"""
downsample_chapman.py
─────────────────────
Preprocesses the Chapman ECG dataset for OOD evaluation against
PTB-XL trained models.

Pipeline
--------
1.  Read record list (chapman_final_records_fixed.txt)
2.  Load SNOMED → acronym map  (ConditionNames_SNOMED-CT.csv)
3.  For each WFDB record:
      a. Load signal  (expected 5000 samples × 12 leads @ 500 Hz)
      b. Downsample 500 Hz → 100 Hz  (5000 → 1000 samples)
      c. Extract SNOMED codes from Dx comment
      d. Map SNOMED → Chapman acronym → PTB superclass
      e. Build 5-class multi-label vector  [NORM, MI, STTC, CD, HYP]
4.  Exclude records with no mappable diagnostic label
5.  Save X_chapman.npy  (N, 12, 1000) float32
       y_chapman.npy  (N, 5)         int8

Key fixes vs original downsamplechapman.py
-------------------------------------------
- ST (Sinus Tachycardia) moved to NORM_SET — was wrongly in STTC_SET
- LBBB and LBBBB added to CD_SET (were missing)
- MI subtypes MIBW/MIFW/MILW/MISW added to MI_SET
- 2AVB1, 2AVB2, AVN variants added to CD_SET
- LVQRSCL, LVQRSLL added to HYP_SET
- NORM label now set from SNOMED mapping (SR/SB/SA), not as fallback
- Records with only excluded rhythm codes are dropped cleanly
- Signal shape check relaxed: accepts any (n≥1000, 12) not just exactly 5000
- dtype: y saved as int8 (not float64)

Usage
-----
    python downsample_chapman.py

    # or override paths:
    python downsample_chapman.py \
        --base_path   data/chapman \
        --wfdb_path   data/chapman/WFDBRecords \
        --record_list data/chapman/chapman_final_records_fixed.txt \
        --snomed_csv  data/chapman/ConditionNames_SNOMED-CT.csv \
        --out_path    data/chapman
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample
from tqdm import tqdm

# ── PTB superclass sets ───────────────────────────────────────────────────────
# Index:  0=NORM  1=MI  2=STTC  3=CD  4=HYP

NORM_SET = {
    "SR",       # Sinus Rhythm
    "SB",       # Sinus Bradycardia
    "SA",       # Sinus Arrhythmia / Irregularity
    "ST",       # Sinus Tachycardia  ← FIX: was wrongly in STTC_SET
}

MI_SET = {
    "AQW",      # Abnormal Q Wave
    "MI",       # Myocardial Infarction (generic)
    "MIBW",     # MI back wall
    "MIFW",     # MI front wall
    "MILW",     # MI lower wall
    "MISW",     # MI side wall
}

STTC_SET = {
    "STE",      # ST Extension / Elevation
    "STDD",     # ST Drop Down / Depression
    "STTC",     # ST-T Change (generic)
    "STTU",     # ST Tilt Up
    "TWC",      # T Wave Change
    "TWO",      # T Wave Opposite / Inversion
}

CD_SET = {
    "1AVB",     # 1st Degree AV Block
    "2AVB",     # 2nd Degree AV Block
    "2AVB1",    # 2nd Degree AV Block Type I (Wenckebach)
    "2AVB2",    # 2nd Degree AV Block Type II (Mobitz)
    "3AVB",     # 3rd Degree AV Block (Complete)
    "AVB",      # AV Block (generic)
    "RBBB",     # Right Bundle Branch Block
    "LBBB",     # Left Bundle Branch Block   ← FIX: was missing
    "LBBBB",    # Left Back Bundle Branch Block (Chapman-specific)
    "LFBBB",    # Left Front Bundle Branch Block
    "IVB",      # Intraventricular Block
    "IDC",      # Interior Differences Conduction (merged with IVB)
    "PRIE",     # PR Interval Extension
}

HYP_SET = {
    "LVH",      # Left Ventricular Hypertrophy
    "RVH",      # Right Ventricular Hypertrophy
    "RAH",      # Right Atrial Hypertrophy
    "LVQRSAL",  # Lower voltage QRS all leads
    "LVQRSCL",  # Lower voltage QRS chest leads  ← FIX: was missing
    "LVQRSLL",  # Lower voltage QRS limb leads   ← FIX: was missing
}

# Rhythm-only codes: not part of any PTB diagnostic class
# Records whose ONLY codes fall here are excluded entirely
EXCLUDED_ACRONYMS = {
    "AFIB", "AF", "SVT", "AT", "APB", "VPB", "QTIE",
    "ARS",  "ALS", "WPW", "VPE", "VEB", "VB",  "VET",
    "JPT",  "JEB", "WAVN", "AVNRT", "AVRT", "SAAWR",
    "ABI",  "CCR", "CR",  "ERV",  "FQRS", "UW",  "PWC",
}

# ── Superclass index lookup ───────────────────────────────────────────────────

def acronym_to_superclass(acronym: str) -> int | None:
    """Return PTB superclass index (0-4) or None if not mappable."""
    acronym = acronym.strip().upper()
    if acronym in NORM_SET:
        return 0
    if acronym in MI_SET:
        return 1
    if acronym in STTC_SET:
        return 2
    if acronym in CD_SET:
        return 3
    if acronym in HYP_SET:
        return 4
    return None   # excluded or unknown

# ── Signal downsampling ───────────────────────────────────────────────────────

TARGET_SAMPLES = 1000  # 10 s × 100 Hz

def downsample_signal(signal: np.ndarray) -> np.ndarray:
    """
    Input:  (n_samples, 12) — any n_samples, 500 Hz
    Output: (12, 1000)      — 100 Hz, float32

    Uses scipy.signal.resample (FFT-based, anti-aliased).
    Each lead is resampled independently then cropped/padded to exactly 1000.
    """
    out = np.zeros((12, TARGET_SAMPLES), dtype=np.float32)
    n_in = signal.shape[0]
    # Compute target length proportionally (handles non-5000 lengths gracefully)
    n_out = int(round(n_in * TARGET_SAMPLES / 5000)) if n_in != 5000 else TARGET_SAMPLES

    for lead in range(12):
        resampled = resample(signal[:, lead].astype(np.float32), n_out)
        n_copy = min(len(resampled), TARGET_SAMPLES)
        out[lead, :n_copy] = resampled[:n_copy]
    return out

# ── SNOMED map loader ─────────────────────────────────────────────────────────

def load_snomed_map(csv_path: str) -> dict:
    """
    Returns {snomed_code_str → acronym_str}.
    Strips BOM, normalises float codes (e.g. '164865005.0' → '164865005').
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    # Snomed_CT may be read as float by pandas if column has mixed types
    df["Snomed_CT"] = (df["Snomed_CT"]
                       .astype(str)
                       .str.replace(r"\.0$", "", regex=True)
                       .str.strip())
    df["Acronym Name"] = df["Acronym Name"].str.strip()
    return dict(zip(df["Snomed_CT"], df["Acronym Name"]))

# ── Comment parser ────────────────────────────────────────────────────────────

def extract_snomed_codes(record: wfdb.Record) -> list:
    """
    Parse SNOMED codes from WFDB record comments.
    Chapman stores them as:  'Dx: 426783006,164889003'
    Returns list of code strings, empty list if not found.
    """
    for comment in (record.comments or []):
        comment = str(comment)
        if "Dx" in comment:
            raw = comment.split(":")[-1]
            return [
                c.strip().replace(".0", "")
                for c in raw.split(",")
                if c.strip()
            ]
    return []

# ── Main preprocessing ────────────────────────────────────────────────────────

def preprocess(base_path: str, wfdb_path: str, record_list: str,
               snomed_csv: str, out_path: str) -> None:

    # Load inputs
    with open(record_list, "r") as f:
        records = [line.strip() for line in f if line.strip()]
    print(f"Records to process : {len(records)}")

    snomed_to_acronym = load_snomed_map(snomed_csv)
    print(f"SNOMED map entries : {len(snomed_to_acronym)}")

    X_list: list = []
    y_list: list = []

    # Counters for diagnostics
    n_bad_signal   = 0
    n_no_dx        = 0
    n_only_rhythm  = 0
    n_ok           = 0

    for rec in tqdm(records, desc="Chapman"):
        try:
            record_path = os.path.join(wfdb_path, rec)
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal   # (n_samples, 12)
        except Exception:
            n_bad_signal += 1
            continue

        # Basic signal validation
        if signal is None or signal.ndim != 2 or signal.shape[1] < 12:
            n_bad_signal += 1
            continue
        if signal.shape[0] < TARGET_SAMPLES:
            # Too short to downsample meaningfully
            n_bad_signal += 1
            continue

        # Replace NaN/Inf
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Extract SNOMED codes
        snomed_codes = extract_snomed_codes(record)
        if not snomed_codes:
            n_no_dx += 1
            continue

        # Build label vector
        label_vec = np.zeros(5, dtype=np.int8)
        has_any_valid = False

        for code in snomed_codes:
            acronym = snomed_to_acronym.get(code)
            if acronym is None:
                continue  # unknown SNOMED code
            if acronym in EXCLUDED_ACRONYMS:
                continue  # rhythm-only, not a PTB diagnostic class

            cls = acronym_to_superclass(acronym)
            if cls is not None:
                label_vec[cls] = 1
                has_any_valid = True

        if not has_any_valid:
            n_only_rhythm += 1
            continue

        # Downsample signal
        ecg = downsample_signal(signal)  # (12, 1000)

        X_list.append(ecg)
        y_list.append(label_vec)
        n_ok += 1

    # ── Stack and save ────────────────────────────────────────────────────────
    if not X_list:
        print("[ERROR] No valid samples found. Check paths and WFDB records.")
        return

    X = np.stack(X_list, axis=0)   # (N, 12, 1000)  float32
    y = np.stack(y_list, axis=0)   # (N, 5)          int8

    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, "X_chapman.npy"), X)
    np.save(os.path.join(out_path, "y_chapman.npy"), y)

    # ── Report ────────────────────────────────────────────────────────────────
    label_names = ["NORM", "MI", "STTC", "CD", "HYP"]
    total = len(records)

    print(f"\n{'='*52}")
    print(f"  Chapman preprocessing complete")
    print(f"{'='*52}")
    print(f"  Total records        : {total}")
    print(f"  Saved samples        : {n_ok}  ({n_ok/total*100:.1f}%)")
    print(f"  Skipped – bad signal : {n_bad_signal}")
    print(f"  Skipped – no Dx code : {n_no_dx}")
    print(f"  Skipped – rhythm only: {n_only_rhythm}")
    print(f"\n  Output X shape : {X.shape}   dtype={X.dtype}")
    print(f"  Output y shape : {y.shape}   dtype={y.dtype}")
    print(f"\n  Label distribution:")
    for i, name in enumerate(label_names):
        n = int(y[:, i].sum())
        print(f"    {name:6s}: {n:6d}  ({n/len(y)*100:.1f}%)")
    multi = int((y.sum(axis=1) > 1).sum())
    print(f"\n  Multi-label samples  : {multi}  ({multi/len(y)*100:.1f}%)")
    print(f"\n  Saved to: {out_path}/X_chapman.npy")
    print(f"            {out_path}/y_chapman.npy")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Chapman ECG → X_chapman.npy / y_chapman.npy"
    )
    parser.add_argument("--base_path",   default="data/chapman")
    parser.add_argument("--wfdb_path",   default="data/chapman/WFDBRecords")
    parser.add_argument("--record_list", default="data/chapman/chapman_final_records_fixed.txt")
    parser.add_argument("--snomed_csv",  default="data/chapman/ConditionNames_SNOMED-CT.csv")
    parser.add_argument("--out_path",    default="data/chapman")
    args = parser.parse_args()

    preprocess(
        base_path   = args.base_path,
        wfdb_path   = args.wfdb_path,
        record_list = args.record_list,
        snomed_csv  = args.snomed_csv,
        out_path    = args.out_path,
    )
