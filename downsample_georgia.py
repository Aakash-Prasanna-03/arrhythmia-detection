import os
import argparse
import numpy as np
import wfdb
import pandas as pd
from scipy.signal import resample
from tqdm import tqdm

# Add missing import for Path
from pathlib import Path

# =========================
# PATHS
# =========================

BASE_PATH   = "data/georgia"
RECORD_LIST = os.path.join(BASE_PATH, "RECORDS")
SCORED_CSV   = "data/SNOMED_mappings_scored.csv"
UNSCORED_CSV = "data/SNOMED_mappings_unscored.csv"

# =========================
# LOAD SNOMED MAPPING
# =========================
# Georgia uses two files (scored + unscored), both semicolon-delimited.
# We build one unified SNOMED code → abbreviation dict.
# Scored file takes priority on conflicts.

def load_snomed_map(scored_path: str, unscored_path: str) -> dict:
    maps = {}
    for path in [unscored_path, scored_path]:   # scored loaded last = higher priority
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        df["SNOMED CT Code"] = df["SNOMED CT Code"].astype(str).str.strip()
        df["Abbreviation"]   = df["Abbreviation"].str.strip()
        for _, row in df.iterrows():
            code = row["SNOMED CT Code"]
            abbr = row["Abbreviation"]
            if code and abbr:
                maps[code] = abbr
    return maps

# =========================
# GEORGIA → PTB SUPERCLASS
# =========================
# Only abbreviations with Georgia > 0 in either mapping file are included.
# Source counts confirmed from the CSV columns.

NORM_SET = {
    "SNR",      # sinus rhythm            (Georgia scored:  1752)
    "SB",       # sinus bradycardia       (Georgia scored:  1677)
    "STach",    # sinus tachycardia       (Georgia scored:  1261)
    "SA",       # sinus arrhythmia        (Georgia scored:   455)
}

MI_SET = {
    "QAb",      # Q-wave abnormal         (Georgia scored:   464)
    "MI",       # myocardial infarction   (Georgia unscored:   7)
}

STTC_SET = {
    "TAb",      # T wave abnormal         (Georgia scored:  2306)
    "TInv",     # T wave inversion        (Georgia scored:   812)
    "LQT",      # prolonged QT            (Georgia scored:  1391)
    "STIAb",    # ST interval abnormal    (Georgia unscored: 992)
    "NSSTTA",   # nonspecific ST-T        (Georgia unscored:1883)
    "LIs",      # lateral ischaemia       (Georgia unscored: 903)
    "IIs",      # inferior ischaemia      (Georgia unscored: 451)
    "AnMIs",    # anterior ischaemia      (Georgia unscored: 281)
    "STE",      # ST elevation            (Georgia unscored: 134)
    "STD",      # ST depression           (Georgia scored:    38)
    "STC",      # ST changes              (Georgia unscored:   6)
}

CD_SET = {
    "IAVB",     # 1st degree AV block     (Georgia scored:   769)
    "IRBBB",    # incomplete RBBB         (Georgia scored:   407)
    "LBBB",     # left bundle branch block(Georgia scored:   231)
    "LAnFB",    # left ant. fasc. block   (Georgia scored:   180)
    "NSIVCB",   # nonspec. IV cond. dis.  (Georgia scored:   203)
    "CRBBB",    # complete RBBB           (Georgia scored:    28)
    "RBBB",     # right bundle branch blk (Georgia scored:   542)
    "IIAVB",    # 2nd degree AV block     (Georgia unscored:  23)
    "ILBBB",    # incomplete LBBB         (Georgia unscored:  86)
    "BBB",      # bundle branch block     (Georgia unscored: 116)
    "CHB",      # complete heart block    (Georgia unscored:   8)
    "AVB",      # AV block (generic)      (Georgia unscored:  74)
    "LPFB",     # left post. fasc. block  (Georgia unscored:  25)
}

HYP_SET = {
    "LVH",      # left vent. hypertrophy  (Georgia unscored:1232)
    "LAE",      # left atrial enlargement (Georgia unscored: 870)
    "LQRSV",    # low QRS voltages        (Georgia scored:   374)
    "RVH",      # right vent. hypertrophy (Georgia unscored:  86)
    "VH",       # ventricular hypertrophy (Georgia unscored:  71)
    "LAA",      # left atrial abnormality (Georgia unscored:  72)
    "AH",       # atrial hypertrophy      (Georgia unscored:  60)
    "RAAb",     # right atrial abnormality(Georgia unscored:  14)
}

# Rhythm / artefact / non-PTB abbreviations — records whose only
# codes fall here are excluded entirely
EXCLUDED_ABBREVS = {
    "AF",    "AFL",   "Brady", "PAC",   "PVC",   "SVPB",  "VPB",
    "LPR",   "PR",    "LAD",   "RAD",   "SVT",   "ATach", "AFAFL",
    "AP",    "AJR",   "AVJR",  "JTach", "JE",    "WAP",   "WPW",
    "ERe",   "VBig",  "VEB",   "VEsB",  "VEsR",  "VF",    "VFL",
    "VPEx",  "VTach", "VTrig", "VPP",   "ALR",   "SPRI",  "RAb",
    "BPAC",  "SVB",   "TIA",   "UAb",   "HTV",   "ISTD",
}


def abbrev_to_superclass(abbrev: str) -> int | None:
    """Return PTB superclass index (0-4) or None if not mappable."""
    abbrev = abbrev.strip()
    if abbrev in NORM_SET:
        return 0
    if abbrev in MI_SET:
        return 1
    if abbrev in STTC_SET:
        return 2
    if abbrev in CD_SET:
        return 3
    if abbrev in HYP_SET:
        return 4
    return None

# =========================
# SIGNAL DOWNSAMPLING
# =========================

TARGET_SAMPLES = 1000   # 10 s × 100 Hz

def downsample_signal(signal: np.ndarray) -> np.ndarray:
    """
    Input:  (n_samples, 12)  @ 500 Hz
    Output: (12, 1000)       @ 100 Hz, float32
    """
    signal = signal.T                                    # (12, n_samples)
    signal_100hz = resample(signal, TARGET_SAMPLES, axis=1)
    return signal_100hz.astype(np.float32)              # (12, 1000)

# =========================
# SNOMED CODE EXTRACTION
# =========================

def extract_snomed_codes(record: wfdb.Record) -> list:
    """Parse 'Dx: code1,code2,...' from WFDB record comments."""
    for comment in (record.comments or []):
        comment = str(comment)
        if "Dx" in comment:
            raw = comment.split(":")[-1]
            return [c.strip().replace(".0", "") for c in raw.split(",") if c.strip()]
    return []

# =========================
# RECORD DISCOVERY
# =========================

def discover_records(base_path: str, record_list: str) -> list:
    """
    Return list of full record paths (no extension).
    Uses RECORDS file if present, otherwise scans for .hea files.
    """
    rfile = Path(record_list) if record_list else None
    if rfile and rfile.exists():
        lines = rfile.read_text().splitlines()
        return [str(Path(base_path) / l.strip()) for l in lines if l.strip()]
    # Fallback: scan recursively for .hea files
    from pathlib import Path as P
    hea_files = sorted(P(base_path).rglob("*.hea"))
    return [str(h.with_suffix("")) for h in hea_files]

# =========================
# DATA CONSTRUCTION
# =========================

def preprocess(base_path: str, record_list: str,
               scored_csv: str, unscored_csv: str,
               out_path: str) -> None:

    from pathlib import Path

    snomed_to_abbrev = load_snomed_map(scored_csv, unscored_csv)
    print(f"SNOMED map entries : {len(snomed_to_abbrev)}")

    records = discover_records(base_path, record_list)
    print(f"Records to process : {len(records)}")

    X = []
    y = []

    n_bad_signal  = 0
    n_no_dx       = 0
    n_only_rhythm = 0
    n_ok          = 0

    for rec_path in tqdm(records, desc="Georgia"):
        try:
            record = wfdb.rdrecord(rec_path)
            signal = record.p_signal   # (n_samples, 12)
        except Exception:
            n_bad_signal += 1
            continue

        # Signal validation
        if signal is None or signal.ndim != 2 or signal.shape[1] < 12:
            n_bad_signal += 1
            continue
        if signal.shape[0] < TARGET_SAMPLES:
            n_bad_signal += 1
            continue

        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Extract SNOMED codes
        snomed_codes = extract_snomed_codes(record)
        if not snomed_codes:
            n_no_dx += 1
            continue

        # Build label vector
        label_vec = np.zeros(5, dtype=np.int8)
        has_valid = False

        for code in snomed_codes:
            abbrev = snomed_to_abbrev.get(code)
            if abbrev is None:
                continue
            if abbrev in EXCLUDED_ABBREVS:
                continue

            cls = abbrev_to_superclass(abbrev)
            if cls is not None:
                label_vec[cls] = 1
                has_valid = True

        if not has_valid:
            n_only_rhythm += 1
            continue

        signal_ds = downsample_signal(signal)   # (12, 1000)

        X.append(signal_ds)
        y.append(label_vec)
        n_ok += 1

    # =========================
    # SAVE
    # =========================
    if not X:
        print("[ERROR] No valid samples produced. Check paths and WFDB records.")
        return

    X_arr = np.stack(X)   # (N, 12, 1000) float32
    y_arr = np.stack(y)   # (N, 5)        int8

    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, "X_georgia.npy"), X_arr)
    np.save(os.path.join(out_path, "y_georgia.npy"), y_arr)

    # =========================
    # REPORT
    # =========================
    label_names = ["NORM", "MI", "STTC", "CD", "HYP"]
    total = len(records)

    print(f"\n{'='*52}")
    print(f"  Georgia preprocessing complete")
    print(f"{'='*52}")
    print(f"  Total records        : {total}")
    print(f"  Saved samples        : {n_ok}  ({n_ok/total*100:.1f}%)")
    print(f"  Skipped – bad signal : {n_bad_signal}")
    print(f"  Skipped – no Dx code : {n_no_dx}")
    print(f"  Skipped – rhythm only: {n_only_rhythm}")
    print(f"\n  Output X shape : {X_arr.shape}   dtype={X_arr.dtype}")
    print(f"  Output y shape : {y_arr.shape}   dtype={y_arr.dtype}")
    print(f"\n  Label distribution:")
    for i, name in enumerate(label_names):
        n = int(y_arr[:, i].sum())
        print(f"    {name:6s}: {n:6d}  ({n/len(y_arr)*100:.1f}%)")
    multi = int((y_arr.sum(axis=1) > 1).sum())
    print(f"\n  Multi-label samples  : {multi}  ({multi/len(y_arr)*100:.1f}%)")
    print(f"\n  Saved to: {out_path}/X_georgia.npy")
    print(f"            {out_path}/y_georgia.npy")


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Georgia ECG → X_georgia.npy / y_georgia.npy"
    )
    parser.add_argument("--base_path",    default=BASE_PATH)
    parser.add_argument("--record_list",  default=RECORD_LIST)
    parser.add_argument("--scored_csv",   default=SCORED_CSV)
    parser.add_argument("--unscored_csv", default=UNSCORED_CSV)
    parser.add_argument("--out_path",     default=BASE_PATH)
    args = parser.parse_args()

    preprocess(
        base_path   = args.base_path,
        record_list = args.record_list,
        scored_csv  = args.scored_csv,
        unscored_csv= args.unscored_csv,
        out_path    = args.out_path,
    )
