"""
audit_chapman.py
────────────────
Audits ConditionNames_SNOMED-CT.csv (Chapman ECG dictionary).

Checks
------
1.  Required columns present
2.  Blank / whitespace-only fields
3.  SNOMED CT code format  (digits only, 6-18 chars)
4.  Duplicate acronyms
5.  Duplicate SNOMED codes  (different acronyms sharing one code)
6.  Typos / suspicious full names  (numbers, special chars, very short)
7.  Known clinical SNOMED spot-checks  (ground truth verification)
8.  Specific known clinical issues (IDC/IVB, WAVN/SAAWR, LBBBB, ST ambiguity)
9.  OOD label coverage  (all Chapman acronyms used in PTB mapping present?)

Usage
-----
    python audit_chapman.py [path/to/ConditionNames_SNOMED-CT.csv]

Defaults to  data/chapman/ConditionNames_SNOMED-CT.csv
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_PATH = "data/chapman/ConditionNames_SNOMED-CT.csv"

REQUIRED_COLUMNS = {"Acronym Name", "Full Name", "Snomed_CT"}

# Ground-truth SNOMED codes per acronym (verified against SNOMED CT browser)
# AF  = Atrial Flutter  164890007
# AFIB = Atrial Fibrillation 164889003  (not AF!)
KNOWN_SNOMED = {
    "SR":   "426783006",   # Sinus Rhythm
    "SB":   "426177001",   # Sinus Bradycardia
    "SA":   "427393009",   # Sinus Arrhythmia
    "AF":   "164890007",   # Atrial Flutter   ← NOT fibrillation
    "AFIB": "164889003",   # Atrial Fibrillation
    "SVT":  "426761007",   # Supraventricular Tachycardia
    "RBBB": "59118001",    # Right Bundle Branch Block
    "LBBB": "164909002",   # Left Bundle Branch Block
    "LVH":  "164873001",   # Left Ventricular Hypertrophy
    "RVH":  "89792004",    # Right Ventricular Hypertrophy
    "WPW":  "74390002",    # Wolff-Parkinson-White
    "1AVB": "270492004",   # 1st Degree AV Block
    "2AVB": "195042002",   # 2nd Degree AV Block
    "3AVB": "27885002",    # 3rd Degree AV Block
    "VPB":  "17338001",    # Ventricular Premature Beat
    "APB":  "284470004",   # Atrial Premature Beat
    "AQW":  "164917005",   # Abnormal Q Wave
    "STE":  "164930006",   # ST Elevation
    "STDD": "429622005",   # ST Depression
}

# SNOMED codes intentionally shared across multiple Chapman acronyms
# Check 5 will PASS these silently; anything else with dupes gets a WARN
EXPECTED_SNOMED_DUPES = {
    "164865005": {"MI", "MIBW", "MIFW", "MILW", "MISW"},   # MI subtypes
    "164909002": {"LBBB", "LBBBB", "LFBBB"},                # LBBB variants
    "698252002": {"IDC", "IVB"},                            # flagged in check 8
    "195101003": {"WAVN", "SAAWR"},                         # flagged in check 8
}

# Every acronym used in downsamplechapman.py must be resolvable in the CSV
PTB_MAPPING_ACRONYMS = {
    # NORM
    "SR", "SB", "SA",
    # MI
    "AQW",
    # STTC  (note: ST here = Sinus Tachycardia in Chapman, mapped to NORM in pipeline)
    "ST", "STE", "STDD", "STTC", "STTU", "TWC", "TWO",
    # CD
    "1AVB", "2AVB", "3AVB", "AVB", "RBBB", "LBBB", "LFBBB", "IVB",
    # HYP
    "LVH", "RVH", "LVQRSAL",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "  ✅ PASS"
WARN = "  ⚠️  WARN"
FAIL = "  ❌ FAIL"


def snomed_valid(code: str) -> bool:
    """SNOMED CT codes are purely numeric, typically 6–18 digits."""
    return bool(re.fullmatch(r"\d{6,18}", code))


def print_section(title: str):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")


# ── Main audit ────────────────────────────────────────────────────────────────

def audit(path: str):
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    # Read with utf-8-sig to strip BOM (file has BOM from Excel export)
    with open(p, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_cols = set(reader.fieldnames or [])
        rows = list(reader)

    print(f"\n{'═'*62}")
    print(f"  AUDIT: {p.name}")
    print(f"  Rows loaded: {len(rows)}")
    print(f"{'═'*62}")

    issues = []

    # ── 1. Required columns ───────────────────────────────────────────────────
    print_section("1. Required Columns")
    missing_cols = REQUIRED_COLUMNS - raw_cols
    if missing_cols:
        msg = f"Missing columns: {missing_cols}"
        print(f"{FAIL}  {msg}")
        issues.append(msg)
    else:
        print(f"{PASS}  All required columns present")

    # ── 2. Blank fields ───────────────────────────────────────────────────────
    print_section("2. Blank / Whitespace Fields")
    blank_issues = []
    for i, row in enumerate(rows, 2):  # row 1 = header
        for col in ["Acronym Name", "Full Name", "Snomed_CT"]:
            if not row.get(col, "").strip():
                msg = (f"Row {i}: blank '{col}' "
                       f"(acronym={row.get('Acronym Name', '?')!r})")
                blank_issues.append(msg)
    if blank_issues:
        for m in blank_issues:
            print(f"{FAIL}  {m}")
        issues.extend(blank_issues)
    else:
        print(f"{PASS}  No blank fields found")

    # ── 3. SNOMED format ──────────────────────────────────────────────────────
    print_section("3. SNOMED CT Code Format  (6-18 numeric digits)")
    bad_snomed = []
    for row in rows:
        sn = row.get("Snomed_CT", "").strip()
        if sn and not snomed_valid(sn):
            msg = f"{row['Acronym Name']}: invalid SNOMED '{sn}'"
            bad_snomed.append(msg)
    if bad_snomed:
        for m in bad_snomed:
            print(f"{FAIL}  {m}")
        issues.extend(bad_snomed)
    else:
        print(f"{PASS}  All SNOMED codes are valid numeric format")

    # ── 4. Duplicate acronyms ─────────────────────────────────────────────────
    print_section("4. Duplicate Acronyms")
    seen_acro: dict = defaultdict(list)
    for i, row in enumerate(rows, 2):
        seen_acro[row.get("Acronym Name", "").strip()].append(i)
    dupe_acro = {a: lines for a, lines in seen_acro.items() if len(lines) > 1}
    if dupe_acro:
        for a, lines in dupe_acro.items():
            msg = f"Duplicate acronym '{a}' on rows {lines}"
            print(f"{FAIL}  {msg}")
            issues.append(msg)
    else:
        print(f"{PASS}  No duplicate acronyms")

    # ── 5. Duplicate SNOMED codes ─────────────────────────────────────────────
    print_section("5. Duplicate SNOMED Codes  (same code, different acronyms)")
    snomed_to_acros: dict = defaultdict(set)
    for row in rows:
        sn = row.get("Snomed_CT", "").strip()
        ac = row.get("Acronym Name", "").strip()
        if sn:
            snomed_to_acros[sn].add(ac)

    found_any_dupe = False
    for sn, acros in sorted(snomed_to_acros.items()):
        if len(acros) > 1:
            found_any_dupe = True
            expected = EXPECTED_SNOMED_DUPES.get(sn)
            if expected and acros == expected:
                print(f"{PASS}  SNOMED {sn} shared by {sorted(acros)}  "
                      "[expected / documented]")
            else:
                msg = (f"SNOMED {sn} shared by {sorted(acros)}"
                       " — verify intentional")
                print(f"{WARN}  {msg}")
                issues.append(msg)
    if not found_any_dupe:
        print(f"{PASS}  No duplicate SNOMED codes found")

    # ── 6. Suspicious full names ──────────────────────────────────────────────
    print_section("6. Suspicious Full Names  (typos, digits, whitespace)")
    # AVB acronyms legitimately contain digits in their names
    DIGIT_OK = {"1AVB", "2AVB", "2AVB1", "2AVB2", "3AVB"}
    found_name_issue = False
    for row in rows:
        fn = row.get("Full Name", "").strip()   # already stripped
        ac = row.get("Acronym Name", "").strip()

        if len(fn) < 4:
            msg = f"{ac}: full name suspiciously short: {fn!r}"
            print(f"{WARN}  {msg}")
            issues.append(msg)
            found_name_issue = True

        if re.search(r"\d", fn) and ac not in DIGIT_OK:
            msg = f"{ac}: full name contains digit: {fn!r}"
            print(f"{WARN}  {msg}")
            issues.append(msg)
            found_name_issue = True

        # Check raw (un-stripped) value for whitespace issues
        fn_raw = row.get("Full Name", "")
        if "  " in fn_raw or fn_raw != fn_raw.strip():
            msg = f"{ac}: whitespace issue in raw full name: {fn_raw!r}"
            print(f"{WARN}  {msg}")
            issues.append(msg)
            found_name_issue = True

    if not found_name_issue:
        print(f"{PASS}  No suspicious full names found")

    # ── 7. Clinical SNOMED spot-checks ────────────────────────────────────────
    print_section("7. Clinical SNOMED Spot-Checks  (known ground truth)")
    acro_map = {
        r["Acronym Name"].strip(): r["Snomed_CT"].strip() for r in rows
    }
    for acro, expected_sn in KNOWN_SNOMED.items():
        if acro not in acro_map:
            msg = f"{acro}: not found in file"
            print(f"{WARN}  {msg}")
            issues.append(msg)
        elif acro_map[acro] != expected_sn:
            msg = (f"{acro}: expected SNOMED {expected_sn}, "
                   f"got {acro_map[acro]}")
            print(f"{FAIL}  {msg}")
            issues.append(msg)
        else:
            print(f"{PASS}  {acro}: {expected_sn}")

    # ── 8. Specific known clinical issues ─────────────────────────────────────
    print_section("8. Specific Known Clinical Issues")

    # 8a. IDC and IVB share 698252002
    idc_sn = acro_map.get("IDC", "")
    ivb_sn = acro_map.get("IVB", "")
    if idc_sn and idc_sn == ivb_sn:
        msg = (f"IDC and IVB share SNOMED {idc_sn}. "
               "IDC (Interior Differences Conduction) is non-standard — "
               "consider removing or merging with IVB (Intraventricular Block).")
        print(f"{WARN}  {msg}")
        issues.append(msg)

    # 8b. WAVN and SAAWR share 195101003
    wavn_sn  = acro_map.get("WAVN", "")
    saawr_sn = acro_map.get("SAAWR", "")
    if wavn_sn and wavn_sn == saawr_sn:
        msg = (f"WAVN and SAAWR share SNOMED {wavn_sn}. "
               "Clinically distinct rhythms. "
               "Correct SNOMED for wandering atrial pacemaker is 195101003; "
               "WAVN has no standard SNOMED — consider removing WAVN.")
        print(f"{WARN}  {msg}")
        issues.append(msg)

    # 8c. LBBBB is non-standard
    if "LBBBB" in acro_map:
        if acro_map["LBBBB"] == acro_map.get("LBBB", ""):
            msg = ("LBBBB shares SNOMED code with LBBB and is not a standard "
                   "clinical ECG term. Likely intended as LPFB "
                   "(left posterior fascicular block, 445211001). "
                   "Consider removing or remapping.")
            print(f"{WARN}  {msg}")
            issues.append(msg)

    # 8d. ST = Sinus Tachycardia — pipeline collision risk
    if "ST" in acro_map:
        msg = ("'ST' maps to Sinus Tachycardia (427084000) in Chapman. "
               "In downsamplechapman.py it is placed in STTC_SET — "
               "this is WRONG. ST should be in NORM_SET. "
               "Fix: move 'ST' from STTC_SET to NORM_SET in preprocessing.")
        print(f"{FAIL}  {msg}")
        issues.append(msg)

    # ── 9. OOD label coverage ─────────────────────────────────────────────────
    print_section("9. OOD Label Coverage  (PTB mapping acronyms present in CSV?)")
    missing_in_csv = PTB_MAPPING_ACRONYMS - set(acro_map.keys())
    if missing_in_csv:
        for m in sorted(missing_in_csv):
            msg = f"'{m}' used in PTB mapping but MISSING from CSV"
            print(f"{FAIL}  {msg}")
            issues.append(msg)
    else:
        print(f"{PASS}  All {len(PTB_MAPPING_ACRONYMS)} PTB-mapping acronyms "
              "are present in CSV")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  SUMMARY: {len(issues)} issue(s) found  |  {len(rows)} rows audited")
    print(f"{'═'*62}")
    if issues:
        print("\n  All issues:")
        for idx, iss in enumerate(issues, 1):
            print(f"  {idx:>2}. {iss}")
    else:
        print("  ✅ File is clean.")
    print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    audit(path)
