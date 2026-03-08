"""
audit_georgia.py
────────────────
Audits SNOMED_mappings_scored.csv and SNOMED_mappings_unscored.csv
(Georgia ECG dataset label dictionaries).

These files are used to map Georgia SNOMED codes → PTB superclasses
for OOD evaluation of PTB-XL trained models.

Checks
------
1.  Required columns present  (both files)
2.  Blank / whitespace-only fields
3.  SNOMED CT code format  (digits only, 6-18 chars)
4.  Duplicate SNOMED codes within each file
5.  Cross-file conflicts  (same SNOMED, different abbreviation)
6.  Known clinical SNOMED spot-checks  (ground truth)
7.  PTB superclass coverage  (all 5 classes reachable from Georgia?)
8.  Georgia-column presence  (codes with Georgia=0 contribute nothing to OOD)
9.  Abbreviation collision with Chapman  (same abbrev, different SNOMED)

Usage
-----
    python audit_georgia.py \
        [path/to/SNOMED_mappings_scored.csv] \
        [path/to/SNOMED_mappings_unscored.csv]

Defaults to  data/SNOMED_mappings_scored.csv  and  data/SNOMED_mappings_unscored.csv
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_SCORED   = "data/SNOMED_mappings_scored.csv"
DEFAULT_UNSCORED = "data/SNOMED_mappings_unscored.csv"

# Both files use semicolon delimiter
DELIMITER = ";"

REQUIRED_COLUMNS = {"Dx", "SNOMED CT Code", "Abbreviation", "Georgia"}

# Ground-truth SNOMED → abbreviation spot-checks (verified)
KNOWN_SNOMED = {
    "270492004": "IAVB",    # 1st degree AV block
    "164889003": "AF",      # Atrial fibrillation
    "164890007": "AFL",     # Atrial flutter
    "164909002": "LBBB",    # Left bundle branch block
    "59118001":  "RBBB",    # Right bundle branch block
    "713427006": "CRBBB",   # Complete RBBB
    "426783006": "SNR",     # Sinus (normal) rhythm
    "426177001": "SB",      # Sinus bradycardia
    "427084000": "STach",   # Sinus tachycardia
    "427393009": "SA",      # Sinus arrhythmia
    "164873001": "LVH",     # Left ventricular hypertrophy
    "164934002": "TAb",     # T-wave abnormal
    "429622005": "STD",     # ST depression
    "164931005": "STE",     # ST elevation
    "164865005": "MI",      # Myocardial infarction
    "111975006": "LQT",     # Prolonged QT
    "284470004": "PAC",     # Premature atrial contraction
    "164917005": "QAb",     # Q-wave abnormal
}

# PTB superclass → abbreviations that map to it (Georgia pipeline mapping)
# Used to verify all 5 classes have at least one Georgia record
PTB_CLASS_ABBREVS = {
    "NORM": {"SNR", "SB", "STach", "SA"},
    "MI":   {"MI", "QAb", "AnMI", "OldMI"},
    "STTC": {"STD", "STE", "STIAb", "TAb", "TInv", "LQT", "NSSTTA",
             "AnMIs", "IIs", "LIs", "STC"},
    "CD":   {"IAVB", "LBBB", "RBBB", "CRBBB", "IRBBB", "LAnFB",
             "NSIVCB", "IIAVB", "CHB", "ILBBB", "BBB"},
    "HYP":  {"LVH", "RVH", "LAE", "RAH", "VH", "LQRSV"},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "  ✅ PASS"
WARN = "  ⚠️  WARN"
FAIL = "  ❌ FAIL"


def snomed_valid(code: str) -> bool:
    return bool(re.fullmatch(r"\d{6,18}", code))


def print_section(title: str):
    print(f"\n{'─'*64}")
    print(f"  {title}")
    print(f"{'─'*64}")


def load_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Load semicolon-delimited CSV, return (fieldnames, rows)."""
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=DELIMITER)
        fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        rows = []
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return fieldnames, rows


def georgia_count(row: dict) -> int:
    """Return the Georgia column value as int (0 if missing/blank)."""
    val = row.get("Georgia", "0").strip()
    try:
        return int(val)
    except ValueError:
        return 0


# ── Main audit ────────────────────────────────────────────────────────────────

def audit(scored_path: str, unscored_path: str):

    sp = Path(scored_path)
    up = Path(unscored_path)

    for p in [sp, up]:
        if not p.exists():
            print(f"[ERROR] File not found: {p}")
            sys.exit(1)

    scored_cols,   scored_rows   = load_csv(sp)
    unscored_cols, unscored_rows = load_csv(up)

    all_rows  = [("scored",   r) for r in scored_rows]
    all_rows += [("unscored", r) for r in unscored_rows]

    print(f"\n{'═'*64}")
    print(f"  AUDIT: Georgia SNOMED mapping files")
    print(f"  Scored rows   : {len(scored_rows)}   ({sp.name})")
    print(f"  Unscored rows : {len(unscored_rows)}   ({up.name})")
    print(f"{'═'*64}")

    issues = []

    # ── 1. Required columns ───────────────────────────────────────────────────
    print_section("1. Required Columns")
    for label, cols in [("scored", scored_cols), ("unscored", unscored_cols)]:
        missing = REQUIRED_COLUMNS - set(cols)
        if missing:
            msg = f"{label}: missing columns {missing}"
            print(f"{FAIL}  {msg}")
            issues.append(msg)
        else:
            print(f"{PASS}  {label}: all required columns present")

    # ── 2. Blank fields ───────────────────────────────────────────────────────
    print_section("2. Blank / Whitespace Fields")
    blank_found = False
    for source, row in all_rows:
        for col in ["Dx", "SNOMED CT Code", "Abbreviation"]:
            if not row.get(col, "").strip():
                msg = (f"[{source}] blank '{col}' for "
                       f"dx={row.get('Dx','?')!r}")
                print(f"{FAIL}  {msg}")
                issues.append(msg)
                blank_found = True
    if not blank_found:
        print(f"{PASS}  No blank fields found")

    # ── 3. SNOMED format ──────────────────────────────────────────────────────
    print_section("3. SNOMED CT Code Format  (6-18 numeric digits)")
    bad_found = False
    for source, row in all_rows:
        sn = row.get("SNOMED CT Code", "").strip()
        if sn and not snomed_valid(sn):
            msg = (f"[{source}] {row.get('Abbreviation','?')}: "
                   f"invalid SNOMED '{sn}'")
            print(f"{FAIL}  {msg}")
            issues.append(msg)
            bad_found = True
    if not bad_found:
        print(f"{PASS}  All SNOMED codes are valid numeric format")

    # ── 4. Duplicate SNOMED within each file ──────────────────────────────────
    print_section("4. Duplicate SNOMED Codes Within Each File")
    for label, rows in [("scored", scored_rows), ("unscored", unscored_rows)]:
        sn_seen: dict = defaultdict(list)
        for row in rows:
            sn = row.get("SNOMED CT Code", "").strip()
            ab = row.get("Abbreviation", "").strip()
            if sn:
                sn_seen[sn].append(ab)
        found_dupe = False
        for sn, abbrevs in sorted(sn_seen.items()):
            if len(abbrevs) > 1:
                msg = (f"[{label}] SNOMED {sn} appears {len(abbrevs)}x "
                       f"→ {abbrevs}")
                print(f"{WARN}  {msg}")
                issues.append(msg)
                found_dupe = True
        if not found_dupe:
            print(f"{PASS}  [{label}] No duplicate SNOMED codes")

    # ── 5. Cross-file SNOMED conflicts ────────────────────────────────────────
    print_section("5. Cross-File SNOMED Conflicts  (same code, different abbrev)")
    scored_map   = {r["SNOMED CT Code"]: r["Abbreviation"] for r in scored_rows
                    if r.get("SNOMED CT Code")}
    unscored_map = {r["SNOMED CT Code"]: r["Abbreviation"] for r in unscored_rows
                    if r.get("SNOMED CT Code")}
    conflict_found = False
    for sn in set(scored_map) & set(unscored_map):
        ab_s = scored_map[sn]
        ab_u = unscored_map[sn]
        if ab_s != ab_u:
            msg = (f"SNOMED {sn}: scored='{ab_s}' vs unscored='{ab_u}' "
                   "— abbreviation mismatch across files")
            print(f"{WARN}  {msg}")
            issues.append(msg)
            conflict_found = True
    if not conflict_found:
        print(f"{PASS}  No cross-file abbreviation conflicts")

    # ── 6. Clinical SNOMED spot-checks ────────────────────────────────────────
    print_section("6. Clinical SNOMED Spot-Checks  (known ground truth)")
    # Build combined map: scored takes priority over unscored
    combined_map = {**unscored_map, **scored_map}
    for sn, expected_abbrev in KNOWN_SNOMED.items():
        if sn not in combined_map:
            msg = f"SNOMED {sn} ({expected_abbrev}) not found in either file"
            print(f"{WARN}  {msg}")
            issues.append(msg)
        elif combined_map[sn] != expected_abbrev:
            msg = (f"SNOMED {sn}: expected abbrev '{expected_abbrev}', "
                   f"got '{combined_map[sn]}'")
            print(f"{FAIL}  {msg}")
            issues.append(msg)
        else:
            print(f"{PASS}  {sn} → {expected_abbrev}")

    # ── 7. PTB superclass coverage ────────────────────────────────────────────
    print_section("7. PTB Superclass Coverage  (all 5 classes reachable?)")
    # Build set of abbreviations with Georgia > 0
    georgia_abbrevs = set()
    for _, row in all_rows:
        if georgia_count(row) > 0:
            ab = row.get("Abbreviation", "").strip()
            if ab:
                georgia_abbrevs.add(ab)

    print(f"  Georgia-active abbreviations: {len(georgia_abbrevs)}")

    for cls, abbrevs in PTB_CLASS_ABBREVS.items():
        hit = abbrevs & georgia_abbrevs
        if hit:
            print(f"{PASS}  {cls}: covered by {sorted(hit)}")
        else:
            msg = (f"{cls}: NO Georgia records map to this class! "
                   f"Expected one of {sorted(abbrevs)}")
            print(f"{FAIL}  {msg}")
            issues.append(msg)

    # ── 8. Georgia-zero warning ───────────────────────────────────────────────
    print_section("8. Georgia-Zero Entries  (codes that won't contribute to OOD)")
    georgia_zero = []
    for source, row in all_rows:
        if georgia_count(row) == 0:
            ab = row.get("Abbreviation", "?")
            dx = row.get("Dx", "?")
            georgia_zero.append(f"[{source}] {ab} ({dx})")
    if georgia_zero:
        print(f"  {WARN}  {len(georgia_zero)} entries have Georgia=0 "
              "(informational, not errors):")
        for entry in georgia_zero[:10]:   # show first 10 to avoid flood
            print(f"         {entry}")
        if len(georgia_zero) > 10:
            print(f"         … and {len(georgia_zero)-10} more")
    else:
        print(f"{PASS}  All entries have Georgia > 0")

    # ── 9. Abbreviation collision with Chapman ────────────────────────────────
    print_section("9. Abbreviation Collision with Chapman Acronyms")
    # Chapman uses different SNOMED for same-named abbreviations in some cases
    # Key collisions to watch: SB, SA, SR, ST, RBBB, LBBB
    CHAPMAN_SNOMED = {
        "SR":   "426783006",
        "SB":   "426177001",
        "SA":   "427393009",
        "RBBB": "59118001",
        "LBBB": "164909002",
        "LVH":  "164873001",
        "RVH":  "89792004",
    }
    collision_found = False
    for ab, chap_sn in CHAPMAN_SNOMED.items():
        # Find this abbreviation in Georgia files
        for source, row in all_rows:
            if row.get("Abbreviation", "").strip() == ab:
                geo_sn = row.get("SNOMED CT Code", "").strip()
                if geo_sn and geo_sn != chap_sn:
                    msg = (f"Abbreviation '{ab}': Chapman SNOMED={chap_sn}, "
                           f"Georgia [{source}] SNOMED={geo_sn} — "
                           "same label, different code; verify mapping is consistent")
                    print(f"{WARN}  {msg}")
                    issues.append(msg)
                    collision_found = True
                break
    if not collision_found:
        print(f"{PASS}  No abbreviation/SNOMED collisions with Chapman")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_rows = len(scored_rows) + len(unscored_rows)
    print(f"\n{'═'*64}")
    print(f"  SUMMARY: {len(issues)} issue(s) found  |  "
          f"{total_rows} total rows audited")
    print(f"{'═'*64}")
    if issues:
        print("\n  All issues:")
        for idx, iss in enumerate(issues, 1):
            print(f"  {idx:>2}. {iss}")
    else:
        print("  ✅ Files are clean.")
    print()


if __name__ == "__main__":
    scored_path   = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCORED
    unscored_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_UNSCORED
    audit(scored_path, unscored_path)
