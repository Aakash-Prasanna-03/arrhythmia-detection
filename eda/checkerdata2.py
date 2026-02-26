import wfdb
from pathlib import Path

# ---- PATH ----
root = Path("../data/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords")

print("Scanning dataset...\n")

# ---- STEP 1: Pairing check (.hea <-> .mat) ----
hea_files = list(root.rglob("*.hea"))
mat_files = list(root.rglob("*.mat"))

hea_set = set([f.with_suffix("") for f in hea_files])
mat_set = set([f.with_suffix("") for f in mat_files])

valid_pairs = hea_set.intersection(mat_set)
missing_mat = hea_set - mat_set
missing_hea = mat_set - hea_set

print("=== FILE STRUCTURE CHECK ===")
print("Total .hea files:", len(hea_set))
print("Total .mat files:", len(mat_set))
print("Valid pairs:", len(valid_pairs))
print("Missing .mat:", len(missing_mat))
print("Missing .hea:", len(missing_hea))
print()

# ---- STEP 2: Loadability check ----
print("Checking loadability...\n")

failed_load = []

for i, record_path in enumerate(valid_pairs):
    try:
        signal, meta = wfdb.rdsamp(str(record_path))
    except Exception as e:
        failed_load.append(record_path)

    if i % 5000 == 0:
        print(f"Checked {i}/{len(valid_pairs)}")

print("\n=== LOADABILITY CHECK ===")
print("Failed to load:", len(failed_load))

# ---- STEP 3: Build clean list ----
clean_records = [r for r in valid_pairs if r not in failed_load]

print("\n=== FINAL CLEAN DATASET ===")
print("Clean records:", len(clean_records))

# ---- STEP 4: Save clean record list ----
with open("chapman_clean_records.txt", "w") as f:
    for r in clean_records:
        f.write(str(r) + "\n")

print("\nSaved clean record list to chapman_clean_records.txt")