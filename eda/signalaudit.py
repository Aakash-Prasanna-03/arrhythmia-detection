import wfdb
import numpy as np
from pathlib import Path
from collections import Counter

# Load clean record list
with open("chapman_clean_records.txt") as f:
    records = [Path(line.strip()) for line in f]

fs_set = set()
length_set = set()
lead_count_set = set()
lead_orders = set()
nan_count = 0

print("Auditing signal properties...\n")

for i, record_path in enumerate(records):
    signal, meta = wfdb.rdsamp(str(record_path))

    fs_set.add(meta["fs"])
    length_set.add(signal.shape[0])
    lead_count_set.add(signal.shape[1])
    lead_orders.add(tuple(meta["sig_name"]))

    if np.isnan(signal).any():
        nan_count += 1

    if i % 5000 == 0:
        print(f"Checked {i}/{len(records)}")

print("\n=== SIGNAL AUDIT RESULTS ===")
print("Unique sampling rates:", fs_set)
print("Unique signal lengths:", length_set)
print("Unique lead counts:", lead_count_set)
print("Unique lead orders count:", len(lead_orders))
print("Signals containing NaNs:", nan_count)