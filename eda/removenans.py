import wfdb
import numpy as np
from pathlib import Path

with open("chapman_clean_records.txt") as f:
    records = [Path(line.strip()) for line in f]

final_records = []

for record_path in records:
    signal, meta = wfdb.rdsamp(str(record_path))
    if not np.isnan(signal).any():
        final_records.append(record_path)

print("Final usable records:", len(final_records))

with open("chapman_final_records.txt", "w") as f:
    for r in final_records:
        f.write(str(r) + "\n")