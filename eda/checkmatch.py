import os

base_path = "data/records100"

dat_count = 0
hea_count = 0

for root, dirs, files in os.walk(base_path):
    for f in files:
        if f.endswith("_lr.dat"):
            dat_count += 1
        if f.endswith("_lr.hea"):
            hea_count += 1

print("Total .dat files:", dat_count)
print("Total .hea files:", hea_count)