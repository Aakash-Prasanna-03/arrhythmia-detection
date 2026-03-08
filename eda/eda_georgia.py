import os
from collections import Counter
import pandas as pd

BASE_PATH = r"D:\ad\data\georgia"

code_counter = Counter()
record_counter = 0

for root, dirs, files in os.walk(BASE_PATH):
    for f in files:
        if f.endswith(".hea"):
            record_counter += 1
            fpath = os.path.join(root, f)

            with open(fpath, "r") as file:
                lines = file.readlines()

            for line in lines:
                if line.startswith("# Dx:"):
                    dx_part = line.split(":")[1].strip()
                    codes = [c.strip() for c in dx_part.split(",")]

                    for code in codes:
                        code_counter[code] += 1

print("Total records:", record_counter)
print("Total unique SNOMED codes:", len(code_counter))

print("\nTop 20 most frequent SNOMED codes:")
for code, count in code_counter.most_common(20):
    print(code, "->", count)

# Save full distribution to CSV
df = pd.DataFrame(code_counter.items(), columns=["SNOMED_code", "count"])
df = df.sort_values(by="count", ascending=False)
df.to_csv("georgia_snomed_distribution.csv", index=False)

print("\nSaved full distribution to georgia_snomed_distribution.csv")