import os
import pandas as pd

df = pd.read_csv("data/ptbxl_database.csv")

missing = []

for ecg_id in df["ecg_id"]:
    ecg_str = str(ecg_id).zfill(5)
    folder = ecg_str[:2] + "000"

    dat_path = os.path.join("data/records100", folder, f"{ecg_str}_lr.dat")
    hea_path = os.path.join("data/records100", folder, f"{ecg_str}_lr.hea")

    if not (os.path.exists(dat_path) and os.path.exists(hea_path)):
        missing.append(ecg_id)

print("Missing IDs:")
print(missing)
print("Total:", len(missing))