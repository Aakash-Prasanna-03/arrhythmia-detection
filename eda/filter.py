import os
import pandas as pd

df = pd.read_csv("data/ptbxl_database.csv")

valid_rows = []

for ecg_id in df["ecg_id"]:
    ecg_str = str(ecg_id).zfill(5)
    folder = ecg_str[:2] + "000"

    dat_path = os.path.join("data/records100", folder, f"{ecg_str}_lr.dat")
    hea_path = os.path.join("data/records100", folder, f"{ecg_str}_lr.hea")

    if os.path.exists(dat_path) and os.path.exists(hea_path):
        valid_rows.append(ecg_id)

clean_df = df[df["ecg_id"].isin(valid_rows)]

print("Original records:", len(df))
print("Valid records:", len(clean_df))
print("Removed records:", len(df) - len(clean_df))

clean_df.to_csv("data/ptbxl_database_clean.csv", index=False)

print("Saved cleaned CSV → ptbxl_database_clean.csv")