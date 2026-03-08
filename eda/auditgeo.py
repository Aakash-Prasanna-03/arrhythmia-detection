import os
import pandas as pd
import numpy as np
from collections import Counter

BASE_PATH = "data/georgia"

META_CSV = os.path.join(BASE_PATH, "ptbxl_database.csv")
SNOMED_SCORED = os.path.join(BASE_PATH, "SNOMED_mappings_scored.csv")
SNOMED_UNSCORED = os.path.join(BASE_PATH, "SNOMED_mappings_unscored.csv")

df = pd.read_csv(META_CSV)

print("Total records:", len(df))

# =========================
# LOAD SNOMED MAPS
# =========================

scored = pd.read_csv(SNOMED_SCORED)
unscored = pd.read_csv(SNOMED_UNSCORED)

snomed_to_diag = {}

for _, r in scored.iterrows():
    snomed_to_diag[str(r["SNOMED CT Code"])] = r["Abbreviation"]

for _, r in unscored.iterrows():
    snomed_to_diag[str(r["SNOMED CT Code"])] = r["Abbreviation"]

print("Total SNOMED mappings:", len(snomed_to_diag))

# =========================
# PTB SUPERCLASS MAP
# =========================

MI_SET = {
    "MI","AMI","IMI","ASMI","ALMI","IPLMI","IPMI","LMI","PMI"
}

STTC_SET = {
    "STTC","STD","STE","TINV"
}

CD_SET = {
    "AVB","1AVB","2AVB","2AVB1","2AVB2","3AVB",
    "LBBB","RBBB","CLBBB","CRBBB",
    "ILBBB","IRBBB","LAFB","LPFB","IVCD"
}

HYP_SET = {
    "LVH","RVH","LAH","RAH","HYP"
}

def map_superclass(acronym):

    acronym = str(acronym).upper()

    if acronym in MI_SET:
        return 1

    if acronym in STTC_SET:
        return 2

    if acronym in CD_SET:
        return 3

    if acronym in HYP_SET:
        return 4

    return None


counts = np.zeros(5)
mapped = 0
multi = 0

for _, row in df.iterrows():

    codes = str(row["scp_codes"])

    label_vec = np.zeros(5)

    for code in snomed_to_diag:

        if code in codes:

            acr = snomed_to_diag[code]
            cls = map_superclass(acr)

            if cls is not None:
                label_vec[cls] = 1

    if label_vec.sum() > 0:

        if label_vec[1:].sum() == 0:
            label_vec[0] = 1

        counts += label_vec
        mapped += 1

        if label_vec.sum() > 1:
            multi += 1


print("\nSuperclass distribution")
print("NORM:", int(counts[0]))
print("MI  :", int(counts[1]))
print("STTC:", int(counts[2]))
print("CD  :", int(counts[3]))
print("HYP :", int(counts[4]))

print("\nMapped:", mapped)
print("Coverage:", round(mapped/len(df)*100,2),"%")
print("Multilabel:", multi)