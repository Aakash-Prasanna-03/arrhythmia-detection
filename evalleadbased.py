import torch
import numpy as np
import json
import os
import sys
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from models.resnet1d import ResNet18_1D
from metrics import compute_metrics, print_metrics

# =========================
# DATASET SELECTION
# =========================

if len(sys.argv) < 2:
    print("Usage: python eval.py [chapman | georgia]")
    sys.exit(1)

dataset = sys.argv[1].lower()

if dataset == "chapman":
    X_PATH = "data/chapman/X_chapman.npy"
    Y_PATH = "data/chapman/y_chapman.npy"

elif dataset == "georgia":
    X_PATH = "data/georgia/X_georgia.npy"
    Y_PATH = "data/georgia/y_georgia.npy"

else:
    print("Invalid dataset. Use: chapman or georgia")
    sys.exit(1)

# =========================
# CONFIG
# =========================

MODEL_DIR = "lead_based_models"
SAVE_JSON = os.path.join(MODEL_DIR, f"{dataset}_ood_comparison.json")
SAVE_CSV  = os.path.join(MODEL_DIR, f"{dataset}_ood_comparison.csv")

lead_names  = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
               'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("\n=================================")
print("Dataset     :", dataset.upper())
print("Signal shape:", X.shape)
print("Label shape :", y.shape)
print("Device      :", device)
print("=================================")

print("\nClass distribution:")
for i, c in enumerate(class_names):
    print(f"  {c}: {int(y[:, i].sum())}")

results      = {}
rows_for_csv = []

# =========================
# EVALUATE EACH LEAD
# =========================

for lead_idx in range(12):

    lead_name = lead_names[lead_idx]

    print(f"\n==============================")
    print(f"Evaluating Lead: {lead_name}")
    print(f"==============================")

    X_lead = X[:, [lead_idx], :]

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_lead)),
        batch_size=256, shuffle=False
    )

    model = ResNet18_1D(in_channels=1, num_classes=5).to(device)
    checkpoint = torch.load(
        os.path.join(MODEL_DIR, f"best_lead_{lead_name}.pth"),
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    all_proba = []
    with torch.no_grad():
        for (x_batch,) in loader:
            outputs = model(x_batch.to(device))
            all_proba.append(torch.sigmoid(outputs).cpu().numpy())

    y_proba  = np.vstack(all_proba)
    y_binary = (y_proba >= 0.5).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    m = compute_metrics(y, y_binary, y_proba)
    print_metrics(m, label=f"Lead {lead_name} → {dataset.upper()}")

    # ── Load PTB-XL ID F1 ─────────────────────────────────────────────────────
    id_f1 = None
    id_json_path = os.path.join(MODEL_DIR, f"results_lead_{lead_name}.json")
    if os.path.exists(id_json_path):
        with open(id_json_path) as f:
            id_f1 = json.load(f)["test_macro_f1_id"]

    drop = (id_f1 - m["macro_f1"]) if id_f1 is not None else None

    print(f"\n  PTB-XL ID Macro F1 : {id_f1}")
    print(f"  Absolute Drop      : {drop:.4f}" if drop is not None else "  Absolute Drop: N/A")

    # ── Store ─────────────────────────────────────────────────────────────────
    results[lead_name] = {
        "ptb_id_macro_f1":             id_f1,
        f"{dataset}_ood_macro_f1":     m["macro_f1"],
        "absolute_drop":               float(drop) if drop is not None else None,
        "macro_precision":             m["macro_precision"],
        "macro_recall":                m["macro_recall"],
        "macro_auc":                   m["macro_auc"],
        "hamming_loss":                m["hamming_loss"],
        "subset_accuracy":             m["subset_accuracy"],
        "ood_per_class":               m["per_class"],
    }

    rows_for_csv.append({
        "Lead":                        lead_name,
        "PTB_ID_Macro_F1":             id_f1,
        f"{dataset.upper()}_OOD_Macro_F1": m["macro_f1"],
        "Macro_Precision":             m["macro_precision"],
        "Macro_Recall":                m["macro_recall"],
        "Macro_AUC":                   m["macro_auc"],
        "Hamming_Loss":                m["hamming_loss"],
        "Subset_Accuracy":             m["subset_accuracy"],
        "Absolute_Drop":               drop,
    })

# =========================
# SAVE
# =========================

with open(SAVE_JSON, "w") as f:
    json.dump(results, f, indent=4)

pd.DataFrame(rows_for_csv).to_csv(SAVE_CSV, index=False)

print(f"\n==============================")
print(f"Saved JSON : {SAVE_JSON}")
print(f"Saved CSV  : {SAVE_CSV}")
print(f"==============================")
