import torch
import numpy as np
import json
import os
import sys
import pandas as pd
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from models.resnet1d import ResNet18_1D

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
SAVE_CSV = os.path.join(MODEL_DIR, f"{dataset}_ood_comparison.csv")

lead_names = ['I','II','III','aVR','aVL','aVF',
              'V1','V2','V3','V4','V5','V6']

class_names = ['NORM','MI','STTC','CD','HYP']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("\n=================================")
print("Dataset:", dataset.upper())
print("Signal shape:", X.shape)
print("Label shape:", y.shape)
print("Device:", device)
print("=================================")

# Print label distribution
print("\nClass distribution:")
for i, c in enumerate(class_names):
    print(c, ":", int(y[:, i].sum()))

results = {}
rows_for_csv = []

# =========================
# EVALUATE EACH LEAD
# =========================

for lead_idx in range(12):

    lead_name = lead_names[lead_idx]

    print("\n==============================")
    print("Evaluating Lead:", lead_name)
    print("==============================")

    # Extract single lead
    X_lead = X[:, [lead_idx], :]

    dataset_tensor = TensorDataset(torch.FloatTensor(X_lead))
    loader = DataLoader(dataset_tensor, batch_size=256, shuffle=False)

    # Load PTB trained model
    model = ResNet18_1D(in_channels=1, num_classes=5).to(device)

    checkpoint_path = os.path.join(
        MODEL_DIR,
        f"best_lead_{lead_name}.pth"
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # =========================
    # INFERENCE
    # =========================

    all_preds = []

    with torch.no_grad():
        for (x_batch,) in loader:

            x_batch = x_batch.to(device)

            outputs = model(x_batch)

            preds = torch.sigmoid(outputs).cpu().numpy()

            all_preds.append(preds)

    preds = np.vstack(all_preds)

    pred_binary = (preds >= 0.5).astype(int)

    # =========================
    # METRICS
    # =========================

    ood_macro_f1 = f1_score(
        y,
        pred_binary,
        average='macro',
        zero_division=0
    )

    ood_per_class_f1 = f1_score(
        y,
        pred_binary,
        average=None,
        zero_division=0
    )

    # =========================
    # LOAD PTB ID RESULTS
    # =========================

    id_json_path = os.path.join(
        MODEL_DIR,
        f"results_lead_{lead_name}.json"
    )

    id_f1 = None

    if os.path.exists(id_json_path):
        with open(id_json_path) as f:
            id_data = json.load(f)

        id_f1 = id_data["test_macro_f1_id"]

    drop = None

    if id_f1 is not None:
        drop = id_f1 - ood_macro_f1

    print("PTB ID Macro F1:", id_f1)
    print(f"{dataset.upper()} OOD Macro F1:", ood_macro_f1)
    print("Absolute Drop:", drop)

    # =========================
    # SAVE RESULTS IN MEMORY
    # =========================

    results[lead_name] = {
        "ptb_id_macro_f1": id_f1,
        f"{dataset}_ood_macro_f1": float(ood_macro_f1),
        "absolute_drop": float(drop) if drop is not None else None,
        "ood_per_class_f1": {
            class_names[i]: float(ood_per_class_f1[i])
            for i in range(len(class_names))
        }
    }

    rows_for_csv.append({
        "Lead": lead_name,
        "PTB_ID_Macro_F1": id_f1,
        f"{dataset.upper()}_OOD_Macro_F1": ood_macro_f1,
        "Absolute_Drop": drop
    })

# =========================
# SAVE RESULTS
# =========================

with open(SAVE_JSON, "w") as f:
    json.dump(results, f, indent=4)

df = pd.DataFrame(rows_for_csv)
df.to_csv(SAVE_CSV, index=False)

print("\n==============================")
print("Saved JSON to:", SAVE_JSON)
print("Saved CSV to:", SAVE_CSV)
print("==============================")

