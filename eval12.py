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
    print("Usage: python eval_12lead.py [chapman | georgia]")
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

MODEL_PATH = "lead_based_models/best_12lead.pth"

class_names = ['NORM','MI','STTC','CD','HYP']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("\n==============================")
print("Dataset:", dataset.upper())
print("Signals:", X.shape)
print("Labels :", y.shape)
print("Device :", device)
print("==============================")

# =========================
# DATA LOADER
# =========================

dataset_tensor = TensorDataset(torch.FloatTensor(X))
loader = DataLoader(dataset_tensor, batch_size=256, shuffle=False)

# =========================
# LOAD MODEL
# =========================

model = ResNet18_1D(in_channels=12, num_classes=5).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
print("Checkpoint keys:", checkpoint.keys())
model.load_state_dict(checkpoint["model_state_dict"])

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

macro_f1 = f1_score(
    y,
    pred_binary,
    average="macro",
    zero_division=0
)

per_class_f1 = f1_score(
    y,
    pred_binary,
    average=None,
    zero_division=0
)

print("\n===== RESULTS =====")
print("Macro F1:", macro_f1)

print("\nPer-class F1:")
for i, c in enumerate(class_names):
    print(c, ":", per_class_f1[i])

# Save results to JSON
results_dict = {
    "macro_f1": float(macro_f1),
    "per_class_f1": {c: float(per_class_f1[i]) for i, c in enumerate(class_names)}
}
with open("results_12lead.json", "w") as f:
    json.dump(results_dict, f, indent=2)

