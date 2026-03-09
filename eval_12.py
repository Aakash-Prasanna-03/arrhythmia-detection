import torch
import numpy as np
import json
import sys
from torch.utils.data import DataLoader, TensorDataset
from models.resnet1d import ResNet18_1D
from metrics import compute_metrics, print_metrics

# =========================
# DATASET SELECTION
# =========================

if len(sys.argv) < 2:
    print("Usage: python eval12.py [chapman | georgia]")
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

MODEL_PATH  = "lead_based_models/best_12lead.pth"
class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("\n==============================")
print(f"Dataset : {dataset.upper()}")
print(f"Signals : {X.shape}")
print(f"Labels  : {y.shape}")
print(f"Device  : {device}")
print("==============================")

# =========================
# LOAD MODEL
# =========================

model      = ResNet18_1D(in_channels=12, num_classes=5).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
print("Checkpoint keys:", checkpoint.keys())
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# =========================
# INFERENCE
# =========================

loader    = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=256, shuffle=False)
all_proba = []

with torch.no_grad():
    for (x_batch,) in loader:
        outputs = model(x_batch.to(device))
        all_proba.append(torch.sigmoid(outputs).cpu().numpy())

y_proba  = np.vstack(all_proba)
y_binary = (y_proba >= 0.5).astype(int)

# =========================
# METRICS
# =========================

m = compute_metrics(y, y_binary, y_proba)
print_metrics(m, label=f"12-Lead → {dataset.upper()}")

# =========================
# SAVE
# =========================

out = {
    "model":           "ResNet18_1D_12lead",
    "dataset":         dataset,
    "macro_f1":        m["macro_f1"],
    "macro_precision": m["macro_precision"],
    "macro_recall":    m["macro_recall"],
    "macro_auc":       m["macro_auc"],
    "hamming_loss":    m["hamming_loss"],
    "subset_accuracy": m["subset_accuracy"],
    "per_class":       m["per_class"],
}

out_path = f"results_12lead_{dataset}.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n  ✓ Saved {out_path}")
