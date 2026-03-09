import torch
import numpy as np
import json
import sys
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from models.resnet1d import ResNet18_1D

# =========================
# DATASET SELECTION
# =========================

if len(sys.argv) < 2:
    print("Usage: python eval_avr_lead1.py [chapman | georgia]")
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
# Lead I = index 0, aVR = index 3
# Must match LEAD_INDICES used in train_avr_lead1.py
# =========================

LEAD_INDICES = [0, 3]
LEAD_NAMES   = ['I', 'aVR']
MODEL_NAME   = "aVR_LeadI"
MODEL_PATH   = f"lead_based_models/best_{MODEL_NAME}.pth"

class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================

X = np.load(X_PATH)   # shape (N, 12, 1000)
y = np.load(Y_PATH)   # shape (N, 5)

# Select the same two leads used during training
X = X[:, LEAD_INDICES, :]   # → (N, 2, 1000)

print("\n==============================")
print(f"Dataset : {dataset.upper()}")
print(f"Leads   : {LEAD_NAMES}  (indices {LEAD_INDICES})")
print(f"Signals : {X.shape}")
print(f"Labels  : {y.shape}")
print(f"Device  : {device}")
print("==============================")

# =========================
# DATA LOADER
# =========================

loader = DataLoader(
    TensorDataset(torch.FloatTensor(X)),
    batch_size=256,
    shuffle=False
)

# =========================
# LOAD MODEL
# =========================

model = ResNet18_1D(in_channels=2, num_classes=5).to(device)

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
        preds   = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(preds)

preds       = np.vstack(all_preds)
pred_binary = (preds >= 0.5).astype(int)

# =========================
# METRICS
# =========================

macro_f1     = f1_score(y, pred_binary, average="macro", zero_division=0)
per_class_f1 = f1_score(y, pred_binary, average=None,    zero_division=0)

print("\n===== RESULTS =====")
print(f"Model  : {MODEL_NAME}  ({' + '.join(LEAD_NAMES)})")
print(f"Macro F1: {macro_f1:.4f}")
print("\nPer-class F1:")
for i, c in enumerate(class_names):
    print(f"  {c:6s}: {per_class_f1[i]:.4f}")

# =========================
# Save Results
# =========================

results_dict = {
    "model":         MODEL_NAME,
    "leads":         LEAD_NAMES,
    "dataset":       dataset,
    "macro_f1":      float(macro_f1),
    "per_class_f1":  {c: float(per_class_f1[i]) for i, c in enumerate(class_names)}
}

out_path = f"results_{MODEL_NAME}_{dataset}.json"
with open(out_path, "w") as f:
    json.dump(results_dict, f, indent=2)

print(f"\n  ✓ Saved {out_path}")