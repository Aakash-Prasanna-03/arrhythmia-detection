import torch
import numpy as np
import json
import sys
from torch.utils.data import DataLoader, TensorDataset
from models.resnet1d import ResNet18_1D
from metrics import compute_metrics, print_metrics

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

LEAD_INDICES = [0, 3]
LEAD_NAMES   = ['I', 'aVR']
MODEL_NAME   = "aVR_LeadI"
MODEL_PATH   = f"lead_based_models/best_{MODEL_NAME}.pth"
class_names  = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = np.load(X_PATH)[:, LEAD_INDICES, :]
y = np.load(Y_PATH)

print("\n==============================")
print(f"Dataset : {dataset.upper()}")
print(f"Leads   : {LEAD_NAMES}  (indices {LEAD_INDICES})")
print(f"Signals : {X.shape}")
print(f"Labels  : {y.shape}")
print(f"Device  : {device}")
print("==============================")

model      = ResNet18_1D(in_channels=2, num_classes=5).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
print("Checkpoint keys:", checkpoint.keys())
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

loader    = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=256, shuffle=False)
all_proba = []

with torch.no_grad():
    for (x_batch,) in loader:
        outputs = model(x_batch.to(device))
        all_proba.append(torch.sigmoid(outputs).cpu().numpy())

y_proba  = np.vstack(all_proba)
y_binary = (y_proba >= 0.5).astype(int)

m = compute_metrics(y, y_binary, y_proba)
print_metrics(m, label=f"{MODEL_NAME} ({' + '.join(LEAD_NAMES)}) -> {dataset.upper()}")

out = {
    "model":           MODEL_NAME,
    "leads":           LEAD_NAMES,
    "dataset":         dataset,
    "macro_f1":        m["macro_f1"],
    "macro_precision": m["macro_precision"],
    "macro_recall":    m["macro_recall"],
    "macro_auc":       m["macro_auc"],
    "hamming_loss":    m["hamming_loss"],
    "subset_accuracy": m["subset_accuracy"],
    "per_class":       m["per_class"],
}

out_path = f"results_{MODEL_NAME}_{dataset}.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\n  Saved {out_path}")
