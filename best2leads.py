import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataloader import PTBXLDataset
from models.resnet1d import ResNet18_1D


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
NUM_EPOCHS  = 50

# Lead I = index 0, aVR = index 3
LEAD_INDICES = [0, 3]
LEAD_NAMES   = ['I', 'aVR']
MODEL_NAME   = "aVR_LeadI"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("lead_based_models", exist_ok=True)

print(f"\n==============================")
print(f"Training 2-lead model: {' + '.join(LEAD_NAMES)}")
print(f"==============================")

# =========================
# Load Dataset
# =========================
dataset = PTBXLDataset('./data')

X_train, y_train = dataset.prepare_dataset('train')
X_val,   y_val   = dataset.prepare_dataset('val')
X_test,  y_test  = dataset.prepare_dataset('test')

# Select the two leads → shape (N, 2, 1000)
X_train = X_train[:, LEAD_INDICES, :]
X_val   = X_val[:,   LEAD_INDICES, :]
X_test  = X_test[:,  LEAD_INDICES, :]

print("Train shape after lead selection:", X_train.shape)
print("Val shape:  ", X_val.shape)
print("Test shape: ", X_test.shape)

# =========================
# Class imbalance weights
# Identical to train12lead.py and leadtrain.py
# =========================
y_train_tensor = torch.FloatTensor(y_train)
pos_counts = y_train_tensor.sum(dim=0)
neg_counts  = len(y_train_tensor) - pos_counts
pos_weight  = (neg_counts / pos_counts.clamp(min=1)).to(device)

print("\nClass counts (train):")
for i, c in enumerate(CLASS_NAMES):
    print(f"  {c}: {int(pos_counts[i])} pos  |  pos_weight={pos_weight[i]:.2f}")

# =========================
# DataLoaders
# =========================
train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
    batch_size=128,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
    batch_size=128,
    shuffle=False,
    num_workers=0
)

# =========================
# Model — in_channels=2 for the 2-lead input
# All other hyperparameters matched exactly to train12lead.py / leadtrain.py
# =========================
model = ResNet18_1D(in_channels=2, num_classes=5).to(device)

for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.p = 0.3

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# =========================
# Loss, Optimizer, Scheduler
# All matched to the other two scripts
# =========================
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
)

best_val_f1 = 0.0
best_epoch  = 0

# =========================
# Training Loop
# =========================
for epoch in range(NUM_EPOCHS):

    model.train()
    total_loss  = 0.0
    num_batches = 0

    for x_batch, y_batch in tqdm(train_loader,
                                  desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
                                  leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    scheduler.step()

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            preds   = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    pred_bin   = (all_preds > 0.5).astype(int)

    val_macro_f1  = f1_score(all_labels, pred_bin, average='macro', zero_division=0)
    val_per_class = f1_score(all_labels, pred_bin, average=None,    zero_division=0)
    avg_loss      = total_loss / num_batches
    cur_lr        = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch+1:>2}/{NUM_EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"LR: {cur_lr:.2e} | "
          f"Val Macro F1: {val_macro_f1:.4f} | "
          f"[NORM:{val_per_class[0]:.3f} "
          f"MI:{val_per_class[1]:.3f} "
          f"STTC:{val_per_class[2]:.3f} "
          f"CD:{val_per_class[3]:.3f} "
          f"HYP:{val_per_class[4]:.3f}]")

    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1
        best_epoch  = epoch + 1

        torch.save({
            'model_state_dict':  model.state_dict(),
            'lead_indices':      LEAD_INDICES,
            'lead_names':        LEAD_NAMES,
            'epoch':             best_epoch,
            'best_val_macro_f1': best_val_f1,
            'val_per_class_f1':  val_per_class.tolist(),
        }, f"lead_based_models/best_{MODEL_NAME}.pth")

        print(f"  ✓ Saved best model (epoch {best_epoch}, val F1={best_val_f1:.4f})")

print(f"\nBest Val Macro F1: {best_val_f1:.4f}  (epoch {best_epoch})")

# =========================
# Test Evaluation (PTB-XL ID)
# =========================
checkpoint = torch.load(f"lead_based_models/best_{MODEL_NAME}.pth",
                        map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        preds   = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y_batch.numpy())

all_preds  = np.vstack(all_preds)
all_labels = np.vstack(all_labels)
pred_bin   = (all_preds > 0.5).astype(int)

test_macro_f1  = f1_score(all_labels, pred_bin, average='macro', zero_division=0)
test_per_class = f1_score(all_labels, pred_bin, average=None,    zero_division=0)

print(f"\n{'='*45}")
print(f"  {MODEL_NAME} PTB-XL Test Results (ID)")
print(f"{'='*45}")
print(f"  Macro F1: {test_macro_f1:.4f}")
print(f"  Per-class F1:")
for i, c in enumerate(CLASS_NAMES):
    print(f"    {c:6s}: {test_per_class[i]:.4f}")

# =========================
# Save Results
# =========================
results = {
    "model":            f"ResNet18_1D_{MODEL_NAME}",
    "leads":            LEAD_NAMES,
    "lead_indices":     LEAD_INDICES,
    "best_val_epoch":   best_epoch,
    "val_macro_f1":     float(best_val_f1),
    "test_macro_f1_id": float(test_macro_f1),
    "test_per_class_f1_id": {
        CLASS_NAMES[i]: float(test_per_class[i]) for i in range(5)
    }
}

with open(f"lead_based_models/results_{MODEL_NAME}.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"\n  ✓ Saved results_{MODEL_NAME}.json")