import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import json
import os
import random

from dataloader import PTBXLDataset
from models.resnet1d import ResNet18_1D


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("lead_based_models", exist_ok=True)

# Load dataset
dataset = PTBXLDataset('./data')

X_train, y_train = dataset.prepare_dataset('train')
X_val, y_val = dataset.prepare_dataset('val')
X_test, y_test = dataset.prepare_dataset('test')

print("Train shape:", X_train.shape)

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
    batch_size=128,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
    batch_size=128,
    shuffle=False
)

model = ResNet18_1D(in_channels=12, num_classes=5).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

best_f1 = 0

for epoch in range(40):

    model.train()
    for x_batch, y_batch in tqdm(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            preds = torch.sigmoid(outputs).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(y_batch.numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    pred_binary = (all_preds > 0.5).astype(int)

    f1 = f1_score(all_labels, pred_binary,
                  average='macro', zero_division=0)

    print(f"Epoch {epoch+1}/40 | Val Macro F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "lead_based_models/best_12lead.pth")

print("\nBest Val F1:", best_f1)

# Test Evaluation
model.load_state_dict(torch.load("lead_based_models/best_12lead.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        preds = torch.sigmoid(outputs).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(y_batch.numpy())

all_preds = np.vstack(all_preds)
all_labels = np.vstack(all_labels)
pred_binary = (all_preds > 0.5).astype(int)

macro_f1 = f1_score(all_labels, pred_binary,
                    average='macro', zero_division=0)

print("12-Lead Test Macro F1 (ID):", macro_f1)

with open("lead_based_models/results_12lead.json", "w") as f:
    json.dump({"test_macro_f1_id": float(macro_f1)}, f, indent=4)