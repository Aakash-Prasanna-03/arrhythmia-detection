import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score, classification_report
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


def train_single_lead(LEAD_INDEX):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs("lead_based_models", exist_ok=True)

    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    lead_name = lead_names[LEAD_INDEX]
    print(f"\n==============================")
    print(f"Training single lead model for: {lead_name}")
    print(f"==============================")

    # =========================
    # Load Dataset
    # =========================
    dataset = PTBXLDataset('./data')

    X_train, y_train = dataset.prepare_dataset('train')
    X_val, y_val = dataset.prepare_dataset('val')
    X_test, y_test = dataset.prepare_dataset('test')

    # Select single lead
    X_train = X_train[:, [LEAD_INDEX], :]
    X_val   = X_val[:, [LEAD_INDEX], :]
    X_test  = X_test[:, [LEAD_INDEX], :]

    print("Train shape after lead selection:", X_train.shape)

    # =========================
    # DataLoaders
    # =========================
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

    # =========================
    # Model
    # =========================
    model = ResNet18_1D(in_channels=1, num_classes=5).to(device)

    print("Total parameters:",
          sum(p.numel() for p in model.parameters()))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1 = 0

    # =========================
    # Training Loop
    # =========================
    for epoch in range(40):

        model.train()
        total_loss = 0

        for x_batch, y_batch in tqdm(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ===== Validation =====
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

        print(f"Epoch {epoch+1}/40 | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Macro F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1

            save_path = f"lead_based_models/best_lead_{lead_name}.pth"

            torch.save({
                'model_state_dict': model.state_dict(),
                'lead_index': LEAD_INDEX,
                'lead_name': lead_name,
                'best_val_macro_f1': best_f1
            }, save_path)

            print("✓ Saved best model")

    print("\nBest Validation Macro F1:", best_f1)

    # =========================
    # Test Evaluation (ID)
    # =========================
    checkpoint = torch.load(f"lead_based_models/best_lead_{lead_name}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
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

    print("\nTest Macro F1 (ID):", macro_f1)

    results = {
        "lead_index": LEAD_INDEX,
        "lead_name": lead_name,
        "val_macro_f1": float(best_f1),
        "test_macro_f1_id": float(macro_f1)
    }

    with open(f"lead_based_models/results_lead_{lead_name}.json", "w") as f:
        json.dump(results, f, indent=4)

    print("✓ Metrics saved\n")


# =========================
# Train All 12 Leads
# =========================
if __name__ == "__main__":
    for i in range(12):
        train_single_lead(i)