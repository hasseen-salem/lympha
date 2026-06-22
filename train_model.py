import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from safetensors.torch import save_file
import joblib
import os
import glob

DATA_PATH = "/content/drive/MyDrive/Lympha/Training-data/NF-CSE-CIC-IDS2018-v2.parquet"
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42
CHECKPOINT_DIR = "/content/drive/MyDrive/Lympha/checkpoints"
CHECKPOINT_INTERVAL = 5

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

COLS_TO_DROP = ["Attack"]


class TrafficClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def find_latest_checkpoint():
    ckpts = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pt")))
    return ckpts[-1] if ckpts else None


def save_checkpoint(model, optimizer, epoch, loss, is_best=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        ckpt_path,
    )
    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            best_path,
        )
    print(f"  [Checkpoint] Saved checkpoint_epoch_{epoch}.pt", flush=True)


def main():
    print("Loading dataset...", flush=True)
    df = pd.read_parquet(DATA_PATH)
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    print(f"Dataset shape: {df.shape}", flush=True)
    print(f"Label distribution:\n{df['Label'].value_counts()}", flush=True)

    # 80/20 stratified split on the full dataset
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["Label"]
    )

    print(f"Train: {len(train_df):,}  Test: {len(test_df):,}", flush=True)

    # save test set for later inspection
    os.makedirs("/content/drive/MyDrive/Lympha", exist_ok=True)
    test_df.to_parquet("/content/drive/MyDrive/Lympha/test_set.parquet", index=False)
    print(f"Test set saved to /content/drive/MyDrive/Lympha/test_set.parquet", flush=True)

    X_train = train_df.drop(columns=["Label"]).values.astype(np.float32)
    y_train = train_df["Label"].values.astype(np.int64)
    X_test = test_df.drop(columns=["Label"]).values.astype(np.float32)
    y_test = test_df["Label"].values.astype(np.int64)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, "/content/drive/MyDrive/Lympha/scaler.pkl")
    print("Scaler saved to /content/drive/MyDrive/Lympha/scaler.pkl", flush=True)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TrafficClassifier(input_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    start_epoch = 1
    best_val_loss = float("inf")

    latest_ckpt = find_latest_checkpoint()
    if latest_ckpt:
        print(f"Resuming from checkpoint: {latest_ckpt}", flush=True)
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["loss"]
        print(f"Resumed at epoch {checkpoint['epoch']}", flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_loss /= len(test_loader.dataset)
        acc = correct / total * 100
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:2d}/{EPOCHS}  "
            f"Train Loss: {train_loss:.4f}  "
            f"Val Loss: {val_loss:.4f}  "
            f"Val Acc: {acc:.2f}%",
            flush=True,
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        if epoch % CHECKPOINT_INTERVAL == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, val_loss, is_best=is_best)

    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, "/content/drive/MyDrive/Lympha/model.safetensors")
    print("Model saved to /content/drive/MyDrive/Lympha/model.safetensors", flush=True)

    with open("/content/drive/MyDrive/Lympha/model_info.txt", "w") as f:
        f.write(f"input_dim={X_train.shape[1]}\n")

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
