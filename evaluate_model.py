import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
import joblib

DATA_PATH = "Training_data/test_set.parquet"
MODEL_PATH = "output/model.safetensors"
SCALER_PATH = "models/v1/scaler.pkl"
BATCH_SIZE = 256
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)


class TrafficClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("Loading dataset...", flush=True)
    df = pd.read_parquet(DATA_PATH)
    df = df.drop(columns=["Attack"], errors="ignore")

    X = df.drop(columns=["Label"]).values.astype(np.float32)
    y = df["Label"].values

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)

    with open("output/model_info.txt") as f:
        input_dim = int(f.read().strip().split("=")[1])

    model = TrafficClassifier(input_dim=input_dim).to(device)
    state_dict = load_file(MODEL_PATH, device=str(device))
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.", flush=True)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(np.int64)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print("\n========== Full Dataset Evaluation ==========")
    print(f"Total samples: {len(all_labels)}")
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"               Predicted 0    Predicted 1")
    print(f"Actual 0       {tn:>8d}        {fp:>8d}")
    print(f"Actual 1       {fn:>8d}        {tp:>8d}")

    benign_probs = all_probs[all_labels == 0]
    sus_probs = all_probs[all_labels == 1]

    print(f"\nProbability distribution (class 1 = suspicious):")
    print(f"  Benign samples:    mean={benign_probs.mean():.4f}, "
          f"median={np.median(benign_probs):.4f}, "
          f"std={benign_probs.std():.4f}")
    print(f"  Suspicious samples: mean={sus_probs.mean():.4f}, "
          f"median={np.median(sus_probs):.4f}, "
          f"std={sus_probs.std():.4f}")

    print(f"\nThreshold analysis (for adjustable threshold):")
    for thresh in [0.5, 0.7, 0.85, 0.9, 0.95]:
        preds_t = (all_probs >= thresh).astype(np.int64)
        acc_t = (preds_t == all_labels).mean() * 100
        tp_t = ((preds_t == 1) & (all_labels == 1)).sum()
        fp_t = ((preds_t == 1) & (all_labels == 0)).sum()
        fn_t = ((preds_t == 0) & (all_labels == 1)).sum()
        prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
        rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        print(f"  Threshold {thresh:.2f}: "
              f"Acc={acc_t:.2f}%  Prec={prec_t:.4f}  Rec={rec_t:.4f}  "
              f"FP={fp_t}  FN={fn_t}", flush=True)


if __name__ == "__main__":
    main()
