import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
import joblib
import os
import glob

TEST_PATH = "test_set.parquet"
MODELS_DIR = "models"
BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)


class TrafficClassifier_512(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)

class TrafficClassifier_256(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)

class TrafficClassifier_128(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 2),
        )
    def forward(self, x): return self.net(x)


ARCH_CLASSES = [
    ("512→256→128→64→2", TrafficClassifier_512),
    ("256→128→64→2", TrafficClassifier_256),
    ("128→64→2", TrafficClassifier_128),
]


def load_model(model_dir):
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    info_path = os.path.join(model_dir, "model_info.txt")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    with open(info_path) as f:
        input_dim = int(f.read().strip().split("=")[1])

    state_dict = load_file(safetensors_path, device="cpu")

    model = None
    arch_name = None
    for name, cls in ARCH_CLASSES:
        try:
            m = cls(input_dim)
            m.load_state_dict(state_dict, strict=True)
            model = m.to(device)
            arch_name = name
            break
        except (RuntimeError, KeyError):
            continue

    if model is None:
        raise RuntimeError(f"Could not load model from {model_dir}: unknown architecture")

    scaler = joblib.load(scaler_path)
    return model, scaler, arch_name


def evaluate(model, scaler, X, y):
    model.eval()
    X_scaled = scaler.transform(X)
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X_scaled), BATCH_SIZE):
            batch = torch.tensor(X_scaled[i:i+BATCH_SIZE], dtype=torch.float32).to(device)
            out = model(batch)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    probs = np.array(all_probs)

    results = {}
    for thresh in [0.3, 0.5, 0.7, 0.85, 0.9, 0.95]:
        preds = (probs >= thresh).astype(int)
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        tn = ((preds == 0) & (y == 0)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / len(y) * 100
        results[thresh] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "fp": fp, "fn": fn}

    benign_probs = probs[y == 0]
    sus_probs = probs[y == 1]

    return results, {
        "benign_mean": benign_probs.mean(),
        "benign_median": np.median(benign_probs),
        "sus_mean": sus_probs.mean(),
        "sus_median": np.median(sus_probs),
    }


def main():
    print("Loading test set...", flush=True)
    df = pd.read_parquet(TEST_PATH)
    X = df.drop(columns=["Label"]).values.astype(np.float32)
    y = df["Label"].values
    total = len(df)
    sus_count = (y == 1).sum()
    ben_count = total - sus_count
    print(f"Test set: {total:,} flows ({ben_count:,} benign / {sus_count:,} suspicious)\n", flush=True)

    model_dirs = sorted(glob.glob(os.path.join(MODELS_DIR, "v*")))
    if not model_dirs:
        print(f"No model directories found in {MODELS_DIR}/", flush=True)
        return

    all_results = []

    for model_dir in model_dirs:
        name = os.path.basename(model_dir)
        print(f"Evaluating {name}...", flush=True)
        try:
            model, scaler, arch = load_model(model_dir)
            results, dist = evaluate(model, scaler, X, y)
            all_results.append((name, arch, results, dist))
            print(f"  Architecture: {arch}  OK", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

    if not all_results:
        return

    # Print comparison table
    print("\n" + "=" * 140)
    header = f"{'Model':<8} {'Arch':<22} {'Threshold':<10} {'Acc%':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'FP':<7} {'FN':<7}"
    print(header)
    print("-" * 140)

    for name, arch, results, dist in all_results:
        first = True
        for thresh in [0.3, 0.5, 0.7, 0.85, 0.9, 0.95]:
            r = results[thresh]
            label = f"{name}" if first else ""
            print(
                f"{label:<8} {arch if first else '':<22} "
                f"{thresh:<10.2f} {r['acc']:<8.2f} {r['prec']:<8.4f} "
                f"{r['rec']:<8.4f} {r['f1']:<8.4f} {r['fp']:<7d} {r['fn']:<7d}",
                flush=True,
            )
            first = False

    # Summary: best threshold per model
    print("\n" + "=" * 140)
    print(f"{'Model':<8} {'Arch':<22} {'Best Thresh':<12} {'Acc%':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'FP':<7} {'FN':<7}")
    print("-" * 140)
    for name, arch, results, dist in all_results:
        # pick threshold that maximizes F1
        best_t = max(results, key=lambda t: results[t]["f1"])
        r = results[best_t]
        print(
            f"{name:<8} {arch:<22} {best_t:<12.2f} {r['acc']:<8.2f} {r['prec']:<8.4f} "
            f"{r['rec']:<8.4f} {r['f1']:<8.4f} {r['fp']:<7d} {r['fn']:<7d}",
            flush=True,
        )

    # Probability distributions
    print("\n" + "=" * 140)
    print(f"{'Model':<8} {'Benign mean':<12} {'Benign median':<14} {'Susp mean':<12} {'Susp median':<14}")
    print("-" * 140)
    for name, arch, results, dist in all_results:
        print(
            f"{name:<8} {dist['benign_mean']:<12.4f} {dist['benign_median']:<14.4f} "
            f"{dist['sus_mean']:<12.4f} {dist['sus_median']:<14.4f}",
            flush=True,
        )

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
