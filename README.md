# Lympha — Network Traffic Anomaly Detection

Lightweight neural network classifier that flags network flows as benign or suspicious, trained on the NF-CSE-CIC-IDS2018-v2 dataset.

## Project Structure

```
├── train_model.py         # Training script
├── evaluate_model.py      # Evaluation / inference test script
├── inspect_flows.py       # Interactive flow-by-flow inspection
├── Training_data/
│   └── NF-CSE-CIC-IDS2018-v2.parquet
├── output/                # Local artifacts (cached splits, etc.)
├── src/                   # Future: feature extraction & inference pipeline
└── README.md
```

Training outputs (saved to Google Drive):
- `/content/drive/MyDrive/Lympha/model.safetensors`
- `/content/drive/MyDrive/Lympha/scaler.pkl`
- `/content/drive/MyDrive/Lympha/test_set.parquet`
- `/content/drive/MyDrive/Lympha/train_set.parquet`
- `/content/drive/MyDrive/Lympha/checkpoints/`

## Requirements

- Python 3.8+
- pandas, numpy, torch, scikit-learn, safetensors, joblib, pyarrow

```bash
pip install pandas numpy torch scikit-learn safetensors joblib pyarrow
```

## Training

```bash
python3 train_model.py
```

The script:
1. Loads the parquet dataset and drops string identifiers (`Attack`)
2. Splits 80/20 (stratified), caches splits to disk for reuse
3. Normalises with `StandardScaler`
4. Trains a lightweight 3-layer MLP for 30 epochs
5. Saves checkpoints every 5 epochs
6. Saves final weights as `model.safetensors`

**Resuming:** Re-run — it loads the latest checkpoint and continues.
**Caching:** On second run, it skips the split step and loads cached train/test sets.

## Memory optimisations

- DataFrames and arrays are deleted and garbage-collected after use
- Train/test splits are cached to disk so the full dataset is only loaded once
- Batch size reduced to 128

## Evaluation

```bash
python3 evaluate_model.py
```

Reports accuracy, precision, recall, F1, confusion matrix, and threshold sweep.

## Model Architecture

| Layer    | Units | Activation | Regularisation |
|----------|-------|------------|----------------|
| Input    | 41    | —          | —              |
| Hidden 1 | 128   | ReLU       | BatchNorm, Dropout 0.3 |
| Hidden 2 | 64    | ReLU       | BatchNorm, Dropout 0.2 |
| Output   | 2     | —          | —              |

Optimiser: Adam (LR 1e-3) with ReduceLROnPlateau scheduler.
