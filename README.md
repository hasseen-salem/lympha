# Lympha — Network Traffic Anomaly Detection

Lightweight neural network classifier that flags network flows as benign or suspicious, trained on the NF-UNSW-NB15-v2 dataset.

## Project Structure

```
├── train_model.py         # Training script
├── evaluate_model.py      # Evaluation / inference test script
├── Training_data/
│   └── NF-UNSW-NB15-V2.parquet
├── output/                # Trained artifacts (created by train_model.py)
│   ├── model.safetensors  # Final model weights
│   ├── scaler.pkl         # Fitted StandardScaler
│   ├── model_info.txt     # Input dimension metadata
│   └── checkpoints/       # Checkpoints (model + optimizer + epoch)
├── src/                   # Future: feature extraction & inference pipeline
└── README.md
```

## Requirements

- Python 3.8+
- pandas, numpy, torch, scikit-learn, safetensors, joblib, pyarrow

Install:

```bash
pip install pandas numpy torch scikit-learn safetensors joblib pyarrow
```

## Training

```bash
python3 train_model.py
```

The script:
1. Loads the parquet dataset and drops string identifiers (`Attack`)
2. Performs stratified balanced sampling (50k benign + 50k suspicious)
3. Splits 80/20, normalises with `StandardScaler`
4. Trains a 4-layer MLP for 30 epochs
5. Saves checkpoints every 5 epochs (`output/checkpoints/`)
6. Saves final weights as `output/model.safetensors`

**Resuming:** If interrupted, re-run `train_model.py` — it loads the latest checkpoint and continues.

## Evaluation

```bash
python3 evaluate_model.py
```

Evaluates the trained model on the **full dataset** (~2M flows) and reports accuracy, precision, recall, F1, confusion matrix, and a threshold sweep for tuning false-alarm rates.

## Model Architecture

| Layer    | Units | Activation | Regularisation |
|----------|-------|------------|----------------|
| Input    | 41    | —          | —              |
| Hidden 1 | 256   | ReLU       | BatchNorm, Dropout 0.3 |
| Hidden 2 | 128   | ReLU       | BatchNorm, Dropout 0.3 |
| Hidden 3 | 64    | ReLU       | BatchNorm, Dropout 0.2 |
| Output   | 2     | —          | —              |

Optimiser: Adam (LR 1e-3) with ReduceLROnPlateau scheduler.
