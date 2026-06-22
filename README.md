# Lympha — Network Traffic Anomaly Detection

Lightweight neural network classifier that flags network flows as benign or suspicious, trained on the NF-UNSW-NB15-v2 dataset.

## Project Structure

```
├── train_model.py         # Training script (Colab)
├── evaluate_model.py      # Single-model evaluation
├── local_test/            # Offline comparison of trained models
│   ├── compare_models.py  # Evaluate all models side by side
│   ├── test_set.parquet   # Held-out test set (397k flows)
│   ├── models/
│   │   ├── v1/            # 256→128→64  (no class weights)
│   │   ├── v2/            # 512→256→128→64  (no class weights)
│   │   └── v3/            # 512→256→128→64  (class weights 1:25)
│   └── README.md
├── output/                # Latest trained artifacts
├── Training_data/         # Raw datasets
├── requirements.txt
└── README.md
```

## Training

```bash
python3 train_model.py
```

- Loads the parquet dataset, drops `Attack` string column
- Splits 80/20 stratisfied, caches splits to Drive for reuse
- Normalises with `StandardScaler`
- Trains on 80% of full data
- Saves checkpoints every 5 epochs + best model
- Saves final weights as `model.safetensors`

**Resuming:** Re-run — loads latest checkpoint and continues from there.

**Class weights:** Edit `class_weights` in the script to adjust the recall/precision trade-off (default `[1.0, 25.0]`).

## Evaluation

```bash
python3 evaluate_model.py
```

Reports accuracy, precision, recall, F1, confusion matrix, and threshold sweep.

## Compare all models

```bash
cd local_test && python3 compare_models.py
```

Loads all models in `models/` and prints a side-by-side comparison table with metrics at multiple thresholds.

## Requirements

```
pip install pandas numpy torch scikit-learn safetensors joblib pyarrow
```
