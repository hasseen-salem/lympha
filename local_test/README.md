# Model Comparison

Three trained models on the NF-UNSW-NB15-v2 dataset, evaluated against a held-out test set of 397,349 flows (382,333 benign / 15,016 suspicious).

## Models

### v1 — 256→128→64 (baseline)

- **Architecture:** `Linear(41,256) → BatchNorm → ReLU → Dropout(0.3) → Linear(256,128) → BatchNorm → ReLU → Dropout(0.3) → Linear(128,64) → BatchNorm → ReLU → Dropout(0.2) → Linear(64,2)`
- **Class weights:** None (1:1)
- **Training data:** 100k stratified sample → 80k train / 20k test
- **Best threshold:** 0.95
- **Character:** Good recall, moderate precision. The lightest model.

### v2 — 512→256→128→64 (best overall)

- **Architecture:** `Linear(41,512) → BatchNorm → ReLU → Dropout(0.3) → Linear(512,256) → BatchNorm → ReLU → Dropout(0.3) → Linear(256,128) → BatchNorm → ReLU → Dropout(0.2) → Linear(128,64) → BatchNorm → ReLU → Dropout(0.1) → Linear(64,2)`
- **Class weights:** None (1:1)
- **Training data:** Full dataset → 80% train / 20% test
- **Best threshold:** 0.50
- **Character:** Highest F1 (0.9495) and precision (0.9297). Best balance of low false alarms and high detection. Recommended for production.

### v3 — 512→256→128→64 (weighted)

- **Architecture:** Same as v2
- **Class weights:** 1:25 (benign:suspicious)
- **Training data:** Full dataset → 80% train / 20% test
- **Best threshold:** 0.95
- **Character:** Near-perfect recall (0.9838) with more false positives than v2. Use when missing an attack is far more costly than a false alarm.

## Results Summary

| Model | Best Thresh | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|---|---|
| v1 | 0.95 | 99.52% | 0.8929 | 0.9919 | 0.9398 | 1,787 | 122 |
| **v2** | **0.50** | **99.61%** | **0.9297** | **0.9701** | **0.9495** | **1,102** | **449** |
| v3 | 0.95 | 99.56% | 0.9065 | 0.9838 | 0.9436 | 1,523 | 243 |

## Running comparison

```bash
python3 compare_models.py
```
