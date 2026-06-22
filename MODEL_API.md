# Model API — Inference Guide

## Input: 41 features (in exact order)

The model expects a single row of 41 numerical features, in this exact order:

| # | Column | Type | Description |
|---|--------|------|-------------|
| 0 | `L4_SRC_PORT` | int32 | Source port |
| 1 | `L4_DST_PORT` | int32 | Destination port |
| 2 | `PROTOCOL` | int16 | IP protocol (6=TCP, 17=UDP, 1=ICMP) |
| 3 | `L7_PROTO` | float32 | L7 protocol ID |
| 4 | `IN_BYTES` | int32 | Incoming bytes |
| 5 | `IN_PKTS` | int16 | Incoming packets |
| 6 | `OUT_BYTES` | int32 | Outgoing bytes |
| 7 | `OUT_PKTS` | int16 | Outgoing packets |
| 8 | `TCP_FLAGS` | int8 | Aggregate TCP flags |
| 9 | `CLIENT_TCP_FLAGS` | int8 | Client-side TCP flags |
| 10 | `SERVER_TCP_FLAGS` | int8 | Server-side TCP flags |
| 11 | `FLOW_DURATION_MILLISECONDS` | int32 | Flow duration in ms |
| 12 | `DURATION_IN` | int16 | Incoming duration |
| 13 | `DURATION_OUT` | int16 | Outgoing duration |
| 14 | `MIN_TTL` | int16 | Minimum TTL |
| 15 | `MAX_TTL` | int16 | Maximum TTL |
| 16 | `LONGEST_FLOW_PKT` | int16 | Longest packet in flow |
| 17 | `SHORTEST_FLOW_PKT` | int16 | Shortest packet in flow |
| 18 | `MIN_IP_PKT_LEN` | int16 | Min IP packet length |
| 19 | `MAX_IP_PKT_LEN` | int16 | Max IP packet length |
| 20 | `SRC_TO_DST_SECOND_BYTES` | float32 | Source to destination bytes/sec |
| 21 | `DST_TO_SRC_SECOND_BYTES` | float32 | Destination to source bytes/sec |
| 22 | `RETRANSMITTED_IN_BYTES` | int32 | Retransmitted incoming bytes |
| 23 | `RETRANSMITTED_IN_PKTS` | int16 | Retransmitted incoming packets |
| 24 | `RETRANSMITTED_OUT_BYTES` | int32 | Retransmitted outgoing bytes |
| 25 | `RETRANSMITTED_OUT_PKTS` | int16 | Retransmitted outgoing packets |
| 26 | `SRC_TO_DST_AVG_THROUGHPUT` | int64 | Average throughput src→dst |
| 27 | `DST_TO_SRC_AVG_THROUGHPUT` | int64 | Average throughput dst→src |
| 28 | `NUM_PKTS_UP_TO_128_BYTES` | int16 | Packets ≤ 128 bytes |
| 29 | `NUM_PKTS_128_TO_256_BYTES` | int16 | Packets 128–256 bytes |
| 30 | `NUM_PKTS_256_TO_512_BYTES` | int16 | Packets 256–512 bytes |
| 31 | `NUM_PKTS_512_TO_1024_BYTES` | int16 | Packets 512–1024 bytes |
| 32 | `NUM_PKTS_1024_TO_1514_BYTES` | int16 | Packets 1024–1514 bytes |
| 33 | `TCP_WIN_MAX_IN` | int32 | Max TCP window incoming |
| 34 | `TCP_WIN_MAX_OUT` | int32 | Max TCP window outgoing |
| 35 | `ICMP_TYPE` | int32 | ICMP type |
| 36 | `ICMP_IPV4_TYPE` | int16 | ICMPv4 type |
| 37 | `DNS_QUERY_ID` | int32 | DNS query ID |
| 38 | `DNS_QUERY_TYPE` | int32 | DNS query type |
| 39 | `DNS_TTL_ANSWER` | int64 | DNS TTL answer |
| 40 | `FTP_COMMAND_RET_CODE` | float32 | FTP return code |

## Preprocessing

Before passing data to the model, you **must** apply the same `StandardScaler` that was fitted during training:

```python
import joblib
import numpy as np

scaler = joblib.load("path/to/scaler.pkl")

# row must be a 2D array: shape (1, 41) or (N, 41)
row_scaled = scaler.transform(row)
```

The scaler standardises each feature to zero mean and unit variance using the training set statistics.

## Inference

```python
import torch
from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TrafficClassifier(input_dim=41).to(device)
state_dict = load_file("path/to/model.safetensors", device=str(device))
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    tensor = torch.tensor(row_scaled, dtype=torch.float32).to(device)
    logits = model(tensor)                         # shape: (1, 2)
    prob = torch.softmax(logits, dim=1)[:, 1]      # suspicious probability
    score = prob.item()                             # float between 0.0 and 1.0
```

> **Note:** The model outputs logits, not probabilities. Always apply `softmax` to get a probability between 0 and 1.

## Output interpretation

- **Score close to 0.0** → benign
- **Score close to 1.0** → suspicious

Apply a threshold to decide:

```python
THRESHOLD = 0.50    # adjust based on your tolerance
is_suspicious = score >= THRESHOLD
```

### Recommended thresholds per model

| Model | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| v1 (256→128→64) | 0.95 | 0.893 | 0.992 | 0.940 |
| **v2 (512→256→128→64)** | **0.50** | **0.930** | **0.970** | **0.950** |
| v3 (512→256→128→64 weighted) | 0.95 | 0.907 | 0.984 | 0.944 |

## Architecture (v2 / v3)

```
Input(41) → Linear(512) → BatchNorm → ReLU → Dropout(0.3)
         → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
         → Linear(128) → BatchNorm → ReLU → Dropout(0.2)
         → Linear(64)  → BatchNorm → ReLU → Dropout(0.1)
         → Linear(2)   → softmax → [benign_prob, suspicious_prob]
```

Total parameters: ~269,000.

## Common mistakes

- **Don't** pass raw features without scaling — the model was trained on standardised data
- **Don't** skip any of the 41 features — the order and count must match exactly
- **Don't** include the `Attack` or `Label` columns — they are not inputs
- **Do** use `float32` — the model weights are float32
- **Do** call `model.eval()` before inference to disable dropout
