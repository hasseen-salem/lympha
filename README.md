# Lympha — Real-Time Network Intrusion Prevention

Lympha is a live network intrusion prevention daemon that captures packets, aggregates them into bidirectional flows, scores them with an embedding-augmented neural network, and blocks threats via nftables — all in a single non-blocking pipeline.

## Architecture

```
┌─────────────┐    ┌───────────────┐    ┌────────────────┐    ┌──────────────────┐
│ PacketSniffer│───▶│ FlowAggregator│───▶│ InferenceEngine│───▶│ FirewallController│
│ (scapy)     │    │ O(1) LRU evict│    │ LymphaNet      │    │ nft -f - stdin   │
│ backpressure│    │ async sweeper │    │ 80-dim fused   │    │ shell=False      │
│ thread-safe │    │ 43 features   │    │ Embed + MLP    │    │ IP validation    │
└─────────────┘    └───────────────┘    └────────────────┘    └──────────────────┘
                                                                    │
                                                                    ▼
                                                              nftables set
                                                              (ip saddr drop)
```

## Modules

### PacketSniffer (`src/ingestion/sniffer.py`)
- Live capture via scapy `L3socket` with BPF filter
- Three-tier backpressure: normal → warning (configurable ratio) → critical (drops packets)
- Thread-safe queue with configurable `maxsize`, stop sentinel pattern
- Simulation replay from pcap files as fallback on permission errors

### FlowAggregator (`src/features/aggregator.py`)
- O(1) per-packet ingestion — `dict` lookup + `OrderedDict.move_to_end()`
- LRU eviction when `max_flows` is exceeded (`popitem(last=False)`)
- Asynchronous background sweeper thread for idle/hard timeout eviction
- 43 bidirectional flow features: packet counts, byte counts, IAT statistics, TCP flags, payload entropy
- Reentrant lock (`threading.RLock()`) protecting all shared state

### LymphaNet (`src/models/lympha_net.py`)
- Hybrid architecture: categorical embeddings + continuous MLP
- `nn.Embedding`: source port (65536 → 16), dest port (65536 → 16), protocol (256 → 8)
- `torch.clamp` guards out-of-bounds indices — runtime-safe against malformed input
- Fused 80-dim input (40 continuous + 40 embedded) → `[512, 256, 128, 64]` → `num_classes`
- Legacy `TrafficClassifier` preserved for backward-compatible checkpoint loading

### InferenceEngine (`src/models/inference.py`)
- `split_features()` separates 40 continuous columns from 3 categorical indices
- `StandardScaler` alignment validation — raises `InferenceAlignmentError` on dimension mismatch
- Multi-batch evaluation with configurable `batch_size`
- `torch.no_grad()` + `model.eval()` enforced on every cycle
- `build_mock_scaler()` for simulation mode when no live scaler artifact exists

### FirewallController (`src/prevention/firewall.py`)
- Nftables interaction via `subprocess.run(["nft", "-f", "-"], input=script, shell=False)`
- `ipaddress.ip_address()` validates all IP inputs — injection strings rejected structurally
- `_validate_identifier()` regex `^[A-Za-z][A-Za-z0-9_-]*$` on table/set/chain names
- Batch `add element` transactions with per-element TTL
- Error hierarchy: `NftablesPrivilegeError`, `NftablesExecutionError`, `NftablesTimeoutError`
- Whitelist support: exact IP match + prefix match

### LymphaDaemon (`src/main.py`)
- Non-blocking event loop: drain sniffer queue → flush aggregator → run inference → block threats
- SIGINT/SIGTERM handling with saved/restored previous handlers
- Exception isolation boundaries — no single component failure crashes the daemon
- Cleanup guaranteed via `run()` `finally` block — reverse-initialization-order teardown
- `request_shutdown()` + `_shutdown_event` converges all paths to `stop()`

## Installation

```bash
git clone https://github.com/hasseen-salem/lympha.git
cd lympha
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Edit `config/settings.yaml`:

```yaml
sniffer:
  interface: "eth0"
  bpf_filter: "ip"
  queue_maxsize: 10000
  backpressure_warn_ratio: 0.8
  backpressure_critical_ratio: 0.95

aggregator:
  idle_timeout_sec: 5.0
  hard_timeout_sec: 10.0
  max_flows: 10000
  flush_interval_sec: 1.0

model:
  continuous_dim: 40
  hidden_dims: [512, 256, 128, 64]
  dropouts: [0.3, 0.3, 0.2, 0.1]
  src_port_embed_dim: 16
  dst_port_embed_dim: 16
  proto_embed_dim: 8

execution:
  simulation_mode: true
  detection_threshold: 0.5
  flush_interval_sec: 2.0

mitigation:
  enabled: true
  table_name: "lympha"
  set_name: "blackhole"
  block_timeout_sec: 3600
  whitelist_ips: ["127.0.0.1"]
```

## Usage

### Simulation mode (no root required)

```bash
python3 -m src.main
```

The daemon uses `build_mock_scaler()` and initialized model weights. Packets are captured live but inference runs without a trained checkpoint.

### Production mode

Export a trained 40-dim scaler and model weights, then:

```bash
python3 -m src.main --config /path/to/config.yaml
```

Run with `CAP_NET_RAW` / root for live packet capture + nftables mitigation.

### Testing

```bash
pytest tests/ -v
```

92 tests across all modules (2 CUDA skips on CPU-only machines):

| Module | Tests | Scope |
|--------|-------|-------|
| `test_sniffer` | 12 | Backpressure, thread safety, sentinel, simulation fallback |
| `test_aggregator` | 17 | LRU eviction, concurrent ingest, sweeper, bidir flows |
| `test_models` | 13 | Embedding dims, gradient flow, OOB clamping, device movement |
| `test_inference` | 14 | Scaler alignment, multi-batch, threshold control, artifact errors |
| `test_firewall` | 19 | Injection rejection, nft script building, privilege errors, whitelist |
| `test_main` | 10 | Signal handling, exception isolation, shutdown convergence, metrics |
| `test_config_loader` | 9 | Path resolution, validation, frozen config, device resolution |

## Requirements

- Python 3.10+
- PyTorch 2.x
- scapy
- pandas, numpy
- scikit-learn
- safetensors
- pydantic
- PyYAML
- nftables (mitigation only)

## License

MIT
