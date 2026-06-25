from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config_loader import AggregatorConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

FiveTuple = tuple[str, str, int, int, str]

FLOW_FEATURES: list[str] = [
    "packet_length",
    "protocol_type",
    "src_port",
    "dst_port",
    "window_size",
    "ttl",
    "flags",
    "payload_entropy",
    "flow_duration",
    "fwd_pkts_tot",
    "bwd_pkts_tot",
    "fwd_pkts_per_sec",
    "bwd_pkts_per_sec",
    "flow_pkts_per_sec",
    "down_up_ratio",
    "fwd_header_len",
    "bwd_header_len",
    "fwd_pkts_small",
    "bwd_pkts_small",
    "fwd_pkts_bulk",
    "bwd_pkts_bulk",
    "fwd_pkt_len_max",
    "fwd_pkt_len_min",
    "fwd_pkt_len_mean",
    "bwd_pkt_len_max",
    "bwd_pkt_len_min",
    "bwd_pkt_len_mean",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    "fin_flag_cnt",
    "syn_flag_cnt",
    "rst_flag_cnt",
    "psh_flag_cnt",
    "ack_flag_cnt",
    "init_win_bytes_fwd",
    "init_win_bytes_bwd",
    "avg_pkt_size",
]

FLUSH_COLUMNS: list[str] = FLOW_FEATURES + ["src_ip", "dst_ip"]

TCP_FLAG_MAP: dict[str, int] = {
    "F": 0x01,
    "S": 0x02,
    "R": 0x04,
    "P": 0x08,
    "A": 0x10,
    "U": 0x20,
}


@dataclass(frozen=True)
class AggregatorMetrics:
    active_flows: int = 0
    completed_flows: int = 0
    lru_evictions: int = 0
    timeout_evictions: int = 0
    packets_ingested: int = 0
    packets_dropped: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "active_flows": self.active_flows,
            "completed_flows": self.completed_flows,
            "lru_evictions": self.lru_evictions,
            "timeout_evictions": self.timeout_evictions,
            "packets_ingested": self.packets_ingested,
            "packets_dropped": self.packets_dropped,
        }


def _ip_layer(packet: Any) -> Any:
    try:
        from scapy.all import IP as ScapyIP
        from scapy.all import Ether as ScapyEther

        if isinstance(packet, ScapyIP):
            return packet
        if isinstance(packet, ScapyEther):
            return _ip_layer(packet.payload)
    except ImportError:
        pass
    if hasattr(packet, "src"):
        return packet
    if hasattr(packet, "payload") and hasattr(packet.payload, "src"):
        return packet.payload
    return packet


def _extract_transport(packet: Any) -> tuple[str, int, int]:
    ip = _ip_layer(packet)
    proto_num = ip.proto if hasattr(ip, "proto") else 0
    transport = ip.payload if hasattr(ip, "payload") else ip

    sport = getattr(transport, "sport", 0)
    dport = getattr(transport, "dport", 0)

    proto_name = {1: "ICMP", 6: "TCP", 17: "UDP"}.get(proto_num, str(proto_num))
    return proto_name, sport, dport


def five_tuple(packet: Any) -> FiveTuple:
    if not hasattr(packet, "payload"):
        return ("", "", 0, 0, "")
    ip = _ip_layer(packet)
    src = ip.src if hasattr(ip, "src") else ""
    dst = ip.dst if hasattr(ip, "dst") else ""
    proto_name, sport, dport = _extract_transport(packet)
    return (src, dst, sport, dport, proto_name)


def _extract_tcp_flags(packet: Any) -> int:
    if not hasattr(packet, "payload"):
        return 0
    ip = packet.payload
    if not hasattr(ip, "payload"):
        return 0
    transport = ip.payload
    flags = getattr(transport, "flags", 0)
    if isinstance(flags, str):
        val = 0
        for ch in flags.upper():
            val |= TCP_FLAG_MAP.get(ch, 0)
        return val
    return int(flags) if flags else 0


def _payload_entropy(payload: bytes) -> float:
    if not payload:
        return 0.0
    counts = np.bincount(np.frombuffer(payload, dtype=np.uint8))
    prob = counts / counts.sum()
    return float(-np.sum(prob * np.log2(prob + 1e-12)))


class _RunningStats:
    __slots__ = ("_min", "_max", "_sum", "_sum_sq", "_count")

    def __init__(self) -> None:
        self._min = 0.0
        self._max = 0.0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0

    def add(self, value: float) -> None:
        if self._count == 0:
            self._min = value
            self._max = value
        else:
            self._min = min(self._min, value)
            self._max = max(self._max, value)
        self._sum += value
        self._sum_sq += value * value
        self._count += 1

    @property
    def min(self) -> float:
        return self._min if self._count > 0 else 0.0

    @property
    def max(self) -> float:
        return self._max if self._count > 0 else 0.0

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0

    @property
    def std(self) -> float:
        if self._count < 2:
            return 0.0
        mean = self.mean
        variance = self._sum_sq / self._count - mean * mean
        return math.sqrt(max(variance, 0.0))


class FlowState:
    __slots__ = (
        "src",
        "dst",
        "sport",
        "dport",
        "proto",
        "start",
        "last",
        "fwd_pkts",
        "bwd_pkts",
        "fwd_bytes",
        "bwd_bytes",
        "fwd_header_len_sum",
        "bwd_header_len_sum",
        "fwd_len_stats",
        "bwd_len_stats",
        "fwd_iat_stats",
        "bwd_iat_stats",
        "last_fwd_ts",
        "last_bwd_ts",
        "fwd_pkts_small",
        "bwd_pkts_small",
        "fwd_pkts_bulk",
        "bwd_pkts_bulk",
        "fin_cnt",
        "syn_cnt",
        "rst_cnt",
        "psh_cnt",
        "ack_cnt",
        "fwd_win",
        "bwd_win",
        "entropy_sum",
        "entropy_count",
        "ttl_val",
    )

    def __init__(self, src: str, dst: str, sport: int, dport: int, proto: str) -> None:
        self.src = src
        self.dst = dst
        self.sport = sport
        self.dport = dport
        self.proto = proto
        self.start = time.time()
        self.last = self.start
        self.fwd_pkts = 0
        self.bwd_pkts = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.fwd_header_len_sum = 0
        self.bwd_header_len_sum = 0
        self.fwd_len_stats = _RunningStats()
        self.bwd_len_stats = _RunningStats()
        self.fwd_iat_stats = _RunningStats()
        self.bwd_iat_stats = _RunningStats()
        self.last_fwd_ts: float | None = None
        self.last_bwd_ts: float | None = None
        self.fwd_pkts_small = 0
        self.bwd_pkts_small = 0
        self.fwd_pkts_bulk = 0
        self.bwd_pkts_bulk = 0
        self.fin_cnt = 0
        self.syn_cnt = 0
        self.rst_cnt = 0
        self.psh_cnt = 0
        self.ack_cnt = 0
        self.fwd_win = 0
        self.bwd_win = 0
        self.entropy_sum = 0.0
        self.entropy_count = 0
        self.ttl_val = 64

    def update(self, packet: Any, direction: str) -> None:
        now = time.time()
        pkt_len = len(packet)
        ihl_bytes = 0
        flags_val = 0
        ttl = 64
        win = 0
        payload = b""

        ip = _ip_layer(packet) if hasattr(packet, "payload") else None
        if ip and hasattr(ip, "ihl"):
            ihl_bytes = (ip.ihl or 0) * 4
            ttl = getattr(ip, "ttl", 64)
        if ip and hasattr(ip, "payload"):
            transport = ip.payload
            win = getattr(transport, "window", 0) if hasattr(transport, "window") else 0
            flags_val = _extract_tcp_flags(packet)
            payload = bytes(transport.payload) if hasattr(transport, "payload") else b""

        header_len = ihl_bytes + (
            20 if ip and hasattr(ip, "payload") and hasattr(ip.payload, "sport") else 0
        )

        if direction == "fwd":
            self.fwd_pkts += 1
            self.fwd_bytes += pkt_len
            self.fwd_header_len_sum += header_len
            self.fwd_len_stats.add(float(pkt_len))
            if self.last_fwd_ts is not None:
                self.fwd_iat_stats.add(now - self.last_fwd_ts)
            self.last_fwd_ts = now
            self.fwd_win = win if win else self.fwd_win
            if pkt_len < 64:
                self.fwd_pkts_small += 1
            if pkt_len > 1500:
                self.fwd_pkts_bulk += 1
        else:
            self.bwd_pkts += 1
            self.bwd_bytes += pkt_len
            self.bwd_header_len_sum += header_len
            self.bwd_len_stats.add(float(pkt_len))
            if self.last_bwd_ts is not None:
                self.bwd_iat_stats.add(now - self.last_bwd_ts)
            self.last_bwd_ts = now
            self.bwd_win = win if win else self.bwd_win
            if pkt_len < 64:
                self.bwd_pkts_small += 1
            if pkt_len > 1500:
                self.bwd_pkts_bulk += 1

        if flags_val & TCP_FLAG_MAP["F"]:
            self.fin_cnt += 1
        if flags_val & TCP_FLAG_MAP["S"]:
            self.syn_cnt += 1
        if flags_val & TCP_FLAG_MAP["R"]:
            self.rst_cnt += 1
        if flags_val & TCP_FLAG_MAP["P"]:
            self.psh_cnt += 1
        if flags_val & TCP_FLAG_MAP["A"]:
            self.ack_cnt += 1

        self.entropy_sum += _payload_entropy(payload)
        self.entropy_count += 1
        self.ttl_val = ttl
        self.last = now

    def to_features(self) -> dict[str, float]:
        duration = max(self.last - self.start, 1e-6)
        total_pkts = self.fwd_pkts + self.bwd_pkts

        return {
            "packet_length": (
                self.fwd_len_stats.mean * self.fwd_pkts
                + self.bwd_len_stats.mean * self.bwd_pkts
            ) / max(total_pkts, 1),
            "protocol_type": 6 if self.proto == "TCP" else (17 if self.proto == "UDP" else 1),
            "src_port": float(self.sport),
            "dst_port": float(self.dport),
            "window_size": float(self.fwd_win or self.bwd_win),
            "ttl": float(self.ttl_val),
            "flags": float(self.syn_cnt > 0 or self.ack_cnt > 0),
            "payload_entropy": self.entropy_sum / max(self.entropy_count, 1),
            "flow_duration": duration,
            "fwd_pkts_tot": float(self.fwd_pkts),
            "bwd_pkts_tot": float(self.bwd_pkts),
            "fwd_pkts_per_sec": self.fwd_pkts / duration,
            "bwd_pkts_per_sec": self.bwd_pkts / duration,
            "flow_pkts_per_sec": total_pkts / duration,
            "down_up_ratio": self.bwd_pkts / max(self.fwd_pkts, 1),
            "fwd_header_len": float(self.fwd_header_len_sum),
            "bwd_header_len": float(self.bwd_header_len_sum),
            "fwd_pkts_small": float(self.fwd_pkts_small),
            "bwd_pkts_small": float(self.bwd_pkts_small),
            "fwd_pkts_bulk": float(self.fwd_pkts_bulk),
            "bwd_pkts_bulk": float(self.bwd_pkts_bulk),
            "fwd_pkt_len_max": self.fwd_len_stats.max,
            "fwd_pkt_len_min": self.fwd_len_stats.min,
            "fwd_pkt_len_mean": self.fwd_len_stats.mean,
            "bwd_pkt_len_max": self.bwd_len_stats.max,
            "bwd_pkt_len_min": self.bwd_len_stats.min,
            "bwd_pkt_len_mean": self.bwd_len_stats.mean,
            "fwd_iat_mean": self.fwd_iat_stats.mean,
            "fwd_iat_std": self.fwd_iat_stats.std,
            "fwd_iat_max": self.fwd_iat_stats.max,
            "fwd_iat_min": self.fwd_iat_stats.min,
            "bwd_iat_mean": self.bwd_iat_stats.mean,
            "bwd_iat_std": self.bwd_iat_stats.std,
            "bwd_iat_max": self.bwd_iat_stats.max,
            "bwd_iat_min": self.bwd_iat_stats.min,
            "fin_flag_cnt": float(self.fin_cnt),
            "syn_flag_cnt": float(self.syn_cnt),
            "rst_flag_cnt": float(self.rst_cnt),
            "psh_flag_cnt": float(self.psh_cnt),
            "ack_flag_cnt": float(self.ack_cnt),
            "init_win_bytes_fwd": float(self.fwd_win),
            "init_win_bytes_bwd": float(self.bwd_win),
            "avg_pkt_size": (
                self.fwd_bytes / max(self.fwd_pkts, 1)
                if self.fwd_pkts
                else (self.bwd_bytes / max(self.bwd_pkts, 1))
            ),
        }


class FlowAggregator:
    """O(1) flow tracking with LRU eviction and asynchronous timeout sweeps."""

    def __init__(self, config: AggregatorConfig) -> None:
        self._config = config
        self._flows: OrderedDict[FiveTuple, FlowState] = OrderedDict()
        self._completed: list[dict[str, float | str]] = []
        self._lock = threading.RLock()
        self._metrics = AggregatorMetrics()
        self._running = False
        self._sweeper_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def config(self) -> AggregatorConfig:
        return self._config

    @property
    def metrics(self) -> AggregatorMetrics:
        with self._lock:
            return AggregatorMetrics(
                active_flows=len(self._flows),
                completed_flows=self._metrics.completed_flows,
                lru_evictions=self._metrics.lru_evictions,
                timeout_evictions=self._metrics.timeout_evictions,
                packets_ingested=self._metrics.packets_ingested,
                packets_dropped=self._metrics.packets_dropped,
            )

    @property
    def active_flow_count(self) -> int:
        with self._lock:
            return len(self._flows)

    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._sweeper_thread = threading.Thread(
            target=self._sweeper_loop,
            name="lympha-aggregator-sweeper",
            daemon=True,
        )
        self._sweeper_thread.start()
        logger.info(
            "Flow aggregator sweeper started (interval=%.2fs idle=%.2fs hard=%.2fs max_flows=%d)",
            self._config.flush_interval_sec,
            self._config.idle_timeout_sec,
            self._config.hard_timeout_sec,
            self._config.max_flows,
        )

    def stop(self, timeout: float = 2.0) -> None:
        if not self._running:
            return
        self._stop_event.set()
        self._running = False
        if self._sweeper_thread and self._sweeper_thread.is_alive():
            self._sweeper_thread.join(timeout=timeout)
        self._sweeper_thread = None
        logger.info("Flow aggregator sweeper stopped")

    def ingest(self, packet: Any) -> None:
        key = five_tuple(packet)
        if not key[0]:
            with self._lock:
                self._metrics = AggregatorMetrics(
                    **{
                        **self._metrics.snapshot(),
                        "packets_dropped": self._metrics.packets_dropped + 1,
                    }
                )
            return

        with self._lock:
            self._metrics = AggregatorMetrics(
                **{
                    **self._metrics.snapshot(),
                    "packets_ingested": self._metrics.packets_ingested + 1,
                }
            )
            direction = self._resolve_direction(packet, key)

            if key in self._flows:
                flow = self._flows[key]
                flow.update(packet, direction)
                self._flows.move_to_end(key)
                return

            if len(self._flows) >= self._config.max_flows:
                self._evict_lru()

            self._flows[key] = FlowState(*key)
            self._flows[key].update(packet, direction)

    def _resolve_direction(self, packet: Any, key: FiveTuple) -> str:
        ip = _ip_layer(packet) if hasattr(packet, "payload") else None
        if ip and hasattr(ip, "src"):
            return "fwd" if ip.src == key[0] else "bwd"
        return "fwd"

    def _evict_lru(self) -> None:
        if not self._flows:
            return
        key, flow = self._flows.popitem(last=False)
        self._append_completed(key, flow)
        self._metrics = AggregatorMetrics(
            **{
                **self._metrics.snapshot(),
                "lru_evictions": self._metrics.lru_evictions + 1,
                "completed_flows": self._metrics.completed_flows + 1,
            }
        )
        logger.warning(
            "LRU eviction at capacity (%d): %s:%d -> %s:%d",
            self._config.max_flows,
            key[0],
            key[2],
            key[1],
            key[3],
        )

    def _append_completed(self, key: FiveTuple, flow: FlowState) -> None:
        features = flow.to_features()
        features["src_ip"] = key[0]
        features["dst_ip"] = key[1]
        self._completed.append(features)

    def _sweeper_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._config.flush_interval_sec):
            self._sweep_expired()

    def _sweep_expired(self) -> int:
        now = time.time()
        expired_keys: list[FiveTuple] = []

        with self._lock:
            for key, flow in self._flows.items():
                age = now - flow.start
                idle = now - flow.last
                if age > self._config.hard_timeout_sec or idle > self._config.idle_timeout_sec:
                    expired_keys.append(key)

            for key in expired_keys:
                flow = self._flows.pop(key, None)
                if flow is None:
                    continue
                self._append_completed(key, flow)
                self._metrics = AggregatorMetrics(
                    **{
                        **self._metrics.snapshot(),
                        "timeout_evictions": self._metrics.timeout_evictions + 1,
                        "completed_flows": self._metrics.completed_flows + 1,
                    }
                )

        if expired_keys:
            logger.debug("Sweeper expired %d flows", len(expired_keys))
        return len(expired_keys)

    def drain(self) -> pd.DataFrame:
        with self._lock:
            if not self._completed:
                return pd.DataFrame(columns=FLUSH_COLUMNS)
            df = pd.DataFrame(self._completed, columns=FLUSH_COLUMNS)
            df = df.fillna(0.0).infer_objects()
            self._completed.clear()
            return df

    def flush(self) -> pd.DataFrame:
        self._sweep_expired()
        with self._lock:
            for key in list(self._flows.keys()):
                flow = self._flows.pop(key)
                self._append_completed(key, flow)
                self._metrics = AggregatorMetrics(
                    **{
                        **self._metrics.snapshot(),
                        "completed_flows": self._metrics.completed_flows + 1,
                    }
                )
            if not self._completed:
                return pd.DataFrame(columns=FLUSH_COLUMNS)
            df = pd.DataFrame(self._completed, columns=FLUSH_COLUMNS)
            df = df.fillna(0.0).infer_objects()
            self._completed.clear()
            return df

    def empty_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(columns=FLUSH_COLUMNS)
