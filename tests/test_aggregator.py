from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.aggregator import (
    FLOW_FEATURES,
    FLUSH_COLUMNS,
    FlowAggregator,
    _payload_entropy,
    five_tuple,
)
from src.utils.config_loader import AggregatorConfig


def _make_config(**overrides) -> AggregatorConfig:
    defaults = {
        "idle_timeout_sec": 0.15,
        "hard_timeout_sec": 1.0,
        "max_flows": 100,
        "flush_interval_sec": 0.05,
    }
    defaults.update(overrides)
    return AggregatorConfig(**defaults)


def _make_mock_ip_packet(
    src: str = "10.0.0.1",
    dst: str = "10.0.0.2",
    sport: int = 12345,
    dport: int = 80,
    proto: str = "TCP",
    payload: bytes = b"hello",
    flags: int = 0x02,
    win: int = 65535,
    ttl: int = 64,
    pkt_len: int = 100,
):
    class MockPayload:
        def __init__(self) -> None:
            self.payload = payload
            self.sport = sport
            self.dport = dport
            self.flags = flags
            self.window = win

    class MockIP:
        def __init__(self) -> None:
            self.src = src
            self.dst = dst
            self.proto = 6 if proto == "TCP" else 17
            self.ihl = 5
            self.ttl = ttl
            self.payload = MockPayload()

    class MockPacket:
        def __init__(self) -> None:
            self.payload = MockIP()

        def __len__(self) -> int:
            return pkt_len

    return MockPacket()


def test_five_tuple_extraction():
    pkt = _make_mock_ip_packet(
        src="192.168.1.10",
        dst="192.168.1.20",
        sport=4444,
        dport=53,
        proto="TCP",
    )
    assert five_tuple(pkt) == ("192.168.1.10", "192.168.1.20", 4444, 53, "TCP")


def test_flow_aggregator_single_flow():
    agg = FlowAggregator(_make_config())
    pkt = _make_mock_ip_packet()
    agg.ingest(pkt)
    time.sleep(0.2)
    agg._sweep_expired()  # noqa: SLF001
    df = agg.drain()
    assert len(df) == 1
    assert list(df.columns) == FLUSH_COLUMNS
    assert df.iloc[0]["protocol_type"] == 6.0
    assert df.iloc[0]["src_port"] == 12345.0
    assert df.iloc[0]["dst_port"] == 80.0
    assert df.iloc[0]["fwd_pkts_tot"] == 1.0


def test_flow_aggregator_bidirectional():
    agg = FlowAggregator(_make_config())
    pkt_fwd = _make_mock_ip_packet(
        src="10.0.0.1",
        dst="10.0.0.2",
        sport=1000,
        dport=80,
        flags=0x02,
    )
    pkt_bwd = _make_mock_ip_packet(
        src="10.0.0.2",
        dst="10.0.0.1",
        sport=80,
        dport=1000,
        flags=0x10,
    )
    agg.ingest(pkt_fwd)
    agg.ingest(pkt_bwd)
    df = agg.flush()
    assert len(df) == 2
    assert "ack_flag_cnt" in df.columns


def test_flow_aggregator_multiple_flows():
    agg = FlowAggregator(_make_config())
    for index in range(5):
        pkt = _make_mock_ip_packet(
            src="10.0.0.1",
            dst="10.0.0.2",
            sport=1000 + index,
            dport=80,
        )
        agg.ingest(pkt)
    df = agg.flush()
    assert len(df) == 5


def test_flow_aggregator_empty_drain():
    agg = FlowAggregator(_make_config())
    df = agg.drain()
    assert len(df) == 0
    assert list(df.columns) == FLUSH_COLUMNS


def test_flow_aggregator_active_flow_count():
    agg = FlowAggregator(_make_config(idle_timeout_sec=10.0, hard_timeout_sec=60.0))
    pkt = _make_mock_ip_packet()
    agg.ingest(pkt)
    assert agg.active_flow_count == 1
    agg.flush()
    assert agg.active_flow_count == 0


def test_payload_entropy():
    assert abs(_payload_entropy(b"aaaa")) < 1e-9
    assert _payload_entropy(b"abcd") > 1.0
    assert _payload_entropy(b"") == 0.0


def test_all_flow_features_present():
    agg = FlowAggregator(_make_config())
    pkt = _make_mock_ip_packet(
        src="192.168.1.1",
        dst="192.168.1.2",
        sport=3000,
        dport=443,
        flags=0x12,
    )
    agg.ingest(pkt)
    df = agg.flush()
    assert len(df.columns) == len(FLUSH_COLUMNS)
    assert set(df.columns) == set(FLUSH_COLUMNS)


def test_feature_ranges():
    agg = FlowAggregator(_make_config())
    pkt = _make_mock_ip_packet(flags=0x12)
    agg.ingest(pkt)
    df = agg.flush()
    row = df.iloc[0]
    assert row["syn_flag_cnt"] >= 1.0
    assert row["ack_flag_cnt"] >= 1.0
    assert row["flow_duration"] > 0.0
    assert row["packet_length"] > 0.0


def test_ingest_does_not_call_sweeper_on_hot_path():
    agg = FlowAggregator(_make_config())
    with patch.object(agg, "_sweep_expired") as sweep_mock:
        for index in range(50):
            agg.ingest(
                _make_mock_ip_packet(
                    src="10.0.0.1",
                    dst="10.0.0.2",
                    sport=2000 + index,
                )
            )
        sweep_mock.assert_not_called()


def test_background_sweeper_expires_idle_flows():
    agg = FlowAggregator(_make_config(idle_timeout_sec=0.1, flush_interval_sec=0.05))
    agg.start()
    agg.ingest(_make_mock_ip_packet(sport=5001))
    assert agg.active_flow_count == 1

    deadline = time.monotonic() + 1.0
    drained = 0
    while time.monotonic() < deadline and drained == 0:
        df = agg.drain()
        drained = len(df)
        time.sleep(0.02)

    agg.stop()
    assert drained == 1
    assert agg.metrics.timeout_evictions >= 1
    assert agg.active_flow_count == 0


def test_max_flows_lru_eviction_flood():
    max_flows = 20
    agg = FlowAggregator(
        _make_config(max_flows=max_flows, idle_timeout_sec=60.0, hard_timeout_sec=120.0),
    )

    for index in range(max_flows + 50):
        agg.ingest(
            _make_mock_ip_packet(
                src="10.0.0.1",
                dst="10.0.0.2",
                sport=10_000 + index,
            )
        )

    assert agg.active_flow_count == max_flows
    assert agg.metrics.lru_evictions >= 50

    df = agg.flush()
    assert len(df) >= 50


def test_lru_keeps_recently_used_flow():
    max_flows = 3
    agg = FlowAggregator(
        _make_config(max_flows=max_flows, idle_timeout_sec=60.0, hard_timeout_sec=120.0),
    )

    agg.ingest(_make_mock_ip_packet(sport=1))
    agg.ingest(_make_mock_ip_packet(sport=2))
    agg.ingest(_make_mock_ip_packet(sport=3))

    agg.ingest(_make_mock_ip_packet(sport=1))
    agg.ingest(_make_mock_ip_packet(sport=4))
    agg.ingest(_make_mock_ip_packet(sport=5))

    assert agg.active_flow_count == max_flows
    assert agg.metrics.lru_evictions >= 2

    df = agg.flush()
    active_ports = set(df["src_port"].astype(int).tolist())
    assert 1 in active_ports
    assert 4 in active_ports
    assert 5 in active_ports


def test_concurrent_ingest_thread_safe():
    agg = FlowAggregator(
        _make_config(max_flows=500, idle_timeout_sec=60.0, hard_timeout_sec=120.0),
    )
    packet_count = 200
    workers = 8

    def ingest_batch(start: int) -> None:
        for index in range(start, start + packet_count // workers):
            agg.ingest(
                _make_mock_ip_packet(
                    src=f"10.0.{index % 50}.1",
                    dst="10.0.0.2",
                    sport=20_000 + index,
                )
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(ingest_batch, batch * (packet_count // workers))
            for batch in range(workers)
        ]
        for future in futures:
            future.result()

    assert agg.metrics.packets_ingested == packet_count
    assert agg.active_flow_count <= 500
    df = agg.flush()
    assert len(df) == packet_count


def test_flush_exports_all_active_flows():
    agg = FlowAggregator(
        _make_config(idle_timeout_sec=60.0, hard_timeout_sec=120.0),
    )
    for index in range(3):
        agg.ingest(_make_mock_ip_packet(sport=7000 + index))
    df = agg.flush()
    assert len(df) == 3
    assert agg.active_flow_count == 0


def test_same_five_tuple_updates_single_flow():
    agg = FlowAggregator(_make_config())
    for _ in range(5):
        agg.ingest(_make_mock_ip_packet(sport=9000))
    assert agg.active_flow_count == 1
    df = agg.flush()
    assert len(df) == 1
    assert df.iloc[0]["fwd_pkts_tot"] == 5.0


def test_hard_timeout_eviction():
    agg = FlowAggregator(
        _make_config(idle_timeout_sec=0.05, hard_timeout_sec=0.1, flush_interval_sec=0.05),
    )
    agg.start()
    agg.ingest(_make_mock_ip_packet(sport=8001))

    deadline = time.monotonic() + 1.0
    expired = 0
    while time.monotonic() < deadline and expired == 0:
        expired = len(agg.drain())
        time.sleep(0.02)

    agg.stop()
    assert expired >= 1
    assert agg.metrics.timeout_evictions >= 1
