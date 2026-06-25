from __future__ import annotations

import queue
import signal
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.aggregator import FLUSH_COLUMNS
from src.main import LymphaDaemon
from src.models.inference import RealTimeInferenceEngine, ThreatProfile
from src.models.lympha_net import LymphaNet
from src.prevention.firewall import BlockResult
from src.utils.config_loader import load_config


def _mock_packet(pkt_id: int = 1):
    class MockPayload:
        def __init__(self) -> None:
            self.payload = b""
            self.sport = 1000 + pkt_id
            self.dport = 80
            self.flags = 0x02
            self.window = 65535

    class MockIP:
        def __init__(self) -> None:
            self.src = f"10.0.0.{pkt_id}"
            self.dst = "10.0.0.99"
            self.proto = 6
            self.ihl = 5
            self.ttl = 64
            self.payload = MockPayload()

    class MockPacket:
        def __init__(self) -> None:
            self.payload = MockIP()

        def __len__(self) -> int:
            return 100

    return MockPacket()


class MockSniffer:
    def __init__(self, packets: list | None = None) -> None:
        self._queue: queue.Queue = queue.Queue()
        for packet in packets or []:
            self._queue.put(packet)
        self.running = False
        self.started = False
        self.stop_called = False
        self.metrics = MagicMock()
        self.metrics.packets_dropped = 0

    @property
    def packet_queue(self) -> queue.Queue:
        return self._queue

    def start(self) -> None:
        self.running = True
        self.started = True

    def stop(self) -> None:
        self.stop_called = True
        self.running = False


class MockAggregator:
    def __init__(self, drain_df: pd.DataFrame | None = None) -> None:
        self.started = False
        self.stopped = False
        self.ingested: list = []
        self._drain_df = drain_df if drain_df is not None else pd.DataFrame(columns=FLUSH_COLUMNS)
        self.metrics = MagicMock()
        self.metrics.active_flows = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def ingest(self, packet) -> None:
        self.ingested.append(packet)

    def drain(self) -> pd.DataFrame:
        df = self._drain_df
        self._drain_df = pd.DataFrame(columns=FLUSH_COLUMNS)
        return df


class MockInference:
    def __init__(
        self,
        profiles: list[ThreatProfile] | None = None,
        *,
        raise_error: bool = False,
    ) -> None:
        self.profiles = profiles or []
        self.raise_error = raise_error
        self.call_count = 0

    def predict_flows(self, df: pd.DataFrame) -> list[ThreatProfile]:
        self.call_count += 1
        if self.raise_error:
            raise RuntimeError("simulated inference failure")
        if df.empty:
            return []
        return list(self.profiles)


class MockFirewall:
    def __init__(self) -> None:
        self.initialized = False
        self.cleaned_up = False
        self.blocked_ips: list[str] = []

    def initialize(self) -> None:
        self.initialized = True

    def block_threats(self, profiles) -> BlockResult:
        blocked = tuple(profile.src_ip for profile in profiles if profile.should_block)
        self.blocked_ips.extend(blocked)
        return BlockResult(blocked=blocked)

    def cleanup(self) -> None:
        self.cleaned_up = True


def _flow_row(src_ip: str = "10.0.0.5", dst_ip: str = "10.0.0.99") -> dict:
    row = {name: 1.0 for name in FLUSH_COLUMNS if name not in ("src_ip", "dst_ip")}
    row["protocol_type"] = 6.0
    row["src_port"] = 1234.0
    row["dst_port"] = 80.0
    row["src_ip"] = src_ip
    row["dst_ip"] = dst_ip
    return row


def _build_daemon(**overrides) -> LymphaDaemon:
    config = load_config()
    defaults = {
        "config": config,
        "sniffer": MockSniffer(),
        "aggregator": MockAggregator(),
        "inference": MockInference(),
        "firewall": MockFirewall(),
        "install_signals": False,
    }
    defaults.update(overrides)
    return LymphaDaemon(**defaults)


def test_daemon_start_is_idempotent():
    daemon = _build_daemon()
    daemon.start()
    daemon.start()
    assert daemon.running
    daemon.stop()


def test_daemon_wires_components_on_start():
    sniffer = MockSniffer()
    aggregator = MockAggregator()
    firewall = MockFirewall()
    daemon = _build_daemon(sniffer=sniffer, aggregator=aggregator, firewall=firewall)

    daemon.start()
    assert sniffer.started
    assert aggregator.started
    assert firewall.initialized
    daemon.stop()


def test_process_once_ingests_packets_from_queue():
    packets = [_mock_packet(1), _mock_packet(2)]
    sniffer = MockSniffer(packets)
    aggregator = MockAggregator()
    daemon = _build_daemon(sniffer=sniffer, aggregator=aggregator)
    daemon.start()

    daemon.process_once()
    assert len(aggregator.ingested) == 2
    assert daemon.metrics.packets_processed == 2
    daemon.stop()


def test_inference_cycle_blocks_threats():
    flow_df = pd.DataFrame([_flow_row()], columns=FLUSH_COLUMNS)
    aggregator = MockAggregator(drain_df=flow_df)
    inference = MockInference(
        [
            ThreatProfile("10.0.0.5", "10.0.0.99", 0.95, True),
        ],
    )
    firewall = MockFirewall()
    daemon = _build_daemon(
        aggregator=aggregator,
        inference=inference,
        firewall=firewall,
    )
    daemon.start()

    daemon._last_flush = 0.0
    daemon.process_once()

    assert inference.call_count == 1
    assert daemon.metrics.flows_scored == 1
    assert daemon.metrics.flows_blocked == 1
    assert firewall.blocked_ips == ["10.0.0.5"]
    assert len(daemon.metrics.detection_events) == 1
    daemon.stop()


def test_inference_exception_is_contained():
    flow_df = pd.DataFrame([_flow_row()], columns=FLUSH_COLUMNS)
    aggregator = MockAggregator(drain_df=flow_df)
    inference = MockInference(raise_error=True)
    sniffer = MockSniffer([_mock_packet(1)])
    daemon = _build_daemon(
        sniffer=sniffer,
        aggregator=aggregator,
        inference=inference,
    )
    daemon.start()

    daemon._last_flush = 0.0
    daemon.process_once()
    daemon.process_once()

    assert daemon.metrics.inference_errors == 1
    assert daemon.metrics.packets_processed >= 1
    assert sniffer.running
    daemon.stop()


def test_request_shutdown_stops_run_loop_and_cleans_up():
    sniffer = MockSniffer()
    aggregator = MockAggregator()
    firewall = MockFirewall()
    daemon = _build_daemon(
        sniffer=sniffer,
        aggregator=aggregator,
        firewall=firewall,
    )

    thread = threading.Thread(
        target=lambda: daemon.run(max_cycles=10_000, poll_interval_sec=0.01),
        daemon=True,
    )
    thread.start()
    time.sleep(0.05)
    daemon.request_shutdown()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert sniffer.stop_called
    assert aggregator.stopped
    assert firewall.cleaned_up
    assert not daemon.running


def test_signal_handler_requests_shutdown():
    daemon = _build_daemon(install_signals=False)
    daemon._handle_signal(signal.SIGTERM, None)
    assert daemon._shutdown_event.is_set()


def test_stop_sentinel_triggers_shutdown():
    sniffer = MockSniffer()
    sniffer.packet_queue.put(None)
    daemon = _build_daemon(sniffer=sniffer)
    daemon.start()
    daemon.process_once()
    assert daemon._shutdown_event.is_set()
    daemon.stop()


def test_build_inference_engine_uses_mock_scaler_in_simulation_mode():
    config = load_config()
    assert config.execution.simulation_mode is True
    daemon = LymphaDaemon(config, install_signals=False)
    assert isinstance(daemon.inference, RealTimeInferenceEngine)
    assert isinstance(daemon.inference._model, LymphaNet)


def test_metrics_snapshot_is_copy():
    daemon = _build_daemon()
    daemon.start()
    metrics = daemon.metrics
    assert metrics.packets_processed == 0
    daemon.stop()
