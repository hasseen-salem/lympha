from __future__ import annotations

import queue
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.sniffer import PacketSniffer
from src.utils.config_loader import SnifferConfig


def _make_config(**overrides) -> SnifferConfig:
    defaults = {
        "interface": "lo",
        "bpf_filter": "ip",
        "queue_maxsize": 10,
        "max_packets": 0,
        "recv_timeout_sec": 0.1,
        "simulation_file": None,
        "backpressure_warn_ratio": 0.8,
        "backpressure_critical_ratio": 0.95,
    }
    defaults.update(overrides)
    return SnifferConfig(**defaults)


def _make_mock_packet(pkt_id: int = 1):
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
            self.dst = "10.0.0.2"
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


def _iter_packets(count: int, delay: float = 0.0) -> Iterator[Packet]:
    for index in range(count):
        if delay:
            time.sleep(delay)
        yield _make_mock_packet(index + 1)


def _blocking_source(count: int) -> Iterator[Packet]:
    return iter(_make_mock_packet(index + 1) for index in range(count))


def test_sniffer_instantiation():
    sniffer = PacketSniffer(_make_config())
    assert sniffer.running is False
    assert sniffer.packet_queue is not None
    assert sniffer.metrics.mode == "idle"
    sniffer.stop()


def test_sniffer_starts_and_queues_mock_packets():
    config = _make_config(queue_maxsize=20, max_packets=5)
    sniffer = PacketSniffer(
        config,
        packet_source=lambda: _iter_packets(5),
    )
    sniffer.start()

    deadline = time.monotonic() + 2.0
    received: list[Packet] = []
    while time.monotonic() < deadline and len(received) < 5:
        try:
            pkt = sniffer.packet_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if pkt is not None:
            received.append(pkt)

    sniffer.stop()

    assert sniffer.metrics.packets_captured == 5
    assert sniffer.metrics.packets_enqueued == 5
    assert sniffer.metrics.packets_dropped == 0
    assert len(received) == 5
    assert sniffer.metrics.mode == "stopped"


def test_sniffer_stop_is_idempotent():
    config = _make_config(max_packets=2)
    sniffer = PacketSniffer(
        config,
        packet_source=lambda: _iter_packets(2),
    )
    sniffer.start()
    time.sleep(0.2)
    sniffer.stop()
    sniffer.stop()
    assert sniffer.running is False


def test_sniffer_delivers_stop_sentinel():
    config = _make_config(max_packets=1)
    sniffer = PacketSniffer(
        config,
        packet_source=lambda: _iter_packets(1),
    )
    sniffer.start()
    time.sleep(0.2)
    sniffer.stop()

    sentinel_seen = False
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            pkt = sniffer.packet_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if pkt is None:
            sentinel_seen = True
            break
    assert sentinel_seen


def test_sniffer_full_queue_drops_packets_and_preserves_state():
    config = _make_config(
        queue_maxsize=2,
        backpressure_warn_ratio=0.5,
        backpressure_critical_ratio=0.9,
        recv_timeout_sec=0.05,
    )
    sniffer = PacketSniffer(
        config,
        packet_source=lambda: _blocking_source(6),
    )
    sniffer.start()
    time.sleep(0.5)
    sniffer.stop()

    metrics = sniffer.metrics
    assert metrics.packets_captured == 6
    assert metrics.packets_enqueued <= 2
    assert metrics.packets_dropped >= 4
    assert metrics.packets_enqueued + metrics.packets_dropped == 6
    assert sniffer.running is False


def test_sniffer_backpressure_warning_threshold():
    config = _make_config(
        queue_maxsize=5,
        backpressure_warn_ratio=0.6,
        backpressure_critical_ratio=0.99,
        recv_timeout_sec=0.05,
    )
    sniffer = PacketSniffer(
        config,
        packet_source=lambda: _blocking_source(5),
    )
    sniffer.start()
    time.sleep(0.3)
    sniffer.stop()

    assert sniffer.metrics.backpressure_warnings >= 1
    assert sniffer.metrics.packets_enqueued >= 1


def test_sniffer_critical_threshold_drops_without_enqueue():
    config = _make_config(
        queue_maxsize=3,
        backpressure_warn_ratio=0.5,
        backpressure_critical_ratio=0.67,
        recv_timeout_sec=0.05,
    )
    sniffer = PacketSniffer(
        config,
        packet_source=lambda: _blocking_source(5),
    )

    for _ in range(3):
        sniffer._enqueue_packet(_make_mock_packet())  # noqa: SLF001

    assert sniffer.packet_queue.qsize() == 3
    sniffer._enqueue_packet(_make_mock_packet())  # noqa: SLF001

    metrics = sniffer.metrics
    assert metrics.backpressure_critical >= 1
    assert metrics.packets_dropped >= 1
    assert sniffer.packet_queue.qsize() == 3


def test_sniffer_permission_error_without_fallback():
    config = _make_config(simulation_file=None)

    class FailingSocket:
        def recv(self, _mtu: int):
            raise AssertionError("recv should not be called")

        def close(self) -> None:
            pass

    def raise_permission(*_args, **_kwargs):
        raise PermissionError("Operation not permitted")

    import src.ingestion.sniffer as sniffer_module

    original_l3 = sniffer_module.conf.L3socket
    sniffer_module.conf.L3socket = raise_permission  # type: ignore[assignment]
    try:
        sniffer = PacketSniffer(config)
        sniffer.start()
        time.sleep(0.3)
        sniffer.stop()
        assert sniffer.last_error is not None
        assert "Permission denied" in sniffer.last_error or "privileges" in sniffer.last_error
        assert sniffer.metrics.capture_errors >= 1
    finally:
        sniffer_module.conf.L3socket = original_l3


def test_sniffer_falls_back_to_simulation_on_permission_error(tmp_path: Path):
    pcap_path = tmp_path / "sample.pcap"
    try:
        from scapy.all import IP, TCP, wrpcap

        packets = [
            IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1234, dport=80),
            IP(src="10.0.0.3", dst="10.0.0.4") / TCP(sport=4321, dport=443),
        ]
        wrpcap(str(pcap_path), packets)
    except Exception:
        pytest.skip("Scapy pcap write unavailable")

    config = _make_config(
        simulation_file=pcap_path,
        max_packets=2,
        queue_maxsize=10,
    )

    import src.ingestion.sniffer as sniffer_module

    original_l3 = sniffer_module.conf.L3socket

    def raise_permission(*_args, **_kwargs):
        raise PermissionError("Operation not permitted")

    sniffer_module.conf.L3socket = raise_permission  # type: ignore[assignment]
    try:
        sniffer = PacketSniffer(config)
        sniffer.start()

        received = 0
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and received < 2:
            try:
                pkt = sniffer.packet_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if pkt is not None:
                received += 1

        sniffer.stop()
        assert received == 2
        assert sniffer.metrics.mode == "stopped"
        assert sniffer.metrics.packets_enqueued == 2
    finally:
        sniffer_module.conf.L3socket = original_l3


def test_sniffer_simulation_file_mode_direct(tmp_path: Path):
    pcap_path = tmp_path / "direct.pcap"
    try:
        from scapy.all import IP, UDP, wrpcap

        wrpcap(str(pcap_path), [IP(src="1.1.1.1", dst="2.2.2.2") / UDP(sport=53, dport=53)])
    except Exception:
        pytest.skip("Scapy pcap write unavailable")

    config = _make_config(
        simulation_file=pcap_path,
        max_packets=1,
        queue_maxsize=5,
    )

    import src.ingestion.sniffer as sniffer_module

    original_l3 = sniffer_module.conf.L3socket

    def raise_permission(*_args, **_kwargs):
        raise PermissionError("Operation not permitted")

    sniffer_module.conf.L3socket = raise_permission  # type: ignore[assignment]
    try:
        sniffer = PacketSniffer(config)
        sniffer.start()
        time.sleep(0.5)
        sniffer.stop()

        assert sniffer.metrics.packets_captured >= 1
        assert sniffer.metrics.packets_enqueued >= 1
    finally:
        sniffer_module.conf.L3socket = original_l3


def test_sniffer_max_packets_honored():
    config = _make_config(max_packets=3, queue_maxsize=10)
    sniffer = PacketSniffer(
        config,
        packet_source=lambda: _iter_packets(10),
    )
    sniffer.start()
    time.sleep(0.3)
    sniffer.stop()
    assert sniffer.metrics.packets_captured == 3


def test_sniffer_double_start_is_safe():
    config = _make_config(max_packets=1)
    sniffer = PacketSniffer(
        config,
        packet_source=lambda: _iter_packets(1),
    )
    sniffer.start()
    sniffer.start()
    time.sleep(0.2)
    sniffer.stop()
    assert sniffer.metrics.packets_captured >= 1
