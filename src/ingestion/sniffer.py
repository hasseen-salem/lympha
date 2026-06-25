from __future__ import annotations

import errno
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scapy.all import conf
from scapy.packet import Packet

from src.utils.config_loader import SnifferConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_MTU: int = 65535
_SENTINEL: object = object()

PacketSource = Callable[[], Iterator[Packet]]


class SnifferError(Exception):
    """Base error for packet capture failures."""


class SnifferPermissionError(SnifferError):
    """Raised when raw socket access is denied (typically missing CAP_NET_RAW)."""


@dataclass
class SnifferMetrics:
    packets_captured: int = 0
    packets_enqueued: int = 0
    packets_dropped: int = 0
    backpressure_warnings: int = 0
    backpressure_critical: int = 0
    capture_errors: int = 0
    mode: str = "idle"

    def snapshot(self) -> dict[str, int | str]:
        return {
            "packets_captured": self.packets_captured,
            "packets_enqueued": self.packets_enqueued,
            "packets_dropped": self.packets_dropped,
            "backpressure_warnings": self.backpressure_warnings,
            "backpressure_critical": self.backpressure_critical,
            "capture_errors": self.capture_errors,
            "mode": self.mode,
        }


@dataclass
class _MetricsState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    metrics: SnifferMetrics = field(default_factory=SnifferMetrics)


class PacketSniffer:
    """Thread-safe packet capture producer feeding a bounded consumer queue."""

    def __init__(
        self,
        config: SnifferConfig,
        *,
        packet_source: PacketSource | None = None,
    ) -> None:
        self._config = config
        self._packet_source = packet_source
        self._packet_queue: queue.Queue[Packet | None] = queue.Queue(
            maxsize=config.queue_maxsize,
        )
        self._metrics_state = _MetricsState()
        self._running = False
        self._started = False
        self._thread: threading.Thread | None = None
        self._socket: Any = None
        self._last_error: str | None = None
        self._stop_event = threading.Event()

    @property
    def config(self) -> SnifferConfig:
        return self._config

    @property
    def packet_queue(self) -> queue.Queue[Packet | None]:
        return self._packet_queue

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def metrics(self) -> SnifferMetrics:
        with self._metrics_state.lock:
            return SnifferMetrics(**self._metrics_state.metrics.snapshot())

    def queue_fill_ratio(self) -> float:
        return self._packet_queue.qsize() / self._config.queue_maxsize

    def start(self) -> None:
        if self._running:
            logger.warning(
                "Sniffer already running (interface=%s mode=%s)",
                self._config.interface,
                self.metrics.mode,
            )
            return

        self._last_error = None
        self._stop_event.clear()
        self._running = True
        self._started = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"lympha-sniffer-{self._config.interface}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Sniffer thread started (interface=%s filter=%s queue_maxsize=%d)",
            self._config.interface,
            self._config.bpf_filter,
            self._config.queue_maxsize,
        )

    def _set_mode(self, mode: str) -> None:
        with self._metrics_state.lock:
            self._metrics_state.metrics.mode = mode

    def _increment_metric(self, field_name: str, amount: int = 1) -> None:
        with self._metrics_state.lock:
            current = getattr(self._metrics_state.metrics, field_name)
            setattr(self._metrics_state.metrics, field_name, current + amount)

    def _capture_loop(self) -> None:
        try:
            if self._packet_source is not None:
                self._run_source_capture(self._packet_source())
                return

            simulation_file = self._config.simulation_file
            try:
                self._run_live_capture()
            except SnifferError as exc:
                self._record_error(str(exc))
                if simulation_file is not None and simulation_file.exists():
                    logger.warning(
                        "Live capture failed (%s); falling back to simulation file %s",
                        exc,
                        simulation_file,
                    )
                    self._run_simulation_capture(simulation_file)
                else:
                    raise
        except SnifferError as exc:
            self._record_error(str(exc))
            logger.error("Sniffer capture failed: %s", exc)
        except Exception as exc:
            self._record_error(str(exc))
            self._increment_metric("capture_errors")
            logger.exception("Unexpected sniffer failure: %s", exc)
        finally:
            self._running = False
            self._close_socket()
            self._set_mode("stopped")
            self._signal_stop_sentinel()

    def _record_error(self, message: str) -> None:
        self._last_error = message
        self._increment_metric("capture_errors")

    def _run_live_capture(self) -> None:
        self._set_mode("live")
        logger.info(
            "Opening live capture on %s (filter=%s)",
            self._config.interface,
            self._config.bpf_filter,
        )
        try:
            self._socket = conf.L3socket(
                iface=self._config.interface,
                filter=self._config.bpf_filter,
            )
        except PermissionError as exc:
            raise SnifferPermissionError(
                f"Permission denied opening raw socket on {self._config.interface}: {exc}",
            ) from exc
        except OSError as exc:
            if exc.errno in {errno.EPERM, errno.EACCES}:
                raise SnifferPermissionError(
                    f"Insufficient privileges for {self._config.interface}: {exc}",
                ) from exc
            raise SnifferError(
                f"Failed to open L3socket on {self._config.interface}: {exc}",
            ) from exc
        except Exception as exc:
            raise SnifferError(
                f"Failed to open L3socket on {self._config.interface}: {exc}",
            ) from exc

        logger.info("L3socket opened on %s", self._config.interface)
        count = 0
        try:
            while self._running and not self._stop_event.is_set():
                try:
                    pkt = self._socket.recv(_MTU)
                except TypeError:
                    continue
                except OSError as exc:
                    if not self._running:
                        break
                    self._increment_metric("capture_errors")
                    logger.warning("Socket recv error on %s: %s", self._config.interface, exc)
                    continue
                if pkt is None:
                    continue
                count += 1
                self._enqueue_packet(pkt)
                if self._config.max_packets > 0 and count >= self._config.max_packets:
                    logger.info(
                        "Reached max_packets=%d on %s; stopping capture",
                        self._config.max_packets,
                        self._config.interface,
                    )
                    break
        finally:
            self._close_socket()

    def _run_simulation_capture(self, simulation_file: Path) -> None:
        self._set_mode("simulation")
        try:
            from scapy.utils import PcapReader
        except ImportError as exc:
            raise SnifferError("Scapy PcapReader unavailable for simulation capture") from exc

        if not simulation_file.exists():
            raise SnifferError(f"Simulation file not found: {simulation_file}")

        logger.info("Replaying packets from %s", simulation_file)
        count = 0
        try:
            with PcapReader(str(simulation_file)) as reader:
                for pkt in reader:
                    if not self._running or self._stop_event.is_set():
                        break
                    count += 1
                    self._enqueue_packet(pkt)
                    if self._config.max_packets > 0 and count >= self._config.max_packets:
                        break
        except Exception as exc:
            raise SnifferError(
                f"Failed to read simulation file {simulation_file}: {exc}",
            ) from exc

    def _run_source_capture(self, packets: Iterator[Packet]) -> None:
        self._set_mode("inject")
        count = 0
        for pkt in packets:
            if not self._running or self._stop_event.is_set():
                break
            count += 1
            self._enqueue_packet(pkt)
            if self._config.max_packets > 0 and count >= self._config.max_packets:
                break

    def _enqueue_packet(self, pkt: Packet) -> None:
        self._increment_metric("packets_captured")
        fill_ratio = self.queue_fill_ratio()

        if fill_ratio >= self._config.backpressure_critical_ratio:
            self._increment_metric("backpressure_critical")
            logger.error(
                "Queue critical on %s: %.1f%% full (%d/%d) — packet dropped",
                self._config.interface,
                fill_ratio * 100,
                self._packet_queue.qsize(),
                self._config.queue_maxsize,
            )
            self._increment_metric("packets_dropped")
            return

        if fill_ratio >= self._config.backpressure_warn_ratio:
            self._increment_metric("backpressure_warnings")
            logger.warning(
                "Backpressure on %s: %.1f%% full (%d/%d)",
                self._config.interface,
                fill_ratio * 100,
                self._packet_queue.qsize(),
                self._config.queue_maxsize,
            )

        try:
            self._packet_queue.put(
                pkt,
                timeout=self._config.recv_timeout_sec,
            )
            self._increment_metric("packets_enqueued")
        except queue.Full:
            self._increment_metric("packets_dropped")
            logger.warning(
                "Packet dropped — queue saturated on %s (%d/%d)",
                self._config.interface,
                self._packet_queue.qsize(),
                self._config.queue_maxsize,
            )

    def _close_socket(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def _signal_stop_sentinel(self) -> None:
        try:
            self._packet_queue.put(None, timeout=self._config.recv_timeout_sec)
        except queue.Full:
            try:
                self._packet_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._packet_queue.put_nowait(None)
            except queue.Full:
                logger.warning(
                    "Could not deliver stop sentinel on %s — queue full",
                    self._config.interface,
                )

    def stop(self, timeout: float = 2.0) -> None:
        if not self._started and not self._running:
            return

        logger.info("Stopping sniffer on %s ...", self._config.interface)
        self._running = False
        self._stop_event.set()
        self._close_socket()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "Sniffer thread did not terminate within %.1fs on %s",
                    timeout,
                    self._config.interface,
                )

        self._thread = None
        self._signal_stop_sentinel()
        self._set_mode("stopped")
        logger.info(
            "Sniffer stopped on %s (captured=%d enqueued=%d dropped=%d)",
            self._config.interface,
            self.metrics.packets_captured,
            self.metrics.packets_enqueued,
            self.metrics.packets_dropped,
        )
