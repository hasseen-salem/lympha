from __future__ import annotations

import argparse
import queue
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.features.aggregator import FlowAggregator
from src.ingestion.sniffer import PacketSniffer
from src.models.inference import RealTimeInferenceEngine, ThreatProfile
from src.models.lympha_net import LymphaNet
from src.prevention.firewall import BlockResult, FirewallController
from src.utils.config_loader import AppConfig, load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_DEFAULT_QUEUE_DRAIN_LIMIT = 256
_DEFAULT_POLL_INTERVAL_SEC = 0.05


@dataclass
class DaemonMetrics:
    packets_processed: int = 0
    flows_scored: int = 0
    flows_blocked: int = 0
    inference_errors: int = 0
    mitigation_errors: int = 0
    sniffer_packets_dropped: int = 0
    detection_events: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "packets_processed": self.packets_processed,
            "flows_scored": self.flows_scored,
            "flows_blocked": self.flows_blocked,
            "inference_errors": self.inference_errors,
            "mitigation_errors": self.mitigation_errors,
            "sniffer_packets_dropped": self.sniffer_packets_dropped,
            "detection_events": list(self.detection_events),
        }


class LymphaDaemon:
    """Production orchestrator for Lympha live intrusion prevention."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        sniffer: PacketSniffer | None = None,
        aggregator: FlowAggregator | None = None,
        inference: RealTimeInferenceEngine | None = None,
        firewall: FirewallController | None = None,
        install_signals: bool = True,
    ) -> None:
        self._config = config or load_config()
        self._shutdown_event = threading.Event()
        self._started = False
        self._lock = threading.RLock()
        self._metrics = DaemonMetrics()
        self._last_flush = time.monotonic()
        self._flush_interval_sec = self._config.execution.flush_interval_sec
        self._install_signals = install_signals
        self._signals_installed = False
        self._previous_handlers: dict[int, Any] = {}

        self.sniffer = sniffer or PacketSniffer(self._config.sniffer)
        self.aggregator = aggregator or FlowAggregator(self._config.aggregator)
        self.inference = inference or self._build_inference_engine()
        self.firewall = firewall or FirewallController(self._config)

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def metrics(self) -> DaemonMetrics:
        with self._lock:
            return DaemonMetrics(**self._metrics.snapshot())

    @property
    def running(self) -> bool:
        return self._started and not self._shutdown_event.is_set()

    def _build_inference_engine(self) -> RealTimeInferenceEngine:
        if self._config.execution.simulation_mode:
            scaler = RealTimeInferenceEngine.build_mock_scaler(
                feature_count=self._config.model.continuous_dim,
            )
            model = LymphaNet.from_config(self._config.model)
            return RealTimeInferenceEngine(
                self._config,
                model=model,
                scaler=scaler,
                require_weights=False,
            )
        return RealTimeInferenceEngine.from_artifacts(
            self._config,
            require_weights=False,
        )

    def _install_signal_handlers(self) -> None:
        if self._signals_installed or not self._install_signals:
            return

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)
            except (ValueError, OSError) as exc:
                logger.warning("Could not register handler for signal %s: %s", sig, exc)

        self._signals_installed = True
        logger.debug("Signal handlers installed for SIGINT and SIGTERM")

    def _restore_signal_handlers(self) -> None:
        if not self._signals_installed:
            return
        for sig, handler in self._previous_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        self._signals_installed = False
        self._previous_handlers.clear()

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        logger.info("Received signal %s — requesting graceful shutdown", signum)
        self.request_shutdown()

    def request_shutdown(self) -> None:
        self._shutdown_event.set()

    def start(self) -> None:
        with self._lock:
            if self._started:
                logger.debug("Lympha daemon already started")
                return

            self._shutdown_event.clear()
            self._install_signal_handlers()
            self._metrics = DaemonMetrics()
            self._last_flush = time.monotonic()

            if self._config.mitigation.enabled:
                self.firewall.initialize()
            self.aggregator.start()
            self.sniffer.start()
            self._started = True
            logger.info(
                "Lympha daemon started (interface=%s flush_interval=%.2fs threshold=%.2f)",
                self._config.sniffer.interface,
                self._flush_interval_sec,
                self._config.execution.detection_threshold,
            )

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return

            self._shutdown_event.set()
            logger.info("Stopping Lympha daemon ...")

            try:
                if self.sniffer.running:
                    self.sniffer.stop()
            except Exception:
                logger.exception("Sniffer shutdown failed")

            try:
                self.aggregator.stop()
            except Exception:
                logger.exception("Aggregator shutdown failed")

            try:
                self.firewall.cleanup()
            except Exception:
                logger.exception("Firewall cleanup failed")

            self._restore_signal_handlers()
            self._started = False
            logger.info(
                "Lympha daemon stopped (packets=%d flows_scored=%d blocked=%d inference_errors=%d)",
                self._metrics.packets_processed,
                self._metrics.flows_scored,
                self._metrics.flows_blocked,
                self._metrics.inference_errors,
            )

    def _drain_sniffer_queue(self, *, limit: int = _DEFAULT_QUEUE_DRAIN_LIMIT) -> int:
        processed = 0
        while processed < limit and not self._shutdown_event.is_set():
            try:
                packet = self.sniffer.packet_queue.get(
                    timeout=self._config.sniffer.recv_timeout_sec,
                )
            except queue.Empty:
                break

            if packet is None:
                logger.debug("Sniffer stop sentinel received")
                self.request_shutdown()
                break

            self.aggregator.ingest(packet)
            processed += 1

        if processed:
            with self._lock:
                self._metrics.packets_processed += processed

        sniffer_metrics = self.sniffer.metrics
        with self._lock:
            self._metrics.sniffer_packets_dropped = sniffer_metrics.packets_dropped

        return processed

    def _record_detection_events(self, profiles: list[ThreatProfile]) -> None:
        limit = self._config.execution.detection_event_limit
        for profile in profiles:
            if not profile.should_block:
                continue
            event = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "src_ip": profile.src_ip,
                "dst_ip": profile.dst_ip,
                "malicious_prob": round(profile.malicious_prob, 4),
                "action": "BLOCKED",
            }
            with self._lock:
                self._metrics.detection_events.append(event)
                if len(self._metrics.detection_events) > limit:
                    self._metrics.detection_events.pop(0)

    def _run_inference_cycle(self) -> None:
        flow_df = self.aggregator.drain()
        if flow_df.empty:
            return

        try:
            profiles = self.inference.predict_flows(flow_df)
        except Exception:
            with self._lock:
                self._metrics.inference_errors += 1
            logger.exception(
                "Inference cycle failed for %d flow(s) — ingress will continue",
                len(flow_df),
            )
            return

        with self._lock:
            self._metrics.flows_scored += len(profiles)

        threats = [profile for profile in profiles if profile.should_block]
        self._record_detection_events(threats)

        if not threats or not self._config.mitigation.enabled:
            return

        try:
            result = self.firewall.block_threats(profiles)
        except Exception:
            with self._lock:
                self._metrics.mitigation_errors += 1
            logger.exception(
                "Mitigation cycle failed for %d threat(s) — ingress will continue",
                len(threats),
            )
            return

        with self._lock:
            self._metrics.flows_blocked += len(result.blocked)

    def process_once(self) -> None:
        self._drain_sniffer_queue()
        now = time.monotonic()
        if now - self._last_flush >= self._flush_interval_sec:
            self._run_inference_cycle()
            self._last_flush = now

    def run(
        self,
        *,
        poll_interval_sec: float = _DEFAULT_POLL_INTERVAL_SEC,
        max_cycles: int | None = None,
    ) -> None:
        self.start()
        cycles = 0
        try:
            while not self._shutdown_event.is_set():
                self.process_once()
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                time.sleep(poll_interval_sec)
        finally:
            self.stop()

    def run_until_shutdown(
        self,
        *,
        poll_interval_sec: float = _DEFAULT_POLL_INTERVAL_SEC,
    ) -> None:
        self.run(poll_interval_sec=poll_interval_sec, max_cycles=None)


def build_daemon(config_path: str | None = None, **kwargs: Any) -> LymphaDaemon:
    config = load_config(config_path) if config_path else load_config()
    return LymphaDaemon(config, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lympha — Unified Network Intrusion Prevention Daemon",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    daemon = build_daemon(args.config)
    try:
        daemon.run_until_shutdown()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        daemon.request_shutdown()
        daemon.stop()


if __name__ == "__main__":
    main()
