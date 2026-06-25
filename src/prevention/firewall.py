from __future__ import annotations

import ipaddress
import re
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from src.models.inference import ThreatProfile
from src.utils.config_loader import AppConfig, MitigationConfig, load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_DEFAULT_TIMEOUT_SEC = 30.0


class FirewallError(RuntimeError):
    """Base firewall controller failure."""


class InvalidIpError(FirewallError):
    """Raised when an IP address cannot be normalized safely."""


class NftablesPrivilegeError(FirewallError):
    """Raised when nftables cannot be modified due to missing privileges."""


class NftablesExecutionError(FirewallError):
    """Raised when an nftables transaction fails."""


class NftablesTimeoutError(FirewallError):
    """Raised when nftables execution exceeds the configured timeout."""


class NftRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        input: str | None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True)
class BanRecord:
    ip: str
    banned_at: float
    expires_at: float | None


@dataclass(frozen=True)
class BlockResult:
    blocked: tuple[str, ...] = ()
    skipped_whitelist: tuple[str, ...] = ()
    skipped_invalid: tuple[str, ...] = ()
    skipped_duplicate: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()


@dataclass
class FirewallState:
    initialized: bool = False
    active_bans: dict[str, BanRecord] = field(default_factory=dict)


def _validate_identifier(name: str, label: str) -> str:
    if not _IDENTIFIER_RE.match(name):
        raise FirewallError(f"Invalid nftables {label}: {name!r}")
    return name


def sanitize_ipv4(ip_address: str) -> str:
    try:
        parsed = ipaddress.ip_address(ip_address.strip())
    except ValueError as exc:
        raise InvalidIpError(f"Malformed IP address: {ip_address!r}") from exc
    if parsed.version != 4:
        raise InvalidIpError(
            f"IPv6 is unsupported by the current nftables set configuration: {ip_address!r}",
        )
    return str(parsed)


def build_batch_block_script(
    *,
    table_name: str,
    set_name: str,
    ips: Sequence[str],
    timeout_sec: int,
) -> str:
    if not ips:
        return ""

    elements: list[str] = []
    for ip in ips:
        if timeout_sec > 0:
            elements.append(f"{ip} timeout {timeout_sec}s")
        else:
            elements.append(ip)

    joined = ", ".join(elements)
    return f"add element inet {table_name} {set_name} {{ {joined} }}\n"


def build_init_script(
    *,
    table_name: str,
    set_name: str,
    chain_name: str,
) -> str:
    return (
        f"add table inet {table_name}\n"
        f"add set inet {table_name} {set_name} "
        f"{{ type ipv4_addr; flags timeout; }}\n"
        f"add chain inet {table_name} {chain_name} "
        f"{{ type filter hook input priority 0; policy accept; }}\n"
        f"add rule inet {table_name} {chain_name} ip saddr @{set_name} drop\n"
    )


def build_cleanup_script(*, table_name: str) -> str:
    return f"delete table inet {table_name}\n"


def _default_nft_runner(
    argv: Sequence[str],
    *,
    input: str | None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        input=input,
        capture_output=True,
        text=True,
        shell=False,
        timeout=timeout,
        check=False,
    )


class FirewallController:
    """Transaction-safe nftables mitigation controller for Lympha."""

    def __init__(
        self,
        config: MitigationConfig | AppConfig | None = None,
        *,
        nft_runner: NftRunner | None = None,
        command_timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    ) -> None:
        if isinstance(config, AppConfig) or config is None:
            app_config = config or load_config()
            self._mitigation = app_config.mitigation
        else:
            self._mitigation = config

        self._table_name = _validate_identifier(self._mitigation.table_name, "table_name")
        self._set_name = _validate_identifier(self._mitigation.set_name, "set_name")
        self._chain_name = _validate_identifier(self._mitigation.chain_name, "chain_name")
        self._nft_runner = nft_runner or _default_nft_runner
        self._command_timeout_sec = command_timeout_sec
        self._state = FirewallState()
        self._lock = threading.RLock()

    @property
    def config(self) -> MitigationConfig:
        return self._mitigation

    @property
    def initialized(self) -> bool:
        with self._lock:
            return self._state.initialized

    @property
    def active_bans(self) -> dict[str, BanRecord]:
        with self._lock:
            return dict(self._state.active_bans)

    def is_whitelisted(self, ip_address: str) -> bool:
        try:
            normalized = sanitize_ipv4(ip_address)
        except InvalidIpError:
            return False

        if normalized in self._mitigation.whitelist_ips:
            return True
        return any(normalized.startswith(prefix) for prefix in self._mitigation.whitelist_prefixes)

    def _classify_ip(self, ip_address: str) -> tuple[str | None, str | None]:
        if self.is_whitelisted(ip_address):
            return None, "whitelist"
        try:
            return sanitize_ipv4(ip_address), None
        except InvalidIpError:
            return None, "invalid"

    def _execute_argv(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            result = self._nft_runner(
                argv,
                input=None,
                timeout=self._command_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise NftablesTimeoutError(
                f"nft command timed out after {self._command_timeout_sec}s: {' '.join(argv)}",
            ) from exc

        if result.returncode == 0:
            return result

        stderr = (result.stderr or "").strip()
        if "Permission denied" in stderr or "Operation not permitted" in stderr:
            raise NftablesPrivilegeError(stderr or "Insufficient privileges for nftables")
        if "No such file or directory" in stderr and "nft" in stderr:
            raise NftablesExecutionError("nft binary not found — install nftables")

        raise NftablesExecutionError(
            f"nft command failed ({result.returncode}): {' '.join(argv)}\n{stderr}",
        )

    def _execute_script(self, script: str) -> subprocess.CompletedProcess[str]:
        if not script.strip():
            return subprocess.CompletedProcess(args=["nft", "-f", "-"], returncode=0, stdout="", stderr="")

        logger.debug("Executing nft script:\n%s", script.strip())
        try:
            result = self._nft_runner(
                ["nft", "-f", "-"],
                input=script,
                timeout=self._command_timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise NftablesTimeoutError(
                f"nft script timed out after {self._command_timeout_sec}s",
            ) from exc

        if result.returncode == 0:
            return result

        stderr = (result.stderr or "").strip()
        if "Permission denied" in stderr or "Operation not permitted" in stderr:
            raise NftablesPrivilegeError(stderr or "Insufficient privileges for nftables")
        if "No such file or directory" in stderr and "nft" in stderr:
            raise NftablesExecutionError("nft binary not found — install nftables")

        if "File exists" in stderr or "already exists" in stderr:
            logger.debug("nft idempotency notice: %s", stderr)
            return result

        raise NftablesExecutionError(
            f"nft script failed ({result.returncode}):\n{script}\n{stderr}",
        )

    def initialize(self) -> None:
        if not self._mitigation.enabled:
            logger.info("Mitigation disabled — skipping nftables initialization")
            return

        with self._lock:
            if self._state.initialized:
                logger.debug("Firewall already initialized for table %s", self._table_name)
                return

            script = build_init_script(
                table_name=self._table_name,
                set_name=self._set_name,
                chain_name=self._chain_name,
            )
            self._execute_script(script)
            self._state.initialized = True
            logger.info(
                "nftables initialized (table=%s set=%s chain=%s timeout=%ss)",
                self._table_name,
                self._set_name,
                self._chain_name,
                self._mitigation.block_timeout_sec,
            )

    def block_ips(self, ip_list: Iterable[str]) -> BlockResult:
        if not self._mitigation.enabled:
            return BlockResult()

        with self._lock:
            if not self._state.initialized:
                self.initialize()

            blocked: list[str] = []
            skipped_whitelist: list[str] = []
            skipped_invalid: list[str] = []
            skipped_duplicate: list[str] = []
            failed: list[str] = []
            to_add: list[str] = []

            for raw_ip in ip_list:
                normalized, reason = self._classify_ip(raw_ip)
                if reason == "whitelist":
                    skipped_whitelist.append(raw_ip)
                    continue
                if reason == "invalid" or normalized is None:
                    skipped_invalid.append(raw_ip)
                    continue
                if normalized in self._state.active_bans:
                    skipped_duplicate.append(normalized)
                    continue
                to_add.append(normalized)

            if not to_add:
                return BlockResult(
                    blocked=tuple(blocked),
                    skipped_whitelist=tuple(skipped_whitelist),
                    skipped_invalid=tuple(skipped_invalid),
                    skipped_duplicate=tuple(skipped_duplicate),
                    failed=tuple(failed),
                )

            script = build_batch_block_script(
                table_name=self._table_name,
                set_name=self._set_name,
                ips=to_add,
                timeout_sec=self._mitigation.block_timeout_sec,
            )

            try:
                self._execute_script(script)
            except FirewallError:
                failed.extend(to_add)
                raise
            except Exception:
                failed.extend(to_add)
                raise

            now = time.time()
            expires_at = (
                now + self._mitigation.block_timeout_sec
                if self._mitigation.block_timeout_sec > 0
                else None
            )
            for ip in to_add:
                self._state.active_bans[ip] = BanRecord(
                    ip=ip,
                    banned_at=now,
                    expires_at=expires_at,
                )
                blocked.append(ip)

            logger.info("Blocked %d IP(s) in single nft transaction", len(blocked))
            return BlockResult(
                blocked=tuple(blocked),
                skipped_whitelist=tuple(skipped_whitelist),
                skipped_invalid=tuple(skipped_invalid),
                skipped_duplicate=tuple(skipped_duplicate),
                failed=tuple(failed),
            )

    def block_threats(self, profiles: Iterable[ThreatProfile]) -> BlockResult:
        ips = [profile.src_ip for profile in profiles if profile.should_block]
        return self.block_ips(ips)

    def unblock_ip(self, ip_address: str) -> None:
        normalized = sanitize_ipv4(ip_address)
        script = f"delete element inet {self._table_name} {self._set_name} {{ {normalized} }}\n"
        self._execute_script(script)
        with self._lock:
            self._state.active_bans.pop(normalized, None)
        logger.info("Unblocked IP %s", normalized)

    def list_blocked(self) -> list[str]:
        try:
            result = self._execute_argv(
                ["nft", "list", "set", "inet", self._table_name, self._set_name],
            )
        except NftablesExecutionError:
            return []

        matches = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", result.stdout or "")
        return sorted(set(matches))

    def cleanup(self) -> None:
        if not self._mitigation.enabled:
            return

        with self._lock:
            if not self._state.initialized:
                return

            script = build_cleanup_script(table_name=self._table_name)
            try:
                self._execute_script(script)
            except NftablesExecutionError as exc:
                logger.warning("Firewall cleanup encountered an error: %s", exc)
            finally:
                self._state.active_bans.clear()
                self._state.initialized = False
                logger.info("Firewall cleanup completed for table %s", self._table_name)

    def teardown(self) -> None:
        self.cleanup()
