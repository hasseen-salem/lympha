from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import ThreatProfile
from src.prevention.firewall import (
    BlockResult,
    FirewallController,
    FirewallError,
    InvalidIpError,
    NftablesExecutionError,
    NftablesPrivilegeError,
    NftablesTimeoutError,
    build_batch_block_script,
    build_cleanup_script,
    build_init_script,
    sanitize_ipv4,
)
from src.utils.config_loader import MitigationConfig


def _mitigation_config(**overrides) -> MitigationConfig:
    defaults = {
        "enabled": True,
        "table_name": "lympha",
        "set_name": "blackhole",
        "chain_name": "input",
        "block_timeout_sec": 3600,
        "whitelist_ips": ["127.0.0.1", "192.168.100.1"],
        "whitelist_prefixes": ["224.", "239."],
    }
    defaults.update(overrides)
    return MitigationConfig(**defaults)


class MockNftRunner:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stderr: str = "",
        fail_on_script_contains: str | None = None,
        timeout: bool = False,
    ) -> None:
        self.calls: list[tuple[list[str], str | None, float]] = []
        self.scripts: list[str] = []
        self.returncode = returncode
        self.stderr = stderr
        self.fail_on_script_contains = fail_on_script_contains
        self.timeout = timeout

    def __call__(
        self,
        argv,
        *,
        input: str | None,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        argv_list = list(argv)
        self.calls.append((argv_list, input, timeout))
        if input:
            self.scripts.append(input)

        if self.timeout:
            raise subprocess.TimeoutExpired(cmd=argv_list, timeout=timeout)

        if self.fail_on_script_contains and input and self.fail_on_script_contains in input:
            return subprocess.CompletedProcess(
                args=argv_list,
                returncode=1,
                stdout="",
                stderr=self.fail_on_script_contains,
            )

        return subprocess.CompletedProcess(
            args=argv_list,
            returncode=self.returncode,
            stdout="",
            stderr=self.stderr,
        )


def test_sanitize_ipv4_valid():
    assert sanitize_ipv4("10.0.0.5") == "10.0.0.5"
    assert sanitize_ipv4(" 192.168.1.1 ") == "192.168.1.1"


def test_sanitize_ipv4_rejects_malformed_and_ipv6():
    with pytest.raises(InvalidIpError):
        sanitize_ipv4("not-an-ip")
    with pytest.raises(InvalidIpError):
        sanitize_ipv4("1.2.3")
    with pytest.raises(InvalidIpError):
        sanitize_ipv4("::1")


def test_build_init_script_structure():
    script = build_init_script(table_name="lympha", set_name="blackhole", chain_name="input")
    assert "add table inet lympha" in script
    assert "add set inet lympha blackhole" in script
    assert "flags timeout" in script
    assert "ip saddr @blackhole drop" in script


def test_build_batch_block_script_single_transaction():
    script = build_batch_block_script(
        table_name="lympha",
        set_name="blackhole",
        ips=["1.2.3.4", "5.6.7.8"],
        timeout_sec=3600,
    )
    assert script.count("add element") == 1
    assert "1.2.3.4 timeout 3600s" in script
    assert "5.6.7.8 timeout 3600s" in script
    assert ";" not in script
    assert "|" not in script


def test_build_cleanup_script():
    assert build_cleanup_script(table_name="lympha") == "delete table inet lympha\n"


def test_initialize_executes_single_script_with_shell_false():
    runner = MockNftRunner()
    controller = FirewallController(_mitigation_config(), nft_runner=runner)
    controller.initialize()

    assert len(runner.scripts) == 1
    assert runner.calls[0][0] == ["nft", "-f", "-"]
    assert "add table inet lympha" in runner.scripts[0]


def test_initialize_is_idempotent():
    runner = MockNftRunner()
    controller = FirewallController(_mitigation_config(), nft_runner=runner)
    controller.initialize()
    controller.initialize()
    assert len(runner.scripts) == 1


def test_block_ips_batches_in_one_transaction():
    runner = MockNftRunner()
    controller = FirewallController(_mitigation_config(), nft_runner=runner)
    controller.initialize()

    result = controller.block_ips(["10.1.1.1", "10.1.1.2", "10.1.1.3"])
    assert result.blocked == ("10.1.1.1", "10.1.1.2", "10.1.1.3")
    assert len(runner.scripts) == 2
    batch_script = runner.scripts[1]
    assert batch_script.count("add element") == 1
    assert "10.1.1.1 timeout 3600s" in batch_script
    assert len(controller.active_bans) == 3


def test_block_ips_skips_whitelist_and_invalid():
    runner = MockNftRunner()
    controller = FirewallController(_mitigation_config(), nft_runner=runner)
    controller.initialize()

    result = controller.block_ips(
        ["127.0.0.1", "bad-ip", "10.2.2.2", "224.0.0.1"],
    )
    assert result.blocked == ("10.2.2.2",)
    assert result.skipped_whitelist == ("127.0.0.1", "224.0.0.1")
    assert result.skipped_invalid == ("bad-ip",)


def test_block_ips_skips_duplicates():
    runner = MockNftRunner()
    controller = FirewallController(_mitigation_config(), nft_runner=runner)
    controller.initialize()

    first = controller.block_ips(["10.9.9.9"])
    second = controller.block_ips(["10.9.9.9", "10.9.9.10"])
    assert first.blocked == ("10.9.9.9",)
    assert second.blocked == ("10.9.9.10",)
    assert second.skipped_duplicate == ("10.9.9.9",)


def test_block_threats_uses_should_block_flag():
    runner = MockNftRunner()
    controller = FirewallController(_mitigation_config(), nft_runner=runner)
    controller.initialize()

    profiles = [
        ThreatProfile("10.3.3.3", "10.0.0.1", 0.95, True),
        ThreatProfile("10.4.4.4", "10.0.0.2", 0.10, False),
    ]
    result = controller.block_threats(profiles)
    assert result.blocked == ("10.3.3.3",)


def test_injection_payload_is_rejected_not_executed():
    runner = MockNftRunner()
    controller = FirewallController(_mitigation_config(), nft_runner=runner)
    controller.initialize()

    malicious = "1.2.3.4; drop table inet lympha"
    result = controller.block_ips([malicious])
    assert result.blocked == ()
    assert len(result.skipped_invalid) == 1
    assert all(malicious not in script for script in runner.scripts)


def test_privilege_error_is_raised_cleanly():
    runner = MockNftRunner(stderr="Operation not permitted")
    runner.returncode = 1
    controller = FirewallController(_mitigation_config(), nft_runner=runner)

    with pytest.raises(NftablesPrivilegeError):
        controller.initialize()


def test_execution_timeout_is_raised():
    runner = MockNftRunner(timeout=True)
    controller = FirewallController(
        _mitigation_config(),
        nft_runner=runner,
        command_timeout_sec=0.01,
    )

    with pytest.raises(NftablesTimeoutError):
        controller.initialize()


def test_cleanup_deletes_table_and_resets_state():
    runner = MockNftRunner()
    controller = FirewallController(_mitigation_config(), nft_runner=runner)
    controller.initialize()
    controller.block_ips(["10.8.8.8"])

    controller.cleanup()
    assert not controller.initialized
    assert controller.active_bans == {}
    assert "delete table inet lympha" in runner.scripts[-1]


def test_cleanup_is_safe_when_not_initialized():
    runner = MockNftRunner()
    controller = FirewallController(_mitigation_config(), nft_runner=runner)
    controller.cleanup()
    assert runner.calls == []


def test_disabled_mitigation_is_noop():
    runner = MockNftRunner()
    controller = FirewallController(_mitigation_config(enabled=False), nft_runner=runner)
    controller.initialize()
    result = controller.block_ips(["10.5.5.5"])
    assert result == BlockResult()
    assert runner.calls == []


def test_invalid_table_name_rejected():
    with pytest.raises(FirewallError, match="Invalid nftables table_name"):
        FirewallController(_mitigation_config(table_name="bad;name"))


def test_list_blocked_parses_nft_output():
    runner = MockNftRunner()

    def runner_with_list(argv, *, input, timeout):
        runner(argv, input=input, timeout=timeout)
        if argv[:2] == ["nft", "list"]:
            return subprocess.CompletedProcess(
                args=list(argv),
                returncode=0,
                stdout="elements = { 1.1.1.1, 2.2.2.2 }",
                stderr="",
            )
        return subprocess.CompletedProcess(args=list(argv), returncode=0, stdout="", stderr="")

    controller = FirewallController(_mitigation_config(), nft_runner=runner_with_list)
    controller.initialize()
    blocked = controller.list_blocked()
    assert blocked == ["1.1.1.1", "2.2.2.2"]
