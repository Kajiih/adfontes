"""Verify built adfontes cp310-abi3 wheels and optional source distributions."""

# omni:disable-file [repeated-literal] -- CLI arguments and JSON-RPC wire keys read clearest inline; constants would only restate them

from __future__ import annotations

import dataclasses
import json
import pathlib
import platform
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from .conftest import McpRoundtrip, NativeRunner, Outcome, Venv, WheelArchive

MAX_WHEEL_BYTES = 15 * 1024 * 1024
MCP_EXPECTED_REPLY_COUNT = 4
SIGINT_EXIT_CODE = 130
EXTENSION_ENTRIES = frozenset({"adfontes/_adfontes.abi3.so", "adfontes/_adfontes.pyd"})
REQUIRED_ENTRIES = frozenset(
    {
        "adfontes/__init__.py",
        "adfontes/__main__.py",
        "adfontes/_cli.py",
        "adfontes/_adfontes.pyi",
        "adfontes/py.typed",
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class CliCase:
    """One CLI invocation and its expected exit status and stream substrings."""

    args: Sequence[str]
    exit_code: int
    stdout_contains: str = ""
    stderr_contains: str = ""


CLI_CASES = (
    CliCase(("--version",), 0, stdout_contains="adfontes "),
    CliCase(("--help",), 0, stdout_contains="Usage: adfontes"),
    CliCase(("fetch", "--help"), 0, stdout_contains="Usage: adfontes fetch"),
    CliCase(("slice-pdf", "--help"), 0, stdout_contains="Usage: adfontes slice-pdf"),
    CliCase(("mcp", "--help"), 0, stdout_contains="Usage: adfontes mcp"),
    CliCase(("fetch", "ftp://example.org/disallowed"), 1, stderr_contains="safety block"),
    CliCase(("unknown-subcommand",), 2, stderr_contains="Usage:"),
)
MODULE_PARITY_FLAGS = ("--version", "--help")
CLOSED_PIPE_CASES = (
    CliCase(("--version",), 0),
    CliCase(("--help",), 0),
)


@pytest.fixture(scope="session")
def wheel_cli_outcomes(installed_venv: Venv) -> dict[tuple[str, ...], Outcome]:
    """Run each `CLI_CASES` invocation once per installed-wheel session."""
    executable = str(installed_venv.executable("adfontes"))
    return {
        tuple(case.args): installed_venv.run([executable, *case.args])
        for case in CLI_CASES
    }


@pytest.fixture(scope="session")
def native_cli_outcomes(native_runner: NativeRunner) -> dict[tuple[str, ...], Outcome]:
    """Run each `CLI_CASES` invocation once for the standalone reference binary."""
    return {tuple(case.args): native_runner.run(case.args) for case in CLI_CASES}


def test_wheel_archive_tag_and_size(wheel_artifact: pathlib.Path) -> None:
    """Verify the built wheel uses the `cp310-abi3` tag and fits within the size ceiling."""
    assert "-cp310-abi3-" in wheel_artifact.name
    assert wheel_artifact.stat().st_size <= MAX_WHEEL_BYTES


def test_wheel_archive_entries_and_no_scripts(wheel_archive: WheelArchive) -> None:
    """Verify required package files are present and no script stubs are packaged."""
    assert REQUIRED_ENTRIES <= wheel_archive.entries
    assert wheel_archive.script_entries == ()


def test_wheel_abi3_tag_and_single_extension(wheel_archive: WheelArchive) -> None:
    """Verify `.dist-info/WHEEL` declares `cp310-abi3` and packs one native extension."""
    assert len(wheel_archive.entries & EXTENSION_ENTRIES) == 1
    assert sum(name.endswith((".so", ".pyd")) for name in wheel_archive.entries) == 1
    assert "Tag: cp310-abi3-" in wheel_archive.wheel_metadata


def test_native_extension_exports_run_cli(installed_venv: Venv) -> None:
    """Verify the compiled extension imports cleanly and exposes callable `run_cli`."""
    outcome = installed_venv.run(
        [
            str(installed_venv.executable("python")),
            "-c",
            "from adfontes._adfontes import run_cli; assert callable(run_cli)",
        ]
    )
    assert outcome.exit_code == 0, outcome.stderr


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"},
    reason="Apple Silicon Mach-O signature check",
)
def test_macos_extension_codesign(installed_venv: Venv) -> None:
    """Verify the installed `_adfontes.abi3.so` retains a valid Mach-O code signature."""
    locate = installed_venv.run(
        [
            str(installed_venv.executable("python")),
            "-c",
            "import adfontes._adfontes as m; print(m.__file__)",
        ]
    )
    assert locate.exit_code == 0, locate.stderr
    installed_venv.check(["codesign", "--verify", "--strict", locate.stdout.rstrip("\r\n")])


@pytest.mark.parametrize("case", CLI_CASES, ids=lambda case: "_".join(case.args))
def test_cli_subcommands(
    wheel_cli_outcomes: dict[tuple[str, ...], Outcome],
    case: CliCase,
) -> None:
    """Verify wheel console entrypoint exit codes and stream snippets across subcommands."""
    outcome = wheel_cli_outcomes[tuple(case.args)]
    assert outcome.exit_code == case.exit_code, outcome.stderr
    assert case.stdout_contains in outcome.stdout
    assert case.stderr_contains in outcome.stderr


@pytest.mark.parametrize("case", CLI_CASES, ids=lambda case: "_".join(case.args))
def test_native_binary_parity(
    native_cli_outcomes: dict[tuple[str, ...], Outcome],
    wheel_cli_outcomes: dict[tuple[str, ...], Outcome],
    case: CliCase,
) -> None:
    """Verify exact exit-code, stdout, and stderr parity against the standalone binary."""
    assert wheel_cli_outcomes[tuple(case.args)] == native_cli_outcomes[tuple(case.args)]


@pytest.mark.parametrize("cli_flag", MODULE_PARITY_FLAGS)
def test_python_module_entrypoint_parity(
    installed_venv: Venv,
    wheel_cli_outcomes: dict[tuple[str, ...], Outcome],
    cli_flag: str,
) -> None:
    """Verify `python -m adfontes` matches the installed `adfontes` console script."""
    via_module = installed_venv.run(
        [str(installed_venv.executable("python")), "-m", "adfontes", cli_flag]
    )
    assert via_module == wheel_cli_outcomes[(cli_flag,)]


def test_mcp_session_exits_cleanly(mcp_roundtrip: McpRoundtrip) -> None:
    """Verify the batched MCP roundtrip process exits zero without stderr and returns 4 lines."""
    assert mcp_roundtrip.outcome.exit_code == 0, mcp_roundtrip.outcome.stderr
    assert mcp_roundtrip.outcome.stderr == ""
    assert len(mcp_roundtrip.replies) == MCP_EXPECTED_REPLY_COUNT, mcp_roundtrip.replies


def test_mcp_handshake_and_tools_list(mcp_roundtrip: McpRoundtrip) -> None:
    """Verify MCP 2025-06-18 initialization negotiation and advertised tool catalog."""
    assert len(mcp_roundtrip.replies) == MCP_EXPECTED_REPLY_COUNT, mcp_roundtrip.outcome.stderr
    init_reply, tools_reply, _, _ = mcp_roundtrip.replies
    assert init_reply["result"]["protocolVersion"] == "2025-06-18"
    tools = tools_reply["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["fetch", "slice_pdf"]


def test_mcp_null_id_ping(mcp_roundtrip: McpRoundtrip) -> None:
    """Verify null-id JSON-RPC ping returns an empty object with preserved null id."""
    assert len(mcp_roundtrip.replies) == MCP_EXPECTED_REPLY_COUNT, mcp_roundtrip.outcome.stderr
    _, _, ping_reply, _ = mcp_roundtrip.replies
    assert ping_reply["id"] is None
    assert ping_reply["result"] == {}


def test_mcp_batch_rejection(mcp_roundtrip: McpRoundtrip) -> None:
    """Verify MCP 2025-06-18 rejects JSON-RPC batch arrays with `-32600`."""
    assert len(mcp_roundtrip.replies) == MCP_EXPECTED_REPLY_COUNT, mcp_roundtrip.outcome.stderr
    _, _, _, batch_reply = mcp_roundtrip.replies
    assert batch_reply["error"]["code"] == -32600


@pytest.mark.skipif(sys.platform == "win32", reason="Unix SIGINT delivery over live stdio")
def test_mcp_live_sigint_exits_130_without_stderr(installed_venv: Venv) -> None:
    """Verify live `SIGINT` interrupts `adfontes mcp` cleanly with exit `130` and no traceback."""
    result = installed_venv.run_mcp_until_init_then_interrupt(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        }
    )
    assert result.init_reply["result"]["protocolVersion"] == "2025-06-18"
    assert result.outcome.exit_code == SIGINT_EXIT_CODE
    assert result.outcome.stderr == ""


@pytest.mark.parametrize("case", CLOSED_PIPE_CASES, ids=lambda case: "_".join(case.args))
def test_cli_closed_stdout_epipe(installed_venv: Venv, case: CliCase) -> None:
    """Verify CLI invocations exit cleanly without stderr noise on closed stdout pipes."""
    outcome = installed_venv.run_closed_stdout(
        [str(installed_venv.executable("adfontes")), *case.args]
    )
    assert outcome.exit_code == case.exit_code
    assert outcome.stdout == ""
    assert outcome.stderr == ""


def test_mcp_closed_stdout_epipe(installed_venv: Venv) -> None:
    """Verify `adfontes mcp` exits cleanly without stderr noise on a closed stdout pipe."""
    outcome = installed_venv.run_closed_stdout(
        [str(installed_venv.executable("adfontes")), "mcp"],
        stdin=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n",
    )
    assert outcome.exit_code == 0
    assert outcome.stdout == ""
    assert outcome.stderr == ""


def test_sdist_locked_build_and_version(sdist_venv: Venv) -> None:
    """Verify the source distribution builds cleanly with `--locked` and runs `--version`."""
    outcome = sdist_venv.run([str(sdist_venv.executable("adfontes")), "--version"])
    assert outcome.exit_code == 0, outcome.stderr
    assert outcome.stdout.startswith("adfontes ")
