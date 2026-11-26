"""Pytest fixtures and isolated virtual environment helpers for wheel verification."""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

TIMEOUT = datetime.timedelta(minutes=2)
INTERRUPT_TIMEOUT = datetime.timedelta(seconds=10)
SDIST_BUILD_TIMEOUT = datetime.timedelta(minutes=10)


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """Exit status and captured streams of one process."""

    exit_code: int
    stdout: str
    stderr: str


@dataclasses.dataclass(frozen=True, slots=True)
class McpRoundtrip:
    """Captured process outcome and decoded JSON-RPC replies from one MCP session."""

    outcome: Outcome
    replies: tuple[Mapping[str, Any], ...]


@dataclasses.dataclass(frozen=True, slots=True)
class McpInterruptOutcome:
    """Initial handshake reply and post-SIGINT process outcome from live `adfontes mcp`."""

    init_reply: Mapping[str, Any]
    outcome: Outcome


@dataclasses.dataclass(frozen=True, slots=True)
class SuiteConfig:
    """Resolved CLI configuration shared across the verification session."""

    wheel: pathlib.Path | None
    sdist: pathlib.Path | None
    interpreters: tuple[str, ...]
    reference_binary: pathlib.Path | None


@dataclasses.dataclass(frozen=True, slots=True)
class WheelArchive:
    """Extracted entry inventory and WHEEL metadata of the built wheel archive."""

    entries: frozenset[str]
    script_entries: tuple[str, ...]
    wheel_metadata: str


@dataclasses.dataclass(frozen=True, slots=True)
class NativeRunner:
    """Isolated runner bound to the standalone native `adfontes` reference binary."""

    binary: pathlib.Path
    root: pathlib.Path
    env: Mapping[str, str]

    def run(self, args: Sequence[str]) -> Outcome:
        """Run the standalone reference binary with `args` inside an isolated temp root."""
        return _run_command([str(self.binary), *args], cwd=self.root, env=self.env)


@dataclasses.dataclass(frozen=True, slots=True)
class Venv:
    """A fresh virtual environment isolated from user Python and adfontes configuration."""

    root: pathlib.Path
    env: Mapping[str, str]

    @classmethod
    def create(cls, root: pathlib.Path, interpreter: str) -> Venv:
        """Create an isolated virtual environment under `root`."""
        xdg_dir = root / "xdg"
        xdg_dir.mkdir()
        env = _isolated_env(xdg_dir)
        _check_command(
            ["uv", "venv", "--python", interpreter, str(root / "venv")],
            cwd=root,
            env=env,
        )
        return cls(root=root, env=env)

    def executable(self, stem: str) -> pathlib.Path:
        """Return the path of console script or interpreter `stem` inside the environment."""
        scripts = ("Scripts", f"{stem}.exe") if os.name == "nt" else ("bin", stem)
        return self.root.joinpath("venv", *scripts)

    def install(
        self, artifact: pathlib.Path, extra_env: Mapping[str, str] | None = None
    ) -> None:
        """Install a wheel or sdist into the environment with `uv pip`."""
        _check_command(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(self.executable("python")),
                str(artifact),
            ],
            cwd=self.root,
            env={**self.env, **(extra_env or {})},
            timeout=SDIST_BUILD_TIMEOUT,
        )

    def run(self, command: Sequence[str], *, stdin: str | None = None) -> Outcome:
        """Run `command` inside the virtual environment and capture its output."""
        return _run_command(command, cwd=self.root, env=self.env, stdin=stdin)

    def run_closed_stdout(
        self, command: Sequence[str], *, stdin: str | None = None
    ) -> Outcome:
        """Run `command` inside the environment with an already-closed stdout pipe."""
        return _run_with_closed_stdout(command, cwd=self.root, env=self.env, stdin=stdin)

    def run_mcp_until_init_then_interrupt(
        self, init_request: Mapping[str, Any]
    ) -> McpInterruptOutcome:
        """Spawn `adfontes mcp`, exchange `initialize`, send `SIGINT`, and collect status."""
        return _run_mcp_interrupt(
            [str(self.executable("adfontes")), "mcp"],
            cwd=self.root,
            env=self.env,
            init_request=init_request,
        )

    def check(self, command: Sequence[str]) -> None:
        """Run `command` inside the environment and fail loudly on non-zero exit."""
        _check_command(command, cwd=self.root, env=self.env)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register wheel, sdist, reference binary, and Python interpreter CLI flags."""
    group = parser.getgroup("adfontes wheel verification")
    group.addoption(
        "--wheel-dir",
        type=pathlib.Path,
        default=None,
        help="Optional directory containing the single built .whl artifact to verify.",
    )
    group.addoption(
        "--sdist-dir",
        type=pathlib.Path,
        default=None,
        help="Optional directory containing the single built .tar.gz source distribution.",
    )
    group.addoption(
        "--python",
        action="append",
        dest="python_interpreters",
        default=[],
        help="Interpreter passed to `uv venv --python` (repeatable; default: current Python).",
    )
    group.addoption(
        "--reference-binary",
        default=None,
        help="Standalone adfontes binary for output parity (`.exe` is resolved on Windows).",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize every interpreter-backed fixture over `--python` entries."""
    if "interpreter" in metafunc.fixturenames:
        interpreters = _resolve_interpreters(metafunc.config)
        metafunc.parametrize("interpreter", interpreters, scope="session")


@pytest.fixture(scope="session")
def suite_config(request: pytest.FixtureRequest) -> SuiteConfig:
    """Return resolved wheel, sdist, interpreter, and reference binary paths."""
    wheel_dir: pathlib.Path | None = request.config.getoption("--wheel-dir")
    sdist_dir: pathlib.Path | None = request.config.getoption("--sdist-dir")
    if wheel_dir is None and sdist_dir is None:
        raise pytest.UsageError("at least one of --wheel-dir or --sdist-dir is required")
    try:
        wheel = _single_match(wheel_dir, "*.whl") if wheel_dir is not None else None
        sdist = _single_match(sdist_dir, "*.tar.gz") if sdist_dir is not None else None
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error

    interpreters = _resolve_interpreters(request.config)

    reference_binary: pathlib.Path | None = None
    raw_reference: str | None = request.config.getoption("--reference-binary")
    if raw_reference is not None:
        resolved = shutil.which(raw_reference)
        if resolved is None:
            raise pytest.UsageError(f"reference binary not found: {raw_reference}")
        reference_binary = pathlib.Path(resolved).resolve()

    return SuiteConfig(
        wheel=wheel,
        sdist=sdist,
        interpreters=interpreters,
        reference_binary=reference_binary,
    )


@pytest.fixture(scope="session")
def wheel_artifact(suite_config: SuiteConfig) -> pathlib.Path:
    """Return the built wheel archive or skip wheel checks when absent."""
    if suite_config.wheel is None:
        pytest.skip("no --wheel-dir provided")
    return suite_config.wheel


@pytest.fixture(scope="session")
def wheel_archive(wheel_artifact: pathlib.Path) -> WheelArchive:
    """Return the built wheel archive's normalized entry inventory and WHEEL metadata."""
    with zipfile.ZipFile(wheel_artifact) as archive:
        entries = frozenset(archive.namelist())
        wheel_files = [entry for entry in entries if entry.endswith(".dist-info/WHEEL")]
        wheel_metadata = (
            archive.read(wheel_files[0]).decode("utf-8") if len(wheel_files) == 1 else ""
        )
    scripts = tuple(sorted(entry for entry in entries if ".data/scripts/" in entry))
    return WheelArchive(
        entries=entries,
        script_entries=scripts,
        wheel_metadata=wheel_metadata,
    )


@pytest.fixture(scope="session")
def reference_binary(suite_config: SuiteConfig) -> pathlib.Path:
    """Return the standalone reference binary or skip parity checks when absent."""
    if suite_config.reference_binary is None:
        pytest.skip("no --reference-binary provided")
    return suite_config.reference_binary


@pytest.fixture(scope="session")
def native_runner(
    reference_binary: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> NativeRunner:
    """Provide an isolated runner for the standalone native reference binary."""
    root = tmp_path_factory.mktemp("adfontes-native")
    xdg_dir = root / "xdg"
    xdg_dir.mkdir()
    return NativeRunner(binary=reference_binary, root=root, env=_isolated_env(xdg_dir))


@pytest.fixture(scope="session")
def sdist_artifact(suite_config: SuiteConfig) -> pathlib.Path:
    """Return the source distribution archive or skip sdist checks when absent."""
    if suite_config.sdist is None:
        pytest.skip("no --sdist-dir provided")
    return suite_config.sdist


@pytest.fixture(scope="session")
def installed_venv(
    interpreter: str,
    wheel_artifact: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Venv:
    """Create an isolated venv for `interpreter` and install the built wheel once."""
    root = tmp_path_factory.mktemp("adfontes-wheel")
    venv = Venv.create(root, interpreter)
    venv.install(wheel_artifact)
    return venv


@pytest.fixture(scope="session")
def sdist_venv(
    suite_config: SuiteConfig,
    sdist_artifact: pathlib.Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Venv:
    """Build and install the source distribution with `--locked` in an isolated venv."""
    root = tmp_path_factory.mktemp("adfontes-sdist")
    venv = Venv.create(root, suite_config.interpreters[0])
    venv.install(sdist_artifact, extra_env={"MATURIN_PEP517_ARGS": "--locked"})
    return venv


@pytest.fixture(scope="session")
def mcp_roundtrip(installed_venv: Venv) -> McpRoundtrip:
    """Run one initialize/tools-list/ping/batch session and return its outcome and replies."""
    frames = "\n".join(
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18"},
                }
            ),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            json.dumps({"jsonrpc": "2.0", "id": None, "method": "ping"}),
            json.dumps([{"jsonrpc": "2.0", "id": 3, "method": "ping"}]),
        ]
    ) + "\n"
    outcome = installed_venv.run(
        [str(installed_venv.executable("adfontes")), "mcp"],
        stdin=frames,
    )
    replies = (
        tuple(
            json.loads(line.rstrip("\r\n"))
            for line in outcome.stdout.splitlines()
            if line.rstrip("\r\n")
        )
        if outcome.exit_code == 0
        else ()
    )
    return McpRoundtrip(outcome=outcome, replies=replies)


def _resolve_interpreters(config: pytest.Config) -> tuple[str, ...]:
    """Return deduplicated interpreter specs from `--python` or the active executable."""
    raw_pythons = config.getoption("python_interpreters") or [sys.executable]
    return tuple(dict.fromkeys(raw_pythons))


def _isolated_env(xdg_dir: pathlib.Path) -> dict[str, str]:
    """Return a clean environment isolated from host Python and user config."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
    }
    env |= {"XDG_CONFIG_HOME": str(xdg_dir), "NO_COLOR": "1"}
    return env


def _single_match(directory: pathlib.Path, pattern: str) -> pathlib.Path:
    """Return the only file matching `pattern` in `directory`."""
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {pattern} in {directory}, found {matches}")
    return matches[0].resolve()


def _run_command(
    command: Sequence[str],
    *,
    cwd: pathlib.Path,
    env: Mapping[str, str],
    stdin: str | None = None,
) -> Outcome:
    """Run `command` to completion and capture its streams as text."""
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        input=stdin,
        stdin=None if stdin is not None else subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=TIMEOUT.total_seconds(),
        check=False,
    )
    return Outcome(completed.returncode, completed.stdout, completed.stderr)


def _run_with_closed_stdout(
    command: Sequence[str],
    *,
    cwd: pathlib.Path,
    env: Mapping[str, str],
    stdin: str | None = None,
) -> Outcome:
    """Run `command` with stdout connected to an already-closed pipe read end."""
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            input=stdin,
            stdin=None if stdin is not None else subprocess.DEVNULL,
            stdout=write_fd,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=TIMEOUT.total_seconds(),
            check=False,
        )
    finally:
        os.close(write_fd)
    return Outcome(completed.returncode, "", completed.stderr)


def _run_mcp_interrupt(
    command: Sequence[str],
    *,
    cwd: pathlib.Path,
    env: Mapping[str, str],
    init_request: Mapping[str, Any],
) -> McpInterruptOutcome:
    """Spawn `adfontes mcp`, exchange `initialize`, send `SIGINT` while stdin stays open."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        process.stdin.write(json.dumps(dict(init_request)) + "\n")
        process.stdin.flush()
        raw_reply = process.stdout.readline().rstrip("\r\n")
        init_reply = json.loads(raw_reply)
        process.send_signal(signal.SIGINT)
        exit_code = process.wait(timeout=INTERRUPT_TIMEOUT.total_seconds())
        stderr = process.stderr.read()
        return McpInterruptOutcome(
            init_reply=init_reply,
            outcome=Outcome(exit_code=exit_code, stdout=raw_reply, stderr=stderr),
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()


def _check_command(
    command: Sequence[str],
    *,
    cwd: pathlib.Path,
    env: Mapping[str, str],
    timeout: datetime.timedelta = TIMEOUT,
) -> None:
    """Run a setup command, raising with its output if it exits non-zero."""
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout.total_seconds(),
        check=False,
    )
    if completed.returncode != 0:
        output = f"{completed.stdout}\n{completed.stderr}"
        raise RuntimeError(f"Expected {command} to succeed, got: {output}")
