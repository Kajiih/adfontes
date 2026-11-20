#!/usr/bin/env python3
"""Verify the built adfontes cp310-abi3 wheel and optional source distribution.

Installs the wheel into a fresh virtual environment per requested Python interpreter and checks
the CLI and stdio MCP server end to end: exit codes and output for subcommands, `python -m adfontes`
parity, MCP JSON-RPC 2.0 handshake/batch-rejection round-trips, macOS ad-hoc code-signature
validity on `_adfontes.abi3.so`, closed stdout pipe (`EPIPE`) handling, and optional byte-for-byte
parity with a standalone binary.
"""

# omni:disable-file [repeated-literal] -- CLI arguments and JSON-RPC wire keys read clearest inline; constants would only restate them

import argparse
import dataclasses
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence

MAX_WHEEL_BYTES = 15 * 1024 * 1024
TIMEOUT = datetime.timedelta(minutes=2)
SDIST_BUILD_TIMEOUT = datetime.timedelta(minutes=10)
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
    """One CLI invocation and its expected exit status and stdout fragment."""

    args: Sequence[str]
    exit_code: int
    stdout_contains: str = ""


CLI_CASES = (
    CliCase(["--version"], 0, "adfontes "),
    CliCase(["--help"], 0, "Usage: adfontes"),
    CliCase(["fetch", "--help"], 0, "Usage: adfontes fetch"),
    CliCase(["slice-pdf", "--help"], 0, "Usage: adfontes slice-pdf"),
    CliCase(["mcp", "--help"], 0, "Usage: adfontes mcp"),
    CliCase(["fetch", "ftp://example.org/disallowed"], 1),
    CliCase(["unknown-subcommand"], 2),
)
MODULE_PARITY_ARGS = (["--version"], ["--help"])
CLOSED_PIPE_CASES = (
    CliCase(["--version"], 0),
    CliCase(["--help"], 0),
)


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """Exit status and captured streams of one process."""

    exit_code: int
    stdout: str
    stderr: str


@dataclasses.dataclass(frozen=True, slots=True)
class Venv:
    """A fresh virtual environment isolated from user Python and adfontes configuration."""

    root: pathlib.Path
    env: Mapping[str, str]

    @classmethod
    def create(cls, root: pathlib.Path, interpreter: str) -> "Venv":
        """Create an isolated virtual environment under `root`."""
        (root / "xdg").mkdir()
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}
        }
        env |= {"XDG_CONFIG_HOME": str(root / "xdg"), "NO_COLOR": "1"}
        _check(
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
        _check(
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


def main() -> int:
    """Run every verification check selected on the command line."""
    options = _parse_args()
    _check_archive(options.wheel)
    for interpreter in options.python:
        with tempfile.TemporaryDirectory(prefix="adfontes-wheel-") as temp_dir:
            _check_installed_wheel(
                root=pathlib.Path(temp_dir),
                interpreter=interpreter,
                wheel=options.wheel,
                reference_binary=options.reference_binary,
            )
    if options.sdist is not None:
        with tempfile.TemporaryDirectory(prefix="adfontes-sdist-") as temp_dir:
            _check_sdist(
                root=pathlib.Path(temp_dir),
                interpreter=options.python[0],
                sdist=options.sdist,
            )
    print(
        f"Verified {options.wheel.name} ({options.wheel.stat().st_size} bytes) "
        f"across {len(options.python)} Python interpreter(s)."
    )
    return 0


def _parse_args() -> argparse.Namespace:
    """Parse flags and resolve wheel, sdist, and reference binary paths."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel-dir", type=pathlib.Path, required=True)
    parser.add_argument("--sdist-dir", type=pathlib.Path)
    parser.add_argument(
        "--python",
        action="append",
        help="Interpreter passed to `uv venv --python` (repeatable; default: current Python).",
    )
    parser.add_argument(
        "--reference-binary",
        help="Standalone adfontes binary for output parity (`.exe` is resolved on Windows).",
    )
    args = parser.parse_args()
    args.wheel = _single_match(args.wheel_dir, "*.whl")
    args.sdist = _single_match(args.sdist_dir, "*.tar.gz") if args.sdist_dir else None
    args.python = args.python or [sys.executable]
    if args.reference_binary is not None:
        resolved = shutil.which(args.reference_binary)
        if resolved is None:
            parser.error(f"reference binary not found: {args.reference_binary}")
        args.reference_binary = pathlib.Path(resolved).resolve()
    return args


def _single_match(directory: pathlib.Path, pattern: str) -> pathlib.Path:
    """Return the only file matching `pattern` in `directory`."""
    matches = sorted(directory.glob(pattern))
    _expect(len(matches) == 1, f"exactly one {pattern} in {directory}, found {matches}")
    return matches[0].resolve()


def _check_archive(wheel: pathlib.Path) -> None:
    """Verify the wheel tag, size ceiling, abi3 extension, and zero bundled scripts."""
    _expect("-cp310-abi3-" in wheel.name, f"cp310-abi3 tag in {wheel.name}")
    _expect(
        wheel.stat().st_size <= MAX_WHEEL_BYTES,
        f"{wheel.name} under {MAX_WHEEL_BYTES} bytes",
    )
    with zipfile.ZipFile(wheel) as archive:
        entries = frozenset(archive.namelist())
    _expect(
        len(entries & EXTENSION_ENTRIES) == 1,
        f"one abi3 extension in {sorted(entries)}",
    )
    _expect(
        REQUIRED_ENTRIES <= entries, f"{sorted(REQUIRED_ENTRIES - entries)} missing"
    )
    scripts = [entry for entry in entries if ".data/scripts/" in entry]
    _expect(not scripts, f"no bundled executables, found {scripts}")


def _check_installed_wheel(
    *,
    root: pathlib.Path,
    interpreter: str,
    wheel: pathlib.Path,
    reference_binary: pathlib.Path | None,
) -> None:
    """Install `wheel` and verify CLI cases, parity, macOS codesign, MCP, and EPIPE.

    Args:
        root: Isolated temporary directory hosting the test virtual environment.
        interpreter: Python interpreter specifier passed to `uv venv --python`.
        wheel: Path to the built cp310-abi3 wheel archive under test.
        reference_binary: Standalone `adfontes` executable to compare against; if `None`,
            skips native binary parity checks. When provided, every CLI test case is also
            run against this binary and checked for exact exit-code, stdout, and stderr
            parity.
    """
    venv = Venv.create(root, interpreter)
    venv.install(wheel)
    if sys.platform == "darwin":
        _check_macos_codesign(venv)

    adfontes = [str(venv.executable("adfontes"))]
    for case in CLI_CASES:
        outcome = _run([*adfontes, *case.args], cwd=root, env=venv.env)
        _expect(
            outcome.exit_code == case.exit_code,
            f"exit {case.exit_code} for {case}: {outcome}",
        )
        _expect(case.stdout_contains in outcome.stdout, f"stdout of {case}: {outcome}")
        if reference_binary is not None:
            native = _run([str(reference_binary), *case.args], cwd=root, env=venv.env)
            _expect(
                outcome == native,
                f"parity with {reference_binary} for {case}: {native}",
            )

    module = [str(venv.executable("python")), "-m", "adfontes"]
    for args in MODULE_PARITY_ARGS:
        via_module = _run([*module, *args], cwd=root, env=venv.env)
        via_script = _run([*adfontes, *args], cwd=root, env=venv.env)
        _expect(via_module == via_script, f"python -m parity for {args}: {via_module}")

    _check_mcp_roundtrip(adfontes=adfontes, cwd=root, env=venv.env)
    for case in CLOSED_PIPE_CASES:
        outcome = _run_with_closed_stdout([*adfontes, *case.args], cwd=root, env=venv.env)
        _expect(
            outcome == Outcome(case.exit_code, "", ""),
            f"closed pipe for {case}: {outcome}",
        )
    mcp_pipe_outcome = _run_with_closed_stdout(
        [*adfontes, "mcp"],
        cwd=root,
        env=venv.env,
        stdin=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n",
    )
    _expect(
        mcp_pipe_outcome == Outcome(0, "", ""),
        f"closed pipe for mcp ping: {mcp_pipe_outcome}",
    )


def _check_macos_codesign(venv: Venv) -> None:
    """Verify the installed `_adfontes.abi3.so` retains a valid Mach-O code signature."""
    locate = _run(
        [
            str(venv.executable("python")),
            "-c",
            "import adfontes._adfontes as m; print(m.__file__)",
        ],
        cwd=venv.root,
        env=venv.env,
    )
    _expect(locate.exit_code == 0, f"locate `_adfontes` module: {locate}")
    extension_path = locate.stdout.strip()
    _check(
        ["codesign", "--verify", "--strict", extension_path],
        cwd=venv.root,
        env=venv.env,
    )


def _check_mcp_roundtrip(
    *, adfontes: Iterable[str], cwd: pathlib.Path, env: Mapping[str, str]
) -> None:
    """Verify MCP 2025-06-18 initialize, tools/list, null-id ping, and batch rejection."""
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
    outcome = _run([*adfontes, "mcp"], cwd=cwd, env=env, stdin=frames)
    _expect(outcome.exit_code == 0 and not outcome.stderr, f"clean MCP exit: {outcome}")
    replies = [json.loads(line) for line in outcome.stdout.splitlines() if line.strip()]
    _expect(len(replies) == 4, f"4 JSON-RPC responses (notification omitted): {replies}")
    _expect(
        replies[0]["result"]["protocolVersion"] == "2025-06-18",
        f"2025-06-18 negotiation: {replies[0]}",
    )
    tool_names = [tool["name"] for tool in replies[1]["result"]["tools"]]
    _expect(tool_names == ["fetch", "slice_pdf"], f"expected MCP tools: {tool_names}")
    _expect(replies[2]["id"] is None and replies[2]["result"] == {}, f"null-id ping: {replies[2]}")
    _expect(replies[3]["error"]["code"] == -32600, f"batch rejection -32600: {replies[3]}")


def _check_sdist(*, root: pathlib.Path, interpreter: str, sdist: pathlib.Path) -> None:
    """Build and install `sdist` with `--locked` and verify `--version`."""
    venv = Venv.create(root, interpreter)
    venv.install(sdist, extra_env={"MATURIN_PEP517_ARGS": "--locked"})
    outcome = _run([str(venv.executable("adfontes")), "--version"], cwd=root, env=venv.env)
    _expect(
        outcome.exit_code == 0 and outcome.stdout.startswith("adfontes "),
        f"sdist build runs adfontes: {outcome}",
    )


def _run(
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
    """Run `command` with stdout connected to an already-closed pipe read end.

    Args:
        command: Executable and arguments to run.
        cwd: Working directory for the child process.
        env: Environment variables for the child process.
        stdin: Text written to the child's stdin; if `None`, connects stdin to `DEVNULL`.
    """
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


def _check(
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
    output = f"{completed.stdout}\n{completed.stderr}"
    _expect(completed.returncode == 0, f"{command} to succeed, got: {output}")


def _expect(condition: bool, expectation: str) -> None:
    """Raise `RuntimeError` describing `expectation` when `condition` is false."""
    if not condition:
        raise RuntimeError(f"Expected {expectation}")


if __name__ == "__main__":
    sys.exit(main())
