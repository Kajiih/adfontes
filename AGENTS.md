# Agent Guidelines

## Verification Scope

Run the narrowest verification checks covering your changes, and run the full gate before committing:

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets --locked -- -D warnings`
- `cargo test --workspace --locked`
- `RUSTDOCFLAGS=-D warnings cargo doc --workspace --no-deps --locked`
- `RUSTDOCFLAGS=-D warnings cargo doc --workspace --no-deps --document-private-items --locked`
- `omnicoach code`
- `cargo publish --dry-run --locked`

## Architecture & Component Boundaries

`adfontes` enforces a strict downward-pointing component Directed Acyclic Graph (**DAG**) via compile-time assertions in `src/architecture.rs` and bidirectional conformance checks in `tests/architecture_conformance.rs`:

- `Model` owns validated external identifiers and domain sum types (`url`, `thiserror`).
- `Archive` owns bounded decompression and `tar` unpacking (`flate2`, `tar`).
- `Transport` owns SSRF-pinned synchronous HTTP(S) requests and charset decoding (`ureq`, `encoding_rs`).
- `Pdf` owns pure-Rust PDF indexing and lossless vector page slicing (`lopdf`).
- `Extract` owns math-preserving HTML5, XML, RFC, TeX, and CommonMark TOC extractors (`scraper`, `roxmltree`).
- `Resolve` is strictly filesystem-free (no `std::fs` writes or staging).
- `Stage` is strictly network-free (no `Transport` or `Resolve` imports) and owns `sha2`, `tempfile`, and `rustix`.
- `Mcp` exposes newline-delimited JSON-RPC `2.0` stdio tools (`fetch`, `slice_pdf`).
- `Cli` owns `clap` and startup environment binding (`CliEnv::from_env()`).
- `Bin` (`src/bin/adfontes.rs`) delegates to `adfontes::cli::run_from_args`, and `crates/adfontes-py` delegates to `adfontes::cli::run_from_args_with_interrupt` so Python `SIGINT`/`KeyboardInterrupt` interrupts blocked MCP stdio reads cleanly.

Never bypass `owns_external` encapsulation in production code or unit tests (for example, `Stage` tests call `crate::pdf::build_test_pdf` rather than importing `lopdf`).

## Security & Packaging Invariants

- **Propagate `AdfontesError::SafetyBlock` fail-closed**: Never swallow SSRF, XML DTD, archive-bomb, symlink, or confinement safety blocks when falling back across upstream endpoints.
- **No post-link `strip` on macOS**: `[profile.release]` intentionally omits `strip = "symbols"` so Apple Silicon binaries and `cp310-abi3` wheels (`_adfontes.abi3.so`) keep their linker-generated ad-hoc `LC_CODE_SIGNATURE` intact (`codesign --verify --strict`). Only Linux release artifacts strip symbols explicitly in `.github/workflows/release.yml` (standalone binaries via `strip` and `manylinux_2_28` wheels via `maturin --strip`).
- **Single-line EPIPE-safe output**: Both CLI and MCP write via unlocked `io::stdout()`/`io::stderr()` handles (avoiding long-held global locks across network/stdin waits) and treat `BrokenPipe` (`EPIPE`) as exit code `0`.

## Structural Navigation

Use `omnicoach graph` before broad file exploration:

- `omnicoach graph <file>` for intra-file call hierarchy and helper clusters.
- `omnicoach graph <dir>` for component and module maps.
- `omnicoach graph find <pattern>` to locate symbols across the repository.
- `omnicoach graph slice <symbol>...` for cross-module callers, callees, and signatures.
