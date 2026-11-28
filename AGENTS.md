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
- `uv run --no-project --with 'pytest>=8.3,<10' pytest -q -p no:cacheprovider tests/wheel --wheel-dir <wheels> --sdist-dir <sdist> --reference-binary target/release/adfontes --python 3.10 --python 3.13`

## Architecture & Component Boundaries

`adfontes` enforces compile-time acyclicity across the declared component catalog in `src/architecture.rs` and checks root component declarations, top-level module parity, the narrowed `src/lib.rs` public surface (`architecture`, `cli`, and `architecture_component!`), and the `crates/adfontes-py` -> `adfontes::cli` boundary in `tests/architecture_conformance.rs`; intra-crate component edges and external crate ownership document the architectural contract checked during review via `omnicoach graph`:

- `Architecture` owns the component DAG catalog and compile-time acyclicity checks (`strum`).
- `Model` owns validated external identifiers and domain sum types (`thiserror`, `url`, `percent-encoding`).
- `Archive` owns bounded decompression and `tar` unpacking (`flate2`, `tar`).
- `Transport` owns SSRF-pinned asynchronous HTTP(S) requests with cached keep-alive/TLS connection pools and `SsrfDnsResolver`, proxy/CIDR matching, MIME parsing, and charset decoding (`reqwest`, `rustls` with `ring`, `rustls-platform-verifier`, `ipnet`, `mime`, `encoding_rs`).
- `Pdf` owns pure-Rust PDF indexing and lossless vector page slicing (`lopdf`).
- `Extract` owns math-preserving HTML5, XML, RFC, TeX, and CommonMark TOC extractors (`cssparser`, `cssparser-color`, `scraper`, `roxmltree`, `pulldown-cmark`).
- `Resolve` (`src/resolve.rs`, `src/resolve/content.rs`, `src/resolve/providers.rs`, and `src/resolve/session.rs`) is strictly filesystem-free (no `std::fs` writes or staging) and drives asynchronous scholarly retrieval waterfalls.
- `Stage` (`src/stage.rs`, `src/stage/manifest.rs`, `src/stage/sandbox.rs`) is strictly network-free (no `Transport` or `Resolve` imports) and owns `sha2`, `tempfile`, `rustix`, and `dunce`.
- `Mcp` (`Mcp => [Stage, Model]`) is strictly synchronous and network-free, owns `serde`, and exposes newline-delimited JSON-RPC `2.0` stdio tools (`fetch`, `slice_pdf`) over an injected synchronous fetcher callback.
- `Cli` owns `clap`, startup environment binding (`CliEnv::from_env()`), and the single-threaded `tokio` runtime bridging `fetch` and `mcp` to async `Resolve`/`Transport`.
- `Bin` (`src/bin/adfontes.rs`) delegates to `adfontes::cli::run_from_args`, and `crates/adfontes-py` delegates to `adfontes::cli::run_from_args_with_interrupt` so Python `SIGINT`/`KeyboardInterrupt` interrupts blocked MCP stdio reads and async `fetch` network I/O cleanly.

Never bypass `owns_external` encapsulation in production code or unit tests (for example, `Stage` tests call `crate::pdf::build_test_pdf` rather than importing `lopdf`). `tokio` and `serde_json` are shared async runtime and JSON crates across the workspace rather than single-component dependencies.

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
