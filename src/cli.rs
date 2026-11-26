//! Command-line interface parsing, startup environment binding, and EPIPE-safe output dispatch.

architecture_component!(Cli);

use std::ffi::OsString;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Duration;

use clap::error::ErrorKind as ClapErrorKind;
use clap::{Parser, Subcommand};

use crate::mcp::serve_io_until;
use crate::model::{AdfontesError, ExtractedDocument, PageSpan, ParsedTarget};
use crate::resolve::{ResolverConfig, resolve_target};
use crate::stage::{StagingPolicy, slice_staged_pdf, stage_document, validate_staging_dir_policy};
use crate::transport::{HttpClient, TransportConfig};

const INTERRUPT_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Exit code for successful invocations and clean `BrokenPipe` (`EPIPE`) terminations.
pub(crate) const EXIT_OK: u8 = 0;
/// Exit code for operational retrieval, extraction, safety, or filesystem errors.
pub(crate) const EXIT_ERROR: u8 = 1;
/// Exit code for CLI usage or argument parsing errors.
pub(crate) const EXIT_USAGE: u8 = 2;

/// Process-level configuration parsed once at CLI startup.
#[derive(Debug, Clone)]
pub(crate) struct CliEnv {
    /// SSRF-validated HTTP transport settings (including environment proxies).
    pub(crate) transport_config: TransportConfig,
    /// Upstream repository endpoints and optional `ADFONTES_CONTACT_EMAIL`.
    pub(crate) resolver_config: ResolverConfig,
    /// Confined staging policy (`ADFONTES_STAGING_DIR`, `ADFONTES_DISABLE_PDF_SLICE`, UID).
    pub(crate) staging_policy: StagingPolicy,
}

impl CliEnv {
    /// Reads all supported process environment variables once at startup.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            transport_config: TransportConfig::from_env(),
            resolver_config: ResolverConfig::from_env(),
            staging_policy: StagingPolicy::from_env(),
        }
    }
}

/// Parses `args` and executes `adfontes` against process `stdin`/`stdout`/`stderr`.
///
/// Returns exit code `0` on success (including `BrokenPipe` on stdout), `1` on operational or
/// safety errors, and `2` on argument parsing failures.
#[must_use]
pub fn run_from_args<I, T>(args: I) -> u8
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    run_from_args_with_interrupt(args, || false)
}

/// Parses `args` and executes `adfontes` against process stdio while polling `should_stop` on
/// interrupted (`EINTR`) MCP stdin reads and during async network I/O.
///
/// Used by language bindings (such as `adfontes-py`) so pending host signals (`SIGINT`) terminate
/// `adfontes mcp` and `adfontes fetch` cleanly without printing spurious stderr errors while benign
/// signals are retried.
#[must_use]
pub fn run_from_args_with_interrupt<I, T, F>(args: I, should_stop: F) -> u8
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
    F: FnMut() -> bool,
{
    let env = CliEnv::from_env();
    // Pass unlocked `io::stdout()` / `io::stderr()` so background DNS threads or panic hooks are
    // never blocked across the lifetime of `fetch` or `mcp`.
    let stdin = io::stdin();
    run_from_args_with_io(
        args,
        &env,
        stdin.lock(),
        io::stdout(),
        io::stderr(),
        should_stop,
    )
}

/// Parses `args` and executes `adfontes` with injected environment configuration, stdio streams, and
/// an interrupt predicate polled on `EINTR` stdin reads and async network waits.
#[must_use]
pub(crate) fn run_from_args_with_io<I, T, R, W, E, F>(
    args: I,
    env: &CliEnv,
    stdin: R,
    mut stdout: W,
    mut stderr: E,
    mut should_stop: F,
) -> u8
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
    R: BufRead,
    W: Write,
    E: Write,
    F: FnMut() -> bool,
{
    let parsed = match CliArgs::try_parse_from(args) {
        Ok(cli) => cli,
        Err(clap_error) => {
            let rendered = clap_error.to_string();
            return match clap_error.kind() {
                ClapErrorKind::DisplayHelp | ClapErrorKind::DisplayVersion => {
                    match write_text_epipe_safe(&mut stdout, &rendered) {
                        Ok(()) => EXIT_OK,
                        Err(error) => {
                            let message = format!("adfontes: {error}\n");
                            write_stderr_best_effort(&mut stderr, &message);
                            EXIT_ERROR
                        }
                    }
                }
                _ => {
                    write_stderr_best_effort(&mut stderr, &rendered);
                    EXIT_USAGE
                }
            };
        }
    };

    // Latch one-shot host predicates (such as CPython `check_signals`) so subsequent stop checks
    // continue observing the signal without emitting spurious stderr errors on clean shutdown.
    let stopped = std::cell::Cell::new(false);
    let outcome = dispatch_command(parsed.command, env, stdin, &mut stdout, || {
        if !stopped.get() && should_stop() {
            stopped.set(true);
        }
        stopped.get()
    });
    match outcome {
        Ok(()) => EXIT_OK,
        Err(_) if stopped.get() => EXIT_OK,
        Err(error) => {
            let message = format!("adfontes: {error}\n");
            write_stderr_best_effort(&mut stderr, &message);
            EXIT_ERROR
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "adfontes",
    bin_name = "adfontes",
    version,
    about = "Full-text academic and technical literature reader CLI and stdio MCP server"
)]
struct CliArgs {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Retrieve full text or PDF via keyless open-access waterfalls and stage locally.
    Fetch {
        /// arXiv ID/URL, DOI, PMCID, PMID, IETF RFC, or http(s):// URL.
        target: String,
        /// Confined output directory (temp, `ADFONTES_EXTRA_ALLOWED_ROOTS`, or `ADFONTES_STAGING_DIR`).
        #[arg(short = 'o', long = "output-dir")]
        output_dir: Option<PathBuf>,
        /// Skip writing companion `<slug>.pages.md` for binary PDFs.
        #[arg(long = "no-pdf-index")]
        no_pdf_index: bool,
    },
    /// Extract a lossless 1..=8-page vector PDF slice from a confined staged PDF.
    SlicePdf {
        /// Path to a confined staged PDF file.
        pdf_path: PathBuf,
        /// 1-indexed page or inclusive range (`N` or `START-END`, max 8 pages).
        #[arg(short = 'p', long = "pages")]
        pages: String,
        /// Confined output directory (defaults to `pdf_path`'s parent directory).
        #[arg(short = 'o', long = "output-dir")]
        output_dir: Option<PathBuf>,
    },
    /// Run the newline-delimited JSON-RPC 2.0 stdio MCP server.
    Mcp,
}

fn dispatch_command<R, W, F>(
    command: CliCommand,
    env: &CliEnv,
    stdin: R,
    stdout: &mut W,
    mut should_stop: F,
) -> Result<(), AdfontesError>
where
    R: BufRead,
    W: Write,
    F: FnMut() -> bool,
{
    match command {
        CliCommand::Fetch {
            target,
            output_dir,
            no_pdf_index,
        } => {
            let parsed_target = ParsedTarget::parse(&target)?;
            validate_staging_dir_policy(output_dir.as_deref(), &env.staging_policy)?;
            let runtime = build_async_runtime()?;
            let client = HttpClient::new(env.transport_config.clone());
            let outcome = block_on_interruptible(
                &runtime,
                &mut should_stop,
                resolve_target(&parsed_target, &client, &env.resolver_config),
            );
            drop(client);
            runtime.shutdown_background();
            let extracted = outcome?;
            let staged = stage_document(
                &parsed_target,
                extracted,
                output_dir.as_deref(),
                !no_pdf_index,
                &env.staging_policy,
            )?;
            write_output_line_epipe_safe(stdout, &staged.compact_index)
        }
        CliCommand::SlicePdf {
            pdf_path,
            pages,
            output_dir,
        } => {
            let span = PageSpan::parse_spec(&pages)?;
            let slice =
                slice_staged_pdf(&pdf_path, span, output_dir.as_deref(), &env.staging_policy)?;
            write_output_line_epipe_safe(stdout, &slice.summary)
        }
        CliCommand::Mcp => {
            let runtime = build_async_runtime()?;
            let client = HttpClient::new(env.transport_config.clone());
            let outcome = {
                let resolver_config = &env.resolver_config;
                let mut fetcher = |target: &ParsedTarget,
                                   stop: &mut dyn FnMut() -> bool|
                 -> Result<ExtractedDocument, AdfontesError> {
                    block_on_interruptible(
                        &runtime,
                        stop,
                        resolve_target(target, &client, resolver_config),
                    )
                };
                serve_io_until(
                    stdin,
                    stdout,
                    &env.staging_policy,
                    &mut fetcher,
                    should_stop,
                )
            };
            drop(client);
            runtime.shutdown_background();
            outcome
        }
    }
}

fn build_async_runtime() -> Result<tokio::runtime::Runtime, AdfontesError> {
    tokio::runtime::Builder::new_current_thread()
        .enable_io()
        .enable_time()
        .build()
        .map_err(|source| AdfontesError::Io {
            context: "initializing async HTTP runtime".to_owned(),
            source,
        })
}

fn block_on_interruptible<F, Fut, T>(
    runtime: &tokio::runtime::Runtime,
    should_stop: &mut F,
    future: Fut,
) -> Result<T, AdfontesError>
where
    F: FnMut() -> bool + ?Sized,
    Fut: Future<Output = Result<T, AdfontesError>>,
{
    if should_stop() {
        return Err(interrupted_error());
    }
    runtime.block_on(async move {
        let mut pinned = std::pin::pin!(future);
        loop {
            tokio::select! {
                biased;
                outcome = &mut pinned => return outcome,
                () = tokio::time::sleep(INTERRUPT_POLL_INTERVAL) => {
                    if should_stop() {
                        return Err(interrupted_error());
                    }
                }
            }
        }
    })
}

fn interrupted_error() -> AdfontesError {
    AdfontesError::Io {
        context: "interrupted during network I/O".to_owned(),
        source: io::Error::new(io::ErrorKind::Interrupted, "operation interrupted"),
    }
}

fn write_output_line_epipe_safe<W: Write>(writer: &mut W, text: &str) -> Result<(), AdfontesError> {
    if text.ends_with('\n') {
        write_text_epipe_safe(writer, text)
    } else {
        write_text_epipe_safe(writer, &format!("{text}\n"))
    }
}

fn write_text_epipe_safe<W: Write>(writer: &mut W, text: &str) -> Result<(), AdfontesError> {
    match writer
        .write_all(text.as_bytes())
        .and_then(|()| writer.flush())
    {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(source) => Err(AdfontesError::Io {
            context: "writing command output".to_owned(),
            source,
        }),
    }
}

fn write_stderr_best_effort<W: Write>(writer: &mut W, text: &str) {
    let _ = writer
        .write_all(text.as_bytes())
        .and_then(|()| writer.flush());
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::io::Cursor;
    use std::net::TcpListener;
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::thread;

    use indoc::indoc;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::stage::TestSandbox;

    struct FailingWriter {
        kind: io::ErrorKind,
    }

    impl Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(self.kind, "synthetic write failure"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn test_env(sandbox: &Path) -> CliEnv {
        CliEnv {
            transport_config: TransportConfig::default(),
            resolver_config: ResolverConfig::default(),
            staging_policy: StagingPolicy::for_sandbox(sandbox),
        }
    }

    #[test]
    fn prints_version_cleanly() {
        let sandbox = TestSandbox::new();
        let env = test_env(sandbox.path());
        let mut version_stdout = Vec::new();
        let mut version_stderr = Vec::new();

        let code = run_from_args_with_io(
            ["adfontes", "--version"],
            &env,
            Cursor::new(b""),
            &mut version_stdout,
            &mut version_stderr,
            || false,
        );

        assert_eq!(code, EXIT_OK);
        assert!(
            String::from_utf8(version_stdout)
                .unwrap()
                .starts_with("adfontes ")
        );
        assert_eq!(version_stderr, b"");
    }

    #[test]
    fn uses_canonical_adfontes_bin_name_for_windows_argv0() {
        let sandbox = TestSandbox::new();
        let env = test_env(sandbox.path());
        let mut win_help = Vec::new();

        let win_code = run_from_args_with_io(
            [r"C:\bin\adfontes.exe", "--help"],
            &env,
            Cursor::new(b""),
            &mut win_help,
            &mut Vec::new(),
            || false,
        );

        assert_eq!(win_code, EXIT_OK);
        assert!(
            String::from_utf8(win_help)
                .unwrap()
                .contains("Usage: adfontes <COMMAND>")
        );
    }

    #[test]
    fn suppresses_broken_pipe_and_reports_non_pipe_stdout_errors() {
        let sandbox = TestSandbox::new();
        let env = test_env(sandbox.path());
        let mut io_stderr = Vec::new();

        let pipe_code = run_from_args_with_io(
            ["adfontes", "--help"],
            &env,
            Cursor::new(b""),
            FailingWriter {
                kind: io::ErrorKind::BrokenPipe,
            },
            &mut Vec::new(),
            || false,
        );
        let io_code = run_from_args_with_io(
            ["adfontes", "--help"],
            &env,
            Cursor::new(b""),
            FailingWriter {
                kind: io::ErrorKind::WriteZero,
            },
            &mut io_stderr,
            || false,
        );

        assert_eq!(pipe_code, EXIT_OK);
        assert_eq!(io_code, EXIT_ERROR);
        assert!(String::from_utf8(io_stderr).unwrap().contains("adfontes:"));
    }

    #[test]
    fn returns_exit_2_on_usage_error_and_exit_1_on_invalid_target() {
        let sandbox = TestSandbox::new();
        let env = test_env(sandbox.path());
        let mut usage_stderr = Vec::new();
        let mut target_stderr = Vec::new();

        let usage_code = run_from_args_with_io(
            ["adfontes", "unknown-cmd"],
            &env,
            Cursor::new(b""),
            &mut Vec::new(),
            &mut usage_stderr,
            || false,
        );
        let target_code = run_from_args_with_io(
            ["adfontes", "fetch", "ftp://example.org/file"],
            &env,
            Cursor::new(b""),
            &mut Vec::new(),
            &mut target_stderr,
            || false,
        );

        assert_eq!(usage_code, EXIT_USAGE);
        assert_eq!(target_code, EXIT_ERROR);
        assert!(
            String::from_utf8(target_stderr)
                .unwrap()
                .contains("adfontes:")
        );
    }

    #[test]
    fn dispatches_fetch_and_blocks_forbidden_output_dir_before_network() {
        let sandbox = TestSandbox::new();
        let body = indoc! {"
            # CLI Article

            ## Details

            Full markdown body for end-to-end CLI verification with sufficient prose length to pass extraction cleanly.
        "};
        let runtime = build_async_runtime().unwrap();
        let server = runtime.block_on(async {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/article.md"))
                .respond_with(
                    ResponseTemplate::new(200).set_body_raw(body, "text/markdown; charset=utf-8"),
                )
                .expect(1)
                .mount(&server)
                .await;
            server
        });
        let addr = *server.address();

        let mut allowed_loopback_addrs = BTreeSet::new();
        allowed_loopback_addrs.insert(addr);
        let env = CliEnv {
            transport_config: TransportConfig {
                allowed_loopback_addrs,
                ..TransportConfig::default()
            },
            resolver_config: ResolverConfig::default(),
            staging_policy: StagingPolicy::for_sandbox(sandbox.path()),
        };
        let url = format!("{}/article.md", server.uri());
        let fetch_dir = sandbox.path().join("fetched");
        let mut fetch_stdout = Vec::new();

        let blocked_dir = sandbox.path().join("home/.ssh/blocked");
        let blocked_code = run_from_args_with_io(
            [
                "adfontes",
                "fetch",
                &url,
                "-o",
                &blocked_dir.display().to_string(),
            ],
            &env,
            Cursor::new(b""),
            &mut Vec::new(),
            &mut Vec::new(),
            || false,
        );
        let fetch_code = run_from_args_with_io(
            [
                "adfontes",
                "fetch",
                &url,
                "-o",
                &fetch_dir.display().to_string(),
                "--no-pdf-index",
            ],
            &env,
            Cursor::new(b""),
            &mut fetch_stdout,
            &mut Vec::new(),
            || false,
        );
        let recorded = runtime.block_on(server.received_requests()).unwrap();
        drop(server);

        assert_eq!(blocked_code, EXIT_ERROR);
        assert_eq!(fetch_code, EXIT_OK);
        assert!(
            String::from_utf8(fetch_stdout)
                .unwrap()
                .contains("CLI Article")
        );
        assert_eq!(recorded.len(), 1);
    }

    #[test]
    fn dispatches_slice_pdf_using_injected_io() {
        let sandbox = TestSandbox::new();
        let env = test_env(sandbox.path());
        let pdf_dir = sandbox.path().join("pdfs");
        let pdf_path = env
            .staging_policy
            .write_test_pdf(&pdf_dir, "doc.pdf", &["One", "Two", "Three"])
            .unwrap();
        let slice_out = sandbox.path().join("slices");
        let mut slice_stdout = Vec::new();

        let slice_code = run_from_args_with_io(
            [
                "adfontes",
                "slice-pdf",
                &pdf_path.display().to_string(),
                "-p",
                "2-3",
                "-o",
                &slice_out.display().to_string(),
            ],
            &env,
            Cursor::new(b""),
            &mut slice_stdout,
            &mut Vec::new(),
            || false,
        );

        assert_eq!(slice_code, EXIT_OK);
        assert!(
            String::from_utf8(slice_stdout)
                .unwrap()
                .contains("PAGES: 2-3 (of 3)")
        );
    }

    #[test]
    fn dispatches_mcp_using_injected_io() {
        let sandbox = TestSandbox::new();
        let env = test_env(sandbox.path());
        let mut mcp_stdout = Vec::new();

        let mcp_code = run_from_args_with_io(
            ["adfontes", "mcp"],
            &env,
            Cursor::new(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n"),
            &mut mcp_stdout,
            &mut Vec::new(),
            || false,
        );

        assert_eq!(mcp_code, EXIT_OK);
        assert!(
            String::from_utf8(mcp_stdout)
                .unwrap()
                .contains("\"result\":{}")
        );
    }

    #[test]
    fn cancels_stalled_fetch_within_budget_when_one_shot_stop_trips() {
        let sandbox = TestSandbox::new();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let accepted_at = Arc::new(std::sync::Mutex::new(None::<std::time::Instant>));
        let server_accepted_at = Arc::clone(&accepted_at);
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let server = thread::spawn(move || {
            if let Ok((_stream, _)) = listener.accept() {
                *server_accepted_at.lock().unwrap() = Some(std::time::Instant::now());
                let _ = release_rx.recv_timeout(Duration::from_secs(2));
            }
        });

        let mut allowed_loopback_addrs = BTreeSet::new();
        allowed_loopback_addrs.insert(addr);
        let env = CliEnv {
            transport_config: TransportConfig {
                timeout: Duration::from_secs(2),
                allowed_loopback_addrs,
                ..TransportConfig::default()
            },
            resolver_config: ResolverConfig::default(),
            staging_policy: StagingPolicy::for_sandbox(sandbox.path()),
        };
        let url = format!("http://{addr}/stalled.md");
        let fired = std::sync::atomic::AtomicBool::new(false);
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();

        let exit_code = run_from_args_with_io(
            ["adfontes", "fetch", &url],
            &env,
            Cursor::new(b""),
            &mut stdout,
            &mut stderr,
            || {
                let server_ready = accepted_at.lock().unwrap().is_some();
                server_ready
                    && !fired.load(Ordering::Relaxed)
                    && fired
                        .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                        .is_ok()
            },
        );
        let cancel_elapsed = accepted_at
            .lock()
            .unwrap()
            .expect("stalled test server must have accepted the client connection")
            .elapsed();
        let _ = release_tx.send(());
        server.join().unwrap();

        assert_eq!(exit_code, EXIT_OK);
        assert_eq!(stderr, b"");
        assert_eq!(stdout, b"");
        assert!(
            cancel_elapsed < Duration::from_millis(750),
            "cancellation took {cancel_elapsed:?}, expected < 750ms"
        );
    }
}
