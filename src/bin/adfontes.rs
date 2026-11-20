//! Standalone `adfontes` command-line and stdio MCP binary.

adfontes::architecture_component!(Bin);

use std::process::ExitCode;

fn main() -> ExitCode {
    ExitCode::from(adfontes::cli::run_from_args(std::env::args_os()))
}
