//! `CPython` `abi3-py310` extension entrypoint delegating CLI and stdio MCP execution to
//! [`adfontes::cli`].

use std::ffi::OsString;

use pyo3::prelude::*;

/// Registers the internal `_adfontes` native extension module.
#[pymodule(name = "_adfontes")]
fn adfontes_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_cli, module)?)?;
    Ok(())
}

/// Executes the `adfontes` CLI or stdio MCP server from `args`, releasing the Python GIL during
/// execution while checking pending Python signals (such as `SIGINT` / `KeyboardInterrupt`) on
/// interrupted stdio reads and on return.
#[pyfunction]
fn run_cli(py: Python<'_>, args: Vec<OsString>) -> PyResult<u8> {
    let mut pending_signal_error: Option<PyErr> = None;
    let exit_code = py.detach(|| {
        adfontes::cli::run_from_args_with_interrupt(args, || {
            Python::attach(|attached_py| match attached_py.check_signals() {
                Ok(()) => false,
                Err(error) => {
                    pending_signal_error = Some(error);
                    true
                }
            })
        })
    });
    if let Some(error) = pending_signal_error {
        return Err(error);
    }
    py.check_signals()?;
    Ok(exit_code)
}
