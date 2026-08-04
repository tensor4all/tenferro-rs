use tenferro_cpu::{with_cpu_exec_session, CpuBackend, CpuExecSession};
use tenferro_runtime::{CompiledGraph, Runtime, RuntimeConfigError};
use tenferro_tensor::{BackendSessionHost, Tensor};

/// Run a test operation through the provider-owned CPU execution session.
pub fn with_cpu_linalg<R>(
    backend: &mut CpuBackend,
    f: impl for<'a> FnOnce(&'a mut CpuExecSession<'a>) -> R + Send,
) -> R
where
    R: Send,
{
    backend.with_backend_session(|session| {
        with_cpu_exec_session(session, f).expect("CPU backend session should expose CpuExecSession")
    })
}

/// Build a CPU runtime with the linalg extension module installed.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] when CPU engine registration, linalg module
/// construction, module installation, or final runtime construction fails.
pub fn cpu_runtime_with_linalg(backend: &CpuBackend) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(backend)?)?;
    builder.install_extension_module(tenferro_linalg::extension_module::<CpuBackend>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    builder.build()
}

/// Execute a compiled linalg test program and return all outputs.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::RuntimeState`] when constructing the test
/// runtime fails. Propagates typed input binding, validation, and backend
/// execution errors from
/// [`Runtime::run_compiled`].
pub fn run_all(
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let backend = CpuBackend::new();
    let runtime = cpu_runtime_with_linalg(&backend).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "linalg_test_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            source,
        )
    })?;
    runtime.run_compiled(program, inputs)
}
