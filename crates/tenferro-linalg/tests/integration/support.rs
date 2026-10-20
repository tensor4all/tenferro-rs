use tenferro_cpu::CpuEngine;
use tenferro_runtime::ExecutionEngine;
use tenferro_runtime::{CompiledGraph, Runtime, RuntimeConfigError};
use tenferro_tensor::{Tensor, TensorRead};

/// Run a test operation on a CPU engine.
pub fn with_cpu_linalg<R>(
    backend: &mut CpuEngine,
    f: impl for<'a> FnOnce(&'a mut CpuEngine) -> R + Send,
) -> R
where
    R: Send,
{
    f(backend)
}

/// Build a CPU runtime with the linalg extension module installed.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] when CPU engine registration, linalg module
/// construction, module installation, or final runtime construction fails.
pub fn cpu_runtime_with_linalg(backend: &CpuEngine) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(backend)?)?;
    builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
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
/// [`Runtime::run`].
pub fn run_all(
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let mut engine = tenferro_cpu::CpuEngine::serial();
    let runtime = cpu_runtime_with_linalg(&engine).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "linalg_test_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            source,
        )
    })?;
    run_program(&runtime, &mut engine, program, inputs)
}

/// Prepare `program` for `engine` and run it, returning owned outputs.
///
/// # Errors
///
/// Propagates preparation, input validation and execution errors.
pub fn run_program(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let reads: Vec<TensorRead<'_>> = inputs
        .iter()
        .map(|tensor| TensorRead::from_tensor(tensor))
        .collect();
    let prepared = runtime.prepare(program, &reads, &[engine.target()])?;
    let outputs = {
        let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut *engine];
        runtime.run(&mut engines, &prepared, &reads)?
    };
    outputs
        .into_iter()
        .map(|output| Ok(output.into_tensor(&mut *engine)?))
        .collect()
}
