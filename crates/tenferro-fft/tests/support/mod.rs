use tenferro_cpu::CpuEngine;
use tenferro_runtime::{CompiledGraph, ExecutionEngine, Runtime, RuntimeConfigError};
use tenferro_tensor::{Tensor, TensorRead};

/// Build a CPU runtime with the FFT extension module installed for `engine`'s kind.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] when CPU engine registration, FFT module
/// construction, module installation, or final runtime construction fails.
pub fn cpu_runtime_with_fft(engine: &CpuEngine) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(engine)?)?;
    builder.install_extension_module(tenferro_fft::extension_module::<CpuEngine>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    builder.build()
}

/// Prepare and run `program` on `engine`, returning owned outputs.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::RuntimeState`] when the target is not
/// registered or the prepared run belongs to another runtime,
/// [`tenferro_runtime::Error::Unsupported`] when the FFT provider declines an
/// operation, [`tenferro_runtime::Error::TensorRuntime`] when the engine fails
/// or an output cannot be copied, and [`tenferro_runtime::Error::Extension`]
/// for an FFT kernel's typed failure.
pub fn run_on_engine(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let reads: Vec<TensorRead<'_>> = inputs.iter().map(|t| TensorRead::from_tensor(t)).collect();
    let prepared = runtime.prepare(program, &reads, &[engine.target()])?;
    let outputs = runtime.run(&mut [engine as &mut dyn ExecutionEngine], &prepared, &reads)?;
    outputs
        .into_iter()
        .map(|output| output.into_tensor(&mut *engine))
        .collect::<tenferro_tensor::Result<Vec<_>>>()
        .map_err(Into::into)
}

/// Execute a compiled FFT test program on a serial CPU engine and return its single output.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::RuntimeState`] when constructing the test
/// runtime fails or when the compiled program returns any output count other
/// than one. Propagates typed input binding, validation, and backend execution
/// errors from [`run_on_engine`].
pub fn run_one(program: &CompiledGraph, inputs: &[&Tensor]) -> tenferro_runtime::Result<Tensor> {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime_with_fft(&engine).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "fft_test_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            source,
        )
    })?;
    let mut outputs = run_on_engine(&runtime, &mut engine, program, inputs)?;
    let actual = outputs.len();
    if actual != 1 {
        return Err(tenferro_runtime::Error::runtime_state(
            "fft_test_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            format!("expected one runtime output, got {actual}"),
        ));
    }
    outputs.pop().ok_or_else(|| {
        tenferro_runtime::Error::runtime_state(
            "fft_test_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            "runtime returned no output after successful output-count validation",
        )
    })
}
