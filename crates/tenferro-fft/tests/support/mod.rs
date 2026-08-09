use tenferro_cpu::CpuBackend;
use tenferro_runtime::{CompiledGraph, Runtime, RuntimeConfigError};
use tenferro_tensor::Tensor;

/// Build a CPU runtime with the FFT extension module installed.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] when CPU engine registration, FFT module
/// construction, module installation, or final runtime construction fails.
pub fn cpu_runtime_with_fft(backend: &CpuBackend) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(backend)?)?;
    builder.install_extension_module(tenferro_fft::extension_module::<CpuBackend>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    builder.build()
}

/// Execute a compiled FFT test program and return its single output.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::RuntimeState`] when constructing the test
/// runtime fails or when the compiled program returns any output count other
/// than one. Propagates typed input binding, validation, and backend execution
/// errors from [`Runtime::run_compiled`].
pub fn run_one(program: &CompiledGraph, inputs: &[&Tensor]) -> tenferro_runtime::Result<Tensor> {
    let backend = CpuBackend::new();
    let runtime = cpu_runtime_with_fft(&backend).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "fft_test_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            source,
        )
    })?;
    let mut outputs = runtime.run_compiled(program, inputs)?;
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
