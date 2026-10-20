use tenferro_cpu::CpuEngine;
#[cfg(feature = "autodiff")]
use tenferro_runtime::ExtensionCacheSelector;
use tenferro_runtime::{CompiledGraph, ExecutionEngine, Runtime, RuntimeConfigError};
#[cfg(feature = "autodiff")]
use tenferro_tensor::CacheStats;
use tenferro_tensor::{Tensor, TensorRead};

/// Build a CPU runtime with the einsum extension module installed.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] when CPU engine registration, einsum module
/// construction, module installation, or final runtime construction fails.
pub fn cpu_runtime_with_einsum(backend: &CpuEngine) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(backend)?)?;
    builder.install_extension_module(tenferro_einsum::extension_module::<CpuEngine>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    builder.build()
}

/// Read the extension cache stats owned by `engine`.
#[cfg(feature = "autodiff")]
pub fn extension_cache_stats(engine: &mut CpuEngine) -> CacheStats {
    let caches = engine.take_extension_caches();
    let stats = caches.stats(ExtensionCacheSelector::All);
    engine.restore_extension_caches(caches);
    stats
}

/// Execute a compiled einsum test program on `engine` and return its outputs.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::RuntimeState`] when the target is not
/// registered or the prepared run belongs to another runtime,
/// [`tenferro_runtime::Error::Unsupported`] when no installed module or
/// provider serves the program, [`tenferro_runtime::Error::TensorRuntime`]
/// when the engine fails or an output cannot be copied, and
/// [`tenferro_runtime::Error::Extension`] for an extension kernel's typed
/// failure.
pub fn run_on_cpu_with_engine(
    runtime: &Runtime,
    engine: &mut CpuEngine,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> tenferro_runtime::Result<Vec<Tensor>> {
    let reads: Vec<TensorRead<'_>> = inputs.iter().map(|t| TensorRead::from_tensor(t)).collect();
    let target = engine.target();
    let prepared = runtime.prepare(program, &reads, std::slice::from_ref(&target))?;
    let outputs = runtime.run(&mut [engine], &prepared, &reads)?;
    outputs
        .into_iter()
        .map(|output| {
            output
                .into_tensor(engine)
                .map_err(tenferro_runtime::Error::from)
        })
        .collect()
}

/// Execute a compiled einsum test program and return its single output.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::RuntimeState`] when constructing the test
/// runtime fails or when the compiled program returns any output count other
/// than one. Propagates typed input binding, validation, and engine execution
/// errors from [`Runtime::prepare`] and [`Runtime::run`].
pub fn run_one(program: &CompiledGraph, inputs: &[&Tensor]) -> tenferro_runtime::Result<Tensor> {
    let mut engine = CpuEngine::serial();
    let runtime = cpu_runtime_with_einsum(&engine).map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "einsum_test_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            source,
        )
    })?;
    let mut outputs = run_on_cpu_with_engine(&runtime, &mut engine, program, inputs)?;
    let actual = outputs.len();
    if actual != 1 {
        return Err(tenferro_runtime::Error::runtime_state(
            "einsum_test_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            format!("expected one runtime output, got {actual}"),
        ));
    }
    outputs.pop().ok_or_else(|| {
        tenferro_runtime::Error::runtime_state(
            "einsum_test_runtime",
            tenferro_runtime::ErrorPhase::Execution,
            "runtime returned no output after successful output-count validation",
        )
    })
}
