// The provider-pointer registry is process-global, so every test that
// registers or observes an injected CBLAS/LAPACK symbol serializes on this lock.
#[cfg(all(feature = "blas", feature = "provider-inject"))]
pub(crate) static PROVIDER_INJECT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[path = "integration/backend_capability_contracts.rs"]
mod backend_capability_contracts;
#[path = "integration/inject_dual_abi_tests.rs"]
mod inject_dual_abi_tests;
#[path = "integration/inject_tests.rs"]
mod inject_tests;
#[path = "integration/runtime_error_tests.rs"]
mod runtime_error_tests;
#[path = "integration/static_replay.rs"]
mod static_replay;

/// An engine with `threads` workers (serial for one).
#[allow(dead_code)]
pub(crate) fn engine(threads: usize) -> tenferro_cpu::CpuEngine {
    if threads <= 1 {
        tenferro_cpu::CpuEngine::serial()
    } else {
        tenferro_cpu::CpuEngine::new(
            tenferro_cpu::Pool::builder()
                .threads(threads)
                .build()
                .expect("test pool"),
        )
    }
}

/// Run `f` on `engine` viewed as `&mut dyn Engine`.
#[allow(dead_code)]
pub(crate) fn run<R>(
    engine: &mut tenferro_cpu::CpuEngine,
    f: impl FnOnce(&mut dyn tenferro_tensor::Engine) -> R,
) -> tenferro_tensor::Result<R> {
    Ok(f(engine))
}
