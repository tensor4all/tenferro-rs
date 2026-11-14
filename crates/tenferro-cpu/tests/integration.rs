#[path = "integration/backend_capability_contracts.rs"]
mod backend_capability_contracts;
#[path = "integration/lanes.rs"]
mod lanes;
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
