use tenferro_cpu::{runtime_engine_registration_with_id, CpuBackend, CpuBackendKind};
use tenferro_runtime::{EngineId, Runtime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let primary = CpuBackend::with_threads_and_kind(2, CpuBackendKind::Faer)?;
    let secondary = CpuBackend::with_threads_and_kind(2, CpuBackendKind::Blas)?;

    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration_with_id(
        &primary,
        EngineId::new("example.cpu.faer.v1")?,
    )?)?;
    builder.register_engine(runtime_engine_registration_with_id(
        &secondary,
        EngineId::new("example.cpu.blas.v1")?,
    )?)?;
    let runtime = builder.build()?;
    assert_eq!(runtime.snapshot()?.engine_count(), 2);
    Ok(())
}
