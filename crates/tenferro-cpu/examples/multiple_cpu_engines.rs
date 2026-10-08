use tenferro_cpu::{runtime_engine_registration_with_id, CpuBackend, CpuId, CpuSet};
use tenferro_runtime::{EngineId, Runtime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cpus = CpuSet::new([CpuId::new(0)])?;
    let primary = CpuBackend::builder()
        .cpus(cpus.clone())
        .threads(1)?
        .build()?;
    let secondary = CpuBackend::builder().cpus(cpus).threads(1)?.build()?;

    let mut builder = Runtime::builder();
    builder.register_engine(runtime_engine_registration_with_id(
        &primary,
        EngineId::new("example.cpu.primary.v1")?,
    )?)?;
    builder.register_engine(runtime_engine_registration_with_id(
        &secondary,
        EngineId::new("example.cpu.secondary.v1")?,
    )?)?;
    let runtime = builder.build()?;
    assert_eq!(runtime.snapshot()?.engine_count(), 2);
    Ok(())
}
