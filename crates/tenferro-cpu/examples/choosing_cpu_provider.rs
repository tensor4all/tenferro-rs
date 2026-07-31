use tenferro_cpu::{CpuBackend, CpuBackendKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let faer = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Faer)?;
    let blas = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Blas)?;
    assert_eq!(faer.kind(), CpuBackendKind::Faer);
    assert_eq!(blas.kind(), CpuBackendKind::Blas);
    Ok(())
}
