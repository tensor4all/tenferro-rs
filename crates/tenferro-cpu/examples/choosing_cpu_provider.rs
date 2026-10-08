use tenferro_cpu::{CpuBackend, CpuId, CpuSet};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The CPU backend is chosen at compile time: `native` (the default) or
    // `blas`. Both expose the same tenferro-owned placement API.
    let default_backend = CpuBackend::new();
    let pinned = CpuBackend::builder()
        .cpus(CpuSet::new([CpuId::new(0)])?)
        .threads(1)?
        .build()?;
    assert!(default_backend.num_threads() >= 1);
    assert_eq!(pinned.num_threads(), 1);
    Ok(())
}
