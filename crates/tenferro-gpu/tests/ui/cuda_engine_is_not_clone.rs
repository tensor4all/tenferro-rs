use tenferro_gpu::cuda::CudaEngine;

/// An engine owns its stream, vendor handles and pending work, so it cannot be duplicated: two
/// handles would issue work on one stream without either knowing about the other.
fn duplicate(engine: &CudaEngine) -> CudaEngine {
    engine.clone()
}

fn main() {
    let _ = duplicate;
}
