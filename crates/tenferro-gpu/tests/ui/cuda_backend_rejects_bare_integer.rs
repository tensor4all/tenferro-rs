use tenferro_gpu::cuda::CudaEngine;

fn main() {
    let _ = CudaEngine::open(0);
}
