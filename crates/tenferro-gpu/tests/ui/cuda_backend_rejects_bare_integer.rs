use tenferro_gpu::cuda::CudaBackend;

fn main() {
    let _ = CudaBackend::new(0);
}
