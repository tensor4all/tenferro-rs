use tenferro_gpu::CudaBackend;

fn main() {
    let _ = CudaBackend::new(0);
}
