use tenferro_gpu::cuda::CudaHeldSession;

fn escape<'a>(session: CudaHeldSession<'a>) -> CudaHeldSession<'static> {
    session
}

fn main() {
    let _ = escape;
}
