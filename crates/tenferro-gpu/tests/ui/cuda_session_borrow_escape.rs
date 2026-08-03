use tenferro_gpu::CudaExecSession;

fn escape<'a>(session: CudaExecSession<'a>) -> CudaExecSession<'static> {
    session
}

fn main() {
    let _ = escape;
}
