use tenferro_gpu::{cuda::CudaEngine, cuda::CudaExecSession};

fn owner_accessor<'a>(session: &'a mut CudaExecSession<'a>) -> &'a mut CudaEngine {
    session.backend_mut()
}

fn owner_deref<'a>(session: &'a mut CudaExecSession<'a>) -> &'a mut CudaEngine {
    &mut *session
}

fn main() {
    let _ = owner_accessor;
    let _ = owner_deref;
}
