use tenferro_gpu::{CudaBackend, CudaExecSession};

fn owner_accessor<'a>(session: &'a mut CudaExecSession<'a>) -> &'a mut CudaBackend {
    session.backend_mut()
}

fn owner_deref<'a>(session: &'a mut CudaExecSession<'a>) -> &'a mut CudaBackend {
    &mut *session
}

fn main() {
    let _ = owner_accessor;
    let _ = owner_deref;
}
