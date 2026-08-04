use tenferro_gpu::{webgpu::WebGpuBackend, webgpu::WebGpuExecSession};

fn owner_accessor<'a>(session: &'a mut WebGpuExecSession<'a>) -> &'a mut WebGpuBackend {
    session.backend_mut()
}

fn owner_deref<'a>(session: &'a mut WebGpuExecSession<'a>) -> &'a mut WebGpuBackend {
    &mut *session
}

fn main() {
    let _ = owner_accessor;
    let _ = owner_deref;
}
