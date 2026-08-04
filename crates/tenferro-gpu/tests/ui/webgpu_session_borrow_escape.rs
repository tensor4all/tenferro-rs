use tenferro_gpu::webgpu::WebGpuExecSession;

fn escape<'a>(session: WebGpuExecSession<'a>) -> WebGpuExecSession<'static> {
    session
}

fn main() {
    let _ = escape;
}
