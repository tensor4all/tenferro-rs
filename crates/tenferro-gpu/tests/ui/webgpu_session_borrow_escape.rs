use tenferro_gpu::WebGpuExecSession;

fn escape<'a>(session: WebGpuExecSession<'a>) -> WebGpuExecSession<'static> {
    session
}

fn main() {
    let _ = escape;
}
