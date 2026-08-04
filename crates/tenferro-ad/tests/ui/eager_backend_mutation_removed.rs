use tenferro_ad::{EagerBackend, EagerRuntime};
use tenferro_cpu::CpuBackend;

fn main() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();

    runtime
        .with_backend_mut(|backend| {
            *backend = EagerBackend::Cpu(CpuBackend::new());
        })
        .unwrap();
}
