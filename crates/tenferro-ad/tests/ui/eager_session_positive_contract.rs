use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;

fn main() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new());

    runtime
        .with_execution_session(|_session| 7_u8)
        .unwrap();
    runtime.replace_cpu_backend(CpuBackend::new()).unwrap();
}
