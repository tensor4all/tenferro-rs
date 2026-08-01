use tenferro_cpu::CpuBackend;
use tenferro_tensor::backend::BackendSession;

fn recover_cpu_owner(session: &mut dyn BackendSession) -> &mut CpuBackend {
    session
}

fn main() {}
