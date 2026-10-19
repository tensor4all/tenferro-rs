use tenferro_cpu::CpuBackend;
use tenferro_tensor::backend::Engine;

fn recover_cpu_owner(session: &mut dyn Engine) -> &mut CpuBackend {
    session
}

fn main() {}
