use tenferro_tensor::backend::Engine;

fn recover_cpu_owner(session: &mut dyn Engine) -> &mut tenferro_cpu::CpuEngine {
    session
}

fn main() {}
