use tenferro_ad::EagerBackend;
use tenferro_cpu::CpuBackend;

fn main() {
    let _backend = EagerBackend::Cpu(CpuBackend::new());
}
