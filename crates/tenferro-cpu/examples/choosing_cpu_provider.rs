use tenferro_cpu::{CpuEngine, Pool};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The caller builds the pool (and pins its workers if it wants to); the
    // engine computes on it. A serial engine has no pool at all.
    let serial = CpuEngine::serial();
    let pooled = CpuEngine::new(Pool::builder().threads(2).build()?);
    println!(
        "provider={} serial_threads={} pooled_threads={}",
        tenferro_cpu::cpu_provider_id(),
        serial.num_threads(),
        pooled.num_threads()
    );
    Ok(())
}
