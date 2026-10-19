use tenferro_ad::EagerRuntime;

fn main() {
    let runtime = EagerRuntime::new().unwrap();
    let mut engine = tenferro_cpu::CpuEngine::serial();

    runtime
        .with_eager_session(&mut engine, |_session| Ok::<_, tenferro_ad::Error>(7_u8))
        .unwrap();
}
