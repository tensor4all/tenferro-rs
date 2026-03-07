use super::*;

#[test]
fn default_runtime_roundtrip() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let runtime = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
    assert_eq!(runtime, "cpu");
}
