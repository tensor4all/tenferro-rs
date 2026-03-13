use super::*;
use tenferro_prims::CpuContext;

#[test]
fn set_and_restore_runtime_context() {
    let guard0 = set_runtime_context(RuntimeContext::Cpu(CpuContext::new(1)));
    let value = with_runtime_context(|ctx| Ok(ctx.name())).unwrap();
    assert_eq!(value, "cpu");

    let guard1 = set_runtime_context(RuntimeContext::Cpu(CpuContext::new(2)));
    let value = with_runtime_context(|ctx| match ctx {
        RuntimeContext::Cpu(inner) => Ok(inner.num_threads()),
        _ => unreachable!(),
    })
    .unwrap();
    assert_eq!(value, 2);

    drop(guard1);
    let value = with_runtime_context(|ctx| match ctx {
        RuntimeContext::Cpu(inner) => Ok(inner.num_threads()),
        _ => unreachable!(),
    })
    .unwrap();
    assert_eq!(value, 1);

    drop(guard0);
    let missing = with_runtime_context(|ctx| Ok(ctx.name()));
    assert!(matches!(missing, Err(Error::RuntimeNotConfigured)));
}
