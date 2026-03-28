use super::*;
use crate::Error;
use crate::RuntimeContext;
use tenferro_prims::CpuContext;

#[test]
fn context_module_reexports_internal_runtime_holder_helpers() {
    let _guard = set_runtime_context(RuntimeContext::Cpu(CpuContext::new(1)));
    let value = with_runtime_context(|ctx| Ok(ctx.name())).unwrap();
    assert_eq!(value, "cpu");

    drop(_guard);
    let missing = with_runtime_context(|ctx| Ok(ctx.name()));
    assert!(matches!(missing, Err(Error::RuntimeNotConfigured)));
}
