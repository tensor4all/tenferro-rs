use tenferro_internal_error::Error;
use tenferro_prims::CpuContext;

use crate::{set_default_runtime, with_default_runtime, RuntimeContext};

#[test]
fn context_module_reexports_internal_runtime_holder_helpers() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let value = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
    assert_eq!(value, "cpu");

    drop(_guard);
    let missing = with_default_runtime(|ctx| Ok(ctx.name()));
    assert!(matches!(missing, Err(Error::RuntimeNotConfigured)));
}
