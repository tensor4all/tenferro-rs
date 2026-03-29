use tenferro_prims::CpuContext;

use crate::{set_default_runtime, with_default_runtime, with_runtime, RuntimeContext};

#[test]
fn public_runtime_reexports_install_and_read_the_internal_runtime() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let name = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
    assert_eq!(name, "cpu");

    let scoped = with_runtime(RuntimeContext::Cpu(CpuContext::new(2)), || {
        with_default_runtime(|ctx| Ok(ctx.name()))
    })
    .unwrap();
    assert_eq!(scoped, "cpu");
}
