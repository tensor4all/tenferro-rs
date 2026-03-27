use tenferro_prims::CpuContext;

use crate::{runtime, Error, RuntimeContext, Tensor};

#[test]
fn with_runtime_installs_scope_and_restores_previous_default() {
    assert!(matches!(
        crate::with_default_runtime(|ctx| Ok(ctx.name())),
        Err(Error::RuntimeNotConfigured)
    ));

    let out = runtime::with_runtime(RuntimeContext::Cpu(CpuContext::new(1)), || {
        let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])?;
        x.sum()
    })
    .unwrap();

    assert!(out.dims().is_empty());
    assert!(matches!(
        crate::with_default_runtime(|ctx| Ok(ctx.name())),
        Err(Error::RuntimeNotConfigured)
    ));
}

#[test]
fn with_runtime_restores_outer_default_runtime_after_nested_scope() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let outer_before = crate::with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
    assert_eq!(outer_before, "cpu");

    let nested = runtime::with_runtime(RuntimeContext::Cpu(CpuContext::new(2)), || {
        crate::with_default_runtime(|ctx| Ok(ctx.name()))
    })
    .unwrap();
    assert_eq!(nested, "cpu");

    let outer_after = crate::with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
    assert_eq!(outer_after, "cpu");
}
