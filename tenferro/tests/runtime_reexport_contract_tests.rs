use tenferro::{
    runtime, set_default_runtime, with_default_runtime, DefaultRuntimeGuard, RuntimeContext,
};
use tenferro_internal_runtime::{
    set_default_runtime as set_internal_default_runtime,
    with_default_runtime as with_internal_default_runtime, with_runtime as with_internal_runtime,
    DefaultRuntimeGuard as InternalGuard, RuntimeContext as InternalRuntimeContext,
};
use tenferro_prims::CpuContext;

fn keep_internal_guard(guard: InternalGuard) -> InternalGuard {
    guard
}

fn keep_public_guard(guard: DefaultRuntimeGuard) -> DefaultRuntimeGuard {
    guard
}

#[test]
fn public_runtime_surface_reexports_the_internal_runtime_types() {
    let public = RuntimeContext::Cpu(CpuContext::new(1));
    let _: InternalRuntimeContext = public;

    let internal = InternalRuntimeContext::Cpu(CpuContext::new(2));
    let _: RuntimeContext = internal;

    let public_guard =
        keep_internal_guard(set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1))));
    let public_name = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
    let internal_name = with_internal_default_runtime(|ctx| Ok(ctx.name())).unwrap();
    assert_eq!(public_name, internal_name);
    drop(public_guard);

    let internal_guard = keep_public_guard(set_internal_default_runtime(
        InternalRuntimeContext::Cpu(CpuContext::new(1)),
    ));
    let public_name = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
    let internal_name = with_internal_default_runtime(|ctx| Ok(ctx.name())).unwrap();
    assert_eq!(public_name, internal_name);
    drop(internal_guard);

    let scoped_name = runtime::with_runtime(RuntimeContext::Cpu(CpuContext::new(3)), || {
        with_default_runtime(|ctx| Ok(ctx.name()))
    })
    .unwrap();
    assert_eq!(scoped_name, "cpu");

    let internal_scoped =
        with_internal_runtime(InternalRuntimeContext::Cpu(CpuContext::new(4)), || {
            with_internal_default_runtime(|ctx| Ok(ctx.name()))
        })
        .unwrap();
    assert_eq!(internal_scoped, "cpu");
}
