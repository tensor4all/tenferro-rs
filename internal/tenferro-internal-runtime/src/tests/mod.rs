use tenferro_internal_error::{Error, Result};
use tenferro_prims::{CpuContext, CudaContext, RocmContext};

use crate::{set_default_runtime, with_default_runtime, with_runtime, RuntimeContext};

mod context_shim_contract;
mod organization;
mod scope_contract;

#[test]
fn default_runtime_roundtrip() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let runtime = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
    assert_eq!(runtime, "cpu");
}

#[test]
fn with_runtime_installs_scope_and_restores_previous_default() {
    assert!(matches!(
        with_default_runtime(|ctx| Ok(ctx.name())),
        Err(Error::RuntimeNotConfigured)
    ));

    let name = with_runtime(RuntimeContext::Cpu(CpuContext::new(1)), || {
        with_default_runtime(|ctx| Ok(ctx.name()))
    })
    .unwrap();
    assert_eq!(name, "cpu");

    assert!(matches!(
        with_default_runtime(|ctx| Ok(ctx.name())),
        Err(Error::RuntimeNotConfigured)
    ));
}

#[test]
fn with_runtime_restores_outer_default_runtime_after_nested_scope() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let outer_before = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
    assert_eq!(outer_before, "cpu");

    let nested = with_runtime(RuntimeContext::Cpu(CpuContext::new(2)), || {
        with_default_runtime(|ctx| Ok(ctx.name()))
    })
    .unwrap();
    assert_eq!(nested, "cpu");

    let outer_after = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
    assert_eq!(outer_after, "cpu");
}

#[test]
fn with_runtime_restores_previous_runtime_after_error() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let err: Result<()> = with_runtime(RuntimeContext::Cpu(CpuContext::new(2)), || {
        Err(Error::InvalidAdTensor {
            message: "boom".to_string(),
        })
    });
    assert!(matches!(err, Err(Error::InvalidAdTensor { .. })));

    let outer_after = with_default_runtime(|ctx| Ok(ctx.name())).unwrap();
    assert_eq!(outer_after, "cpu");
}

#[test]
fn runtime_context_from_impls_cover_all_supported_variants() {
    let cpu: RuntimeContext = CpuContext::new(1).into();
    let cuda: RuntimeContext = CudaContext::new().into();
    let rocm: RuntimeContext = RocmContext::new().into();

    assert_eq!(cpu.name(), "cpu");
    assert_eq!(cuda.name(), "cuda");
    assert_eq!(rocm.name(), "rocm");
}

#[test]
fn with_runtime_installs_cuda_and_rocm_variants_for_scope_lookup() {
    let cuda_name = with_runtime(RuntimeContext::Cuda(CudaContext::new()), || {
        with_default_runtime(|ctx| Ok(ctx.name()))
    })
    .unwrap();
    assert_eq!(cuda_name, "cuda");

    let rocm_name = with_runtime(RuntimeContext::Rocm(RocmContext::new()), || {
        with_default_runtime(|ctx| Ok(ctx.name()))
    })
    .unwrap();
    assert_eq!(rocm_name, "rocm");

    assert!(matches!(
        with_default_runtime(|ctx| Ok(ctx.name())),
        Err(Error::RuntimeNotConfigured)
    ));
}
