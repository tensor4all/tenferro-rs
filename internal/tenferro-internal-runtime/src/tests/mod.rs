use tenferro_internal_error::{Error, Result};
use tenferro_linalg::backend::LinalgCapabilityOp;
use tenferro_prims::{CpuContext, CudaContext, RocmContext};

use crate::dispatch::{unsupported_runtime_capability, with_einsum_runtime, with_linalg_runtime};
use crate::{set_default_runtime, with_default_runtime, with_runtime, RuntimeContext};

mod context_shim_contract;
mod dispatch;
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
        Err(Error::InvalidTensorOperands {
            message: "boom".to_string(),
        })
    });
    assert!(matches!(err, Err(Error::InvalidTensorOperands { .. })));

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

#[test]
fn unsupported_runtime_capability_builds_expected_error() {
    let err = unsupported_runtime_capability("qr", "cuda");
    assert!(matches!(
        err,
        Error::UnsupportedRuntimeOp {
            op: "qr",
            runtime: "cuda"
        }
    ));
    assert_eq!(
        err.to_string(),
        "operation `qr` is not supported on runtime `cuda`"
    );
}

#[test]
fn with_einsum_runtime_runs_cpu_and_checks_gpu_capabilities() {
    let _cpu = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let cpu_name = with_einsum_runtime::<f64, _>(
        "einsum_probe",
        |_| Ok("cpu"),
        |_| Ok("cuda"),
        |_| Ok("rocm"),
    )
    .unwrap();
    assert_eq!(cpu_name, "cpu");
    drop(_cpu);

    let _cuda = set_default_runtime(RuntimeContext::Cuda(CudaContext::new()));
    match with_einsum_runtime::<f64, _>(
        "einsum_probe",
        |_| Ok("cpu"),
        |_| Ok("cuda"),
        |_| Ok("rocm"),
    ) {
        Ok(name) => assert_eq!(name, "cuda"),
        Err(Error::UnsupportedRuntimeOp { op, runtime }) => {
            assert_eq!(op, "einsum_probe");
            assert_eq!(runtime, "cuda");
        }
        Err(other) => panic!("unexpected cuda einsum dispatch result: {other:?}"),
    }
    drop(_cuda);

    let _rocm = set_default_runtime(RuntimeContext::Rocm(RocmContext::new()));
    match with_einsum_runtime::<f64, _>(
        "einsum_probe",
        |_| Ok("cpu"),
        |_| Ok("cuda"),
        |_| Ok("rocm"),
    ) {
        Ok(name) => assert_eq!(name, "rocm"),
        Err(Error::UnsupportedRuntimeOp { op, runtime }) => {
            assert_eq!(op, "einsum_probe");
            assert_eq!(runtime, "rocm");
        }
        Err(other) => panic!("unexpected rocm einsum dispatch result: {other:?}"),
    }
}

#[test]
fn with_linalg_runtime_runs_cpu_and_checks_gpu_capabilities() {
    let _cpu = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let cpu_name = with_linalg_runtime::<f64, _>(
        "solve_probe",
        LinalgCapabilityOp::Solve,
        |_| Ok("cpu"),
        |_| Ok("cuda"),
        |_| Ok("rocm"),
    )
    .unwrap();
    assert_eq!(cpu_name, "cpu");
    drop(_cpu);

    let _cuda = set_default_runtime(RuntimeContext::Cuda(CudaContext::new()));
    match with_linalg_runtime::<f64, _>(
        "solve_probe",
        LinalgCapabilityOp::Solve,
        |_| Ok("cpu"),
        |_| Ok("cuda"),
        |_| Ok("rocm"),
    ) {
        Ok(name) => assert_eq!(name, "cuda"),
        Err(Error::UnsupportedRuntimeOp { op, runtime }) => {
            assert_eq!(op, "solve_probe");
            assert_eq!(runtime, "cuda");
        }
        Err(other) => panic!("unexpected cuda linalg dispatch result: {other:?}"),
    }
    drop(_cuda);

    let _rocm = set_default_runtime(RuntimeContext::Rocm(RocmContext::new()));
    match with_linalg_runtime::<f64, _>(
        "solve_probe",
        LinalgCapabilityOp::Solve,
        |_| Ok("cpu"),
        |_| Ok("cuda"),
        |_| Ok("rocm"),
    ) {
        Ok(name) => assert_eq!(name, "rocm"),
        Err(Error::UnsupportedRuntimeOp { op, runtime }) => {
            assert_eq!(op, "solve_probe");
            assert_eq!(runtime, "rocm");
        }
        Err(other) => panic!("unexpected rocm linalg dispatch result: {other:?}"),
    }
}
