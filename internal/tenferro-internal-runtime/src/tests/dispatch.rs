use tenferro_internal_error::Error;
use tenferro_linalg::backend::LinalgCapabilityOp;
use tenferro_prims::{CpuContext, CudaContext};

use crate::{set_default_runtime, RuntimeContext};

#[test]
fn unsupported_runtime_capability_reports_op_and_runtime() {
    assert!(matches!(
        crate::dispatch::unsupported_runtime_capability("solve", "cuda"),
        Error::UnsupportedRuntimeOp {
            op: "solve",
            runtime: "cuda",
        }
    ));
}

#[test]
fn with_einsum_runtime_uses_the_current_cpu_runtime_branch() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let name = crate::dispatch::with_einsum_runtime::<f64, _>(
        "einsum",
        |_ctx: &mut CpuContext| Ok::<_, Error>("cpu"),
        |_ctx| Ok::<_, Error>("cuda"),
        |_ctx| Ok::<_, Error>("rocm"),
    )
    .unwrap();

    assert_eq!(name, "cpu");
}

#[test]
fn with_linalg_runtime_reports_unsupported_cuda_capability() {
    let _guard = set_default_runtime(RuntimeContext::Cuda(CudaContext::new()));

    let err = crate::dispatch::with_linalg_runtime::<f64, _>(
        "solve_ex",
        LinalgCapabilityOp::SolveEx,
        |_| Ok::<_, Error>(()),
        |_| Ok::<_, Error>(()),
        |_| Ok::<_, Error>(()),
    )
    .unwrap_err();

    assert!(matches!(
        err,
        Error::UnsupportedRuntimeOp {
            op: "solve_ex",
            runtime: "cuda",
        }
    ));
}
