use super::*;

#[test]
fn run_requires_runtime() {
    let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let err = qr(&t).run().err();
    assert!(matches!(err, Some(Error::RuntimeNotConfigured)));
}

#[test]
fn run_with_cuda_runtime_returns_unsupported_runtime_error() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cuda(CudaContext::new()));
    let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let err = qr(&t).run().err();
    assert!(matches!(
        err,
        Some(Error::UnsupportedRuntimeOp {
            op: "qr",
            runtime: "cuda"
        })
    ));
}

#[test]
fn run_with_rocm_runtime_returns_unsupported_runtime_error() {
    let _guard = crate::set_default_runtime(RuntimeContext::Rocm(RocmContext::new()));
    let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let err = qr(&t).run().err();
    assert!(matches!(
        err,
        Some(Error::UnsupportedRuntimeOp {
            op: "qr",
            runtime: "rocm"
        })
    ));
}

#[test]
fn primal_einsum_builder_runs() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let out = einsum("ij,jk->ik", &[&a, &b]).run().unwrap();
    assert_eq!(out.dims(), &[2, 2]);
    assert_eq!(as_slice(&out).len(), 4);
}

#[test]
fn primal_qr_builder_runs() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let out = qr(&t).run().unwrap();
    assert_eq!(out.q.dims(), &[2, 2]);
    assert_eq!(out.r.dims(), &[2, 2]);
}

#[test]
fn solve_triangular_ad_supports_forward_mode() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = Tensor::<f64>::from_slice(&[2.0, 0.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let da = Tensor::<f64>::from_slice(&[0.1, 0.0, -0.2, 0.1], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let db = Tensor::<f64>::from_slice(&[0.2, -0.1], &[2], MemoryOrder::ColumnMajor).unwrap();

    let ad_a = AdTensor::new_forward(a, da).unwrap();
    let ad_b = AdTensor::new_forward(b, db).unwrap();
    let out = solve_triangular_ad(&ad_a, &ad_b).run().unwrap();
    assert!(matches!(out.as_value(), AdValue::Forward { .. }));
    assert_eq!(out.dims(), &[2]);
}
