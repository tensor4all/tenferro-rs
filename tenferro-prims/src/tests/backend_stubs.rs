use tenferro_algebra::Standard;
use tenferro_device::{Error, LogicalMemorySpace};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    AnalyticPrimsDescriptor, AnalyticUnaryOp, RocmBackend, RocmContext, ScalarPrimsDescriptor,
    ScalarUnaryOp, TensorAnalyticPrims, TensorScalarPrims,
};
#[cfg(not(feature = "cuda"))]
use crate::{CudaBackend, CudaContext};

fn tensor_f64(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn assert_scalar_stub_backend<Backend, Context>(ctx: &mut Context, backend_name: &str)
where
    Backend: TensorScalarPrims<Standard<f64>, Plan = (), Context = Context>,
{
    let desc = ScalarPrimsDescriptor::PointwiseUnary {
        op: ScalarUnaryOp::Neg,
    };

    assert!(!<Backend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(desc.clone()));

    let err =
        <Backend as TensorScalarPrims<Standard<f64>>>::plan(ctx, &desc, &[&[1], &[1]]).unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidArgument(message)
            if message.contains("scalar family descriptor")
                && message.contains(backend_name)
                && message.contains("phase 1")
    ));

    let input = tensor_f64(&[1.0], &[1]);
    let mut output = Tensor::<f64>::zeros(
        &[1],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let err = <Backend as TensorScalarPrims<Standard<f64>>>::execute(
        ctx,
        &(),
        1.0,
        &[&input],
        0.0,
        &mut output,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidArgument(message)
            if message.contains("scalar family execution")
                && message.contains(backend_name)
                && message.contains("phase 1")
    ));
}

fn assert_analytic_stub_backend<Backend, Context>(ctx: &mut Context, backend_name: &str)
where
    Backend: TensorAnalyticPrims<Standard<f64>, Plan = (), Context = Context>,
{
    let desc = AnalyticPrimsDescriptor::PointwiseUnary {
        op: AnalyticUnaryOp::Sqrt,
    };

    assert!(!<Backend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(desc.clone()));

    let err = <Backend as TensorAnalyticPrims<Standard<f64>>>::plan(ctx, &desc, &[&[1], &[1]])
        .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidArgument(message)
            if message.contains("analytic family descriptor")
                && message.contains(backend_name)
                && message.contains("phase 1")
    ));

    let input = tensor_f64(&[4.0], &[1]);
    let mut output = Tensor::<f64>::zeros(
        &[1],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let err = <Backend as TensorAnalyticPrims<Standard<f64>>>::execute(
        ctx,
        &(),
        1.0,
        &[&input],
        0.0,
        &mut output,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidArgument(message)
            if message.contains("analytic family execution")
                && message.contains(backend_name)
                && message.contains("phase 1")
    ));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_scalar_and_analytic_family_stubs_reject_phase1_execution() {
    let mut scalar_ctx = CudaContext::new();
    assert_scalar_stub_backend::<CudaBackend, _>(&mut scalar_ctx, "CudaBackend");

    let mut analytic_ctx = CudaContext::new();
    assert_analytic_stub_backend::<CudaBackend, _>(&mut analytic_ctx, "CudaBackend");
}

#[test]
fn rocm_scalar_and_analytic_family_stubs_reject_phase1_execution() {
    let mut scalar_ctx = RocmContext::new();
    assert_scalar_stub_backend::<RocmBackend, _>(&mut scalar_ctx, "RocmBackend");

    let mut analytic_ctx = RocmContext::new();
    assert_analytic_stub_backend::<RocmBackend, _>(&mut analytic_ctx, "RocmBackend");
}
