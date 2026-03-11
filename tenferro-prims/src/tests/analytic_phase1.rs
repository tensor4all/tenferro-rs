use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    AnalyticBinaryOp, AnalyticPrimsDescriptor, AnalyticUnaryOp, CpuBackend, CpuContext,
    TensorAnalyticPrims,
};

fn tensor_f64(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn cpu_analytic_phase1_supports_exp_log_tanh_and_pow() {
    for op in [
        AnalyticUnaryOp::Exp,
        AnalyticUnaryOp::Log,
        AnalyticUnaryOp::Tanh,
    ] {
        assert!(
            <CpuBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(
                AnalyticPrimsDescriptor::PointwiseUnary { op }
            )
        );
    }
    assert!(
        <CpuBackend as TensorAnalyticPrims<Standard<f64>>>::has_analytic_support(
            AnalyticPrimsDescriptor::PointwiseBinary {
                op: AnalyticBinaryOp::Pow,
            }
        )
    );
}

#[test]
fn cpu_analytic_phase1_executes_exp_and_pow() {
    let mut ctx = CpuContext::new(1);

    let exp_desc = AnalyticPrimsDescriptor::PointwiseUnary {
        op: AnalyticUnaryOp::Exp,
    };
    let exp_plan = <CpuBackend as TensorAnalyticPrims<Standard<f64>>>::plan(
        &mut ctx,
        &exp_desc,
        &[&[2], &[2]],
    )
    .unwrap();
    let input = tensor_f64(&[0.0, 1.0], &[2]);
    let mut exp_out = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    <CpuBackend as TensorAnalyticPrims<Standard<f64>>>::execute(
        &mut ctx,
        &exp_plan,
        1.0,
        &[&input],
        0.0,
        &mut exp_out,
    )
    .unwrap();
    let exp_data = exp_out.buffer().as_slice().unwrap();
    assert!((exp_data[0] - 1.0).abs() < 1.0e-12);
    assert!((exp_data[1] - std::f64::consts::E).abs() < 1.0e-12);

    let pow_desc = AnalyticPrimsDescriptor::PointwiseBinary {
        op: AnalyticBinaryOp::Pow,
    };
    let pow_plan = <CpuBackend as TensorAnalyticPrims<Standard<f64>>>::plan(
        &mut ctx,
        &pow_desc,
        &[&[2], &[2], &[2]],
    )
    .unwrap();
    let bases = tensor_f64(&[2.0, 9.0], &[2]);
    let exponents = tensor_f64(&[3.0, 0.5], &[2]);
    let mut pow_out = Tensor::<f64>::zeros(
        &[2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    <CpuBackend as TensorAnalyticPrims<Standard<f64>>>::execute(
        &mut ctx,
        &pow_plan,
        1.0,
        &[&bases, &exponents],
        0.0,
        &mut pow_out,
    )
    .unwrap();
    let pow_data = pow_out.buffer().as_slice().unwrap();
    assert!((pow_data[0] - 8.0).abs() < 1.0e-12);
    assert!((pow_data[1] - 3.0).abs() < 1.0e-12);
}

#[test]
fn cuda_analytic_phase1_does_not_advertise_unimplemented_ops() {
    for desc in [
        AnalyticPrimsDescriptor::PointwiseUnary {
            op: AnalyticUnaryOp::Exp,
        },
        AnalyticPrimsDescriptor::PointwiseUnary {
            op: AnalyticUnaryOp::Log,
        },
        AnalyticPrimsDescriptor::PointwiseBinary {
            op: AnalyticBinaryOp::Pow,
        },
    ] {
        assert!(!<crate::CudaBackend as TensorAnalyticPrims<
            Standard<f64>,
        >>::has_analytic_support(desc));
    }
}
