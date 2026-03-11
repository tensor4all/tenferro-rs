use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    CpuBackend, CpuContext, ScalarBinaryOp, ScalarPrimsDescriptor, ScalarReductionOp,
    TensorScalarPrims,
};

fn tensor_f64(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn cpu_scalar_phase1_supports_add_div_and_mean() {
    assert!(<CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
        ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::Add,
        }
    ));
    assert!(<CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
        ScalarPrimsDescriptor::PointwiseBinary {
            op: ScalarBinaryOp::Div,
        }
    ));
    assert!(<CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
        ScalarPrimsDescriptor::Reduction {
            modes_a: vec![0, 1],
            modes_c: vec![1],
            op: ScalarReductionOp::Mean,
        }
    ));
}

#[test]
fn cpu_scalar_phase1_executes_add_and_mean_reduction() {
    let mut ctx = CpuContext::new(1);

    let add_desc = ScalarPrimsDescriptor::PointwiseBinary {
        op: ScalarBinaryOp::Add,
    };
    let add_plan = <CpuBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut ctx,
        &add_desc,
        &[&[2, 2], &[2, 2], &[2, 2]],
    )
    .unwrap();
    let lhs = tensor_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let rhs = tensor_f64(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let mut add_out = Tensor::<f64>::zeros(
        &[2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    <CpuBackend as TensorScalarPrims<Standard<f64>>>::execute(
        &mut ctx,
        &add_plan,
        1.0,
        &[&lhs, &rhs],
        0.0,
        &mut add_out,
    )
    .unwrap();
    assert_eq!(add_out.buffer().as_slice().unwrap(), &[11.0, 22.0, 33.0, 44.0]);

    let mean_desc = ScalarPrimsDescriptor::Reduction {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        op: ScalarReductionOp::Mean,
    };
    let mean_plan = <CpuBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut ctx,
        &mean_desc,
        &[&[2, 2], &[2]],
    )
    .unwrap();
    let input = tensor_f64(&[1.0, 3.0, 5.0, 7.0], &[2, 2]);
    let mut mean_out =
        Tensor::<f64>::zeros(&[2], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor);
    <CpuBackend as TensorScalarPrims<Standard<f64>>>::execute(
        &mut ctx,
        &mean_plan,
        1.0,
        &[&input],
        0.0,
        &mut mean_out,
    )
    .unwrap();
    assert_eq!(mean_out.buffer().as_slice().unwrap(), &[2.0, 6.0]);
}

#[test]
fn cuda_scalar_phase1_does_not_advertise_unimplemented_ops() {
    assert!(
        !<crate::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseBinary {
                op: ScalarBinaryOp::Add,
            }
        )
    );
    assert!(
        !<crate::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::Reduction {
                modes_a: vec![0, 1],
                modes_c: vec![1],
                op: ScalarReductionOp::Mean,
            }
        )
    );
}
