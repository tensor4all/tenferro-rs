use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{CpuBackend, CpuContext, ScalarPrimsDescriptor, ScalarTernaryOp, TensorScalarPrims};

fn tensor_f64(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn cpu_scalar_phase2_supports_where_for_ordered_real_and_rejects_complex() {
    assert!(
        <CpuBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseTernary {
                op: ScalarTernaryOp::Where,
            }
        )
    );
    assert!(!<CpuBackend as TensorScalarPrims<
        Standard<num_complex::Complex64>,
    >>::has_scalar_support(
        ScalarPrimsDescriptor::PointwiseTernary {
            op: ScalarTernaryOp::Where,
        }
    ));
}

#[test]
fn cpu_scalar_phase2_executes_where_for_same_shape_real_tensors() {
    let mut ctx = CpuContext::new(1);
    let desc = ScalarPrimsDescriptor::PointwiseTernary {
        op: ScalarTernaryOp::Where,
    };
    let plan = <CpuBackend as TensorScalarPrims<Standard<f64>>>::plan(
        &mut ctx,
        &desc,
        &[&[2, 2], &[2, 2], &[2, 2], &[2, 2]],
    )
    .unwrap();
    let mask = tensor_f64(&[1.0, 0.0, 2.0, -3.0], &[2, 2]);
    let on_true = tensor_f64(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let on_false = tensor_f64(&[-1.0, -2.0, -3.0, -4.0], &[2, 2]);
    let mut output = Tensor::<f64>::zeros(
        &[2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    <CpuBackend as TensorScalarPrims<Standard<f64>>>::execute(
        &mut ctx,
        &plan,
        1.0,
        &[&mask, &on_true, &on_false],
        0.0,
        &mut output,
    )
    .unwrap();

    assert_eq!(
        output.buffer().as_slice().unwrap(),
        &[10.0, -2.0, 30.0, 40.0]
    );
}

#[test]
fn cuda_scalar_phase2_advertises_where_only_when_wired() {
    let where_supported =
        <crate::CudaBackend as TensorScalarPrims<Standard<f64>>>::has_scalar_support(
            ScalarPrimsDescriptor::PointwiseTernary {
                op: ScalarTernaryOp::Where,
            },
        );

    if cfg!(feature = "cuda") {
        assert!(where_supported);
    } else {
        assert!(!where_supported);
    }
}
