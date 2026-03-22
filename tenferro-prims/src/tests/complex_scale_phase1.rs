use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{ComplexScalePrimsDescriptor, CpuBackend, CpuContext, TensorComplexScalePrims};

fn assert_close_slice_f64(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!(
            (*lhs - *rhs).norm() <= tol,
            "got {lhs:?}, expected {rhs:?}, tol={tol}"
        );
    }
}

fn assert_close_slice_f32(actual: &[Complex32], expected: &[Complex32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!(
            (*lhs - *rhs).norm() <= tol,
            "got {lhs:?}, expected {rhs:?}, tol={tol}"
        );
    }
}

#[test]
fn cpu_complex_scale_phase1_supports_pointwise_mul_for_complex32_and_complex64() {
    let desc = ComplexScalePrimsDescriptor::PointwiseMul;

    assert!(
        <CpuBackend as TensorComplexScalePrims<Complex32>>::has_complex_scale_support(desc.clone())
    );
    assert!(<CpuBackend as TensorComplexScalePrims<Complex64>>::has_complex_scale_support(desc));
}

#[test]
fn cpu_complex_scale_phase1_executes_pointwise_mul_for_complex64() {
    let mut ctx = CpuContext::new(1);
    let complex = Tensor::from_slice(
        &[
            Complex64::new(1.0, -2.0),
            Complex64::new(-3.0, 4.0),
            Complex64::new(5.0, 0.5),
            Complex64::new(-7.0, -1.5),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let real = Tensor::from_slice(
        &[2.0_f64, -0.5, 3.0, 4.0],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let desc = ComplexScalePrimsDescriptor::PointwiseMul;
    let mut output = Tensor::<Complex64>::zeros(
        &[2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let plan = <CpuBackend as TensorComplexScalePrims<Complex64>>::plan(
        &mut ctx,
        &desc,
        &[complex.dims(), real.dims(), output.dims()],
    )
    .unwrap();
    <CpuBackend as TensorComplexScalePrims<Complex64>>::execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &complex,
        &real,
        Complex64::zero(),
        &mut output,
    )
    .unwrap();

    assert_close_slice_f64(
        output.buffer().as_slice().unwrap(),
        &[
            Complex64::new(2.0, -4.0),
            Complex64::new(1.5, -2.0),
            Complex64::new(15.0, 1.5),
            Complex64::new(-28.0, -6.0),
        ],
        1.0e-12,
    );
}

#[test]
fn cpu_complex_scale_phase1_executes_pointwise_mul_for_complex32() {
    let mut ctx = CpuContext::new(1);
    let complex = Tensor::from_slice(
        &[
            Complex32::new(1.0, -2.0),
            Complex32::new(-3.0, 4.0),
            Complex32::new(5.0, 0.5),
            Complex32::new(-7.0, -1.5),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let real = Tensor::from_slice(
        &[2.0_f32, -0.5, 3.0, 4.0],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let desc = ComplexScalePrimsDescriptor::PointwiseMul;
    let mut output = Tensor::<Complex32>::zeros(
        &[2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let plan = <CpuBackend as TensorComplexScalePrims<Complex32>>::plan(
        &mut ctx,
        &desc,
        &[complex.dims(), real.dims(), output.dims()],
    )
    .unwrap();
    <CpuBackend as TensorComplexScalePrims<Complex32>>::execute(
        &mut ctx,
        &plan,
        Complex32::new(1.0, 0.0),
        &complex,
        &real,
        Complex32::zero(),
        &mut output,
    )
    .unwrap();

    assert_close_slice_f32(
        output.buffer().as_slice().unwrap(),
        &[
            Complex32::new(2.0, -4.0),
            Complex32::new(1.5, -2.0),
            Complex32::new(15.0, 1.5),
            Complex32::new(-28.0, -6.0),
        ],
        1.0e-5,
    );
}
