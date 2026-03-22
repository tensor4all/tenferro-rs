use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tenferro_algebra::Scalar;
use tenferro_device::LogicalMemorySpace;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{
    ComplexRealPrimsDescriptor, ComplexRealUnaryOp, CpuBackend, CpuContext, TensorComplexRealPrims,
};

fn assert_close_slice_f64(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!((lhs - rhs).abs() <= tol, "got {lhs}, expected {rhs}");
    }
}

fn assert_close_slice_f32(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (lhs, rhs) in actual.iter().zip(expected.iter()) {
        assert!((lhs - rhs).abs() <= tol, "got {lhs}, expected {rhs}");
    }
}

fn execute_abs_real_cpu<C>(values: &[C], dims: &[usize]) -> Tensor<C::Real>
where
    C: Scalar + num_complex::ComplexFloat,
    C::Real: Scalar + Zero + One,
    CpuBackend: TensorComplexRealPrims<C, Real = C::Real, Context = CpuContext>,
{
    let mut ctx = CpuContext::new(1);
    let input = Tensor::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap();
    let desc = ComplexRealPrimsDescriptor::PointwiseUnary {
        op: ComplexRealUnaryOp::Abs,
    };
    let mut output = Tensor::<C::Real>::zeros(
        dims,
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let plan = <CpuBackend as TensorComplexRealPrims<C>>::plan(
        &mut ctx,
        &desc,
        &[input.dims(), output.dims()],
    )
    .unwrap();
    <CpuBackend as TensorComplexRealPrims<C>>::execute(
        &mut ctx,
        &plan,
        <C::Real as One>::one(),
        &[&input],
        <C::Real as Zero>::zero(),
        &mut output,
    )
    .unwrap();
    output
}

#[test]
fn cpu_complex_real_phase1_supports_abs_for_complex32_and_complex64() {
    let desc = ComplexRealPrimsDescriptor::PointwiseUnary {
        op: ComplexRealUnaryOp::Abs,
    };

    assert!(
        <CpuBackend as TensorComplexRealPrims<Complex32>>::has_complex_real_support(desc.clone())
    );
    assert!(<CpuBackend as TensorComplexRealPrims<Complex64>>::has_complex_real_support(desc));
}

#[test]
fn cpu_complex_real_phase1_supports_real_and_imag_for_complex32_and_complex64() {
    let real_desc = ComplexRealPrimsDescriptor::PointwiseUnary {
        op: ComplexRealUnaryOp::Real,
    };
    let imag_desc = ComplexRealPrimsDescriptor::PointwiseUnary {
        op: ComplexRealUnaryOp::Imag,
    };

    assert!(
        <CpuBackend as TensorComplexRealPrims<Complex32>>::has_complex_real_support(
            real_desc.clone()
        )
    );
    assert!(
        <CpuBackend as TensorComplexRealPrims<Complex32>>::has_complex_real_support(
            imag_desc.clone()
        )
    );
    assert!(<CpuBackend as TensorComplexRealPrims<Complex64>>::has_complex_real_support(real_desc));
    assert!(<CpuBackend as TensorComplexRealPrims<Complex64>>::has_complex_real_support(imag_desc));
}

#[test]
fn cpu_complex_real_phase1_executes_abs_for_complex64() {
    let output = execute_abs_real_cpu::<Complex64>(
        &[
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 12.0),
            Complex64::new(8.0, 15.0),
            Complex64::new(7.0, 24.0),
        ],
        &[2, 2],
    );
    let actual = output.buffer().as_slice().unwrap();
    assert_close_slice_f64(actual, &[5.0, 13.0, 17.0, 25.0], 1.0e-12);
}

#[test]
fn cpu_complex_real_phase1_executes_abs_for_complex32() {
    let output = execute_abs_real_cpu::<Complex32>(
        &[
            Complex32::new(3.0, 4.0),
            Complex32::new(5.0, 12.0),
            Complex32::new(8.0, 15.0),
            Complex32::new(7.0, 24.0),
        ],
        &[2, 2],
    );
    let actual = output.buffer().as_slice().unwrap();
    assert_close_slice_f32(actual, &[5.0, 13.0, 17.0, 25.0], 1.0e-5);
}

#[test]
fn cpu_complex_real_phase1_executes_real_for_complex64() {
    let mut ctx = CpuContext::new(1);
    let input = Tensor::from_slice(
        &[
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 12.0),
            Complex64::new(8.0, 15.0),
            Complex64::new(7.0, 24.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let desc = ComplexRealPrimsDescriptor::PointwiseUnary {
        op: ComplexRealUnaryOp::Real,
    };
    let mut output = Tensor::<f64>::zeros(
        &[2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let plan = <CpuBackend as TensorComplexRealPrims<Complex64>>::plan(
        &mut ctx,
        &desc,
        &[input.dims(), output.dims()],
    )
    .unwrap();
    <CpuBackend as TensorComplexRealPrims<Complex64>>::execute(
        &mut ctx,
        &plan,
        1.0,
        &[&input],
        0.0,
        &mut output,
    )
    .unwrap();
    assert_close_slice_f64(
        output.buffer().as_slice().unwrap(),
        &[3.0, 5.0, 8.0, 7.0],
        1.0e-12,
    );
}

#[test]
fn cpu_complex_real_phase1_executes_imag_for_complex32() {
    let mut ctx = CpuContext::new(1);
    let input = Tensor::from_slice(
        &[
            Complex32::new(3.0, 4.0),
            Complex32::new(5.0, 12.0),
            Complex32::new(8.0, 15.0),
            Complex32::new(7.0, 24.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let desc = ComplexRealPrimsDescriptor::PointwiseUnary {
        op: ComplexRealUnaryOp::Imag,
    };
    let mut output = Tensor::<f32>::zeros(
        &[2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let plan = <CpuBackend as TensorComplexRealPrims<Complex32>>::plan(
        &mut ctx,
        &desc,
        &[input.dims(), output.dims()],
    )
    .unwrap();
    <CpuBackend as TensorComplexRealPrims<Complex32>>::execute(
        &mut ctx,
        &plan,
        1.0,
        &[&input],
        0.0,
        &mut output,
    )
    .unwrap();
    assert_close_slice_f32(
        output.buffer().as_slice().unwrap(),
        &[4.0, 12.0, 15.0, 24.0],
        1.0e-5,
    );
}
