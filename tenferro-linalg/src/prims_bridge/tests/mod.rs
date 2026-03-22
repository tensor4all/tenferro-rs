use tenferro_algebra::Standard;
use tenferro_prims::{CpuBackend, CpuContext, TensorSemiringCore};

use num_complex::{Complex32, Complex64};

use super::{
    batched_gemm_with_semiring_context, batched_gemm_with_semiring_core,
    complex_real_reduce_keep_axes, complex_real_unary_same_shape, scalar_where_same_shape,
};

struct SemiringCoreOnlyCpuBackend;

impl<T> TensorSemiringCore<Standard<T>> for SemiringCoreOnlyCpuBackend
where
    T: crate::LinalgScalar,
    CpuBackend: TensorSemiringCore<Standard<T>, Context = CpuContext>,
{
    type Plan = <CpuBackend as TensorSemiringCore<Standard<T>>>::Plan;
    type Context = CpuContext;

    fn plan(
        ctx: &mut Self::Context,
        desc: &tenferro_prims::SemiringCoreDescriptor,
        shapes: &[&[usize]],
    ) -> tenferro_device::Result<Self::Plan> {
        <CpuBackend as TensorSemiringCore<Standard<T>>>::plan(ctx, desc, shapes)
    }

    fn execute(
        ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: T,
        inputs: &[&tenferro_tensor::Tensor<T>],
        beta: T,
        output: &mut tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<()> {
        <CpuBackend as TensorSemiringCore<Standard<T>>>::execute(
            ctx, plan, alpha, inputs, beta, output,
        )
    }
}

#[test]
fn batched_gemm_with_semiring_context_multiplies_real_col_major_matrices() {
    let mut ctx = CpuContext::new(1);
    let a = vec![1.0_f64, 2.0, 3.0, 4.0];
    let b = vec![5.0_f64, 6.0, 7.0, 8.0];

    let c = batched_gemm_with_semiring_context(&mut ctx, &a, 2, 2, &b, 2).unwrap();

    assert_eq!(c, vec![23.0, 34.0, 31.0, 46.0]);
}

#[test]
fn generic_batched_gemm_bridge_accepts_semiring_core_only_backend() {
    let mut ctx = CpuContext::new(1);
    let a = vec![1.0_f64, 2.0, 3.0, 4.0];
    let b = vec![5.0_f64, 6.0, 7.0, 8.0];

    let c = batched_gemm_with_semiring_core::<f64, SemiringCoreOnlyCpuBackend>(
        &mut ctx, &a, 2, 2, &b, 2,
    )
    .unwrap();

    assert_eq!(c, vec![23.0, 34.0, 31.0, 46.0]);
}

#[test]
fn batched_gemm_with_semiring_context_multiplies_complex_col_major_matrices() {
    let mut ctx = CpuContext::new(1);
    let a = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    let b = vec![
        Complex64::new(5.0, 0.0),
        Complex64::new(6.0, 0.0),
        Complex64::new(7.0, 0.0),
        Complex64::new(8.0, 0.0),
    ];

    let c = batched_gemm_with_semiring_context(&mut ctx, &a, 2, 2, &b, 2).unwrap();

    assert_eq!(
        c,
        vec![
            Complex64::new(23.0, 0.0),
            Complex64::new(34.0, 0.0),
            Complex64::new(31.0, 0.0),
            Complex64::new(46.0, 0.0),
        ]
    );
}

#[test]
fn batched_gemm_with_semiring_context_multiplies_complex32_col_major_matrices() {
    let mut ctx = CpuContext::new(1);
    let a = vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(4.0, 0.0),
    ];
    let b = vec![
        Complex32::new(5.0, 0.0),
        Complex32::new(6.0, 0.0),
        Complex32::new(7.0, 0.0),
        Complex32::new(8.0, 0.0),
    ];

    let c = batched_gemm_with_semiring_context(&mut ctx, &a, 2, 2, &b, 2).unwrap();

    assert_eq!(
        c,
        vec![
            Complex32::new(23.0, 0.0),
            Complex32::new(34.0, 0.0),
            Complex32::new(31.0, 0.0),
            Complex32::new(46.0, 0.0),
        ]
    );
}

#[test]
fn complex_real_unary_same_shape_abs_materializes_real_tensor_on_cpu() {
    let mut ctx = CpuContext::new(1);
    let input = tenferro_tensor::Tensor::from_slice(
        &[
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 12.0),
            Complex64::new(8.0, 15.0),
            Complex64::new(7.0, 24.0),
        ],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let output =
        complex_real_unary_same_shape(&mut ctx, &input, tenferro_prims::ComplexRealUnaryOp::Abs)
            .unwrap();

    assert_eq!(output.dims(), &[2, 2]);
    assert_eq!(
        output.buffer().as_slice().unwrap(),
        &[5.0, 13.0, 17.0, 25.0]
    );
}

#[test]
fn complex_real_reduce_keep_axes_sums_and_maxes_abs_values_on_cpu() {
    let mut ctx = CpuContext::new(1);
    let input = tenferro_tensor::Tensor::from_slice(
        &[
            Complex32::new(3.0, 4.0),
            Complex32::new(8.0, 15.0),
            Complex32::new(5.0, 12.0),
            Complex32::new(7.0, 24.0),
        ],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let column_sums = complex_real_reduce_keep_axes(
        &mut ctx,
        &input,
        tenferro_prims::ComplexRealUnaryOp::Abs,
        &[1],
        tenferro_prims::ScalarReductionOp::Sum,
    )
    .unwrap();
    let global_max = complex_real_reduce_keep_axes(
        &mut ctx,
        &input,
        tenferro_prims::ComplexRealUnaryOp::Abs,
        &[],
        tenferro_prims::ScalarReductionOp::Max,
    )
    .unwrap();

    assert_eq!(column_sums.buffer().as_slice().unwrap(), &[22.0, 38.0]);
    assert_eq!(global_max.buffer().as_slice().unwrap(), &[25.0]);
}

#[test]
fn scalar_where_same_shape_selects_by_numeric_mask_on_cpu() {
    let mut ctx = CpuContext::new(1);
    let mask = tenferro_tensor::Tensor::from_slice(
        &[1.0_f64, 0.0, -2.0, 0.0],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let on_true = tenferro_tensor::Tensor::from_slice(
        &[10.0_f64, 20.0, 30.0, 40.0],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let on_false = tenferro_tensor::Tensor::from_slice(
        &[-1.0_f64, -2.0, -3.0, -4.0],
        &[2, 2],
        tenferro_tensor::MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let output = scalar_where_same_shape(&mut ctx, &mask, &on_true, &on_false).unwrap();

    assert_eq!(
        output.buffer().as_slice().unwrap(),
        &[10.0, -2.0, 30.0, -4.0]
    );
}
