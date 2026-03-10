use tenferro_algebra::Standard;
use tenferro_prims::{CpuBackend, CpuContext, TensorSemiringCore};

use num_complex::{Complex32, Complex64};

use super::{batched_gemm_via_prims, batched_gemm_with_semiring_core};

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
fn batched_gemm_via_prims_multiplies_real_col_major_matrices() {
    let a = vec![1.0_f64, 2.0, 3.0, 4.0];
    let b = vec![5.0_f64, 6.0, 7.0, 8.0];

    let c = batched_gemm_via_prims(&a, 2, 2, &b, 2).unwrap();

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
fn batched_gemm_via_prims_multiplies_complex_col_major_matrices() {
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

    let c = batched_gemm_via_prims(&a, 2, 2, &b, 2).unwrap();

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
fn batched_gemm_via_prims_multiplies_complex32_col_major_matrices() {
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

    let c = batched_gemm_via_prims(&a, 2, 2, &b, 2).unwrap();

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
