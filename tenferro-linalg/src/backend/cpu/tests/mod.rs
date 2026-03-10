use super::*;
use crate::LinalgScalar;
use tenferro_tensor::{MemoryOrder, Tensor};

/// Convert a slice of f64 pairs into scalar type T for test matrices.
trait TestScalar: LinalgScalar {
    fn from_f64(v: f64) -> Self;
}

impl TestScalar for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
}

impl TestScalar for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl TestScalar for num_complex::Complex64 {
    fn from_f64(v: f64) -> Self {
        Self::new(v, 0.0)
    }
}

impl TestScalar for num_complex::Complex32 {
    fn from_f64(v: f64) -> Self {
        Self::new(v as f32, 0.0)
    }
}

fn make<T: TestScalar>(data: &[f64], dims: &[usize]) -> Tensor<T> {
    let typed: Vec<T> = data.iter().map(|&v| T::from_f64(v)).collect();
    Tensor::from_slice(&typed, dims, MemoryOrder::ColumnMajor).unwrap()
}

macro_rules! cpu_backend_tests {
    ($mod_name:ident, $scalar:ty) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn solve() {
                let mut ctx = tenferro_prims::CpuContext::new(1);
                let a = make::<$scalar>(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
                let b = make::<$scalar>(&[4.0, 7.0], &[2, 1]);
                let x = <CpuTensorLinalgBackend as TensorLinalgBackend<$scalar>>::solve(
                    &mut ctx, &a, &b,
                )
                .unwrap();
                assert_eq!(x.dims(), &[2, 1]);
            }

            #[test]
            fn solve_accepts_vector_rhs() {
                let mut ctx = tenferro_prims::CpuContext::new(1);
                let a = make::<$scalar>(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
                let b = make::<$scalar>(&[4.0, 7.0], &[2]);
                let x = <CpuTensorLinalgBackend as TensorLinalgBackend<$scalar>>::solve(
                    &mut ctx, &a, &b,
                )
                .unwrap();
                assert_eq!(x.dims(), &[2]);
            }

            #[test]
            fn solve_rejects_scalar_rhs_without_panic() {
                let mut ctx = tenferro_prims::CpuContext::new(1);
                let a = make::<$scalar>(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
                let b = make::<$scalar>(&[4.0], &[]);
                assert!(
                    <CpuTensorLinalgBackend as TensorLinalgBackend<$scalar>>::solve(
                        &mut ctx, &a, &b
                    )
                    .is_err()
                );
            }

            #[test]
            fn solve_triangular() {
                let mut ctx = tenferro_prims::CpuContext::new(1);
                let a = make::<$scalar>(&[2.0, 0.0, 1.0, 3.0], &[2, 2]);
                let b = make::<$scalar>(&[5.0, 6.0], &[2, 1]);
                let x = <CpuTensorLinalgBackend as TensorLinalgBackend<$scalar>>::solve_triangular(
                    &mut ctx, &a, &b, true,
                )
                .unwrap();
                assert_eq!(x.dims(), &[2, 1]);
            }

            #[test]
            fn solve_triangular_accepts_vector_rhs() {
                let mut ctx = tenferro_prims::CpuContext::new(1);
                let a = make::<$scalar>(&[2.0, 0.0, 1.0, 3.0], &[2, 2]);
                let b = make::<$scalar>(&[5.0, 6.0], &[2]);
                let x = <CpuTensorLinalgBackend as TensorLinalgBackend<$scalar>>::solve_triangular(
                    &mut ctx, &a, &b, true,
                )
                .unwrap();
                assert_eq!(x.dims(), &[2]);
            }

            #[test]
            fn qr() {
                let mut ctx = tenferro_prims::CpuContext::new(1);
                let a = make::<$scalar>(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
                let result =
                    <CpuTensorLinalgBackend as TensorLinalgBackend<$scalar>>::qr(&mut ctx, &a)
                        .unwrap();
                assert_eq!(result.q.dims(), &[2, 2]);
                assert_eq!(result.r.dims(), &[2, 2]);
            }

            #[test]
            fn thin_svd() {
                let mut ctx = tenferro_prims::CpuContext::new(1);
                let a = make::<$scalar>(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
                let result = <CpuTensorLinalgBackend as TensorLinalgBackend<$scalar>>::thin_svd(
                    &mut ctx, &a,
                )
                .unwrap();
                assert_eq!(result.u.dims(), &[2, 2]);
                assert_eq!(result.s.dims(), &[2]);
                assert_eq!(result.vt.dims(), &[2, 2]);
            }

            #[test]
            fn lu_factor() {
                let mut ctx = tenferro_prims::CpuContext::new(1);
                let a = make::<$scalar>(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
                let result = <CpuTensorLinalgBackend as TensorLinalgBackend<$scalar>>::lu_factor(
                    &mut ctx, &a,
                )
                .unwrap();
                assert_eq!(result.l.dims(), &[2, 2]);
                assert_eq!(result.u.dims(), &[2, 2]);
                assert_eq!(result.pivots.len(), 2);
            }

            #[test]
            fn cholesky() {
                let mut ctx = tenferro_prims::CpuContext::new(1);
                // SPD (Hermitian positive-definite): [[4, 2], [2, 3]]
                let a = make::<$scalar>(&[4.0, 2.0, 2.0, 3.0], &[2, 2]);
                let l = <CpuTensorLinalgBackend as TensorLinalgBackend<$scalar>>::cholesky(
                    &mut ctx, &a,
                )
                .unwrap();
                assert_eq!(l.dims(), &[2, 2]);
            }

            #[test]
            fn eigen_sym() {
                let mut ctx = tenferro_prims::CpuContext::new(1);
                let a = make::<$scalar>(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
                let result = <CpuTensorLinalgBackend as TensorLinalgBackend<$scalar>>::eigen_sym(
                    &mut ctx, &a,
                )
                .unwrap();
                assert_eq!(result.values.dims(), &[2]);
                assert_eq!(result.vectors.dims(), &[2, 2]);
            }

            #[test]
            fn eig() {
                let mut ctx = tenferro_prims::CpuContext::new(1);
                let a = make::<$scalar>(&[1.0, 2.0, 0.0, 3.0], &[2, 2]);
                let result =
                    <CpuTensorLinalgBackend as TensorLinalgBackend<$scalar>>::eig(&mut ctx, &a)
                        .unwrap();
                assert_eq!(result.values.dims(), &[2]);
                assert_eq!(result.vectors.dims(), &[2, 2]);
            }
        }
    };
}

cpu_backend_tests!(f64_tests, f64);
cpu_backend_tests!(f32_tests, f32);
cpu_backend_tests!(complex64_tests, num_complex::Complex64);
cpu_backend_tests!(complex32_tests, num_complex::Complex32);

fn run_generic_backend_solve_smoke<B, T>()
where
    B: TensorLinalgBackend<T, Context = tenferro_prims::CpuContext>,
    T: TestScalar + CpuLinalgScalar,
{
    let mut ctx = tenferro_prims::CpuContext::new(1);
    let a = make::<T>(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
    let b = make::<T>(&[4.0, 7.0], &[2]);
    let x = B::solve(&mut ctx, &a, &b).unwrap();
    assert_eq!(x.dims(), &[2]);
}

fn run_generic_backend_qr_smoke<B, T>()
where
    B: TensorLinalgBackend<T, Context = tenferro_prims::CpuContext>,
    T: TestScalar + CpuLinalgScalar,
{
    let mut ctx = tenferro_prims::CpuContext::new(1);
    let a = make::<T>(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
    let qr = B::qr(&mut ctx, &a).unwrap();
    assert_eq!(qr.q.dims(), &[2, 2]);
    assert_eq!(qr.r.dims(), &[2, 2]);
}

#[test]
fn solve_is_generic_over_cpu_backend_and_scalar() {
    run_generic_backend_solve_smoke::<CpuTensorLinalgBackend, f64>();
    run_generic_backend_solve_smoke::<CpuTensorLinalgBackend, f32>();
    run_generic_backend_solve_smoke::<CpuTensorLinalgBackend, num_complex::Complex64>();
    run_generic_backend_solve_smoke::<CpuTensorLinalgBackend, num_complex::Complex32>();
}

#[test]
fn qr_is_generic_over_cpu_backend_and_scalar() {
    run_generic_backend_qr_smoke::<CpuTensorLinalgBackend, f64>();
    run_generic_backend_qr_smoke::<CpuTensorLinalgBackend, f32>();
    run_generic_backend_qr_smoke::<CpuTensorLinalgBackend, num_complex::Complex64>();
    run_generic_backend_qr_smoke::<CpuTensorLinalgBackend, num_complex::Complex32>();
}

#[test]
fn solve_slices_wrapper_uses_selected_cpu_backend() {
    let a = [2.0_f64, 1.0, 1.0, 3.0];
    let b = [4.0_f64, 7.0];
    let mut x = [0.0_f64; 2];
    super::solve_slices(&a, &b, 2, 1, &mut x).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-12);
    assert!((x[1] - 2.0).abs() < 1e-12);
}

#[test]
fn lu_slices_wrapper_uses_selected_cpu_backend() {
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let mut perm = [0usize; 2];
    let mut l = [0.0_f64; 4];
    let mut u = [0.0_f64; 4];
    super::lu_slices(&a, 2, 2, &mut perm, &mut l, &mut u).unwrap();
    assert_eq!(perm, [0, 1]);
    assert_eq!(l, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(u, [1.0, 0.0, 0.0, 1.0]);
}
