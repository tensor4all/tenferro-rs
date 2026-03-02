//! CPU tensor linalg backend.
//!
//! The actual provider implementation is selected at compile time via
//! `linalg-faer` or `linalg-lapack` features.

use super::tensor_api::TensorLinalgBackend;
use super::tensor_context::TensorLinalgContextFor;
use crate::LinalgScalar;

#[cfg(feature = "linalg-faer")]
use super::cpu_faer;

/// Marker type for the CPU tensor linalg backend.
///
/// The backend type provides the trait implementation, while
/// [`tenferro_prims::CpuContext`] owns execution state (thread pool,
/// plan cache, etc.).
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::CpuTensorLinalgBackend;
///
/// let _backend = CpuTensorLinalgBackend;
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuTensorLinalgBackend;

#[cfg(feature = "linalg-faer")]
impl<T> TensorLinalgBackend<T> for CpuTensorLinalgBackend
where
    T: LinalgScalar,
    crate::backend::faer_backend::FaerBackend: crate::backend::LinalgBackend<T, Real = T::Real>,
{
    type Context = tenferro_prims::CpuContext;

    fn solve(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
        b: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<tenferro_tensor::Tensor<T>> {
        cpu_faer::solve(ctx, a, b)
    }

    fn solve_triangular(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
        b: &tenferro_tensor::Tensor<T>,
        upper: bool,
    ) -> tenferro_device::Result<tenferro_tensor::Tensor<T>> {
        cpu_faer::solve_triangular(ctx, a, b, upper)
    }

    fn qr(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::QrTensorResult<T>> {
        cpu_faer::qr(ctx, a)
    }

    fn thin_svd(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::SvdTensorResult<T>> {
        cpu_faer::thin_svd(ctx, a)
    }

    fn lu_factor(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::LuTensorResult<T>> {
        cpu_faer::lu_factor(ctx, a)
    }

    fn cholesky(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<tenferro_tensor::Tensor<T>> {
        cpu_faer::cholesky(ctx, a)
    }

    fn eigen_sym(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::EigenTensorResult<T>> {
        cpu_faer::eigen_sym(ctx, a)
    }

    fn eig(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::EigTensorResult<T>> {
        cpu_faer::eig(ctx, a)
    }
}

#[cfg(feature = "linalg-faer")]
impl<T> TensorLinalgContextFor<T> for tenferro_prims::CpuContext
where
    T: LinalgScalar,
    crate::backend::faer_backend::FaerBackend: crate::backend::LinalgBackend<T, Real = T::Real>,
{
    type Backend = CpuTensorLinalgBackend;
}

#[cfg(test)]
mod tests {
    use super::*;
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
                    let x = CpuTensorLinalgBackend::solve(&mut ctx, &a, &b).unwrap();
                    assert_eq!(x.dims(), &[2, 1]);
                }

                #[test]
                fn solve_triangular() {
                    let mut ctx = tenferro_prims::CpuContext::new(1);
                    let a = make::<$scalar>(&[2.0, 0.0, 1.0, 3.0], &[2, 2]);
                    let b = make::<$scalar>(&[5.0, 6.0], &[2, 1]);
                    let x =
                        CpuTensorLinalgBackend::solve_triangular(&mut ctx, &a, &b, true).unwrap();
                    assert_eq!(x.dims(), &[2, 1]);
                }

                #[test]
                fn qr() {
                    let mut ctx = tenferro_prims::CpuContext::new(1);
                    let a = make::<$scalar>(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
                    let result = CpuTensorLinalgBackend::qr(&mut ctx, &a).unwrap();
                    assert_eq!(result.q.dims(), &[2, 2]);
                    assert_eq!(result.r.dims(), &[2, 2]);
                }

                #[test]
                fn thin_svd() {
                    let mut ctx = tenferro_prims::CpuContext::new(1);
                    let a = make::<$scalar>(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
                    let result = CpuTensorLinalgBackend::thin_svd(&mut ctx, &a).unwrap();
                    assert_eq!(result.u.dims(), &[2, 2]);
                    assert_eq!(result.s.dims(), &[2]);
                    assert_eq!(result.vt.dims(), &[2, 2]);
                }

                #[test]
                fn lu_factor() {
                    let mut ctx = tenferro_prims::CpuContext::new(1);
                    let a = make::<$scalar>(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
                    let result = CpuTensorLinalgBackend::lu_factor(&mut ctx, &a).unwrap();
                    assert_eq!(result.l.dims(), &[2, 2]);
                    assert_eq!(result.u.dims(), &[2, 2]);
                    assert_eq!(result.pivots.len(), 2);
                }

                #[test]
                fn cholesky() {
                    let mut ctx = tenferro_prims::CpuContext::new(1);
                    // SPD (Hermitian positive-definite): [[4, 2], [2, 3]]
                    let a = make::<$scalar>(&[4.0, 2.0, 2.0, 3.0], &[2, 2]);
                    let l = CpuTensorLinalgBackend::cholesky(&mut ctx, &a).unwrap();
                    assert_eq!(l.dims(), &[2, 2]);
                }

                #[test]
                fn eigen_sym() {
                    let mut ctx = tenferro_prims::CpuContext::new(1);
                    let a = make::<$scalar>(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
                    let result = CpuTensorLinalgBackend::eigen_sym(&mut ctx, &a).unwrap();
                    assert_eq!(result.values.dims(), &[2]);
                    assert_eq!(result.vectors.dims(), &[2, 2]);
                }

                #[test]
                fn eig() {
                    let mut ctx = tenferro_prims::CpuContext::new(1);
                    let a = make::<$scalar>(&[1.0, 2.0, 0.0, 3.0], &[2, 2]);
                    let result = CpuTensorLinalgBackend::eig(&mut ctx, &a).unwrap();
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
}
