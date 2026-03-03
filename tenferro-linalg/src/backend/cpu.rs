//! CPU tensor linalg backend.
//!
//! The actual provider implementation is selected at compile time via
//! `linalg-faer` or `linalg-lapack` features.

use num_complex::{Complex32, Complex64};

use super::tensor_api::TensorLinalgBackend;
use super::tensor_context::TensorLinalgContextFor;
use super::LinalgBackend;
use tenferro_device::Result;

#[cfg(feature = "linalg-faer")]
use super::cpu_faer as cpu_impl;
#[cfg(feature = "linalg-lapack")]
use super::cpu_lapack as cpu_impl;

#[cfg(feature = "linalg-faer")]
type SelectedCpuSliceBackend = super::faer_backend::FaerBackend;
#[cfg(feature = "linalg-lapack")]
type SelectedCpuSliceBackend = super::cpu_lapack::LapackBackend;

mod private {
    use num_complex::{Complex32, Complex64};

    use super::{LinalgBackend, SelectedCpuSliceBackend};
    use crate::LinalgScalar;
    use tenferro_device::Result;

    pub trait CpuLinalgOps: LinalgScalar {
        fn solve_slices(
            a: &[Self],
            b: &[Self],
            n: usize,
            nrhs: usize,
            x: &mut [Self],
        ) -> Result<()>;
        fn solve_triangular_slices(
            a: &[Self],
            b: &[Self],
            n: usize,
            nrhs: usize,
            upper: bool,
            x: &mut [Self],
        ) -> Result<()>;
        fn thin_svd_slices(
            a: &[Self],
            m: usize,
            n: usize,
            u: &mut [Self],
            s: &mut [Self::Real],
            vt: &mut [Self],
        ) -> Result<()>;
        fn qr_slices(a: &[Self], m: usize, n: usize, q: &mut [Self], r: &mut [Self]) -> Result<()>;
        fn lu_slices(
            a: &[Self],
            m: usize,
            n: usize,
            perm: &mut [usize],
            l: &mut [Self],
            u_out: &mut [Self],
        ) -> Result<()>;
        fn cholesky_slices(a: &[Self], n: usize, l: &mut [Self]) -> Result<()>;
        fn eigen_sym_slices(
            a: &[Self],
            n: usize,
            values: &mut [Self::Real],
            vectors: &mut [Self],
        ) -> Result<()>;
        fn eig_slices(
            a: &[Self],
            n: usize,
            values_ri: &mut [Self],
            vectors_ri: &mut [Self],
        ) -> Result<()>;
    }

    macro_rules! impl_cpu_linalg_ops {
        ($ty:ty) => {
            impl CpuLinalgOps for $ty {
                fn solve_slices(
                    a: &[Self],
                    b: &[Self],
                    n: usize,
                    nrhs: usize,
                    x: &mut [Self],
                ) -> Result<()> {
                    SelectedCpuSliceBackend::new().solve(a, b, n, nrhs, x)
                }

                fn solve_triangular_slices(
                    a: &[Self],
                    b: &[Self],
                    n: usize,
                    nrhs: usize,
                    upper: bool,
                    x: &mut [Self],
                ) -> Result<()> {
                    SelectedCpuSliceBackend::new().solve_triangular(a, b, n, nrhs, upper, x)
                }

                fn thin_svd_slices(
                    a: &[Self],
                    m: usize,
                    n: usize,
                    u: &mut [Self],
                    s: &mut [Self::Real],
                    vt: &mut [Self],
                ) -> Result<()> {
                    SelectedCpuSliceBackend::new().thin_svd(a, m, n, u, s, vt)
                }

                fn qr_slices(
                    a: &[Self],
                    m: usize,
                    n: usize,
                    q: &mut [Self],
                    r: &mut [Self],
                ) -> Result<()> {
                    SelectedCpuSliceBackend::new().qr(a, m, n, q, r)
                }

                fn lu_slices(
                    a: &[Self],
                    m: usize,
                    n: usize,
                    perm: &mut [usize],
                    l: &mut [Self],
                    u_out: &mut [Self],
                ) -> Result<()> {
                    SelectedCpuSliceBackend::new().lu(a, m, n, perm, l, u_out)
                }

                fn cholesky_slices(a: &[Self], n: usize, l: &mut [Self]) -> Result<()> {
                    SelectedCpuSliceBackend::new().cholesky(a, n, l)
                }

                fn eigen_sym_slices(
                    a: &[Self],
                    n: usize,
                    values: &mut [Self::Real],
                    vectors: &mut [Self],
                ) -> Result<()> {
                    SelectedCpuSliceBackend::new().eigen_sym(a, n, values, vectors)
                }

                fn eig_slices(
                    a: &[Self],
                    n: usize,
                    values_ri: &mut [Self],
                    vectors_ri: &mut [Self],
                ) -> Result<()> {
                    SelectedCpuSliceBackend::new().eig_general(a, n, values_ri, vectors_ri)
                }
            }
        };
    }

    impl_cpu_linalg_ops!(f64);
    impl_cpu_linalg_ops!(f32);
    impl_cpu_linalg_ops!(Complex64);
    impl_cpu_linalg_ops!(Complex32);
}

/// Scalar types supported by the CPU linalg provider selected at build time.
pub trait CpuLinalgScalar: private::CpuLinalgOps {}

impl CpuLinalgScalar for f64 {}
impl CpuLinalgScalar for f32 {}
impl CpuLinalgScalar for Complex64 {}
impl CpuLinalgScalar for Complex32 {}

pub(crate) fn solve_slices<T: CpuLinalgScalar>(
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
    x: &mut [T],
) -> Result<()> {
    <T as private::CpuLinalgOps>::solve_slices(a, b, n, nrhs, x)
}

pub(crate) fn solve_triangular_slices<T: CpuLinalgScalar>(
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
    upper: bool,
    x: &mut [T],
) -> Result<()> {
    <T as private::CpuLinalgOps>::solve_triangular_slices(a, b, n, nrhs, upper, x)
}

pub(crate) fn thin_svd_slices<T: CpuLinalgScalar>(
    a: &[T],
    m: usize,
    n: usize,
    u: &mut [T],
    s: &mut [T::Real],
    vt: &mut [T],
) -> Result<()> {
    <T as private::CpuLinalgOps>::thin_svd_slices(a, m, n, u, s, vt)
}

pub(crate) fn qr_slices<T: CpuLinalgScalar>(
    a: &[T],
    m: usize,
    n: usize,
    q: &mut [T],
    r: &mut [T],
) -> Result<()> {
    <T as private::CpuLinalgOps>::qr_slices(a, m, n, q, r)
}

pub(crate) fn lu_slices<T: CpuLinalgScalar>(
    a: &[T],
    m: usize,
    n: usize,
    perm: &mut [usize],
    l: &mut [T],
    u_out: &mut [T],
) -> Result<()> {
    <T as private::CpuLinalgOps>::lu_slices(a, m, n, perm, l, u_out)
}

#[cfg(feature = "linalg-faer")]
pub(crate) fn cholesky_slices<T: CpuLinalgScalar>(a: &[T], n: usize, l: &mut [T]) -> Result<()> {
    <T as private::CpuLinalgOps>::cholesky_slices(a, n, l)
}

#[cfg(feature = "linalg-faer")]
pub(crate) fn eigen_sym_slices<T: CpuLinalgScalar>(
    a: &[T],
    n: usize,
    values: &mut [T::Real],
    vectors: &mut [T],
) -> Result<()> {
    <T as private::CpuLinalgOps>::eigen_sym_slices(a, n, values, vectors)
}

#[cfg(feature = "linalg-faer")]
pub(crate) fn eig_slices<T: CpuLinalgScalar>(
    a: &[T],
    n: usize,
    values_ri: &mut [T],
    vectors_ri: &mut [T],
) -> Result<()> {
    <T as private::CpuLinalgOps>::eig_slices(a, n, values_ri, vectors_ri)
}

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

impl<T> TensorLinalgBackend<T> for CpuTensorLinalgBackend
where
    T: CpuLinalgScalar,
{
    type Context = tenferro_prims::CpuContext;

    fn solve(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
        b: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<tenferro_tensor::Tensor<T>> {
        cpu_impl::solve(ctx, a, b)
    }

    fn solve_triangular(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
        b: &tenferro_tensor::Tensor<T>,
        upper: bool,
    ) -> tenferro_device::Result<tenferro_tensor::Tensor<T>> {
        cpu_impl::solve_triangular(ctx, a, b, upper)
    }

    fn qr(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::QrTensorResult<T>> {
        cpu_impl::qr(ctx, a)
    }

    fn thin_svd(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::SvdTensorResult<T>> {
        cpu_impl::thin_svd(ctx, a)
    }

    fn lu_factor(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::LuTensorResult<T>> {
        cpu_impl::lu_factor(ctx, a)
    }

    fn cholesky(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<tenferro_tensor::Tensor<T>> {
        cpu_impl::cholesky(ctx, a)
    }

    fn eigen_sym(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::EigenTensorResult<T>> {
        cpu_impl::eigen_sym(ctx, a)
    }

    fn eig(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::EigTensorResult<T>> {
        cpu_impl::eig(ctx, a)
    }
}

impl<T> TensorLinalgContextFor<T> for tenferro_prims::CpuContext
where
    T: CpuLinalgScalar,
{
    type Backend = CpuTensorLinalgBackend;
}

#[cfg(all(test, feature = "linalg-faer"))]
mod tests {
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
                    let x = CpuTensorLinalgBackend::solve(&mut ctx, &a, &b).unwrap();
                    assert_eq!(x.dims(), &[2, 1]);
                }

                #[test]
                fn solve_accepts_vector_rhs() {
                    let mut ctx = tenferro_prims::CpuContext::new(1);
                    let a = make::<$scalar>(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
                    let b = make::<$scalar>(&[4.0, 7.0], &[2]);
                    let x = CpuTensorLinalgBackend::solve(&mut ctx, &a, &b).unwrap();
                    assert_eq!(x.dims(), &[2]);
                }

                #[test]
                fn solve_rejects_scalar_rhs_without_panic() {
                    let mut ctx = tenferro_prims::CpuContext::new(1);
                    let a = make::<$scalar>(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
                    let b = make::<$scalar>(&[4.0], &[]);
                    assert!(CpuTensorLinalgBackend::solve(&mut ctx, &a, &b).is_err());
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
                fn solve_triangular_accepts_vector_rhs() {
                    let mut ctx = tenferro_prims::CpuContext::new(1);
                    let a = make::<$scalar>(&[2.0, 0.0, 1.0, 3.0], &[2, 2]);
                    let b = make::<$scalar>(&[5.0, 6.0], &[2]);
                    let x =
                        CpuTensorLinalgBackend::solve_triangular(&mut ctx, &a, &b, true).unwrap();
                    assert_eq!(x.dims(), &[2]);
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
}
