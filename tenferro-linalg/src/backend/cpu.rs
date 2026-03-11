//! CPU tensor linalg backend.
//!
//! The actual provider implementation is selected at compile time via
//! `linalg-faer` or `linalg-lapack` features.

use num_complex::{Complex32, Complex64};
use tenferro_linalg_prims::LapackEigScalar;

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
pub trait CpuLinalgScalar: private::CpuLinalgOps + LapackEigScalar {}

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

pub(crate) fn cholesky_slices<T: CpuLinalgScalar>(a: &[T], n: usize, l: &mut [T]) -> Result<()> {
    <T as private::CpuLinalgOps>::cholesky_slices(a, n, l)
}

pub(crate) fn eigen_sym_slices<T: CpuLinalgScalar>(
    a: &[T],
    n: usize,
    values: &mut [T::Real],
    vectors: &mut [T],
) -> Result<()> {
    <T as private::CpuLinalgOps>::eigen_sym_slices(a, n, values, vectors)
}

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

    fn has_linalg_support(_op: super::tensor_api::LinalgCapabilityOp) -> bool {
        true
    }

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
mod tests;
