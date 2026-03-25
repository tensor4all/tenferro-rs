//! CPU tensor linalg backend.
//!
//! The actual provider implementation is selected at compile time via
//! `linalg-faer` or `linalg-lapack` features.

use std::any::TypeId;

use num_complex::{Complex32, Complex64};
use tenferro_linalg_prims::{KernelLinalgScalar, LapackEigScalar};

use super::tensor_api::TensorLinalgBackend;
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
    use crate::KernelLinalgScalar;
    use tenferro_device::Result;

    /// Slice-level CPU linalg operations dispatched to faer or LAPACK.
    ///
    /// This trait is internal and sealed by the `private` module.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Internal trait: not accessible outside the crate.
    /// ```
    pub trait CpuLinalgOps: KernelLinalgScalar + super::LapackEigScalar {
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

macro_rules! cast_slice {
    ($slice:expr, $from:ty, $to:ty) => {{
        unsafe { &*($slice as *const [$from] as *const [$to]) }
    }};
}

macro_rules! cast_slice_mut {
    ($slice:expr, $from:ty, $to:ty) => {{
        unsafe { &mut *($slice as *mut [$from] as *mut [$to]) }
    }};
}

macro_rules! dispatch_kernel_linalg_scalar_type {
    ($generic:ty, $concrete:ident, $body:block) => {{
        let tid = TypeId::of::<$generic>();
        if tid == TypeId::of::<f64>() {
            type $concrete = f64;
            $body
        } else if tid == TypeId::of::<f32>() {
            type $concrete = f32;
            $body
        } else if tid == TypeId::of::<Complex64>() {
            type $concrete = Complex64;
            $body
        } else if tid == TypeId::of::<Complex32>() {
            type $concrete = Complex32;
            $body
        } else {
            unreachable!("KernelLinalgScalar must be one of the standard linalg dtypes")
        }
    }};
}

pub(crate) fn solve_slices<T: KernelLinalgScalar>(
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
    x: &mut [T],
) -> Result<()> {
    dispatch_kernel_linalg_scalar_type!(T, Concrete, {
        <Concrete as private::CpuLinalgOps>::solve_slices(
            cast_slice!(a, T, Concrete),
            cast_slice!(b, T, Concrete),
            n,
            nrhs,
            cast_slice_mut!(x, T, Concrete),
        )
    })
}

pub(crate) fn solve_triangular_slices<T: KernelLinalgScalar>(
    a: &[T],
    b: &[T],
    n: usize,
    nrhs: usize,
    upper: bool,
    x: &mut [T],
) -> Result<()> {
    dispatch_kernel_linalg_scalar_type!(T, Concrete, {
        <Concrete as private::CpuLinalgOps>::solve_triangular_slices(
            cast_slice!(a, T, Concrete),
            cast_slice!(b, T, Concrete),
            n,
            nrhs,
            upper,
            cast_slice_mut!(x, T, Concrete),
        )
    })
}

pub(crate) fn thin_svd_slices<T: KernelLinalgScalar>(
    a: &[T],
    m: usize,
    n: usize,
    u: &mut [T],
    s: &mut [T::Real],
    vt: &mut [T],
) -> Result<()> {
    dispatch_kernel_linalg_scalar_type!(T, Concrete, {
        type ConcreteReal = <Concrete as crate::LinalgScalar>::Real;
        <Concrete as private::CpuLinalgOps>::thin_svd_slices(
            cast_slice!(a, T, Concrete),
            m,
            n,
            cast_slice_mut!(u, T, Concrete),
            cast_slice_mut!(s, T::Real, ConcreteReal),
            cast_slice_mut!(vt, T, Concrete),
        )
    })
}

pub(crate) fn qr_slices<T: KernelLinalgScalar>(
    a: &[T],
    m: usize,
    n: usize,
    q: &mut [T],
    r: &mut [T],
) -> Result<()> {
    dispatch_kernel_linalg_scalar_type!(T, Concrete, {
        <Concrete as private::CpuLinalgOps>::qr_slices(
            cast_slice!(a, T, Concrete),
            m,
            n,
            cast_slice_mut!(q, T, Concrete),
            cast_slice_mut!(r, T, Concrete),
        )
    })
}

pub(crate) fn lu_slices<T: KernelLinalgScalar>(
    a: &[T],
    m: usize,
    n: usize,
    perm: &mut [usize],
    l: &mut [T],
    u_out: &mut [T],
) -> Result<()> {
    dispatch_kernel_linalg_scalar_type!(T, Concrete, {
        <Concrete as private::CpuLinalgOps>::lu_slices(
            cast_slice!(a, T, Concrete),
            m,
            n,
            perm,
            cast_slice_mut!(l, T, Concrete),
            cast_slice_mut!(u_out, T, Concrete),
        )
    })
}

pub(crate) fn cholesky_slices<T: KernelLinalgScalar>(a: &[T], n: usize, l: &mut [T]) -> Result<()> {
    dispatch_kernel_linalg_scalar_type!(T, Concrete, {
        <Concrete as private::CpuLinalgOps>::cholesky_slices(
            cast_slice!(a, T, Concrete),
            n,
            cast_slice_mut!(l, T, Concrete),
        )
    })
}

pub(crate) fn eigen_sym_slices<T: KernelLinalgScalar>(
    a: &[T],
    n: usize,
    values: &mut [T::Real],
    vectors: &mut [T],
) -> Result<()> {
    dispatch_kernel_linalg_scalar_type!(T, Concrete, {
        type ConcreteReal = <Concrete as crate::LinalgScalar>::Real;
        <Concrete as private::CpuLinalgOps>::eigen_sym_slices(
            cast_slice!(a, T, Concrete),
            n,
            cast_slice_mut!(values, T::Real, ConcreteReal),
            cast_slice_mut!(vectors, T, Concrete),
        )
    })
}

pub(crate) fn eig_slices<T: KernelLinalgScalar>(
    a: &[T],
    n: usize,
    values_ri: &mut [T],
    vectors_ri: &mut [T],
) -> Result<()> {
    dispatch_kernel_linalg_scalar_type!(T, Concrete, {
        <Concrete as private::CpuLinalgOps>::eig_slices(
            cast_slice!(a, T, Concrete),
            n,
            cast_slice_mut!(values_ri, T, Concrete),
            cast_slice_mut!(vectors_ri, T, Concrete),
        )
    })
}

pub(crate) fn eig_buffer_sizes<T: KernelLinalgScalar>(n: usize) -> (usize, usize) {
    dispatch_kernel_linalg_scalar_type!(T, Concrete, {
        <Concrete as LapackEigScalar>::eig_buffer_sizes(n)
    })
}

pub(crate) fn eig_ri_to_complex<T: KernelLinalgScalar>(
    n: usize,
    val_ri: &[T],
    vec_ri: &[T],
    values_out: &mut [T::Complex],
    vectors_out: &mut [T::Complex],
) {
    dispatch_kernel_linalg_scalar_type!(T, Concrete, {
        type ConcreteComplex = <Concrete as crate::LinalgScalar>::Complex;
        <Concrete as LapackEigScalar>::eig_ri_to_complex(
            n,
            cast_slice!(val_ri, T, Concrete),
            cast_slice!(vec_ri, T, Concrete),
            cast_slice_mut!(values_out, T::Complex, ConcreteComplex),
            cast_slice_mut!(vectors_out, T::Complex, ConcreteComplex),
        )
    })
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
    T: KernelLinalgScalar,
{
    type Context = tenferro_prims::CpuContext;

    fn has_linalg_support(_op: super::tensor_api::LinalgCapabilityOp) -> bool {
        true
    }

    fn solve_ex(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
        b: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::SolveTensorExResult<T>> {
        super::cpu_tensor_impl::solve_ex(ctx, a, b)
    }

    fn solve(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
        b: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<tenferro_tensor::Tensor<T>> {
        cpu_impl::solve(ctx, a, b)
    }

    fn lu_solve(
        ctx: &mut Self::Context,
        factors: &tenferro_tensor::Tensor<T>,
        pivots: &tenferro_tensor::Tensor<i32>,
        b: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<tenferro_tensor::Tensor<T>> {
        super::cpu_tensor_impl::lu_solve(ctx, factors, pivots, b)
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

    fn svdvals(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<tenferro_tensor::Tensor<T::Real>> {
        Ok(cpu_impl::thin_svd(ctx, a)?.s)
    }

    fn lu_factor_ex(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::LuTensorExResult<T>> {
        super::cpu_tensor_impl::lu_factor_ex(ctx, a)
    }

    fn lu_factor(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::LuTensorResult<T>> {
        cpu_impl::lu_factor(ctx, a)
    }

    fn lu_factor_no_pivot(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::LuTensorResult<T>> {
        super::cpu_tensor_impl::lu_factor_no_pivot(ctx, a)
    }

    fn cholesky_ex(
        ctx: &mut Self::Context,
        a: &tenferro_tensor::Tensor<T>,
    ) -> tenferro_device::Result<super::tensor_api::CholeskyTensorExResult<T>> {
        super::cpu_tensor_impl::cholesky_ex(ctx, a)
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

#[cfg(all(test, feature = "linalg-faer"))]
mod tests;
