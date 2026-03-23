//! Backend-facing linalg kernel contracts for the tenferro workspace.
//!
//! This crate holds the low-level tensor linalg protocol that device backends
//! implement. High-level composite APIs remain in `tenferro-linalg`.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_linalg_prims::{TensorLinalgPrims, QrTensorResult};
//! use tenferro_tensor::Tensor;
//!
//! fn accepts_backend<B: TensorLinalgPrims<f64>>() {}
//! let _: Option<QrTensorResult<f64>> = None;
//! let _: Option<Tensor<f64>> = None;
//! ```

#[cfg(all(feature = "provider-src", not(feature = "linalg-lapack")))]
compile_error!("provider-src requires linalg-lapack");
#[cfg(all(feature = "provider-inject", not(feature = "linalg-lapack")))]
compile_error!("provider-inject requires linalg-lapack");
#[cfg(all(feature = "linalg-faer", feature = "linalg-lapack"))]
compile_error!(
    "Features `linalg-faer` and `linalg-lapack` are mutually exclusive. Enable exactly one."
);
#[cfg(not(any(feature = "linalg-faer", feature = "linalg-lapack")))]
compile_error!("No CPU linalg provider selected. Enable `linalg-faer` or `linalg-lapack`.");
#[cfg(all(
    any(
        feature = "src-openblas",
        feature = "src-netlib",
        feature = "src-accelerate",
        feature = "src-r",
        feature = "src-intel-mkl-dynamic-sequential",
        feature = "src-intel-mkl-dynamic-parallel",
        feature = "src-intel-mkl-static-sequential",
        feature = "src-intel-mkl-static-parallel"
    ),
    not(feature = "linalg-lapack")
))]
compile_error!("src-* features require linalg-lapack and provider-src");

#[cfg(feature = "linalg-lapack")]
const _: () = {
    let provider_count =
        (cfg!(feature = "provider-src") as usize) + (cfg!(feature = "provider-inject") as usize);
    assert!(
        provider_count == 1,
        "linalg-lapack requires exactly one provider: provider-src or provider-inject"
    );

    let src_count = (cfg!(feature = "src-openblas") as usize)
        + (cfg!(feature = "src-netlib") as usize)
        + (cfg!(feature = "src-accelerate") as usize)
        + (cfg!(feature = "src-r") as usize)
        + (cfg!(feature = "src-intel-mkl-dynamic-sequential") as usize)
        + (cfg!(feature = "src-intel-mkl-dynamic-parallel") as usize)
        + (cfg!(feature = "src-intel-mkl-static-sequential") as usize)
        + (cfg!(feature = "src-intel-mkl-static-parallel") as usize);

    if cfg!(feature = "provider-src") {
        assert!(
            src_count == 1,
            "provider-src requires exactly one src-* feature"
        );
    }
    if cfg!(feature = "provider-inject") {
        assert!(src_count == 0, "provider-inject forbids src-* features");
    }
};

#[cfg(feature = "provider-src")]
extern crate blas_src as _;
#[cfg(feature = "provider-src")]
extern crate cblas_src as _;
#[cfg(feature = "provider-src")]
extern crate lapack_src as _;

#[cfg(feature = "provider-inject")]
extern crate cblas_inject as _;
#[cfg(feature = "provider-inject")]
extern crate lapack_inject as _;

use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use tenferro_algebra::Scalar;
use tenferro_device::Result;
use tenferro_tensor::Tensor;

pub mod backend;

/// Scalar types supported by linalg kernel contracts.
///
/// # Examples
///
/// ```
/// use tenferro_linalg_prims::LinalgScalar;
///
/// fn needs_linalg_scalar<T: LinalgScalar>(x: T) -> T { x }
/// assert_eq!(needs_linalg_scalar(1.0_f64), 1.0);
/// ```
pub trait LinalgScalar:
    Scalar
    + std::ops::Sub<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::Div<Output = Self>
    + num_traits::NumCast
    + std::fmt::Debug
    + 'static
{
    type Real: LinalgScalar<Real = Self::Real, Complex = Self::Complex> + num_traits::Float;
    type Complex: LinalgScalar<Real = Self::Real, Complex = Self::Complex>;

    /// Return the scalar magnitude in the associated real field.
    fn abs_real(&self) -> Self::Real;
    /// Return a reasonable machine epsilon for the associated real field.
    fn real_epsilon() -> Self::Real;
    /// Return the algebraic conjugate.
    fn conj(&self) -> Self;
    /// Build a scalar from explicit real/imaginary parts.
    fn from_parts(real: Self::Real, imag: Self::Real) -> Self;
    /// Build a scalar from the associated real field.
    fn from_real(real: Self::Real) -> Self {
        Self::from_parts(real, Self::Real::zero())
    }
    /// Return the real part in the associated real field.
    fn real_part(&self) -> Self::Real;
    /// Return the imaginary part in the associated real field.
    fn imag_part(&self) -> Self::Real;
}

/// Scalar types with concrete backend kernel support in the current workspace.
///
/// This marker keeps public/high-level linalg bounds generic over backends
/// without leaking provider-specific names such as `Cpu*` into higher layers.
///
/// # Examples
///
/// ```
/// use tenferro_linalg_prims::KernelLinalgScalar;
///
/// fn needs_kernel_scalar<T: KernelLinalgScalar>(x: T) -> T { x }
/// assert_eq!(needs_kernel_scalar(1.0_f64), 1.0);
/// ```
pub trait KernelLinalgScalar: LinalgScalar {}

/// LAPACK-oriented eigen helper contract for CPU eigendecomposition paths.
///
/// This trait is intentionally narrower than [`LinalgScalar`]. It exists so
/// CPU eigensolver glue can request the real/imag buffer conversion helpers it
/// needs without forcing every backend-generic scalar contract to carry LAPACK
/// details.
///
/// # Examples
///
/// ```
/// use tenferro_linalg_prims::LapackEigScalar;
///
/// let (vals, vecs) = <f64 as LapackEigScalar>::eig_buffer_sizes(2);
/// assert_eq!((vals, vecs), (4, 8));
/// ```
pub trait LapackEigScalar: LinalgScalar {
    /// Return the temporary value/vector buffer sizes used by the CPU eig path.
    fn eig_buffer_sizes(n: usize) -> (usize, usize);

    /// Convert LAPACK-style real/imag outputs into complex values/vectors.
    fn eig_ri_to_complex(
        n: usize,
        val_ri: &[Self],
        vec_ri: &[Self],
        values_out: &mut [Self::Complex],
        vectors_out: &mut [Self::Complex],
    );
}

macro_rules! impl_real_linalg_scalar {
    ($ty:ty, $complex:ty) => {
        impl LinalgScalar for $ty {
            type Real = $ty;
            type Complex = $complex;

            fn abs_real(&self) -> $ty {
                num_traits::Float::abs(*self)
            }

            fn real_epsilon() -> $ty {
                <$ty as num_traits::Float>::epsilon()
            }

            fn conj(&self) -> $ty {
                *self
            }

            fn from_parts(real: Self::Real, _imag: Self::Real) -> Self {
                real
            }

            fn real_part(&self) -> Self::Real {
                *self
            }

            fn imag_part(&self) -> Self::Real {
                0.0
            }
        }

        impl KernelLinalgScalar for $ty {}

        impl LapackEigScalar for $ty {
            fn eig_buffer_sizes(n: usize) -> (usize, usize) {
                (2 * n, 2 * n * n)
            }

            fn eig_ri_to_complex(
                n: usize,
                val_ri: &[Self],
                vec_ri: &[Self],
                values_out: &mut [$complex],
                vectors_out: &mut [$complex],
            ) {
                for i in 0..n {
                    values_out[i] = <$complex>::new(val_ri[2 * i], val_ri[2 * i + 1]);
                }
                for k in 0..(n * n) {
                    vectors_out[k] = <$complex>::new(vec_ri[2 * k], vec_ri[2 * k + 1]);
                }
            }
        }
    };
}

macro_rules! impl_complex_linalg_scalar {
    ($ty:ty, $real:ty) => {
        impl LinalgScalar for $ty {
            type Real = $real;
            type Complex = $ty;

            fn abs_real(&self) -> $real {
                self.norm()
            }

            fn real_epsilon() -> $real {
                <$real as num_traits::Float>::epsilon()
            }

            fn conj(&self) -> $ty {
                self.conj()
            }

            fn from_parts(real: Self::Real, imag: Self::Real) -> Self {
                <$ty>::new(real, imag)
            }

            fn real_part(&self) -> Self::Real {
                self.re
            }

            fn imag_part(&self) -> Self::Real {
                self.im
            }
        }

        impl KernelLinalgScalar for $ty {}

        impl LapackEigScalar for $ty {
            fn eig_buffer_sizes(n: usize) -> (usize, usize) {
                (n, n * n)
            }

            fn eig_ri_to_complex(
                _n: usize,
                val_ri: &[Self],
                vec_ri: &[Self],
                values_out: &mut [$ty],
                vectors_out: &mut [$ty],
            ) {
                values_out.copy_from_slice(val_ri);
                vectors_out.copy_from_slice(vec_ri);
            }
        }
    };
}

impl_real_linalg_scalar!(f64, Complex64);
impl_real_linalg_scalar!(f32, Complex32);
impl_complex_linalg_scalar!(Complex64, f64);
impl_complex_linalg_scalar!(Complex32, f32);

/// Result of a tensor-level QR decomposition.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg_prims::QrTensorResult;
/// let _result: Option<QrTensorResult<f64>> = None;
/// ```
#[derive(Clone)]
pub struct QrTensorResult<T: LinalgScalar> {
    pub q: Tensor<T>,
    pub r: Tensor<T>,
}

/// Result of a tensor-level thin SVD.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg_prims::SvdTensorResult;
/// let _result: Option<SvdTensorResult<f64>> = None;
/// ```
#[derive(Clone)]
pub struct SvdTensorResult<T: LinalgScalar> {
    pub u: Tensor<T>,
    pub s: Tensor<T::Real>,
    pub vt: Tensor<T>,
}

/// Result of a tensor-level LU factorization.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg_prims::LuTensorResult;
/// let _result: Option<LuTensorResult<f64>> = None;
/// ```
#[derive(Clone)]
pub struct LuTensorResult<T: LinalgScalar> {
    pub l: Tensor<T>,
    pub u: Tensor<T>,
    pub pivots: Tensor<i32>,
}

/// Result of a tensor-level LU factorization with numerical status.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg_prims::LuTensorExResult;
/// let _result: Option<LuTensorExResult<f64>> = None;
/// ```
#[derive(Clone)]
pub struct LuTensorExResult<T: LinalgScalar> {
    /// Unit-lower-triangular factor.
    pub l: Tensor<T>,
    /// Upper-triangular factor.
    pub u: Tensor<T>,
    /// Backend pivot tensor.
    pub pivots: Tensor<i32>,
    /// Per-batch numerical status tensor.
    pub info: Tensor<i32>,
}

/// Result of a tensor-level linear solve with numerical status.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg_prims::SolveTensorExResult;
/// let _result: Option<SolveTensorExResult<f64>> = None;
/// ```
#[derive(Clone)]
pub struct SolveTensorExResult<T: LinalgScalar> {
    /// Solution tensor.
    pub solution: Tensor<T>,
    /// Per-batch numerical status tensor.
    pub info: Tensor<i32>,
}

/// Result of a tensor-level Cholesky factorization with numerical status.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg_prims::CholeskyTensorExResult;
/// let _result: Option<CholeskyTensorExResult<f64>> = None;
/// ```
#[derive(Clone)]
pub struct CholeskyTensorExResult<T: LinalgScalar> {
    /// Lower-triangular Cholesky factor.
    pub l: Tensor<T>,
    /// Per-batch numerical status tensor.
    pub info: Tensor<i32>,
}

/// Result of a tensor-level Hermitian eigendecomposition.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg_prims::EigenTensorResult;
/// let _result: Option<EigenTensorResult<f64>> = None;
/// ```
#[derive(Clone)]
pub struct EigenTensorResult<T: LinalgScalar> {
    pub values: Tensor<T::Real>,
    pub vectors: Tensor<T>,
}

/// Result of a tensor-level general eigendecomposition.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg_prims::EigTensorResult;
/// let _result: Option<EigTensorResult<f64>> = None;
/// ```
#[derive(Clone)]
pub struct EigTensorResult<T: LinalgScalar> {
    pub values: Tensor<T::Complex>,
    pub vectors: Tensor<T::Complex>,
}

/// Backend-facing tensor linalg protocol.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg_prims::TensorLinalgPrims;
///
/// fn accepts_backend<B: TensorLinalgPrims<f64>>() {}
/// let _ = accepts_backend::<todo!()>;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinalgCapabilityOp {
    Solve,
    SolveTriangular,
    Qr,
    ThinSvd,
    LuFactor,
    Cholesky,
    EigenSym,
    Eig,
    LuSolve,
    Lstsq,
    LuFactorEx,
    CholeskyEx,
    SolveEx,
    Inv,
    Det,
    Slogdet,
    Pinv,
    MatrixExp,
    MatrixPower,
    Cross,
    HouseholderProduct,
    Vander,
    TensorInv,
    TensorSolve,
    Norm,
}

pub trait TensorLinalgPrims<T: KernelLinalgScalar> {
    type Context;

    fn has_linalg_support(op: LinalgCapabilityOp) -> bool;

    /// Solve a square linear system while returning per-batch numerical status.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_linalg_prims::TensorLinalgPrims;
    ///
    /// fn accepts_backend<B: TensorLinalgPrims<f64>>() {}
    /// let _ = accepts_backend::<todo!()>;
    /// ```
    fn solve_ex(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
        b: &Tensor<T>,
    ) -> Result<SolveTensorExResult<T>>;

    fn solve(ctx: &mut Self::Context, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;
    fn lu_solve(
        ctx: &mut Self::Context,
        factors: &Tensor<T>,
        pivots: &Tensor<i32>,
        b: &Tensor<T>,
    ) -> Result<Tensor<T>>;
    fn solve_triangular(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
        b: &Tensor<T>,
        upper: bool,
    ) -> Result<Tensor<T>>;
    fn qr(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<QrTensorResult<T>>;
    fn thin_svd(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<SvdTensorResult<T>>;
    fn svdvals(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<Tensor<T::Real>>;
    /// Compute an LU factorization while returning per-batch numerical status.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_linalg_prims::TensorLinalgPrims;
    ///
    /// fn accepts_backend<B: TensorLinalgPrims<f64>>() {}
    /// let _ = accepts_backend::<todo!()>;
    /// ```
    fn lu_factor_ex(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<LuTensorExResult<T>>;
    fn lu_factor(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<LuTensorResult<T>>;
    /// Compute a Cholesky factorization while returning per-batch numerical status.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_linalg_prims::TensorLinalgPrims;
    ///
    /// fn accepts_backend<B: TensorLinalgPrims<f64>>() {}
    /// let _ = accepts_backend::<todo!()>;
    /// ```
    fn cholesky_ex(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<CholeskyTensorExResult<T>>;
    fn cholesky(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<Tensor<T>>;
    fn eigen_sym(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<EigenTensorResult<T>>;
    fn eig(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<EigTensorResult<T>>;
}

#[cfg(test)]
mod tests;
