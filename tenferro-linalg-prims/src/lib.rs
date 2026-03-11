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

use num_complex::{Complex32, Complex64};
use tenferro_algebra::Scalar;
use tenferro_device::Result;
use tenferro_tensor::Tensor;

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
}

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

impl LinalgScalar for f64 {
    type Real = f64;
    type Complex = Complex64;

    fn abs_real(&self) -> f64 {
        num_traits::Float::abs(*self)
    }

    fn real_epsilon() -> f64 {
        <f64 as num_traits::Float>::epsilon()
    }

    fn conj(&self) -> f64 {
        *self
    }
}

impl LapackEigScalar for f64 {
    fn eig_buffer_sizes(n: usize) -> (usize, usize) {
        (2 * n, 2 * n * n)
    }

    fn eig_ri_to_complex(
        n: usize,
        val_ri: &[Self],
        vec_ri: &[Self],
        values_out: &mut [Complex64],
        vectors_out: &mut [Complex64],
    ) {
        for i in 0..n {
            values_out[i] = Complex64::new(val_ri[2 * i], val_ri[2 * i + 1]);
        }
        for k in 0..(n * n) {
            vectors_out[k] = Complex64::new(vec_ri[2 * k], vec_ri[2 * k + 1]);
        }
    }
}

impl LinalgScalar for f32 {
    type Real = f32;
    type Complex = Complex32;

    fn abs_real(&self) -> f32 {
        num_traits::Float::abs(*self)
    }

    fn real_epsilon() -> f32 {
        <f32 as num_traits::Float>::epsilon()
    }

    fn conj(&self) -> f32 {
        *self
    }
}

impl LapackEigScalar for f32 {
    fn eig_buffer_sizes(n: usize) -> (usize, usize) {
        (2 * n, 2 * n * n)
    }

    fn eig_ri_to_complex(
        n: usize,
        val_ri: &[Self],
        vec_ri: &[Self],
        values_out: &mut [Complex32],
        vectors_out: &mut [Complex32],
    ) {
        for i in 0..n {
            values_out[i] = Complex32::new(val_ri[2 * i], val_ri[2 * i + 1]);
        }
        for k in 0..(n * n) {
            vectors_out[k] = Complex32::new(vec_ri[2 * k], vec_ri[2 * k + 1]);
        }
    }
}

impl LinalgScalar for Complex64 {
    type Real = f64;
    type Complex = Complex64;

    fn abs_real(&self) -> f64 {
        self.norm()
    }

    fn real_epsilon() -> f64 {
        <f64 as num_traits::Float>::epsilon()
    }

    fn conj(&self) -> Complex64 {
        self.conj()
    }
}

impl LapackEigScalar for Complex64 {
    fn eig_buffer_sizes(n: usize) -> (usize, usize) {
        (n, n * n)
    }

    fn eig_ri_to_complex(
        _n: usize,
        val_ri: &[Self],
        vec_ri: &[Self],
        values_out: &mut [Complex64],
        vectors_out: &mut [Complex64],
    ) {
        values_out.copy_from_slice(val_ri);
        vectors_out.copy_from_slice(vec_ri);
    }
}

impl LinalgScalar for Complex32 {
    type Real = f32;
    type Complex = Complex32;

    fn abs_real(&self) -> f32 {
        self.norm()
    }

    fn real_epsilon() -> f32 {
        <f32 as num_traits::Float>::epsilon()
    }

    fn conj(&self) -> Complex32 {
        self.conj()
    }
}

impl LapackEigScalar for Complex32 {
    fn eig_buffer_sizes(n: usize) -> (usize, usize) {
        (n, n * n)
    }

    fn eig_ri_to_complex(
        _n: usize,
        val_ri: &[Self],
        vec_ri: &[Self],
        values_out: &mut [Complex32],
        vectors_out: &mut [Complex32],
    ) {
        values_out.copy_from_slice(val_ri);
        vectors_out.copy_from_slice(vec_ri);
    }
}

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
    pub pivots: Vec<i32>,
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

pub trait TensorLinalgPrims<T: LinalgScalar> {
    type Context;

    fn has_linalg_support(op: LinalgCapabilityOp) -> bool;

    fn solve(ctx: &mut Self::Context, a: &Tensor<T>, b: &Tensor<T>) -> Result<Tensor<T>>;
    fn solve_triangular(
        ctx: &mut Self::Context,
        a: &Tensor<T>,
        b: &Tensor<T>,
        upper: bool,
    ) -> Result<Tensor<T>>;
    fn qr(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<QrTensorResult<T>>;
    fn thin_svd(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<SvdTensorResult<T>>;
    fn lu_factor(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<LuTensorResult<T>>;
    fn cholesky(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<Tensor<T>>;
    fn eigen_sym(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<EigenTensorResult<T>>;
    fn eig(ctx: &mut Self::Context, a: &Tensor<T>) -> Result<EigTensorResult<T>>;
}

#[cfg(test)]
mod tests;
