use num_complex::Complex;
use tenferro_algebra::Scalar;

use crate::{AdTensor, DynTensorTyped};

/// Typed SVD result for internal builder wiring.
///
/// # Examples
///
/// ```text
/// use tenferro_dyadtensor::{Tensor, set_default_runtime, RuntimeContext};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let ad_a = Tensor::from_tensor(a);
/// let out = ad_a.svd().unwrap();
/// assert_eq!(out.s.dims(), &[2]);
/// ```
#[derive(Clone)]
pub struct TypedSvdResult<T: Scalar + DynTensorTyped> {
    /// Left singular vectors.
    pub u: AdTensor<T>,
    /// Singular values.
    pub s: AdTensor<T>,
    /// Right singular vectors transposed.
    pub vt: AdTensor<T>,
}

/// Typed QR result for internal builder wiring.
///
/// # Examples
///
/// ```text
/// let out = qr_ad(&ad_a).run().unwrap();
/// let _q = &out.q;
/// let _r = &out.r;
/// ```
#[derive(Clone)]
pub struct TypedQrResult<T: Scalar + DynTensorTyped> {
    /// Q factor.
    pub q: AdTensor<T>,
    /// R factor.
    pub r: AdTensor<T>,
}

/// Typed LU result for internal builder wiring.
///
/// # Examples
///
/// ```text
/// let out = lu_ad(&ad_a).run().unwrap();
/// let _l = &out.l;
/// let _u = &out.u;
/// ```
#[derive(Clone)]
pub struct TypedLuResult<T: Scalar + DynTensorTyped> {
    /// Permutation indices.
    pub p: Option<Vec<usize>>,
    /// Lower factor.
    pub l: AdTensor<T>,
    /// Upper factor.
    pub u: AdTensor<T>,
}

/// Typed Hermitian eigen decomposition result for internal builder wiring.
///
/// # Examples
///
/// ```text
/// let out = eigen_ad(&ad_a).run().unwrap();
/// let _values = &out.values;
/// let _vectors = &out.vectors;
/// ```
#[derive(Clone)]
pub struct TypedEigenResult<T: Scalar + DynTensorTyped> {
    /// Eigenvalues.
    pub values: AdTensor<T>,
    /// Eigenvectors.
    pub vectors: AdTensor<T>,
}

/// Typed general eigendecomposition result for internal builder wiring.
///
/// # Examples
///
/// ```text
/// let out = eig_ad(&ad_a).run().unwrap();
/// let _values = &out.values;
/// let _vectors = &out.vectors;
/// ```
#[derive(Clone)]
pub struct TypedEigResult<T>
where
    T: Scalar,
    Complex<T>: Scalar + DynTensorTyped,
{
    /// Complex eigenvalues.
    pub values: AdTensor<Complex<T>>,
    /// Complex eigenvectors.
    pub vectors: AdTensor<Complex<T>>,
}

/// Typed sign/logabsdet result for internal builder wiring.
///
/// # Examples
///
/// ```text
/// let out = slogdet_ad(&ad_a).run().unwrap();
/// let _sign = &out.sign;
/// let _logabsdet = &out.logabsdet;
/// ```
#[derive(Clone)]
pub struct TypedSlogdetResult<T: Scalar + DynTensorTyped> {
    /// Sign tensor.
    pub sign: AdTensor<T>,
    /// Log-absolute-determinant tensor.
    pub logabsdet: AdTensor<T>,
}

/// Typed least-squares result for internal builder wiring.
///
/// # Examples
///
/// ```text
/// let out = lstsq_ad(&ad_a, &ad_b).run().unwrap();
/// let _x = &out.x;
/// let _residual = &out.residual;
/// ```
#[derive(Clone)]
pub struct TypedLstsqResult<T: Scalar + DynTensorTyped> {
    /// Least squares solution.
    pub x: AdTensor<T>,
    /// Residual tensor.
    pub residual: AdTensor<T>,
}
