use num_complex::Complex;
use tenferro_algebra::Scalar;

use crate::AdTensor;

/// AD-aware SVD result.
///
/// # Examples
///
/// ```rust
/// use tenferro_dyadtensor::{svd_ad, set_default_runtime, RuntimeContext};
/// use tenferro_prims::CpuContext;
/// use tenferro_tensor::{MemoryOrder, Tensor};
///
/// let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
/// let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
/// let ad_a = tenferro_dyadtensor::AdTensor::new_primal(a);
/// let out = svd_ad(&ad_a).run().unwrap();
/// assert_eq!(out.s.dims(), &[2]);
/// ```
#[derive(Clone)]
pub struct AdSvdResult<T: Scalar> {
    /// Left singular vectors.
    pub u: AdTensor<T>,
    /// Singular values.
    pub s: AdTensor<T>,
    /// Right singular vectors transposed.
    pub vt: AdTensor<T>,
}

/// AD-aware QR result.
///
/// # Examples
///
/// ```ignore
/// let out = qr_ad(&ad_a).run().unwrap();
/// let _q = &out.q;
/// let _r = &out.r;
/// ```
#[derive(Clone)]
pub struct AdQrResult<T: Scalar> {
    /// Q factor.
    pub q: AdTensor<T>,
    /// R factor.
    pub r: AdTensor<T>,
}

/// AD-aware LU result.
///
/// # Examples
///
/// ```ignore
/// let out = lu_ad(&ad_a).run().unwrap();
/// let _l = &out.l;
/// let _u = &out.u;
/// ```
#[derive(Clone)]
pub struct AdLuResult<T: Scalar> {
    /// Permutation indices.
    pub p: Option<Vec<usize>>,
    /// Lower factor.
    pub l: AdTensor<T>,
    /// Upper factor.
    pub u: AdTensor<T>,
}

/// AD-aware eigen decomposition result.
///
/// # Examples
///
/// ```ignore
/// let out = eigen_ad(&ad_a).run().unwrap();
/// let _values = &out.values;
/// let _vectors = &out.vectors;
/// ```
#[derive(Clone)]
pub struct AdEigenResult<T: Scalar> {
    /// Eigenvalues.
    pub values: AdTensor<T>,
    /// Eigenvectors.
    pub vectors: AdTensor<T>,
}

/// AD-aware general eigendecomposition result.
///
/// # Examples
///
/// ```ignore
/// let out = eig_ad(&ad_a).run().unwrap();
/// let _values = &out.values;
/// let _vectors = &out.vectors;
/// ```
#[derive(Clone)]
pub struct AdEigResult<T>
where
    T: Scalar,
    Complex<T>: Scalar,
{
    /// Complex eigenvalues.
    pub values: AdTensor<Complex<T>>,
    /// Complex eigenvectors.
    pub vectors: AdTensor<Complex<T>>,
}

/// AD-aware sign/logabsdet result.
///
/// # Examples
///
/// ```ignore
/// let out = slogdet_ad(&ad_a).run().unwrap();
/// let _sign = &out.sign;
/// let _logabsdet = &out.logabsdet;
/// ```
#[derive(Clone)]
pub struct AdSlogdetResult<T: Scalar> {
    /// Sign tensor.
    pub sign: AdTensor<T>,
    /// Log-absolute-determinant tensor.
    pub logabsdet: AdTensor<T>,
}

/// AD-aware least squares result.
///
/// # Examples
///
/// ```ignore
/// let out = lstsq_ad(&ad_a, &ad_b).run().unwrap();
/// let _x = &out.x;
/// let _residual = &out.residual;
/// ```
#[derive(Clone)]
pub struct AdLstsqResult<T: Scalar> {
    /// Least squares solution.
    pub x: AdTensor<T>,
    /// Residual tensor.
    pub residual: AdTensor<T>,
}
