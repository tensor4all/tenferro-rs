use super::Tensor;

/// Dynamic AD-aware SVD result.
///
/// # Examples
///
/// ```ignore
/// let out = x.svd()?;
/// let _u = &out.u;
/// let _s = &out.s;
/// let _vt = &out.vt;
/// ```
#[derive(Clone)]
pub struct SvdResult {
    /// Left singular vectors.
    pub u: Tensor,
    /// Singular values.
    pub s: Tensor,
    /// Right singular vectors transposed.
    pub vt: Tensor,
}

/// Dynamic AD-aware QR result.
///
/// # Examples
///
/// ```ignore
/// let out = x.qr()?;
/// let _q = &out.q;
/// let _r = &out.r;
/// ```
#[derive(Clone)]
pub struct QrResult {
    /// Q factor.
    pub q: Tensor,
    /// R factor.
    pub r: Tensor,
}

/// Dynamic AD-aware LU result.
///
/// # Examples
///
/// ```ignore
/// let out = x.lu()?;
/// let _l = &out.l;
/// let _u = &out.u;
/// ```
#[derive(Clone)]
pub struct LuResult {
    /// Permutation indices.
    pub p: Option<Vec<usize>>,
    /// Lower factor.
    pub l: Tensor,
    /// Upper factor.
    pub u: Tensor,
}

/// Dynamic AD-aware symmetric/Hermitian eigen result.
///
/// # Examples
///
/// ```ignore
/// let out = x.eigen()?;
/// let _values = &out.values;
/// let _vectors = &out.vectors;
/// ```
#[derive(Clone)]
pub struct EigenResult {
    /// Eigenvalues.
    pub values: Tensor,
    /// Eigenvectors.
    pub vectors: Tensor,
}

/// Dynamic AD-aware general eigendecomposition result.
///
/// # Examples
///
/// ```ignore
/// let out = x.eig()?;
/// let _values = &out.values;
/// let _vectors = &out.vectors;
/// ```
#[derive(Clone)]
pub struct EigResult {
    /// Complex eigenvalues.
    pub values: Tensor,
    /// Complex eigenvectors.
    pub vectors: Tensor,
}

/// Dynamic AD-aware sign/logabsdet result.
///
/// # Examples
///
/// ```ignore
/// let out = x.slogdet()?;
/// let _sign = &out.sign;
/// let _logabsdet = &out.logabsdet;
/// ```
#[derive(Clone)]
pub struct SlogdetResult {
    /// Sign tensor.
    pub sign: Tensor,
    /// Log-absolute-determinant tensor.
    pub logabsdet: Tensor,
}

/// Dynamic AD-aware least squares result.
///
/// # Examples
///
/// ```ignore
/// let out = a.lstsq(&b)?;
/// let _x = &out.x;
/// let _residual = &out.residual;
/// ```
#[derive(Clone)]
pub struct LstsqResult {
    /// Least squares solution.
    pub x: Tensor,
    /// Residual tensor.
    pub residual: Tensor,
}
