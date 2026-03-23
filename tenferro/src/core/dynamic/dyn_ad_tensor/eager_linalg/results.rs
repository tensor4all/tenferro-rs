use super::Tensor;
use tenferro_tensor::Tensor as DenseTensor;

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

/// Dynamic AD-aware LU factorization result.
///
/// # Examples
///
/// ```ignore
/// let out = x.lu_factor()?;
/// let _factors = &out.factors;
/// ```
#[derive(Clone)]
pub struct LuFactorResult {
    /// Packed LU factors.
    pub factors: Tensor,
    /// Backend pivot tensor in 1-indexed step-pivot form.
    pub pivots: DenseTensor<i32>,
}

/// Dynamic AD-aware LU factorization result with status codes.
///
/// # Examples
///
/// ```ignore
/// let out = x.lu_factor_ex()?;
/// let _info = &out.info;
/// ```
#[derive(Clone)]
pub struct LuFactorExResult {
    /// Packed LU factors.
    pub factors: Tensor,
    /// Backend pivot tensor in 1-indexed step-pivot form.
    pub pivots: DenseTensor<i32>,
    /// Per-batch numerical status tensor.
    pub info: DenseTensor<i32>,
}

/// Dynamic AD-aware solve result with status codes.
///
/// # Examples
///
/// ```ignore
/// let out = a.solve_ex(&b)?;
/// let _solution = &out.solution;
/// ```
#[derive(Clone)]
pub struct SolveExResult {
    /// Solution tensor.
    pub solution: Tensor,
    /// Per-batch numerical status.
    pub info: Vec<i32>,
}

/// Dynamic AD-aware inverse result with status codes.
///
/// # Examples
///
/// ```ignore
/// let out = x.inv_ex()?;
/// let _inverse = &out.inverse;
/// ```
#[derive(Clone)]
pub struct InvExResult {
    /// Inverse tensor.
    pub inverse: Tensor,
    /// Per-batch numerical status.
    pub info: Vec<i32>,
}

/// Dynamic AD-aware Cholesky result with status codes.
///
/// # Examples
///
/// ```ignore
/// let out = x.cholesky_ex()?;
/// let _l = &out.l;
/// ```
#[derive(Clone)]
pub struct CholeskyExResult {
    /// Lower-triangular factor.
    pub l: Tensor,
    /// Per-batch numerical status.
    pub info: Vec<i32>,
}
