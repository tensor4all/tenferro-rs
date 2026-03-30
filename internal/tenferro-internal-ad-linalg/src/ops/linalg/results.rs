use tenferro_internal_ad_core::DynAdTensor;
use tenferro_tensor::Tensor;

/// Typed SVD result for internal builder wiring.
///
/// # Examples
///
/// ```text
/// use tenferro::{Tensor, set_default_runtime, RuntimeContext};
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
pub struct SvdResult<U, S = U, Vt = U> {
    /// Left singular vectors.
    pub u: U,
    /// Singular values.
    pub s: S,
    /// Right singular vectors transposed.
    pub vt: Vt,
}

/// Typed SVD result for internal builder wiring.
pub type TypedSvdResult = SvdResult<DynAdTensor>;

/// Erased SVD result for dynamic entrypoints.
pub type DynSvdResult = SvdResult<DynAdTensor>;

/// QR result for internal builder wiring and dynamic entrypoints.
///
/// # Examples
///
/// ```text
/// let out = qr_ad(&ad_a).run().unwrap();
/// let _q = &out.q;
/// let _r = &out.r;
/// ```
#[derive(Clone)]
pub struct QrResult<Q, R = Q> {
    pub q: Q,
    pub r: R,
}

pub type TypedQrResult = QrResult<DynAdTensor>;
pub type DynQrResult = QrResult<DynAdTensor>;

/// LU result for internal builder wiring and dynamic entrypoints.
///
/// # Examples
///
/// ```text
/// let out = lu_ad(&ad_a).run().unwrap();
/// let _p = &out.p;
/// let _l = &out.l;
/// let _u = &out.u;
/// ```
#[derive(Clone)]
pub struct LuResult<P, L = P, U = P> {
    pub p: P,
    pub l: L,
    pub u: U,
}

pub type TypedLuResult = LuResult<DynAdTensor>;
pub type DynLuResult = LuResult<DynAdTensor>;

/// Hermitian eigen decomposition result for internal builder wiring and
/// dynamic entrypoints.
///
/// # Examples
///
/// ```text
/// let out = eigen_ad(&ad_a).run().unwrap();
/// let _values = &out.values;
/// let _vectors = &out.vectors;
/// ```
#[derive(Clone)]
pub struct EigenResult<Values, Vectors = Values> {
    pub values: Values,
    pub vectors: Vectors,
}

pub type TypedEigenResult = EigenResult<DynAdTensor>;
pub type DynEigenResult = EigenResult<DynAdTensor>;

/// General eigendecomposition result for internal builder wiring and dynamic
/// entrypoints.
///
/// # Examples
///
/// ```text
/// let out = eig_ad(&ad_a).run().unwrap();
/// let _values = &out.values;
/// let _vectors = &out.vectors;
/// ```
#[derive(Clone)]
pub struct EigResult<Values, Vectors = Values> {
    pub values: Values,
    pub vectors: Vectors,
}

pub type TypedEigResult = EigResult<DynAdTensor>;
pub type DynEigResult = EigResult<DynAdTensor>;

/// Sign/logabsdet result for internal builder wiring and dynamic entrypoints.
///
/// # Examples
///
/// ```text
/// let out = slogdet_ad(&ad_a).run().unwrap();
/// let _sign = &out.sign;
/// let _logabsdet = &out.logabsdet;
/// ```
#[derive(Clone)]
pub struct SlogdetResult<Sign, LogAbsDet = Sign> {
    pub sign: Sign,
    pub logabsdet: LogAbsDet,
}

pub type TypedSlogdetResult = SlogdetResult<DynAdTensor>;
pub type DynSlogdetResult = SlogdetResult<DynAdTensor>;

/// Least-squares result for internal builder wiring and dynamic entrypoints.
///
/// # Examples
///
/// ```text
/// let out = lstsq_ad(&ad_a, &ad_b).run().unwrap();
/// let _x = &out.x;
/// let _residual = &out.residual;
/// ```
#[derive(Clone)]
pub struct LstsqResult<X, Residual = X> {
    pub x: X,
    pub residual: Residual,
}

pub type TypedLstsqResult = LstsqResult<DynAdTensor>;
pub type DynLstsqResult = LstsqResult<DynAdTensor>;

/// LU factorization result for dynamic eager entrypoints.
#[derive(Clone)]
pub struct LuFactorResult<C> {
    pub factors: C,
    pub pivots: Tensor<i32>,
}

pub type DynLuFactorResult = LuFactorResult<DynAdTensor>;

/// LU factorization result with status codes for dynamic eager entrypoints.
#[derive(Clone)]
pub struct LuFactorExResult<C> {
    pub factors: C,
    pub pivots: Tensor<i32>,
    pub info: Tensor<i32>,
}

pub type DynLuFactorExResult = LuFactorExResult<DynAdTensor>;

/// Solve result with status codes for dynamic eager entrypoints.
#[derive(Clone)]
pub struct SolveExResult<C> {
    pub solution: C,
    pub info: Tensor<i32>,
}

pub type DynSolveExResult = SolveExResult<DynAdTensor>;

/// Inverse result with status codes for dynamic eager entrypoints.
#[derive(Clone)]
pub struct InvExResult<C> {
    pub inverse: C,
    pub info: Tensor<i32>,
}

pub type DynInvExResult = InvExResult<DynAdTensor>;

/// Cholesky result with status codes for dynamic eager entrypoints.
#[derive(Clone)]
pub struct CholeskyExResult<C> {
    pub l: C,
    pub info: Tensor<i32>,
}

pub type DynCholeskyExResult = CholeskyExResult<DynAdTensor>;
