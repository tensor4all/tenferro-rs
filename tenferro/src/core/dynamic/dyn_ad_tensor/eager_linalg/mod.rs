use super::Tensor;
use crate::ops::ad;
use crate::{AdMode, Error, Result, ScalarType};
use tenferro_algebra::Scalar;

mod extra;
mod results;

pub use results::{
    CholeskyExResult, EigResult, EigenResult, InvExResult, LstsqResult, LuFactorExResult,
    LuFactorResult, LuResult, QrResult, SlogdetResult, SolveExResult, SvdResult,
};

pub(super) fn real_only_error(op: &'static str, dtype: ScalarType) -> Error {
    Error::InvalidAdTensor {
        message: format!("{op} currently requires a real Tensor input, got {dtype:?}"),
    }
}

pub(super) fn primal_only_error(op: &'static str) -> Error {
    Error::InvalidAdTensor {
        message: format!("{op} currently supports only primal tensors"),
    }
}

pub(super) fn primal_complex_only_error(op: &'static str) -> Error {
    Error::InvalidAdTensor {
        message: format!("{op} currently supports complex tensors only in primal mode"),
    }
}

pub(super) fn same_dtype_error(op: &'static str, lhs: ScalarType, rhs: ScalarType) -> Error {
    Error::InvalidAdTensor {
        message: format!("{op} requires matching dtypes, got lhs={lhs:?}, rhs={rhs:?}"),
    }
}

impl Tensor {
    fn dense_primal_with_mode_error<T>(
        value: &crate::AdTensor<T>,
        op: &'static str,
        mode_error: fn(&'static str) -> Error,
    ) -> Result<tenferro_tensor::Tensor<T>>
    where
        T: Scalar + crate::DynTensorTyped + crate::runtime::contracts::LinalgRuntimeValue,
    {
        if !value.is_dense() {
            return Err(Error::UnsupportedStructuredLinalg { op });
        }
        if value.mode() != AdMode::Primal {
            return Err(mode_error(op));
        }
        value
            .structured_primal()
            .to_dense()
            .map_err(|e| Error::InvalidAdTensor {
                message: format!("{op} failed to densify structured complex input: {e}"),
            })
    }

    pub(super) fn dense_primal_only<T>(
        value: &crate::AdTensor<T>,
        op: &'static str,
    ) -> Result<tenferro_tensor::Tensor<T>>
    where
        T: Scalar + crate::DynTensorTyped + crate::runtime::contracts::LinalgRuntimeValue,
    {
        Self::dense_primal_with_mode_error(value, op, primal_only_error)
    }

    pub(super) fn dense_primal_complex_only<T>(
        value: &crate::AdTensor<T>,
        op: &'static str,
    ) -> Result<tenferro_tensor::Tensor<T>>
    where
        T: Scalar + crate::DynTensorTyped + crate::runtime::contracts::LinalgRuntimeValue,
    {
        Self::dense_primal_with_mode_error(value, op, primal_complex_only_error)
    }

    /// Runs eager AD SVD on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.svd()?;
    /// let _s = &out.s;
    /// ```
    pub fn svd(&self) -> Result<SvdResult> {
        match self {
            Self::F32(value) => {
                let out = ad::svd(value)?;
                Ok(SvdResult {
                    u: out.u.into(),
                    s: out.s.into(),
                    vt: out.vt.into(),
                })
            }
            Self::F64(value) => {
                let out = ad::svd(value)?;
                Ok(SvdResult {
                    u: out.u.into(),
                    s: out.s.into(),
                    vt: out.vt.into(),
                })
            }
            Self::C32(value) => {
                let out = ad::svd(value)?;
                Ok(SvdResult {
                    u: out.u.into(),
                    s: out.s.into(),
                    vt: out.vt.into(),
                })
            }
            Self::C64(value) => {
                let out = ad::svd(value)?;
                Ok(SvdResult {
                    u: out.u.into(),
                    s: out.s.into(),
                    vt: out.vt.into(),
                })
            }
        }
    }

    /// Runs eager AD QR on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.qr()?;
    /// let _q = &out.q;
    /// ```
    pub fn qr(&self) -> Result<QrResult> {
        match self {
            Self::F32(value) => {
                let out = ad::qr(value)?;
                Ok(QrResult {
                    q: out.q.into(),
                    r: out.r.into(),
                })
            }
            Self::F64(value) => {
                let out = ad::qr(value)?;
                Ok(QrResult {
                    q: out.q.into(),
                    r: out.r.into(),
                })
            }
            Self::C32(value) => {
                let dense = Self::dense_primal_complex_only(value, "qr")?;
                let out = crate::ops::qr(&dense).run()?;
                Ok(QrResult {
                    q: Tensor::from_tensor(out.q),
                    r: Tensor::from_tensor(out.r),
                })
            }
            Self::C64(value) => {
                let dense = Self::dense_primal_complex_only(value, "qr")?;
                let out = crate::ops::qr(&dense).run()?;
                Ok(QrResult {
                    q: Tensor::from_tensor(out.q),
                    r: Tensor::from_tensor(out.r),
                })
            }
        }
    }

    /// Runs eager AD LU on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.lu()?;
    /// let _u = &out.u;
    /// ```
    pub fn lu(&self) -> Result<LuResult> {
        match self {
            Self::F32(value) => {
                let out = ad::lu(value)?;
                Ok(LuResult {
                    p: out.p,
                    l: out.l.into(),
                    u: out.u.into(),
                })
            }
            Self::F64(value) => {
                let out = ad::lu(value)?;
                Ok(LuResult {
                    p: out.p,
                    l: out.l.into(),
                    u: out.u.into(),
                })
            }
            Self::C32(value) => {
                let dense = Self::dense_primal_complex_only(value, "lu")?;
                let out = crate::ops::lu(&dense).run()?;
                Ok(LuResult {
                    p: out.p,
                    l: Tensor::from_tensor(out.l),
                    u: Tensor::from_tensor(out.u),
                })
            }
            Self::C64(value) => {
                let dense = Self::dense_primal_complex_only(value, "lu")?;
                let out = crate::ops::lu(&dense).run()?;
                Ok(LuResult {
                    p: out.p,
                    l: Tensor::from_tensor(out.l),
                    u: Tensor::from_tensor(out.u),
                })
            }
        }
    }

    /// Runs eager AD symmetric/Hermitian eigen decomposition on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.eigen()?;
    /// let _values = &out.values;
    /// ```
    pub fn eigen(&self) -> Result<EigenResult> {
        match self {
            Self::F32(value) => {
                let out = ad::eigen(value)?;
                Ok(EigenResult {
                    values: out.values.into(),
                    vectors: out.vectors.into(),
                })
            }
            Self::F64(value) => {
                let out = ad::eigen(value)?;
                Ok(EigenResult {
                    values: out.values.into(),
                    vectors: out.vectors.into(),
                })
            }
            Self::C32(value) => {
                let dense = Self::dense_primal_complex_only(value, "eigen")?;
                let out = crate::ops::eigen(&dense).run()?;
                Ok(EigenResult {
                    values: Tensor::from_tensor(out.values),
                    vectors: Tensor::from_tensor(out.vectors),
                })
            }
            Self::C64(value) => {
                let dense = Self::dense_primal_complex_only(value, "eigen")?;
                let out = crate::ops::eigen(&dense).run()?;
                Ok(EigenResult {
                    values: Tensor::from_tensor(out.values),
                    vectors: Tensor::from_tensor(out.vectors),
                })
            }
        }
    }

    /// Runs eager AD general eigendecomposition on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.eig()?;
    /// let _vectors = &out.vectors;
    /// ```
    pub fn eig(&self) -> Result<EigResult> {
        match self {
            Self::F32(value) => {
                let out = ad::eig(value)?;
                Ok(EigResult {
                    values: out.values.into(),
                    vectors: out.vectors.into(),
                })
            }
            Self::F64(value) => {
                let out = ad::eig(value)?;
                Ok(EigResult {
                    values: out.values.into(),
                    vectors: out.vectors.into(),
                })
            }
            _ => Err(real_only_error("eig", self.scalar_type())),
        }
    }

    /// Runs eager AD least-squares solve on dynamic tensors.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = a.lstsq(&b)?;
    /// let _x = &out.x;
    /// ```
    pub fn lstsq(&self, rhs: &Self) -> Result<LstsqResult> {
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                let out = ad::lstsq(lhs, rhs)?;
                Ok(LstsqResult {
                    x: out.x.into(),
                    residual: out.residual.into(),
                })
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                let out = ad::lstsq(lhs, rhs)?;
                Ok(LstsqResult {
                    x: out.x.into(),
                    residual: out.residual.into(),
                })
            }
            (lhs, rhs) => Err(same_dtype_error(
                "lstsq",
                lhs.scalar_type(),
                rhs.scalar_type(),
            )),
        }
    }

    /// Runs eager AD Cholesky on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let chol = x.cholesky()?;
    /// ```
    pub fn cholesky(&self) -> Result<Self> {
        match self {
            Self::F32(value) => Ok(Self::F32(ad::cholesky(value)?)),
            Self::F64(value) => Ok(Self::F64(ad::cholesky(value)?)),
            Self::C32(value) => {
                let dense = Self::dense_primal_complex_only(value, "cholesky")?;
                Ok(Self::from_tensor(crate::ops::cholesky(&dense).run()?))
            }
            Self::C64(value) => {
                let dense = Self::dense_primal_complex_only(value, "cholesky")?;
                Ok(Self::from_tensor(crate::ops::cholesky(&dense).run()?))
            }
        }
    }

    /// Runs eager AD linear solve on dynamic tensors.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = a.solve(&b)?;
    /// ```
    pub fn solve(&self, rhs: &Self) -> Result<Self> {
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => Ok(Self::F32(ad::solve(lhs, rhs)?)),
            (Self::F64(lhs), Self::F64(rhs)) => Ok(Self::F64(ad::solve(lhs, rhs)?)),
            (Self::C32(lhs), Self::C32(rhs)) => Ok(Self::C32(ad::solve(lhs, rhs)?)),
            (Self::C64(lhs), Self::C64(rhs)) => Ok(Self::C64(ad::solve(lhs, rhs)?)),
            (lhs, rhs) => Err(same_dtype_error(
                "solve",
                lhs.scalar_type(),
                rhs.scalar_type(),
            )),
        }
    }

    /// Runs eager AD triangular solve on dynamic tensors.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = a.solve_triangular(&b)?;
    /// ```
    pub fn solve_triangular(&self, rhs: &Self) -> Result<Self> {
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => Ok(Self::F32(ad::solve_triangular(lhs, rhs)?)),
            (Self::F64(lhs), Self::F64(rhs)) => Ok(Self::F64(ad::solve_triangular(lhs, rhs)?)),
            (Self::C32(lhs), Self::C32(rhs)) => Ok(Self::C32(ad::solve_triangular(lhs, rhs)?)),
            (Self::C64(lhs), Self::C64(rhs)) => Ok(Self::C64(ad::solve_triangular(lhs, rhs)?)),
            (lhs, rhs) => Err(same_dtype_error(
                "solve_triangular",
                lhs.scalar_type(),
                rhs.scalar_type(),
            )),
        }
    }

    /// Runs eager AD inverse on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let inv = x.inv()?;
    /// ```
    pub fn inv(&self) -> Result<Self> {
        match self {
            Self::F32(value) => Ok(Self::F32(ad::inv(value)?)),
            Self::F64(value) => Ok(Self::F64(ad::inv(value)?)),
            Self::C32(value) => {
                let dense = Self::dense_primal_complex_only(value, "inv")?;
                Ok(Self::from_tensor(crate::ops::inv(&dense).run()?))
            }
            Self::C64(value) => {
                let dense = Self::dense_primal_complex_only(value, "inv")?;
                Ok(Self::from_tensor(crate::ops::inv(&dense).run()?))
            }
        }
    }

    /// Runs eager AD determinant on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let det = x.det()?;
    /// ```
    pub fn det(&self) -> Result<Self> {
        match self {
            Self::F32(value) => Ok(Self::F32(ad::det(value)?)),
            Self::F64(value) => Ok(Self::F64(ad::det(value)?)),
            _ => Err(real_only_error("det", self.scalar_type())),
        }
    }

    /// Runs eager AD slogdet on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.slogdet()?;
    /// let _sign = &out.sign;
    /// ```
    pub fn slogdet(&self) -> Result<SlogdetResult> {
        match self {
            Self::F32(value) => {
                let out = ad::slogdet(value)?;
                Ok(SlogdetResult {
                    sign: out.sign.into(),
                    logabsdet: out.logabsdet.into(),
                })
            }
            Self::F64(value) => {
                let out = ad::slogdet(value)?;
                Ok(SlogdetResult {
                    sign: out.sign.into(),
                    logabsdet: out.logabsdet.into(),
                })
            }
            _ => Err(real_only_error("slogdet", self.scalar_type())),
        }
    }

    /// Runs eager AD pseudoinverse on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let pinv = x.pinv()?;
    /// ```
    pub fn pinv(&self) -> Result<Self> {
        match self {
            Self::F32(value) => Ok(Self::F32(ad::pinv(value)?)),
            Self::F64(value) => Ok(Self::F64(ad::pinv(value)?)),
            _ => Err(real_only_error("pinv", self.scalar_type())),
        }
    }

    /// Runs eager AD matrix exponential on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.matrix_exp()?;
    /// ```
    pub fn matrix_exp(&self) -> Result<Self> {
        match self {
            Self::F32(value) => Ok(Self::F32(ad::matrix_exp(value)?)),
            Self::F64(value) => Ok(Self::F64(ad::matrix_exp(value)?)),
            Self::C32(value) => {
                let dense = Self::dense_primal_complex_only(value, "matrix_exp")?;
                Ok(Self::from_tensor(crate::ops::matrix_exp(&dense).run()?))
            }
            Self::C64(value) => {
                let dense = Self::dense_primal_complex_only(value, "matrix_exp")?;
                Ok(Self::from_tensor(crate::ops::matrix_exp(&dense).run()?))
            }
        }
    }

    /// Runs eager AD norm on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.norm()?;
    /// ```
    pub fn norm(&self) -> Result<Self> {
        match self {
            Self::F32(value) => Ok(Self::F32(ad::norm(value)?)),
            Self::F64(value) => Ok(Self::F64(ad::norm(value)?)),
            _ => Err(real_only_error("norm", self.scalar_type())),
        }
    }
}
