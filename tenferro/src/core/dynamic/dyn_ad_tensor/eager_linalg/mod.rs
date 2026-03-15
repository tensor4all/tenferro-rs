use super::Tensor;
use crate::ops::ad;
use crate::{AdMode, Error, Result, ScalarType};
use num_complex::{Complex32, Complex64};

mod results;

pub use results::{
    EigResult, EigenResult, LstsqResult, LuResult, QrResult, SlogdetResult, SvdResult,
};

fn real_only_error(op: &'static str, dtype: ScalarType) -> Error {
    Error::InvalidAdTensor {
        message: format!("{op} currently requires a real Tensor input, got {dtype:?}"),
    }
}

fn primal_complex_only_error(op: &'static str) -> Error {
    Error::InvalidAdTensor {
        message: format!("{op} currently supports complex tensors only in primal mode"),
    }
}

fn same_dtype_error(op: &'static str, lhs: ScalarType, rhs: ScalarType) -> Error {
    Error::InvalidAdTensor {
        message: format!("{op} requires matching dtypes, got lhs={lhs:?}, rhs={rhs:?}"),
    }
}

impl Tensor {
    fn dense_primal_c32(
        value: &crate::AdTensor<Complex32>,
        op: &'static str,
    ) -> Result<tenferro_tensor::Tensor<Complex32>> {
        if value.mode() != AdMode::Primal {
            return Err(primal_complex_only_error(op));
        }
        value
            .structured_primal()
            .to_dense()
            .map_err(|e| Error::InvalidAdTensor {
                message: format!("{op} failed to densify structured complex input: {e}"),
            })
    }

    fn dense_primal_c64(
        value: &crate::AdTensor<Complex64>,
        op: &'static str,
    ) -> Result<tenferro_tensor::Tensor<Complex64>> {
        if value.mode() != AdMode::Primal {
            return Err(primal_complex_only_error(op));
        }
        value
            .structured_primal()
            .to_dense()
            .map_err(|e| Error::InvalidAdTensor {
                message: format!("{op} failed to densify structured complex input: {e}"),
            })
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
                let dense = Self::dense_primal_c32(value, "svd")?;
                let out = crate::ops::svd(&dense).run()?;
                Ok(SvdResult {
                    u: Tensor::from_tensor(out.u),
                    s: Tensor::from_tensor(out.s),
                    vt: Tensor::from_tensor(out.vt),
                })
            }
            Self::C64(value) => {
                let dense = Self::dense_primal_c64(value, "svd")?;
                let out = crate::ops::svd(&dense).run()?;
                Ok(SvdResult {
                    u: Tensor::from_tensor(out.u),
                    s: Tensor::from_tensor(out.s),
                    vt: Tensor::from_tensor(out.vt),
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
                let dense = Self::dense_primal_c32(value, "qr")?;
                let out = crate::ops::qr(&dense).run()?;
                Ok(QrResult {
                    q: Tensor::from_tensor(out.q),
                    r: Tensor::from_tensor(out.r),
                })
            }
            Self::C64(value) => {
                let dense = Self::dense_primal_c64(value, "qr")?;
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
            _ => Err(real_only_error("lu", self.scalar_type())),
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
            _ => Err(real_only_error("eigen", self.scalar_type())),
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
            _ => Err(real_only_error("cholesky", self.scalar_type())),
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
            _ => Err(real_only_error("inv", self.scalar_type())),
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
            _ => Err(real_only_error("matrix_exp", self.scalar_type())),
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
