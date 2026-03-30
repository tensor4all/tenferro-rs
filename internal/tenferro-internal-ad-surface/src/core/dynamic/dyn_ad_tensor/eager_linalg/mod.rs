use super::{accessors::TypedTensorBorrowTyped, Tensor, TypedTensorRef};
use crate::ops::ad;
use crate::{AdMode, Error, Result, ScalarType};
use tenferro_algebra::Scalar;
use tenferro_internal_ad_core::{AdTensor, DynAdTensorRef};

mod extra;
mod extra_tensorized;
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

macro_rules! match_real_dyn_ad_tensor_ref {
    ($tensor:expr, $tensor_ref:ident, $op:literal, $body:expr) => {{
        let $tensor_ref = $tensor.as_dyn_ad_ref();
        match $tensor_ref {
            DynAdTensorRef::F32(_) | DynAdTensorRef::F64(_) => $body,
            _ => Err(real_only_error($op, $tensor_ref.scalar_type())),
        }
    }};
}

macro_rules! match_real_or_complex_primal_only_dyn_ad_tensor_ref {
    ($tensor:expr, $tensor_ref:ident, $op:literal, $real_body:expr, |$value:ident| $complex_body:block) => {{
        let $tensor_ref = $tensor.as_dyn_ad_ref();
        match $tensor_ref {
            DynAdTensorRef::F32(_) | DynAdTensorRef::F64(_) => $real_body,
            DynAdTensorRef::C32($value) => $complex_body,
            DynAdTensorRef::C64($value) => $complex_body,
        }
    }};
}

impl Tensor {
    fn dense_primal_with_mode_error<T>(
        value: &AdTensor<T>,
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
                message: format!("{op} failed to densify structured tensor input: {e}"),
            })
    }

    pub(super) fn dense_primal_complex_only<T>(
        value: &AdTensor<T>,
        op: &'static str,
    ) -> Result<tenferro_tensor::Tensor<T>>
    where
        T: Scalar + crate::DynTensorTyped + crate::runtime::contracts::LinalgRuntimeValue,
    {
        Self::dense_primal_with_mode_error(value, op, primal_complex_only_error)
    }

    fn dense_primal_only_typed<T>(
        value: TypedTensorRef<'_, T>,
        op: &'static str,
    ) -> Result<tenferro_tensor::Tensor<T>>
    where
        T: Scalar
            + TypedTensorBorrowTyped
            + crate::DynTensorTyped
            + crate::runtime::contracts::LinalgRuntimeValue,
    {
        if !value.is_dense() {
            return Err(Error::UnsupportedStructuredLinalg { op });
        }
        if value.mode() != AdMode::Primal {
            return Err(primal_only_error(op));
        }
        value
            .structured_primal()
            .to_dense()
            .map_err(|e| Error::InvalidAdTensor {
                message: format!("{op} failed to densify structured tensor input: {e}"),
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
        let out = ad::svd_dyn(self.as_dyn_ad_ref())?;
        Ok(SvdResult {
            u: out.u.into(),
            s: out.s.into(),
            vt: out.vt.into(),
        })
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
        match_real_or_complex_primal_only_dyn_ad_tensor_ref!(
            self,
            tensor_ref,
            "qr",
            {
                let out = ad::qr_dyn(tensor_ref)?;
                Ok(QrResult {
                    q: out.q.into(),
                    r: out.r.into(),
                })
            },
            |value| {
                let dense = Self::dense_primal_complex_only(value, "qr")?;
                let out = crate::ops::qr(&dense).run()?;
                Ok(QrResult {
                    q: Tensor::from_tensor(out.q),
                    r: Tensor::from_tensor(out.r),
                })
            }
        )
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
        match_real_or_complex_primal_only_dyn_ad_tensor_ref!(
            self,
            tensor_ref,
            "lu",
            {
                let out = ad::lu_dyn(tensor_ref)?;
                Ok(LuResult {
                    p: out.p.into(),
                    l: out.l.into(),
                    u: out.u.into(),
                })
            },
            |value| {
                let dense = Self::dense_primal_complex_only(value, "lu")?;
                let out = crate::ops::lu(&dense).run()?;
                Ok(LuResult {
                    p: out.p.into(),
                    l: Tensor::from_tensor(out.l),
                    u: Tensor::from_tensor(out.u),
                })
            }
        )
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
        match_real_or_complex_primal_only_dyn_ad_tensor_ref!(
            self,
            tensor_ref,
            "eigen",
            {
                let out = ad::eigen_dyn(tensor_ref)?;
                Ok(EigenResult {
                    values: out.values.into(),
                    vectors: out.vectors.into(),
                })
            },
            |value| {
                let dense = Self::dense_primal_complex_only(value, "eigen")?;
                let out = crate::ops::eigen(&dense).run()?;
                Ok(EigenResult {
                    values: Tensor::from_tensor(out.values),
                    vectors: Tensor::from_tensor(out.vectors),
                })
            }
        )
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
        match self.as_dyn_ad_ref() {
            DynAdTensorRef::F32(value) => {
                let out = ad::eig_dyn(value.into())?;
                Ok(EigResult {
                    values: out.values.into(),
                    vectors: out.vectors.into(),
                })
            }
            DynAdTensorRef::F64(value) => {
                let out = ad::eig_dyn(value.into())?;
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
        match (self.as_dyn_ad_ref(), rhs.as_dyn_ad_ref()) {
            (DynAdTensorRef::F32(_), DynAdTensorRef::F32(_))
            | (DynAdTensorRef::F64(_), DynAdTensorRef::F64(_)) => {
                let out = ad::lstsq_dyn(self.as_dyn_ad_ref(), rhs.as_dyn_ad_ref())?;
                Ok(LstsqResult {
                    x: out.x.into(),
                    residual: out.residual.into(),
                })
            }
            (lhs @ DynAdTensorRef::F32(_), rhs @ DynAdTensorRef::F64(_))
            | (lhs @ DynAdTensorRef::F64(_), rhs @ DynAdTensorRef::F32(_)) => Err(
                same_dtype_error("lstsq", lhs.scalar_type(), rhs.scalar_type()),
            ),
            (lhs, rhs) => Err(Error::InvalidAdTensor {
                message: format!(
                    "lstsq requires real-valued operands, got lhs={:?}, rhs={:?}",
                    lhs.scalar_type(),
                    rhs.scalar_type()
                ),
            }),
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
        match_real_or_complex_primal_only_dyn_ad_tensor_ref!(
            self,
            tensor_ref,
            "cholesky",
            Ok(ad::cholesky_dyn(tensor_ref)?.into()),
            |value| {
                let dense = Self::dense_primal_complex_only(value, "cholesky")?;
                Ok(Self::from_tensor(crate::ops::cholesky(&dense).run()?))
            }
        )
    }

    /// Runs eager AD linear solve on dynamic tensors.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = a.solve(&b)?;
    /// ```
    pub fn solve(&self, rhs: &Self) -> Result<Self> {
        match (self.as_dyn_ad_ref(), rhs.as_dyn_ad_ref()) {
            (DynAdTensorRef::F32(_), DynAdTensorRef::F32(_))
            | (DynAdTensorRef::F64(_), DynAdTensorRef::F64(_))
            | (DynAdTensorRef::C32(_), DynAdTensorRef::C32(_))
            | (DynAdTensorRef::C64(_), DynAdTensorRef::C64(_)) => {
                Ok(ad::solve_dyn(self.as_dyn_ad_ref(), rhs.as_dyn_ad_ref())?.into())
            }
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
        match (self.as_dyn_ad_ref(), rhs.as_dyn_ad_ref()) {
            (DynAdTensorRef::F32(_), DynAdTensorRef::F32(_))
            | (DynAdTensorRef::F64(_), DynAdTensorRef::F64(_))
            | (DynAdTensorRef::C32(_), DynAdTensorRef::C32(_))
            | (DynAdTensorRef::C64(_), DynAdTensorRef::C64(_)) => {
                Ok(ad::solve_triangular_dyn(self.as_dyn_ad_ref(), rhs.as_dyn_ad_ref())?.into())
            }
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
        match_real_or_complex_primal_only_dyn_ad_tensor_ref!(
            self,
            tensor_ref,
            "inv",
            Ok(ad::inv_dyn(tensor_ref)?.into()),
            |value| {
                let dense = Self::dense_primal_complex_only(value, "inv")?;
                Ok(Self::from_tensor(crate::ops::inv(&dense).run()?))
            }
        )
    }

    /// Runs eager AD determinant on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let det = x.det()?;
    /// ```
    pub fn det(&self) -> Result<Self> {
        match_real_dyn_ad_tensor_ref!(self, tensor_ref, "det", Ok(ad::det_dyn(tensor_ref)?.into()))
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
        match_real_dyn_ad_tensor_ref!(self, tensor_ref, "slogdet", {
            let out = ad::slogdet_dyn(tensor_ref)?;
            Ok(SlogdetResult {
                sign: out.sign.into(),
                logabsdet: out.logabsdet.into(),
            })
        })
    }

    /// Runs eager AD pseudoinverse on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let pinv = x.pinv()?;
    /// ```
    pub fn pinv(&self) -> Result<Self> {
        match_real_dyn_ad_tensor_ref!(
            self,
            tensor_ref,
            "pinv",
            Ok(ad::pinv_dyn(tensor_ref)?.into())
        )
    }

    /// Runs eager AD matrix exponential on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.matrix_exp()?;
    /// ```
    pub fn matrix_exp(&self) -> Result<Self> {
        match_real_or_complex_primal_only_dyn_ad_tensor_ref!(
            self,
            tensor_ref,
            "matrix_exp",
            Ok(ad::matrix_exp_dyn(tensor_ref)?.into()),
            |value| {
                let dense = Self::dense_primal_complex_only(value, "matrix_exp")?;
                Ok(Self::from_tensor(crate::ops::matrix_exp(&dense).run()?))
            }
        )
    }

    /// Runs eager AD norm on a dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.norm()?;
    /// ```
    pub fn norm(&self) -> Result<Self> {
        match_real_dyn_ad_tensor_ref!(
            self,
            tensor_ref,
            "norm",
            Ok(ad::norm_dyn(tensor_ref)?.into())
        )
    }
}
