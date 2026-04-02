use super::super::accessors::TypedTensorBorrowTyped;
use super::{
    same_dtype_error, CholeskyExResult, InvExResult, LuFactorExResult, LuFactorResult,
    SolveExResult, Tensor,
};
use crate::ops::ad;
use crate::{DynTensorTyped, Error, Result, TypedTensorRef};
use tenferro_algebra::Scalar;
use tenferro_internal_ad_core::DynAdTensorRef;
use tenferro_internal_ad_linalg::results::{
    DynCholeskyExResult, DynInvExResult, DynLuFactorExResult, DynLuFactorResult, DynSolveExResult,
};
use tenferro_tensor::Tensor as DenseTensor;

macro_rules! match_same_dtype_dyn_ad_tensor_ref_pair {
    ($lhs:expr, $rhs:expr, $op:literal, |$lhs_ref:ident, $rhs_ref:ident| $body:block) => {{
        let $lhs_ref = $lhs.as_dyn_ad_ref();
        let $rhs_ref = $rhs.as_dyn_ad_ref();
        match ($lhs_ref, $rhs_ref) {
            (DynAdTensorRef::F32(_), DynAdTensorRef::F32(_))
            | (DynAdTensorRef::F64(_), DynAdTensorRef::F64(_))
            | (DynAdTensorRef::C32(_), DynAdTensorRef::C32(_))
            | (DynAdTensorRef::C64(_), DynAdTensorRef::C64(_)) => $body,
            (lhs, rhs) => Err(same_dtype_error($op, lhs.scalar_type(), rhs.scalar_type())),
        }
    }};
}

pub(super) fn with_dense_primal_typed<T, R, F>(
    value: TypedTensorRef<'_, T>,
    op: &'static str,
    f: F,
) -> Result<R>
where
    T: Scalar
        + TypedTensorBorrowTyped
        + DynTensorTyped
        + crate::runtime::contracts::LinalgRuntimeValue,
    F: FnOnce(&tenferro_tensor::Tensor<T>) -> Result<R>,
{
    let dense = Tensor::dense_primal_only_typed(value, op)?;
    f(&dense)
}

pub(super) fn with_dense_primal_pair_typed<T, R, F>(
    lhs: TypedTensorRef<'_, T>,
    rhs: TypedTensorRef<'_, T>,
    op: &'static str,
    f: F,
) -> Result<R>
where
    T: Scalar
        + TypedTensorBorrowTyped
        + DynTensorTyped
        + crate::runtime::contracts::LinalgRuntimeValue,
    F: FnOnce(&tenferro_tensor::Tensor<T>, &tenferro_tensor::Tensor<T>) -> Result<R>,
{
    let lhs = Tensor::dense_primal_only_typed(lhs, op)?;
    let rhs = Tensor::dense_primal_only_typed(rhs, op)?;
    f(&lhs, &rhs)
}

fn map_lu_factor_result(out: DynLuFactorResult) -> LuFactorResult {
    LuFactorResult {
        factors: out.factors.into(),
        pivots: out.pivots,
    }
}

fn map_lu_factor_ex_result(out: DynLuFactorExResult) -> LuFactorExResult {
    LuFactorExResult {
        factors: out.factors.into(),
        pivots: out.pivots,
        info: out.info,
    }
}

fn map_solve_ex_result(out: DynSolveExResult) -> SolveExResult {
    SolveExResult {
        solution: out.solution.into(),
        info: out.info,
    }
}

fn map_inv_ex_result(out: DynInvExResult) -> InvExResult {
    InvExResult {
        inverse: out.inverse.into(),
        info: out.info,
    }
}

fn map_cholesky_ex_result(out: DynCholeskyExResult) -> CholeskyExResult {
    CholeskyExResult {
        l: out.l.into(),
        info: out.info,
    }
}

impl Tensor {
    /// Computes an LU factorization and returns the packed factors plus pivots.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.lu_factor()?;
    /// let _pivots = &out.pivots;
    /// ```
    pub fn lu_factor(&self) -> Result<LuFactorResult> {
        Ok(map_lu_factor_result(ad::lu_factor_dyn(
            self.as_dyn_ad_ref(),
        )?))
    }

    /// Computes an LU factorization with numerical status information.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.lu_factor_ex()?;
    /// let _info = &out.info;
    /// ```
    pub fn lu_factor_ex(&self) -> Result<LuFactorExResult> {
        Ok(map_lu_factor_ex_result(ad::lu_factor_ex_dyn(
            self.as_dyn_ad_ref(),
        )?))
    }

    /// Solves `LU x = b` using pre-factorized packed LU factors and pivots.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = factors.lu_solve(&rhs, &pivots)?;
    /// ```
    pub fn lu_solve(&self, rhs: &Self, pivots: &DenseTensor<i32>) -> Result<Self> {
        match_same_dtype_dyn_ad_tensor_ref_pair!(self, rhs, "lu_solve", |lhs_ref, rhs_ref| {
            Ok(ad::lu_solve_dyn(lhs_ref, rhs_ref, pivots)?.into())
        })
    }

    /// Solves a linear system and returns numerical status information.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = a.solve_ex(&b)?;
    /// let _solution = &out.solution;
    /// ```
    pub fn solve_ex(&self, rhs: &Self) -> Result<SolveExResult> {
        match_same_dtype_dyn_ad_tensor_ref_pair!(self, rhs, "solve_ex", |lhs_ref, rhs_ref| {
            Ok(map_solve_ex_result(ad::solve_ex_dyn(lhs_ref, rhs_ref)?))
        })
    }

    /// Computes an inverse with numerical status information.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.inv_ex()?;
    /// let _inverse = &out.inverse;
    /// ```
    pub fn inv_ex(&self) -> Result<InvExResult> {
        Ok(map_inv_ex_result(ad::inv_ex_dyn(self.as_dyn_ad_ref())?))
    }

    /// Computes a Cholesky factorization with numerical status information.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = x.cholesky_ex()?;
    /// let _factor = &out.l;
    /// ```
    pub fn cholesky_ex(&self) -> Result<CholeskyExResult> {
        Ok(map_cholesky_ex_result(ad::cholesky_ex_dyn(
            self.as_dyn_ad_ref(),
        )?))
    }

    /// Raises a square matrix to an integer power.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let squared = x.matrix_power(2)?;
    /// ```
    pub fn matrix_power(&self, exponent: i64) -> Result<Self> {
        match self.as_dyn_ad_ref() {
            DynAdTensorRef::F32(_) => {
                let value = self.as_f32().ok_or_else(|| Error::InvalidAdTensor {
                    message: "matrix_power: internal type mismatch after matching F32".into(),
                })?;
                with_dense_primal_typed(value, "matrix_power", |dense| {
                    Ok(Self::from_tensor(
                        crate::ops::matrix_power(dense).exponent(exponent).run()?,
                    ))
                })
            }
            DynAdTensorRef::F64(_) => {
                let value = self.as_f64().ok_or_else(|| Error::InvalidAdTensor {
                    message: "matrix_power: internal type mismatch after matching F64".into(),
                })?;
                with_dense_primal_typed(value, "matrix_power", |dense| {
                    Ok(Self::from_tensor(
                        crate::ops::matrix_power(dense).exponent(exponent).run()?,
                    ))
                })
            }
            DynAdTensorRef::C32(_) => {
                let value = self.as_c32().ok_or_else(|| Error::InvalidAdTensor {
                    message: "matrix_power: internal type mismatch after matching C32".into(),
                })?;
                with_dense_primal_typed(value, "matrix_power", |dense| {
                    Ok(Self::from_tensor(
                        crate::ops::matrix_power(dense).exponent(exponent).run()?,
                    ))
                })
            }
            DynAdTensorRef::C64(_) => {
                let value = self.as_c64().ok_or_else(|| Error::InvalidAdTensor {
                    message: "matrix_power: internal type mismatch after matching C64".into(),
                })?;
                with_dense_primal_typed(value, "matrix_power", |dense| {
                    Ok(Self::from_tensor(
                        crate::ops::matrix_power(dense).exponent(exponent).run()?,
                    ))
                })
            }
        }
    }

    /// Computes the matrix condition number with the default spectral norm.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let value = x.cond()?;
    /// ```
    pub fn cond(&self) -> Result<Self> {
        if let Some(value) = self.as_f32() {
            return with_dense_primal_typed(value, "cond", |dense| {
                Ok(Self::from_tensor(crate::ops::cond(dense).run()?))
            });
        }
        if let Some(value) = self.as_f64() {
            return with_dense_primal_typed(value, "cond", |dense| {
                Ok(Self::from_tensor(crate::ops::cond(dense).run()?))
            });
        }
        Err(super::real_only_error("cond", self.scalar_type()))
    }

    /// Computes the vector cross product.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let z = x.cross(&y)?;
    /// ```
    pub fn cross(&self, rhs: &Self) -> Result<Self> {
        if let (Some(lhs), Some(rhs)) = (self.as_f32(), rhs.as_f32()) {
            return with_dense_primal_pair_typed(lhs, rhs, "cross", |a, b| {
                Ok(Self::from_tensor(crate::ops::cross(a, b).run()?))
            });
        }
        if let (Some(lhs), Some(rhs)) = (self.as_f64(), rhs.as_f64()) {
            return with_dense_primal_pair_typed(lhs, rhs, "cross", |a, b| {
                Ok(Self::from_tensor(crate::ops::cross(a, b).run()?))
            });
        }
        if let (Some(lhs), Some(rhs)) = (self.as_c32(), rhs.as_c32()) {
            return with_dense_primal_pair_typed(lhs, rhs, "cross", |a, b| {
                Ok(Self::from_tensor(crate::ops::cross(a, b).run()?))
            });
        }
        if let (Some(lhs), Some(rhs)) = (self.as_c64(), rhs.as_c64()) {
            return with_dense_primal_pair_typed(lhs, rhs, "cross", |a, b| {
                Ok(Self::from_tensor(crate::ops::cross(a, b).run()?))
            });
        }
        Err(same_dtype_error(
            "cross",
            self.scalar_type(),
            rhs.scalar_type(),
        ))
    }

    /// Forms the orthogonal/unitary matrix from Householder reflectors and `tau`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let q = reflectors.householder_product(&tau)?;
    /// ```
    pub fn householder_product(&self, tau: &Self) -> Result<Self> {
        if let (Some(lhs), Some(rhs)) = (self.as_f32(), tau.as_f32()) {
            return with_dense_primal_pair_typed(lhs, rhs, "householder_product", |a, b| {
                Ok(Self::from_tensor(
                    crate::ops::householder_product(a, b).run()?,
                ))
            });
        }
        if let (Some(lhs), Some(rhs)) = (self.as_f64(), tau.as_f64()) {
            return with_dense_primal_pair_typed(lhs, rhs, "householder_product", |a, b| {
                Ok(Self::from_tensor(
                    crate::ops::householder_product(a, b).run()?,
                ))
            });
        }
        if let (Some(lhs), Some(rhs)) = (self.as_c32(), tau.as_c32()) {
            return with_dense_primal_pair_typed(lhs, rhs, "householder_product", |a, b| {
                Ok(Self::from_tensor(
                    crate::ops::householder_product(a, b).run()?,
                ))
            });
        }
        if let (Some(lhs), Some(rhs)) = (self.as_c64(), tau.as_c64()) {
            return with_dense_primal_pair_typed(lhs, rhs, "householder_product", |a, b| {
                Ok(Self::from_tensor(
                    crate::ops::householder_product(a, b).run()?,
                ))
            });
        }
        Err(same_dtype_error(
            "householder_product",
            self.scalar_type(),
            tau.scalar_type(),
        ))
    }
}
