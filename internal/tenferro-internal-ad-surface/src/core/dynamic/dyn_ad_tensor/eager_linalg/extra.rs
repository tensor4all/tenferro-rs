use super::{
    same_dtype_error, CholeskyExResult, InvExResult, LuFactorExResult, LuFactorResult,
    SolveExResult, Tensor,
};
use crate::{AdTensor, DynTensorTyped, Result};
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor as DenseTensor;

pub(super) fn with_dense_primal<T, R, F>(value: &AdTensor<T>, op: &'static str, f: F) -> Result<R>
where
    T: Scalar + DynTensorTyped + crate::runtime::contracts::LinalgRuntimeValue,
    F: FnOnce(&tenferro_tensor::Tensor<T>) -> Result<R>,
{
    let dense = Tensor::dense_primal_only(value, op)?;
    f(&dense)
}

pub(super) fn with_dense_primal_pair<T, R, F>(
    lhs: &AdTensor<T>,
    rhs: &AdTensor<T>,
    op: &'static str,
    f: F,
) -> Result<R>
where
    T: Scalar + DynTensorTyped + crate::runtime::contracts::LinalgRuntimeValue,
    F: FnOnce(&tenferro_tensor::Tensor<T>, &tenferro_tensor::Tensor<T>) -> Result<R>,
{
    let lhs = Tensor::dense_primal_only(lhs, op)?;
    let rhs = Tensor::dense_primal_only(rhs, op)?;
    f(&lhs, &rhs)
}

fn map_lu_factor_result<T>(out: tenferro_linalg::LuFactorResult<T>) -> LuFactorResult
where
    T: Scalar + DynTensorTyped,
    AdTensor<T>: Into<Tensor>,
{
    LuFactorResult {
        factors: Tensor::from_tensor(out.factors),
        pivots: out.pivots,
    }
}

fn map_lu_factor_ex_result<T>(out: tenferro_linalg::LuFactorExResult<T>) -> LuFactorExResult
where
    T: Scalar + DynTensorTyped,
    AdTensor<T>: Into<Tensor>,
{
    LuFactorExResult {
        factors: Tensor::from_tensor(out.factors),
        pivots: out.pivots,
        info: out.info,
    }
}

fn map_solve_ex_result<T>(out: tenferro_linalg::SolveExResult<T>) -> SolveExResult
where
    T: Scalar + DynTensorTyped,
    AdTensor<T>: Into<Tensor>,
{
    SolveExResult {
        solution: Tensor::from_tensor(out.solution),
        info: out.info,
    }
}

fn map_inv_ex_result<T>(out: tenferro_linalg::InvExResult<T>) -> InvExResult
where
    T: Scalar + DynTensorTyped,
    AdTensor<T>: Into<Tensor>,
{
    InvExResult {
        inverse: Tensor::from_tensor(out.inverse),
        info: out.info,
    }
}

fn map_cholesky_ex_result<T>(out: tenferro_linalg::CholeskyExResult<T>) -> CholeskyExResult
where
    T: Scalar + DynTensorTyped,
    AdTensor<T>: Into<Tensor>,
{
    CholeskyExResult {
        l: Tensor::from_tensor(out.l),
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
        match self {
            Self::F32(value) => with_dense_primal(value, "lu_factor", |dense| {
                Ok(map_lu_factor_result(crate::ops::lu_factor(dense).run()?))
            }),
            Self::F64(value) => with_dense_primal(value, "lu_factor", |dense| {
                Ok(map_lu_factor_result(crate::ops::lu_factor(dense).run()?))
            }),
            Self::C32(value) => with_dense_primal(value, "lu_factor", |dense| {
                Ok(map_lu_factor_result(crate::ops::lu_factor(dense).run()?))
            }),
            Self::C64(value) => with_dense_primal(value, "lu_factor", |dense| {
                Ok(map_lu_factor_result(crate::ops::lu_factor(dense).run()?))
            }),
        }
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
        match self {
            Self::F32(value) => with_dense_primal(value, "lu_factor_ex", |dense| {
                Ok(map_lu_factor_ex_result(
                    crate::ops::lu_factor_ex(dense).run()?,
                ))
            }),
            Self::F64(value) => with_dense_primal(value, "lu_factor_ex", |dense| {
                Ok(map_lu_factor_ex_result(
                    crate::ops::lu_factor_ex(dense).run()?,
                ))
            }),
            Self::C32(value) => with_dense_primal(value, "lu_factor_ex", |dense| {
                Ok(map_lu_factor_ex_result(
                    crate::ops::lu_factor_ex(dense).run()?,
                ))
            }),
            Self::C64(value) => with_dense_primal(value, "lu_factor_ex", |dense| {
                Ok(map_lu_factor_ex_result(
                    crate::ops::lu_factor_ex(dense).run()?,
                ))
            }),
        }
    }

    /// Solves `LU x = b` using pre-factorized packed LU factors and pivots.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = factors.lu_solve(&rhs, &pivots)?;
    /// ```
    pub fn lu_solve(&self, rhs: &Self, pivots: &DenseTensor<i32>) -> Result<Self> {
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "lu_solve", |factors, rhs| {
                    Ok(Self::from_tensor(
                        crate::ops::lu_solve(factors, rhs).pivots(pivots).run()?,
                    ))
                })
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "lu_solve", |factors, rhs| {
                    Ok(Self::from_tensor(
                        crate::ops::lu_solve(factors, rhs).pivots(pivots).run()?,
                    ))
                })
            }
            (Self::C32(lhs), Self::C32(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "lu_solve", |factors, rhs| {
                    Ok(Self::from_tensor(
                        crate::ops::lu_solve(factors, rhs).pivots(pivots).run()?,
                    ))
                })
            }
            (Self::C64(lhs), Self::C64(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "lu_solve", |factors, rhs| {
                    Ok(Self::from_tensor(
                        crate::ops::lu_solve(factors, rhs).pivots(pivots).run()?,
                    ))
                })
            }
            (lhs, rhs) => Err(same_dtype_error(
                "lu_solve",
                lhs.scalar_type(),
                rhs.scalar_type(),
            )),
        }
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
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "solve_ex", |a, b| {
                    Ok(map_solve_ex_result(crate::ops::solve_ex(a, b).run()?))
                })
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "solve_ex", |a, b| {
                    Ok(map_solve_ex_result(crate::ops::solve_ex(a, b).run()?))
                })
            }
            (Self::C32(lhs), Self::C32(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "solve_ex", |a, b| {
                    Ok(map_solve_ex_result(crate::ops::solve_ex(a, b).run()?))
                })
            }
            (Self::C64(lhs), Self::C64(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "solve_ex", |a, b| {
                    Ok(map_solve_ex_result(crate::ops::solve_ex(a, b).run()?))
                })
            }
            (lhs, rhs) => Err(same_dtype_error(
                "solve_ex",
                lhs.scalar_type(),
                rhs.scalar_type(),
            )),
        }
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
        match self {
            Self::F32(value) => with_dense_primal(value, "inv_ex", |dense| {
                Ok(map_inv_ex_result(crate::ops::inv_ex(dense).run()?))
            }),
            Self::F64(value) => with_dense_primal(value, "inv_ex", |dense| {
                Ok(map_inv_ex_result(crate::ops::inv_ex(dense).run()?))
            }),
            Self::C32(value) => with_dense_primal(value, "inv_ex", |dense| {
                Ok(map_inv_ex_result(crate::ops::inv_ex(dense).run()?))
            }),
            Self::C64(value) => with_dense_primal(value, "inv_ex", |dense| {
                Ok(map_inv_ex_result(crate::ops::inv_ex(dense).run()?))
            }),
        }
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
        match self {
            Self::F32(value) => with_dense_primal(value, "cholesky_ex", |dense| {
                Ok(map_cholesky_ex_result(
                    crate::ops::cholesky_ex(dense).run()?,
                ))
            }),
            Self::F64(value) => with_dense_primal(value, "cholesky_ex", |dense| {
                Ok(map_cholesky_ex_result(
                    crate::ops::cholesky_ex(dense).run()?,
                ))
            }),
            Self::C32(value) => with_dense_primal(value, "cholesky_ex", |dense| {
                Ok(map_cholesky_ex_result(
                    crate::ops::cholesky_ex(dense).run()?,
                ))
            }),
            Self::C64(value) => with_dense_primal(value, "cholesky_ex", |dense| {
                Ok(map_cholesky_ex_result(
                    crate::ops::cholesky_ex(dense).run()?,
                ))
            }),
        }
    }

    /// Raises a square matrix to an integer power.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let squared = x.matrix_power(2)?;
    /// ```
    pub fn matrix_power(&self, exponent: i64) -> Result<Self> {
        match self {
            Self::F32(value) => with_dense_primal(value, "matrix_power", |dense| {
                Ok(Self::from_tensor(
                    crate::ops::matrix_power(dense).exponent(exponent).run()?,
                ))
            }),
            Self::F64(value) => with_dense_primal(value, "matrix_power", |dense| {
                Ok(Self::from_tensor(
                    crate::ops::matrix_power(dense).exponent(exponent).run()?,
                ))
            }),
            Self::C32(value) => with_dense_primal(value, "matrix_power", |dense| {
                Ok(Self::from_tensor(
                    crate::ops::matrix_power(dense).exponent(exponent).run()?,
                ))
            }),
            Self::C64(value) => with_dense_primal(value, "matrix_power", |dense| {
                Ok(Self::from_tensor(
                    crate::ops::matrix_power(dense).exponent(exponent).run()?,
                ))
            }),
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
        match self {
            Self::F32(value) => with_dense_primal(value, "cond", |dense| {
                Ok(Self::from_tensor(crate::ops::cond(dense).run()?))
            }),
            Self::F64(value) => with_dense_primal(value, "cond", |dense| {
                Ok(Self::from_tensor(crate::ops::cond(dense).run()?))
            }),
            _ => Err(super::real_only_error("cond", self.scalar_type())),
        }
    }

    /// Computes the vector cross product.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let z = x.cross(&y)?;
    /// ```
    pub fn cross(&self, rhs: &Self) -> Result<Self> {
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "cross", |a, b| {
                    Ok(Self::from_tensor(crate::ops::cross(a, b).run()?))
                })
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "cross", |a, b| {
                    Ok(Self::from_tensor(crate::ops::cross(a, b).run()?))
                })
            }
            (Self::C32(lhs), Self::C32(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "cross", |a, b| {
                    Ok(Self::from_tensor(crate::ops::cross(a, b).run()?))
                })
            }
            (Self::C64(lhs), Self::C64(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "cross", |a, b| {
                    Ok(Self::from_tensor(crate::ops::cross(a, b).run()?))
                })
            }
            (lhs, rhs) => Err(same_dtype_error(
                "cross",
                lhs.scalar_type(),
                rhs.scalar_type(),
            )),
        }
    }

    /// Forms the orthogonal/unitary matrix from Householder reflectors and `tau`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let q = reflectors.householder_product(&tau)?;
    /// ```
    pub fn householder_product(&self, tau: &Self) -> Result<Self> {
        match (self, tau) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "householder_product", |a, b| {
                    Ok(Self::from_tensor(
                        crate::ops::householder_product(a, b).run()?,
                    ))
                })
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "householder_product", |a, b| {
                    Ok(Self::from_tensor(
                        crate::ops::householder_product(a, b).run()?,
                    ))
                })
            }
            (Self::C32(lhs), Self::C32(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "householder_product", |a, b| {
                    Ok(Self::from_tensor(
                        crate::ops::householder_product(a, b).run()?,
                    ))
                })
            }
            (Self::C64(lhs), Self::C64(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "householder_product", |a, b| {
                    Ok(Self::from_tensor(
                        crate::ops::householder_product(a, b).run()?,
                    ))
                })
            }
            (lhs, rhs) => Err(same_dtype_error(
                "householder_product",
                lhs.scalar_type(),
                rhs.scalar_type(),
            )),
        }
    }
}
