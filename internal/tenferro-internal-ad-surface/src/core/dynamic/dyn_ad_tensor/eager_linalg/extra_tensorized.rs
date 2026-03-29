use super::{same_dtype_error, Tensor};
use crate::Result;

use super::extra::{with_dense_primal, with_dense_primal_pair};

impl Tensor {
    /// Builds a Vandermonde matrix using the default options.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let v = x.vander()?;
    /// ```
    pub fn vander(&self) -> Result<Self> {
        self.vander_with(None, false)
    }

    /// Builds a Vandermonde matrix with explicit column-count and order controls.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let v = x.vander_with(Some(4), true)?;
    /// ```
    pub fn vander_with(&self, columns: Option<usize>, increasing: bool) -> Result<Self> {
        match self {
            Self::F32(value) => with_dense_primal(value, "vander", |dense| {
                let mut builder = crate::ops::vander(dense).increasing(increasing);
                if let Some(columns) = columns {
                    builder = builder.columns(columns);
                }
                Ok(Self::from_tensor(builder.run()?))
            }),
            Self::F64(value) => with_dense_primal(value, "vander", |dense| {
                let mut builder = crate::ops::vander(dense).increasing(increasing);
                if let Some(columns) = columns {
                    builder = builder.columns(columns);
                }
                Ok(Self::from_tensor(builder.run()?))
            }),
            Self::C32(value) => with_dense_primal(value, "vander", |dense| {
                let mut builder = crate::ops::vander(dense).increasing(increasing);
                if let Some(columns) = columns {
                    builder = builder.columns(columns);
                }
                Ok(Self::from_tensor(builder.run()?))
            }),
            Self::C64(value) => with_dense_primal(value, "vander", |dense| {
                let mut builder = crate::ops::vander(dense).increasing(increasing);
                if let Some(columns) = columns {
                    builder = builder.columns(columns);
                }
                Ok(Self::from_tensor(builder.run()?))
            }),
        }
    }

    /// Inverts a tensorized linear map by splitting the first `ind` axes to the left.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let inv = x.tensorinv(2)?;
    /// ```
    pub fn tensorinv(&self, ind: usize) -> Result<Self> {
        match self {
            Self::F32(value) => with_dense_primal(value, "tensorinv", |dense| {
                Ok(Self::from_tensor(
                    crate::ops::tensorinv(dense).ind(ind).run()?,
                ))
            }),
            Self::F64(value) => with_dense_primal(value, "tensorinv", |dense| {
                Ok(Self::from_tensor(
                    crate::ops::tensorinv(dense).ind(ind).run()?,
                ))
            }),
            Self::C32(value) => with_dense_primal(value, "tensorinv", |dense| {
                Ok(Self::from_tensor(
                    crate::ops::tensorinv(dense).ind(ind).run()?,
                ))
            }),
            Self::C64(value) => with_dense_primal(value, "tensorinv", |dense| {
                Ok(Self::from_tensor(
                    crate::ops::tensorinv(dense).ind(ind).run()?,
                ))
            }),
        }
    }

    /// Solves a tensorized linear system using the default axis ordering.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = a.tensorsolve(&b)?;
    /// ```
    pub fn tensorsolve(&self, rhs: &Self) -> Result<Self> {
        self.tensorsolve_with_dims(rhs, &[])
    }

    /// Solves a tensorized linear system after moving the listed axes before solving.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let x = a.tensorsolve_with_dims(&b, &[3, 2])?;
    /// ```
    pub fn tensorsolve_with_dims(&self, rhs: &Self, dims: &[usize]) -> Result<Self> {
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "tensorsolve", |a, b| {
                    let mut builder = crate::ops::tensorsolve(a, b);
                    if !dims.is_empty() {
                        builder = builder.dims(dims);
                    }
                    Ok(Self::from_tensor(builder.run()?))
                })
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "tensorsolve", |a, b| {
                    let mut builder = crate::ops::tensorsolve(a, b);
                    if !dims.is_empty() {
                        builder = builder.dims(dims);
                    }
                    Ok(Self::from_tensor(builder.run()?))
                })
            }
            (Self::C32(lhs), Self::C32(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "tensorsolve", |a, b| {
                    let mut builder = crate::ops::tensorsolve(a, b);
                    if !dims.is_empty() {
                        builder = builder.dims(dims);
                    }
                    Ok(Self::from_tensor(builder.run()?))
                })
            }
            (Self::C64(lhs), Self::C64(rhs)) => {
                with_dense_primal_pair(lhs, rhs, "tensorsolve", |a, b| {
                    let mut builder = crate::ops::tensorsolve(a, b);
                    if !dims.is_empty() {
                        builder = builder.dims(dims);
                    }
                    Ok(Self::from_tensor(builder.run()?))
                })
            }
            (lhs, rhs) => Err(same_dtype_error(
                "tensorsolve",
                lhs.scalar_type(),
                rhs.scalar_type(),
            )),
        }
    }
}
