use super::{same_dtype_error, Tensor};
use crate::Result;

use super::extra::{with_dense_primal_pair_typed, with_dense_primal_typed};

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
        if let Some(value) = self.as_f32() {
            return with_dense_primal_typed(value, "vander", |dense| {
                let mut builder = crate::ops::vander(dense).increasing(increasing);
                if let Some(columns) = columns {
                    builder = builder.columns(columns);
                }
                Ok(Self::from_tensor(builder.run()?))
            });
        }
        if let Some(value) = self.as_f64() {
            return with_dense_primal_typed(value, "vander", |dense| {
                let mut builder = crate::ops::vander(dense).increasing(increasing);
                if let Some(columns) = columns {
                    builder = builder.columns(columns);
                }
                Ok(Self::from_tensor(builder.run()?))
            });
        }
        if let Some(value) = self.as_c32() {
            return with_dense_primal_typed(value, "vander", |dense| {
                let mut builder = crate::ops::vander(dense).increasing(increasing);
                if let Some(columns) = columns {
                    builder = builder.columns(columns);
                }
                Ok(Self::from_tensor(builder.run()?))
            });
        }
        if let Some(value) = self.as_c64() {
            return with_dense_primal_typed(value, "vander", |dense| {
                let mut builder = crate::ops::vander(dense).increasing(increasing);
                if let Some(columns) = columns {
                    builder = builder.columns(columns);
                }
                Ok(Self::from_tensor(builder.run()?))
            });
        }
        unreachable!("Tensor::vander_with should have one of the supported scalar dtypes")
    }

    /// Inverts a tensorized linear map by splitting the first `ind` axes to the left.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let inv = x.tensorinv(2)?;
    /// ```
    pub fn tensorinv(&self, ind: usize) -> Result<Self> {
        if let Some(value) = self.as_f32() {
            return with_dense_primal_typed(value, "tensorinv", |dense| {
                Ok(Self::from_tensor(
                    crate::ops::tensorinv(dense).ind(ind).run()?,
                ))
            });
        }
        if let Some(value) = self.as_f64() {
            return with_dense_primal_typed(value, "tensorinv", |dense| {
                Ok(Self::from_tensor(
                    crate::ops::tensorinv(dense).ind(ind).run()?,
                ))
            });
        }
        if let Some(value) = self.as_c32() {
            return with_dense_primal_typed(value, "tensorinv", |dense| {
                Ok(Self::from_tensor(
                    crate::ops::tensorinv(dense).ind(ind).run()?,
                ))
            });
        }
        if let Some(value) = self.as_c64() {
            return with_dense_primal_typed(value, "tensorinv", |dense| {
                Ok(Self::from_tensor(
                    crate::ops::tensorinv(dense).ind(ind).run()?,
                ))
            });
        }
        unreachable!("Tensor::tensorinv should have one of the supported scalar dtypes")
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
        if let (Some(lhs), Some(rhs)) = (self.as_f32(), rhs.as_f32()) {
            return with_dense_primal_pair_typed(lhs, rhs, "tensorsolve", |a, b| {
                let mut builder = crate::ops::tensorsolve(a, b);
                if !dims.is_empty() {
                    builder = builder.dims(dims);
                }
                Ok(Self::from_tensor(builder.run()?))
            });
        }
        if let (Some(lhs), Some(rhs)) = (self.as_f64(), rhs.as_f64()) {
            return with_dense_primal_pair_typed(lhs, rhs, "tensorsolve", |a, b| {
                let mut builder = crate::ops::tensorsolve(a, b);
                if !dims.is_empty() {
                    builder = builder.dims(dims);
                }
                Ok(Self::from_tensor(builder.run()?))
            });
        }
        if let (Some(lhs), Some(rhs)) = (self.as_c32(), rhs.as_c32()) {
            return with_dense_primal_pair_typed(lhs, rhs, "tensorsolve", |a, b| {
                let mut builder = crate::ops::tensorsolve(a, b);
                if !dims.is_empty() {
                    builder = builder.dims(dims);
                }
                Ok(Self::from_tensor(builder.run()?))
            });
        }
        if let (Some(lhs), Some(rhs)) = (self.as_c64(), rhs.as_c64()) {
            return with_dense_primal_pair_typed(lhs, rhs, "tensorsolve", |a, b| {
                let mut builder = crate::ops::tensorsolve(a, b);
                if !dims.is_empty() {
                    builder = builder.dims(dims);
                }
                Ok(Self::from_tensor(builder.run()?))
            });
        }
        Err(same_dtype_error(
            "tensorsolve",
            self.scalar_type(),
            rhs.scalar_type(),
        ))
    }
}
