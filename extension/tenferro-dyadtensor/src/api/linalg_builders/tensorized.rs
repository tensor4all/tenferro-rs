use super::common::{binary_linalg_builder, dispatch_linalg_runtime};
use super::*;

binary_linalg_builder!(
    CrossBuilder,
    cross,
    returns = Tensor<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::Cross,
    op = "cross",
    bounds = (T: LinalgScalar + CpuLinalgScalar),
    call = |ctx, builder| tenferro_linalg::cross::<T, _>(ctx, builder.a, builder.b).map_err(Error::from)
);

binary_linalg_builder!(
    HouseholderProductBuilder,
    householder_product,
    returns = Tensor<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::HouseholderProduct,
    op = "householder_product",
    bounds = (T: LinalgScalar + CpuLinalgScalar),
    call = |ctx, builder| tenferro_linalg::householder_product::<T, _>(ctx, builder.a, builder.b).map_err(Error::from)
);

/// Builder for `vander`.
/// # Examples
///
/// ```ignore
/// let _builder = vander(/* ... */);
/// ```
pub struct VanderBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    columns: Option<usize>,
    increasing: bool,
}

impl<'a, T> VanderBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets the output column count.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.columns(4);
    /// ```
    pub fn columns(mut self, columns: usize) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Sets whether powers increase from left to right.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.increasing(true);
    /// ```
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Executes `vander`.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        dispatch_linalg_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Vander,
            "vander",
            |ctx| {
                tenferro_linalg::vander::<T, _>(ctx, self.tensor, self.columns, self.increasing)
                    .map_err(Error::from)
            }
        )
    }
}

/// Creates a `vander` builder.
/// # Examples
///
/// ```ignore
/// let _ = vander(/* ... */);
/// ```
pub fn vander<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> VanderBuilder<'a, T> {
    VanderBuilder {
        tensor,
        columns: None,
        increasing: false,
    }
}

/// Builder for `tensorinv`.
/// # Examples
///
/// ```ignore
/// let _builder = tensorinv(/* ... */);
/// ```
pub struct TensorinvBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    ind: usize,
}

impl<'a, T> TensorinvBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets the partition point between left and right tensor dimensions.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.ind(2);
    /// ```
    pub fn ind(mut self, ind: usize) -> Self {
        self.ind = ind;
        self
    }

    /// Executes `tensorinv`.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        dispatch_linalg_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::TensorInv,
            "tensorinv",
            |ctx| {
                tenferro_linalg::tensorinv::<T, _>(ctx, self.tensor, self.ind).map_err(Error::from)
            }
        )
    }
}

/// Creates a `tensorinv` builder.
/// # Examples
///
/// ```ignore
/// let _ = tensorinv(/* ... */);
/// ```
pub fn tensorinv<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> TensorinvBuilder<'a, T> {
    TensorinvBuilder { tensor, ind: 1 }
}

/// Builder for `tensorsolve`.
/// # Examples
///
/// ```ignore
/// let _builder = tensorsolve(/* ... */);
/// ```
pub struct TensorsolveBuilder<'a, T: LinalgScalar> {
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
    dims: Option<&'a [usize]>,
}

impl<'a, T> TensorsolveBuilder<'a, T>
where
    T: LinalgScalar + CpuLinalgScalar,
{
    /// Sets the solution axes to move before solving.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.dims(&[3, 2]);
    /// ```
    pub fn dims(mut self, dims: &'a [usize]) -> Self {
        self.dims = Some(dims);
        self
    }

    /// Executes `tensorsolve`.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        dispatch_linalg_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::TensorSolve,
            "tensorsolve",
            |ctx| {
                tenferro_linalg::tensorsolve::<T, _>(ctx, self.a, self.b, self.dims)
                    .map_err(Error::from)
            }
        )
    }
}

/// Creates a `tensorsolve` builder.
/// # Examples
///
/// ```ignore
/// let _ = tensorsolve(/* ... */);
/// ```
pub fn tensorsolve<'a, T: LinalgScalar>(
    a: &'a Tensor<T>,
    b: &'a Tensor<T>,
) -> TensorsolveBuilder<'a, T> {
    TensorsolveBuilder { a, b, dims: None }
}
