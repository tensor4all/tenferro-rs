use super::common::{binary_linalg_builder, dispatch_linalg_runtime, unary_linalg_builder};
use super::*;

/// Builder for SVD.
/// # Examples
///
/// ```ignore
/// let _builder = svd(/* ... */);
/// ```
pub struct SvdBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    options: Option<&'a SvdOptions>,
}

impl<'a, T> SvdBuilder<'a, T>
where
    T: LinalgRuntimeValue,
{
    /// Sets optional SVD options.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.options(&options);
    /// ```
    pub fn options(mut self, options: &'a SvdOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Executes SVD.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<SvdResult<T, T::Real>> {
        dispatch_linalg_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::ThinSvd,
            "svd",
            |ctx| {
                tenferro_linalg::svd::<T, _>(ctx, self.tensor, self.options).map_err(Error::from)
            }
        )
    }
}

/// Creates an SVD builder.
/// # Examples
///
/// ```ignore
/// let _ = svd(/* ... */);
/// ```
pub fn svd<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> SvdBuilder<'a, T> {
    SvdBuilder {
        tensor,
        options: None,
    }
}

unary_linalg_builder!(
    QrBuilder,
    qr,
    returns = QrResult<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::Qr,
    op = "qr",
    bounds = (T: LinalgRuntimeValue),
    call = |ctx, builder| tenferro_linalg::qr::<T, _>(ctx, builder.tensor).map_err(Error::from)
);

/// Builder for `lu`.
/// # Examples
///
/// ```ignore
/// let _builder = lu(/* ... */);
/// ```
pub struct LuBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    pivot: LuPivot,
}

impl<'a, T> LuBuilder<'a, T>
where
    T: LinalgRuntimeValue,
{
    /// Sets LU pivoting policy.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.pivot(pivot);
    /// ```
    pub fn pivot(mut self, pivot: LuPivot) -> Self {
        self.pivot = pivot;
        self
    }

    /// Executes `lu`.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<LuResult<T>> {
        dispatch_linalg_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
            "lu",
            |ctx| {
                tenferro_linalg::lu::<T, _>(ctx, self.tensor, self.pivot).map_err(Error::from)
            }
        )
    }
}

/// Creates a `lu` builder.
/// # Examples
///
/// ```ignore
/// let _ = lu(/* ... */);
/// ```
pub fn lu<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> LuBuilder<'a, T> {
    LuBuilder {
        tensor,
        pivot: LuPivot::Partial,
    }
}

unary_linalg_builder!(
    LuFactorBuilder,
    lu_factor,
    returns = LuFactorResult<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
    op = "lu_factor",
    bounds = (T: LinalgRuntimeValue),
    call = |ctx, builder| tenferro_linalg::lu_factor::<T, _>(ctx, builder.tensor).map_err(Error::from)
);

unary_linalg_builder!(
    LuFactorExBuilder,
    lu_factor_ex,
    returns = LuFactorExResult<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
    op = "lu_factor_ex",
    bounds = (T: LinalgRuntimeValue),
    call = |ctx, builder| tenferro_linalg::lu_factor_ex::<T, _>(ctx, builder.tensor).map_err(Error::from)
);

unary_linalg_builder!(
    EigenBuilder,
    eigen,
    returns = EigenResult<T, T::Real>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::EigenSym,
    op = "eigen",
    bounds = (T: LinalgRuntimeValue),
    call = |ctx, builder| tenferro_linalg::eigen::<T, _>(ctx, builder.tensor).map_err(Error::from)
);

binary_linalg_builder!(
    LstsqBuilder,
    lstsq,
    returns = LstsqResult<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::Lstsq,
    op = "lstsq",
    bounds = (T: LinalgRuntimeValue),
    call = |ctx, builder| tenferro_linalg::lstsq::<T, _>(ctx, builder.a, builder.b).map_err(Error::from)
);

unary_linalg_builder!(
    CholeskyBuilder,
    cholesky,
    returns = Tensor<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::Cholesky,
    op = "cholesky",
    bounds = (T: LinalgRuntimeValue),
    call = |ctx, builder| tenferro_linalg::cholesky::<T, _>(ctx, builder.tensor).map_err(Error::from)
);

unary_linalg_builder!(
    CholeskyExBuilder,
    cholesky_ex,
    returns = CholeskyExResult<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::CholeskyEx,
    op = "cholesky_ex",
    bounds = (T: LinalgRuntimeValue),
    call = |ctx, builder| tenferro_linalg::cholesky_ex::<T, _>(ctx, builder.tensor).map_err(Error::from)
);
