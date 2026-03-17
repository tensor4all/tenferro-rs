use super::common::{
    dispatch_cpu_composite_runtime, dispatch_linalg_runtime, unary_linalg_builder,
};
use super::*;

unary_linalg_builder!(
    MatrixExpBuilder,
    matrix_exp,
    returns = Tensor<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::MatrixExp,
    op = "matrix_exp",
    bounds = (T: LinalgRuntimeValue),
    call = |ctx, builder| tenferro_linalg::matrix_exp::<T, _>(ctx, builder.tensor).map_err(Error::from)
);

/// Builder for `matrix_power`.
/// # Examples
///
/// ```ignore
/// let _builder = matrix_power(/* ... */);
/// ```
pub struct MatrixPowerBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    exponent: i64,
}

impl<'a, T> MatrixPowerBuilder<'a, T>
where
    T: LinalgRuntimeValue,
{
    /// Sets the integer exponent.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.exponent(3);
    /// ```
    pub fn exponent(mut self, exponent: i64) -> Self {
        self.exponent = exponent;
        self
    }

    /// Executes `matrix_power`.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        dispatch_linalg_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::MatrixPower,
            "matrix_power",
            |ctx| {
                tenferro_linalg::matrix_power::<T, _>(ctx, self.tensor, self.exponent)
                    .map_err(Error::from)
            }
        )
    }
}

/// Creates a `matrix_power` builder.
/// # Examples
///
/// ```ignore
/// let _ = matrix_power(/* ... */);
/// ```
pub fn matrix_power<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> MatrixPowerBuilder<'a, T> {
    MatrixPowerBuilder {
        tensor,
        exponent: 1,
    }
}

/// Builder for `cond`.
/// # Examples
///
/// ```ignore
/// let _builder = cond(/* ... */);
/// ```
pub struct CondBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    kind: NormKind,
}

impl<'a, T> CondBuilder<'a, T>
where
    T: RealLinalgRuntimeValue,
{
    /// Sets the norm kind used in the condition number.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.kind(NormKind::Spectral);
    /// ```
    #[allow(dead_code)]
    pub fn kind(mut self, kind: NormKind) -> Self {
        self.kind = kind;
        self
    }

    /// Executes `cond`.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        dispatch_cpu_composite_runtime!("cond", |ctx| {
            tenferro_linalg::cond::<T, _>(ctx, self.tensor, self.kind).map_err(Error::from)
        })
    }
}

/// Creates a `cond` builder.
/// # Examples
///
/// ```ignore
/// let _ = cond(/* ... */);
/// ```
pub fn cond<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> CondBuilder<'a, T> {
    CondBuilder {
        tensor,
        kind: NormKind::Spectral,
    }
}
