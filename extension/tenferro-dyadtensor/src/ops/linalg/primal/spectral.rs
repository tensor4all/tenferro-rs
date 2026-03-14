use super::common::{
    dispatch_cpu_composite_runtime, dispatch_linalg_runtime, unary_linalg_builder,
};
use super::*;
use num_complex::Complex;

use crate::DynTensorTyped;

unary_linalg_builder!(
    DetBuilder,
    det,
    returns = Tensor<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::Det,
    op = "det",
    bounds = (T: RealLinalgRuntimeValue),
    call = |ctx, builder| tenferro_linalg::det::<T, _>(ctx, builder.tensor).map_err(Error::from)
);

unary_linalg_builder!(
    SlogdetBuilder,
    slogdet,
    returns = SlogdetResult<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::Slogdet,
    op = "slogdet",
    bounds = (T: RealLinalgRuntimeValue),
    call = |ctx, builder| tenferro_linalg::slogdet::<T, _>(ctx, builder.tensor).map_err(Error::from)
);

unary_linalg_builder!(
    EigBuilder,
    eig,
    returns = EigResult<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::Eig,
    op = "eig",
    bounds = (T: ComplexLinalgRuntimeValue, Complex<T>: DynTensorTyped),
    call = |ctx, builder| tenferro_linalg::eig::<T, _>(ctx, builder.tensor).map_err(Error::from)
);

/// Builder for `pinv`.
/// # Examples
///
/// ```ignore
/// let _builder = pinv(/* ... */);
/// ```
pub struct PinvBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    rcond: Option<f64>,
}

impl<'a, T> PinvBuilder<'a, T>
where
    T: RealLinalgRuntimeValue,
{
    /// Sets rcond.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.rcond(1e-12);
    /// ```
    pub fn rcond(mut self, rcond: f64) -> Self {
        self.rcond = Some(rcond);
        self
    }

    /// Executes `pinv`.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        dispatch_linalg_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Pinv,
            "pinv",
            |ctx| {
                tenferro_linalg::pinv::<T, _>(ctx, self.tensor, self.rcond).map_err(Error::from)
            }
        )
    }
}

/// Creates a `pinv` builder.
/// # Examples
///
/// ```ignore
/// let _ = pinv(/* ... */);
/// ```
pub fn pinv<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> PinvBuilder<'a, T> {
    PinvBuilder {
        tensor,
        rcond: None,
    }
}

unary_linalg_builder!(
    MatrixExpBuilder,
    matrix_exp,
    returns = Tensor<T>,
    capability = tenferro_linalg::backend::LinalgCapabilityOp::MatrixExp,
    op = "matrix_exp",
    bounds = (T: RealLinalgRuntimeValue),
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

/// Builder for `norm`.
/// # Examples
///
/// ```ignore
/// let _builder = norm(/* ... */);
/// ```
pub struct NormBuilder<'a, T: LinalgScalar> {
    tensor: &'a Tensor<T>,
    kind: NormKind,
}

impl<'a, T> NormBuilder<'a, T>
where
    T: RealLinalgRuntimeValue,
{
    /// Sets norm kind.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.kind(kind);
    /// ```
    pub fn kind(mut self, kind: NormKind) -> Self {
        self.kind = kind;
        self
    }

    /// Executes `norm`.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<Tensor<T>> {
        dispatch_linalg_runtime!(
            T,
            tenferro_linalg::backend::LinalgCapabilityOp::Norm,
            "norm",
            |ctx| {
                tenferro_linalg::norm::<T, _>(ctx, self.tensor, self.kind).map_err(Error::from)
            }
        )
    }
}

/// Creates a `norm` builder.
/// # Examples
///
/// ```ignore
/// let _ = norm(/* ... */);
/// ```
pub fn norm<'a, T: LinalgScalar>(tensor: &'a Tensor<T>) -> NormBuilder<'a, T> {
    NormBuilder {
        tensor,
        kind: NormKind::Fro,
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
