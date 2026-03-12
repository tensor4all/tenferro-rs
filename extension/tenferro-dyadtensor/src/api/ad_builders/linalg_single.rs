use super::super::*;
use super::common::*;

/// Builder for AD Cholesky.
/// # Examples
///
/// ```text
/// // Construct `CholeskyAdBuilder` via its corresponding operation constructor.
/// ```
pub struct CholeskyAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> CholeskyAdBuilder<'a, T>
where
    T: RealLinalgRuntimeValue,
{
    /// Executes AD Cholesky.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Cholesky,
            op = "cholesky_ad",
            pullback = "cholesky_ad_pullback",
            input = self.tensor,
            primal = |ctx, tensor| {
                tenferro_linalg::cholesky::<T, _>(ctx, tensor).map_err(Error::from)
            },
            frule = |ctx, tensor, dt| {
                tenferro_linalg::cholesky_frule::<T, _>(ctx, tensor, dt).map_err(Error::from)
            },
            rrule = |ctx, tensor, cotangent| {
                tenferro_linalg::cholesky_rrule::<T, _>(ctx, tensor, cotangent).map_err(Error::from)
            },
        )
    }
}

/// Creates an AD cholesky builder.
/// # Examples
///
/// ```ignore
/// let _ = cholesky_ad(/* ... */);
/// ```
pub fn cholesky_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> CholeskyAdBuilder<'a, T> {
    CholeskyAdBuilder { tensor }
}

/// Builder for AD solve.
/// # Examples
///
/// ```text
/// // Construct `SolveAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveAdBuilder<'a, T: Scalar> {
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
}

impl<'a, T> SolveAdBuilder<'a, T>
where
    T: RealLinalgRuntimeValue,
{
    /// Executes AD solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_binary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Solve,
            op = "solve_ad",
            pullback = "solve_ad_pullback",
            lhs = self.a,
            rhs = self.b,
            primal = |ctx, a, b| tenferro_linalg::solve::<T, _>(ctx, a, b).map_err(Error::from),
            frule = |ctx, a, b, da, db| {
                tenferro_linalg::solve_frule::<T, _>(ctx, a, b, da, db).map_err(Error::from)
            },
            rrule = |ctx, a, b, cotangent| {
                let grad = tenferro_linalg::solve_rrule::<T, _>(ctx, a, b, cotangent)
                    .map_err(Error::from)?;
                Ok((grad.a, grad.b))
            },
        )
    }
}

/// Creates an AD solve builder.
/// # Examples
///
/// ```ignore
/// let _ = solve_ad(/* ... */);
/// ```
pub fn solve_ad<'a, T: Scalar>(a: &'a AdTensor<T>, b: &'a AdTensor<T>) -> SolveAdBuilder<'a, T> {
    SolveAdBuilder { a, b }
}

/// Builder for AD inverse.
/// # Examples
///
/// ```text
/// // Construct `InvAdBuilder` via its corresponding operation constructor.
/// ```
pub struct InvAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> InvAdBuilder<'a, T>
where
    T: RealLinalgRuntimeValue,
{
    /// Executes AD inverse.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Inv,
            op = "inv_ad",
            pullback = "inv_ad_pullback",
            input = self.tensor,
            primal = |ctx, tensor| tenferro_linalg::inv::<T, _>(ctx, tensor).map_err(Error::from),
            frule = |ctx, tensor, dt| {
                tenferro_linalg::inv_frule::<T, _>(ctx, tensor, dt).map_err(Error::from)
            },
            rrule = |ctx, tensor, cotangent| {
                tenferro_linalg::inv_rrule::<T, _>(ctx, tensor, cotangent).map_err(Error::from)
            },
        )
    }
}

/// Creates an AD inv builder.
/// # Examples
///
/// ```ignore
/// let _ = inv_ad(/* ... */);
/// ```
pub fn inv_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> InvAdBuilder<'a, T> {
    InvAdBuilder { tensor }
}

/// Builder for AD det.
/// # Examples
///
/// ```text
/// // Construct `DetAdBuilder` via its corresponding operation constructor.
/// ```
pub struct DetAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> DetAdBuilder<'a, T>
where
    T: RealLinalgRuntimeValue,
{
    /// Executes AD determinant.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Det,
            op = "det_ad",
            pullback = "det_ad_pullback",
            input = self.tensor,
            primal = |ctx, tensor| tenferro_linalg::det::<T, _>(ctx, tensor).map_err(Error::from),
            frule = |ctx, tensor, dt| {
                tenferro_linalg::det_frule::<T, _>(ctx, tensor, dt).map_err(Error::from)
            },
            rrule = |ctx, tensor, cotangent| {
                tenferro_linalg::det_rrule::<T, _>(ctx, tensor, cotangent).map_err(Error::from)
            },
        )
    }
}

/// Creates an AD det builder.
/// # Examples
///
/// ```ignore
/// let _ = det_ad(/* ... */);
/// ```
pub fn det_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> DetAdBuilder<'a, T> {
    DetAdBuilder { tensor }
}

/// Builder for AD pinv.
/// # Examples
///
/// ```text
/// // Construct `PinvAdBuilder` via its corresponding operation constructor.
/// ```
pub struct PinvAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    rcond: Option<f64>,
}

impl<'a, T> PinvAdBuilder<'a, T>
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

    /// Executes AD pinv.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Pinv,
            op = "pinv_ad",
            pullback = "pinv_ad_pullback",
            input = self.tensor,
            primal = |ctx, tensor| {
                tenferro_linalg::pinv::<T, _>(ctx, tensor, self.rcond).map_err(Error::from)
            },
            frule = |ctx, tensor, dt| {
                tenferro_linalg::pinv_frule::<T, _>(ctx, tensor, dt, self.rcond)
                    .map_err(Error::from)
            },
            rrule = |ctx, tensor, cotangent| {
                tenferro_linalg::pinv_rrule::<T, _>(ctx, tensor, cotangent, self.rcond)
                    .map_err(Error::from)
            },
        )
    }
}

/// Creates an AD pinv builder.
/// # Examples
///
/// ```ignore
/// let _ = pinv_ad(/* ... */);
/// ```
pub fn pinv_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> PinvAdBuilder<'a, T> {
    PinvAdBuilder {
        tensor,
        rcond: None,
    }
}

/// Builder for AD matrix exponential.
/// # Examples
///
/// ```text
/// // Construct `MatrixExpAdBuilder` via its corresponding operation constructor.
/// ```
pub struct MatrixExpAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> MatrixExpAdBuilder<'a, T>
where
    T: RealLinalgRuntimeValue,
{
    /// Executes AD matrix exponential.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::MatrixExp,
            op = "matrix_exp_ad",
            pullback = "matrix_exp_ad_pullback",
            input = self.tensor,
            primal = |ctx, tensor| {
                tenferro_linalg::matrix_exp::<T, _>(ctx, tensor).map_err(Error::from)
            },
            frule = |ctx, tensor, dt| {
                tenferro_linalg::matrix_exp_frule::<T, _>(ctx, tensor, dt).map_err(Error::from)
            },
            rrule = |ctx, tensor, cotangent| {
                tenferro_linalg::matrix_exp_rrule::<T, _>(ctx, tensor, cotangent)
                    .map_err(Error::from)
            },
        )
    }
}

/// Creates an AD matrix_exp builder.
/// # Examples
///
/// ```ignore
/// let _ = matrix_exp_ad(/* ... */);
/// ```
pub fn matrix_exp_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> MatrixExpAdBuilder<'a, T> {
    MatrixExpAdBuilder { tensor }
}

/// Builder for AD solve_triangular.
/// # Examples
///
/// ```text
/// // Construct `SolveTriangularAdBuilder` via its corresponding operation constructor.
/// ```
pub struct SolveTriangularAdBuilder<'a, T: Scalar> {
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
    upper: bool,
}

impl<'a, T> SolveTriangularAdBuilder<'a, T>
where
    T: LinalgRuntimeValue,
{
    /// Sets whether the matrix is upper triangular.
    /// # Examples
    ///
    /// ```ignore
    /// let _builder = builder.upper(true);
    /// ```
    pub fn upper(mut self, upper: bool) -> Self {
        self.upper = upper;
        self
    }

    /// Executes AD triangular solve.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>>
    where
        T: 'static,
    {
        run_binary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::SolveTriangular,
            op = "solve_triangular_ad",
            pullback = "solve_triangular_ad_pullback",
            lhs = self.a,
            rhs = self.b,
            primal = |ctx, a, b| {
                tenferro_linalg::solve_triangular::<T, _>(ctx, a, b, self.upper)
                    .map_err(Error::from)
            },
            frule = |ctx, a, b, da, db| {
                tenferro_linalg::solve_triangular_frule::<T, _>(ctx, a, b, da, db, self.upper)
                    .map_err(Error::from)
            },
            rrule = |ctx, a, b, cotangent| {
                let grad = tenferro_linalg::solve_triangular_rrule::<T, _>(
                    ctx, a, b, cotangent, self.upper,
                )
                .map_err(Error::from)?;
                Ok((grad.a, grad.b))
            },
        )
    }
}

/// Creates an AD solve_triangular builder.
/// # Examples
///
/// ```ignore
/// let _ = solve_triangular_ad(/* ... */);
/// ```
pub fn solve_triangular_ad<'a, T: Scalar>(
    a: &'a AdTensor<T>,
    b: &'a AdTensor<T>,
) -> SolveTriangularAdBuilder<'a, T> {
    SolveTriangularAdBuilder { a, b, upper: true }
}

/// Builder for AD norm.
/// # Examples
///
/// ```text
/// // Construct `NormAdBuilder` via its corresponding operation constructor.
/// ```
pub struct NormAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
    kind: NormKind,
}

impl<'a, T> NormAdBuilder<'a, T>
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

    /// Executes AD norm.
    /// # Examples
    ///
    /// ```ignore
    /// let _out = builder.run();
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Norm,
            op = "norm_ad",
            pullback = "norm_ad_pullback",
            input = self.tensor,
            primal = |ctx, tensor| {
                tenferro_linalg::norm::<T, _>(ctx, tensor, self.kind).map_err(Error::from)
            },
            frule = |ctx, tensor, dt| {
                tenferro_linalg::norm_frule::<T, _>(ctx, tensor, dt, self.kind).map_err(Error::from)
            },
            rrule = |ctx, tensor, cotangent| {
                tenferro_linalg::norm_rrule::<T, _>(ctx, tensor, cotangent, self.kind)
                    .map_err(Error::from)
            },
        )
    }
}

/// Creates an AD norm builder.
/// # Examples
///
/// ```ignore
/// let _ = norm_ad(/* ... */);
/// ```
pub fn norm_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> NormAdBuilder<'a, T> {
    NormAdBuilder {
        tensor,
        kind: NormKind::Fro,
    }
}
