use super::super::*;
use num_complex::Complex;
use tenferro_internal_ad_core::{AdMode, DynAdTensor, DynAdTensorRef};
use tenferro_internal_frontend_core::{DynTensorTyped, ScalarType};

macro_rules! eager_into_dyn_result {
    ($expr:expr, $ty:ident { $($field:ident),+ $(,)? }) => {{
        let out = $expr?;
        Ok($ty {
            $($field: out.$field.into(),)+
        })
    }};
}

macro_rules! dyn_unary_dispatch_all {
    ($tensor:expr, |$value:ident| $body:expr) => {{
        match $tensor {
            DynAdTensorRef::F32($value) => $body,
            DynAdTensorRef::F64($value) => $body,
            DynAdTensorRef::C32($value) => $body,
            DynAdTensorRef::C64($value) => $body,
        }
    }};
}

macro_rules! dyn_unary_dispatch_real_only {
    ($tensor:expr, $op:expr, |$value:ident| $body:expr) => {{
        match $tensor {
            DynAdTensorRef::F32($value) => $body,
            DynAdTensorRef::F64($value) => $body,
            DynAdTensorRef::C32(_) | DynAdTensorRef::C64(_) => Err(Error::InvalidAdTensor {
                message: format!("{} currently requires a real DynAdTensor input", $op),
            }),
        }
    }};
}

macro_rules! dyn_unary_dispatch_split {
    ($tensor:expr, |$real:ident| $real_body:expr, |$complex:ident| $complex_body:expr) => {{
        match $tensor {
            DynAdTensorRef::F32($real) => $real_body,
            DynAdTensorRef::F64($real) => $real_body,
            DynAdTensorRef::C32($complex) => $complex_body,
            DynAdTensorRef::C64($complex) => $complex_body,
        }
    }};
}

macro_rules! dyn_binary_dispatch_same_dtype {
    ($lhs:expr, $rhs:expr, $op:expr, |$lhs_var:ident, $rhs_var:ident| $body:expr) => {{
        match ($lhs, $rhs) {
            (DynAdTensorRef::F32($lhs_var), DynAdTensorRef::F32($rhs_var)) => $body,
            (DynAdTensorRef::F64($lhs_var), DynAdTensorRef::F64($rhs_var)) => $body,
            (DynAdTensorRef::C32($lhs_var), DynAdTensorRef::C32($rhs_var)) => $body,
            (DynAdTensorRef::C64($lhs_var), DynAdTensorRef::C64($rhs_var)) => $body,
            (lhs, rhs) => Err(same_dtype_error($op, lhs.scalar_type(), rhs.scalar_type())),
        }
    }};
}

macro_rules! eager_unary {
    ($(#[$meta:meta])* fn $name:ident -> $ret:ty => $builder:ident ; where { $($bounds:tt)* }) => {
        $(#[$meta])*
        pub fn $name<T: Scalar>(tensor: &AdTensor<T>) -> Result<$ret>
        where
            $($bounds)*
        {
            $builder(tensor).run()
        }
    };
}

macro_rules! eager_binary {
    ($(#[$meta:meta])* fn $name:ident -> $ret:ty => $builder:ident ; where { $($bounds:tt)* }) => {
        $(#[$meta])*
        pub fn $name<T: Scalar>(a: &AdTensor<T>, b: &AdTensor<T>) -> Result<$ret>
        where
            $($bounds)*
        {
            $builder(a, b).run()
        }
    };
}

fn primal_only_error(op: &'static str) -> Error {
    Error::InvalidAdTensor {
        message: format!("{op} currently supports only primal tensors"),
    }
}

fn same_dtype_error(op: &'static str, lhs: ScalarType, rhs: ScalarType) -> Error {
    Error::InvalidAdTensor {
        message: format!("{op} requires matching DynAdTensor inputs, got lhs={lhs:?}, rhs={rhs:?}"),
    }
}

fn dense_primal_only<T>(value: &AdTensor<T>, op: &'static str) -> Result<tenferro_tensor::Tensor<T>>
where
    T: Scalar + DynTensorTyped + crate::runtime::contracts::LinalgRuntimeValue,
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

fn dense_primal_pair<T>(
    lhs: &AdTensor<T>,
    rhs: &AdTensor<T>,
    op: &'static str,
) -> Result<(tenferro_tensor::Tensor<T>, tenferro_tensor::Tensor<T>)>
where
    T: Scalar + DynTensorTyped + crate::runtime::contracts::LinalgRuntimeValue,
{
    Ok((dense_primal_only(lhs, op)?, dense_primal_only(rhs, op)?))
}

fn to_dyn_primal<T>(tensor: tenferro_tensor::Tensor<T>) -> Result<DynAdTensor>
where
    T: Scalar + DynTensorTyped + tenferro_internal_ad_core::DynAdTensorTyped,
{
    Ok(
        AdTensor::try_from(tenferro_internal_ad_core::AdTensorSnapshot::Primal(
            tensor.into(),
        ))?
        .into(),
    )
}

fn lu_factor_dyn_impl<T>(value: &AdTensor<T>, op: &'static str) -> Result<DynLuFactorResult>
where
    T: Scalar
        + DynTensorTyped
        + tenferro_internal_ad_core::DynAdTensorTyped
        + crate::runtime::contracts::LinalgRuntimeValue,
{
    let dense = dense_primal_only(value, op)?;
    let out = crate::runtime::dispatch::with_linalg_runtime::<T, _>(
        op,
        tenferro_linalg::backend::LinalgCapabilityOp::LuFactor,
        |ctx| tenferro_linalg::lu_factor::<T, _>(ctx, &dense).map_err(Error::from),
        |ctx| tenferro_linalg::lu_factor::<T, _>(ctx, &dense).map_err(Error::from),
        |ctx| tenferro_linalg::lu_factor::<T, _>(ctx, &dense).map_err(Error::from),
    )?;
    Ok(DynLuFactorResult {
        factors: to_dyn_primal(out.factors)?,
        pivots: out.pivots,
    })
}

fn lu_factor_ex_dyn_impl<T>(value: &AdTensor<T>, op: &'static str) -> Result<DynLuFactorExResult>
where
    T: Scalar
        + DynTensorTyped
        + tenferro_internal_ad_core::DynAdTensorTyped
        + crate::runtime::contracts::LinalgRuntimeValue,
{
    let dense = dense_primal_only(value, op)?;
    let out = crate::runtime::dispatch::with_linalg_runtime::<T, _>(
        op,
        tenferro_linalg::backend::LinalgCapabilityOp::LuFactorEx,
        |ctx| tenferro_linalg::lu_factor_ex::<T, _>(ctx, &dense).map_err(Error::from),
        |ctx| tenferro_linalg::lu_factor_ex::<T, _>(ctx, &dense).map_err(Error::from),
        |ctx| tenferro_linalg::lu_factor_ex::<T, _>(ctx, &dense).map_err(Error::from),
    )?;
    Ok(DynLuFactorExResult {
        factors: to_dyn_primal(out.factors)?,
        pivots: out.pivots,
        info: out.info,
    })
}

fn lu_solve_dyn_impl<T>(
    lhs: &AdTensor<T>,
    rhs: &AdTensor<T>,
    pivots: &tenferro_tensor::Tensor<i32>,
    op: &'static str,
) -> Result<DynAdTensor>
where
    T: Scalar
        + DynTensorTyped
        + tenferro_internal_ad_core::DynAdTensorTyped
        + crate::runtime::contracts::LinalgRuntimeValue,
{
    let (lhs, rhs) = dense_primal_pair(lhs, rhs, op)?;
    let out = crate::runtime::dispatch::with_linalg_runtime::<T, _>(
        op,
        tenferro_linalg::backend::LinalgCapabilityOp::LuSolve,
        |ctx| tenferro_linalg::lu_solve::<T, _>(ctx, &lhs, pivots, &rhs).map_err(Error::from),
        |ctx| tenferro_linalg::lu_solve::<T, _>(ctx, &lhs, pivots, &rhs).map_err(Error::from),
        |ctx| tenferro_linalg::lu_solve::<T, _>(ctx, &lhs, pivots, &rhs).map_err(Error::from),
    )?;
    to_dyn_primal(out)
}

fn solve_ex_dyn_impl<T>(
    lhs: &AdTensor<T>,
    rhs: &AdTensor<T>,
    op: &'static str,
) -> Result<DynSolveExResult>
where
    T: Scalar
        + DynTensorTyped
        + tenferro_internal_ad_core::DynAdTensorTyped
        + crate::runtime::contracts::LinalgRuntimeValue,
{
    let (lhs, rhs) = dense_primal_pair(lhs, rhs, op)?;
    let out = crate::runtime::dispatch::with_linalg_runtime::<T, _>(
        op,
        tenferro_linalg::backend::LinalgCapabilityOp::SolveEx,
        |ctx| tenferro_linalg::solve_ex::<T, _>(ctx, &lhs, &rhs).map_err(Error::from),
        |ctx| tenferro_linalg::solve_ex::<T, _>(ctx, &lhs, &rhs).map_err(Error::from),
        |ctx| tenferro_linalg::solve_ex::<T, _>(ctx, &lhs, &rhs).map_err(Error::from),
    )?;
    Ok(DynSolveExResult {
        solution: to_dyn_primal(out.solution)?,
        info: out.info,
    })
}

fn inv_ex_dyn_impl<T>(value: &AdTensor<T>, op: &'static str) -> Result<DynInvExResult>
where
    T: Scalar
        + DynTensorTyped
        + tenferro_internal_ad_core::DynAdTensorTyped
        + crate::runtime::contracts::LinalgRuntimeValue,
{
    let dense = dense_primal_only(value, op)?;
    let out = crate::runtime::dispatch::with_linalg_runtime::<T, _>(
        op,
        tenferro_linalg::backend::LinalgCapabilityOp::Inv,
        |ctx| tenferro_linalg::inv_ex::<T, _>(ctx, &dense).map_err(Error::from),
        |ctx| tenferro_linalg::inv_ex::<T, _>(ctx, &dense).map_err(Error::from),
        |ctx| tenferro_linalg::inv_ex::<T, _>(ctx, &dense).map_err(Error::from),
    )?;
    Ok(DynInvExResult {
        inverse: to_dyn_primal(out.inverse)?,
        info: out.info,
    })
}

fn cholesky_ex_dyn_impl<T>(value: &AdTensor<T>, op: &'static str) -> Result<DynCholeskyExResult>
where
    T: Scalar
        + DynTensorTyped
        + tenferro_internal_ad_core::DynAdTensorTyped
        + crate::runtime::contracts::LinalgRuntimeValue,
{
    let dense = dense_primal_only(value, op)?;
    let out = crate::runtime::dispatch::with_linalg_runtime::<T, _>(
        op,
        tenferro_linalg::backend::LinalgCapabilityOp::CholeskyEx,
        |ctx| tenferro_linalg::cholesky_ex::<T, _>(ctx, &dense).map_err(Error::from),
        |ctx| tenferro_linalg::cholesky_ex::<T, _>(ctx, &dense).map_err(Error::from),
        |ctx| tenferro_linalg::cholesky_ex::<T, _>(ctx, &dense).map_err(Error::from),
    )?;
    Ok(DynCholeskyExResult {
        l: to_dyn_primal(out.l)?,
        info: out.info,
    })
}

/// Eager AD SVD.
///
/// Equivalent to `crate::svd_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro::ad::svd(&a)?;
/// ```
pub fn svd<T>(tensor: &AdTensor<T>) -> Result<TypedSvdResult>
where
    T: LinalgRuntimeValue + DynAdTensorTyped,
    T::Real: DynTensorTyped + DynAdTensorTyped + tenferro_tensor::KeepCountScalar,
{
    svd_ad(tensor).run()
}

/// Real-valued eager AD SVD with edge-based reverse fast path.
pub fn svd_real<T>(tensor: &AdTensor<T>) -> Result<TypedSvdResult>
where
    T: RealLinalgRuntimeValue + DynAdTensorTyped,
{
    if can_use_edge_svd_real_reverse(tensor) {
        return edge_svd_real(tensor, None);
    }
    svd_ad(tensor).run()
}

/// Eager AD QR.
///
/// Equivalent to `crate::qr_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro::ad::qr(&a)?;
/// ```
pub fn qr<T>(tensor: &AdTensor<T>) -> Result<DynQrResult>
where
    T: RealLinalgRuntimeValue + DynAdTensorTyped,
{
    if can_use_edge_qr_real_reverse(tensor) {
        return edge_qr(tensor);
    }
    qr_ad(tensor).run()
}

eager_unary!(
    /// Eager AD LU (partial pivot by default).
    ///
    /// Equivalent to `crate::lu_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro::ad::lu(&a)?;
    /// ```
    fn lu -> DynLuResult => lu_ad;
    where {
        T: crate::runtime::dispatch::RealLuLinalgDispatchValue + DynAdTensorTyped,
    }
);

eager_unary!(
    /// Eager AD symmetric/Hermitian eigen decomposition.
    ///
    /// Equivalent to `crate::eigen_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro::ad::eigen(&a)?;
    /// ```
    fn eigen -> DynEigenResult => eigen_ad;
    where {
        T: RealLinalgRuntimeValue + DynAdTensorTyped,
    }
);

eager_binary!(
    /// Eager AD least-squares solve.
    ///
    /// Equivalent to `crate::lstsq_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro::ad::lstsq(&a, &b)?;
    /// ```
    fn lstsq -> DynLstsqResult => lstsq_ad;
    where {
        T: RealLinalgRuntimeValue + DynAdTensorTyped,
    }
);

eager_unary!(
    /// Eager AD Cholesky.
    ///
    /// Equivalent to `crate::cholesky_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro::ad::cholesky(&a)?;
    /// ```
    fn cholesky -> AdTensor<T> => cholesky_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_binary!(
    /// Eager AD linear solve.
    ///
    /// Equivalent to `crate::solve_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro::ad::solve(&a, &b)?;
    /// ```
    fn solve -> AdTensor<T> => solve_ad;
    where {
        T: LinalgRuntimeValue,
    }
);

eager_unary!(
    /// Eager AD inverse.
    ///
    /// Equivalent to `crate::inv_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro::ad::inv(&a)?;
    /// ```
    fn inv -> AdTensor<T> => inv_ad;
    where {
        T: RealLinalgRuntimeValue,
    }
);

eager_unary!(
    /// Eager AD determinant.
    ///
    /// Equivalent to `crate::det_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro::ad::det(&a)?;
    /// ```
    fn det -> AdTensor<T> => det_ad;
    where {
        T: crate::runtime::dispatch::ScaledRealLinalgDispatchValue,
    }
);

eager_unary!(
    /// Eager AD slogdet.
    ///
    /// Equivalent to `crate::slogdet_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro::ad::slogdet(&a)?;
    /// ```
    fn slogdet -> DynSlogdetResult => slogdet_ad;
    where {
        T: crate::runtime::dispatch::SlogdetLinalgDispatchValue + DynAdTensorTyped,
    }
);

eager_unary!(
    /// Eager AD general eigendecomposition.
    ///
    /// Equivalent to `crate::eig_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro::ad::eig(&a)?;
    /// ```
    fn eig -> DynEigResult => eig_ad;
    where {
        T: ComplexLinalgRuntimeValue,
        Complex<T>: DynTensorTyped + DynAdTensorTyped,
    }
);

/// Eager LU factorization from an erased input carrier.
pub fn lu_factor_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynLuFactorResult> {
    dyn_unary_dispatch_all!(tensor, |value| lu_factor_dyn_impl(value, "lu_factor"))
}

/// Eager LU factorization with numerical status information from an erased input carrier.
pub fn lu_factor_ex_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynLuFactorExResult> {
    dyn_unary_dispatch_all!(tensor, |value| lu_factor_ex_dyn_impl(value, "lu_factor_ex"))
}

/// Eager LU solve from erased input carriers.
pub fn lu_solve_dyn(
    factors: DynAdTensorRef<'_>,
    rhs: DynAdTensorRef<'_>,
    pivots: &tenferro_tensor::Tensor<i32>,
) -> Result<DynAdTensor> {
    dyn_binary_dispatch_same_dtype!(factors, rhs, "lu_solve_dyn", |lhs, rhs| {
        lu_solve_dyn_impl(lhs, rhs, pivots, "lu_solve")
    })
}

/// Eager solve with numerical status information from an erased input carrier.
pub fn solve_ex_dyn(a: DynAdTensorRef<'_>, b: DynAdTensorRef<'_>) -> Result<DynSolveExResult> {
    dyn_binary_dispatch_same_dtype!(a, b, "solve_ex_dyn", |lhs, rhs| {
        solve_ex_dyn_impl(lhs, rhs, "solve_ex")
    })
}

/// Eager inverse with numerical status information from an erased input carrier.
pub fn inv_ex_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynInvExResult> {
    dyn_unary_dispatch_all!(tensor, |value| inv_ex_dyn_impl(value, "inv_ex"))
}

/// Eager Cholesky with numerical status information from an erased input carrier.
pub fn cholesky_ex_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynCholeskyExResult> {
    dyn_unary_dispatch_all!(tensor, |value| cholesky_ex_dyn_impl(value, "cholesky_ex"))
}

/// Eager AD SVD from an erased input carrier.
pub fn svd_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynSvdResult> {
    dyn_unary_dispatch_split!(
        tensor,
        |value| eager_into_dyn_result!(svd_real(value), DynSvdResult { u, s, vt }),
        |value| eager_into_dyn_result!(svd(value), DynSvdResult { u, s, vt })
    )
}

/// Eager AD QR from an erased input carrier.
pub fn qr_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynQrResult> {
    dyn_unary_dispatch_real_only!(tensor, "qr_dyn", |value| {
        eager_into_dyn_result!(qr(value), DynQrResult { q, r })
    })
}

/// Eager AD LU from an erased input carrier.
pub fn lu_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynLuResult> {
    dyn_unary_dispatch_real_only!(tensor, "lu_dyn", |value| {
        eager_into_dyn_result!(lu(value), DynLuResult { p, l, u })
    })
}

/// Eager AD symmetric eigen decomposition from an erased input carrier.
pub fn eigen_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynEigenResult> {
    dyn_unary_dispatch_real_only!(tensor, "eigen_dyn", |value| {
        eager_into_dyn_result!(eigen(value), DynEigenResult { values, vectors })
    })
}

/// Eager AD general eigendecomposition from an erased input carrier.
pub fn eig_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynEigResult> {
    dyn_unary_dispatch_real_only!(tensor, "eig_dyn", |value| {
        eager_into_dyn_result!(eig(value), DynEigResult { values, vectors })
    })
}

/// Eager AD slogdet from an erased input carrier.
pub fn slogdet_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynSlogdetResult> {
    dyn_unary_dispatch_real_only!(tensor, "slogdet_dyn", |value| {
        eager_into_dyn_result!(slogdet(value), DynSlogdetResult { sign, logabsdet })
    })
}

/// Eager AD Cholesky from an erased input carrier.
pub fn cholesky_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    dyn_unary_dispatch_real_only!(tensor, "cholesky_dyn", |value| Ok(cholesky(value)?.into()))
}

/// Eager AD linear solve from erased input carriers.
pub fn solve_dyn(a: DynAdTensorRef<'_>, b: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    dyn_binary_dispatch_same_dtype!(a, b, "solve_dyn", |lhs, rhs| Ok(solve(lhs, rhs)?.into()))
}

/// Eager AD triangular solve from erased input carriers.
pub fn solve_triangular_dyn(a: DynAdTensorRef<'_>, b: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    dyn_binary_dispatch_same_dtype!(a, b, "solve_triangular_dyn", |lhs, rhs| Ok(
        solve_triangular(lhs, rhs)?.into()
    ))
}

/// Eager AD inverse from an erased input carrier.
pub fn inv_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    dyn_unary_dispatch_real_only!(tensor, "inv_dyn", |value| Ok(inv(value)?.into()))
}

/// Eager AD determinant from an erased input carrier.
pub fn det_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    dyn_unary_dispatch_real_only!(tensor, "det_dyn", |value| Ok(det(value)?.into()))
}

/// Eager AD pseudoinverse from an erased input carrier.
pub fn pinv_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    dyn_unary_dispatch_real_only!(tensor, "pinv_dyn", |value| Ok(pinv(value)?.into()))
}

/// Eager AD matrix exponential from an erased input carrier.
pub fn matrix_exp_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    dyn_unary_dispatch_real_only!(tensor, "matrix_exp_dyn", |value| Ok(
        matrix_exp(value)?.into()
    ))
}

/// Eager AD norm from an erased input carrier.
pub fn norm_dyn(tensor: DynAdTensorRef<'_>) -> Result<DynAdTensor> {
    dyn_unary_dispatch_real_only!(tensor, "norm_dyn", |value| Ok(norm(value)?.into()))
}

/// Eager AD least-squares from erased input carriers.
pub fn lstsq_dyn(a: DynAdTensorRef<'_>, b: DynAdTensorRef<'_>) -> Result<DynLstsqResult> {
    match (a, b) {
        (DynAdTensorRef::F32(lhs), DynAdTensorRef::F32(rhs)) => {
            eager_into_dyn_result!(lstsq(lhs, rhs), DynLstsqResult { x, residual })
        }
        (DynAdTensorRef::F64(lhs), DynAdTensorRef::F64(rhs)) => {
            eager_into_dyn_result!(lstsq(lhs, rhs), DynLstsqResult { x, residual })
        }
        (lhs @ DynAdTensorRef::F32(_), rhs @ DynAdTensorRef::F64(_))
        | (lhs @ DynAdTensorRef::F64(_), rhs @ DynAdTensorRef::F32(_)) => Err(same_dtype_error(
            "lstsq_dyn",
            lhs.scalar_type(),
            rhs.scalar_type(),
        )),
        (lhs, rhs) => Err(Error::InvalidAdTensor {
            message: format!(
                "lstsq_dyn requires real-valued operands, got lhs={:?}, rhs={:?}",
                lhs.scalar_type(),
                rhs.scalar_type()
            ),
        }),
    }
}

eager_unary!(
    /// Eager AD pseudoinverse.
    ///
    /// Equivalent to `crate::pinv_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro::ad::pinv(&a)?;
    /// ```
    fn pinv -> AdTensor<T> => pinv_ad;
    where {
        T: crate::runtime::dispatch::ScaledRealLinalgDispatchValue,
    }
);

eager_unary!(
    /// Eager AD matrix exponential.
    ///
    /// Equivalent to `crate::matrix_exp_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro::ad::matrix_exp(&a)?;
    /// ```
    fn matrix_exp -> AdTensor<T> => matrix_exp_ad;
    where {
        T: crate::runtime::dispatch::RealMatrixExpLinalgDispatchValue,
    }
);

eager_binary!(
    /// Eager AD triangular solve (upper=true by default).
    ///
    /// Equivalent to `crate::solve_triangular_ad(...).run()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = tenferro::ad::solve_triangular(&a, &b)?;
    /// ```
    fn solve_triangular -> AdTensor<T> => solve_triangular_ad;
    where {
        T: LinalgRuntimeValue,
    }
);

/// Eager AD norm (Frobenius by default).
///
/// Equivalent to `crate::norm_ad(...).run()`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro::ad::norm(&a)?;
/// ```
pub fn norm<T: Scalar>(tensor: &AdTensor<T>) -> Result<AdTensor<T>>
where
    T: crate::runtime::dispatch::NormLinalgDispatchValue,
{
    norm_ad(tensor).kind(NormKind::Fro).run()
}
