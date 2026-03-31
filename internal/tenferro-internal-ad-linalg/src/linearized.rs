use chainrules_core::AutodiffError;
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use tenferro_algebra::{Conjugate, Scalar};
use tenferro_internal_ad_core::{
    AdResult, CheckpointHint, DynValue, LinearizableOp, LinearizedOp, Schema, SlotSchema,
};
use tenferro_internal_frontend_core::{DynTensor, DynTensorTyped, StructuredTensor};
use tenferro_internal_runtime::contracts::{LinalgRuntimeValue, RealLinalgRuntimeValue};
use tenferro_internal_runtime::dispatch::{
    with_linalg_runtime, LuLinalgDispatchValue, NormLinalgDispatchValue,
    RealMatrixExpLinalgDispatchValue, ScaledRealLinalgDispatchValue, SlogdetLinalgDispatchValue,
};
use tenferro_linalg::backend::LinalgCapabilityOp;
use tenferro_linalg::{
    cholesky, cholesky_frule, cholesky_rrule, det, det_frule, det_rrule, eig, eig_frule, eig_rrule,
    eigen, eigen_frule, eigen_rrule, inv, inv_frule, inv_rrule, lstsq, lstsq_frule, lstsq_rrule,
    lu, lu_frule, lu_rrule, matrix_exp, matrix_exp_frule, matrix_exp_rrule, norm, norm_frule,
    norm_rrule, pinv, pinv_frule, pinv_rrule, qr, qr_frule, qr_rrule, slogdet, slogdet_frule,
    slogdet_rrule, solve, solve_frule, solve_rrule, solve_triangular, solve_triangular_frule,
    solve_triangular_rrule, svd, svd_frule, svd_rrule, EigCotangent, EigenCotangent,
    KernelLinalgScalar, LuCotangent, LuPivot, NormKind, QrCotangent, SlogdetCotangent,
    SvdCotangent, SvdOptions,
};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

use crate::{Error, Result};

#[derive(Clone, Copy)]
pub struct SolveOp;

#[derive(Clone, Copy)]
pub struct LstsqOp;

#[derive(Clone, Copy)]
pub struct SolveTriangularOp {
    upper: bool,
}

#[derive(Clone, Copy)]
pub struct NormOp {
    kind: NormKind,
}

#[derive(Clone, Copy)]
pub struct DetOp;

#[derive(Clone, Copy)]
pub struct InvOp;

#[derive(Clone, Copy)]
pub struct SlogdetOp;

#[derive(Clone, Copy)]
pub struct CholeskyOp;

#[derive(Clone, Copy)]
pub struct LuOp {
    pivot: LuPivot,
}

#[derive(Clone, Copy)]
pub struct QrOp;

#[derive(Clone, Copy)]
pub struct EigOp;

#[derive(Clone, Copy)]
pub struct EigenOp;

#[derive(Clone, Default)]
pub struct SvdOp {
    options: Option<SvdOptions>,
}

#[derive(Clone, Default)]
pub struct PInvOp {
    rcond: Option<f64>,
}

#[derive(Clone, Copy)]
pub struct MatrixExpOp;

pub struct DynQrValues {
    pub q: DynValue,
    pub r: DynValue,
}

pub struct DynLstsqValues {
    pub x: DynValue,
    pub residual: DynValue,
}

pub struct DynLuValues {
    pub p: DynValue,
    pub l: DynValue,
    pub u: DynValue,
}

pub struct DynEigValues {
    pub values: DynValue,
    pub vectors: DynValue,
}

pub struct DynEigenValues {
    pub values: DynValue,
    pub vectors: DynValue,
}

pub struct DynSlogdetValues {
    pub sign: DynValue,
    pub logabsdet: DynValue,
}

pub struct DynSvdValues {
    pub u: DynValue,
    pub s: DynValue,
    pub vt: DynValue,
}

#[doc(hidden)]
pub struct SolveLinearized {
    a: DynTensor,
    b: DynTensor,
}

#[doc(hidden)]
pub struct LstsqLinearized {
    a: DynTensor,
    b: DynTensor,
}

#[doc(hidden)]
pub struct SolveTriangularLinearized {
    a: DynTensor,
    b: DynTensor,
    upper: bool,
}

#[doc(hidden)]
pub struct NormLinearized {
    input: DynTensor,
    kind: NormKind,
}

#[doc(hidden)]
pub struct DetLinearized {
    input: DynTensor,
}

#[doc(hidden)]
pub struct InvLinearized {
    input: DynTensor,
}

#[doc(hidden)]
pub struct SlogdetLinearized {
    input: DynTensor,
}

#[doc(hidden)]
pub struct CholeskyLinearized {
    input: DynTensor,
}

#[doc(hidden)]
pub struct LuLinearized {
    input: DynTensor,
    pivot: LuPivot,
}

#[doc(hidden)]
pub struct QrLinearized {
    input: DynTensor,
}

#[doc(hidden)]
pub struct EigLinearized {
    input: DynTensor,
}

#[doc(hidden)]
pub struct EigenLinearized {
    input: DynTensor,
}

#[doc(hidden)]
pub struct SvdLinearized {
    input: DynTensor,
    options: Option<SvdOptions>,
}

#[doc(hidden)]
pub struct PInvLinearized {
    input: DynTensor,
    rcond: Option<f64>,
}

#[doc(hidden)]
pub struct MatrixExpLinearized {
    input: DynTensor,
}

fn differentiable_schema(slots: usize) -> Schema {
    Schema {
        slots: (0..slots)
            .map(|_| SlotSchema {
                differentiable: true,
                auxiliary: false,
            })
            .collect(),
    }
}

fn slogdet_output_schema() -> Schema {
    Schema {
        slots: vec![
            SlotSchema {
                differentiable: false,
                auxiliary: true,
            },
            SlotSchema {
                differentiable: true,
                auxiliary: false,
            },
        ],
    }
}

fn lstsq_output_schema() -> Schema {
    Schema {
        slots: vec![
            SlotSchema {
                differentiable: true,
                auxiliary: false,
            },
            SlotSchema {
                differentiable: false,
                auxiliary: true,
            },
        ],
    }
}

fn lu_output_schema() -> Schema {
    Schema {
        slots: vec![
            SlotSchema {
                differentiable: false,
                auxiliary: true,
            },
            SlotSchema {
                differentiable: true,
                auxiliary: false,
            },
            SlotSchema {
                differentiable: true,
                auxiliary: false,
            },
        ],
    }
}

fn invalid_argument(message: impl Into<String>) -> Error {
    AutodiffError::InvalidArgument(message.into()).into()
}

fn into_ad_error(error: Error) -> AutodiffError {
    match error {
        Error::Autodiff(error) => error,
        other => AutodiffError::InvalidArgument(other.to_string()),
    }
}

macro_rules! dispatch_linalg {
    ($ty:ty, $op:expr, $cap:expr, |$ctx:ident| $body:expr) => {{
        with_linalg_runtime::<$ty, _>($op, $cap, |$ctx| $body, |$ctx| $body, |$ctx| $body)
    }};
}

fn dense_dyn_tensor_typed<T>(value: &DynTensor, context: &str) -> Result<DenseTensor<T>>
where
    T: DynTensorTyped + Copy,
{
    let structured = T::structured_ref(value)
        .ok_or_else(|| invalid_argument(format!("{context} requires matching dtypes")))?;
    Ok(structured.to_dense()?)
}

fn optional_dense_dyn_tensor_typed<T>(
    value: &Option<DynTensor>,
    context: &str,
) -> Result<Option<DenseTensor<T>>>
where
    T: DynTensorTyped + Copy,
{
    value
        .as_ref()
        .map(|tensor| dense_dyn_tensor_typed::<T>(tensor, context))
        .transpose()
}

fn dense_zeros_like<T>(like: &DenseTensor<T>) -> Result<DenseTensor<T>>
where
    T: Scalar + Zero + Copy,
{
    let total: usize = like.dims().iter().product();
    DenseTensor::from_slice(
        &vec![T::zero(); total],
        like.dims(),
        MemoryOrder::ColumnMajor,
    )
    .map_err(Error::from)
}

fn dense_optional_or_zero<T>(
    value: &Option<DynTensor>,
    like: &DenseTensor<T>,
    context: &str,
) -> Result<DenseTensor<T>>
where
    T: DynTensorTyped + Scalar + Zero + Copy,
{
    optional_dense_dyn_tensor_typed::<T>(value, context)?.map_or_else(|| dense_zeros_like(like), Ok)
}

fn dyn_from_dense<T>(value: DenseTensor<T>) -> DynTensor
where
    T: DynTensorTyped + Copy,
{
    T::into_dyn(StructuredTensor::from(value))
}

fn solve_primal_t<T>(a: &StructuredTensor<T>, b: &StructuredTensor<T>) -> Result<DynTensor>
where
    T: LinalgRuntimeValue + DynTensorTyped + Copy,
{
    let dense_a = a.to_dense()?;
    let dense_b = b.to_dense()?;
    let output = dispatch_linalg!(T, "solve_dyn_value", LinalgCapabilityOp::Solve, |ctx| {
        solve(ctx, &dense_a, &dense_b).map_err(Error::from)
    })?;
    Ok(dyn_from_dense(output))
}

fn solve_jvp_t<T>(
    a: &StructuredTensor<T>,
    b: &StructuredTensor<T>,
    tangents: &[Option<DynTensor>],
) -> Result<Option<DynTensor>>
where
    T: LinalgRuntimeValue + DynTensorTyped + Scalar + Zero + Copy,
{
    if tangents.iter().all(Option::is_none) {
        return Ok(None);
    }
    let dense_a = a.to_dense()?;
    let dense_b = b.to_dense()?;
    let tangent_a = dense_optional_or_zero(&tangents[0], &dense_a, "solve_jvp tangent_a")?;
    let tangent_b = dense_optional_or_zero(&tangents[1], &dense_b, "solve_jvp tangent_b")?;
    let (_, tangent) = dispatch_linalg!(T, "solve_jvp", LinalgCapabilityOp::Solve, |ctx| {
        solve_frule(ctx, &dense_a, &dense_b, &tangent_a, &tangent_b).map_err(Error::from)
    })?;
    Ok(Some(dyn_from_dense(tangent)))
}

fn solve_vjp_t<T>(
    a: &StructuredTensor<T>,
    b: &StructuredTensor<T>,
    cotangent: &DynTensor,
    input_grad_mask: &[bool],
) -> Result<Vec<Option<DynTensor>>>
where
    T: LinalgRuntimeValue + DynTensorTyped + Copy,
{
    if !input_grad_mask.iter().any(|needed| *needed) {
        return Ok(vec![None, None]);
    }
    let dense_a = a.to_dense()?;
    let dense_b = b.to_dense()?;
    let dense_cotangent = dense_dyn_tensor_typed::<T>(cotangent, "solve_vjp")?;
    let grad = dispatch_linalg!(T, "solve_vjp", LinalgCapabilityOp::Solve, |ctx| {
        solve_rrule(ctx, &dense_a, &dense_b, &dense_cotangent).map_err(Error::from)
    })?;
    Ok(vec![
        input_grad_mask[0].then(|| dyn_from_dense(grad.a)),
        input_grad_mask[1].then(|| dyn_from_dense(grad.b)),
    ])
}

fn lstsq_primal_t<T>(a: &StructuredTensor<T>, b: &StructuredTensor<T>) -> Result<Vec<DynTensor>>
where
    T: RealLinalgRuntimeValue + DynTensorTyped + Conjugate + Copy,
{
    let dense_a = a.to_dense()?;
    let dense_b = b.to_dense()?;
    let output = dispatch_linalg!(T, "lstsq_dyn_values", LinalgCapabilityOp::Lstsq, |ctx| {
        lstsq(ctx, &dense_a, &dense_b).map_err(Error::from)
    })?;
    Ok(vec![
        dyn_from_dense(output.x),
        dyn_from_dense(output.residual),
    ])
}

fn lstsq_jvp_t<T>(
    a: &StructuredTensor<T>,
    b: &StructuredTensor<T>,
    tangents: &[Option<DynTensor>],
) -> Result<Vec<Option<DynTensor>>>
where
    T: RealLinalgRuntimeValue + DynTensorTyped + Scalar + Zero + Conjugate + Copy,
{
    if tangents.iter().all(Option::is_none) {
        return Ok(vec![None, None]);
    }
    let dense_a = a.to_dense()?;
    let dense_b = b.to_dense()?;
    let tangent_a = dense_optional_or_zero(&tangents[0], &dense_a, "lstsq_jvp tangent_a")?;
    let tangent_b = dense_optional_or_zero(&tangents[1], &dense_b, "lstsq_jvp tangent_b")?;
    let (_, tangent) = dispatch_linalg!(T, "lstsq_jvp", LinalgCapabilityOp::Lstsq, |ctx| {
        lstsq_frule(ctx, &dense_a, &dense_b, &tangent_a, &tangent_b).map_err(Error::from)
    })?;
    Ok(vec![
        Some(dyn_from_dense(tangent.x)),
        Some(dyn_from_dense(tangent.residual)),
    ])
}

fn lstsq_vjp_t<T>(
    a: &StructuredTensor<T>,
    b: &StructuredTensor<T>,
    output_cotangents: &[Option<DynTensor>],
    input_grad_mask: &[bool],
) -> Result<Vec<Option<DynTensor>>>
where
    T: RealLinalgRuntimeValue + DynTensorTyped + Conjugate + Copy,
{
    if !input_grad_mask.iter().any(|needed| *needed) {
        return Ok(vec![None, None]);
    }
    if output_cotangents
        .get(1)
        .and_then(|value| value.as_ref())
        .is_some()
    {
        return Err(invalid_argument(
            "lstsq residual cotangent is unsupported; residual is an auxiliary output",
        ));
    }
    let Some(cotangent_x) = output_cotangents[0].as_ref() else {
        return Ok(vec![None, None]);
    };
    let dense_a = a.to_dense()?;
    let dense_b = b.to_dense()?;
    let dense_cotangent_x = dense_dyn_tensor_typed::<T>(cotangent_x, "lstsq_vjp x")?;
    let grad = dispatch_linalg!(T, "lstsq_vjp", LinalgCapabilityOp::Lstsq, |ctx| {
        lstsq_rrule(ctx, &dense_a, &dense_b, &dense_cotangent_x).map_err(Error::from)
    })?;
    Ok(vec![
        input_grad_mask[0].then(|| dyn_from_dense(grad.a)),
        input_grad_mask[1].then(|| dyn_from_dense(grad.b)),
    ])
}

fn solve_triangular_primal_t<T>(
    a: &StructuredTensor<T>,
    b: &StructuredTensor<T>,
    upper: bool,
) -> Result<DynTensor>
where
    T: LinalgRuntimeValue + DynTensorTyped + Copy,
{
    let dense_a = a.to_dense()?;
    let dense_b = b.to_dense()?;
    let output = dispatch_linalg!(
        T,
        "solve_triangular_dyn_value",
        LinalgCapabilityOp::SolveTriangular,
        |ctx| { solve_triangular(ctx, &dense_a, &dense_b, upper).map_err(Error::from) }
    )?;
    Ok(dyn_from_dense(output))
}

fn solve_triangular_jvp_t<T>(
    a: &StructuredTensor<T>,
    b: &StructuredTensor<T>,
    tangents: &[Option<DynTensor>],
    upper: bool,
) -> Result<Option<DynTensor>>
where
    T: LinalgRuntimeValue + DynTensorTyped + Scalar + Zero + Copy,
{
    if tangents.iter().all(Option::is_none) {
        return Ok(None);
    }
    let dense_a = a.to_dense()?;
    let dense_b = b.to_dense()?;
    let tangent_a =
        dense_optional_or_zero(&tangents[0], &dense_a, "solve_triangular_jvp tangent_a")?;
    let tangent_b =
        dense_optional_or_zero(&tangents[1], &dense_b, "solve_triangular_jvp tangent_b")?;
    let (_, tangent) = dispatch_linalg!(
        T,
        "solve_triangular_jvp",
        LinalgCapabilityOp::SolveTriangular,
        |ctx| {
            solve_triangular_frule(ctx, &dense_a, &dense_b, &tangent_a, &tangent_b, upper)
                .map_err(Error::from)
        }
    )?;
    Ok(Some(dyn_from_dense(tangent)))
}

fn solve_triangular_vjp_t<T>(
    a: &StructuredTensor<T>,
    b: &StructuredTensor<T>,
    cotangent: &DynTensor,
    input_grad_mask: &[bool],
    upper: bool,
) -> Result<Vec<Option<DynTensor>>>
where
    T: LinalgRuntimeValue + DynTensorTyped + Copy,
{
    if !input_grad_mask.iter().any(|needed| *needed) {
        return Ok(vec![None, None]);
    }
    let dense_a = a.to_dense()?;
    let dense_b = b.to_dense()?;
    let dense_cotangent = dense_dyn_tensor_typed::<T>(cotangent, "solve_triangular_vjp")?;
    let grad = dispatch_linalg!(
        T,
        "solve_triangular_vjp",
        LinalgCapabilityOp::SolveTriangular,
        |ctx| {
            solve_triangular_rrule(ctx, &dense_a, &dense_b, &dense_cotangent, upper)
                .map_err(Error::from)
        }
    )?;
    Ok(vec![
        input_grad_mask[0].then(|| dyn_from_dense(grad.a)),
        input_grad_mask[1].then(|| dyn_from_dense(grad.b)),
    ])
}

fn inv_primal_t<T>(input: &StructuredTensor<T>) -> Result<DynTensor>
where
    T: RealLinalgRuntimeValue + DynTensorTyped + Copy,
{
    let dense_input = input.to_dense()?;
    let output = dispatch_linalg!(T, "inv_dyn_value", LinalgCapabilityOp::Inv, |ctx| {
        inv(ctx, &dense_input).map_err(Error::from)
    })?;
    Ok(dyn_from_dense(output))
}

fn inv_jvp_t<T>(
    input: &StructuredTensor<T>,
    tangent: &Option<DynTensor>,
) -> Result<Option<DynTensor>>
where
    T: RealLinalgRuntimeValue + DynTensorTyped + Scalar + Zero + Copy,
{
    if tangent.is_none() {
        return Ok(None);
    }
    let dense_input = input.to_dense()?;
    let dense_tangent = dense_optional_or_zero(tangent, &dense_input, "inv_jvp tangent")?;
    let (_, output_tangent) = dispatch_linalg!(T, "inv_jvp", LinalgCapabilityOp::Inv, |ctx| {
        inv_frule(ctx, &dense_input, &dense_tangent).map_err(Error::from)
    })?;
    Ok(Some(dyn_from_dense(output_tangent)))
}

fn inv_vjp_t<T>(
    input: &StructuredTensor<T>,
    cotangent: &DynTensor,
    input_grad_mask: &[bool],
) -> Result<Vec<Option<DynTensor>>>
where
    T: RealLinalgRuntimeValue + DynTensorTyped + Copy,
{
    if !input_grad_mask[0] {
        return Ok(vec![None]);
    }
    let dense_input = input.to_dense()?;
    let dense_cotangent = dense_dyn_tensor_typed::<T>(cotangent, "inv_vjp")?;
    let grad = dispatch_linalg!(T, "inv_vjp", LinalgCapabilityOp::Inv, |ctx| {
        inv_rrule(ctx, &dense_input, &dense_cotangent).map_err(Error::from)
    })?;
    Ok(vec![Some(dyn_from_dense(grad))])
}

fn slogdet_primal_t<T>(input: &StructuredTensor<T>) -> Result<Vec<DynTensor>>
where
    T: SlogdetLinalgDispatchValue + DynTensorTyped + Copy,
{
    let dense_input = input.to_dense()?;
    let output = dispatch_linalg!(T, "slogdet_dyn_value", LinalgCapabilityOp::Slogdet, |ctx| {
        slogdet(ctx, &dense_input).map_err(Error::from)
    })?;
    Ok(vec![
        dyn_from_dense(output.sign),
        dyn_from_dense(output.logabsdet),
    ])
}

fn slogdet_jvp_t<T>(
    input: &StructuredTensor<T>,
    tangent: &Option<DynTensor>,
) -> Result<Vec<Option<DynTensor>>>
where
    T: SlogdetLinalgDispatchValue + DynTensorTyped + Scalar + Zero + Copy,
{
    if tangent.is_none() {
        return Ok(vec![None, None]);
    }
    let dense_input = input.to_dense()?;
    let dense_tangent = dense_optional_or_zero(tangent, &dense_input, "slogdet_jvp tangent")?;
    let (_, tangent_output) =
        dispatch_linalg!(T, "slogdet_jvp", LinalgCapabilityOp::Slogdet, |ctx| {
            slogdet_frule(ctx, &dense_input, &dense_tangent).map_err(Error::from)
        })?;
    Ok(vec![
        Some(dyn_from_dense(tangent_output.sign)),
        Some(dyn_from_dense(tangent_output.logabsdet)),
    ])
}

fn slogdet_vjp_t<T>(
    input: &StructuredTensor<T>,
    output_cotangents: &[Option<DynTensor>],
    input_grad_mask: &[bool],
) -> Result<Vec<Option<DynTensor>>>
where
    T: SlogdetLinalgDispatchValue + DynTensorTyped + Copy,
{
    if !input_grad_mask[0] {
        return Ok(vec![None]);
    }
    let dense_input = input.to_dense()?;
    let cotangent = SlogdetCotangent {
        logabsdet: optional_dense_dyn_tensor_typed::<T>(
            &output_cotangents[1],
            "slogdet_vjp logabsdet",
        )?,
    };
    if cotangent.logabsdet.is_none() {
        return Ok(vec![None]);
    }
    let grad = dispatch_linalg!(T, "slogdet_vjp", LinalgCapabilityOp::Slogdet, |ctx| {
        slogdet_rrule(ctx, &dense_input, &cotangent).map_err(Error::from)
    })?;
    Ok(vec![Some(dyn_from_dense(grad))])
}

fn cholesky_primal_t<T>(input: &StructuredTensor<T>) -> Result<DynTensor>
where
    T: RealLinalgRuntimeValue + DynTensorTyped + Copy,
{
    let dense_input = input.to_dense()?;
    let output = dispatch_linalg!(
        T,
        "cholesky_dyn_value",
        LinalgCapabilityOp::Cholesky,
        |ctx| { cholesky(ctx, &dense_input).map_err(Error::from) }
    )?;
    Ok(dyn_from_dense(output))
}

fn cholesky_jvp_t<T>(
    input: &StructuredTensor<T>,
    tangent: &Option<DynTensor>,
) -> Result<Option<DynTensor>>
where
    T: RealLinalgRuntimeValue + DynTensorTyped + Scalar + Zero + Copy,
{
    if tangent.is_none() {
        return Ok(None);
    }
    let dense_input = input.to_dense()?;
    let dense_tangent = dense_optional_or_zero(tangent, &dense_input, "cholesky_jvp tangent")?;
    let (_, output_tangent) =
        dispatch_linalg!(T, "cholesky_jvp", LinalgCapabilityOp::Cholesky, |ctx| {
            cholesky_frule(ctx, &dense_input, &dense_tangent).map_err(Error::from)
        })?;
    Ok(Some(dyn_from_dense(output_tangent)))
}

fn cholesky_vjp_t<T>(
    input: &StructuredTensor<T>,
    cotangent: &DynTensor,
    input_grad_mask: &[bool],
) -> Result<Vec<Option<DynTensor>>>
where
    T: RealLinalgRuntimeValue + DynTensorTyped + Copy,
{
    if !input_grad_mask[0] {
        return Ok(vec![None]);
    }
    let dense_input = input.to_dense()?;
    let dense_cotangent = dense_dyn_tensor_typed::<T>(cotangent, "cholesky_vjp")?;
    let grad = dispatch_linalg!(T, "cholesky_vjp", LinalgCapabilityOp::Cholesky, |ctx| {
        cholesky_rrule(ctx, &dense_input, &dense_cotangent).map_err(Error::from)
    })?;
    Ok(vec![Some(dyn_from_dense(grad))])
}

fn lu_primal_t<T>(input: &StructuredTensor<T>, pivot: LuPivot) -> Result<Vec<DynTensor>>
where
    T: LuLinalgDispatchValue + DynTensorTyped + Copy,
{
    let dense_input = input.to_dense()?;
    let output = dispatch_linalg!(T, "lu_dyn_value", LinalgCapabilityOp::LuFactor, |ctx| {
        lu(ctx, &dense_input, pivot).map_err(Error::from)
    })?;
    Ok(vec![
        dyn_from_dense(output.p),
        dyn_from_dense(output.l),
        dyn_from_dense(output.u),
    ])
}

fn lu_jvp_t<T>(
    input: &StructuredTensor<T>,
    tangent: &Option<DynTensor>,
    pivot: LuPivot,
) -> Result<Vec<Option<DynTensor>>>
where
    T: LuLinalgDispatchValue + DynTensorTyped + Scalar + Zero + Copy,
{
    if tangent.is_none() {
        return Ok(vec![None, None, None]);
    }
    let dense_input = input.to_dense()?;
    let dense_tangent = dense_optional_or_zero(tangent, &dense_input, "lu_jvp tangent")?;
    let (_, tangent_output) = dispatch_linalg!(T, "lu_jvp", LinalgCapabilityOp::LuFactor, |ctx| {
        lu_frule(ctx, &dense_input, &dense_tangent, pivot).map_err(Error::from)
    })?;
    Ok(vec![
        None,
        Some(dyn_from_dense(tangent_output.l)),
        Some(dyn_from_dense(tangent_output.u)),
    ])
}

fn lu_vjp_t<T>(
    input: &StructuredTensor<T>,
    output_cotangents: &[Option<DynTensor>],
    input_grad_mask: &[bool],
    pivot: LuPivot,
) -> Result<Vec<Option<DynTensor>>>
where
    T: LuLinalgDispatchValue + DynTensorTyped + Copy,
{
    if !input_grad_mask[0] {
        return Ok(vec![None]);
    }
    if output_cotangents
        .first()
        .and_then(|value| value.as_ref())
        .is_some()
    {
        return Err(invalid_argument(
            "lu permutation cotangent is unsupported; permutation output is auxiliary",
        ));
    }
    let dense_input = input.to_dense()?;
    let cotangent = LuCotangent {
        l: optional_dense_dyn_tensor_typed::<T>(&output_cotangents[1], "lu_vjp l")?,
        u: optional_dense_dyn_tensor_typed::<T>(&output_cotangents[2], "lu_vjp u")?,
    };
    if cotangent.l.is_none() && cotangent.u.is_none() {
        return Ok(vec![None]);
    }
    let grad = dispatch_linalg!(T, "lu_vjp", LinalgCapabilityOp::LuFactor, |ctx| {
        lu_rrule(ctx, &dense_input, &cotangent, pivot).map_err(Error::from)
    })?;
    Ok(vec![Some(dyn_from_dense(grad))])
}

fn pinv_primal_t<T>(input: &StructuredTensor<T>, rcond: Option<f64>) -> Result<DynTensor>
where
    T: ScaledRealLinalgDispatchValue + DynTensorTyped + Conjugate + Copy,
{
    let dense_input = input.to_dense()?;
    let output = dispatch_linalg!(T, "pinv_dyn_value", LinalgCapabilityOp::Pinv, |ctx| {
        pinv(ctx, &dense_input, rcond).map_err(Error::from)
    })?;
    Ok(dyn_from_dense(output))
}

fn pinv_jvp_t<T>(
    input: &StructuredTensor<T>,
    tangent: &Option<DynTensor>,
    rcond: Option<f64>,
) -> Result<Option<DynTensor>>
where
    T: ScaledRealLinalgDispatchValue + DynTensorTyped + Scalar + Zero + Conjugate + Copy,
{
    if tangent.is_none() {
        return Ok(None);
    }
    let dense_input = input.to_dense()?;
    let dense_tangent = dense_optional_or_zero(tangent, &dense_input, "pinv_jvp tangent")?;
    let (_, output_tangent) = dispatch_linalg!(T, "pinv_jvp", LinalgCapabilityOp::Pinv, |ctx| {
        pinv_frule(ctx, &dense_input, &dense_tangent, rcond).map_err(Error::from)
    })?;
    Ok(Some(dyn_from_dense(output_tangent)))
}

fn pinv_vjp_t<T>(
    input: &StructuredTensor<T>,
    cotangent: &DynTensor,
    input_grad_mask: &[bool],
    rcond: Option<f64>,
) -> Result<Vec<Option<DynTensor>>>
where
    T: ScaledRealLinalgDispatchValue + DynTensorTyped + Conjugate + Copy,
{
    if !input_grad_mask[0] {
        return Ok(vec![None]);
    }
    let dense_input = input.to_dense()?;
    let dense_cotangent = dense_dyn_tensor_typed::<T>(cotangent, "pinv_vjp")?;
    let grad = dispatch_linalg!(T, "pinv_vjp", LinalgCapabilityOp::Pinv, |ctx| {
        pinv_rrule(ctx, &dense_input, &dense_cotangent, rcond).map_err(Error::from)
    })?;
    Ok(vec![Some(dyn_from_dense(grad))])
}

fn matrix_exp_primal_t<T>(input: &StructuredTensor<T>) -> Result<DynTensor>
where
    T: RealMatrixExpLinalgDispatchValue + DynTensorTyped + Copy,
{
    let dense_input = input.to_dense()?;
    let output = dispatch_linalg!(
        T,
        "matrix_exp_dyn_value",
        LinalgCapabilityOp::MatrixExp,
        |ctx| { matrix_exp(ctx, &dense_input).map_err(Error::from) }
    )?;
    Ok(dyn_from_dense(output))
}

fn matrix_exp_jvp_t<T>(
    input: &StructuredTensor<T>,
    tangent: &Option<DynTensor>,
) -> Result<Option<DynTensor>>
where
    T: RealMatrixExpLinalgDispatchValue + DynTensorTyped + Scalar + Zero + Copy,
{
    if tangent.is_none() {
        return Ok(None);
    }
    let dense_input = input.to_dense()?;
    let dense_tangent = dense_optional_or_zero(tangent, &dense_input, "matrix_exp_jvp tangent")?;
    let (_, output_tangent) =
        dispatch_linalg!(T, "matrix_exp_jvp", LinalgCapabilityOp::MatrixExp, |ctx| {
            matrix_exp_frule(ctx, &dense_input, &dense_tangent).map_err(Error::from)
        })?;
    Ok(Some(dyn_from_dense(output_tangent)))
}

fn matrix_exp_vjp_t<T>(
    input: &StructuredTensor<T>,
    cotangent: &DynTensor,
    input_grad_mask: &[bool],
) -> Result<Vec<Option<DynTensor>>>
where
    T: RealMatrixExpLinalgDispatchValue + DynTensorTyped + Copy,
{
    if !input_grad_mask[0] {
        return Ok(vec![None]);
    }
    let dense_input = input.to_dense()?;
    let dense_cotangent = dense_dyn_tensor_typed::<T>(cotangent, "matrix_exp_vjp")?;
    let grad = dispatch_linalg!(T, "matrix_exp_vjp", LinalgCapabilityOp::MatrixExp, |ctx| {
        matrix_exp_rrule(ctx, &dense_input, &dense_cotangent).map_err(Error::from)
    })?;
    Ok(vec![Some(dyn_from_dense(grad))])
}

fn eig_primal_t<T>(input: &StructuredTensor<T>) -> Result<Vec<DynTensor>>
where
    T: RealLinalgRuntimeValue
        + DynTensorTyped
        + num_traits::Float
        + KernelLinalgScalar<Real = T, Complex = num_complex::Complex<T>>
        + Copy,
    num_complex::Complex<T>: DynTensorTyped + Copy,
{
    let dense_input = input.to_dense()?;
    let output = dispatch_linalg!(T, "eig_dyn_value", LinalgCapabilityOp::Eig, |ctx| {
        eig(ctx, &dense_input).map_err(Error::from)
    })?;
    Ok(vec![
        dyn_from_dense(output.values),
        dyn_from_dense(output.vectors),
    ])
}

fn eig_jvp_t<T>(
    input: &StructuredTensor<T>,
    tangent: &Option<DynTensor>,
) -> Result<Vec<Option<DynTensor>>>
where
    T: RealLinalgRuntimeValue
        + DynTensorTyped
        + Scalar
        + Zero
        + num_traits::Float
        + KernelLinalgScalar<Real = T, Complex = num_complex::Complex<T>>
        + Copy,
    num_complex::Complex<T>: DynTensorTyped + Copy,
{
    if tangent.is_none() {
        return Ok(vec![None, None]);
    }
    let dense_input = input.to_dense()?;
    let dense_tangent = dense_optional_or_zero(tangent, &dense_input, "eig_jvp tangent")?;
    let (_, tangent_output) = dispatch_linalg!(T, "eig_jvp", LinalgCapabilityOp::Eig, |ctx| {
        eig_frule(ctx, &dense_input, &dense_tangent).map_err(Error::from)
    })?;
    Ok(vec![
        Some(dyn_from_dense(tangent_output.values)),
        Some(dyn_from_dense(tangent_output.vectors)),
    ])
}

fn eig_vjp_t<T>(
    input: &StructuredTensor<T>,
    output_cotangents: &[Option<DynTensor>],
    input_grad_mask: &[bool],
) -> Result<Vec<Option<DynTensor>>>
where
    T: RealLinalgRuntimeValue
        + DynTensorTyped
        + num_traits::Float
        + KernelLinalgScalar<Real = T, Complex = num_complex::Complex<T>>
        + Copy,
    num_complex::Complex<T>: DynTensorTyped + Copy,
{
    if !input_grad_mask[0] {
        return Ok(vec![None]);
    }
    let dense_input = input.to_dense()?;
    let cotangent = EigCotangent {
        values: optional_dense_dyn_tensor_typed::<num_complex::Complex<T>>(
            &output_cotangents[0],
            "eig_vjp values",
        )?,
        vectors: optional_dense_dyn_tensor_typed::<num_complex::Complex<T>>(
            &output_cotangents[1],
            "eig_vjp vectors",
        )?,
    };
    if cotangent.values.is_none() && cotangent.vectors.is_none() {
        return Ok(vec![None]);
    }
    let grad = dispatch_linalg!(T, "eig_vjp", LinalgCapabilityOp::Eig, |ctx| {
        eig_rrule(ctx, &dense_input, &cotangent).map_err(Error::from)
    })?;
    Ok(vec![Some(dyn_from_dense(grad))])
}

fn eigen_primal_t<T>(input: &StructuredTensor<T>) -> Result<Vec<DynTensor>>
where
    T: RealLinalgRuntimeValue + DynTensorTyped + Copy,
{
    let dense_input = input.to_dense()?;
    let output = dispatch_linalg!(T, "eigen_dyn_value", LinalgCapabilityOp::EigenSym, |ctx| {
        eigen(ctx, &dense_input).map_err(Error::from)
    })?;
    Ok(vec![
        dyn_from_dense(output.values),
        dyn_from_dense(output.vectors),
    ])
}

fn eigen_jvp_t<T>(
    input: &StructuredTensor<T>,
    tangent: &Option<DynTensor>,
) -> Result<Vec<Option<DynTensor>>>
where
    T: RealLinalgRuntimeValue + DynTensorTyped + Scalar + Zero + num_traits::Float + Copy,
{
    if tangent.is_none() {
        return Ok(vec![None, None]);
    }
    let dense_input = input.to_dense()?;
    let dense_tangent = dense_optional_or_zero(tangent, &dense_input, "eigen_jvp tangent")?;
    let (_, tangent_output) =
        dispatch_linalg!(T, "eigen_jvp", LinalgCapabilityOp::EigenSym, |ctx| {
            eigen_frule(ctx, &dense_input, &dense_tangent).map_err(Error::from)
        })?;
    Ok(vec![
        Some(dyn_from_dense(tangent_output.values)),
        Some(dyn_from_dense(tangent_output.vectors)),
    ])
}

fn eigen_vjp_t<T>(
    input: &StructuredTensor<T>,
    output_cotangents: &[Option<DynTensor>],
    input_grad_mask: &[bool],
) -> Result<Vec<Option<DynTensor>>>
where
    T: RealLinalgRuntimeValue + DynTensorTyped + num_traits::Float + Copy,
{
    if !input_grad_mask[0] {
        return Ok(vec![None]);
    }
    let dense_input = input.to_dense()?;
    let cotangent = EigenCotangent {
        values: optional_dense_dyn_tensor_typed::<T>(&output_cotangents[0], "eigen_vjp values")?,
        vectors: optional_dense_dyn_tensor_typed::<T>(&output_cotangents[1], "eigen_vjp vectors")?,
    };
    if cotangent.values.is_none() && cotangent.vectors.is_none() {
        return Ok(vec![None]);
    }
    let grad = dispatch_linalg!(T, "eigen_vjp", LinalgCapabilityOp::EigenSym, |ctx| {
        eigen_rrule(ctx, &dense_input, &cotangent).map_err(Error::from)
    })?;
    Ok(vec![Some(dyn_from_dense(grad))])
}

fn norm_primal_t<T>(input: &StructuredTensor<T>, kind: NormKind) -> Result<DynTensor>
where
    T: NormLinalgDispatchValue + DynTensorTyped + Copy,
{
    let dense_input = input.to_dense()?;
    let output = dispatch_linalg!(T, "norm_dyn_value", LinalgCapabilityOp::Norm, |ctx| {
        norm(ctx, &dense_input, kind).map_err(Error::from)
    })?;
    Ok(dyn_from_dense(output))
}

fn norm_jvp_t<T>(
    input: &StructuredTensor<T>,
    tangent: &Option<DynTensor>,
    kind: NormKind,
) -> Result<Option<DynTensor>>
where
    T: NormLinalgDispatchValue + DynTensorTyped + Scalar + Zero + Copy,
{
    if tangent.is_none() {
        return Ok(None);
    }
    let dense_input = input.to_dense()?;
    let dense_tangent = dense_optional_or_zero(tangent, &dense_input, "norm_jvp tangent")?;
    let (_, output_tangent) = dispatch_linalg!(T, "norm_jvp", LinalgCapabilityOp::Norm, |ctx| {
        norm_frule(ctx, &dense_input, &dense_tangent, kind).map_err(Error::from)
    })?;
    Ok(Some(dyn_from_dense(output_tangent)))
}

fn norm_vjp_t<T>(
    input: &StructuredTensor<T>,
    cotangent: &DynTensor,
    kind: NormKind,
    input_grad_mask: &[bool],
) -> Result<Vec<Option<DynTensor>>>
where
    T: NormLinalgDispatchValue + DynTensorTyped + Copy,
{
    if !input_grad_mask[0] {
        return Ok(vec![None]);
    }
    let dense_input = input.to_dense()?;
    let dense_cotangent = dense_dyn_tensor_typed::<T>(cotangent, "norm_vjp")?;
    let grad = dispatch_linalg!(T, "norm_vjp", LinalgCapabilityOp::Norm, |ctx| {
        norm_rrule(ctx, &dense_input, &dense_cotangent, kind).map_err(Error::from)
    })?;
    Ok(vec![Some(dyn_from_dense(grad))])
}

fn det_primal_t<T>(input: &StructuredTensor<T>) -> Result<DynTensor>
where
    T: ScaledRealLinalgDispatchValue + DynTensorTyped + Copy,
{
    let dense_input = input.to_dense()?;
    let output = dispatch_linalg!(T, "det_dyn_value", LinalgCapabilityOp::Det, |ctx| {
        det(ctx, &dense_input).map_err(Error::from)
    })?;
    Ok(dyn_from_dense(output))
}

fn det_jvp_t<T>(
    input: &StructuredTensor<T>,
    tangent: &Option<DynTensor>,
) -> Result<Option<DynTensor>>
where
    T: ScaledRealLinalgDispatchValue + DynTensorTyped + Scalar + Zero + Copy,
{
    if tangent.is_none() {
        return Ok(None);
    }
    let dense_input = input.to_dense()?;
    let dense_tangent = dense_optional_or_zero(tangent, &dense_input, "det_jvp tangent")?;
    let (_, output_tangent) = dispatch_linalg!(T, "det_jvp", LinalgCapabilityOp::Det, |ctx| {
        det_frule(ctx, &dense_input, &dense_tangent).map_err(Error::from)
    })?;
    Ok(Some(dyn_from_dense(output_tangent)))
}

fn det_vjp_t<T>(
    input: &StructuredTensor<T>,
    cotangent: &DynTensor,
    input_grad_mask: &[bool],
) -> Result<Vec<Option<DynTensor>>>
where
    T: ScaledRealLinalgDispatchValue + DynTensorTyped + Copy,
{
    if !input_grad_mask[0] {
        return Ok(vec![None]);
    }
    let dense_input = input.to_dense()?;
    let dense_cotangent = dense_dyn_tensor_typed::<T>(cotangent, "det_vjp")?;
    let grad = dispatch_linalg!(T, "det_vjp", LinalgCapabilityOp::Det, |ctx| {
        det_rrule(ctx, &dense_input, &dense_cotangent).map_err(Error::from)
    })?;
    Ok(vec![Some(dyn_from_dense(grad))])
}

fn qr_primal_t<T>(input: &StructuredTensor<T>) -> Result<Vec<DynTensor>>
where
    T: LinalgRuntimeValue + DynTensorTyped + Copy,
{
    let dense_input = input.to_dense()?;
    let output = dispatch_linalg!(T, "qr_dyn_value", LinalgCapabilityOp::Qr, |ctx| {
        qr(ctx, &dense_input).map_err(Error::from)
    })?;
    Ok(vec![dyn_from_dense(output.q), dyn_from_dense(output.r)])
}

fn qr_jvp_t<T>(
    input: &StructuredTensor<T>,
    tangent: &Option<DynTensor>,
) -> Result<Vec<Option<DynTensor>>>
where
    T: LinalgRuntimeValue + DynTensorTyped + Scalar + Zero + Copy,
{
    if tangent.is_none() {
        return Ok(vec![None, None]);
    }
    let dense_input = input.to_dense()?;
    let dense_tangent = dense_optional_or_zero(tangent, &dense_input, "qr_jvp tangent")?;
    let (_, tangent_output) = dispatch_linalg!(T, "qr_jvp", LinalgCapabilityOp::Qr, |ctx| {
        qr_frule(ctx, &dense_input, &dense_tangent).map_err(Error::from)
    })?;
    Ok(vec![
        Some(dyn_from_dense(tangent_output.q)),
        Some(dyn_from_dense(tangent_output.r)),
    ])
}

fn qr_vjp_t<T>(
    input: &StructuredTensor<T>,
    output_cotangents: &[Option<DynTensor>],
    input_grad_mask: &[bool],
) -> Result<Vec<Option<DynTensor>>>
where
    T: LinalgRuntimeValue + DynTensorTyped + Copy,
{
    if !input_grad_mask[0] {
        return Ok(vec![None]);
    }
    let dense_input = input.to_dense()?;
    let cotangent = QrCotangent {
        q: optional_dense_dyn_tensor_typed::<T>(&output_cotangents[0], "qr_vjp q")?,
        r: optional_dense_dyn_tensor_typed::<T>(&output_cotangents[1], "qr_vjp r")?,
    };
    if cotangent.q.is_none() && cotangent.r.is_none() {
        return Ok(vec![None]);
    }
    let grad = dispatch_linalg!(T, "qr_vjp", LinalgCapabilityOp::Qr, |ctx| {
        qr_rrule(ctx, &dense_input, &cotangent).map_err(Error::from)
    })?;
    Ok(vec![Some(dyn_from_dense(grad))])
}

fn svd_primal_t<T>(
    input: &StructuredTensor<T>,
    options: Option<&SvdOptions>,
) -> Result<Vec<DynTensor>>
where
    T: LinalgRuntimeValue + DynTensorTyped + Copy,
    T::Real: DynTensorTyped + Copy + tenferro_tensor::KeepCountScalar,
{
    let dense_input = input.to_dense()?;
    let output = dispatch_linalg!(T, "svd_dyn_value", LinalgCapabilityOp::ThinSvd, |ctx| {
        svd(ctx, &dense_input, options).map_err(Error::from)
    })?;
    Ok(vec![
        dyn_from_dense(output.u),
        dyn_from_dense(output.s),
        dyn_from_dense(output.vt),
    ])
}

fn svd_jvp_t<T>(
    input: &StructuredTensor<T>,
    tangent: &Option<DynTensor>,
    options: Option<&SvdOptions>,
) -> Result<Vec<Option<DynTensor>>>
where
    T: LinalgRuntimeValue + DynTensorTyped + Scalar + Zero + Copy,
    T::Real: DynTensorTyped + Copy + num_traits::Float + tenferro_tensor::KeepCountScalar,
{
    if tangent.is_none() {
        return Ok(vec![None, None, None]);
    }
    let dense_input = input.to_dense()?;
    let dense_tangent = dense_optional_or_zero(tangent, &dense_input, "svd_jvp tangent")?;
    let (_, tangent_output) = dispatch_linalg!(T, "svd_jvp", LinalgCapabilityOp::ThinSvd, |ctx| {
        svd_frule(ctx, &dense_input, &dense_tangent, options).map_err(Error::from)
    })?;
    Ok(vec![
        Some(dyn_from_dense(tangent_output.u)),
        Some(dyn_from_dense(tangent_output.s)),
        Some(dyn_from_dense(tangent_output.vt)),
    ])
}

fn svd_vjp_t<T>(
    input: &StructuredTensor<T>,
    output_cotangents: &[Option<DynTensor>],
    input_grad_mask: &[bool],
    options: Option<&SvdOptions>,
) -> Result<Vec<Option<DynTensor>>>
where
    T: LinalgRuntimeValue + DynTensorTyped + Copy,
    T::Real: DynTensorTyped + Copy + num_traits::Float + tenferro_tensor::KeepCountScalar,
{
    if !input_grad_mask[0] {
        return Ok(vec![None]);
    }
    let dense_input = input.to_dense()?;
    let cotangent = SvdCotangent {
        u: optional_dense_dyn_tensor_typed::<T>(&output_cotangents[0], "svd_vjp u")?,
        s: optional_dense_dyn_tensor_typed::<T::Real>(&output_cotangents[1], "svd_vjp s")?,
        vt: optional_dense_dyn_tensor_typed::<T>(&output_cotangents[2], "svd_vjp vt")?,
    };
    if cotangent.u.is_none() && cotangent.s.is_none() && cotangent.vt.is_none() {
        return Ok(vec![None]);
    }
    let grad = dispatch_linalg!(T, "svd_vjp", LinalgCapabilityOp::ThinSvd, |ctx| {
        svd_rrule(ctx, &dense_input, &cotangent, options).map_err(Error::from)
    })?;
    Ok(vec![Some(dyn_from_dense(grad))])
}

impl NormOp {
    pub fn new(kind: NormKind) -> Self {
        Self { kind }
    }
}

impl SolveTriangularOp {
    pub fn new(upper: bool) -> Self {
        Self { upper }
    }
}

impl LuOp {
    pub fn new(pivot: LuPivot) -> Self {
        Self { pivot }
    }
}

impl SvdOp {
    pub fn new(options: Option<SvdOptions>) -> Self {
        Self { options }
    }
}

impl PInvOp {
    pub fn new(rcond: Option<f64>) -> Self {
        Self { rcond }
    }
}

impl LinearizableOp<DynTensor> for SolveOp {
    type Linearized = SolveLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        let output = match (inputs[0], inputs[1]) {
            (DynTensor::F32(a), DynTensor::F32(b)) => solve_primal_t::<f32>(a, b),
            (DynTensor::F64(a), DynTensor::F64(b)) => solve_primal_t::<f64>(a, b),
            (DynTensor::C32(a), DynTensor::C32(b)) => solve_primal_t::<Complex32>(a, b),
            (DynTensor::C64(a), DynTensor::C64(b)) => solve_primal_t::<Complex64>(a, b),
            _ => Err(invalid_argument("solve requires matching dtypes")),
        }
        .map_err(into_ad_error)?;
        Ok(vec![output])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(2))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(SolveLinearized {
            a: inputs[0].clone(),
            b: inputs[1].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for SolveLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        let tangent = match (&self.a, &self.b) {
            (DynTensor::F32(a), DynTensor::F32(b)) => solve_jvp_t::<f32>(a, b, input_tangents),
            (DynTensor::F64(a), DynTensor::F64(b)) => solve_jvp_t::<f64>(a, b, input_tangents),
            (DynTensor::C32(a), DynTensor::C32(b)) => {
                solve_jvp_t::<Complex32>(a, b, input_tangents)
            }
            (DynTensor::C64(a), DynTensor::C64(b)) => {
                solve_jvp_t::<Complex64>(a, b, input_tangents)
            }
            _ => Err(invalid_argument(
                "solve linearization requires matching dtypes",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![tangent])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        let Some(cotangent) = output_cotangents[0].as_ref() else {
            return Ok(vec![None, None]);
        };
        match (&self.a, &self.b) {
            (DynTensor::F32(a), DynTensor::F32(b)) => {
                solve_vjp_t::<f32>(a, b, cotangent, input_grad_mask)
            }
            (DynTensor::F64(a), DynTensor::F64(b)) => {
                solve_vjp_t::<f64>(a, b, cotangent, input_grad_mask)
            }
            (DynTensor::C32(a), DynTensor::C32(b)) => {
                solve_vjp_t::<Complex32>(a, b, cotangent, input_grad_mask)
            }
            (DynTensor::C64(a), DynTensor::C64(b)) => {
                solve_vjp_t::<Complex64>(a, b, cotangent, input_grad_mask)
            }
            _ => Err(invalid_argument(
                "solve linearization requires matching dtypes",
            )),
        }
        .map_err(into_ad_error)
    }
}

impl LinearizableOp<DynTensor> for LstsqOp {
    type Linearized = LstsqLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        match (inputs[0], inputs[1]) {
            (DynTensor::F32(a), DynTensor::F32(b)) => lstsq_primal_t::<f32>(a, b),
            (DynTensor::F64(a), DynTensor::F64(b)) => lstsq_primal_t::<f64>(a, b),
            (DynTensor::C32(_), DynTensor::C32(_)) | (DynTensor::C64(_), DynTensor::C64(_)) => Err(
                invalid_argument("lstsq AD currently supports real dtypes only"),
            ),
            _ => Err(invalid_argument("lstsq requires matching dtypes")),
        }
        .map_err(into_ad_error)
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(2))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(lstsq_output_schema())
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(LstsqLinearized {
            a: inputs[0].clone(),
            b: inputs[1].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for LstsqLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        match (&self.a, &self.b) {
            (DynTensor::F32(a), DynTensor::F32(b)) => lstsq_jvp_t::<f32>(a, b, input_tangents),
            (DynTensor::F64(a), DynTensor::F64(b)) => lstsq_jvp_t::<f64>(a, b, input_tangents),
            (DynTensor::C32(_), DynTensor::C32(_)) | (DynTensor::C64(_), DynTensor::C64(_)) => Err(
                invalid_argument("lstsq AD currently supports real dtypes only"),
            ),
            _ => Err(invalid_argument(
                "lstsq linearization requires matching dtypes",
            )),
        }
        .map_err(into_ad_error)
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        match (&self.a, &self.b) {
            (DynTensor::F32(a), DynTensor::F32(b)) => {
                lstsq_vjp_t::<f32>(a, b, output_cotangents, input_grad_mask)
            }
            (DynTensor::F64(a), DynTensor::F64(b)) => {
                lstsq_vjp_t::<f64>(a, b, output_cotangents, input_grad_mask)
            }
            (DynTensor::C32(_), DynTensor::C32(_)) | (DynTensor::C64(_), DynTensor::C64(_)) => Err(
                invalid_argument("lstsq AD currently supports real dtypes only"),
            ),
            _ => Err(invalid_argument(
                "lstsq linearization requires matching dtypes",
            )),
        }
        .map_err(into_ad_error)
    }
}

impl LinearizableOp<DynTensor> for NormOp {
    type Linearized = NormLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        let output = match inputs[0] {
            DynTensor::F32(input) => norm_primal_t::<f32>(input, self.kind),
            DynTensor::F64(input) => norm_primal_t::<f64>(input, self.kind),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "norm AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![output])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(NormLinearized {
            input: inputs[0].clone(),
            kind: self.kind,
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::CheapReplay
    }
}

impl LinearizableOp<DynTensor> for SolveTriangularOp {
    type Linearized = SolveTriangularLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        let output = match (inputs[0], inputs[1]) {
            (DynTensor::F32(a), DynTensor::F32(b)) => {
                solve_triangular_primal_t::<f32>(a, b, self.upper)
            }
            (DynTensor::F64(a), DynTensor::F64(b)) => {
                solve_triangular_primal_t::<f64>(a, b, self.upper)
            }
            (DynTensor::C32(a), DynTensor::C32(b)) => {
                solve_triangular_primal_t::<Complex32>(a, b, self.upper)
            }
            (DynTensor::C64(a), DynTensor::C64(b)) => {
                solve_triangular_primal_t::<Complex64>(a, b, self.upper)
            }
            _ => Err(invalid_argument(
                "solve_triangular requires matching dtypes",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![output])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(2))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(SolveTriangularLinearized {
            a: inputs[0].clone(),
            b: inputs[1].clone(),
            upper: self.upper,
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for SolveTriangularLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        let tangent = match (&self.a, &self.b) {
            (DynTensor::F32(a), DynTensor::F32(b)) => {
                solve_triangular_jvp_t::<f32>(a, b, input_tangents, self.upper)
            }
            (DynTensor::F64(a), DynTensor::F64(b)) => {
                solve_triangular_jvp_t::<f64>(a, b, input_tangents, self.upper)
            }
            (DynTensor::C32(a), DynTensor::C32(b)) => {
                solve_triangular_jvp_t::<Complex32>(a, b, input_tangents, self.upper)
            }
            (DynTensor::C64(a), DynTensor::C64(b)) => {
                solve_triangular_jvp_t::<Complex64>(a, b, input_tangents, self.upper)
            }
            _ => Err(invalid_argument(
                "solve_triangular linearization requires matching dtypes",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![tangent])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        let Some(cotangent) = output_cotangents[0].as_ref() else {
            return Ok(vec![None, None]);
        };
        match (&self.a, &self.b) {
            (DynTensor::F32(a), DynTensor::F32(b)) => {
                solve_triangular_vjp_t::<f32>(a, b, cotangent, input_grad_mask, self.upper)
            }
            (DynTensor::F64(a), DynTensor::F64(b)) => {
                solve_triangular_vjp_t::<f64>(a, b, cotangent, input_grad_mask, self.upper)
            }
            (DynTensor::C32(a), DynTensor::C32(b)) => {
                solve_triangular_vjp_t::<Complex32>(a, b, cotangent, input_grad_mask, self.upper)
            }
            (DynTensor::C64(a), DynTensor::C64(b)) => {
                solve_triangular_vjp_t::<Complex64>(a, b, cotangent, input_grad_mask, self.upper)
            }
            _ => Err(invalid_argument(
                "solve_triangular linearization requires matching dtypes",
            )),
        }
        .map_err(into_ad_error)
    }
}

impl LinearizedOp<DynTensor> for NormLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        let tangent = match &self.input {
            DynTensor::F32(input) => norm_jvp_t::<f32>(input, &input_tangents[0], self.kind),
            DynTensor::F64(input) => norm_jvp_t::<f64>(input, &input_tangents[0], self.kind),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "norm AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![tangent])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        let Some(cotangent) = output_cotangents[0].as_ref() else {
            return Ok(vec![None]);
        };
        match &self.input {
            DynTensor::F32(input) => {
                norm_vjp_t::<f32>(input, cotangent, self.kind, input_grad_mask)
            }
            DynTensor::F64(input) => {
                norm_vjp_t::<f64>(input, cotangent, self.kind, input_grad_mask)
            }
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "norm AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)
    }
}

impl LinearizableOp<DynTensor> for InvOp {
    type Linearized = InvLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        let output = match inputs[0] {
            DynTensor::F32(input) => inv_primal_t::<f32>(input),
            DynTensor::F64(input) => inv_primal_t::<f64>(input),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "inv AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![output])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(InvLinearized {
            input: inputs[0].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for InvLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        let tangent = match &self.input {
            DynTensor::F32(input) => inv_jvp_t::<f32>(input, &input_tangents[0]),
            DynTensor::F64(input) => inv_jvp_t::<f64>(input, &input_tangents[0]),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "inv AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![tangent])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        let Some(cotangent) = output_cotangents[0].as_ref() else {
            return Ok(vec![None]);
        };
        match &self.input {
            DynTensor::F32(input) => inv_vjp_t::<f32>(input, cotangent, input_grad_mask),
            DynTensor::F64(input) => inv_vjp_t::<f64>(input, cotangent, input_grad_mask),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "inv AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)
    }
}

impl LinearizableOp<DynTensor> for SlogdetOp {
    type Linearized = SlogdetLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        match inputs[0] {
            DynTensor::F32(input) => slogdet_primal_t::<f32>(input),
            DynTensor::F64(input) => slogdet_primal_t::<f64>(input),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "slogdet AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(slogdet_output_schema())
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(SlogdetLinearized {
            input: inputs[0].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for SlogdetLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        match &self.input {
            DynTensor::F32(input) => slogdet_jvp_t::<f32>(input, &input_tangents[0]),
            DynTensor::F64(input) => slogdet_jvp_t::<f64>(input, &input_tangents[0]),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "slogdet AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        match &self.input {
            DynTensor::F32(input) => {
                slogdet_vjp_t::<f32>(input, output_cotangents, input_grad_mask)
            }
            DynTensor::F64(input) => {
                slogdet_vjp_t::<f64>(input, output_cotangents, input_grad_mask)
            }
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "slogdet AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)
    }
}

impl LinearizableOp<DynTensor> for CholeskyOp {
    type Linearized = CholeskyLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        let output = match inputs[0] {
            DynTensor::F32(input) => cholesky_primal_t::<f32>(input),
            DynTensor::F64(input) => cholesky_primal_t::<f64>(input),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "cholesky AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![output])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(CholeskyLinearized {
            input: inputs[0].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizableOp<DynTensor> for LuOp {
    type Linearized = LuLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        match inputs[0] {
            DynTensor::F32(input) => lu_primal_t::<f32>(input, self.pivot),
            DynTensor::F64(input) => lu_primal_t::<f64>(input, self.pivot),
            DynTensor::C32(input) => lu_primal_t::<Complex32>(input, self.pivot),
            DynTensor::C64(input) => lu_primal_t::<Complex64>(input, self.pivot),
        }
        .map_err(into_ad_error)
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(lu_output_schema())
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(LuLinearized {
            input: inputs[0].clone(),
            pivot: self.pivot,
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for LuLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        match &self.input {
            DynTensor::F32(input) => lu_jvp_t::<f32>(input, &input_tangents[0], self.pivot),
            DynTensor::F64(input) => lu_jvp_t::<f64>(input, &input_tangents[0], self.pivot),
            DynTensor::C32(input) => lu_jvp_t::<Complex32>(input, &input_tangents[0], self.pivot),
            DynTensor::C64(input) => lu_jvp_t::<Complex64>(input, &input_tangents[0], self.pivot),
        }
        .map_err(into_ad_error)
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        match &self.input {
            DynTensor::F32(input) => {
                lu_vjp_t::<f32>(input, output_cotangents, input_grad_mask, self.pivot)
            }
            DynTensor::F64(input) => {
                lu_vjp_t::<f64>(input, output_cotangents, input_grad_mask, self.pivot)
            }
            DynTensor::C32(input) => {
                lu_vjp_t::<Complex32>(input, output_cotangents, input_grad_mask, self.pivot)
            }
            DynTensor::C64(input) => {
                lu_vjp_t::<Complex64>(input, output_cotangents, input_grad_mask, self.pivot)
            }
        }
        .map_err(into_ad_error)
    }
}

impl LinearizedOp<DynTensor> for CholeskyLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        let tangent = match &self.input {
            DynTensor::F32(input) => cholesky_jvp_t::<f32>(input, &input_tangents[0]),
            DynTensor::F64(input) => cholesky_jvp_t::<f64>(input, &input_tangents[0]),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "cholesky AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![tangent])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        let Some(cotangent) = output_cotangents[0].as_ref() else {
            return Ok(vec![None]);
        };
        match &self.input {
            DynTensor::F32(input) => cholesky_vjp_t::<f32>(input, cotangent, input_grad_mask),
            DynTensor::F64(input) => cholesky_vjp_t::<f64>(input, cotangent, input_grad_mask),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "cholesky AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)
    }
}

impl LinearizableOp<DynTensor> for EigOp {
    type Linearized = EigLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        match inputs[0] {
            DynTensor::F32(input) => eig_primal_t::<f32>(input),
            DynTensor::F64(input) => eig_primal_t::<f64>(input),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "eig AD currently supports real inputs only",
            )),
        }
        .map_err(into_ad_error)
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(2))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(EigLinearized {
            input: inputs[0].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for EigLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        match &self.input {
            DynTensor::F32(input) => eig_jvp_t::<f32>(input, &input_tangents[0]),
            DynTensor::F64(input) => eig_jvp_t::<f64>(input, &input_tangents[0]),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "eig AD currently supports real inputs only",
            )),
        }
        .map_err(into_ad_error)
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        match &self.input {
            DynTensor::F32(input) => eig_vjp_t::<f32>(input, output_cotangents, input_grad_mask),
            DynTensor::F64(input) => eig_vjp_t::<f64>(input, output_cotangents, input_grad_mask),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "eig AD currently supports real inputs only",
            )),
        }
        .map_err(into_ad_error)
    }
}

impl LinearizableOp<DynTensor> for EigenOp {
    type Linearized = EigenLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        match inputs[0] {
            DynTensor::F32(input) => eigen_primal_t::<f32>(input),
            DynTensor::F64(input) => eigen_primal_t::<f64>(input),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "eigen AD currently supports real inputs only",
            )),
        }
        .map_err(into_ad_error)
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(2))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(EigenLinearized {
            input: inputs[0].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for EigenLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        match &self.input {
            DynTensor::F32(input) => eigen_jvp_t::<f32>(input, &input_tangents[0]),
            DynTensor::F64(input) => eigen_jvp_t::<f64>(input, &input_tangents[0]),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "eigen AD currently supports real inputs only",
            )),
        }
        .map_err(into_ad_error)
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        match &self.input {
            DynTensor::F32(input) => eigen_vjp_t::<f32>(input, output_cotangents, input_grad_mask),
            DynTensor::F64(input) => eigen_vjp_t::<f64>(input, output_cotangents, input_grad_mask),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "eigen AD currently supports real inputs only",
            )),
        }
        .map_err(into_ad_error)
    }
}

impl LinearizableOp<DynTensor> for DetOp {
    type Linearized = DetLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        let output = match inputs[0] {
            DynTensor::F32(input) => det_primal_t::<f32>(input),
            DynTensor::F64(input) => det_primal_t::<f64>(input),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "det AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![output])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(DetLinearized {
            input: inputs[0].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for DetLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        let tangent = match &self.input {
            DynTensor::F32(input) => det_jvp_t::<f32>(input, &input_tangents[0]),
            DynTensor::F64(input) => det_jvp_t::<f64>(input, &input_tangents[0]),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "det AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![tangent])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        let Some(cotangent) = output_cotangents[0].as_ref() else {
            return Ok(vec![None]);
        };
        match &self.input {
            DynTensor::F32(input) => det_vjp_t::<f32>(input, cotangent, input_grad_mask),
            DynTensor::F64(input) => det_vjp_t::<f64>(input, cotangent, input_grad_mask),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "det AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)
    }
}

impl LinearizableOp<DynTensor> for PInvOp {
    type Linearized = PInvLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        let output = match inputs[0] {
            DynTensor::F32(input) => pinv_primal_t::<f32>(input, self.rcond),
            DynTensor::F64(input) => pinv_primal_t::<f64>(input, self.rcond),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "pinv AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![output])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(PInvLinearized {
            input: inputs[0].clone(),
            rcond: self.rcond,
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for PInvLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        let tangent = match &self.input {
            DynTensor::F32(input) => pinv_jvp_t::<f32>(input, &input_tangents[0], self.rcond),
            DynTensor::F64(input) => pinv_jvp_t::<f64>(input, &input_tangents[0], self.rcond),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "pinv AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![tangent])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        let Some(cotangent) = output_cotangents[0].as_ref() else {
            return Ok(vec![None]);
        };
        match &self.input {
            DynTensor::F32(input) => {
                pinv_vjp_t::<f32>(input, cotangent, input_grad_mask, self.rcond)
            }
            DynTensor::F64(input) => {
                pinv_vjp_t::<f64>(input, cotangent, input_grad_mask, self.rcond)
            }
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "pinv AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)
    }
}

impl LinearizableOp<DynTensor> for MatrixExpOp {
    type Linearized = MatrixExpLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        let output = match inputs[0] {
            DynTensor::F32(input) => matrix_exp_primal_t::<f32>(input),
            DynTensor::F64(input) => matrix_exp_primal_t::<f64>(input),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "matrix_exp AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![output])
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(MatrixExpLinearized {
            input: inputs[0].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for MatrixExpLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        let tangent = match &self.input {
            DynTensor::F32(input) => matrix_exp_jvp_t::<f32>(input, &input_tangents[0]),
            DynTensor::F64(input) => matrix_exp_jvp_t::<f64>(input, &input_tangents[0]),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "matrix_exp AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)?;
        Ok(vec![tangent])
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        let Some(cotangent) = output_cotangents[0].as_ref() else {
            return Ok(vec![None]);
        };
        match &self.input {
            DynTensor::F32(input) => matrix_exp_vjp_t::<f32>(input, cotangent, input_grad_mask),
            DynTensor::F64(input) => matrix_exp_vjp_t::<f64>(input, cotangent, input_grad_mask),
            DynTensor::C32(_) | DynTensor::C64(_) => Err(invalid_argument(
                "matrix_exp AD currently supports real dtypes only",
            )),
        }
        .map_err(into_ad_error)
    }
}

impl LinearizableOp<DynTensor> for QrOp {
    type Linearized = QrLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        match inputs[0] {
            DynTensor::F32(input) => qr_primal_t::<f32>(input),
            DynTensor::F64(input) => qr_primal_t::<f64>(input),
            DynTensor::C32(input) => qr_primal_t::<Complex32>(input),
            DynTensor::C64(input) => qr_primal_t::<Complex64>(input),
        }
        .map_err(into_ad_error)
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(2))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(QrLinearized {
            input: inputs[0].clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for QrLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        match &self.input {
            DynTensor::F32(input) => qr_jvp_t::<f32>(input, &input_tangents[0]),
            DynTensor::F64(input) => qr_jvp_t::<f64>(input, &input_tangents[0]),
            DynTensor::C32(input) => qr_jvp_t::<Complex32>(input, &input_tangents[0]),
            DynTensor::C64(input) => qr_jvp_t::<Complex64>(input, &input_tangents[0]),
        }
        .map_err(into_ad_error)
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        match &self.input {
            DynTensor::F32(input) => qr_vjp_t::<f32>(input, output_cotangents, input_grad_mask),
            DynTensor::F64(input) => qr_vjp_t::<f64>(input, output_cotangents, input_grad_mask),
            DynTensor::C32(input) => {
                qr_vjp_t::<Complex32>(input, output_cotangents, input_grad_mask)
            }
            DynTensor::C64(input) => {
                qr_vjp_t::<Complex64>(input, output_cotangents, input_grad_mask)
            }
        }
        .map_err(into_ad_error)
    }
}

impl LinearizableOp<DynTensor> for SvdOp {
    type Linearized = SvdLinearized;

    fn primal(&self, inputs: &[&DynTensor]) -> AdResult<Vec<DynTensor>> {
        match inputs[0] {
            DynTensor::F32(input) => svd_primal_t::<f32>(input, self.options.as_ref()),
            DynTensor::F64(input) => svd_primal_t::<f64>(input, self.options.as_ref()),
            DynTensor::C32(input) => svd_primal_t::<Complex32>(input, self.options.as_ref()),
            DynTensor::C64(input) => svd_primal_t::<Complex64>(input, self.options.as_ref()),
        }
        .map_err(into_ad_error)
    }

    fn input_schema(&self, _inputs: &[&DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(1))
    }

    fn output_schema(&self, _inputs: &[&DynTensor], _outputs: &[DynTensor]) -> AdResult<Schema> {
        Ok(differentiable_schema(3))
    }

    fn linearize(
        &self,
        inputs: &[&DynTensor],
        _outputs: &[DynTensor],
    ) -> AdResult<Self::Linearized> {
        Ok(SvdLinearized {
            input: inputs[0].clone(),
            options: self.options.clone(),
        })
    }

    fn checkpoint_hint(&self) -> CheckpointHint {
        CheckpointHint::ExpensiveReplay
    }
}

impl LinearizedOp<DynTensor> for SvdLinearized {
    fn jvp(&self, input_tangents: &[Option<DynTensor>]) -> AdResult<Vec<Option<DynTensor>>> {
        match &self.input {
            DynTensor::F32(input) => {
                svd_jvp_t::<f32>(input, &input_tangents[0], self.options.as_ref())
            }
            DynTensor::F64(input) => {
                svd_jvp_t::<f64>(input, &input_tangents[0], self.options.as_ref())
            }
            DynTensor::C32(input) => {
                svd_jvp_t::<Complex32>(input, &input_tangents[0], self.options.as_ref())
            }
            DynTensor::C64(input) => {
                svd_jvp_t::<Complex64>(input, &input_tangents[0], self.options.as_ref())
            }
        }
        .map_err(into_ad_error)
    }

    fn vjp(
        &self,
        output_cotangents: &[Option<DynTensor>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<DynTensor>>> {
        match &self.input {
            DynTensor::F32(input) => svd_vjp_t::<f32>(
                input,
                output_cotangents,
                input_grad_mask,
                self.options.as_ref(),
            ),
            DynTensor::F64(input) => svd_vjp_t::<f64>(
                input,
                output_cotangents,
                input_grad_mask,
                self.options.as_ref(),
            ),
            DynTensor::C32(input) => svd_vjp_t::<Complex32>(
                input,
                output_cotangents,
                input_grad_mask,
                self.options.as_ref(),
            ),
            DynTensor::C64(input) => svd_vjp_t::<Complex64>(
                input,
                output_cotangents,
                input_grad_mask,
                self.options.as_ref(),
            ),
        }
        .map_err(into_ad_error)
    }
}

pub fn solve_dyn_values(a: &DynValue, b: &DynValue) -> AdResult<DynValue> {
    SolveOp.apply_one(&[a, b])
}

pub fn lstsq_dyn_values(a: &DynValue, b: &DynValue) -> AdResult<DynLstsqValues> {
    let mut outputs = LstsqOp.apply(&[a, b])?;
    if outputs.len() != 2 {
        return Err(AutodiffError::InvalidArgument(format!(
            "LstsqOp expected 2 outputs, got {}",
            outputs.len()
        )));
    }
    let residual = outputs.pop().unwrap();
    let x = outputs.pop().unwrap();
    Ok(DynLstsqValues { x, residual })
}

pub fn solve_triangular_dyn_value(a: &DynValue, b: &DynValue, upper: bool) -> AdResult<DynValue> {
    SolveTriangularOp::new(upper).apply_one(&[a, b])
}

pub fn norm_dyn_value(input: &DynValue, kind: NormKind) -> AdResult<DynValue> {
    NormOp::new(kind).apply_one(&[input])
}

pub fn det_dyn_value(input: &DynValue) -> AdResult<DynValue> {
    DetOp.apply_one(&[input])
}

pub fn inv_dyn_value(input: &DynValue) -> AdResult<DynValue> {
    InvOp.apply_one(&[input])
}

pub fn slogdet_dyn_value(input: &DynValue) -> AdResult<DynSlogdetValues> {
    let mut outputs = SlogdetOp.apply(&[input])?;
    if outputs.len() != 2 {
        return Err(AutodiffError::InvalidArgument(format!(
            "SlogdetOp expected 2 outputs, got {}",
            outputs.len()
        )));
    }
    let logabsdet = outputs.pop().unwrap();
    let sign = outputs.pop().unwrap();
    Ok(DynSlogdetValues { sign, logabsdet })
}

pub fn cholesky_dyn_value(input: &DynValue) -> AdResult<DynValue> {
    CholeskyOp.apply_one(&[input])
}

pub fn lu_dyn_value(input: &DynValue, pivot: LuPivot) -> AdResult<DynLuValues> {
    let mut outputs = LuOp::new(pivot).apply(&[input])?;
    if outputs.len() != 3 {
        return Err(AutodiffError::InvalidArgument(format!(
            "LuOp expected 3 outputs, got {}",
            outputs.len()
        )));
    }
    let u = outputs.pop().unwrap();
    let l = outputs.pop().unwrap();
    let p = outputs.pop().unwrap();
    Ok(DynLuValues { p, l, u })
}

pub fn qr_dyn_value(input: &DynValue) -> AdResult<DynQrValues> {
    let mut outputs = QrOp.apply(&[input])?;
    if outputs.len() != 2 {
        return Err(AutodiffError::InvalidArgument(format!(
            "QrOp expected 2 outputs, got {}",
            outputs.len()
        )));
    }
    let r = outputs.pop().unwrap();
    let q = outputs.pop().unwrap();
    Ok(DynQrValues { q, r })
}

pub fn svd_dyn_value(input: &DynValue, options: Option<SvdOptions>) -> AdResult<DynSvdValues> {
    let mut outputs = SvdOp::new(options).apply(&[input])?;
    if outputs.len() != 3 {
        return Err(AutodiffError::InvalidArgument(format!(
            "SvdOp expected 3 outputs, got {}",
            outputs.len()
        )));
    }
    let vt = outputs.pop().unwrap();
    let s = outputs.pop().unwrap();
    let u = outputs.pop().unwrap();
    Ok(DynSvdValues { u, s, vt })
}

pub fn eig_dyn_value(input: &DynValue) -> AdResult<DynEigValues> {
    let mut outputs = EigOp.apply(&[input])?;
    if outputs.len() != 2 {
        return Err(AutodiffError::InvalidArgument(format!(
            "EigOp expected 2 outputs, got {}",
            outputs.len()
        )));
    }
    let vectors = outputs.pop().unwrap();
    let values = outputs.pop().unwrap();
    Ok(DynEigValues { values, vectors })
}

pub fn eigen_dyn_value(input: &DynValue) -> AdResult<DynEigenValues> {
    let mut outputs = EigenOp.apply(&[input])?;
    if outputs.len() != 2 {
        return Err(AutodiffError::InvalidArgument(format!(
            "EigenOp expected 2 outputs, got {}",
            outputs.len()
        )));
    }
    let vectors = outputs.pop().unwrap();
    let values = outputs.pop().unwrap();
    Ok(DynEigenValues { values, vectors })
}

pub fn pinv_dyn_value(input: &DynValue, rcond: Option<f64>) -> AdResult<DynValue> {
    PInvOp::new(rcond).apply_one(&[input])
}

pub fn matrix_exp_dyn_value(input: &DynValue) -> AdResult<DynValue> {
    MatrixExpOp.apply_one(&[input])
}
