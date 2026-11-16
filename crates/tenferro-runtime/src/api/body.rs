//! The dtype-erased bodies of the core grammar.
//!
//! Every body reads its operands through [`TensorRead`] and runs the same
//! engine `*_read` entries the former typed surface ran, so owned, borrowed,
//! strided and typed inputs reach one implementation. Axes arrive normalized.

use num_complex::Complex64;
use tenferro_ops::broadcast::{broadcast_error_to_validation, broadcast_shape, broadcast_shapes};
use tenferro_tensor::validate::matmul_config_for_shapes;
use tenferro_tensor::{
    ActivationOp, CompareDir, DType, DotGeneralConfig, Engine, Error, GatherConfig, PadConfig,
    Result, ScatterConfig, SliceConfig, Tensor, TensorRead, TensorWrite,
};

use crate::composite;
use crate::composite::session::run_session_composite;
use crate::typed_tensor::{broadcast_to_in_read, ReadInput};

type R<'a> = TensorRead<'a>;

fn input(a: R<'_>) -> ReadInput<'_> {
    ReadInput::Borrowed(a)
}

fn all_axes(rank: usize, axes: Option<&[usize]>) -> std::borrow::Cow<'_, [usize]> {
    crate::tensor::all_axes_if_none(rank, axes)
}

fn broadcast_error(err: tenferro_ops::broadcast::BroadcastError) -> Error {
    Error::validation("broadcast", broadcast_error_to_validation(err))
}

fn binary_in<'a>(
    lhs: R<'a>,
    rhs: R<'a>,
    e: &mut dyn Engine,
) -> Result<(ReadInput<'a>, ReadInput<'a>)> {
    let shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(broadcast_error)?;
    Ok((
        broadcast_to_in_read(lhs, &shape, e)?,
        broadcast_to_in_read(rhs, &shape, e)?,
    ))
}

fn ternary_in<'a>(
    first: R<'a>,
    second: R<'a>,
    third: R<'a>,
    e: &mut dyn Engine,
) -> Result<(ReadInput<'a>, ReadInput<'a>, ReadInput<'a>)> {
    let shape = broadcast_shapes([first.shape(), second.shape(), third.shape()])
        .map_err(broadcast_error)?;
    Ok((
        broadcast_to_in_read(first, &shape, e)?,
        broadcast_to_in_read(second, &shape, e)?,
        broadcast_to_in_read(third, &shape, e)?,
    ))
}

macro_rules! unary {
    ($($name:ident => $read:ident;)*) => {$(
        pub(crate) fn $name(a: R<'_>, e: &mut dyn Engine) -> Result<Tensor> {
            e.$read(a)
        }
    )*};
}

unary! {
    exp => exp_read;
    neg => neg_read;
    abs => abs_read;
    sign => sign_read;
    conj => conj_read;
    log => log_read;
    expm1 => expm1_read;
    log1p => log1p_read;
    erf => erf_read;
    sin => sin_read;
    cos => cos_read;
    tanh => tanh_read;
    sqrt => sqrt_read;
    rsqrt => rsqrt_read;
}

macro_rules! binary {
    ($($name:ident => $read:ident;)*) => {$(
        pub(crate) fn $name(a: R<'_>, b: R<'_>, e: &mut dyn Engine) -> Result<Tensor> {
            let (lhs, rhs) = binary_in(a, b, e)?;
            e.$read(lhs.tensor_read(), rhs.tensor_read())
        }
    )*};
}

binary! {
    add => add_read;
    mul => mul_read;
    sub => sub_read;
    div => div_read;
    rem => rem_read;
    pow => pow_read;
    maximum => maximum_read;
    minimum => minimum_read;
}

pub(crate) fn add_into(a: R<'_>, b: R<'_>, e: &mut dyn Engine, out: TensorWrite<'_>) -> Result<()> {
    let (lhs, rhs) = binary_in(a, b, e)?;
    e.add_read_into(lhs.tensor_read(), rhs.tensor_read(), out)
}

pub(crate) fn compare(a: R<'_>, b: R<'_>, dir: CompareDir, e: &mut dyn Engine) -> Result<Tensor> {
    let (lhs, rhs) = binary_in(a, b, e)?;
    e.compare_read(lhs.tensor_read(), rhs.tensor_read(), &dir)
}

pub(crate) fn clamp(a: R<'_>, lower: R<'_>, upper: R<'_>, e: &mut dyn Engine) -> Result<Tensor> {
    let (input, lower, upper) = ternary_in(a, lower, upper, e)?;
    e.clamp_read(
        input.tensor_read(),
        lower.tensor_read(),
        upper.tensor_read(),
    )
}

pub(crate) fn where_select(
    condition: R<'_>,
    on_true: R<'_>,
    on_false: R<'_>,
    e: &mut dyn Engine,
) -> Result<Tensor> {
    let (condition, on_true, on_false) = ternary_in(condition, on_true, on_false, e)?;
    e.select_read(
        condition.tensor_read(),
        on_true.tensor_read(),
        on_false.tensor_read(),
    )
}

pub(crate) fn matmul(a: R<'_>, b: R<'_>, e: &mut dyn Engine) -> Result<Tensor> {
    let config = matmul_config_for_shapes("matmul", a.shape(), b.shape())?;
    e.dot_general_read(a, b, &config)
}

pub(crate) fn dot_general(
    a: R<'_>,
    b: R<'_>,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
    e: &mut dyn Engine,
) -> Result<Tensor> {
    if lhs_conj || rhs_conj {
        e.dot_general_with_conj_read(a, b, config, lhs_conj, rhs_conj)
    } else {
        e.dot_general_read(a, b, config)
    }
}

pub(crate) fn reshape(a: R<'_>, shape: &[usize], e: &mut dyn Engine) -> Result<Tensor> {
    e.reshape_read(a, shape)
}

pub(crate) fn transpose(a: R<'_>, perm: &[usize], e: &mut dyn Engine) -> Result<Tensor> {
    e.transpose_read(a, perm)
}

pub(crate) fn broadcast_in_dim(
    a: R<'_>,
    shape: &[usize],
    dims: &[usize],
    e: &mut dyn Engine,
) -> Result<Tensor> {
    e.broadcast_in_dim_read(a, shape, dims)
}

macro_rules! reduction {
    ($($name:ident => $read:ident;)*) => {$(
        pub(crate) fn $name(a: R<'_>, axes: Option<&[usize]>, e: &mut dyn Engine) -> Result<Tensor> {
            let axes = all_axes(a.shape().len(), axes);
            e.$read(a, &axes)
        }
    )*};
}

reduction! {
    reduce_sum => reduce_sum_read;
    reduce_max => reduce_max_read;
    reduce_min => reduce_min_read;
    reduce_prod => reduce_prod_read;
    reduce_sum_squares => reduce_sum_squares_read;
}

pub(crate) fn reduce_mean(a: R<'_>, axes: Option<&[usize]>, e: &mut dyn Engine) -> Result<Tensor> {
    run_session_composite(e, |ops| composite::reduce_mean(ops, &input(a), axes))
}

pub(crate) fn convert(a: R<'_>, to: DType, e: &mut dyn Engine) -> Result<Tensor> {
    match a.as_tensor() {
        Some(t) => e.convert(t, to),
        None => e.convert_read(a, to),
    }
}

pub(crate) fn cast(a: R<'_>, to: DType, e: &mut dyn Engine) -> Result<Tensor> {
    match a.as_tensor() {
        Some(t) => e.cast(t, to),
        None => e.cast_read(a, to),
    }
}

pub(crate) fn scale_real(a: R<'_>, factor: f64, e: &mut dyn Engine) -> Result<Tensor> {
    let scalar = crate::scale::real_scale_scalar(a.dtype(), factor)?;
    let scalar = e.upload_host_tensor(TensorRead::from_tensor(&scalar))?;
    mul(a, TensorRead::from_tensor(&scalar), e)
}

pub(crate) fn scale_complex(a: R<'_>, factor: Complex64, e: &mut dyn Engine) -> Result<Tensor> {
    let scalar = crate::scale::complex_scale_scalar(a.dtype(), factor)?;
    let scalar = e.upload_host_tensor(TensorRead::from_tensor(&scalar))?;
    mul(a, TensorRead::from_tensor(&scalar), e)
}

macro_rules! activation {
    ($($name:ident => $op:ident, $composite:ident;)*) => {$(
        pub(crate) fn $name(a: R<'_>, e: &mut dyn Engine) -> Result<Tensor> {
            if let Some(out) = e.fused_activation_read(ActivationOp::$op, a.clone())? {
                return Ok(out);
            }
            run_session_composite(e, |ops| composite::$composite(ops, &input(a)))
        }
    )*};
}

activation! {
    sigmoid => Sigmoid, sigmoid;
    silu => Silu, silu;
    softplus => Softplus, softplus;
    gelu => Gelu, gelu;
    gelu_tanh => GeluTanh, gelu_tanh;
}

pub(crate) fn softmax(a: R<'_>, axis: usize, e: &mut dyn Engine) -> Result<Tensor> {
    if let Some(out) = e.softmax_read(a.clone(), axis)? {
        return Ok(out);
    }
    run_session_composite(e, |ops| composite::softmax(ops, &input(a), axis))
}

pub(crate) fn log_softmax(a: R<'_>, axis: usize, e: &mut dyn Engine) -> Result<Tensor> {
    run_session_composite(e, |ops| composite::log_softmax(ops, &input(a), axis))
}

pub(crate) fn masked_softmax(
    a: R<'_>,
    mask: R<'_>,
    axis: usize,
    e: &mut dyn Engine,
) -> Result<Tensor> {
    let mask = input(mask);
    run_session_composite(e, |ops| {
        composite::masked_softmax(ops, &input(a), &mask, axis)
    })
}

pub(crate) fn masked_log_softmax(
    a: R<'_>,
    mask: R<'_>,
    axis: usize,
    e: &mut dyn Engine,
) -> Result<Tensor> {
    let mask = input(mask);
    run_session_composite(e, |ops| {
        composite::masked_log_softmax(ops, &input(a), &mask, axis)
    })
}

pub(crate) fn layer_norm(
    a: R<'_>,
    axis: usize,
    weight: Option<R<'_>>,
    bias: Option<R<'_>>,
    eps: f64,
    e: &mut dyn Engine,
) -> Result<Tensor> {
    let weight = weight.map(input);
    let bias = bias.map(input);
    run_session_composite(e, |ops| {
        composite::layer_norm(ops, &input(a), axis, weight.as_ref(), bias.as_ref(), eps)
    })
}

pub(crate) fn rms_norm(
    a: R<'_>,
    axis: usize,
    weight: Option<R<'_>>,
    bias: Option<R<'_>>,
    eps: f64,
    e: &mut dyn Engine,
) -> Result<Tensor> {
    let weight = weight.map(input);
    let bias = bias.map(input);
    run_session_composite(e, |ops| {
        composite::rms_norm(ops, &input(a), axis, weight.as_ref(), bias.as_ref(), eps)
    })
}

pub(crate) fn take_along_axis(
    a: R<'_>,
    indices: R<'_>,
    axis: usize,
    e: &mut dyn Engine,
) -> Result<Tensor> {
    let indices = input(indices);
    run_session_composite(e, |ops| {
        composite::take_along_axis(ops, &input(a), &indices, axis)
    })
}

pub(crate) fn gather(
    a: R<'_>,
    indices: R<'_>,
    config: &GatherConfig,
    e: &mut dyn Engine,
) -> Result<Tensor> {
    e.gather_read(a, indices, config)
}

pub(crate) fn scatter(
    a: R<'_>,
    indices: R<'_>,
    updates: R<'_>,
    config: &ScatterConfig,
    e: &mut dyn Engine,
) -> Result<Tensor> {
    e.scatter_read(a, indices, updates, config)
}

pub(crate) fn slice(a: R<'_>, config: &SliceConfig, e: &mut dyn Engine) -> Result<Tensor> {
    e.slice_read(a, config)
}

pub(crate) fn dynamic_slice(
    a: R<'_>,
    starts: R<'_>,
    sizes: &[usize],
    e: &mut dyn Engine,
) -> Result<Tensor> {
    e.dynamic_slice_read(a, starts, sizes)
}

pub(crate) fn dynamic_update_slice(
    a: R<'_>,
    update: R<'_>,
    starts: R<'_>,
    e: &mut dyn Engine,
) -> Result<Tensor> {
    e.dynamic_update_slice_read(a, update, starts)
}

pub(crate) fn pad(a: R<'_>, config: &PadConfig, e: &mut dyn Engine) -> Result<Tensor> {
    e.pad_read(a, config)
}

pub(crate) fn concatenate(inputs: &[R<'_>], axis: usize, e: &mut dyn Engine) -> Result<Tensor> {
    e.concatenate_read(inputs, axis)
}

pub(crate) fn reverse(a: R<'_>, axes: &[usize], e: &mut dyn Engine) -> Result<Tensor> {
    e.reverse_read(a, axes)
}

pub(crate) fn tril(a: R<'_>, k: i64, e: &mut dyn Engine) -> Result<Tensor> {
    e.tril_read(a, k)
}

pub(crate) fn triu(a: R<'_>, k: i64, e: &mut dyn Engine) -> Result<Tensor> {
    e.triu_read(a, k)
}

pub(crate) fn extract_diag(
    a: R<'_>,
    axis_a: usize,
    axis_b: usize,
    e: &mut dyn Engine,
) -> Result<Tensor> {
    e.extract_diagonal_read(a, axis_a, axis_b)
}

pub(crate) fn embed_diag(
    a: R<'_>,
    axis_a: usize,
    axis_b: usize,
    e: &mut dyn Engine,
) -> Result<Tensor> {
    e.embed_diagonal_read(a, axis_a, axis_b)
}
