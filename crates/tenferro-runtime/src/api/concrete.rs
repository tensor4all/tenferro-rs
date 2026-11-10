//! Concrete forward bodies of the core registry (`CoreOps`).
//!
//! Each entry reads its operands through `AsView::as_read`, normalizes signed
//! axes once, runs the erased body of `crate::api::body` and re-wraps the result
//! through `CoreRepr`'s checked downcasts for typed inputs.

#![allow(clippy::too_many_arguments)]

use num_complex::Complex64;
use tenferro_tensor::{
    AsView, CompareDir, DType, Dest, DestMode, DotGeneralConfig, DotGeneralOptions, Engine, Error,
    GatherConfig, PadConfig, Repr, ScatterConfig, SliceConfig, Tensor,
};

use super::{axes, body};
use crate::{AddOptions, CoreRepr, SelectOperand};

type Res<T> = tenferro_tensor::Result<T>;
type Val<X> = <<X as AsView>::Repr as Repr>::Value;
type Re<X> = <<X as AsView>::Repr as CoreRepr>::Real;
type Bo<X> = <<X as AsView>::Repr as CoreRepr>::Bool;

fn val<X: AsView + ?Sized>(op: &'static str, t: Tensor) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    <X::Repr as CoreRepr>::wrap_value_of(op, t)
}

fn re<X: AsView + ?Sized>(op: &'static str, t: Tensor) -> Res<Re<X>>
where
    X::Repr: CoreRepr,
{
    <X::Repr as CoreRepr>::wrap_real(op, t)
}

fn boolean<X: AsView + ?Sized>(op: &'static str, t: Tensor) -> Res<Bo<X>>
where
    X::Repr: CoreRepr,
{
    <X::Repr as CoreRepr>::wrap_bool(op, t)
}

pub(crate) fn add_into<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    _options: &AddOptions,
    e: &mut dyn Engine,
    dest: Dest<<X::Repr as Repr>::Write<'_>>,
) -> Res<()>
where
    X::Repr: CoreRepr,
{
    // Rejected before `dest` is touched.
    if dest.mode() != DestMode::Overwrite {
        return Err(Error::unsupported(
            "add_into",
            format!(
                "destination mode {:?} is not supported; add_into overwrites its destination",
                dest.mode()
            ),
        ));
    }
    let out = <X::Repr as Repr>::erase_write(dest.into_target());
    body::add_into(a.as_read(), b.as_read(), e, out)
}

pub(crate) fn where_select<X: AsView + ?Sized, Y: SelectOperand<X::Repr> + ?Sized>(
    condition: &X,
    on_true: &Y,
    on_false: &impl AsView<Repr = Y::Repr>,
    e: &mut dyn Engine,
) -> Res<<Y::Repr as Repr>::Value>
where
    X::Repr: CoreRepr,
{
    <Y::Repr as Repr>::wrap_value(body::where_select(
        condition.as_read(),
        on_true.as_read(),
        on_false.as_read(),
        e,
    )?)
}

pub(crate) fn concatenate<X: AsView>(inputs: &[&X], axis: isize, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    // An empty list is reported by the engine (as before #2552).
    let axis = match inputs.first() {
        Some(first) => axes::axis("concatenate", axis, first.as_read().shape().len())?,
        None => 0,
    };
    // Up to four inputs keep their reads on the stack, as the former erased
    // entry passed the caller's slice without allocating.
    let out = match inputs {
        [] => body::concatenate(&[], axis, e)?,
        [a] => body::concatenate(&[a.as_read()], axis, e)?,
        [a, b] => body::concatenate(&[a.as_read(), b.as_read()], axis, e)?,
        [a, b, c] => body::concatenate(&[a.as_read(), b.as_read(), c.as_read()], axis, e)?,
        [a, b, c, d] => body::concatenate(
            &[a.as_read(), b.as_read(), c.as_read(), d.as_read()],
            axis,
            e,
        )?,
        _ => {
            let reads: Vec<_> = inputs.iter().map(|x| x.as_read()).collect();
            body::concatenate(&reads, axis, e)?
        }
    };
    val::<X>("concatenate", out)
}

pub(crate) fn exp<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("exp", body::exp(a.as_read(), e)?)
}

pub(crate) fn neg<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("neg", body::neg(a.as_read(), e)?)
}

pub(crate) fn sign<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("sign", body::sign(a.as_read(), e)?)
}

pub(crate) fn conj<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("conj", body::conj(a.as_read(), e)?)
}

pub(crate) fn log<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("log", body::log(a.as_read(), e)?)
}

pub(crate) fn expm1<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("expm1", body::expm1(a.as_read(), e)?)
}

pub(crate) fn log1p<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("log1p", body::log1p(a.as_read(), e)?)
}

pub(crate) fn erf<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("erf", body::erf(a.as_read(), e)?)
}

pub(crate) fn sin<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("sin", body::sin(a.as_read(), e)?)
}

pub(crate) fn cos<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("cos", body::cos(a.as_read(), e)?)
}

pub(crate) fn tanh<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("tanh", body::tanh(a.as_read(), e)?)
}

pub(crate) fn sqrt<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("sqrt", body::sqrt(a.as_read(), e)?)
}

pub(crate) fn rsqrt<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("rsqrt", body::rsqrt(a.as_read(), e)?)
}

pub(crate) fn abs<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Re<X>>
where
    X::Repr: CoreRepr,
{
    re::<X>("abs", body::abs(a.as_read(), e)?)
}

pub(crate) fn sigmoid<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("sigmoid", body::sigmoid(a.as_read(), e)?)
}

pub(crate) fn silu<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("silu", body::silu(a.as_read(), e)?)
}

pub(crate) fn softplus<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("softplus", body::softplus(a.as_read(), e)?)
}

pub(crate) fn gelu<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("gelu", body::gelu(a.as_read(), e)?)
}

pub(crate) fn gelu_tanh<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("gelu_tanh", body::gelu_tanh(a.as_read(), e)?)
}

pub(crate) fn mul<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("mul", body::mul(a.as_read(), b.as_read(), e)?)
}

pub(crate) fn sub<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("sub", body::sub(a.as_read(), b.as_read(), e)?)
}

pub(crate) fn div<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("div", body::div(a.as_read(), b.as_read(), e)?)
}

pub(crate) fn rem<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("rem", body::rem(a.as_read(), b.as_read(), e)?)
}

pub(crate) fn pow<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("pow", body::pow(a.as_read(), b.as_read(), e)?)
}

pub(crate) fn maximum<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("maximum", body::maximum(a.as_read(), b.as_read(), e)?)
}

pub(crate) fn minimum<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("minimum", body::minimum(a.as_read(), b.as_read(), e)?)
}

pub(crate) fn add<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    _options: &AddOptions,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("add", body::add(a.as_read(), b.as_read(), e)?)
}

pub(crate) fn matmul<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("matmul", body::matmul(a.as_read(), b.as_read(), e)?)
}

pub(crate) fn compare<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    dir: CompareDir,
    e: &mut dyn Engine,
) -> Res<Bo<X>>
where
    X::Repr: CoreRepr,
{
    boolean::<X>("compare", body::compare(a.as_read(), b.as_read(), dir, e)?)
}

pub(crate) fn clamp<X: AsView + ?Sized>(
    a: &X,
    lower: &impl AsView<Repr = X::Repr>,
    upper: &impl AsView<Repr = X::Repr>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "clamp",
        body::clamp(a.as_read(), lower.as_read(), upper.as_read(), e)?,
    )
}

pub(crate) fn dot_general<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    config: DotGeneralConfig,
    options: &DotGeneralOptions,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "dot_general",
        body::dot_general(
            a.as_read(),
            b.as_read(),
            &config,
            options.lhs.is_yes(),
            options.rhs.is_yes(),
            e,
        )?,
    )
}

pub(crate) fn reduce_sum<X: AsView + ?Sized>(
    a: &X,
    axes: Option<&[isize]>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "reduce_sum",
        body::reduce_sum(
            a.as_read(),
            axes::opt_axes("reduce_sum", axes, a.as_read().shape().len())?.as_deref(),
            e,
        )?,
    )
}

pub(crate) fn reduce_max<X: AsView + ?Sized>(
    a: &X,
    axes: Option<&[isize]>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "reduce_max",
        body::reduce_max(
            a.as_read(),
            axes::opt_axes("reduce_max", axes, a.as_read().shape().len())?.as_deref(),
            e,
        )?,
    )
}

pub(crate) fn reduce_min<X: AsView + ?Sized>(
    a: &X,
    axes: Option<&[isize]>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "reduce_min",
        body::reduce_min(
            a.as_read(),
            axes::opt_axes("reduce_min", axes, a.as_read().shape().len())?.as_deref(),
            e,
        )?,
    )
}

pub(crate) fn reduce_prod<X: AsView + ?Sized>(
    a: &X,
    axes: Option<&[isize]>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "reduce_prod",
        body::reduce_prod(
            a.as_read(),
            axes::opt_axes("reduce_prod", axes, a.as_read().shape().len())?.as_deref(),
            e,
        )?,
    )
}

pub(crate) fn reduce_sum_squares<X: AsView + ?Sized>(
    a: &X,
    axes: Option<&[isize]>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "reduce_sum_squares",
        body::reduce_sum_squares(
            a.as_read(),
            axes::opt_axes("reduce_sum_squares", axes, a.as_read().shape().len())?.as_deref(),
            e,
        )?,
    )
}

pub(crate) fn reduce_mean<X: AsView + ?Sized>(
    a: &X,
    axes: Option<&[isize]>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "reduce_mean",
        body::reduce_mean(
            a.as_read(),
            axes::opt_axes("reduce_mean", axes, a.as_read().shape().len())?.as_deref(),
            e,
        )?,
    )
}

pub(crate) fn reshape<X: AsView + ?Sized>(a: &X, shape: &[usize], e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("reshape", body::reshape(a.as_read(), shape, e)?)
}

pub(crate) fn broadcast_in_dim<X: AsView + ?Sized>(
    a: &X,
    shape: &[usize],
    dims: &[usize],
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "broadcast_in_dim",
        body::broadcast_in_dim(a.as_read(), shape, dims, e)?,
    )
}

pub(crate) fn transpose<X: AsView + ?Sized>(
    a: &X,
    perm: &[isize],
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "transpose",
        body::transpose(
            a.as_read(),
            &axes::axes("transpose", perm, a.as_read().shape().len())?,
            e,
        )?,
    )
}

pub(crate) fn reverse<X: AsView + ?Sized>(a: &X, axes: &[isize], e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "reverse",
        body::reverse(
            a.as_read(),
            &axes::axes("reverse", axes, a.as_read().shape().len())?,
            e,
        )?,
    )
}

pub(crate) fn tril<X: AsView + ?Sized>(a: &X, k: i64, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("tril", body::tril(a.as_read(), k, e)?)
}

pub(crate) fn triu<X: AsView + ?Sized>(a: &X, k: i64, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("triu", body::triu(a.as_read(), k, e)?)
}

pub(crate) fn extract_diag<X: AsView + ?Sized>(
    a: &X,
    axis_a: isize,
    axis_b: isize,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "extract_diag",
        body::extract_diag(
            a.as_read(),
            axes::axis("extract_diag", axis_a, a.as_read().shape().len())?,
            axes::axis("extract_diag", axis_b, a.as_read().shape().len())?,
            e,
        )?,
    )
}

pub(crate) fn embed_diag<X: AsView + ?Sized>(
    a: &X,
    axis_a: isize,
    axis_b: isize,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "embed_diag",
        body::embed_diag(
            a.as_read(),
            axes::axis("embed_diag", axis_a, a.as_read().shape().len())?,
            axes::axis("embed_diag", axis_b, a.as_read().shape().len() + 1)?,
            e,
        )?,
    )
}

pub(crate) fn slice<X: AsView + ?Sized>(
    a: &X,
    config: SliceConfig,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("slice", body::slice(a.as_read(), &config, e)?)
}

pub(crate) fn pad<X: AsView + ?Sized>(a: &X, config: PadConfig, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("pad", body::pad(a.as_read(), &config, e)?)
}

pub(crate) fn gather<X: AsView + ?Sized>(
    a: &X,
    indices: &impl AsView,
    config: GatherConfig,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "gather",
        body::gather(a.as_read(), indices.as_read(), &config, e)?,
    )
}

pub(crate) fn scatter<X: AsView + ?Sized>(
    a: &X,
    indices: &impl AsView,
    updates: &impl AsView,
    config: ScatterConfig,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "scatter",
        body::scatter(
            a.as_read(),
            indices.as_read(),
            updates.as_read(),
            &config,
            e,
        )?,
    )
}

pub(crate) fn dynamic_slice<X: AsView + ?Sized>(
    a: &X,
    starts: &impl AsView,
    sizes: &[usize],
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "dynamic_slice",
        body::dynamic_slice(a.as_read(), starts.as_read(), sizes, e)?,
    )
}

pub(crate) fn dynamic_update_slice<X: AsView + ?Sized>(
    a: &X,
    update: &impl AsView<Repr = X::Repr>,
    starts: &impl AsView,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "dynamic_update_slice",
        body::dynamic_update_slice(a.as_read(), update.as_read(), starts.as_read(), e)?,
    )
}

pub(crate) fn convert<X: AsView + ?Sized>(a: &X, to: DType, e: &mut dyn Engine) -> Res<Tensor>
where
    X::Repr: CoreRepr,
{
    body::convert(a.as_read(), to, e)
}

pub(crate) fn cast<X: AsView + ?Sized>(a: &X, to: DType, e: &mut dyn Engine) -> Res<Tensor>
where
    X::Repr: CoreRepr,
{
    body::cast(a.as_read(), to, e)
}

pub(crate) fn scale_real<X: AsView + ?Sized>(a: &X, factor: f64, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>("scale_real", body::scale_real(a.as_read(), factor, e)?)
}

pub(crate) fn scale_complex<X: AsView + ?Sized>(
    a: &X,
    factor: Complex64,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "scale_complex",
        body::scale_complex(a.as_read(), factor, e)?,
    )
}

pub(crate) fn softmax<X: AsView + ?Sized>(a: &X, axis: isize, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "softmax",
        body::softmax(
            a.as_read(),
            axes::axis("softmax", axis, a.as_read().shape().len())?,
            e,
        )?,
    )
}

pub(crate) fn log_softmax<X: AsView + ?Sized>(a: &X, axis: isize, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "log_softmax",
        body::log_softmax(
            a.as_read(),
            axes::axis("log_softmax", axis, a.as_read().shape().len())?,
            e,
        )?,
    )
}

pub(crate) fn masked_softmax<X: AsView + ?Sized>(
    a: &X,
    mask: &impl AsView,
    axis: isize,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "masked_softmax",
        body::masked_softmax(
            a.as_read(),
            mask.as_read(),
            axes::axis("masked_softmax", axis, a.as_read().shape().len())?,
            e,
        )?,
    )
}

pub(crate) fn masked_log_softmax<X: AsView + ?Sized>(
    a: &X,
    mask: &impl AsView,
    axis: isize,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "masked_log_softmax",
        body::masked_log_softmax(
            a.as_read(),
            mask.as_read(),
            axes::axis("masked_log_softmax", axis, a.as_read().shape().len())?,
            e,
        )?,
    )
}

pub(crate) fn take_along_axis<X: AsView + ?Sized>(
    a: &X,
    indices: &impl AsView,
    axis: isize,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "take_along_axis",
        body::take_along_axis(
            a.as_read(),
            indices.as_read(),
            axes::axis("take_along_axis", axis, a.as_read().shape().len())?,
            e,
        )?,
    )
}

pub(crate) fn layer_norm<X: AsView + ?Sized>(
    a: &X,
    axis: isize,
    weight: Option<&X>,
    bias: Option<&X>,
    eps: f64,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "layer_norm",
        body::layer_norm(
            a.as_read(),
            axes::axis("layer_norm", axis, a.as_read().shape().len())?,
            weight.map(|w| w.as_read()),
            bias.map(|b| b.as_read()),
            eps,
            e,
        )?,
    )
}

pub(crate) fn rms_norm<X: AsView + ?Sized>(
    a: &X,
    axis: isize,
    weight: Option<&X>,
    bias: Option<&X>,
    eps: f64,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: CoreRepr,
{
    val::<X>(
        "rms_norm",
        body::rms_norm(
            a.as_read(),
            axes::axis("rms_norm", axis, a.as_read().shape().len())?,
            weight.map(|w| w.as_read()),
            bias.map(|b| b.as_read()),
            eps,
            e,
        )?,
    )
}
