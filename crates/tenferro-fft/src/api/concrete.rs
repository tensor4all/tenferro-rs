//! Concrete forward bodies of the registry (`FftOps`).
//!
//! Every entry is generic over the input adapter and runs one dtype-erased
//! body over [`TensorRead`]. A one-axis transform of an operand that is
//! exactly an owned [`Tensor`] takes the owned route (`FftBackend::execute_fft`);
//! any other operand takes the borrowed route (`FftBackend::execute_fft_read`).
//! These are the routes of the former owned and borrowed concrete entries, so
//! results are bit-identical to them. The N-dimensional transforms run the
//! same multi-axis route as [`FftExecutor`](crate::FftExecutor), with a
//! one-entry plan cache owned by the call.

use std::num::NonZeroUsize;

use tenferro_tensor::{AsView, DType, Engine, Tensor, TensorRead};

use super::options::{FftOptions, FftnOptions};
use super::repr::FftRepr;
use crate::{
    concrete_fft_operation, concrete_fft_spec, concrete_ifft_operation, concrete_irfft_operation,
    concrete_rfft_operation, execute_concrete_fft_op, execute_concrete_fft_read_op, execute_fft_op,
    validated_fft_op, with_fft_exec_session, FftExecutionCache, FftOperation, FftPlanCache,
};

type Res<T> = tenferro_tensor::Result<T>;
type Cx<X> = <<X as AsView>::Repr as FftRepr>::Complex;
type Re<X> = <<X as AsView>::Repr as FftRepr>::Real;
type OperationOf = fn(&'static str, DType) -> Res<FftOperation>;

/// One-axis transform: owned route for an owned tensor, borrowed route
/// otherwise (the former owned and borrowed concrete bodies).
pub(crate) fn one_axis(
    a: TensorRead<'_>,
    op: &'static str,
    operation_of: OperationOf,
    o: &FftOptions,
    e: &mut dyn Engine,
) -> Res<Tensor> {
    match a.as_tensor() {
        Some(t) => {
            let spec = concrete_fft_spec(
                op,
                operation_of(op, t.dtype())?,
                t.dtype(),
                t.shape(),
                o.n,
                o.axis,
                o.norm,
            )?;
            with_fft_exec_session(e, op, |backend| execute_concrete_fft_op(t, &spec, backend))
        }
        None => with_fft_exec_session(e, op, |backend| {
            execute_concrete_fft_read_op(
                &a,
                operation_of(op, a.dtype())?,
                op,
                o.n,
                o.axis,
                o.norm,
                backend,
            )
        }),
    }
}

/// Multi-axis transform over the axes `o` resolves for this input.
pub(crate) fn n_axes(
    a: TensorRead<'_>,
    op: &'static str,
    operation_of: OperationOf,
    o: &FftnOptions,
    e: &mut dyn Engine,
) -> Res<Tensor> {
    let rank = a.shape().len();
    let axes = o.resolved_axes(rank);
    let fft_op = validated_fft_op(
        op,
        operation_of(op, a.dtype())?,
        Some(a.dtype()),
        rank,
        Some(a.shape()),
        o.shape.as_deref(),
        "shape",
        &axes,
        o.norm,
    )?;
    with_fft_exec_session(e, op, |backend| {
        let mut plans = FftPlanCache::with_capacity(NonZeroUsize::MIN);
        execute_fft_op(
            &fft_op,
            &a,
            backend,
            FftExecutionCache::caller_owned(&mut plans),
        )
    })
}

macro_rules! entries {
    ($($name:ident, $kind:ident, $options:ty, $operation:path, $wrap:ident, $out:ident;)*) => {$(
        pub(crate) fn $name<X: AsView + ?Sized>(
            a: &X,
            o: &$options,
            e: &mut dyn Engine,
        ) -> Res<$out<X>>
        where
            X::Repr: FftRepr,
        {
            <X::Repr as FftRepr>::$wrap($kind(a.as_read(), stringify!($name), $operation, o, e)?)
        }
    )*};
}

entries! {
    fft, one_axis, FftOptions, concrete_fft_operation, wrap_complex, Cx;
    ifft, one_axis, FftOptions, concrete_ifft_operation, wrap_complex, Cx;
    rfft, one_axis, FftOptions, concrete_rfft_operation, wrap_complex, Cx;
    irfft, one_axis, FftOptions, concrete_irfft_operation, wrap_real, Re;
    fftn, n_axes, FftnOptions, concrete_fft_operation, wrap_complex, Cx;
    ifftn, n_axes, FftnOptions, concrete_ifft_operation, wrap_complex, Cx;
    rfftn, n_axes, FftnOptions, concrete_rfft_operation, wrap_complex, Cx;
    irfftn, n_axes, FftnOptions, concrete_irfft_operation, wrap_real, Re;
}
