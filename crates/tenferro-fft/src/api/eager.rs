//! Eager forward bodies of the registry (`EagerFftOps`).
//!
//! The one-axis entries run the eager extension op of the former
//! session extension; the N-dimensional entries run the same extension op
//! with several axes, as the traced surface records it. The consuming
//! in-place transforms (exception E01) run the former in-place body on the
//! session's engine.

use tenferro_ad::{EagerSession, EagerTensor, Result};
use tenferro_runtime::DType;

use super::options::{FftOptions, FftnOptions};
use crate::eager_ext::{apply_eager_fft_in_session, apply_eager_fft_op_in_session};
use crate::{
    graph_build_fft_error, require_runtime_dtype, runtime_forward_fft_operation, validated_fft_op,
    EagerFftInPlaceError, FftOperation,
};

type S<'a, 'b> = &'a mut EagerSession<'b>;
type T = EagerTensor;

const COMPLEX: &[DType] = &[DType::C32, DType::C64];
const REAL: &[DType] = &[DType::F32, DType::F64];
const ANY: &[DType] = &[DType::F32, DType::F64, DType::C32, DType::C64];

fn one_axis(
    s: S<'_, '_>,
    op: &'static str,
    a: &T,
    operation: FftOperation,
    o: &FftOptions,
) -> Result<T> {
    apply_eager_fft_in_session(s, op, a, operation, o.n, o.axis, o.norm)
}

fn n_axes(
    s: S<'_, '_>,
    op: &'static str,
    a: &T,
    operation: FftOperation,
    o: &FftnOptions,
) -> Result<T> {
    let axes = o.resolved_axes(a.shape().len());
    let fft_op = validated_fft_op(
        op,
        operation,
        None,
        a.shape().len(),
        Some(a.shape()),
        o.shape.as_deref(),
        "shape",
        &axes,
        o.norm,
    )
    .map_err(graph_build_fft_error)?;
    apply_eager_fft_op_in_session(s, fft_op, a)
}

pub(crate) fn fft(s: S<'_, '_>, a: &T, o: &FftOptions) -> Result<T> {
    let operation = runtime_forward_fft_operation(a.dtype())?;
    one_axis(s, "fft", a, operation, o)
}

pub(crate) fn ifft(s: S<'_, '_>, a: &T, o: &FftOptions) -> Result<T> {
    require_runtime_dtype("ifft", a.dtype(), COMPLEX, "C32 or C64")?;
    one_axis(s, "ifft", a, FftOperation::C2cInverse, o)
}

pub(crate) fn rfft(s: S<'_, '_>, a: &T, o: &FftOptions) -> Result<T> {
    require_runtime_dtype("rfft", a.dtype(), REAL, "F32 or F64")?;
    one_axis(s, "rfft", a, FftOperation::R2cOnesided, o)
}

pub(crate) fn irfft(s: S<'_, '_>, a: &T, o: &FftOptions) -> Result<T> {
    require_runtime_dtype("irfft", a.dtype(), COMPLEX, "C32 or C64")?;
    one_axis(s, "irfft", a, FftOperation::C2r, o)
}

pub(crate) fn fftn(s: S<'_, '_>, a: &T, o: &FftnOptions) -> Result<T> {
    require_runtime_dtype("fftn", a.dtype(), ANY, "F32, F64, C32, or C64")?;
    let operation = runtime_forward_fft_operation(a.dtype())?;
    n_axes(s, "fftn", a, operation, o)
}

pub(crate) fn ifftn(s: S<'_, '_>, a: &T, o: &FftnOptions) -> Result<T> {
    require_runtime_dtype("ifftn", a.dtype(), COMPLEX, "C32 or C64")?;
    n_axes(s, "ifftn", a, FftOperation::C2cInverse, o)
}

pub(crate) fn rfftn(s: S<'_, '_>, a: &T, o: &FftnOptions) -> Result<T> {
    require_runtime_dtype("rfftn", a.dtype(), REAL, "F32 or F64")?;
    n_axes(s, "rfftn", a, FftOperation::R2cOnesided, o)
}

pub(crate) fn irfftn(s: S<'_, '_>, a: &T, o: &FftnOptions) -> Result<T> {
    require_runtime_dtype("irfftn", a.dtype(), COMPLEX, "C32 or C64")?;
    n_axes(s, "irfftn", a, FftOperation::C2r, o)
}

/// Consuming in-place transform (E01). A length change is impossible in
/// place, so `n: Some(_)` is a validation error that returns the input.
pub(crate) fn in_place(
    s: S<'_, '_>,
    op: &'static str,
    x: T,
    operation: FftOperation,
    o: &FftOptions,
) -> std::result::Result<T, EagerFftInPlaceError> {
    if let Err(source) = tenferro_ad::extension::ensure_session_owns(s, &x) {
        return Err(EagerFftInPlaceError::Rejected {
            input: Box::new(x),
            source,
        });
    }
    if let Some(n) = o.n {
        return Err(EagerFftInPlaceError::Rejected {
            input: Box::new(x),
            source: tenferro_tensor::Error::invalid_argument(
                op,
                "n",
                format!(
                    "an in-place FFT keeps the input length; got n = Some({n}), use the out-of-place `fft_with` to change the length"
                ),
            )
            .into(),
        });
    }
    crate::eager_in_place::apply(x, s.engine(), op, operation, o.axis, o.norm)
}
