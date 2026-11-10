//! Traced forward bodies of the registry (`TracedFftOps`).
//!
//! Each entry records the FFT extension op the former traced surface
//! recorded, with the same dtype checks and the same graph-build validation.

use tenferro_runtime::{DType, Result, TracedTensor};

use super::options::{FftOptions, FftnOptions};
use crate::{
    apply_unary_fftn, require_runtime_dtype, runtime_forward_fft_operation, single_axis_lengths,
    FftOperation,
};

type T = TracedTensor;

const COMPLEX: &[DType] = &[DType::C32, DType::C64];
const REAL: &[DType] = &[DType::F32, DType::F64];
const ANY: &[DType] = &[DType::F32, DType::F64, DType::C32, DType::C64];

fn one_axis(op: &'static str, a: &T, operation: FftOperation, o: &FftOptions) -> Result<T> {
    let lengths = single_axis_lengths(o.n);
    apply_unary_fftn(
        op,
        a,
        operation,
        lengths.as_ref().map(<[usize; 1]>::as_slice),
        "n",
        &[o.axis],
        o.norm,
    )
}

fn n_axes(op: &'static str, a: &T, operation: FftOperation, o: &FftnOptions) -> Result<T> {
    let axes = o.resolved_axes(a.rank());
    apply_unary_fftn(op, a, operation, o.shape.as_deref(), "shape", &axes, o.norm)
}

pub(crate) fn fft(a: &T, o: &FftOptions) -> Result<T> {
    let operation = runtime_forward_fft_operation(a.dtype())?;
    one_axis("fft", a, operation, o)
}

pub(crate) fn ifft(a: &T, o: &FftOptions) -> Result<T> {
    require_runtime_dtype("ifft", a.dtype(), COMPLEX, "C32 or C64")?;
    one_axis("ifft", a, FftOperation::C2cInverse, o)
}

pub(crate) fn rfft(a: &T, o: &FftOptions) -> Result<T> {
    require_runtime_dtype("rfft", a.dtype(), REAL, "F32 or F64")?;
    one_axis("rfft", a, FftOperation::R2cOnesided, o)
}

pub(crate) fn irfft(a: &T, o: &FftOptions) -> Result<T> {
    require_runtime_dtype("irfft", a.dtype(), COMPLEX, "C32 or C64")?;
    one_axis("irfft", a, FftOperation::C2r, o)
}

pub(crate) fn fftn(a: &T, o: &FftnOptions) -> Result<T> {
    require_runtime_dtype("fftn", a.dtype(), ANY, "F32, F64, C32, or C64")?;
    let operation = runtime_forward_fft_operation(a.dtype())?;
    n_axes("fftn", a, operation, o)
}

pub(crate) fn ifftn(a: &T, o: &FftnOptions) -> Result<T> {
    require_runtime_dtype("ifftn", a.dtype(), COMPLEX, "C32 or C64")?;
    n_axes("ifftn", a, FftOperation::C2cInverse, o)
}

pub(crate) fn rfftn(a: &T, o: &FftnOptions) -> Result<T> {
    require_runtime_dtype("rfftn", a.dtype(), REAL, "F32 or F64")?;
    n_axes("rfftn", a, FftOperation::R2cOnesided, o)
}

pub(crate) fn irfftn(a: &T, o: &FftnOptions) -> Result<T> {
    require_runtime_dtype("irfftn", a.dtype(), COMPLEX, "C32 or C64")?;
    n_axes("irfftn", a, FftOperation::C2r, o)
}
