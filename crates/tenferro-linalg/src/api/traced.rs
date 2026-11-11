//! Traced forward bodies of the registry (`TracedLinalgOps`).
//!
//! Each entry maps the options onto the arguments of the existing traced
//! construction and builds the result struct from its tuple, in tuple order.
//! Options values the former API could not express return `Unsupported` before
//! any graph node is built.

use std::sync::Arc;

use tenferro_runtime::extension::apply;
use tenferro_runtime::{Error, Result, TracedTensor};

use super::concrete::{require_lower, triangular_flags};
use super::options::{
    CholeskyOptions, EigvalshOptions, NormOptions, PinvOptions, SvdvalsOptions,
    TriangularSolveOptions,
};
use crate::extension::{
    EighOptions, LinalgExtensionOp, LinalgOp, QrOptions, SvdOptions,
    DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
};
use crate::flags::KeepDims;
use crate::results::{Eig, Eigh, FullPivLu, Lu, Qr, Slogdet, Svd};
use crate::traced::TracedLinalg as Old;
use crate::validation::ensure_float_or_complex;
use crate::{EighDriver, HouseholderQr, RankRevealingQrOptions, RankRevealingQrResult, SvdShape};

type T = TracedTensor;

fn unsupported(op: &'static str, message: &str) -> Error {
    tenferro_tensor::Error::unsupported(op, message).into()
}

fn single(outputs: Vec<T>, name: &str) -> Result<T> {
    let mut it = outputs.into_iter();
    match (it.next(), it.next()) {
        (Some(out), None) => Ok(out),
        _ => Err(Error::Internal(format!(
            "{name} must produce exactly 1 output"
        ))),
    }
}

pub(crate) fn svd(a: &T, o: &SvdOptions) -> Result<Svd<T, T>> {
    let (u, s, vh) = match o.shape {
        SvdShape::Thin if *o == SvdOptions::default() => Old::svd(a)?,
        SvdShape::Thin => Old::svd_with_options(a, *o)?,
        SvdShape::Full => {
            if *o != SvdOptions::default().shape(SvdShape::Full) {
                return Err(unsupported(
                    "svd",
                    "SvdShape::Full does not support a non-default gauge, derivative_eps or driver yet; use SvdShape::Thin or the default options",
                ));
            }
            Old::svd_full(a)?
        }
    };
    Ok(Svd { u, s, vh })
}

pub(crate) fn svdvals(a: &T, o: &SvdvalsOptions) -> Result<T> {
    ensure_float_or_complex("svdvals", a.dtype())?;
    single(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::SvdVals {
                derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                driver: o.driver,
            })),
            &[a],
        )?,
        "svdvals",
    )
}

pub(crate) fn qr(a: &T, o: &QrOptions) -> Result<Qr<T>> {
    let (q, r) = if *o == QrOptions::default() {
        Old::qr(a)?
    } else {
        Old::qr_with_options(a, *o)?
    };
    Ok(Qr { q, r })
}

pub(crate) fn householder_qr(a: &T) -> Result<HouseholderQr<T>> {
    Old::householder_qr(a)
}

pub(crate) fn rank_revealing_qr(
    a: &T,
    o: &RankRevealingQrOptions,
) -> Result<RankRevealingQrResult<T, T>> {
    Old::rank_revealing_qr(a, *o)
}

pub(crate) fn lu(a: &T) -> Result<Lu<T, T>> {
    let (p, l, u, parity) = Old::lu(a)?;
    Ok(Lu { p, l, u, parity })
}

pub(crate) fn full_piv_lu(a: &T) -> Result<FullPivLu<T, T>> {
    let (p, l, u, q, parity) = Old::full_piv_lu(a)?;
    Ok(FullPivLu { p, l, u, q, parity })
}

pub(crate) fn full_piv_lu_solve(a: &T, b: &T) -> Result<T> {
    Old::full_piv_lu_solve(a, b)
}

pub(crate) fn solve(a: &T, b: &T) -> Result<T> {
    Old::solve(a, b)
}

pub(crate) fn lstsq(a: &T, b: &T) -> Result<T> {
    Old::lstsq(a, b)
}

pub(crate) fn triangular_solve(a: &T, b: &T, o: &TriangularSolveOptions) -> Result<T> {
    let complex = matches!(
        a.dtype(),
        tenferro_tensor::DType::C32 | tenferro_tensor::DType::C64
    );
    let (left, lower, transpose, unit) = triangular_flags(o, complex)?;
    Old::triangular_solve(a, b, left, lower, transpose, unit)
}

pub(crate) fn cholesky(a: &T, o: &CholeskyOptions) -> Result<T> {
    require_lower(o)?;
    Old::cholesky(a)
}

pub(crate) fn eig(a: &T) -> Result<Eig<T>> {
    let (values, vectors) = Old::eig(a)?;
    Ok(Eig { values, vectors })
}

pub(crate) fn eigvals(a: &T) -> Result<T> {
    Old::eigvals(a)
}

pub(crate) fn eigh(a: &T, o: &EighOptions) -> Result<Eigh<T, T>> {
    let (values, vectors) = if *o == EighOptions::default() {
        Old::eigh(a)?
    } else {
        Old::eigh_with_options(a, *o)?
    };
    Ok(Eigh { values, vectors })
}

pub(crate) fn eigvalsh(a: &T, o: &EigvalshOptions) -> Result<T> {
    if o.driver == EighDriver::Auto {
        return Old::eigvalsh(a);
    }
    ensure_float_or_complex("eigvalsh", a.dtype())?;
    single(
        apply(
            Arc::new(LinalgExtensionOp::new(LinalgOp::EighVals {
                derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                driver: o.driver,
            })),
            &[a],
        )?,
        "eigvalsh",
    )
}

pub(crate) fn slogdet(a: &T) -> Result<Slogdet<T, T>> {
    let (sign, log_abs_det) = Old::slogdet(a)?;
    Ok(Slogdet { sign, log_abs_det })
}

pub(crate) fn det(a: &T) -> Result<T> {
    Old::det(a)
}

pub(crate) fn inv(a: &T) -> Result<T> {
    Old::inv(a)
}

pub(crate) fn pinv(a: &T, o: &PinvOptions) -> Result<T> {
    match o.rtol {
        None => Old::pinv(a),
        Some(rtol) => Old::pinv_with_rtol(a, rtol),
    }
}

pub(crate) fn norm(a: &T, o: &NormOptions) -> Result<T> {
    let ord = o.ord.to_ord()?;
    let dim = o.unsigned_axes(a.rank())?;
    Old::norm(a, ord, dim.as_deref(), o.keep_dims == KeepDims::Yes)
}
