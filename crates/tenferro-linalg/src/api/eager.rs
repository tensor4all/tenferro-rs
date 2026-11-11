//! Eager forward bodies of the registry (`EagerLinalgOps`).
//!
//! Each entry maps the options onto the arguments of the existing eager
//! session operation and builds the result struct from its tuple, in tuple
//! order. Options values the former API could not express return `Unsupported`
//! before any operation is recorded.

use tenferro_ad::{EagerSession, EagerTensor, Error, Result};

use super::concrete::{require_lower, triangular_flags};
use super::options::{
    CholeskyOptions, EigvalshOptions, NormOptions, PinvOptions, SvdvalsOptions,
    TriangularSolveOptions,
};
use crate::eager_ext::EagerLinalg as Old;
use crate::eager_ext::{apply_linalg_eager_in_session, one_output};
use crate::extension::{
    EighOptions, LinalgOp, QrOptions, SvdOptions, DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
};
use crate::flags::KeepDims;
use crate::results::{Eig, Eigh, FullPivLu, Lu, Qr, Slogdet, Svd};
use crate::validation::ensure_float_or_complex;
use crate::{EighDriver, HouseholderQr, RankRevealingQrOptions, RankRevealingQrResult, SvdShape};

type S<'a, 'b> = &'a mut EagerSession<'b>;
type T = EagerTensor;

fn unsupported(op: &'static str, message: &str) -> Error {
    tenferro_tensor::Error::unsupported(op, message).into()
}

pub(crate) fn svd(s: S<'_, '_>, a: &T, o: &SvdOptions) -> Result<Svd<T, T>> {
    let (u, sv, vh) = match o.shape {
        SvdShape::Thin if *o == SvdOptions::default() => Old::svd(s, a)?,
        SvdShape::Thin => Old::svd_with_options(s, a, *o)?,
        SvdShape::Full => {
            if *o != SvdOptions::default().shape(SvdShape::Full) {
                return Err(unsupported(
                    "svd",
                    "SvdShape::Full does not support a non-default gauge, derivative_eps or driver yet; use SvdShape::Thin or the default options",
                ));
            }
            Old::svd_full(s, a)?
        }
    };
    Ok(Svd { u, s: sv, vh })
}

pub(crate) fn svdvals(s: S<'_, '_>, a: &T, o: &SvdvalsOptions) -> Result<T> {
    ensure_float_or_complex("svdvals", a.dtype())?;
    one_output(
        apply_linalg_eager_in_session(
            s,
            LinalgOp::SvdVals {
                derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                driver: o.driver,
            },
            &[a],
        )?,
        "svdvals",
    )
}

pub(crate) fn qr(s: S<'_, '_>, a: &T, o: &QrOptions) -> Result<Qr<T>> {
    let (q, r) = if *o == QrOptions::default() {
        Old::qr(s, a)?
    } else {
        Old::qr_with_options(s, a, *o)?
    };
    Ok(Qr { q, r })
}

pub(crate) fn householder_qr(s: S<'_, '_>, a: &T) -> Result<HouseholderQr<T>> {
    Old::householder_qr(s, a)
}

pub(crate) fn rank_revealing_qr(
    s: S<'_, '_>,
    a: &T,
    o: &RankRevealingQrOptions,
) -> Result<RankRevealingQrResult<T, T>> {
    Old::rank_revealing_qr(s, a, *o)
}

pub(crate) fn lu(s: S<'_, '_>, a: &T) -> Result<Lu<T, T>> {
    let (p, l, u, parity) = Old::lu(s, a)?;
    Ok(Lu { p, l, u, parity })
}

pub(crate) fn full_piv_lu(s: S<'_, '_>, a: &T) -> Result<FullPivLu<T, T>> {
    let (p, l, u, q, parity) = Old::full_piv_lu(s, a)?;
    Ok(FullPivLu { p, l, u, q, parity })
}

pub(crate) fn full_piv_lu_solve(s: S<'_, '_>, a: &T, b: &T) -> Result<T> {
    Old::full_piv_lu_solve(s, a, b)
}

pub(crate) fn solve(s: S<'_, '_>, a: &T, b: &T) -> Result<T> {
    Old::solve(s, a, b)
}

pub(crate) fn lstsq(s: S<'_, '_>, a: &T, b: &T) -> Result<T> {
    Old::lstsq(s, a, b)
}

pub(crate) fn triangular_solve(
    s: S<'_, '_>,
    a: &T,
    b: &T,
    o: &TriangularSolveOptions,
) -> Result<T> {
    let complex = matches!(
        a.dtype(),
        tenferro_tensor::DType::C32 | tenferro_tensor::DType::C64
    );
    let (left, lower, transpose, unit) = triangular_flags(o, complex)?;
    Old::triangular_solve(s, a, b, left, lower, transpose, unit)
}

pub(crate) fn cholesky(s: S<'_, '_>, a: &T, o: &CholeskyOptions) -> Result<T> {
    require_lower(o)?;
    Old::cholesky(s, a)
}

pub(crate) fn eig(s: S<'_, '_>, a: &T) -> Result<Eig<T>> {
    let (values, vectors) = Old::eig(s, a)?;
    Ok(Eig { values, vectors })
}

pub(crate) fn eigvals(s: S<'_, '_>, a: &T) -> Result<T> {
    Old::eigvals(s, a)
}

pub(crate) fn eigh(s: S<'_, '_>, a: &T, o: &EighOptions) -> Result<Eigh<T, T>> {
    let (values, vectors) = if *o == EighOptions::default() {
        Old::eigh(s, a)?
    } else {
        Old::eigh_with_options(s, a, *o)?
    };
    Ok(Eigh { values, vectors })
}

pub(crate) fn eigvalsh(s: S<'_, '_>, a: &T, o: &EigvalshOptions) -> Result<T> {
    if o.driver == EighDriver::Auto {
        return Old::eigvalsh(s, a);
    }
    one_output(
        apply_linalg_eager_in_session(
            s,
            LinalgOp::EighVals {
                derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                driver: o.driver,
            },
            &[a],
        )?,
        "eigvalsh",
    )
}

pub(crate) fn slogdet(s: S<'_, '_>, a: &T) -> Result<Slogdet<T, T>> {
    let (sign, log_abs_det) = Old::slogdet(s, a)?;
    Ok(Slogdet { sign, log_abs_det })
}

pub(crate) fn det(s: S<'_, '_>, a: &T) -> Result<T> {
    Old::det(s, a)
}

pub(crate) fn inv(s: S<'_, '_>, a: &T) -> Result<T> {
    Old::inv(s, a)
}

pub(crate) fn pinv(s: S<'_, '_>, a: &T, o: &PinvOptions) -> Result<T> {
    match o.rtol {
        None => Old::pinv(s, a),
        Some(rtol) => Old::pinv_with_rtol(s, a, rtol),
    }
}

pub(crate) fn norm(s: S<'_, '_>, a: &T, o: &NormOptions) -> Result<T> {
    let ord = o.ord.to_ord()?;
    let dim = o.unsigned_axes(a.shape().len())?;
    Old::norm(s, a, ord, dim.as_deref(), o.keep_dims == KeepDims::Yes)
}
