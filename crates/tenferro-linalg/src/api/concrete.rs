//! Concrete forward bodies of the registry (`LinalgOps`).
//!
//! Every public entry is generic over the input adapter and delegates to one
//! dtype-erased body over [`TensorRead`]; the body re-uses the existing entries
//! so every input kind returns bit-identical results to the former API:
//!
//! - a lane engine (`CpuEngine::with_lanes`) is tried first by the entry
//!   itself (see `crate::lane`);
//! - an operand that is exactly an owned [`Tensor`] takes the owned route of
//!   [`OwnedLinalg`](crate::OwnedLinalg);
//! - any other operand (a view, a typed tensor) takes the borrowed route of
//!   [`ReadLinalg`](crate::ReadLinalg).
//!
//! The result is re-wrapped through [`LinalgRepr`](crate::LinalgRepr)'s checked
//! downcasts for typed inputs.

use tenferro_tensor::{AsView, Dest, DestMode, Engine, Error, Repr, Tensor, TensorRead};

use super::options::{
    CholeskyOptions, EigvalshOptions, NormOptions, PinvOptions, SvdvalsOptions,
    TriangularSolveOptions,
};
use super::repr::LinalgRepr;
use crate::extension::{EighOptions, QrOptions, SvdDriver, SvdOptions};
use crate::flags::{Diag, KeepDims, MatOp, Side, Uplo};
use crate::results::{Eig, Eigh, FullPivLu, Lu, Qr, Slogdet, Svd};
use crate::tensor_ext::with_linalg_backend;
use crate::{
    lane, EighDriver, HouseholderQr, OwnedLinalg, RankRevealingQrOptions, RankRevealingQrResult,
    ReadLinalg, SvdShape,
};

type Res<T> = tenferro_tensor::Result<T>;
type Val<X> = <<X as AsView>::Repr as Repr>::Value;
type Re<X> = <<X as AsView>::Repr as LinalgRepr>::Real;
type Cx<X> = <<X as AsView>::Repr as LinalgRepr>::Complex;
type Ix<X> = <<X as AsView>::Repr as LinalgRepr>::Index;

fn val<X: AsView + ?Sized>(t: Tensor) -> Res<Val<X>>
where
    X::Repr: LinalgRepr,
{
    <X::Repr as Repr>::wrap_value(t)
}

fn re<X: AsView + ?Sized>(t: Tensor) -> Res<Re<X>>
where
    X::Repr: LinalgRepr,
{
    <X::Repr as LinalgRepr>::wrap_real(t)
}

fn cx<X: AsView + ?Sized>(t: Tensor) -> Res<Cx<X>>
where
    X::Repr: LinalgRepr,
{
    <X::Repr as LinalgRepr>::wrap_complex(t)
}

fn ix<X: AsView + ?Sized>(t: Tensor) -> Res<Ix<X>>
where
    X::Repr: LinalgRepr,
{
    <X::Repr as LinalgRepr>::wrap_index(t)
}

/// Runs `owned` for an operand that is exactly an owned tensor, `read` otherwise.
fn route<R>(
    a: TensorRead<'_>,
    e: &mut dyn Engine,
    owned: impl FnOnce(&Tensor, &mut dyn Engine) -> R,
    read: impl FnOnce(TensorRead<'_>, &mut dyn Engine) -> R,
) -> R {
    match a.as_tensor() {
        Some(t) => owned(t, e),
        None => read(a, e),
    }
}

/// Like [`route`] for two operands: the owned route needs both to be owned.
fn route2<R>(
    a: TensorRead<'_>,
    b: TensorRead<'_>,
    e: &mut dyn Engine,
    owned: impl FnOnce(&Tensor, &Tensor, &mut dyn Engine) -> R,
    read: impl FnOnce(TensorRead<'_>, TensorRead<'_>, &mut dyn Engine) -> R,
) -> R {
    match (a.as_tensor(), b.as_tensor()) {
        (Some(ta), Some(tb)) => owned(ta, tb, e),
        _ => read(a, b, e),
    }
}

fn is_complex(read: &TensorRead<'_>) -> bool {
    matches!(
        read.dtype(),
        tenferro_tensor::DType::C32 | tenferro_tensor::DType::C64
    )
}

// ---------------------------------------------------------------- svd / svdvals

fn svd_body(a: TensorRead<'_>, o: &SvdOptions, e: &mut dyn Engine) -> Res<Svd<Tensor>> {
    let (u, s, vh) = match o.shape {
        SvdShape::Thin if *o == SvdOptions::default() => route(
            a,
            e,
            |t, e| OwnedLinalg::svd(t, e),
            |r, e| ReadLinalg::svd_read(r, e),
        ),
        SvdShape::Thin => route(
            a,
            e,
            |t, e| OwnedLinalg::svd_with_options(t, *o, e),
            |r, e| ReadLinalg::svd_with_options_read(r, *o, e),
        ),
        SvdShape::Full => {
            if *o != SvdOptions::default().shape(SvdShape::Full) {
                return Err(Error::unsupported(
                    "svd",
                    "SvdShape::Full does not support a non-default gauge, derivative_eps or driver yet; use SvdShape::Thin or the default options",
                ));
            }
            route(
                a,
                e,
                |t, e| OwnedLinalg::svd_full(t, e),
                |r, e| ReadLinalg::svd_full_read(r, e),
            )
        }
    }?;
    Ok(Svd { u, s, vh })
}

pub(crate) fn svd<X: AsView + ?Sized>(
    a: &X,
    o: &SvdOptions,
    e: &mut dyn Engine,
) -> Res<Svd<Val<X>, Re<X>>>
where
    X::Repr: LinalgRepr,
{
    let r = svd_body(a.as_read(), o, e)?;
    Ok(Svd {
        u: val::<X>(r.u)?,
        s: re::<X>(r.s)?,
        vh: val::<X>(r.vh)?,
    })
}

fn svdvals_body(a: TensorRead<'_>, o: &SvdvalsOptions, e: &mut dyn Engine) -> Res<Tensor> {
    if o.driver == SvdDriver::Auto {
        return route(
            a,
            e,
            |t, e| OwnedLinalg::svdvals(t, e),
            |r, e| ReadLinalg::svdvals_read(r, e),
        );
    }
    if let Some(result) = lane::try_lane_svdvals(e, a.clone()) {
        return result;
    }
    with_linalg_backend(e, "svdvals", |backend| {
        backend.svd_values_with_driver_read(a, o.driver)
    })
}

pub(crate) fn svdvals<X: AsView + ?Sized>(
    a: &X,
    o: &SvdvalsOptions,
    e: &mut dyn Engine,
) -> Res<Re<X>>
where
    X::Repr: LinalgRepr,
{
    re::<X>(svdvals_body(a.as_read(), o, e)?)
}

// ------------------------------------------------------------------------- qr

fn qr_body(a: TensorRead<'_>, o: &QrOptions, e: &mut dyn Engine) -> Res<Qr<Tensor>> {
    let (q, r) = if *o == QrOptions::default() {
        route(
            a,
            e,
            |t, e| OwnedLinalg::qr(t, e),
            |r, e| ReadLinalg::qr_read(r, e),
        )
    } else {
        route(
            a,
            e,
            |t, e| OwnedLinalg::qr_with_options(t, *o, e),
            |r, e| ReadLinalg::qr_with_options_read(r, *o, e),
        )
    }?;
    Ok(Qr { q, r })
}

pub(crate) fn qr<X: AsView + ?Sized>(a: &X, o: &QrOptions, e: &mut dyn Engine) -> Res<Qr<Val<X>>>
where
    X::Repr: LinalgRepr,
{
    let r = qr_body(a.as_read(), o, e)?;
    Ok(Qr {
        q: val::<X>(r.q)?,
        r: val::<X>(r.r)?,
    })
}

fn householder_qr_body(a: TensorRead<'_>, e: &mut dyn Engine) -> Res<HouseholderQr<Tensor>> {
    route(
        a,
        e,
        |t, e| OwnedLinalg::householder_qr(t, e),
        |r, e| crate::tensor_ext::householder_qr_read(r, e),
    )
}

pub(crate) fn householder_qr<X: AsView + ?Sized>(
    a: &X,
    e: &mut dyn Engine,
) -> Res<HouseholderQr<Val<X>>>
where
    X::Repr: LinalgRepr,
{
    let r = householder_qr_body(a.as_read(), e)?;
    Ok(HouseholderQr {
        packed: val::<X>(r.packed)?,
        coeff: val::<X>(r.coeff)?,
    })
}

fn rank_revealing_qr_body(
    a: TensorRead<'_>,
    o: &RankRevealingQrOptions,
    e: &mut dyn Engine,
) -> Res<RankRevealingQrResult<Tensor>> {
    route(
        a,
        e,
        |t, e| OwnedLinalg::rank_revealing_qr(t, *o, e),
        |r, e| ReadLinalg::rank_revealing_qr_read(r, *o, e),
    )
}

pub(crate) fn rank_revealing_qr<X: AsView + ?Sized>(
    a: &X,
    o: &RankRevealingQrOptions,
    e: &mut dyn Engine,
) -> Res<RankRevealingQrResult<Val<X>, Ix<X>>>
where
    X::Repr: LinalgRepr,
{
    let r = rank_revealing_qr_body(a.as_read(), o, e)?;
    Ok(RankRevealingQrResult {
        q: val::<X>(r.q)?,
        r: val::<X>(r.r)?,
        column_permutation: ix::<X>(r.column_permutation)?,
        rank: ix::<X>(r.rank)?,
    })
}

// ------------------------------------------------------------------------- lu

fn lu_body(a: TensorRead<'_>, e: &mut dyn Engine) -> Res<Lu<Tensor>> {
    let (p, l, u, parity) = route(
        a,
        e,
        |t, e| OwnedLinalg::lu(t, e),
        |r, e| ReadLinalg::lu_read(r, e),
    )?;
    Ok(Lu { p, l, u, parity })
}

pub(crate) fn lu<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Lu<Val<X>>>
where
    X::Repr: LinalgRepr,
{
    let r = lu_body(a.as_read(), e)?;
    Ok(Lu {
        p: val::<X>(r.p)?,
        l: val::<X>(r.l)?,
        u: val::<X>(r.u)?,
        parity: val::<X>(r.parity)?,
    })
}

fn full_piv_lu_body(a: TensorRead<'_>, e: &mut dyn Engine) -> Res<FullPivLu<Tensor>> {
    let (p, l, u, q, parity) = route(
        a,
        e,
        |t, e| OwnedLinalg::full_piv_lu(t, e),
        |r, e| ReadLinalg::full_piv_lu_read(r, e),
    )?;
    Ok(FullPivLu { p, l, u, q, parity })
}

pub(crate) fn full_piv_lu<X: AsView + ?Sized>(
    a: &X,
    e: &mut dyn Engine,
) -> Res<FullPivLu<Val<X>, Re<X>>>
where
    X::Repr: LinalgRepr,
{
    let r = full_piv_lu_body(a.as_read(), e)?;
    // The owned route returns `parity` in the real dtype, the borrowed route
    // (which every typed input takes) in the input dtype for complex inputs
    // (#2552 Q21). A typed complex call cannot be given a single static
    // parity type until the routes agree, so it is a declared omission.
    let parity = match re::<X>(r.parity) {
        Ok(parity) => parity,
        Err(_) if is_complex(&a.as_read()) => {
            return Err(Error::unsupported(
                "full_piv_lu",
                "a typed Complex32/Complex64 full_piv_lu is not supported yet: the borrowed route returns parity in the input dtype while the result type declares it real (#2552 Q21); call it on an owned `Tensor`",
            ));
        }
        Err(err) => return Err(err),
    };
    Ok(FullPivLu {
        p: val::<X>(r.p)?,
        l: val::<X>(r.l)?,
        u: val::<X>(r.u)?,
        q: val::<X>(r.q)?,
        parity,
    })
}

// ---------------------------------------------------------------------- solves

fn full_piv_lu_solve_body(a: TensorRead<'_>, b: TensorRead<'_>, e: &mut dyn Engine) -> Res<Tensor> {
    route2(
        a,
        b,
        e,
        |ta, tb, e| OwnedLinalg::full_piv_lu_solve(ta, tb, e),
        |ra, rb, e| ReadLinalg::full_piv_lu_solve_read(ra, rb, e),
    )
}

pub(crate) fn full_piv_lu_solve<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: LinalgRepr,
{
    val::<X>(full_piv_lu_solve_body(a.as_read(), b.as_read(), e)?)
}

fn solve_body(a: TensorRead<'_>, b: TensorRead<'_>, e: &mut dyn Engine) -> Res<Tensor> {
    route2(
        a,
        b,
        e,
        |ta, tb, e| OwnedLinalg::solve(ta, tb, e),
        |ra, rb, e| ReadLinalg::solve_read(ra, rb, e),
    )
}

pub(crate) fn solve<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: LinalgRepr,
{
    val::<X>(solve_body(a.as_read(), b.as_read(), e)?)
}

pub(crate) fn solve_into<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    e: &mut dyn Engine,
    dest: Dest<<X::Repr as Repr>::Write<'_>>,
) -> Res<()>
where
    X::Repr: LinalgRepr,
{
    // Rejected before `dest` is touched.
    match dest.mode() {
        DestMode::Overwrite => {}
        mode => {
            return Err(Error::unsupported(
                "solve_into",
                format!("destination mode {mode:?} is not supported; solve_into overwrites its destination"),
            ));
        }
    }
    let out = <X::Repr as Repr>::erase_write(dest.into_target());
    ReadLinalg::solve_read_into(a.as_read(), b.as_read(), out, e)
}

fn lstsq_body(a: TensorRead<'_>, b: TensorRead<'_>, e: &mut dyn Engine) -> Res<Tensor> {
    crate::tensor_ext::lstsq_read(a, b, e)
}

pub(crate) fn lstsq<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: LinalgRepr,
{
    val::<X>(lstsq_body(a.as_read(), b.as_read(), e)?)
}

/// The former `(left_side, lower, transpose_a, unit_diagonal)` flags of
/// `o`, or `Unsupported` for an adjoint of a complex matrix.
pub(crate) fn triangular_flags(
    o: &TriangularSolveOptions,
    complex: bool,
) -> Res<(bool, bool, bool, bool)> {
    let transpose_a = match o.op {
        MatOp::None => false,
        MatOp::Transpose => true,
        MatOp::Adjoint if complex => {
            return Err(Error::unsupported(
                "triangular_solve",
                "MatOp::Adjoint of a complex matrix is not implemented; use MatOp::Transpose on a conjugated matrix",
            ));
        }
        MatOp::Adjoint => true,
    };
    Ok((
        o.side == Side::Left,
        o.uplo == Uplo::Lower,
        transpose_a,
        o.diag == Diag::Unit,
    ))
}

fn triangular_solve_body(
    a: TensorRead<'_>,
    b: TensorRead<'_>,
    o: &TriangularSolveOptions,
    e: &mut dyn Engine,
) -> Res<Tensor> {
    let (left, lower, transpose, unit) = triangular_flags(o, is_complex(&a))?;
    route2(
        a,
        b,
        e,
        |ta, tb, e| OwnedLinalg::triangular_solve(ta, tb, left, lower, transpose, unit, e),
        |ra, rb, e| ReadLinalg::triangular_solve_read(ra, rb, left, lower, transpose, unit, e),
    )
}

pub(crate) fn triangular_solve<X: AsView + ?Sized>(
    a: &X,
    b: &impl AsView<Repr = X::Repr>,
    o: &TriangularSolveOptions,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: LinalgRepr,
{
    val::<X>(triangular_solve_body(a.as_read(), b.as_read(), o, e)?)
}

pub(crate) fn require_lower(o: &CholeskyOptions) -> Res<()> {
    match o.uplo {
        Uplo::Lower => Ok(()),
        Uplo::Upper => Err(Error::unsupported(
            "cholesky",
            "Uplo::Upper is not implemented; the factorization returns the lower factor",
        )),
    }
}

fn cholesky_body(a: TensorRead<'_>, o: &CholeskyOptions, e: &mut dyn Engine) -> Res<Tensor> {
    require_lower(o)?;
    route(
        a,
        e,
        |t, e| OwnedLinalg::cholesky(t, e),
        |r, e| ReadLinalg::cholesky_read(r, e),
    )
}

pub(crate) fn cholesky<X: AsView + ?Sized>(
    a: &X,
    o: &CholeskyOptions,
    e: &mut dyn Engine,
) -> Res<Val<X>>
where
    X::Repr: LinalgRepr,
{
    val::<X>(cholesky_body(a.as_read(), o, e)?)
}

// ------------------------------------------------------------------ eigensolvers

fn eig_body(a: TensorRead<'_>, e: &mut dyn Engine) -> Res<Eig<Tensor>> {
    let (values, vectors) = route(
        a,
        e,
        |t, e| OwnedLinalg::eig(t, e),
        |r, e| ReadLinalg::eig_read(r, e),
    )?;
    Ok(Eig { values, vectors })
}

pub(crate) fn eig<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Eig<Cx<X>>>
where
    X::Repr: LinalgRepr,
{
    let r = eig_body(a.as_read(), e)?;
    Ok(Eig {
        values: cx::<X>(r.values)?,
        vectors: cx::<X>(r.vectors)?,
    })
}

fn eigvals_body(a: TensorRead<'_>, e: &mut dyn Engine) -> Res<Tensor> {
    route(
        a,
        e,
        |t, e| OwnedLinalg::eigvals(t, e),
        |r, e| ReadLinalg::eigvals_read(r, e),
    )
}

pub(crate) fn eigvals<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Cx<X>>
where
    X::Repr: LinalgRepr,
{
    cx::<X>(eigvals_body(a.as_read(), e)?)
}

fn eigh_body(a: TensorRead<'_>, o: &EighOptions, e: &mut dyn Engine) -> Res<Eigh<Tensor>> {
    let (values, vectors) = if *o == EighOptions::default() {
        route(
            a,
            e,
            |t, e| OwnedLinalg::eigh(t, e),
            |r, e| ReadLinalg::eigh_read(r, e),
        )
    } else {
        route(
            a,
            e,
            |t, e| OwnedLinalg::eigh_with_options(t, *o, e),
            |r, e| ReadLinalg::eigh_with_options_read(r, *o, e),
        )
    }?;
    Ok(Eigh { values, vectors })
}

pub(crate) fn eigh<X: AsView + ?Sized>(
    a: &X,
    o: &EighOptions,
    e: &mut dyn Engine,
) -> Res<Eigh<Val<X>, Re<X>>>
where
    X::Repr: LinalgRepr,
{
    let r = eigh_body(a.as_read(), o, e)?;
    Ok(Eigh {
        values: re::<X>(r.values)?,
        vectors: val::<X>(r.vectors)?,
    })
}

fn eigvalsh_body(a: TensorRead<'_>, o: &EigvalshOptions, e: &mut dyn Engine) -> Res<Tensor> {
    if o.driver == EighDriver::Auto {
        return route(
            a,
            e,
            |t, e| OwnedLinalg::eigvalsh(t, e),
            |r, e| ReadLinalg::eigvalsh_read(r, e),
        );
    }
    if let Some(result) = lane::try_lane_eigvalsh(e, a.clone()) {
        return result;
    }
    with_linalg_backend(e, "eigvalsh", |backend| {
        backend.eigh_values_with_driver_read(a, o.driver)
    })
}

pub(crate) fn eigvalsh<X: AsView + ?Sized>(
    a: &X,
    o: &EigvalshOptions,
    e: &mut dyn Engine,
) -> Res<Re<X>>
where
    X::Repr: LinalgRepr,
{
    re::<X>(eigvalsh_body(a.as_read(), o, e)?)
}

// ------------------------------------------------------------ determinant family

fn slogdet_body(a: TensorRead<'_>, e: &mut dyn Engine) -> Res<Slogdet<Tensor>> {
    let (sign, log_abs_det) = route(
        a,
        e,
        |t, e| OwnedLinalg::slogdet(t, e),
        |r, e| ReadLinalg::slogdet_read(r, e),
    )?;
    Ok(Slogdet { sign, log_abs_det })
}

pub(crate) fn slogdet<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Slogdet<Val<X>, Re<X>>>
where
    X::Repr: LinalgRepr,
{
    let r = slogdet_body(a.as_read(), e)?;
    Ok(Slogdet {
        sign: val::<X>(r.sign)?,
        log_abs_det: re::<X>(r.log_abs_det)?,
    })
}

fn det_body(a: TensorRead<'_>, e: &mut dyn Engine) -> Res<Tensor> {
    route(
        a,
        e,
        |t, e| OwnedLinalg::det(t, e),
        |r, e| ReadLinalg::det_read(r, e),
    )
}

pub(crate) fn det<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: LinalgRepr,
{
    val::<X>(det_body(a.as_read(), e)?)
}

fn inv_body(a: TensorRead<'_>, e: &mut dyn Engine) -> Res<Tensor> {
    route(
        a,
        e,
        |t, e| OwnedLinalg::inv(t, e),
        |r, e| ReadLinalg::inv_read(r, e),
    )
}

pub(crate) fn inv<X: AsView + ?Sized>(a: &X, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: LinalgRepr,
{
    val::<X>(inv_body(a.as_read(), e)?)
}

fn pinv_body(a: TensorRead<'_>, o: &PinvOptions, e: &mut dyn Engine) -> Res<Tensor> {
    match o.rtol {
        None => route(
            a,
            e,
            |t, e| OwnedLinalg::pinv(t, e),
            |r, e| ReadLinalg::pinv_read(r, e),
        ),
        Some(rtol) => route(
            a,
            e,
            |t, e| OwnedLinalg::pinv_with_rtol(t, rtol, e),
            |r, e| ReadLinalg::pinv_with_rtol_read(r, rtol, e),
        ),
    }
}

pub(crate) fn pinv<X: AsView + ?Sized>(a: &X, o: &PinvOptions, e: &mut dyn Engine) -> Res<Val<X>>
where
    X::Repr: LinalgRepr,
{
    val::<X>(pinv_body(a.as_read(), o, e)?)
}

fn norm_body(a: TensorRead<'_>, o: &NormOptions, e: &mut dyn Engine) -> Res<Tensor> {
    let ord = o.ord.to_ord()?;
    let dim = o.unsigned_axes(a.shape().len())?;
    let keepdim = o.keep_dims == KeepDims::Yes;
    route(
        a,
        e,
        |t, e| OwnedLinalg::norm(t, ord, dim.as_deref(), keepdim, e),
        |r, e| ReadLinalg::norm_read(r, ord, dim.as_deref(), keepdim, e),
    )
}

pub(crate) fn norm<X: AsView + ?Sized>(a: &X, o: &NormOptions, e: &mut dyn Engine) -> Res<Re<X>>
where
    X::Repr: LinalgRepr,
{
    re::<X>(norm_body(a.as_read(), o, e)?)
}
