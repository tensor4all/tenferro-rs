//! One forwarding case per (operation, layer): the plain call and the call
//! with default options must agree bit for bit.
//!
//! The list is checked against `API_OPS` by
//! `every_registry_entry_has_a_forwarding_case`, so a new registry entry
//! without a case fails.

use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::TraceContext;

use super::support::*;
#[cfg(feature = "autodiff")]
use crate::EagerLinalgOps;
use crate::{
    CholeskyOptions, DetOptions, EigOptions, EighOptions, EigvalsOptions, EigvalshOptions,
    FullPivLuOptions, FullPivLuSolveOptions, HouseholderQrOptions, InvOptions, LinalgOps,
    LstsqOptions, LuOptions, NormOptions, PinvOptions, QrOptions, RankRevealingQrOptions,
    SlogdetOptions, SolveOptions, SvdOptions, SvdvalsOptions, TracedLinalgOps,
    TriangularSolveOptions,
};

/// A forwarding case for one operation on one layer; `run` holds the
/// instantiation for `f64` and for `Complex64`.
pub(super) struct Case {
    pub op: &'static str,
    pub layer: &'static str,
    pub run: [fn() -> TestResult<Pair>; 2],
}

fn concrete_svd<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(tall::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.svd(&mut e)?;
    let w = a.svd_with(&SvdOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_svd<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = tall::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.svd()?;
    let w = a.svd_with(&SvdOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_svd<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(tall::<T>())?)?;
        let p = s.svd(&a)?;
        let w = s.svd_with(&a, &SvdOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_svdvals<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(tall::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.svdvals(&mut e)?;
    let w = a.svdvals_with(&SvdvalsOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_svdvals<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = tall::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.svdvals()?;
    let w = a.svdvals_with(&SvdvalsOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_svdvals<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(tall::<T>())?)?;
        let p = s.svdvals(&a)?;
        let w = s.svdvals_with(&a, &SvdvalsOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_qr<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(tall::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.qr(&mut e)?;
    let w = a.qr_with(&QrOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_qr<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = tall::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.qr()?;
    let w = a.qr_with(&QrOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_qr<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(tall::<T>())?)?;
        let p = s.qr(&a)?;
        let w = s.qr_with(&a, &QrOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_householder_qr<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(tall::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.householder_qr(&mut e)?;
    let w = a.householder_qr_with(&HouseholderQrOptions::default(), &mut e)?;
    let rp = p.r(QrOptions::default(), &mut e)?;
    let qp = p.q_columns(0..2, QrOptions::default(), &mut e)?;
    let rw = w.r(QrOptions::default(), &mut e)?;
    let qw = w.q_columns(0..2, QrOptions::default(), &mut e)?;
    Ok((
        vec![tensor_bits(&rp)?, tensor_bits(&qp)?],
        vec![tensor_bits(&rw)?, tensor_bits(&qw)?],
    ))
}

fn traced_householder_qr<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = tall::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.householder_qr()?;
    let w = a.householder_qr_with(&HouseholderQrOptions::default())?;
    traced_pair(&[p.r(QrOptions::default())?], &[w.r(QrOptions::default())?])
}

#[cfg(feature = "autodiff")]
fn eager_householder_qr<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(tall::<T>())?)?;
        let p = s.householder_qr(&a)?;
        let w = s.householder_qr_with(&a, &HouseholderQrOptions::default())?;
        let rp = p.r(QrOptions::default(), s)?;
        let rw = w.r(QrOptions::default(), s)?;
        Ok::<_, tenferro_ad::Error>((rp, rw))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_rank_revealing_qr<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(tall::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.rank_revealing_qr(&mut e)?;
    let w = a.rank_revealing_qr_with(&RankRevealingQrOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_rank_revealing_qr<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = tall::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.rank_revealing_qr()?;
    let w = a.rank_revealing_qr_with(&RankRevealingQrOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_rank_revealing_qr<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(tall::<T>())?)?;
        let p = s.rank_revealing_qr(&a)?;
        let w = s.rank_revealing_qr_with(&a, &RankRevealingQrOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_lu<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.lu(&mut e)?;
    let w = a.lu_with(&LuOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_lu<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.lu()?;
    let w = a.lu_with(&LuOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_lu<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let p = s.lu(&a)?;
        let w = s.lu_with(&a, &LuOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_full_piv_lu<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.full_piv_lu(&mut e)?;
    let w = a.full_piv_lu_with(&FullPivLuOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_full_piv_lu<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.full_piv_lu()?;
    let w = a.full_piv_lu_with(&FullPivLuOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_full_piv_lu<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let p = s.full_piv_lu(&a)?;
        let w = s.full_piv_lu_with(&a, &FullPivLuOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_full_piv_lu_solve<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let b = tensor::<T>(rhs::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.full_piv_lu_solve(&b, &mut e)?;
    let w = a.full_piv_lu_solve_with(&b, &FullPivLuSolveOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_full_piv_lu_solve<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let (shape, values) = rhs::<T>();
    let b = trace.from_vec_col_major(shape, values)?;
    let p = a.full_piv_lu_solve(&b)?;
    let w = a.full_piv_lu_solve_with(&b, &FullPivLuSolveOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_full_piv_lu_solve<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let b = s.constant_from(raw::<T>(rhs::<T>())?)?;
        let p = s.full_piv_lu_solve(&a, &b)?;
        let w = s.full_piv_lu_solve_with(&a, &b, &FullPivLuSolveOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_solve<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let b = tensor::<T>(rhs::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.solve(&b, &mut e)?;
    let w = a.solve_with(&b, &SolveOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_solve<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let (shape, values) = rhs::<T>();
    let b = trace.from_vec_col_major(shape, values)?;
    let p = a.solve(&b)?;
    let w = a.solve_with(&b, &SolveOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_solve<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let b = s.constant_from(raw::<T>(rhs::<T>())?)?;
        let p = s.solve(&a, &b)?;
        let w = s.solve_with(&a, &b, &SolveOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_lstsq<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(tall::<T>())?;
    let b = tensor::<T>(rhs_tall::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.lstsq(&b, &mut e)?;
    let w = a.lstsq_with(&b, &LstsqOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_lstsq<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = tall::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let (shape, values) = rhs_tall::<T>();
    let b = trace.from_vec_col_major(shape, values)?;
    let p = a.lstsq(&b)?;
    let w = a.lstsq_with(&b, &LstsqOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_lstsq<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(tall::<T>())?)?;
        let b = s.constant_from(raw::<T>(rhs_tall::<T>())?)?;
        let p = s.lstsq(&a, &b)?;
        let w = s.lstsq_with(&a, &b, &LstsqOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_triangular_solve<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let b = tensor::<T>(rhs::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.triangular_solve(&b, &mut e)?;
    let w = a.triangular_solve_with(&b, &TriangularSolveOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_triangular_solve<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let (shape, values) = rhs::<T>();
    let b = trace.from_vec_col_major(shape, values)?;
    let p = a.triangular_solve(&b)?;
    let w = a.triangular_solve_with(&b, &TriangularSolveOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_triangular_solve<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let b = s.constant_from(raw::<T>(rhs::<T>())?)?;
        let p = s.triangular_solve(&a, &b)?;
        let w = s.triangular_solve_with(&a, &b, &TriangularSolveOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_cholesky<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.cholesky(&mut e)?;
    let w = a.cholesky_with(&CholeskyOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_cholesky<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.cholesky()?;
    let w = a.cholesky_with(&CholeskyOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_cholesky<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let p = s.cholesky(&a)?;
        let w = s.cholesky_with(&a, &CholeskyOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_eig<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.eig(&mut e)?;
    let w = a.eig_with(&EigOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_eig<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.eig()?;
    let w = a.eig_with(&EigOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_eig<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let p = s.eig(&a)?;
        let w = s.eig_with(&a, &EigOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_eigvals<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.eigvals(&mut e)?;
    let w = a.eigvals_with(&EigvalsOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_eigvals<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.eigvals()?;
    let w = a.eigvals_with(&EigvalsOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_eigvals<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let p = s.eigvals(&a)?;
        let w = s.eigvals_with(&a, &EigvalsOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_eigh<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.eigh(&mut e)?;
    let w = a.eigh_with(&EighOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_eigh<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.eigh()?;
    let w = a.eigh_with(&EighOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_eigh<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let p = s.eigh(&a)?;
        let w = s.eigh_with(&a, &EighOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_eigvalsh<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.eigvalsh(&mut e)?;
    let w = a.eigvalsh_with(&EigvalshOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_eigvalsh<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.eigvalsh()?;
    let w = a.eigvalsh_with(&EigvalshOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_eigvalsh<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let p = s.eigvalsh(&a)?;
        let w = s.eigvalsh_with(&a, &EigvalshOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_slogdet<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.slogdet(&mut e)?;
    let w = a.slogdet_with(&SlogdetOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_slogdet<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.slogdet()?;
    let w = a.slogdet_with(&SlogdetOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_slogdet<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let p = s.slogdet(&a)?;
        let w = s.slogdet_with(&a, &SlogdetOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_det<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.det(&mut e)?;
    let w = a.det_with(&DetOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_det<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.det()?;
    let w = a.det_with(&DetOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_det<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let p = s.det(&a)?;
        let w = s.det_with(&a, &DetOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_inv<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.inv(&mut e)?;
    let w = a.inv_with(&InvOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_inv<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.inv()?;
    let w = a.inv_with(&InvOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_inv<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let p = s.inv(&a)?;
        let w = s.inv_with(&a, &InvOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_pinv<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(tall::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.pinv(&mut e)?;
    let w = a.pinv_with(&PinvOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_pinv<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = tall::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.pinv()?;
    let w = a.pinv_with(&PinvOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_pinv<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(tall::<T>())?)?;
        let p = s.pinv(&a)?;
        let w = s.pinv_with(&a, &PinvOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

fn concrete_norm<T: Sample>() -> TestResult<Pair> {
    let a = tensor::<T>(sq::<T>())?;
    let mut e = CpuEngine::serial();
    let p = a.norm(&mut e)?;
    let w = a.norm_with(&NormOptions::default(), &mut e)?;
    Ok((p.bits()?, w.bits()?))
}

fn traced_norm<T: Sample>() -> TestResult<Pair> {
    let trace = TraceContext::new();
    let (shape, values) = sq::<T>();
    let a = trace.from_vec_col_major(shape, values)?;
    let p = a.norm()?;
    let w = a.norm_with(&NormOptions::default())?;
    traced_pair(&p.leaves(), &w.leaves())
}

#[cfg(feature = "autodiff")]
fn eager_norm<T: Sample>() -> TestResult<Pair> {
    let runtime = tenferro_ad::EagerRuntime::new()?;
    let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(raw::<T>(sq::<T>())?)?;
        let p = s.norm(&a)?;
        let w = s.norm_with(&a, &NormOptions::default())?;
        Ok::<_, tenferro_ad::Error>((p, w))
    })?;
    Ok((p.bits()?, w.bits()?))
}

/// Every forwarding case.
pub(super) fn all() -> Vec<Case> {
    // Extended with the eager cases when `autodiff` is on.
    #[cfg_attr(not(feature = "autodiff"), allow(unused_mut))]
    let mut cases = vec![
        Case {
            op: "svd",
            layer: "concrete",
            run: [concrete_svd::<f64>, concrete_svd::<Complex64>],
        },
        Case {
            op: "svd",
            layer: "traced",
            run: [traced_svd::<f64>, traced_svd::<Complex64>],
        },
        Case {
            op: "svdvals",
            layer: "concrete",
            run: [concrete_svdvals::<f64>, concrete_svdvals::<Complex64>],
        },
        Case {
            op: "svdvals",
            layer: "traced",
            run: [traced_svdvals::<f64>, traced_svdvals::<Complex64>],
        },
        Case {
            op: "qr",
            layer: "concrete",
            run: [concrete_qr::<f64>, concrete_qr::<Complex64>],
        },
        Case {
            op: "qr",
            layer: "traced",
            run: [traced_qr::<f64>, traced_qr::<Complex64>],
        },
        Case {
            op: "householder_qr",
            layer: "concrete",
            run: [
                concrete_householder_qr::<f64>,
                concrete_householder_qr::<Complex64>,
            ],
        },
        Case {
            op: "householder_qr",
            layer: "traced",
            run: [
                traced_householder_qr::<f64>,
                traced_householder_qr::<Complex64>,
            ],
        },
        Case {
            op: "rank_revealing_qr",
            layer: "concrete",
            run: [
                concrete_rank_revealing_qr::<f64>,
                concrete_rank_revealing_qr::<Complex64>,
            ],
        },
        Case {
            op: "rank_revealing_qr",
            layer: "traced",
            run: [
                traced_rank_revealing_qr::<f64>,
                traced_rank_revealing_qr::<Complex64>,
            ],
        },
        Case {
            op: "lu",
            layer: "concrete",
            run: [concrete_lu::<f64>, concrete_lu::<Complex64>],
        },
        Case {
            op: "lu",
            layer: "traced",
            run: [traced_lu::<f64>, traced_lu::<Complex64>],
        },
        Case {
            op: "full_piv_lu",
            layer: "concrete",
            run: [
                concrete_full_piv_lu::<f64>,
                concrete_full_piv_lu::<Complex64>,
            ],
        },
        Case {
            op: "full_piv_lu",
            layer: "traced",
            run: [traced_full_piv_lu::<f64>, traced_full_piv_lu::<Complex64>],
        },
        Case {
            op: "full_piv_lu_solve",
            layer: "concrete",
            run: [
                concrete_full_piv_lu_solve::<f64>,
                concrete_full_piv_lu_solve::<Complex64>,
            ],
        },
        Case {
            op: "full_piv_lu_solve",
            layer: "traced",
            run: [
                traced_full_piv_lu_solve::<f64>,
                traced_full_piv_lu_solve::<Complex64>,
            ],
        },
        Case {
            op: "solve",
            layer: "concrete",
            run: [concrete_solve::<f64>, concrete_solve::<Complex64>],
        },
        Case {
            op: "solve",
            layer: "traced",
            run: [traced_solve::<f64>, traced_solve::<Complex64>],
        },
        Case {
            op: "lstsq",
            layer: "concrete",
            run: [concrete_lstsq::<f64>, concrete_lstsq::<Complex64>],
        },
        Case {
            op: "lstsq",
            layer: "traced",
            run: [traced_lstsq::<f64>, traced_lstsq::<Complex64>],
        },
        Case {
            op: "triangular_solve",
            layer: "concrete",
            run: [
                concrete_triangular_solve::<f64>,
                concrete_triangular_solve::<Complex64>,
            ],
        },
        Case {
            op: "triangular_solve",
            layer: "traced",
            run: [
                traced_triangular_solve::<f64>,
                traced_triangular_solve::<Complex64>,
            ],
        },
        Case {
            op: "cholesky",
            layer: "concrete",
            run: [concrete_cholesky::<f64>, concrete_cholesky::<Complex64>],
        },
        Case {
            op: "cholesky",
            layer: "traced",
            run: [traced_cholesky::<f64>, traced_cholesky::<Complex64>],
        },
        Case {
            op: "eig",
            layer: "concrete",
            run: [concrete_eig::<f64>, concrete_eig::<Complex64>],
        },
        Case {
            op: "eig",
            layer: "traced",
            run: [traced_eig::<f64>, traced_eig::<Complex64>],
        },
        Case {
            op: "eigvals",
            layer: "concrete",
            run: [concrete_eigvals::<f64>, concrete_eigvals::<Complex64>],
        },
        Case {
            op: "eigvals",
            layer: "traced",
            run: [traced_eigvals::<f64>, traced_eigvals::<Complex64>],
        },
        Case {
            op: "eigh",
            layer: "concrete",
            run: [concrete_eigh::<f64>, concrete_eigh::<Complex64>],
        },
        Case {
            op: "eigh",
            layer: "traced",
            run: [traced_eigh::<f64>, traced_eigh::<Complex64>],
        },
        Case {
            op: "eigvalsh",
            layer: "concrete",
            run: [concrete_eigvalsh::<f64>, concrete_eigvalsh::<Complex64>],
        },
        Case {
            op: "eigvalsh",
            layer: "traced",
            run: [traced_eigvalsh::<f64>, traced_eigvalsh::<Complex64>],
        },
        Case {
            op: "slogdet",
            layer: "concrete",
            run: [concrete_slogdet::<f64>, concrete_slogdet::<Complex64>],
        },
        Case {
            op: "slogdet",
            layer: "traced",
            run: [traced_slogdet::<f64>, traced_slogdet::<Complex64>],
        },
        Case {
            op: "det",
            layer: "concrete",
            run: [concrete_det::<f64>, concrete_det::<Complex64>],
        },
        Case {
            op: "det",
            layer: "traced",
            run: [traced_det::<f64>, traced_det::<Complex64>],
        },
        Case {
            op: "inv",
            layer: "concrete",
            run: [concrete_inv::<f64>, concrete_inv::<Complex64>],
        },
        Case {
            op: "inv",
            layer: "traced",
            run: [traced_inv::<f64>, traced_inv::<Complex64>],
        },
        Case {
            op: "pinv",
            layer: "concrete",
            run: [concrete_pinv::<f64>, concrete_pinv::<Complex64>],
        },
        Case {
            op: "pinv",
            layer: "traced",
            run: [traced_pinv::<f64>, traced_pinv::<Complex64>],
        },
        Case {
            op: "norm",
            layer: "concrete",
            run: [concrete_norm::<f64>, concrete_norm::<Complex64>],
        },
        Case {
            op: "norm",
            layer: "traced",
            run: [traced_norm::<f64>, traced_norm::<Complex64>],
        },
    ];
    #[cfg(feature = "autodiff")]
    cases.extend([
        Case {
            op: "svd",
            layer: "eager",
            run: [eager_svd::<f64>, eager_svd::<Complex64>],
        },
        Case {
            op: "svdvals",
            layer: "eager",
            run: [eager_svdvals::<f64>, eager_svdvals::<Complex64>],
        },
        Case {
            op: "qr",
            layer: "eager",
            run: [eager_qr::<f64>, eager_qr::<Complex64>],
        },
        Case {
            op: "householder_qr",
            layer: "eager",
            run: [
                eager_householder_qr::<f64>,
                eager_householder_qr::<Complex64>,
            ],
        },
        Case {
            op: "rank_revealing_qr",
            layer: "eager",
            run: [
                eager_rank_revealing_qr::<f64>,
                eager_rank_revealing_qr::<Complex64>,
            ],
        },
        Case {
            op: "lu",
            layer: "eager",
            run: [eager_lu::<f64>, eager_lu::<Complex64>],
        },
        Case {
            op: "full_piv_lu",
            layer: "eager",
            run: [eager_full_piv_lu::<f64>, eager_full_piv_lu::<Complex64>],
        },
        Case {
            op: "full_piv_lu_solve",
            layer: "eager",
            run: [
                eager_full_piv_lu_solve::<f64>,
                eager_full_piv_lu_solve::<Complex64>,
            ],
        },
        Case {
            op: "solve",
            layer: "eager",
            run: [eager_solve::<f64>, eager_solve::<Complex64>],
        },
        Case {
            op: "lstsq",
            layer: "eager",
            run: [eager_lstsq::<f64>, eager_lstsq::<Complex64>],
        },
        Case {
            op: "triangular_solve",
            layer: "eager",
            run: [
                eager_triangular_solve::<f64>,
                eager_triangular_solve::<Complex64>,
            ],
        },
        Case {
            op: "cholesky",
            layer: "eager",
            run: [eager_cholesky::<f64>, eager_cholesky::<Complex64>],
        },
        Case {
            op: "eig",
            layer: "eager",
            run: [eager_eig::<f64>, eager_eig::<Complex64>],
        },
        Case {
            op: "eigvals",
            layer: "eager",
            run: [eager_eigvals::<f64>, eager_eigvals::<Complex64>],
        },
        Case {
            op: "eigh",
            layer: "eager",
            run: [eager_eigh::<f64>, eager_eigh::<Complex64>],
        },
        Case {
            op: "eigvalsh",
            layer: "eager",
            run: [eager_eigvalsh::<f64>, eager_eigvalsh::<Complex64>],
        },
        Case {
            op: "slogdet",
            layer: "eager",
            run: [eager_slogdet::<f64>, eager_slogdet::<Complex64>],
        },
        Case {
            op: "det",
            layer: "eager",
            run: [eager_det::<f64>, eager_det::<Complex64>],
        },
        Case {
            op: "inv",
            layer: "eager",
            run: [eager_inv::<f64>, eager_inv::<Complex64>],
        },
        Case {
            op: "pinv",
            layer: "eager",
            run: [eager_pinv::<f64>, eager_pinv::<Complex64>],
        },
        Case {
            op: "norm",
            layer: "eager",
            run: [eager_norm::<f64>, eager_norm::<Complex64>],
        },
    ]);
    cases
}
