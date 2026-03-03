//! Runtime BLAS/LAPACK function-pointer registration for `provider-inject`.
//!
//! This module is available only with:
//! - `linalg-lapack`
//! - `provider-inject`

use cblas_inject::{CgemmFnPtr, DgemmFnPtr, SgemmFnPtr, ZgemmFnPtr};
use lapack_inject::{
    CgeevFnPtr, CgeqrfFnPtr, CgesvFnPtr, CgesvdFnPtr, CgetrfFnPtr, CheevFnPtr, CpotrfFnPtr,
    CtrtrsFnPtr, CungqrFnPtr, DgeevFnPtr, DgeqrfFnPtr, DgesvFnPtr, DgesvdFnPtr, DgetrfFnPtr,
    DorgqrFnPtr, DpotrfFnPtr, DsyevFnPtr, DtrtrsFnPtr, SgeevFnPtr, SgeqrfFnPtr, SgesvFnPtr,
    SgesvdFnPtr, SgetrfFnPtr, SorgqrFnPtr, SpotrfFnPtr, SsyevFnPtr, StrtrsFnPtr, ZgeevFnPtr,
    ZgeqrfFnPtr, ZgesvFnPtr, ZgesvdFnPtr, ZgetrfFnPtr, ZheevFnPtr, ZpotrfFnPtr, ZtrtrsFnPtr,
    ZungqrFnPtr,
};

/// Set of BLAS/LAPACK function pointers to register in one call.
///
/// Any field set to `None` is skipped, so callers can register only the
/// scalar/routine subset they actually use.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::inject::{BlasLapackFnPtrSet, register_blas_lapack_fn_ptrs};
///
/// unsafe {
///     register_blas_lapack_fn_ptrs(BlasLapackFnPtrSet {
///         dgemm: Some(dgemm_ptr),
///         dgesvd: Some(dgesvd_ptr),
///         ..BlasLapackFnPtrSet::new()
///     });
/// }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BlasLapackFnPtrSet {
    /// Fortran `sgemm` function pointer.
    pub sgemm: Option<SgemmFnPtr>,
    /// Fortran `dgemm` function pointer.
    pub dgemm: Option<DgemmFnPtr>,
    /// Fortran `cgemm` function pointer.
    pub cgemm: Option<CgemmFnPtr>,
    /// Fortran `zgemm` function pointer.
    pub zgemm: Option<ZgemmFnPtr>,

    /// Fortran `sgesvd` function pointer.
    pub sgesvd: Option<SgesvdFnPtr>,
    /// Fortran `dgesvd` function pointer.
    pub dgesvd: Option<DgesvdFnPtr>,
    /// Fortran `cgesvd` function pointer.
    pub cgesvd: Option<CgesvdFnPtr>,
    /// Fortran `zgesvd` function pointer.
    pub zgesvd: Option<ZgesvdFnPtr>,

    /// Fortran `sgeqrf` function pointer.
    pub sgeqrf: Option<SgeqrfFnPtr>,
    /// Fortran `dgeqrf` function pointer.
    pub dgeqrf: Option<DgeqrfFnPtr>,
    /// Fortran `cgeqrf` function pointer.
    pub cgeqrf: Option<CgeqrfFnPtr>,
    /// Fortran `zgeqrf` function pointer.
    pub zgeqrf: Option<ZgeqrfFnPtr>,

    /// Fortran `sorgqr` function pointer.
    pub sorgqr: Option<SorgqrFnPtr>,
    /// Fortran `dorgqr` function pointer.
    pub dorgqr: Option<DorgqrFnPtr>,
    /// Fortran `cungqr` function pointer.
    pub cungqr: Option<CungqrFnPtr>,
    /// Fortran `zungqr` function pointer.
    pub zungqr: Option<ZungqrFnPtr>,

    /// Fortran `sgetrf` function pointer.
    pub sgetrf: Option<SgetrfFnPtr>,
    /// Fortran `dgetrf` function pointer.
    pub dgetrf: Option<DgetrfFnPtr>,
    /// Fortran `cgetrf` function pointer.
    pub cgetrf: Option<CgetrfFnPtr>,
    /// Fortran `zgetrf` function pointer.
    pub zgetrf: Option<ZgetrfFnPtr>,

    /// Fortran `spotrf` function pointer.
    pub spotrf: Option<SpotrfFnPtr>,
    /// Fortran `dpotrf` function pointer.
    pub dpotrf: Option<DpotrfFnPtr>,
    /// Fortran `cpotrf` function pointer.
    pub cpotrf: Option<CpotrfFnPtr>,
    /// Fortran `zpotrf` function pointer.
    pub zpotrf: Option<ZpotrfFnPtr>,

    /// Fortran `ssyev` function pointer.
    pub ssyev: Option<SsyevFnPtr>,
    /// Fortran `dsyev` function pointer.
    pub dsyev: Option<DsyevFnPtr>,
    /// Fortran `cheev` function pointer.
    pub cheev: Option<CheevFnPtr>,
    /// Fortran `zheev` function pointer.
    pub zheev: Option<ZheevFnPtr>,

    /// Fortran `sgesv` function pointer.
    pub sgesv: Option<SgesvFnPtr>,
    /// Fortran `dgesv` function pointer.
    pub dgesv: Option<DgesvFnPtr>,
    /// Fortran `cgesv` function pointer.
    pub cgesv: Option<CgesvFnPtr>,
    /// Fortran `zgesv` function pointer.
    pub zgesv: Option<ZgesvFnPtr>,

    /// Fortran `strtrs` function pointer.
    pub strtrs: Option<StrtrsFnPtr>,
    /// Fortran `dtrtrs` function pointer.
    pub dtrtrs: Option<DtrtrsFnPtr>,
    /// Fortran `ctrtrs` function pointer.
    pub ctrtrs: Option<CtrtrsFnPtr>,
    /// Fortran `ztrtrs` function pointer.
    pub ztrtrs: Option<ZtrtrsFnPtr>,

    /// Fortran `sgeev` function pointer.
    pub sgeev: Option<SgeevFnPtr>,
    /// Fortran `dgeev` function pointer.
    pub dgeev: Option<DgeevFnPtr>,
    /// Fortran `cgeev` function pointer.
    pub cgeev: Option<CgeevFnPtr>,
    /// Fortran `zgeev` function pointer.
    pub zgeev: Option<ZgeevFnPtr>,
}

impl BlasLapackFnPtrSet {
    /// Create an empty set (all pointers are `None`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_linalg::inject::BlasLapackFnPtrSet;
    ///
    /// let ptrs = BlasLapackFnPtrSet::new();
    /// assert!(ptrs.dgemm.is_none());
    /// assert!(ptrs.dgesvd.is_none());
    /// ```
    pub const fn new() -> Self {
        Self {
            sgemm: None,
            dgemm: None,
            cgemm: None,
            zgemm: None,
            sgesvd: None,
            dgesvd: None,
            cgesvd: None,
            zgesvd: None,
            sgeqrf: None,
            dgeqrf: None,
            cgeqrf: None,
            zgeqrf: None,
            sorgqr: None,
            dorgqr: None,
            cungqr: None,
            zungqr: None,
            sgetrf: None,
            dgetrf: None,
            cgetrf: None,
            zgetrf: None,
            spotrf: None,
            dpotrf: None,
            cpotrf: None,
            zpotrf: None,
            ssyev: None,
            dsyev: None,
            cheev: None,
            zheev: None,
            sgesv: None,
            dgesv: None,
            cgesv: None,
            zgesv: None,
            strtrs: None,
            dtrtrs: None,
            ctrtrs: None,
            ztrtrs: None,
            sgeev: None,
            dgeev: None,
            cgeev: None,
            zgeev: None,
        }
    }
}

macro_rules! register_if_some {
    ($ptrs:expr, $field:ident, $register:path) => {
        if let Some(f) = $ptrs.$field {
            unsafe { $register(f) };
        }
    };
}

/// Register BLAS/LAPACK function pointers in one call.
///
/// This is a thin bulk wrapper over `cblas-inject` and `lapack-inject`
/// per-symbol `register_*` functions.
///
/// # Safety
///
/// Each provided function pointer must have the exact Fortran BLAS/LAPACK ABI
/// expected by `cblas-inject` / `lapack-inject`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::inject::{BlasLapackFnPtrSet, register_blas_lapack_fn_ptrs};
///
/// unsafe {
///     register_blas_lapack_fn_ptrs(BlasLapackFnPtrSet {
///         dgemm: Some(dgemm_ptr),
///         dgesvd: Some(dgesvd_ptr),
///         dgeqrf: Some(dgeqrf_ptr),
///         dorgqr: Some(dorgqr_ptr),
///         ..BlasLapackFnPtrSet::new()
///     });
/// }
/// ```
pub unsafe fn register_blas_lapack_fn_ptrs(ptrs: BlasLapackFnPtrSet) {
    register_if_some!(ptrs, sgemm, cblas_inject::register_sgemm);
    register_if_some!(ptrs, dgemm, cblas_inject::register_dgemm);
    register_if_some!(ptrs, cgemm, cblas_inject::register_cgemm);
    register_if_some!(ptrs, zgemm, cblas_inject::register_zgemm);

    register_if_some!(ptrs, sgesvd, lapack_inject::register_sgesvd);
    register_if_some!(ptrs, dgesvd, lapack_inject::register_dgesvd);
    register_if_some!(ptrs, cgesvd, lapack_inject::register_cgesvd);
    register_if_some!(ptrs, zgesvd, lapack_inject::register_zgesvd);

    register_if_some!(ptrs, sgeqrf, lapack_inject::register_sgeqrf);
    register_if_some!(ptrs, dgeqrf, lapack_inject::register_dgeqrf);
    register_if_some!(ptrs, cgeqrf, lapack_inject::register_cgeqrf);
    register_if_some!(ptrs, zgeqrf, lapack_inject::register_zgeqrf);

    register_if_some!(ptrs, sorgqr, lapack_inject::register_sorgqr);
    register_if_some!(ptrs, dorgqr, lapack_inject::register_dorgqr);
    register_if_some!(ptrs, cungqr, lapack_inject::register_cungqr);
    register_if_some!(ptrs, zungqr, lapack_inject::register_zungqr);

    register_if_some!(ptrs, sgetrf, lapack_inject::register_sgetrf);
    register_if_some!(ptrs, dgetrf, lapack_inject::register_dgetrf);
    register_if_some!(ptrs, cgetrf, lapack_inject::register_cgetrf);
    register_if_some!(ptrs, zgetrf, lapack_inject::register_zgetrf);

    register_if_some!(ptrs, spotrf, lapack_inject::register_spotrf);
    register_if_some!(ptrs, dpotrf, lapack_inject::register_dpotrf);
    register_if_some!(ptrs, cpotrf, lapack_inject::register_cpotrf);
    register_if_some!(ptrs, zpotrf, lapack_inject::register_zpotrf);

    register_if_some!(ptrs, ssyev, lapack_inject::register_ssyev);
    register_if_some!(ptrs, dsyev, lapack_inject::register_dsyev);
    register_if_some!(ptrs, cheev, lapack_inject::register_cheev);
    register_if_some!(ptrs, zheev, lapack_inject::register_zheev);

    register_if_some!(ptrs, sgesv, lapack_inject::register_sgesv);
    register_if_some!(ptrs, dgesv, lapack_inject::register_dgesv);
    register_if_some!(ptrs, cgesv, lapack_inject::register_cgesv);
    register_if_some!(ptrs, zgesv, lapack_inject::register_zgesv);

    register_if_some!(ptrs, strtrs, lapack_inject::register_strtrs);
    register_if_some!(ptrs, dtrtrs, lapack_inject::register_dtrtrs);
    register_if_some!(ptrs, ctrtrs, lapack_inject::register_ctrtrs);
    register_if_some!(ptrs, ztrtrs, lapack_inject::register_ztrtrs);

    register_if_some!(ptrs, sgeev, lapack_inject::register_sgeev);
    register_if_some!(ptrs, dgeev, lapack_inject::register_dgeev);
    register_if_some!(ptrs, cgeev, lapack_inject::register_cgeev);
    register_if_some!(ptrs, zgeev, lapack_inject::register_zgeev);
}
