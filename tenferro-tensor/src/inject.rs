//! Runtime BLAS/LAPACK function-pointer registration for `provider-inject`.
//!
//! This module is available only with:
//! - `cpu-blas`
//! - `provider-inject`
//!
//! ## ABI conventions
//!
//! tenferro itself is an LP64 consumer (it calls BLAS/LAPACK through normal
//! `i32` integer arguments).  This module registers *provider* function
//! pointers that may use either LP64 (`i32`) or ILP64 (`i64`).
//!
//! - [`ProviderAbi::Lp64`] -- provider uses 32-bit BLAS/LAPACK integer
//!   arguments (same as tenferro's own consumer ABI).
//! - [`ProviderAbi::Ilp64`] -- provider uses 64-bit BLAS/LAPACK integer
//!   arguments (e.g. libblastrampoline configured with 64-bit integer
//!   providers).  tenferro's consumer path remains LP64; the `cblas-inject`
//!   and `lapack-inject` crates transparently bridge the integer types.
//!
//! ## Existing typed wrappers vs raw-pointer API
//!
//! - [`register_blas_gemm_fn_ptrs`] and
//!   [`register_lapack_full_piv_lu_fn_ptrs`] accept typed LP64 function
//!   pointers and remain available for backwards compatibility.
//! - The new raw-pointer entry points
//!   ([`register_blas_gemm_provider_ptrs`],
//!   [`register_lapack_provider_ptrs`]) accept opaque `*const c_void`
//!   pointers together with a [`ProviderAbi`] selector.  They cover the
//!   full GEMM and LAPACK surface.

use std::ffi::c_void;

use cblas_inject::{CgemmFnPtr, DgemmFnPtr, SgemmFnPtr, ZgemmFnPtr};
use lapack_inject::{
    Cgesc2Lp64FnPtr, Cgetc2Lp64FnPtr, Dgesc2Lp64FnPtr, Dgetc2Lp64FnPtr, Sgesc2Lp64FnPtr,
    Sgetc2Lp64FnPtr, Zgesc2Lp64FnPtr, Zgetc2Lp64FnPtr,
};

// --- Public types --------------------------------------------------------

/// Integer ABI used by provider function pointers.
///
/// tenferro itself is always an LP64 BLAS/LAPACK consumer.  This enum
/// describes the ABI of the *provider* pointers being registered.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::inject::ProviderAbi;
///
/// assert_eq!(ProviderAbi::Lp64, ProviderAbi::Lp64);
/// assert_ne!(ProviderAbi::Lp64, ProviderAbi::Ilp64);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderAbi {
    /// Provider uses 32-bit BLAS/LAPACK integer arguments.
    Lp64,
    /// Provider uses 64-bit BLAS/LAPACK integer arguments.
    Ilp64,
}

/// Error returned while registering runtime provider pointers.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::inject::ProviderRegistrationError;
///
/// let err = ProviderRegistrationError::NullPointer { symbol: "dgemm" };
/// assert_eq!(err.to_string(), "dgemm provider pointer is null");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ProviderRegistrationError {
    /// The `Some` pointer field contained a raw null pointer where a
    /// non-null function pointer was required.
    #[error("{symbol} provider pointer is null")]
    NullPointer { symbol: &'static str },
    /// The underlying inject crate reported that a symbol was already registered.
    #[error("{symbol} provider pointer is already registered")]
    AlreadyRegistered { symbol: &'static str },
    /// The underlying inject crate returned an unknown status code.
    #[error("{symbol} provider registration returned status {status}")]
    UnknownStatus { symbol: &'static str, status: i32 },
}

// --- Raw-pointer provider sets -------------------------------------------

/// Set of BLAS GEMM provider pointers (opaque `*const c_void`) to register.
///
/// Any field set to `None` is skipped.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::inject::BlasGemmProviderPtrSet;
///
/// let ptrs = BlasGemmProviderPtrSet {
///     dgemm: Some(std::ptr::null()),
///     ..BlasGemmProviderPtrSet::new()
/// };
/// assert!(ptrs.dgemm.is_some());
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BlasGemmProviderPtrSet {
    /// Fortran `sgemm` raw function pointer.
    pub sgemm: Option<*const c_void>,
    /// Fortran `dgemm` raw function pointer.
    pub dgemm: Option<*const c_void>,
    /// Fortran `cgemm` raw function pointer.
    pub cgemm: Option<*const c_void>,
    /// Fortran `zgemm` raw function pointer.
    pub zgemm: Option<*const c_void>,
}

impl BlasGemmProviderPtrSet {
    /// Create an empty set (all pointers are `None`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::inject::BlasGemmProviderPtrSet;
    ///
    /// let ptrs = BlasGemmProviderPtrSet::new();
    /// assert!(ptrs.dgemm.is_none());
    /// ```
    pub const fn new() -> Self {
        Self {
            sgemm: None,
            dgemm: None,
            cgemm: None,
            zgemm: None,
        }
    }
}

/// Set of LAPACK provider pointers (opaque `*const c_void`) to register.
///
/// Any field set to `None` is skipped.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::inject::LapackProviderPtrSet;
///
/// let ptrs = LapackProviderPtrSet {
///     dgesvd: Some(std::ptr::null()),
///     ..LapackProviderPtrSet::new()
/// };
/// assert!(ptrs.dgesvd.is_some());
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct LapackProviderPtrSet {
    /// Fortran `sgesvd` raw function pointer.
    pub sgesvd: Option<*const c_void>,
    /// Fortran `dgesvd` raw function pointer.
    pub dgesvd: Option<*const c_void>,
    /// Fortran `cgesvd` raw function pointer.
    pub cgesvd: Option<*const c_void>,
    /// Fortran `zgesvd` raw function pointer.
    pub zgesvd: Option<*const c_void>,
    /// Fortran `sgeqrf` raw function pointer.
    pub sgeqrf: Option<*const c_void>,
    /// Fortran `dgeqrf` raw function pointer.
    pub dgeqrf: Option<*const c_void>,
    /// Fortran `cgeqrf` raw function pointer.
    pub cgeqrf: Option<*const c_void>,
    /// Fortran `zgeqrf` raw function pointer.
    pub zgeqrf: Option<*const c_void>,
    /// Fortran `sorgqr` raw function pointer.
    pub sorgqr: Option<*const c_void>,
    /// Fortran `dorgqr` raw function pointer.
    pub dorgqr: Option<*const c_void>,
    /// Fortran `cungqr` raw function pointer.
    pub cungqr: Option<*const c_void>,
    /// Fortran `zungqr` raw function pointer.
    pub zungqr: Option<*const c_void>,
    /// Fortran `strtrs` raw function pointer.
    pub strtrs: Option<*const c_void>,
    /// Fortran `dtrtrs` raw function pointer.
    pub dtrtrs: Option<*const c_void>,
    /// Fortran `ctrtrs` raw function pointer.
    pub ctrtrs: Option<*const c_void>,
    /// Fortran `ztrtrs` raw function pointer.
    pub ztrtrs: Option<*const c_void>,
    /// Fortran `spotrf` raw function pointer.
    pub spotrf: Option<*const c_void>,
    /// Fortran `dpotrf` raw function pointer.
    pub dpotrf: Option<*const c_void>,
    /// Fortran `cpotrf` raw function pointer.
    pub cpotrf: Option<*const c_void>,
    /// Fortran `zpotrf` raw function pointer.
    pub zpotrf: Option<*const c_void>,
    /// Fortran `sgetrf` raw function pointer.
    pub sgetrf: Option<*const c_void>,
    /// Fortran `dgetrf` raw function pointer.
    pub dgetrf: Option<*const c_void>,
    /// Fortran `cgetrf` raw function pointer.
    pub cgetrf: Option<*const c_void>,
    /// Fortran `zgetrf` raw function pointer.
    pub zgetrf: Option<*const c_void>,
    /// Fortran `ssyev` raw function pointer.
    pub ssyev: Option<*const c_void>,
    /// Fortran `dsyev` raw function pointer.
    pub dsyev: Option<*const c_void>,
    /// Fortran `cheev` raw function pointer.
    pub cheev: Option<*const c_void>,
    /// Fortran `zheev` raw function pointer.
    pub zheev: Option<*const c_void>,
    /// Fortran `sgeev` raw function pointer.
    pub sgeev: Option<*const c_void>,
    /// Fortran `dgeev` raw function pointer.
    pub dgeev: Option<*const c_void>,
    /// Fortran `cgeev` raw function pointer.
    pub cgeev: Option<*const c_void>,
    /// Fortran `zgeev` raw function pointer.
    pub zgeev: Option<*const c_void>,
    /// Fortran `sgetc2` raw function pointer.
    pub sgetc2: Option<*const c_void>,
    /// Fortran `dgetc2` raw function pointer.
    pub dgetc2: Option<*const c_void>,
    /// Fortran `sgesc2` raw function pointer.
    pub sgesc2: Option<*const c_void>,
    /// Fortran `dgesc2` raw function pointer.
    pub dgesc2: Option<*const c_void>,
    /// Fortran `cgetc2` raw function pointer.
    pub cgetc2: Option<*const c_void>,
    /// Fortran `zgetc2` raw function pointer.
    pub zgetc2: Option<*const c_void>,
    /// Fortran `cgesc2` raw function pointer.
    pub cgesc2: Option<*const c_void>,
    /// Fortran `zgesc2` raw function pointer.
    pub zgesc2: Option<*const c_void>,
}

impl LapackProviderPtrSet {
    /// Create an empty set (all pointers are `None`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::inject::LapackProviderPtrSet;
    ///
    /// let ptrs = LapackProviderPtrSet::new();
    /// assert!(ptrs.dgesvd.is_none());
    /// ```
    pub const fn new() -> Self {
        Self {
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
            strtrs: None,
            dtrtrs: None,
            ctrtrs: None,
            ztrtrs: None,
            spotrf: None,
            dpotrf: None,
            cpotrf: None,
            zpotrf: None,
            sgetrf: None,
            dgetrf: None,
            cgetrf: None,
            zgetrf: None,
            ssyev: None,
            dsyev: None,
            cheev: None,
            zheev: None,
            sgeev: None,
            dgeev: None,
            cgeev: None,
            zgeev: None,
            sgetc2: None,
            dgetc2: None,
            sgesc2: None,
            dgesc2: None,
            cgetc2: None,
            zgetc2: None,
            cgesc2: None,
            zgesc2: None,
        }
    }
}

// --- Typed function-pointer sets (compatibility wrappers) ----------------

/// Set of CBLAS GEMM function pointers to register in one call.
///
/// Any field set to `None` is skipped, so callers can register only the
/// scalar types they plan to use.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::inject::{register_blas_gemm_fn_ptrs, BlasGemmFnPtrSet};
///
/// unsafe {
///     register_blas_gemm_fn_ptrs(BlasGemmFnPtrSet {
///         dgemm: Some(dgemm_ptr),
///         ..BlasGemmFnPtrSet::new()
///     });
/// }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BlasGemmFnPtrSet {
    /// Fortran `sgemm` function pointer.
    pub sgemm: Option<SgemmFnPtr>,
    /// Fortran `dgemm` function pointer.
    pub dgemm: Option<DgemmFnPtr>,
    /// Fortran `cgemm` function pointer.
    pub cgemm: Option<CgemmFnPtr>,
    /// Fortran `zgemm` function pointer.
    pub zgemm: Option<ZgemmFnPtr>,
}

impl BlasGemmFnPtrSet {
    /// Create an empty set (all pointers are `None`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::inject::BlasGemmFnPtrSet;
    ///
    /// let ptrs = BlasGemmFnPtrSet::new();
    /// assert!(ptrs.dgemm.is_none());
    /// ```
    pub const fn new() -> Self {
        Self {
            sgemm: None,
            dgemm: None,
            cgemm: None,
            zgemm: None,
        }
    }
}

/// Set of LAPACK complete-pivoting LU function pointers to register.
///
/// tenferro currently uses `xGETC2` for the factorization and `xGESC2` for
/// solves. Any field set to `None` is skipped.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::inject::{
///     register_lapack_full_piv_lu_fn_ptrs, LapackFullPivLuFnPtrSet,
/// };
///
/// unsafe {
///     register_lapack_full_piv_lu_fn_ptrs(LapackFullPivLuFnPtrSet {
///         dgetc2: Some(dgetc2_ptr),
///         dgesc2: Some(dgesc2_ptr),
///         ..LapackFullPivLuFnPtrSet::new()
///     });
/// }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct LapackFullPivLuFnPtrSet {
    /// Fortran `sgetc2` LP64 function pointer.
    pub sgetc2: Option<Sgetc2Lp64FnPtr>,
    /// Fortran `sgesc2` LP64 function pointer.
    pub sgesc2: Option<Sgesc2Lp64FnPtr>,
    /// Fortran `dgetc2` LP64 function pointer.
    pub dgetc2: Option<Dgetc2Lp64FnPtr>,
    /// Fortran `dgesc2` LP64 function pointer.
    pub dgesc2: Option<Dgesc2Lp64FnPtr>,
    /// Fortran `cgetc2` LP64 function pointer.
    pub cgetc2: Option<Cgetc2Lp64FnPtr>,
    /// Fortran `cgesc2` LP64 function pointer.
    pub cgesc2: Option<Cgesc2Lp64FnPtr>,
    /// Fortran `zgetc2` LP64 function pointer.
    pub zgetc2: Option<Zgetc2Lp64FnPtr>,
    /// Fortran `zgesc2` LP64 function pointer.
    pub zgesc2: Option<Zgesc2Lp64FnPtr>,
}

impl LapackFullPivLuFnPtrSet {
    /// Create an empty set (all pointers are `None`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::inject::LapackFullPivLuFnPtrSet;
    ///
    /// let ptrs = LapackFullPivLuFnPtrSet::new();
    /// assert!(ptrs.dgetc2.is_none());
    /// ```
    pub const fn new() -> Self {
        Self {
            sgetc2: None,
            sgesc2: None,
            dgetc2: None,
            dgesc2: None,
            cgetc2: None,
            cgesc2: None,
            zgetc2: None,
            zgesc2: None,
        }
    }
}

// --- Status helpers ------------------------------------------------------

/// Map a `cblas-inject` status code to a `Result`.
fn blas_registration_status(
    symbol: &'static str,
    status: i32,
) -> Result<(), ProviderRegistrationError> {
    match status {
        0 => Ok(()),
        1 => Err(ProviderRegistrationError::NullPointer { symbol }),
        2 => Err(ProviderRegistrationError::AlreadyRegistered { symbol }),
        _ => Err(ProviderRegistrationError::UnknownStatus { symbol, status }),
    }
}

/// Map a `lapack-inject` status code to a `Result`.
///
/// lapack-inject typed registration functions accept non-null function
/// pointers; tenferro null-checks the raw pointer before `transmute`, so
/// status 1 (null pointer) is never returned here.
fn lapack_registration_status(
    symbol: &'static str,
    status: i32,
) -> Result<(), ProviderRegistrationError> {
    match status {
        0 => Ok(()),
        2 => Err(ProviderRegistrationError::AlreadyRegistered { symbol }),
        _ => Err(ProviderRegistrationError::UnknownStatus { symbol, status }),
    }
}

// --- Compatibility wrappers (typed LP64) ---------------------------------

/// Register CBLAS GEMM function pointers in one call.
///
/// This is a thin bulk wrapper over `cblas-inject`'s per-symbol
/// `register_*` functions.
///
/// # Safety
///
/// Each provided function pointer must have the exact Fortran BLAS ABI
/// expected by `cblas-inject`.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::inject::{register_blas_gemm_fn_ptrs, BlasGemmFnPtrSet};
///
/// unsafe {
///     register_blas_gemm_fn_ptrs(BlasGemmFnPtrSet {
///         sgemm: Some(sgemm_ptr),
///         dgemm: Some(dgemm_ptr),
///         cgemm: Some(cgemm_ptr),
///         zgemm: Some(zgemm_ptr),
///     });
/// }
/// ```
pub unsafe fn register_blas_gemm_fn_ptrs(ptrs: BlasGemmFnPtrSet) {
    if let Some(f) = ptrs.sgemm {
        unsafe { cblas_inject::register_sgemm(f) };
    }
    if let Some(f) = ptrs.dgemm {
        unsafe { cblas_inject::register_dgemm(f) };
    }
    if let Some(f) = ptrs.cgemm {
        unsafe { cblas_inject::register_cgemm(f) };
    }
    if let Some(f) = ptrs.zgemm {
        unsafe { cblas_inject::register_zgemm(f) };
    }
}

/// Register LAPACK complete-pivoting LU function pointers in one call.
///
/// This is a thin bulk wrapper over `lapack-inject`'s per-symbol
/// `register_*` functions.
///
/// # Safety
///
/// Each provided function pointer must have the exact Fortran LAPACK ABI
/// expected by `lapack-inject`.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::inject::{
///     register_lapack_full_piv_lu_fn_ptrs, LapackFullPivLuFnPtrSet,
/// };
///
/// unsafe {
///     register_lapack_full_piv_lu_fn_ptrs(LapackFullPivLuFnPtrSet {
///         dgetc2: Some(dgetc2_ptr),
///         dgesc2: Some(dgesc2_ptr),
///         zgetc2: Some(zgetc2_ptr),
///         zgesc2: Some(zgesc2_ptr),
///     });
/// }
/// ```
pub unsafe fn register_lapack_full_piv_lu_fn_ptrs(ptrs: LapackFullPivLuFnPtrSet) {
    if let Some(f) = ptrs.sgetc2 {
        unsafe { lapack_inject::register_sgetc2_lp64(f) };
    }
    if let Some(f) = ptrs.sgesc2 {
        unsafe { lapack_inject::register_sgesc2_lp64(f) };
    }
    if let Some(f) = ptrs.dgetc2 {
        unsafe { lapack_inject::register_dgetc2_lp64(f) };
    }
    if let Some(f) = ptrs.dgesc2 {
        unsafe { lapack_inject::register_dgesc2_lp64(f) };
    }
    if let Some(f) = ptrs.cgetc2 {
        unsafe { lapack_inject::register_cgetc2_lp64(f) };
    }
    if let Some(f) = ptrs.cgesc2 {
        unsafe { lapack_inject::register_cgesc2_lp64(f) };
    }
    if let Some(f) = ptrs.zgetc2 {
        unsafe { lapack_inject::register_zgetc2_lp64(f) };
    }
    if let Some(f) = ptrs.zgesc2 {
        unsafe { lapack_inject::register_zgesc2_lp64(f) };
    }
}

// --- Raw-pointer BLAS GEMM registration ----------------------------------

/// Register BLAS GEMM provider pointers with a selected ABI.
///
/// Accepts opaque `*const c_void` pointers and dispatches to the
/// appropriate `cblas-inject` LP64 or ILP64 registration function.
///
/// # Safety
///
/// Each provided pointer must be a valid function pointer with the Fortran
/// BLAS ABI for the corresponding scalar type and integer width.
/// Passing a null pointer for a `Some` field returns
/// [`ProviderRegistrationError::NullPointer`] instead of calling into the
/// provider crate.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::inject::{
///     register_blas_gemm_provider_ptrs, BlasGemmProviderPtrSet, ProviderAbi,
///     ProviderRegistrationError,
/// };
///
/// let err = unsafe {
///     register_blas_gemm_provider_ptrs(
///         ProviderAbi::Ilp64,
///         BlasGemmProviderPtrSet {
///             dgemm: Some(std::ptr::null()),
///             ..BlasGemmProviderPtrSet::new()
///         },
///     )
/// }
/// .unwrap_err();
/// assert_eq!(
///     err,
///     ProviderRegistrationError::NullPointer { symbol: "dgemm" }
/// );
/// ```
pub unsafe fn register_blas_gemm_provider_ptrs(
    abi: ProviderAbi,
    ptrs: BlasGemmProviderPtrSet,
) -> Result<(), ProviderRegistrationError> {
    macro_rules! register_one {
        ($field:ident, $symbol:literal, $lp64_fn:path, $ilp64_fn:path) => {
            if let Some(ptr) = ptrs.$field {
                if ptr.is_null() {
                    return Err(ProviderRegistrationError::NullPointer { symbol: $symbol });
                }
                let status = match abi {
                    ProviderAbi::Lp64 => unsafe { $lp64_fn(ptr) },
                    ProviderAbi::Ilp64 => unsafe { $ilp64_fn(ptr) },
                };
                blas_registration_status($symbol, status)?;
            }
        };
    }

    register_one!(
        sgemm,
        "sgemm",
        cblas_inject::cblas_inject_register_sgemm_lp64,
        cblas_inject::cblas_inject_register_sgemm_ilp64
    );
    register_one!(
        dgemm,
        "dgemm",
        cblas_inject::cblas_inject_register_dgemm_lp64,
        cblas_inject::cblas_inject_register_dgemm_ilp64
    );
    register_one!(
        cgemm,
        "cgemm",
        cblas_inject::cblas_inject_register_cgemm_lp64,
        cblas_inject::cblas_inject_register_cgemm_ilp64
    );
    register_one!(
        zgemm,
        "zgemm",
        cblas_inject::cblas_inject_register_zgemm_lp64,
        cblas_inject::cblas_inject_register_zgemm_ilp64
    );

    Ok(())
}

// --- Raw-pointer LAPACK registration -------------------------------------

/// Register a single LAPACK symbol with transmute and null-check.
macro_rules! register_lapack_symbol {
    ($abi:expr, $symbol:literal, $raw:expr, $lp64_ty:ty, $ilp64_ty:ty,
     $lp64_reg:path, $ilp64_reg:path) => {{
        let raw = $raw;
        if raw.is_null() {
            return Err(ProviderRegistrationError::NullPointer { symbol: $symbol });
        }
        let status = match $abi {
            ProviderAbi::Lp64 => {
                let f = unsafe { std::mem::transmute::<*const std::ffi::c_void, $lp64_ty>(raw) };
                unsafe { $lp64_reg(f) }
            }
            ProviderAbi::Ilp64 => {
                let f = unsafe { std::mem::transmute::<*const std::ffi::c_void, $ilp64_ty>(raw) };
                unsafe { $ilp64_reg(f) }
            }
        };
        lapack_registration_status($symbol, status)?;
    }};
}

/// Register LAPACK provider pointers with a selected ABI.
///
/// Accepts opaque `*const c_void` pointers and dispatches to the
/// appropriate `lapack-inject` LP64 or ILP64 registration function.
///
/// # Safety
///
/// Each provided pointer must be a valid function pointer with the Fortran
/// LAPACK ABI for the corresponding scalar type and integer width.
/// Passing a null pointer for a `Some` field returns
/// [`ProviderRegistrationError::NullPointer`] instead of calling into the
/// provider crate.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::inject::{
///     register_lapack_provider_ptrs, LapackProviderPtrSet, ProviderAbi,
///     ProviderRegistrationError,
/// };
///
/// let err = unsafe {
///     register_lapack_provider_ptrs(
///         ProviderAbi::Ilp64,
///         LapackProviderPtrSet {
///             dgesvd: Some(std::ptr::null()),
///             ..LapackProviderPtrSet::new()
///         },
///     )
/// }
/// .unwrap_err();
/// assert_eq!(
///     err,
///     ProviderRegistrationError::NullPointer { symbol: "dgesvd" }
/// );
/// ```
pub unsafe fn register_lapack_provider_ptrs(
    abi: ProviderAbi,
    ptrs: LapackProviderPtrSet,
) -> Result<(), ProviderRegistrationError> {
    use lapack_inject::*;

    macro_rules! register_field {
        ($field:ident, $symbol:literal,
         $lp64_ty:ident, $ilp64_ty:ident,
         $lp64_reg:ident, $ilp64_reg:ident) => {
            if let Some(raw) = ptrs.$field {
                register_lapack_symbol!(
                    abi, $symbol, raw, $lp64_ty, $ilp64_ty, $lp64_reg, $ilp64_reg
                );
            }
        };
    }

    register_field!(
        sgesvd,
        "sgesvd",
        SgesvdLp64FnPtr,
        SgesvdIlp64FnPtr,
        register_sgesvd_lp64,
        register_sgesvd_ilp64
    );
    register_field!(
        dgesvd,
        "dgesvd",
        DgesvdLp64FnPtr,
        DgesvdIlp64FnPtr,
        register_dgesvd_lp64,
        register_dgesvd_ilp64
    );
    register_field!(
        cgesvd,
        "cgesvd",
        CgesvdLp64FnPtr,
        CgesvdIlp64FnPtr,
        register_cgesvd_lp64,
        register_cgesvd_ilp64
    );
    register_field!(
        zgesvd,
        "zgesvd",
        ZgesvdLp64FnPtr,
        ZgesvdIlp64FnPtr,
        register_zgesvd_lp64,
        register_zgesvd_ilp64
    );
    register_field!(
        sgeqrf,
        "sgeqrf",
        SgeqrfLp64FnPtr,
        SgeqrfIlp64FnPtr,
        register_sgeqrf_lp64,
        register_sgeqrf_ilp64
    );
    register_field!(
        dgeqrf,
        "dgeqrf",
        DgeqrfLp64FnPtr,
        DgeqrfIlp64FnPtr,
        register_dgeqrf_lp64,
        register_dgeqrf_ilp64
    );
    register_field!(
        cgeqrf,
        "cgeqrf",
        CgeqrfLp64FnPtr,
        CgeqrfIlp64FnPtr,
        register_cgeqrf_lp64,
        register_cgeqrf_ilp64
    );
    register_field!(
        zgeqrf,
        "zgeqrf",
        ZgeqrfLp64FnPtr,
        ZgeqrfIlp64FnPtr,
        register_zgeqrf_lp64,
        register_zgeqrf_ilp64
    );
    register_field!(
        sorgqr,
        "sorgqr",
        SorgqrLp64FnPtr,
        SorgqrIlp64FnPtr,
        register_sorgqr_lp64,
        register_sorgqr_ilp64
    );
    register_field!(
        dorgqr,
        "dorgqr",
        DorgqrLp64FnPtr,
        DorgqrIlp64FnPtr,
        register_dorgqr_lp64,
        register_dorgqr_ilp64
    );
    register_field!(
        cungqr,
        "cungqr",
        CungqrLp64FnPtr,
        CungqrIlp64FnPtr,
        register_cungqr_lp64,
        register_cungqr_ilp64
    );
    register_field!(
        zungqr,
        "zungqr",
        ZungqrLp64FnPtr,
        ZungqrIlp64FnPtr,
        register_zungqr_lp64,
        register_zungqr_ilp64
    );
    register_field!(
        strtrs,
        "strtrs",
        StrtrsLp64FnPtr,
        StrtrsIlp64FnPtr,
        register_strtrs_lp64,
        register_strtrs_ilp64
    );
    register_field!(
        dtrtrs,
        "dtrtrs",
        DtrtrsLp64FnPtr,
        DtrtrsIlp64FnPtr,
        register_dtrtrs_lp64,
        register_dtrtrs_ilp64
    );
    register_field!(
        ctrtrs,
        "ctrtrs",
        CtrtrsLp64FnPtr,
        CtrtrsIlp64FnPtr,
        register_ctrtrs_lp64,
        register_ctrtrs_ilp64
    );
    register_field!(
        ztrtrs,
        "ztrtrs",
        ZtrtrsLp64FnPtr,
        ZtrtrsIlp64FnPtr,
        register_ztrtrs_lp64,
        register_ztrtrs_ilp64
    );
    register_field!(
        spotrf,
        "spotrf",
        SpotrfLp64FnPtr,
        SpotrfIlp64FnPtr,
        register_spotrf_lp64,
        register_spotrf_ilp64
    );
    register_field!(
        dpotrf,
        "dpotrf",
        DpotrfLp64FnPtr,
        DpotrfIlp64FnPtr,
        register_dpotrf_lp64,
        register_dpotrf_ilp64
    );
    register_field!(
        cpotrf,
        "cpotrf",
        CpotrfLp64FnPtr,
        CpotrfIlp64FnPtr,
        register_cpotrf_lp64,
        register_cpotrf_ilp64
    );
    register_field!(
        zpotrf,
        "zpotrf",
        ZpotrfLp64FnPtr,
        ZpotrfIlp64FnPtr,
        register_zpotrf_lp64,
        register_zpotrf_ilp64
    );
    register_field!(
        sgetrf,
        "sgetrf",
        SgetrfLp64FnPtr,
        SgetrfIlp64FnPtr,
        register_sgetrf_lp64,
        register_sgetrf_ilp64
    );
    register_field!(
        dgetrf,
        "dgetrf",
        DgetrfLp64FnPtr,
        DgetrfIlp64FnPtr,
        register_dgetrf_lp64,
        register_dgetrf_ilp64
    );
    register_field!(
        cgetrf,
        "cgetrf",
        CgetrfLp64FnPtr,
        CgetrfIlp64FnPtr,
        register_cgetrf_lp64,
        register_cgetrf_ilp64
    );
    register_field!(
        zgetrf,
        "zgetrf",
        ZgetrfLp64FnPtr,
        ZgetrfIlp64FnPtr,
        register_zgetrf_lp64,
        register_zgetrf_ilp64
    );
    register_field!(
        ssyev,
        "ssyev",
        SsyevLp64FnPtr,
        SsyevIlp64FnPtr,
        register_ssyev_lp64,
        register_ssyev_ilp64
    );
    register_field!(
        dsyev,
        "dsyev",
        DsyevLp64FnPtr,
        DsyevIlp64FnPtr,
        register_dsyev_lp64,
        register_dsyev_ilp64
    );
    register_field!(
        cheev,
        "cheev",
        CheevLp64FnPtr,
        CheevIlp64FnPtr,
        register_cheev_lp64,
        register_cheev_ilp64
    );
    register_field!(
        zheev,
        "zheev",
        ZheevLp64FnPtr,
        ZheevIlp64FnPtr,
        register_zheev_lp64,
        register_zheev_ilp64
    );
    register_field!(
        sgeev,
        "sgeev",
        SgeevLp64FnPtr,
        SgeevIlp64FnPtr,
        register_sgeev_lp64,
        register_sgeev_ilp64
    );
    register_field!(
        dgeev,
        "dgeev",
        DgeevLp64FnPtr,
        DgeevIlp64FnPtr,
        register_dgeev_lp64,
        register_dgeev_ilp64
    );
    register_field!(
        cgeev,
        "cgeev",
        CgeevLp64FnPtr,
        CgeevIlp64FnPtr,
        register_cgeev_lp64,
        register_cgeev_ilp64
    );
    register_field!(
        zgeev,
        "zgeev",
        ZgeevLp64FnPtr,
        ZgeevIlp64FnPtr,
        register_zgeev_lp64,
        register_zgeev_ilp64
    );
    register_field!(
        sgetc2,
        "sgetc2",
        Sgetc2Lp64FnPtr,
        Sgetc2Ilp64FnPtr,
        register_sgetc2_lp64,
        register_sgetc2_ilp64
    );
    register_field!(
        dgetc2,
        "dgetc2",
        Dgetc2Lp64FnPtr,
        Dgetc2Ilp64FnPtr,
        register_dgetc2_lp64,
        register_dgetc2_ilp64
    );
    register_field!(
        sgesc2,
        "sgesc2",
        Sgesc2Lp64FnPtr,
        Sgesc2Ilp64FnPtr,
        register_sgesc2_lp64,
        register_sgesc2_ilp64
    );
    register_field!(
        dgesc2,
        "dgesc2",
        Dgesc2Lp64FnPtr,
        Dgesc2Ilp64FnPtr,
        register_dgesc2_lp64,
        register_dgesc2_ilp64
    );
    register_field!(
        cgetc2,
        "cgetc2",
        Cgetc2Lp64FnPtr,
        Cgetc2Ilp64FnPtr,
        register_cgetc2_lp64,
        register_cgetc2_ilp64
    );
    register_field!(
        zgetc2,
        "zgetc2",
        Zgetc2Lp64FnPtr,
        Zgetc2Ilp64FnPtr,
        register_zgetc2_lp64,
        register_zgetc2_ilp64
    );
    register_field!(
        cgesc2,
        "cgesc2",
        Cgesc2Lp64FnPtr,
        Cgesc2Ilp64FnPtr,
        register_cgesc2_lp64,
        register_cgesc2_ilp64
    );
    register_field!(
        zgesc2,
        "zgesc2",
        Zgesc2Lp64FnPtr,
        Zgesc2Ilp64FnPtr,
        register_zgesc2_lp64,
        register_zgesc2_ilp64
    );

    Ok(())
}
