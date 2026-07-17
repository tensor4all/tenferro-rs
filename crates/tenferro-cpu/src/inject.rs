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
//! ## Raw-pointer provider API
//!
//! [`register_blas_gemm_provider_ptrs`] and [`register_lapack_provider_ptrs`]
//! accept opaque `*const c_void` pointers together with a [`ProviderAbi`]
//! selector. They are the only public registration API, so LP64 and ILP64
//! providers use the same contract.

use std::ffi::c_void;

// --- Public types --------------------------------------------------------

/// Integer ABI used by provider function pointers.
///
/// tenferro itself is always an LP64 BLAS/LAPACK consumer.  This enum
/// describes the ABI of the *provider* pointers being registered.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::inject::ProviderAbi;
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
/// use tenferro_cpu::inject::ProviderRegistrationError;
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
    /// A raw provider pointer cannot be represented as the requested function pointer type.
    #[error(
        "{symbol} provider pointer size {pointer_size} does not match function pointer size {function_pointer_size}"
    )]
    FunctionPointerSizeMismatch {
        symbol: &'static str,
        pointer_size: usize,
        function_pointer_size: usize,
    },
}

// --- Raw-pointer provider sets -------------------------------------------

/// Set of BLAS GEMM provider pointers (opaque `*const c_void`) to register.
///
/// Any field set to `None` is skipped.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::inject::BlasGemmProviderPtrSet;
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
    /// use tenferro_cpu::inject::BlasGemmProviderPtrSet;
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
/// use tenferro_cpu::inject::LapackProviderPtrSet;
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
    /// Fortran `sgetrs` raw function pointer.
    pub sgetrs: Option<*const c_void>,
    /// Fortran `dgetrf` raw function pointer.
    pub dgetrf: Option<*const c_void>,
    /// Fortran `dgetrs` raw function pointer.
    pub dgetrs: Option<*const c_void>,
    /// Fortran `cgetrf` raw function pointer.
    pub cgetrf: Option<*const c_void>,
    /// Fortran `cgetrs` raw function pointer.
    pub cgetrs: Option<*const c_void>,
    /// Fortran `zgetrf` raw function pointer.
    pub zgetrf: Option<*const c_void>,
    /// Fortran `zgetrs` raw function pointer.
    pub zgetrs: Option<*const c_void>,
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
    /// use tenferro_cpu::inject::LapackProviderPtrSet;
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
            sgetrs: None,
            dgetrf: None,
            dgetrs: None,
            cgetrf: None,
            cgetrs: None,
            zgetrf: None,
            zgetrs: None,
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
/// use tenferro_cpu::inject::{
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
///
/// # Errors
///
/// Returns [`ProviderRegistrationError::NullPointer`] for a null supplied
/// pointer, `FunctionPointerSizeMismatch` when the ABI pointer width is not
/// representable, or `AlreadyRegistered`/`UnknownStatus` from the injector.
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
                    // SAFETY: Null was rejected above. The public unsafe
                    // contract requires the pointer to match `$symbol` and ABI.
                    ProviderAbi::Lp64 => unsafe { $lp64_fn(ptr) },
                    // SAFETY: Null was rejected above. The public unsafe
                    // contract requires the pointer to match `$symbol` and ABI.
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
                // SAFETY: Null was rejected above. The public unsafe contract
                // requires `$raw` to be a `$symbol` pointer with the LP64 ABI.
                let f = unsafe { cast_provider_function_pointer::<$lp64_ty>($symbol, raw)? };
                // SAFETY: `f` has the provider registration function's exact
                // typed signature after the guarded cast above.
                unsafe { $lp64_reg(f) }
            }
            ProviderAbi::Ilp64 => {
                // SAFETY: Null was rejected above. The public unsafe contract
                // requires `$raw` to be a `$symbol` pointer with the ILP64 ABI.
                let f = unsafe { cast_provider_function_pointer::<$ilp64_ty>($symbol, raw)? };
                // SAFETY: `f` has the provider registration function's exact
                // typed signature after the guarded cast above.
                unsafe { $ilp64_reg(f) }
            }
        };
        lapack_registration_status($symbol, status)?;
    }};
}

unsafe fn cast_provider_function_pointer<F: Copy>(
    symbol: &'static str,
    raw: *const c_void,
) -> Result<F, ProviderRegistrationError> {
    let pointer_size = std::mem::size_of::<*const c_void>();
    let function_pointer_size = std::mem::size_of::<F>();
    if pointer_size != function_pointer_size {
        return Err(ProviderRegistrationError::FunctionPointerSizeMismatch {
            symbol,
            pointer_size,
            function_pointer_size,
        });
    }

    // SAFETY: Callers provide a non-null provider symbol pointer for the exact
    // LAPACK ABI type `F`. The size check above rejects targets where an opaque
    // provider pointer cannot be represented as that function pointer type.
    Ok(unsafe { std::mem::transmute_copy::<*const c_void, F>(&raw) })
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
/// use tenferro_cpu::inject::{
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
///
/// # Errors
///
/// Returns [`ProviderRegistrationError::NullPointer`] for a null supplied
/// pointer, `FunctionPointerSizeMismatch` when the ABI pointer width is not
/// representable, or `AlreadyRegistered`/`UnknownStatus` from the injector.
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
        sgetrs,
        "sgetrs",
        SgetrsLp64FnPtr,
        SgetrsIlp64FnPtr,
        register_sgetrs_lp64,
        register_sgetrs_ilp64
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
        dgetrs,
        "dgetrs",
        DgetrsLp64FnPtr,
        DgetrsIlp64FnPtr,
        register_dgetrs_lp64,
        register_dgetrs_ilp64
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
        cgetrs,
        "cgetrs",
        CgetrsLp64FnPtr,
        CgetrsIlp64FnPtr,
        register_cgetrs_lp64,
        register_cgetrs_ilp64
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
        zgetrs,
        "zgetrs",
        ZgetrsLp64FnPtr,
        ZgetrsIlp64FnPtr,
        register_zgetrs_lp64,
        register_zgetrs_ilp64
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_function_pointer_cast_rejects_size_mismatch() {
        let err = unsafe {
            cast_provider_function_pointer::<[usize; 2]>("dgesvd", 1usize as *const c_void)
        }
        .unwrap_err();

        assert!(matches!(
            err,
            ProviderRegistrationError::FunctionPointerSizeMismatch {
                symbol: "dgesvd",
                ..
            }
        ));
    }

    #[test]
    fn typed_registration_status_helpers_report_errors() {
        assert!(matches!(
            blas_registration_status("dgemm", 2),
            Err(ProviderRegistrationError::AlreadyRegistered { symbol: "dgemm" })
        ));
        assert!(matches!(
            blas_registration_status("dgemm", 9),
            Err(ProviderRegistrationError::UnknownStatus {
                symbol: "dgemm",
                status: 9
            })
        ));
        assert!(matches!(
            lapack_registration_status("dgetc2", 2),
            Err(ProviderRegistrationError::AlreadyRegistered { symbol: "dgetc2" })
        ));
        assert!(matches!(
            lapack_registration_status("dgetc2", 9),
            Err(ProviderRegistrationError::UnknownStatus {
                symbol: "dgetc2",
                status: 9
            })
        ));
    }
}
