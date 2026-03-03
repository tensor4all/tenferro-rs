//! Runtime BLAS function-pointer registration for `provider-inject`.
//!
//! This module is available only with:
//! - `gemm-blas`
//! - `provider-inject`

use cblas_inject::{CgemmFnPtr, DgemmFnPtr, SgemmFnPtr, ZgemmFnPtr};

/// Set of CBLAS GEMM function pointers to register in one call.
///
/// Any field set to `None` is skipped, so callers can register only the
/// scalar types they plan to use.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::inject::{BlasGemmFnPtrSet, register_blas_gemm_fn_ptrs};
///
/// unsafe {
///     register_blas_gemm_fn_ptrs(BlasGemmFnPtrSet {
///         dgemm: Some(dgemm_ptr),
///         zgemm: Some(zgemm_ptr),
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
    /// use tenferro_prims::inject::BlasGemmFnPtrSet;
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
/// ```ignore
/// use tenferro_prims::inject::{BlasGemmFnPtrSet, register_blas_gemm_fn_ptrs};
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
