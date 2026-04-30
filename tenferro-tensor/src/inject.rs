//! Runtime BLAS function-pointer registration for `provider-inject`.
//!
//! This module is available only with:
//! - `cpu-blas`
//! - `provider-inject`

use cblas_inject::{CgemmFnPtr, DgemmFnPtr, SgemmFnPtr, ZgemmFnPtr};
use lapack_inject::{Dgesc2FnPtr, Dgetc2FnPtr, Zgesc2FnPtr, Zgetc2FnPtr};

/// Set of CBLAS GEMM function pointers to register in one call.
///
/// Any field set to `None` is skipped, so callers can register only the
/// scalar types they plan to use.
///
/// # Examples
///
/// ```ignore
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

/// Set of LAPACK complete-pivoting LU function pointers to register.
///
/// tenferro currently uses `xGETC2` for the factorization and `xGESC2` for
/// solves. Any field set to `None` is skipped.
///
/// # Examples
///
/// ```ignore
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
    /// Fortran `dgetc2` function pointer.
    pub dgetc2: Option<Dgetc2FnPtr>,
    /// Fortran `dgesc2` function pointer.
    pub dgesc2: Option<Dgesc2FnPtr>,
    /// Fortran `zgetc2` function pointer.
    pub zgetc2: Option<Zgetc2FnPtr>,
    /// Fortran `zgesc2` function pointer.
    pub zgesc2: Option<Zgesc2FnPtr>,
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
            dgetc2: None,
            dgesc2: None,
            zgetc2: None,
            zgesc2: None,
        }
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
/// ```ignore
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
    if let Some(f) = ptrs.dgetc2 {
        unsafe { lapack_inject::register_dgetc2(f) };
    }
    if let Some(f) = ptrs.dgesc2 {
        unsafe { lapack_inject::register_dgesc2(f) };
    }
    if let Some(f) = ptrs.zgetc2 {
        unsafe { lapack_inject::register_zgetc2(f) };
    }
    if let Some(f) = ptrs.zgesc2 {
        unsafe { lapack_inject::register_zgesc2(f) };
    }
}
