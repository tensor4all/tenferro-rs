//! cuTENSOR v2 FFI types and vtable for runtime-loaded cuTENSOR library.
//!
//! All types are gated behind `cfg(feature = "cuda")`.
//!
//! The vtable is populated by [`CudaBackend::load`](super::cuda::CudaBackend)
//! via `libloading` and used by `execute()` to call cuTENSOR functions.
//!
//! # Examples
//!
//! ```ignore
//! // Aspirational API — not yet functional.
//! use tenferro_prims::cuda_ffi::CutensorVtable;
//!
//! let vtable = CutensorVtable::load("/usr/lib/libcutensor.so").unwrap();
//! ```

use std::ffi::c_void;
use std::os::raw::c_int;

/// cuTENSOR data type enum (mirrors `cutensorDataType_t`).
///
/// Maps tensor scalar types to cuTENSOR's internal representation.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CutensorDataType;
///
/// let dt = CutensorDataType::Float64;
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutensorDataType {
    /// 32-bit floating point (`float`).
    Float32 = 0,
    /// 64-bit floating point (`double`).
    Float64 = 1,
    /// 32-bit complex (`cuComplex`).
    Complex32 = 2,
    /// 64-bit complex (`cuDoubleComplex`).
    Complex64 = 3,
}

/// cuTENSOR status code (mirrors `cutensorStatus_t`).
pub type CutensorStatus = c_int;

/// Opaque cuTENSOR library handle (mirrors `cutensorHandle_t`).
pub type CutensorHandle = *mut c_void;

/// Opaque cuTENSOR operation plan (mirrors `cutensorOperationPlan_t`).
pub type CutensorOperationPlan = *mut c_void;

/// Vtable of cuTENSOR v2 function pointers, loaded at runtime via libloading.
///
/// Each field corresponds to a cuTENSOR v2 API function. The vtable is
/// populated by `CudaBackend::load()` and used by `execute()`.
///
/// **Status: Not yet implemented.** Function pointer types are placeholders;
/// real signatures will be filled in during Phase 3 GPU testing.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CutensorVtable;
///
/// let vtable = CutensorVtable::load("/usr/lib/libcutensor.so").unwrap();
/// ```
pub struct CutensorVtable {
    /// `cutensorCreate` — initialize library handle.
    pub create: unsafe extern "C" fn(*mut CutensorHandle) -> CutensorStatus,
    /// `cutensorDestroy` — release library handle.
    pub destroy: unsafe extern "C" fn(CutensorHandle) -> CutensorStatus,
    /// `cutensorContract` — tensor contraction.
    pub contract: *const c_void,
    /// `cutensorPermute` — tensor permutation.
    pub permute: *const c_void,
    /// `cutensorReduce` — tensor reduction.
    pub reduce: *const c_void,
    /// `cutensorElementwiseBinary` — element-wise binary operation.
    pub elementwise_binary: *const c_void,
    /// `cutensorElementwiseTrinary` — element-wise ternary operation.
    pub elementwise_trinary: *const c_void,
}
