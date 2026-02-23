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
//! use tenferro_prims::cuda_ffi::CutensorVtable;
//!
//! let lib = unsafe { libloading::Library::new("/usr/lib/libcutensor.so") }.unwrap();
//! let vtable = unsafe { CutensorVtable::load(&lib) }.unwrap();
//! ```

use std::ffi::c_void;

// ============================================================================
// cuTENSOR status codes
// ============================================================================

/// cuTENSOR status code (mirrors `cutensorStatus_t`).
pub type cutensorStatus_t = i32;

/// cuTENSOR success status.
pub const CUTENSOR_STATUS_SUCCESS: cutensorStatus_t = 0;

// ============================================================================
// Opaque handle types
// ============================================================================

/// Opaque cuTENSOR library handle.
pub type cutensorHandle_t = *mut c_void;
/// Opaque tensor descriptor.
pub type cutensorTensorDescriptor_t = *mut c_void;
/// Opaque operation descriptor (contraction, permutation, reduction, etc.).
pub type cutensorOperationDescriptor_t = *mut c_void;
/// Opaque plan preference.
pub type cutensorPlanPreference_t = *mut c_void;
/// Opaque execution plan.
pub type cutensorPlan_t = *mut c_void;
/// Opaque compute descriptor (pre-defined global constants in libcutensor).
pub type cutensorComputeDescriptor_t = *const c_void;

// ============================================================================
// Enums
// ============================================================================

/// cuTENSOR data type (mirrors `cutensorDataType_t`).
///
/// Maps Rust scalar types to cuTENSOR's internal type identifiers.
/// Values must match the cuTENSOR v2 header exactly.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CutensorDataType;
/// let dt = CutensorDataType::R_64F;
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutensorDataType {
    /// 32-bit float (`float`).
    R_32F = 0,
    /// 64-bit float (`double`).
    R_64F = 1,
    /// 32-bit complex (`cuComplex`).
    C_32F = 4,
    /// 64-bit complex (`cuDoubleComplex`).
    C_64F = 5,
}

/// cuTENSOR unary element-wise operator (mirrors `cutensorOperator_t`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutensorOperator {
    /// Identity (no transformation).
    Identity = 1,
    /// Complex conjugate.
    Conj = 2,
}

/// cuTENSOR algorithm selection (mirrors `cutensorAlgo_t`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutensorAlgo {
    /// Let cuTENSOR choose the best algorithm.
    Default = -1,
}

/// cuTENSOR JIT compilation mode (mirrors `cutensorJitMode_t`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutensorJitMode {
    /// Disable JIT compilation.
    None = 0,
}

/// cuTENSOR workspace size preference (mirrors `cutensorWorksizePreference_t`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutensorWorksizePref {
    /// Minimum workspace.
    Min = 1,
    /// Recommended workspace (balance of speed and memory).
    Recommended = 2,
    /// Maximum workspace (fastest execution).
    Max = 3,
}

/// Reduction operator for `cutensorCreateReduction`.
///
/// Mirrors the cuTENSOR v2 `cutensorOperator_t` values used in
/// reduction contexts.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutensorReduceOp {
    /// Addition (sum) reduction.
    Add = 3,
    /// Maximum value reduction.
    Max = 5,
    /// Minimum value reduction.
    Min = 6,
}

// ============================================================================
// Function pointer type aliases
// ============================================================================

// -- Handle lifecycle --
pub type FnCreate = unsafe extern "C" fn(*mut cutensorHandle_t) -> cutensorStatus_t;
pub type FnDestroy = unsafe extern "C" fn(cutensorHandle_t) -> cutensorStatus_t;

// -- Tensor descriptor --
pub type FnCreateTensorDescriptor = unsafe extern "C" fn(
    cutensorHandle_t,
    *mut cutensorTensorDescriptor_t,
    u32,        // num_modes
    *const i64, // extent
    *const i64, // stride
    CutensorDataType,
    u32, // alignment_requirement
) -> cutensorStatus_t;

pub type FnDestroyTensorDescriptor =
    unsafe extern "C" fn(cutensorTensorDescriptor_t) -> cutensorStatus_t;

// -- Operation descriptors --
pub type FnCreateContraction = unsafe extern "C" fn(
    cutensorHandle_t,
    *mut cutensorOperationDescriptor_t,
    cutensorTensorDescriptor_t,
    *const i32,
    CutensorOperator, // A
    cutensorTensorDescriptor_t,
    *const i32,
    CutensorOperator, // B
    cutensorTensorDescriptor_t,
    *const i32,
    CutensorOperator, // C
    cutensorTensorDescriptor_t,
    *const i32, // D
    cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

pub type FnCreatePermutation = unsafe extern "C" fn(
    cutensorHandle_t,
    *mut cutensorOperationDescriptor_t,
    cutensorTensorDescriptor_t,
    *const i32,
    CutensorOperator, // A
    cutensorTensorDescriptor_t,
    *const i32, // B
    cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

pub type FnCreateReduction = unsafe extern "C" fn(
    cutensorHandle_t,
    *mut cutensorOperationDescriptor_t,
    cutensorTensorDescriptor_t,
    *const i32,
    CutensorOperator, // A
    cutensorTensorDescriptor_t,
    *const i32,
    CutensorOperator, // C
    cutensorTensorDescriptor_t,
    *const i32, // D
    CutensorReduceOp,
    cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

pub type FnCreateElementwiseBinary = unsafe extern "C" fn(
    cutensorHandle_t,
    *mut cutensorOperationDescriptor_t,
    cutensorTensorDescriptor_t,
    *const i32,
    CutensorOperator, // A
    cutensorTensorDescriptor_t,
    *const i32,
    CutensorOperator, // C
    cutensorTensorDescriptor_t,
    *const i32,       // D
    CutensorReduceOp, // op_ac
    cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

pub type FnCreateElementwiseTrinary = unsafe extern "C" fn(
    cutensorHandle_t,
    *mut cutensorOperationDescriptor_t,
    cutensorTensorDescriptor_t,
    *const i32,
    CutensorOperator, // A
    cutensorTensorDescriptor_t,
    *const i32,
    CutensorOperator, // B
    cutensorTensorDescriptor_t,
    *const i32,
    CutensorOperator, // C
    cutensorTensorDescriptor_t,
    *const i32,       // D
    CutensorReduceOp, // op_ab
    CutensorReduceOp, // op_abc
    cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

pub type FnDestroyOperationDescriptor =
    unsafe extern "C" fn(cutensorOperationDescriptor_t) -> cutensorStatus_t;

// -- Plan --
pub type FnCreatePlanPreference = unsafe extern "C" fn(
    cutensorHandle_t,
    *mut cutensorPlanPreference_t,
    CutensorAlgo,
    CutensorJitMode,
) -> cutensorStatus_t;

pub type FnDestroyPlanPreference =
    unsafe extern "C" fn(cutensorPlanPreference_t) -> cutensorStatus_t;

pub type FnEstimateWorkspaceSize = unsafe extern "C" fn(
    cutensorHandle_t,
    cutensorOperationDescriptor_t,
    cutensorPlanPreference_t,
    CutensorWorksizePref,
    *mut u64,
) -> cutensorStatus_t;

pub type FnCreatePlan = unsafe extern "C" fn(
    cutensorHandle_t,
    *mut cutensorPlan_t,
    cutensorOperationDescriptor_t,
    cutensorPlanPreference_t,
    u64, // workspace_size
) -> cutensorStatus_t;

pub type FnDestroyPlan = unsafe extern "C" fn(cutensorPlan_t) -> cutensorStatus_t;

// -- Execution --
pub type FnContract = unsafe extern "C" fn(
    cutensorHandle_t,
    cutensorPlan_t,
    *const c_void, // alpha
    *const c_void, // A data
    *const c_void, // B data
    *const c_void, // beta
    *const c_void, // C data
    *mut c_void,   // D data (output)
    *mut c_void,   // workspace
    u64,           // workspace_size
    *mut c_void,   // stream
) -> cutensorStatus_t;

pub type FnPermute = unsafe extern "C" fn(
    cutensorHandle_t,
    cutensorPlan_t,
    *const c_void, // alpha
    *const c_void, // A data
    *mut c_void,   // B data (output)
    *mut c_void,   // stream
) -> cutensorStatus_t;

pub type FnReduce = unsafe extern "C" fn(
    cutensorHandle_t,
    cutensorPlan_t,
    *const c_void, // alpha
    *const c_void, // A data
    *const c_void, // beta
    *const c_void, // C data
    *mut c_void,   // D data (output)
    *mut c_void,   // workspace
    u64,           // workspace_size
    *mut c_void,   // stream
) -> cutensorStatus_t;

pub type FnElementwiseBinaryExecute = unsafe extern "C" fn(
    cutensorHandle_t,
    cutensorPlan_t,
    *const c_void, // alpha
    *const c_void, // A data
    *const c_void, // gamma
    *const c_void, // C data
    *mut c_void,   // D data (output)
    *mut c_void,   // stream
) -> cutensorStatus_t;

pub type FnElementwiseTrinaryExecute = unsafe extern "C" fn(
    cutensorHandle_t,
    cutensorPlan_t,
    *const c_void, // alpha
    *const c_void, // A data
    *const c_void, // beta
    *const c_void, // B data
    *const c_void, // gamma
    *const c_void, // C data
    *mut c_void,   // D data (output)
    *mut c_void,   // stream
) -> cutensorStatus_t;

// ============================================================================
// CutensorVtable — 20 function pointers loaded via libloading
// ============================================================================

/// Vtable of cuTENSOR v2 function pointers, loaded at runtime via libloading.
///
/// Populated by [`CudaBackend::load`](super::cuda::CudaBackend::load).
/// Each field is a typed function pointer matching the exact cuTENSOR v2
/// C API signature.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CutensorVtable;
///
/// let lib = unsafe { libloading::Library::new("/usr/lib/libcutensor.so") }.unwrap();
/// let vtable = unsafe { CutensorVtable::load(&lib) }.unwrap();
/// ```
pub struct CutensorVtable {
    // Handle lifecycle (2)
    pub create: FnCreate,
    pub destroy: FnDestroy,

    // Tensor descriptor (2)
    pub create_tensor_descriptor: FnCreateTensorDescriptor,
    pub destroy_tensor_descriptor: FnDestroyTensorDescriptor,

    // Operation descriptors (6)
    pub create_contraction: FnCreateContraction,
    pub create_permutation: FnCreatePermutation,
    pub create_reduction: FnCreateReduction,
    pub create_elementwise_binary: FnCreateElementwiseBinary,
    pub create_elementwise_trinary: FnCreateElementwiseTrinary,
    pub destroy_operation_descriptor: FnDestroyOperationDescriptor,

    // Plan (5)
    pub create_plan_preference: FnCreatePlanPreference,
    pub destroy_plan_preference: FnDestroyPlanPreference,
    pub estimate_workspace_size: FnEstimateWorkspaceSize,
    pub create_plan: FnCreatePlan,
    pub destroy_plan: FnDestroyPlan,

    // Execution (5)
    pub contract: FnContract,
    pub permute: FnPermute,
    pub reduce: FnReduce,
    pub elementwise_binary_execute: FnElementwiseBinaryExecute,
    pub elementwise_trinary_execute: FnElementwiseTrinaryExecute,

    // Compute descriptor global symbols (loaded as pointers)
    pub compute_desc_32f: cutensorComputeDescriptor_t,
    pub compute_desc_64f: cutensorComputeDescriptor_t,
}

impl CutensorVtable {
    /// Load all cuTENSOR v2 function pointers from a libloading Library.
    ///
    /// # Safety
    ///
    /// The caller must ensure `lib` points to a valid cuTENSOR v2 shared
    /// library. Function pointer signatures must match the loaded library
    /// version.
    pub unsafe fn load(lib: &libloading::Library) -> std::result::Result<Self, libloading::Error> {
        Ok(Self {
            create: *lib.get(b"cutensorCreate\0")?,
            destroy: *lib.get(b"cutensorDestroy\0")?,
            create_tensor_descriptor: *lib.get(b"cutensorCreateTensorDescriptor\0")?,
            destroy_tensor_descriptor: *lib.get(b"cutensorDestroyTensorDescriptor\0")?,
            create_contraction: *lib.get(b"cutensorCreateContraction\0")?,
            create_permutation: *lib.get(b"cutensorCreatePermutation\0")?,
            create_reduction: *lib.get(b"cutensorCreateReduction\0")?,
            create_elementwise_binary: *lib.get(b"cutensorCreateElementwiseBinary\0")?,
            create_elementwise_trinary: *lib.get(b"cutensorCreateElementwiseTrinary\0")?,
            destroy_operation_descriptor: *lib.get(b"cutensorDestroyOperationDescriptor\0")?,
            create_plan_preference: *lib.get(b"cutensorCreatePlanPreference\0")?,
            destroy_plan_preference: *lib.get(b"cutensorDestroyPlanPreference\0")?,
            estimate_workspace_size: *lib.get(b"cutensorEstimateWorkspaceSize\0")?,
            create_plan: *lib.get(b"cutensorCreatePlan\0")?,
            destroy_plan: *lib.get(b"cutensorDestroyPlan\0")?,
            contract: *lib.get(b"cutensorContract\0")?,
            permute: *lib.get(b"cutensorPermute\0")?,
            reduce: *lib.get(b"cutensorReduce\0")?,
            elementwise_binary_execute: *lib.get(b"cutensorElementwiseBinaryExecute\0")?,
            elementwise_trinary_execute: *lib.get(b"cutensorElementwiseTrinaryExecute\0")?,
            compute_desc_32f: *lib
                .get::<cutensorComputeDescriptor_t>(b"CUTENSOR_COMPUTE_DESC_32F\0")?,
            compute_desc_64f: *lib
                .get::<cutensorComputeDescriptor_t>(b"CUTENSOR_COMPUTE_DESC_64F\0")?,
        })
    }
}
