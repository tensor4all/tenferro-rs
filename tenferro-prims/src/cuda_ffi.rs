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

use std::ffi::{c_void, CStr, CString};

#[cfg(unix)]
#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const std::os::raw::c_char, flags: i32) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
    fn dlsym(handle: *mut c_void, symbol: *const std::os::raw::c_char) -> *mut c_void;
    fn dlerror() -> *const std::os::raw::c_char;
}

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

/// cuTENSOR data type (mirrors `cutensorDataType_t` / `cudaDataType_t`).
pub type CutensorDataType = i32;
/// 32-bit float (`float`).
pub const CUTENSOR_R_32F: CutensorDataType = 0;
/// 64-bit float (`double`).
pub const CUTENSOR_R_64F: CutensorDataType = 1;
/// 32-bit complex (`cuComplex`).
pub const CUTENSOR_C_32F: CutensorDataType = 4;
/// 64-bit complex (`cuDoubleComplex`).
pub const CUTENSOR_C_64F: CutensorDataType = 5;

/// cuTENSOR unary/binary operator (mirrors `cutensorOperator_t`).
pub type CutensorOperator = i32;
/// Identity (no transformation).
pub const CUTENSOR_OP_IDENTITY: CutensorOperator = 1;
/// Addition.
pub const CUTENSOR_OP_ADD: CutensorOperator = 3;
/// Multiplication.
pub const CUTENSOR_OP_MUL: CutensorOperator = 5;
/// Maximum.
pub const CUTENSOR_OP_MAX: CutensorOperator = 6;
/// Minimum.
pub const CUTENSOR_OP_MIN: CutensorOperator = 7;
/// Complex conjugate.
pub const CUTENSOR_OP_CONJ: CutensorOperator = 9;

/// cuTENSOR algorithm selection (mirrors `cutensorAlgo_t`).
pub type CutensorAlgo = i32;
/// Let cuTENSOR choose the best algorithm.
pub const CUTENSOR_ALGO_DEFAULT: CutensorAlgo = -1;

/// cuTENSOR JIT compilation mode (mirrors `cutensorJitMode_t`).
pub type CutensorJitMode = i32;
/// Disable JIT compilation.
pub const CUTENSOR_JIT_MODE_NONE: CutensorJitMode = 0;

/// cuTENSOR workspace size preference (mirrors `cutensorWorksizePreference_t`).
pub type CutensorWorksizePref = i32;
/// Minimum workspace.
pub const CUTENSOR_WORKSPACE_MIN: CutensorWorksizePref = 1;
/// Recommended workspace (balance of speed and memory).
pub const CUTENSOR_WORKSPACE_DEFAULT: CutensorWorksizePref = 2;
/// Maximum workspace (fastest execution).
pub const CUTENSOR_WORKSPACE_MAX: CutensorWorksizePref = 3;

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
    CutensorOperator,
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
    CutensorOperator, // op_ac
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
    CutensorOperator, // op_ab
    CutensorOperator, // op_abc
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
#[derive(Debug)]
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

#[cfg(unix)]
fn dlerror_string() -> String {
    unsafe {
        let err = dlerror();
        if err.is_null() {
            "unknown dlerror".into()
        } else {
            CStr::from_ptr(err).to_string_lossy().into_owned()
        }
    }
}

#[cfg(unix)]
unsafe fn load_symbol<T: Copy>(
    handle: *mut c_void,
    symbol: &str,
) -> std::result::Result<T, String> {
    let symbol = CString::new(symbol).map_err(|e| e.to_string())?;
    let ptr = dlsym(handle, symbol.as_ptr());
    if ptr.is_null() {
        Err(dlerror_string())
    } else {
        Ok(std::mem::transmute_copy(&ptr))
    }
}

#[cfg(unix)]
unsafe fn load_data_symbol<T: Copy>(
    handle: *mut c_void,
    symbol: &str,
) -> std::result::Result<T, String> {
    let symbol = CString::new(symbol).map_err(|e| e.to_string())?;
    let ptr = dlsym(handle, symbol.as_ptr());
    if ptr.is_null() {
        Err(dlerror_string())
    } else {
        Ok(*(ptr as *const T))
    }
}

#[cfg(unix)]
pub(crate) struct DynamicLibrary {
    handle: *mut c_void,
}

/// # Safety
///
/// `DynamicLibrary` can be safely sent across threads because:
/// - The handle is an opaque `dlopen` pointer that represents loaded library state
/// - The library handle is managed by the dynamic linker and is thread-safe for symbol lookup
/// - The handle remains valid as long as the `DynamicLibrary` exists (until `dlclose` in Drop)
/// - Symbol loading via `dlsym` is thread-safe on POSIX systems
unsafe impl Send for DynamicLibrary {}

/// # Safety
///
/// `DynamicLibrary` can be safely shared across threads because:
/// - The `dlopen`/`dlsym`/`dlclose` functions are thread-safe on POSIX systems
/// - Multiple threads can safely call `dlsym` on the same handle concurrently
/// - The handle is read-only after construction - only Drop modifies it via `dlclose`
/// - Drop uses `dlclose` which is safe to call once; the Rust borrow checker ensures
///   no concurrent access during Drop since `&mut self` is required
unsafe impl Sync for DynamicLibrary {}

#[cfg(unix)]
impl DynamicLibrary {
    pub unsafe fn open(path: &str, flags: i32) -> std::result::Result<Self, String> {
        let path = CString::new(path).map_err(|e| e.to_string())?;
        let handle = dlopen(path.as_ptr(), flags);
        if handle.is_null() {
            Err(dlerror_string())
        } else {
            Ok(Self { handle })
        }
    }

    pub fn handle(&self) -> *mut c_void {
        self.handle
    }
}

#[cfg(unix)]
impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = dlclose(self.handle);
            }
        }
    }
}

impl CutensorVtable {
    /// Load all cuTENSOR v2 function pointers from a libloading Library.
    ///
    /// # Safety
    ///
    /// The caller must ensure `lib` points to a valid cuTENSOR v2 shared
    /// library. Function pointer signatures must match the loaded library
    /// version.
    #[cfg(unix)]
    pub unsafe fn load(handle: *mut c_void) -> std::result::Result<Self, String> {
        Ok(Self {
            create: load_symbol(handle, "cutensorCreate")?,
            destroy: load_symbol(handle, "cutensorDestroy")?,
            create_tensor_descriptor: load_symbol(handle, "cutensorCreateTensorDescriptor")?,
            destroy_tensor_descriptor: load_symbol(handle, "cutensorDestroyTensorDescriptor")?,
            create_contraction: load_symbol(handle, "cutensorCreateContraction")?,
            create_permutation: load_symbol(handle, "cutensorCreatePermutation")?,
            create_reduction: load_symbol(handle, "cutensorCreateReduction")?,
            create_elementwise_binary: load_symbol(handle, "cutensorCreateElementwiseBinary")?,
            create_elementwise_trinary: load_symbol(handle, "cutensorCreateElementwiseTrinary")?,
            destroy_operation_descriptor: load_symbol(
                handle,
                "cutensorDestroyOperationDescriptor",
            )?,
            create_plan_preference: load_symbol(handle, "cutensorCreatePlanPreference")?,
            destroy_plan_preference: load_symbol(handle, "cutensorDestroyPlanPreference")?,
            estimate_workspace_size: load_symbol(handle, "cutensorEstimateWorkspaceSize")?,
            create_plan: load_symbol(handle, "cutensorCreatePlan")?,
            destroy_plan: load_symbol(handle, "cutensorDestroyPlan")?,
            contract: load_symbol(handle, "cutensorContract")?,
            permute: load_symbol(handle, "cutensorPermute")?,
            reduce: load_symbol(handle, "cutensorReduce")?,
            elementwise_binary_execute: load_symbol(handle, "cutensorElementwiseBinaryExecute")?,
            elementwise_trinary_execute: load_symbol(handle, "cutensorElementwiseTrinaryExecute")?,
            compute_desc_32f: load_data_symbol(handle, "CUTENSOR_COMPUTE_DESC_32F")?,
            compute_desc_64f: load_data_symbol(handle, "CUTENSOR_COMPUTE_DESC_64F")?,
        })
    }
}
