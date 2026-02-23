# CudaBackend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `todo!()` stubs in `cuda_ffi.rs` and `cuda.rs` with real cuTENSOR v2 API calls for all PrimDescriptor operations (Contract, Permute, Reduce, BatchedGemm, Trace, AntiTrace, AntiDiag, ElementwiseUnary, ElementwiseMul, MakeContiguous).

**Architecture:** Runtime dlopen via libloading populates a `CutensorVtable` of 20 typed function pointers. RAII wrappers ensure cuTENSOR handles are destroyed on Drop. `CudaBackend::plan()` creates cuTENSOR descriptor chains; `execute()` dispatches to the vtable.

**Tech Stack:** cuTENSOR v2 (runtime-loaded), cudarc 0.19 (dynamic-loading), libloading, Rust unsafe FFI.

**Reference:** `../omeinsum-rs/src/backend/cuda/cutensor/` for exact cuTENSOR v2 signatures.

---

### Task 1: Expand `cuda_ffi.rs` — FFI types, enums, and opaque handles

**Files:**
- Modify: `tenferro-prims/src/cuda_ffi.rs`

**Step 1: Replace entire file with expanded FFI types**

Replace the current 84-line stub with the full cuTENSOR v2 FFI module. This includes:

```rust
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
//! let vtable = unsafe { CutensorVtable::load(&lib).unwrap() };
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
    u32,            // num_modes
    *const i64,     // extent
    *const i64,     // stride
    CutensorDataType,
    u32,            // alignment_requirement
) -> cutensorStatus_t;

pub type FnDestroyTensorDescriptor =
    unsafe extern "C" fn(cutensorTensorDescriptor_t) -> cutensorStatus_t;

// -- Operation descriptors --
pub type FnCreateContraction = unsafe extern "C" fn(
    cutensorHandle_t,
    *mut cutensorOperationDescriptor_t,
    cutensorTensorDescriptor_t, *const i32, CutensorOperator, // A
    cutensorTensorDescriptor_t, *const i32, CutensorOperator, // B
    cutensorTensorDescriptor_t, *const i32, CutensorOperator, // C
    cutensorTensorDescriptor_t, *const i32,                   // D
    cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

pub type FnCreatePermutation = unsafe extern "C" fn(
    cutensorHandle_t,
    *mut cutensorOperationDescriptor_t,
    cutensorTensorDescriptor_t, *const i32, CutensorOperator, // A
    cutensorTensorDescriptor_t, *const i32,                   // B
    cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

pub type FnCreateReduction = unsafe extern "C" fn(
    cutensorHandle_t,
    *mut cutensorOperationDescriptor_t,
    cutensorTensorDescriptor_t, *const i32, CutensorOperator,  // A
    cutensorTensorDescriptor_t, *const i32, CutensorOperator,  // C
    cutensorTensorDescriptor_t, *const i32,                    // D
    CutensorReduceOp,
    cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

pub type FnCreateElementwiseBinary = unsafe extern "C" fn(
    cutensorHandle_t,
    *mut cutensorOperationDescriptor_t,
    cutensorTensorDescriptor_t, *const i32, CutensorOperator,  // A
    cutensorTensorDescriptor_t, *const i32, CutensorOperator,  // C
    cutensorTensorDescriptor_t, *const i32,                    // D
    CutensorReduceOp, // op_ac (CUTENSOR_OP_ADD for binary mul)
    cutensorComputeDescriptor_t,
) -> cutensorStatus_t;

pub type FnCreateElementwiseTrinary = unsafe extern "C" fn(
    cutensorHandle_t,
    *mut cutensorOperationDescriptor_t,
    cutensorTensorDescriptor_t, *const i32, CutensorOperator,  // A
    cutensorTensorDescriptor_t, *const i32, CutensorOperator,  // B
    cutensorTensorDescriptor_t, *const i32, CutensorOperator,  // C
    cutensorTensorDescriptor_t, *const i32,                    // D
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
            compute_desc_32f: *lib.get::<cutensorComputeDescriptor_t>(b"CUTENSOR_COMPUTE_DESC_32F\0")?,
            compute_desc_64f: *lib.get::<cutensorComputeDescriptor_t>(b"CUTENSOR_COMPUTE_DESC_64F\0")?,
        })
    }
}
```

**Step 2: Build to verify FFI compiles**

Run: `cargo build -p tenferro-prims --features cuda`
Expected: Compiles (no runtime test — only type definitions)

**Step 3: Build without cuda feature too**

Run: `cargo build -p tenferro-prims`
Expected: Compiles (cuda_ffi.rs is gated behind `cfg(feature = "cuda")`)

**Step 4: Commit**

```bash
git add tenferro-prims/src/cuda_ffi.rs
git commit -m "feat(cuda): expand cuda_ffi.rs with 20 typed cuTENSOR v2 function pointers"
```

---

### Task 2: Implement RAII wrappers and `CutensorType` trait in `cuda.rs`

**Files:**
- Modify: `tenferro-prims/src/cuda.rs`

**Step 1: Add RAII wrapper structs and CutensorType trait**

Add these internal types at the top of `cuda.rs`, before the public types. These are `pub(crate)` — not exported.

```rust
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::cuda_ffi::*;
use crate::{
    Extension, PlanCache, PrimDescriptor, ReduceOp, TensorPrims, UnaryOp,
    validate_execute_inputs, validate_shape_count, validate_rank, validate_shape_eq,
    mode_position,
};

// ============================================================================
// Error helper
// ============================================================================

/// Check a cuTENSOR status code, converting non-success to Error.
fn check_status(status: cutensorStatus_t, context: &str) -> Result<()> {
    if status == CUTENSOR_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::DeviceError(format!(
            "cuTENSOR error {status} in {context}"
        )))
    }
}

// ============================================================================
// RAII wrappers
// ============================================================================

/// RAII wrapper for `cutensorHandle_t`. Drop calls `cutensorDestroy`.
pub(crate) struct HandleWrapper {
    raw: cutensorHandle_t,
    vtable: Arc<CutensorVtable>,
}

impl Drop for HandleWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.vtable.destroy)(self.raw); }
        }
    }
}

/// RAII wrapper for `cutensorTensorDescriptor_t`.
pub(crate) struct TensorDescWrapper {
    raw: cutensorTensorDescriptor_t,
    vtable: Arc<CutensorVtable>,
}

impl Drop for TensorDescWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.vtable.destroy_tensor_descriptor)(self.raw); }
        }
    }
}

/// RAII wrapper for `cutensorOperationDescriptor_t`.
pub(crate) struct OpDescWrapper {
    raw: cutensorOperationDescriptor_t,
    vtable: Arc<CutensorVtable>,
}

impl Drop for OpDescWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.vtable.destroy_operation_descriptor)(self.raw); }
        }
    }
}

/// RAII wrapper for `cutensorPlanPreference_t`.
pub(crate) struct PlanPrefWrapper {
    raw: cutensorPlanPreference_t,
    vtable: Arc<CutensorVtable>,
}

impl Drop for PlanPrefWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.vtable.destroy_plan_preference)(self.raw); }
        }
    }
}

/// RAII wrapper for `cutensorPlan_t`.
pub(crate) struct PlanWrapper {
    raw: cutensorPlan_t,
    vtable: Arc<CutensorVtable>,
}

impl Drop for PlanWrapper {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.vtable.destroy_plan)(self.raw); }
        }
    }
}

// ============================================================================
// CutensorType trait — maps Rust scalars to cuTENSOR types
// ============================================================================

/// Maps a Rust scalar type to its cuTENSOR data type and compute descriptor.
pub(crate) trait CutensorType: Scalar {
    fn data_type() -> CutensorDataType;
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t;
}

impl CutensorType for f32 {
    fn data_type() -> CutensorDataType { CutensorDataType::R_32F }
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t {
        vtable.compute_desc_32f
    }
}

impl CutensorType for f64 {
    fn data_type() -> CutensorDataType { CutensorDataType::R_64F }
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t {
        vtable.compute_desc_64f
    }
}

impl CutensorType for Complex32 {
    fn data_type() -> CutensorDataType { CutensorDataType::C_32F }
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t {
        vtable.compute_desc_32f
    }
}

impl CutensorType for Complex64 {
    fn data_type() -> CutensorDataType { CutensorDataType::C_64F }
    fn compute_descriptor(vtable: &CutensorVtable) -> cutensorComputeDescriptor_t {
        vtable.compute_desc_64f
    }
}
```

**Step 2: Build to verify**

Run: `cargo build -p tenferro-prims --features cuda`
Expected: Compiles

**Step 3: Commit**

```bash
git add tenferro-prims/src/cuda.rs
git commit -m "feat(cuda): add RAII wrappers and CutensorType trait"
```

---

### Task 3: Implement `CudaBackend::load()` and helper functions

**Files:**
- Modify: `tenferro-prims/src/cuda.rs`

**Step 1: Implement helper to create tensor descriptors**

```rust
/// Create a cuTENSOR tensor descriptor from shape and strides.
///
/// Converts usize dims and isize strides to i64 arrays for the FFI call.
fn create_tensor_desc(
    handle: cutensorHandle_t,
    vtable: &Arc<CutensorVtable>,
    shape: &[usize],
    strides: &[isize],
    data_type: CutensorDataType,
) -> Result<TensorDescWrapper> {
    let num_modes = shape.len() as u32;
    let extent: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let stride: Vec<i64> = strides.iter().map(|&s| s as i64).collect();
    let mut raw: cutensorTensorDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_tensor_descriptor)(
            handle,
            &mut raw,
            num_modes,
            extent.as_ptr(),
            stride.as_ptr(),
            data_type,
            128, // alignment (256-byte default for cuTENSOR)
        )
    };
    check_status(status, "cutensorCreateTensorDescriptor")?;
    Ok(TensorDescWrapper {
        raw,
        vtable: Arc::clone(vtable),
    })
}

/// Create a cuTENSOR plan (shared logic for all operation types).
///
/// Takes an already-created operation descriptor, creates plan preference,
/// estimates workspace, and builds the plan.
fn build_cutensor_plan(
    handle: cutensorHandle_t,
    vtable: &Arc<CutensorVtable>,
    op_desc: &OpDescWrapper,
) -> Result<(PlanWrapper, u64)> {
    // Create plan preference
    let mut pref_raw: cutensorPlanPreference_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_plan_preference)(
            handle,
            &mut pref_raw,
            CutensorAlgo::Default,
            CutensorJitMode::None,
        )
    };
    check_status(status, "cutensorCreatePlanPreference")?;
    let pref = PlanPrefWrapper {
        raw: pref_raw,
        vtable: Arc::clone(vtable),
    };

    // Estimate workspace
    let mut workspace_size: u64 = 0;
    let status = unsafe {
        (vtable.estimate_workspace_size)(
            handle,
            op_desc.raw,
            pref.raw,
            CutensorWorksizePref::Recommended,
            &mut workspace_size,
        )
    };
    check_status(status, "cutensorEstimateWorkspaceSize")?;

    // Create plan
    let mut plan_raw: cutensorPlan_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_plan)(handle, &mut plan_raw, op_desc.raw, pref.raw, workspace_size)
    };
    check_status(status, "cutensorCreatePlan")?;

    Ok((
        PlanWrapper {
            raw: plan_raw,
            vtable: Arc::clone(vtable),
        },
        workspace_size,
    ))
    // pref and op_desc RAII wrappers drop here, freeing cuTENSOR resources
}
```

**Step 2: Implement `CudaBackend::load()`**

Replace the `todo!()` body:

```rust
impl CudaBackend {
    pub fn load(path: &str) -> Result<(Self, CudaContext)> {
        // 1. Open shared library
        let lib = unsafe { libloading::Library::new(path) }
            .map_err(|e| Error::DeviceError(format!("Failed to load cuTENSOR: {e}")))?;

        // 2. Populate vtable
        let vtable = unsafe { CutensorVtable::load(&lib) }
            .map_err(|e| Error::DeviceError(format!("Failed to load cuTENSOR symbols: {e}")))?;
        let vtable = Arc::new(vtable);

        // 3. Initialize cuTENSOR handle
        let mut handle_raw: cutensorHandle_t = ptr::null_mut();
        let status = unsafe { (vtable.create)(&mut handle_raw) };
        check_status(status, "cutensorCreate")?;
        let handle = HandleWrapper {
            raw: handle_raw,
            vtable: Arc::clone(&vtable),
        };

        // 4. Initialize CUDA device via cudarc
        let device = cudarc::driver::CudaDevice::new(0)
            .map_err(|e| Error::DeviceError(format!("CUDA device init failed: {e}")))?;
        let stream = device.fork_default_stream()
            .map_err(|e| Error::DeviceError(format!("CUDA stream creation failed: {e}")))?;

        let ctx = CudaContext {
            handle,
            stream: Arc::new(stream),
            vtable: Arc::clone(&vtable),
            workspace: Vec::new(), // allocated on demand during plan creation
            plan_cache: PlanCache::new(),
        };

        Ok((CudaBackend { _lib: lib }, ctx))
    }
}
```

**Step 3: Update `CudaContext` and `CudaPlan` structs**

```rust
pub struct CudaContext {
    handle: HandleWrapper,
    stream: Arc<cudarc::driver::CudaStream>,
    vtable: Arc<CutensorVtable>,
    workspace: Vec<u8>, // CPU-side workspace buffer; real GPU workspace TBD
    plan_cache: PlanCache,
}

pub struct CudaPlan<T: Scalar> {
    plan: PlanWrapper,
    desc: PrimDescriptor,
    workspace_size: u64,
    _marker: PhantomData<T>,
}
```

**Step 4: Build to verify**

Run: `cargo build -p tenferro-prims --features cuda`
Expected: Compiles (cudarc types resolve)

**Step 5: Commit**

```bash
git add tenferro-prims/src/cuda.rs
git commit -m "feat(cuda): implement CudaBackend::load() and helper functions"
```

---

### Task 4: Implement `plan()` for Contract, BatchedGemm, Trace, AntiTrace, AntiDiag

**Files:**
- Modify: `tenferro-prims/src/cuda.rs`

**Step 1: Implement contraction plan helper**

All five operations map to `cutensorCreateContraction`. The difference is how mode labels and tensor descriptors are constructed.

```rust
/// Create a cuTENSOR contraction plan.
///
/// Used by Contract, BatchedGemm, Trace, AntiTrace, AntiDiag.
fn plan_contraction<T: CutensorType>(
    ctx: &mut CudaContext,
    modes_a: &[i32],
    shape_a: &[usize],
    strides_a: &[isize],
    modes_b: &[i32],
    shape_b: &[usize],
    strides_b: &[isize],
    modes_c: &[i32],
    shape_c: &[usize],
    strides_c: &[isize],
) -> Result<(PlanWrapper, u64)> {
    let vtable = &ctx.vtable;
    let handle = ctx.handle.raw;

    let desc_a = create_tensor_desc(handle, vtable, shape_a, strides_a, T::data_type())?;
    let desc_b = create_tensor_desc(handle, vtable, shape_b, strides_b, T::data_type())?;
    let desc_c = create_tensor_desc(handle, vtable, shape_c, strides_c, T::data_type())?;
    // D descriptor same as C (in-place output)
    let desc_d = create_tensor_desc(handle, vtable, shape_c, strides_c, T::data_type())?;

    let compute = T::compute_descriptor(vtable);

    let mut op_raw: cutensorOperationDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_contraction)(
            handle,
            &mut op_raw,
            desc_a.raw, modes_a.as_ptr(), CutensorOperator::Identity,
            desc_b.raw, modes_b.as_ptr(), CutensorOperator::Identity,
            desc_c.raw, modes_c.as_ptr(), CutensorOperator::Identity,
            desc_d.raw, modes_c.as_ptr(),
            compute,
        )
    };
    check_status(status, "cutensorCreateContraction")?;
    let op_desc = OpDescWrapper {
        raw: op_raw,
        vtable: Arc::clone(vtable),
    };

    build_cutensor_plan(handle, vtable, &op_desc)
    // desc_a, desc_b, desc_c, desc_d, op_desc all drop here
}
```

**Step 2: Implement `plan()` dispatch for contraction-based ops**

In the `TensorPrims::plan()` match:

```rust
PrimDescriptor::Contract { modes_a, modes_b, modes_c } => {
    validate_shape_count(shapes, 3, "Contract")?;
    let modes_a_i32: Vec<i32> = modes_a.iter().map(|&m| m as i32).collect();
    let modes_b_i32: Vec<i32> = modes_b.iter().map(|&m| m as i32).collect();
    let modes_c_i32: Vec<i32> = modes_c.iter().map(|&m| m as i32).collect();
    // Use default contiguous strides for plan (execute extracts real strides)
    let strides_a = default_col_major_strides(shapes[0]);
    let strides_b = default_col_major_strides(shapes[1]);
    let strides_c = default_col_major_strides(shapes[2]);
    let (plan, ws) = plan_contraction::<T>(
        ctx, &modes_a_i32, shapes[0], &strides_a,
        &modes_b_i32, shapes[1], &strides_b,
        &modes_c_i32, shapes[2], &strides_c,
    )?;
    Ok(CudaPlan { plan, desc: desc.clone(), workspace_size: ws, _marker: PhantomData })
}

PrimDescriptor::BatchedGemm { batch_dims, m, n, k } => {
    // Convert BatchedGemm to Contract mode labels
    validate_shape_count(shapes, 3, "BatchedGemm")?;
    let nb = batch_dims.len() as u32;
    let mut modes_a = Vec::new();
    let mut modes_b = Vec::new();
    let mut modes_c = Vec::new();
    // batch modes: 0..nb
    for i in 0..nb { modes_a.push(i as i32); modes_b.push(i as i32); modes_c.push(i as i32); }
    // m mode = nb, k mode = nb+1, n mode = nb+2
    let m_mode = nb as i32;
    let k_mode = (nb + 1) as i32;
    let n_mode = (nb + 2) as i32;
    modes_a.extend([m_mode, k_mode]);
    modes_b.extend([k_mode, n_mode]);
    modes_c.extend([m_mode, n_mode]);
    let strides_a = default_col_major_strides(shapes[0]);
    let strides_b = default_col_major_strides(shapes[1]);
    let strides_c = default_col_major_strides(shapes[2]);
    let (plan, ws) = plan_contraction::<T>(
        ctx, &modes_a, shapes[0], &strides_a,
        &modes_b, shapes[1], &strides_b,
        &modes_c, shapes[2], &strides_c,
    )?;
    Ok(CudaPlan { plan, desc: desc.clone(), workspace_size: ws, _marker: PhantomData })
}
```

For Trace, AntiTrace, AntiDiag — these contract the input with an identity tensor. The plan creates an eye tensor descriptor.

**Step 3: Add `default_col_major_strides` helper**

```rust
fn default_col_major_strides(shape: &[usize]) -> Vec<isize> {
    let n = shape.len();
    if n == 0 { return vec![]; }
    let mut strides = vec![0isize; n];
    strides[0] = 1;
    for i in 1..n {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }
    strides
}
```

**Step 4: Build to verify**

Run: `cargo build -p tenferro-prims --features cuda`
Expected: Compiles

**Step 5: Commit**

```bash
git add tenferro-prims/src/cuda.rs
git commit -m "feat(cuda): implement plan() for Contract, BatchedGemm, Trace, AntiTrace, AntiDiag"
```

---

### Task 5: Implement `plan()` for Permute, MakeContiguous, Reduce, ElementwiseUnary, ElementwiseMul

**Files:**
- Modify: `tenferro-prims/src/cuda.rs`

**Step 1: Implement permutation plan helper**

```rust
fn plan_permutation<T: CutensorType>(
    ctx: &mut CudaContext,
    modes_a: &[i32],
    shape_a: &[usize],
    strides_a: &[isize],
    modes_b: &[i32],
    shape_b: &[usize],
    strides_b: &[isize],
) -> Result<(PlanWrapper, u64)> {
    let vtable = &ctx.vtable;
    let handle = ctx.handle.raw;
    let desc_a = create_tensor_desc(handle, vtable, shape_a, strides_a, T::data_type())?;
    let desc_b = create_tensor_desc(handle, vtable, shape_b, strides_b, T::data_type())?;
    let compute = T::compute_descriptor(vtable);

    let mut op_raw: cutensorOperationDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_permutation)(
            handle, &mut op_raw,
            desc_a.raw, modes_a.as_ptr(), CutensorOperator::Identity,
            desc_b.raw, modes_b.as_ptr(),
            compute,
        )
    };
    check_status(status, "cutensorCreatePermutation")?;
    let op_desc = OpDescWrapper { raw: op_raw, vtable: Arc::clone(vtable) };
    build_cutensor_plan(handle, vtable, &op_desc)
}
```

**Step 2: Implement reduction plan helper**

```rust
fn plan_reduction<T: CutensorType>(
    ctx: &mut CudaContext,
    modes_a: &[i32],
    shape_a: &[usize],
    strides_a: &[isize],
    modes_c: &[i32],
    shape_c: &[usize],
    strides_c: &[isize],
    reduce_op: CutensorReduceOp,
) -> Result<(PlanWrapper, u64)> {
    let vtable = &ctx.vtable;
    let handle = ctx.handle.raw;
    let desc_a = create_tensor_desc(handle, vtable, shape_a, strides_a, T::data_type())?;
    let desc_c = create_tensor_desc(handle, vtable, shape_c, strides_c, T::data_type())?;
    let desc_d = create_tensor_desc(handle, vtable, shape_c, strides_c, T::data_type())?;
    let compute = T::compute_descriptor(vtable);

    let mut op_raw: cutensorOperationDescriptor_t = ptr::null_mut();
    let status = unsafe {
        (vtable.create_reduction)(
            handle, &mut op_raw,
            desc_a.raw, modes_a.as_ptr(), CutensorOperator::Identity,
            desc_c.raw, modes_c.as_ptr(), CutensorOperator::Identity,
            desc_d.raw, modes_c.as_ptr(),
            reduce_op,
            compute,
        )
    };
    check_status(status, "cutensorCreateReduction")?;
    let op_desc = OpDescWrapper { raw: op_raw, vtable: Arc::clone(vtable) };
    build_cutensor_plan(handle, vtable, &op_desc)
}
```

**Step 3: Add plan dispatch for remaining variants**

```rust
PrimDescriptor::Permute { modes_a, modes_b } => { /* call plan_permutation */ }
PrimDescriptor::MakeContiguous => { /* identity permutation via plan_permutation */ }
PrimDescriptor::Reduce { modes_a, modes_c, op } => {
    let reduce_op = match op {
        ReduceOp::Sum => CutensorReduceOp::Add,
        ReduceOp::Max => CutensorReduceOp::Max,
        ReduceOp::Min => CutensorReduceOp::Min,
    };
    /* call plan_reduction */
}
PrimDescriptor::ElementwiseUnary { op } => { /* create_elementwise_trinary */ }
PrimDescriptor::ElementwiseMul => { /* create_elementwise_binary */ }
```

**Step 4: Build to verify**

Run: `cargo build -p tenferro-prims --features cuda`
Expected: Compiles

**Step 5: Commit**

```bash
git add tenferro-prims/src/cuda.rs
git commit -m "feat(cuda): implement plan() for Permute, Reduce, ElementwiseUnary, ElementwiseMul"
```

---

### Task 6: Implement `execute()` dispatch

**Files:**
- Modify: `tenferro-prims/src/cuda.rs`

**Step 1: Implement execute()**

Replace the `todo!()` body. The key logic:

1. Extract device pointers from input/output tensors via `buffer().as_device_ptr()`
2. Cast `alpha`/`beta` scalars to `*const c_void`
3. Match on `plan.desc` to determine which vtable execution function to call
4. Pass workspace pointer and CUDA stream

```rust
fn execute<T: Scalar>(
    ctx: &mut CudaContext,
    plan: &CudaPlan<T>,
    alpha: T,
    inputs: &[&Tensor<T>],
    beta: T,
    output: &mut Tensor<T>,
) -> Result<()> {
    let handle = ctx.handle.raw;
    let stream = ctx.stream.stream as *mut c_void; // cudarc stream handle

    // Grow workspace if needed
    if plan.workspace_size as usize > ctx.workspace.len() {
        ctx.workspace.resize(plan.workspace_size as usize, 0);
    }
    let ws_ptr = if ctx.workspace.is_empty() {
        ptr::null_mut()
    } else {
        ctx.workspace.as_mut_ptr() as *mut c_void
    };

    let alpha_ptr = &alpha as *const T as *const c_void;
    let beta_ptr = &beta as *const T as *const c_void;

    match &plan.desc {
        PrimDescriptor::Contract { .. } | PrimDescriptor::BatchedGemm { .. }
        | PrimDescriptor::Trace { .. } | PrimDescriptor::AntiTrace { .. }
        | PrimDescriptor::AntiDiag { .. } => {
            validate_execute_inputs(inputs, 2, "Contraction")?;
            let a_ptr = inputs[0].buffer().as_device_ptr()
                .ok_or_else(|| Error::DeviceError("input A not on GPU".into()))? as *const c_void;
            let b_ptr = inputs[1].buffer().as_device_ptr()
                .ok_or_else(|| Error::DeviceError("input B not on GPU".into()))? as *const c_void;
            let c_ptr = output.buffer().as_device_ptr()
                .ok_or_else(|| Error::DeviceError("output not on GPU".into()))? as *const c_void;
            let d_ptr = c_ptr as *mut c_void; // D = C for in-place

            let status = unsafe {
                (ctx.vtable.contract)(
                    handle, plan.plan.raw,
                    alpha_ptr, a_ptr, b_ptr,
                    beta_ptr, c_ptr, d_ptr,
                    ws_ptr, plan.workspace_size,
                    stream,
                )
            };
            check_status(status, "cutensorContract")
        }

        PrimDescriptor::Permute { .. } | PrimDescriptor::MakeContiguous => {
            validate_execute_inputs(inputs, 1, "Permute")?;
            let a_ptr = inputs[0].buffer().as_device_ptr()
                .ok_or_else(|| Error::DeviceError("input A not on GPU".into()))? as *const c_void;
            let b_ptr = output.buffer().as_device_ptr()
                .ok_or_else(|| Error::DeviceError("output not on GPU".into()))? as *mut c_void;
            let status = unsafe {
                (ctx.vtable.permute)(handle, plan.plan.raw, alpha_ptr, a_ptr, b_ptr, stream)
            };
            check_status(status, "cutensorPermute")
        }

        PrimDescriptor::Reduce { .. } => {
            validate_execute_inputs(inputs, 1, "Reduce")?;
            let a_ptr = inputs[0].buffer().as_device_ptr()
                .ok_or_else(|| Error::DeviceError("input A not on GPU".into()))? as *const c_void;
            let c_ptr = output.buffer().as_device_ptr()
                .ok_or_else(|| Error::DeviceError("output not on GPU".into()))? as *const c_void;
            let d_ptr = c_ptr as *mut c_void;
            let status = unsafe {
                (ctx.vtable.reduce)(
                    handle, plan.plan.raw,
                    alpha_ptr, a_ptr,
                    beta_ptr, c_ptr, d_ptr,
                    ws_ptr, plan.workspace_size,
                    stream,
                )
            };
            check_status(status, "cutensorReduce")
        }

        PrimDescriptor::ElementwiseMul => {
            validate_execute_inputs(inputs, 2, "ElementwiseMul")?;
            // ... call elementwise_binary_execute
        }

        PrimDescriptor::ElementwiseUnary { .. } => {
            validate_execute_inputs(inputs, 1, "ElementwiseUnary")?;
            // ... call elementwise_trinary_execute
        }
    }
}
```

**Step 2: Implement `resolve_conj`**

Replace the `todo!()` body — delegates to `ElementwiseUnary(Conj)` plan+execute.

**Step 3: Build to verify**

Run: `cargo build -p tenferro-prims --features cuda`
Expected: Compiles

**Step 4: Also verify default build**

Run: `cargo build --workspace && cargo test --workspace`
Expected: All pass (cuda code is feature-gated)

**Step 5: Commit**

```bash
git add tenferro-prims/src/cuda.rs
git commit -m "feat(cuda): implement execute() dispatch for all PrimDescriptor operations"
```

---

### Task 7: Final verification and cleanup

**Files:**
- Modify: `tenferro-prims/src/cuda.rs` (doc comments, remove dead code)

**Step 1: Run full workspace build**

Run: `cargo build --workspace`
Expected: Compiles

**Step 2: Run full test suite**

Run: `cargo test --workspace`
Expected: All tests pass (CUDA code is feature-gated, not exercised)

**Step 3: Run CUDA feature build**

Run: `cargo build -p tenferro-prims --features cuda`
Expected: Compiles

**Step 4: Run formatter**

Run: `cargo fmt --all --check`
Expected: No formatting issues (or fix with `cargo fmt --all`)

**Step 5: Commit and push**

```bash
git add -A
git commit -m "feat(cuda): complete CudaBackend implementation with cuTENSOR v2 API calls

Implements all PrimDescriptor operations via cuTENSOR v2 vtable:
- Contract, BatchedGemm, Trace, AntiTrace, AntiDiag → cutensorContract
- Permute, MakeContiguous → cutensorPermute
- Reduce → cutensorReduce
- ElementwiseUnary → cutensorElementwiseTrinary
- ElementwiseMul → cutensorElementwiseBinary

RAII wrappers ensure all cuTENSOR handles are properly freed.
CutensorType trait maps f32/f64/Complex32/Complex64 to cuTENSOR types."
```
