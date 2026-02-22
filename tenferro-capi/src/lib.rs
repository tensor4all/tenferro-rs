//! C-API (FFI) for tenferro.
//!
//! Exposes tensor lifecycle, einsum, SVD (including AD rules), and DLPack
//! interop to host languages such as Julia, Python (JAX, PyTorch), and C/C++.
//!
//! # Design principles
//!
//! - **Opaque pointers**: `TfeTensorF64` is an opaque handle wrapping
//!   `Tensor<f64>`. Host languages never see Rust internals.
//! - **Status codes**: Every function takes a `*mut tfe_status_t` as its
//!   last argument. Rust panics are caught with `catch_unwind` and
//!   converted to `TFE_INTERNAL_ERROR`.
//! - **Stateless AD rules**: Only `rrule` (VJP) and `frule` (JVP) are
//!   exposed. The AD tape / `TrackedTensor` / `DualTensor` are Rust-internal
//!   and **not** exposed via FFI. Host languages manage their own AD tapes
//!   (ChainRules.jl, PyTorch autograd, JAX custom_vjp).
//! - **f64 only** in this POC phase. All functions carry the `_f64` suffix.
//! - **DLPack interop**: Zero-copy tensor exchange with Julia, Python, and
//!   other frameworks via [`DLManagedTensorVersioned`]. Supports CPU and
//!   GPU memory. Use [`tfe_tensor_f64_to_dlpack`] (export) and
//!   [`tfe_tensor_f64_from_dlpack`] (import).
//! - **Copy semantics** for convenience functions: `tfe_tensor_f64_from_data`
//!   copies the caller's data into a Rust-owned buffer. For zero-copy, use
//!   DLPack.
//!
//! # Memory ownership
//!
//! | Allocation | Freed by |
//! |-----------|----------|
//! | Tensor from `_from_data` / `_zeros` / `_clone` | `tfe_tensor_f64_release` |
//! | Tensor from `_from_dlpack` | `tfe_tensor_f64_release` (calls DLPack deleter) |
//! | Output tensor (via `**_out`) | `tfe_tensor_f64_release` |
//! | Gradient tensor (rrule output) | `tfe_tensor_f64_release` |
//! | `grads_out` array (einsum rrule) | Caller provides buffer |
//! | Input `data` pointer | Caller (data is copied) |
//! | `DLManagedTensorVersioned` from `_to_dlpack` | Consumer calls `deleter` |
//!
//! # Example (C pseudocode)
//!
//! ```c
//! tfe_status_t status;
//! size_t shape[] = {3, 4};
//! double data[12] = { /* ... */ };
//!
//! tfe_tensor_f64 *a = tfe_tensor_f64_from_data(data, 12, shape, 2, &status);
//! assert(status == TFE_SUCCESS);
//!
//! const tfe_tensor_f64 *ops[] = {a, a};
//! tfe_tensor_f64 *c = tfe_einsum_f64("ij,jk->ik", ops, 2, &status);
//!
//! tfe_tensor_f64_release(c);
//! tfe_tensor_f64_release(a);
//! ```

#![allow(clippy::missing_safety_doc)]
#![allow(non_camel_case_types)]

use std::cell::RefCell;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_einsum::{einsum, einsum_frule, einsum_rrule};
use tenferro_linalg::{svd, svd_frule, svd_rrule, SvdCotangent, SvdOptions};
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

// ============================================================================
// Status codes
// ============================================================================

/// Status code type returned by all C-API functions.
pub type tfe_status_t = i32;

/// Operation completed successfully.
pub const TFE_SUCCESS: tfe_status_t = 0;

/// Invalid argument (null pointer, bad subscript string, etc.).
pub const TFE_INVALID_ARGUMENT: tfe_status_t = -1;

/// Tensor shape mismatch for the requested operation.
pub const TFE_SHAPE_MISMATCH: tfe_status_t = -2;

/// Internal error (Rust panic or unexpected failure).
pub const TFE_INTERNAL_ERROR: tfe_status_t = -3;

/// Output buffer is too small for the requested data.
pub const TFE_BUFFER_TOO_SMALL: tfe_status_t = -4;

// ============================================================================
// Thread-local last-error storage
// ============================================================================

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Store an error message in thread-local storage.
fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = msg.to_string();
    });
}

/// Extract a human-readable message from a panic payload.
fn panic_message(payload: &dyn std::any::Any) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Retrieve the last error message (UTF-8, null-terminated).
///
/// - `buf == NULL`: query required length only (written to `*out_len`).
/// - `buf != NULL`: copy message into buffer.
///
/// `out_len` receives the required buffer size including the null terminator.
///
/// # Returns
///
/// - `TFE_SUCCESS` on success (or query-only mode).
/// - `TFE_INVALID_ARGUMENT` if `out_len` is null.
/// - `TFE_BUFFER_TOO_SMALL` if `buf_len` is too small (required size in `*out_len`).
///
/// # Safety
///
/// - `out_len` must be a valid, non-null pointer.
/// - If `buf` is non-null, it must point to a buffer of at least `buf_len` bytes.
///
/// # Examples (C)
///
/// ```c
/// // Query length
/// size_t len;
/// tfe_last_error_message(NULL, 0, &len);
/// if (len > 0) {
///     char *buf = malloc(len);
///     tfe_last_error_message((uint8_t *)buf, len, &len);
///     printf("Error: %s\n", buf);
///     free(buf);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_last_error_message(
    buf: *mut u8,
    buf_len: usize,
    out_len: *mut usize,
) -> tfe_status_t {
    if out_len.is_null() {
        return TFE_INVALID_ARGUMENT;
    }

    LAST_ERROR.with(|cell| {
        let msg = cell.borrow();
        let required = msg.len() + 1; // include null terminator
        *out_len = required;

        if buf.is_null() {
            return TFE_SUCCESS;
        }

        if buf_len < required {
            return TFE_BUFFER_TOO_SMALL;
        }

        std::ptr::copy_nonoverlapping(msg.as_ptr(), buf, msg.len());
        *buf.add(msg.len()) = 0; // null terminator
        TFE_SUCCESS
    })
}

// ============================================================================
// DLPack v1.0 C ABI types
// ============================================================================

/// DLPack version information.
#[repr(C)]
pub struct DLPackVersion {
    /// Major version (1 for DLPack v1.0).
    pub major: u32,
    /// Minor version.
    pub minor: u32,
}

/// DLPack device descriptor.
#[repr(C)]
pub struct DLDevice {
    /// Device type (see `KDLCPU`, `KDLCUDA`, `KDLCUDA_HOST`, `KDLCUDA_MANAGED`,
    /// `KDLROCM`, `KDLROCM_HOST`).
    pub device_type: i32,
    /// Device ID. 0 for CPU, pinned, and managed memory; GPU ordinal for
    /// device-local memory.
    pub device_id: i32,
}

/// DLPack data type descriptor.
#[repr(C)]
pub struct DLDataType {
    /// Type code (see `kDLFloat`, `kDLInt`, `kDLComplex`).
    pub code: u8,
    /// Number of bits per element (e.g., 64 for f64).
    pub bits: u8,
    /// Number of lanes (1 for scalar, >1 for SIMD vector types).
    pub lanes: u16,
}

/// DLPack tensor descriptor (unmanaged).
///
/// Describes the memory layout of a tensor without ownership information.
/// Used as a field within [`DLManagedTensorVersioned`].
#[repr(C)]
pub struct DLTensor {
    /// Pointer to the data. For GPU tensors, this is a device pointer.
    pub data: *mut c_void,
    /// Device where the data resides.
    pub device: DLDevice,
    /// Number of dimensions.
    pub ndim: i32,
    /// Data type.
    pub dtype: DLDataType,
    /// Shape array (length = ndim). Owned by the manager.
    pub shape: *mut i64,
    /// Strides array in **element units** (not bytes). NULL = row-major contiguous.
    pub strides: *mut i64,
    /// Byte offset from `data` pointer to the first element.
    pub byte_offset: u64,
}

/// DLPack managed tensor with version and ownership (DLPack v1.0+).
///
/// This is the primary type for DLPack tensor exchange. The `deleter`
/// callback must be called by the consumer when the data is no longer needed.
#[repr(C)]
pub struct DLManagedTensorVersioned {
    /// DLPack version.
    pub version: DLPackVersion,
    /// Opaque pointer for the producer's use (e.g., Box<Tensor>).
    pub manager_ctx: *mut c_void,
    /// Callback to free resources. Must be called exactly once by the consumer.
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    /// Bitmask flags (see `DLPACK_FLAG_*` constants).
    pub flags: u64,
    /// The tensor descriptor.
    pub dl_tensor: DLTensor,
}

// DLDeviceType constants
/// CPU device.
pub const KDLCPU: i32 = 1;
/// NVIDIA CUDA GPU device memory.
pub const KDLCUDA: i32 = 2;
/// Pinned CUDA CPU memory (`cudaMallocHost`).
pub const KDLCUDA_HOST: i32 = 3;
/// AMD ROCm GPU device memory.
pub const KDLROCM: i32 = 10;
/// Pinned ROCm CPU memory (`hipMallocHost`).
pub const KDLROCM_HOST: i32 = 11;
/// CUDA managed/unified memory (`cudaMallocManaged`).
pub const KDLCUDA_MANAGED: i32 = 13;

// DLDataTypeCode constants
/// Integer type code.
pub const KDLINT: u8 = 0;
/// Floating-point type code.
pub const KDLFLOAT: u8 = 2;
/// Complex type code.
pub const KDLCOMPLEX: u8 = 5;

// DLPack flags
/// Data is read-only (consumer must not write).
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1 << 0;
/// Data was copied (not zero-copy).
pub const DLPACK_FLAG_BITMASK_IS_COPIED: u64 = 1 << 1;

// ============================================================================
// Opaque tensor handle
// ============================================================================

/// Opaque handle wrapping a `Tensor<f64>`.
///
/// Host languages hold a pointer to this type and pass it to all
/// `tfe_*` functions. The internal layout is private; only the C-API
/// functions can access the inner tensor.
///
/// # Examples (C)
///
/// ```c
/// tfe_status_t status;
/// size_t shape[] = {2, 3};
/// double data[] = {1, 2, 3, 4, 5, 6};
/// tfe_tensor_f64 *t = tfe_tensor_f64_from_data(data, 6, shape, 2, &status);
/// // ... use t ...
/// tfe_tensor_f64_release(t);
/// ```
#[repr(C)]
pub struct TfeTensorF64 {
    _private: [u8; 0],
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Convert a `Tensor<f64>` into an opaque handle.
fn tensor_to_handle(tensor: Tensor<f64>) -> *mut TfeTensorF64 {
    Box::into_raw(Box::new(tensor)) as *mut TfeTensorF64
}

/// Borrow the tensor behind an opaque handle.
///
/// # Safety
///
/// `handle` must be a valid, non-null pointer returned by `tensor_to_handle`.
unsafe fn handle_to_ref<'a>(handle: *const TfeTensorF64) -> &'a Tensor<f64> {
    &*(handle as *const Tensor<f64>)
}

/// Take ownership of the tensor behind an opaque handle (frees on drop).
///
/// # Safety
///
/// `handle` must be a valid, non-null pointer returned by `tensor_to_handle`.
/// Must not be used after this call.
unsafe fn handle_take(handle: *mut TfeTensorF64) -> Box<Tensor<f64>> {
    Box::from_raw(handle as *mut Tensor<f64>)
}

/// Build `SvdOptions` from C-API parameters.
fn build_svd_options(max_rank: usize, cutoff: f64) -> Option<SvdOptions> {
    let mr = if max_rank == 0 { None } else { Some(max_rank) };
    let co = if cutoff < 0.0 { None } else { Some(cutoff) };
    if mr.is_none() && co.is_none() {
        None
    } else {
        Some(SvdOptions {
            max_rank: mr,
            cutoff: co,
        })
    }
}

// ============================================================================
// Error mapping helpers
// ============================================================================

/// Map `tenferro_device::Error` to the appropriate status code.
///
/// Also stores the error message in thread-local storage.
fn map_device_error(err: &tenferro_device::Error) -> tfe_status_t {
    set_last_error(&err.to_string());
    use tenferro_device::Error;
    match err {
        Error::InvalidArgument(_)
        | Error::StrideError(_)
        | Error::CrossMemorySpaceOperation { .. } => TFE_INVALID_ARGUMENT,
        Error::ShapeMismatch { .. } | Error::RankMismatch { .. } => TFE_SHAPE_MISMATCH,
        Error::DeviceError(_) | Error::NoCompatibleComputeDevice { .. } => TFE_INTERNAL_ERROR,
    }
}

/// Map `chainrules_core::AutodiffError` to the appropriate status code.
///
/// Also stores the error message in thread-local storage.
fn map_ad_error(err: &chainrules_core::AutodiffError) -> tfe_status_t {
    set_last_error(&err.to_string());
    use chainrules_core::AutodiffError;
    match err {
        AutodiffError::InvalidArgument(_)
        | AutodiffError::ModeNotSupported { .. }
        | AutodiffError::NonScalarLoss { .. }
        | AutodiffError::HvpNotSupported => TFE_INVALID_ARGUMENT,
        AutodiffError::TangentShapeMismatch { .. } => TFE_SHAPE_MISMATCH,
        AutodiffError::MissingNode => TFE_INTERNAL_ERROR,
    }
}

/// Finalize a `catch_unwind` result for functions returning a pointer via status.
///
/// # Safety
///
/// `status` must be a valid, non-null pointer.
unsafe fn finalize_ptr(
    result: std::thread::Result<Result<*mut TfeTensorF64, tfe_status_t>>,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    match result {
        Ok(Ok(ptr)) => {
            *status = TFE_SUCCESS;
            ptr
        }
        Ok(Err(code)) => {
            *status = code;
            std::ptr::null_mut()
        }
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            *status = TFE_INTERNAL_ERROR;
            std::ptr::null_mut()
        }
    }
}

/// Finalize a `catch_unwind` result for functions returning void via status.
///
/// # Safety
///
/// `status` must be a valid, non-null pointer.
unsafe fn finalize_void(
    result: std::thread::Result<Result<(), tfe_status_t>>,
    status: *mut tfe_status_t,
) {
    match result {
        Ok(Ok(())) => {
            *status = TFE_SUCCESS;
        }
        Ok(Err(code)) => {
            *status = code;
        }
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            *status = TFE_INTERNAL_ERROR;
        }
    }
}

/// Matricize a tensor according to left/right dimension indices.
///
/// Returns `(matrix, left_dims, right_dims)` where `matrix` is a 2D
/// column-major contiguous tensor of shape `[m, n]`.
fn matricize(
    tensor: &Tensor<f64>,
    left: &[usize],
    right: &[usize],
) -> Result<(Tensor<f64>, Vec<usize>, Vec<usize>), tfe_status_t> {
    let dims = tensor.dims();
    let mut seen = vec![false; dims.len()];

    // Validate indices
    for &l in left {
        if l >= dims.len() {
            return Err(TFE_INVALID_ARGUMENT);
        }
        if seen[l] {
            return Err(TFE_INVALID_ARGUMENT);
        }
        seen[l] = true;
    }
    for &r in right {
        if r >= dims.len() {
            return Err(TFE_INVALID_ARGUMENT);
        }
        if seen[r] {
            return Err(TFE_INVALID_ARGUMENT);
        }
        seen[r] = true;
    }
    if left.len() + right.len() != dims.len() {
        return Err(TFE_INVALID_ARGUMENT);
    }
    if seen.iter().any(|&v| !v) {
        return Err(TFE_INVALID_ARGUMENT);
    }

    let left_dims: Vec<usize> = left.iter().map(|&i| dims[i]).collect();
    let right_dims: Vec<usize> = right.iter().map(|&i| dims[i]).collect();
    let m: usize = left_dims.iter().product();
    let n: usize = right_dims.iter().product();

    // Build permutation: left dims first, then right dims
    let mut perm: Vec<usize> = Vec::with_capacity(dims.len());
    perm.extend_from_slice(left);
    perm.extend_from_slice(right);

    let permuted = tensor
        .permute(&perm)
        .map_err(|_| TFE_INTERNAL_ERROR)?
        .contiguous(MemoryOrder::ColumnMajor)
        .reshape(&[m, n])
        .map_err(|_| TFE_SHAPE_MISMATCH)?
        .contiguous(MemoryOrder::ColumnMajor);

    Ok((permuted, left_dims, right_dims))
}

/// Compute inverse permutation: `inv_perm[perm[i]] = i`.
fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// Convert public U-cotangent shape `[left_dims..., k]` to matrix shape `[m, k]`.
fn u_cotangent_to_matrix(
    cot_u: &Tensor<f64>,
    left_dims: &[usize],
) -> Result<(Tensor<f64>, usize), tfe_status_t> {
    let u_dims = cot_u.dims();
    if u_dims.len() != left_dims.len() + 1 {
        return Err(TFE_SHAPE_MISMATCH);
    }
    if &u_dims[..left_dims.len()] != left_dims {
        return Err(TFE_SHAPE_MISMATCH);
    }
    let k = u_dims[left_dims.len()];
    let m: usize = left_dims.iter().product();
    let mat = cot_u
        .contiguous(MemoryOrder::ColumnMajor)
        .reshape(&[m, k])
        .map_err(|_| TFE_SHAPE_MISMATCH)?
        .contiguous(MemoryOrder::ColumnMajor);
    Ok((mat, k))
}

/// Convert public Vt-cotangent shape `[k, right_dims...]` to matrix shape `[k, n]`.
fn vt_cotangent_to_matrix(
    cot_vt: &Tensor<f64>,
    right_dims: &[usize],
) -> Result<(Tensor<f64>, usize), tfe_status_t> {
    let vt_dims = cot_vt.dims();
    if vt_dims.len() != right_dims.len() + 1 {
        return Err(TFE_SHAPE_MISMATCH);
    }
    if &vt_dims[1..] != right_dims {
        return Err(TFE_SHAPE_MISMATCH);
    }
    let k = vt_dims[0];
    let n: usize = right_dims.iter().product();
    let mat = cot_vt
        .contiguous(MemoryOrder::ColumnMajor)
        .reshape(&[k, n])
        .map_err(|_| TFE_SHAPE_MISMATCH)?
        .contiguous(MemoryOrder::ColumnMajor);
    Ok((mat, k))
}

/// Validate S-cotangent shape `[k]`.
fn validate_s_cotangent(cot_s: &Tensor<f64>) -> Result<usize, tfe_status_t> {
    let dims = cot_s.dims();
    if dims.len() != 1 {
        return Err(TFE_SHAPE_MISMATCH);
    }
    Ok(dims[0])
}

/// Reshape matrix U (`[m, k]`) back to public shape `[left_dims..., k]`.
fn u_matrix_to_public(u: Tensor<f64>, left_dims: &[usize]) -> Result<Tensor<f64>, tfe_status_t> {
    if u.dims().len() != 2 {
        return Err(TFE_SHAPE_MISMATCH);
    }
    let k = u.dims()[1];
    let mut out_dims = left_dims.to_vec();
    out_dims.push(k);
    u.reshape(&out_dims)
        .map_err(|_| TFE_SHAPE_MISMATCH)
        .map(|t| t.contiguous(MemoryOrder::ColumnMajor))
}

/// Reshape matrix Vt (`[k, n]`) back to public shape `[k, right_dims...]`.
fn vt_matrix_to_public(vt: Tensor<f64>, right_dims: &[usize]) -> Result<Tensor<f64>, tfe_status_t> {
    if vt.dims().len() != 2 {
        return Err(TFE_SHAPE_MISMATCH);
    }
    let k = vt.dims()[0];
    let mut out_dims = vec![k];
    out_dims.extend_from_slice(right_dims);
    vt.reshape(&out_dims)
        .map_err(|_| TFE_SHAPE_MISMATCH)
        .map(|t| t.contiguous(MemoryOrder::ColumnMajor))
}

/// Convert input gradient from matrixized layout `[m, n]` back to original tensor layout.
fn grad_matrix_to_public(
    grad_matrix: Tensor<f64>,
    original_dims: &[usize],
    left: &[usize],
    right: &[usize],
    left_dims: &[usize],
    right_dims: &[usize],
) -> Result<Tensor<f64>, tfe_status_t> {
    if grad_matrix.dims().len() != 2 {
        return Err(TFE_SHAPE_MISMATCH);
    }

    let mut permuted_dims = left_dims.to_vec();
    permuted_dims.extend_from_slice(right_dims);
    let reshaped = grad_matrix
        .contiguous(MemoryOrder::ColumnMajor)
        .reshape(&permuted_dims)
        .map_err(|_| TFE_SHAPE_MISMATCH)?;

    let mut perm = Vec::with_capacity(original_dims.len());
    perm.extend_from_slice(left);
    perm.extend_from_slice(right);
    let inv_perm = inverse_permutation(&perm);

    reshaped
        .permute(&inv_perm)
        .map_err(|_| TFE_INTERNAL_ERROR)
        .map(|t| t.contiguous(MemoryOrder::ColumnMajor))
}

// ============================================================================
// Tensor lifecycle
// ============================================================================

/// Create a tensor from caller-provided data (copy semantics).
///
/// The data is **copied** into Rust-owned storage. The caller retains
/// ownership of the `data` pointer and may free it after this call.
/// The internal memory layout (strides) is implementation-defined.
///
/// For zero-copy tensor exchange with specific memory layouts, use
/// [`tfe_tensor_f64_from_dlpack`] instead.
///
/// # Safety
///
/// - `data` must point to at least `len` valid `f64` values.
/// - `shape` must point to at least `ndim` valid `usize` values.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
/// size_t shape[] = {2, 3};
/// tfe_status_t status;
/// tfe_tensor_f64 *t = tfe_tensor_f64_from_data(data, 6, shape, 2, &status);
/// assert(status == TFE_SUCCESS);
/// tfe_tensor_f64_release(t);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_from_data(
    data: *const f64,
    len: usize,
    shape: *const usize,
    ndim: usize,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if data.is_null() && len > 0 {
            return Err(TFE_INVALID_ARGUMENT);
        }
        if shape.is_null() && ndim > 0 {
            return Err(TFE_INVALID_ARGUMENT);
        }

        let dims = if ndim > 0 {
            std::slice::from_raw_parts(shape, ndim).to_vec()
        } else {
            vec![]
        };

        let data_slice = if len > 0 {
            std::slice::from_raw_parts(data, len)
        } else {
            &[]
        };

        Tensor::from_slice(data_slice, &dims, MemoryOrder::ColumnMajor)
            .map(|t| tensor_to_handle(t))
            .map_err(|e| map_device_error(&e))
    }));

    finalize_ptr(result, status)
}

/// Create a tensor filled with zeros.
///
/// The internal memory layout is implementation-defined.
///
/// # Safety
///
/// - `shape` must point to at least `ndim` valid `usize` values.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// size_t shape[] = {3, 4};
/// tfe_status_t status;
/// tfe_tensor_f64 *t = tfe_tensor_f64_zeros(shape, 2, &status);
/// tfe_tensor_f64_release(t);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_zeros(
    shape: *const usize,
    ndim: usize,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if shape.is_null() && ndim > 0 {
            return Err(TFE_INVALID_ARGUMENT);
        }

        let dims = if ndim > 0 {
            std::slice::from_raw_parts(shape, ndim).to_vec()
        } else {
            vec![]
        };

        let t = Tensor::<f64>::zeros(
            &dims,
            LogicalMemorySpace::MainMemory,
            MemoryOrder::ColumnMajor,
        );
        Ok(tensor_to_handle(t))
    }));

    finalize_ptr(result, status)
}

/// Deep-copy a tensor.
///
/// `Tensor::clone()` is a shallow copy (Arc refcount increment).
/// This C API function performs a deep copy using prims operations
/// (e.g., `Permute(identity)` or `MakeContiguous`) to produce a
/// tensor with its own independent data buffer.
///
/// # Safety
///
/// - `tensor` must be a valid pointer returned by a `tfe_tensor_f64_*`
///   creation function that has not yet been released.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// tfe_tensor_f64 *copy = tfe_tensor_f64_clone(original, &status);
/// tfe_tensor_f64_release(copy);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_clone(
    tensor: *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if tensor.is_null() {
            return Err(TFE_INVALID_ARGUMENT);
        }
        let src = handle_to_ref(tensor);
        let src_data = src.buffer().as_slice().unwrap();
        let n = src.len();
        let off = src.offset() as usize;
        let copy = Tensor::from_slice(
            &src_data[off..off + n],
            src.dims(),
            MemoryOrder::ColumnMajor,
        )
        .map_err(|e| map_device_error(&e))?;
        Ok(tensor_to_handle(copy))
    }));

    finalize_ptr(result, status)
}

/// Release (free) a tensor.
///
/// After this call, `tensor` is invalid and must not be used.
/// Passing a null pointer is a no-op.
///
/// For tensors imported via DLPack, this calls the DLPack deleter
/// to notify the external owner that the data is no longer needed.
///
/// # Safety
///
/// `tensor` must be null or a valid pointer returned by a
/// `tfe_tensor_f64_*` creation function that has not yet been released.
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_release(tensor: *mut TfeTensorF64) {
    if tensor.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(handle_take(tensor));
    }));
}

/// Return the number of dimensions (rank) of the tensor.
///
/// Returns 0 if `tensor` is null (and sets `status` to `TFE_INVALID_ARGUMENT`).
///
/// # Safety
///
/// - `tensor` must be a valid tensor pointer or null.
/// - `status` must be a valid, non-null pointer.
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_ndim(
    tensor: *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> usize {
    if tensor.is_null() {
        *status = TFE_INVALID_ARGUMENT;
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| handle_to_ref(tensor).ndim()));
    match result {
        Ok(n) => {
            *status = TFE_SUCCESS;
            n
        }
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            *status = TFE_INTERNAL_ERROR;
            0
        }
    }
}

/// Write the shape of the tensor into the caller-provided buffer.
///
/// The caller must allocate `out_shape` with at least
/// `tfe_tensor_f64_ndim(tensor)` elements.
///
/// # Safety
///
/// - `tensor` must be a valid, non-null tensor pointer.
/// - `out_shape` must point to a buffer with at least `ndim` `usize` slots.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// tfe_status_t status;
/// size_t ndim = tfe_tensor_f64_ndim(t, &status);
/// size_t *shape = malloc(ndim * sizeof(size_t));
/// tfe_tensor_f64_shape(t, shape, &status);
/// // shape now contains the dimensions
/// free(shape);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_shape(
    tensor: *const TfeTensorF64,
    out_shape: *mut usize,
    status: *mut tfe_status_t,
) {
    if tensor.is_null() || out_shape.is_null() {
        *status = TFE_INVALID_ARGUMENT;
        return;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let t = handle_to_ref(tensor);
        let dims = t.dims();
        std::ptr::copy_nonoverlapping(dims.as_ptr(), out_shape, dims.len());
    }));
    match result {
        Ok(()) => {
            *status = TFE_SUCCESS;
        }
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            *status = TFE_INTERNAL_ERROR;
        }
    }
}

/// Return the total number of elements in the tensor.
///
/// Returns 0 if `tensor` is null (and sets `status` to `TFE_INVALID_ARGUMENT`).
///
/// # Safety
///
/// - `tensor` must be a valid tensor pointer or null.
/// - `status` must be a valid, non-null pointer.
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_len(
    tensor: *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> usize {
    if tensor.is_null() {
        *status = TFE_INVALID_ARGUMENT;
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| handle_to_ref(tensor).len()));
    match result {
        Ok(n) => {
            *status = TFE_SUCCESS;
            n
        }
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            *status = TFE_INTERNAL_ERROR;
            0
        }
    }
}

/// Return a pointer to the tensor's raw data buffer.
///
/// The pointer is valid until `tfe_tensor_f64_release` is called on
/// the tensor. Returns null if `tensor` is null.
///
/// # Safety
///
/// - `tensor` must be a valid tensor pointer or null.
/// - `status` must be a valid, non-null pointer.
/// - The returned pointer must not be used after `tfe_tensor_f64_release(tensor)`.
///
/// # Examples (C)
///
/// ```c
/// tfe_status_t status;
/// const double *ptr = tfe_tensor_f64_data(t, &status);
/// size_t n = tfe_tensor_f64_len(t, &status);
/// for (size_t i = 0; i < n; i++) {
///     printf("%f ", ptr[i]);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_data(
    tensor: *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> *const f64 {
    if tensor.is_null() {
        *status = TFE_INVALID_ARGUMENT;
        return std::ptr::null();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let t = handle_to_ref(tensor);
        let slice = t.buffer().as_slice().unwrap();
        slice.as_ptr().add(t.offset() as usize)
    }));
    match result {
        Ok(ptr) => {
            *status = TFE_SUCCESS;
            ptr
        }
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            *status = TFE_INTERNAL_ERROR;
            std::ptr::null()
        }
    }
}

// ============================================================================
// DLPack interop
// ============================================================================

/// Export a tensor as a DLPack managed tensor (zero-copy).
///
/// The tensor handle is **consumed** by this call and must not be
/// used afterwards (do not call `tfe_tensor_f64_release` on it).
///
/// The returned `DLManagedTensorVersioned` must be consumed by the
/// caller (e.g., passed to Julia `DLPack.from_dlpack()` or Python
/// `numpy.from_dlpack()`). The consumer must call the `deleter`
/// callback when done with the data.
///
/// If the tensor is NULL, returns NULL and sets status to `TFE_INVALID_ARGUMENT`.
///
/// # Safety
///
/// - `tensor` must be a valid tensor pointer or NULL.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// tfe_status_t status;
/// tfe_tensor_f64 *t = tfe_tensor_f64_zeros(shape, 2, &status);
///
/// // Export to DLPack (tensor handle is consumed)
/// DLManagedTensorVersioned *dl = tfe_tensor_f64_to_dlpack(t, &status);
/// // t is now invalid — do NOT call tfe_tensor_f64_release(t)
///
/// // Pass dl to Julia/Python, which calls dl->deleter(dl) when done
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_to_dlpack(
    _tensor: *mut TfeTensorF64,
    _status: *mut tfe_status_t,
) -> *mut DLManagedTensorVersioned {
    todo!()
}

/// Import a DLPack managed tensor as a tenferro tensor (zero-copy).
///
/// Takes ownership of the `DLManagedTensorVersioned`. The deleter
/// callback will be called when the returned tensor is released
/// via `tfe_tensor_f64_release`.
///
/// The tensor data is NOT copied. The returned tensor references
/// the same memory as the DLPack tensor.
///
/// Currently only `kDLCPU` device and float64 dtype are accepted
/// (POC phase). Returns NULL with `TFE_INVALID_ARGUMENT` for other
/// device types or dtypes.
///
/// # Safety
///
/// - `managed` must be a valid pointer to a `DLManagedTensorVersioned`.
/// - The DLPack tensor's data must remain valid until the deleter is called.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// // Obtain DLManagedTensorVersioned* from Julia/Python
/// DLManagedTensorVersioned *dl = /* ... */;
///
/// tfe_status_t status;
/// tfe_tensor_f64 *t = tfe_tensor_f64_from_dlpack(dl, &status);
/// // dl is now owned by t — do NOT call dl->deleter(dl)
///
/// // Use t in einsum, SVD, etc.
/// tfe_tensor_f64_release(t); // calls DLPack deleter internally
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_from_dlpack(
    _managed: *mut DLManagedTensorVersioned,
    _status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    todo!()
}

// ============================================================================
// Einsum
// ============================================================================

/// Execute einsum using string notation.
///
/// Returns a new tensor. The caller must release it with
/// `tfe_tensor_f64_release`.
///
/// # Safety
///
/// - `subscripts` must be a valid null-terminated C string.
/// - `operands` must point to an array of `num_operands` valid tensor pointers.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// const tfe_tensor_f64 *ops[] = {a, b};
/// tfe_status_t status;
/// tfe_tensor_f64 *c = tfe_einsum_f64("ij,jk->ik", ops, 2, &status);
/// assert(status == TFE_SUCCESS);
/// tfe_tensor_f64_release(c);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_einsum_f64(
    subscripts: *const c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if subscripts.is_null() || operands.is_null() {
            return Err(TFE_INVALID_ARGUMENT);
        }

        let subs = CStr::from_ptr(subscripts)
            .to_str()
            .map_err(|_| TFE_INVALID_ARGUMENT)?;

        let op_ptrs = std::slice::from_raw_parts(operands, num_operands);
        let ops: Vec<&Tensor<f64>> = op_ptrs
            .iter()
            .map(|&p| {
                if p.is_null() {
                    Err(TFE_INVALID_ARGUMENT)
                } else {
                    Ok(handle_to_ref(p))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut ctx = CpuContext::new(1);
        einsum::<f64, Standard<f64>, CpuBackend>(&mut ctx, subs, &ops, None)
            .map(|t| tensor_to_handle(t))
            .map_err(|e| map_device_error(&e))
    }));

    finalize_ptr(result, status)
}

/// Reverse-mode rule (VJP) for einsum.
///
/// Computes one gradient tensor per input operand given the output
/// cotangent. The caller must provide `grads_out` as a pre-allocated
/// array of `num_operands` pointers. Each returned tensor must be
/// released by the caller.
///
/// # Safety
///
/// - `subscripts` must be a valid null-terminated C string.
/// - `operands` must point to an array of `num_operands` valid tensor pointers.
/// - `cotangent` must be a valid, non-null tensor pointer.
/// - `grads_out` must point to a caller-allocated array of `num_operands`
///   mutable `*mut TfeTensorF64` pointers.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// // After computing c = einsum("ij,jk->ik", [a, b]):
/// tfe_tensor_f64 *grads[2];
/// tfe_status_t status;
/// const tfe_tensor_f64 *ops[] = {a, b};
/// tfe_einsum_rrule_f64("ij,jk->ik", ops, 2, grad_c, grads, &status);
/// // grads[0] = gradient w.r.t. a
/// // grads[1] = gradient w.r.t. b
/// tfe_tensor_f64_release(grads[0]);
/// tfe_tensor_f64_release(grads[1]);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_einsum_rrule_f64(
    subscripts: *const c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
    cotangent: *const TfeTensorF64,
    grads_out: *mut *mut TfeTensorF64,
    status: *mut tfe_status_t,
) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if subscripts.is_null() || operands.is_null() || cotangent.is_null() || grads_out.is_null()
        {
            return Err(TFE_INVALID_ARGUMENT);
        }

        let subs = CStr::from_ptr(subscripts)
            .to_str()
            .map_err(|_| TFE_INVALID_ARGUMENT)?;

        let op_ptrs = std::slice::from_raw_parts(operands, num_operands);
        let ops: Vec<&Tensor<f64>> = op_ptrs
            .iter()
            .map(|&p| {
                if p.is_null() {
                    Err(TFE_INVALID_ARGUMENT)
                } else {
                    Ok(handle_to_ref(p))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let cot = handle_to_ref(cotangent);

        let mut ctx = CpuContext::new(1);
        let grads = einsum_rrule::<f64, Standard<f64>, CpuBackend>(&mut ctx, subs, &ops, cot)
            .map_err(|e| map_device_error(&e))?;

        let out_slice = std::slice::from_raw_parts_mut(grads_out, num_operands);
        for (i, g) in grads.into_iter().enumerate() {
            out_slice[i] = tensor_to_handle(g);
        }

        Ok(())
    }));

    finalize_void(result, status)
}

/// Forward-mode rule (JVP) for einsum.
///
/// Returns the output tangent. Elements of `tangents` may be null
/// (interpreted as zero tangent for that operand).
///
/// # Safety
///
/// - `subscripts` must be a valid null-terminated C string.
/// - `primals` must point to an array of `num_operands` valid tensor pointers.
/// - `tangents` must point to an array of `num_operands` tensor pointers
///   (elements may be null).
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// const tfe_tensor_f64 *primals[] = {a, b};
/// const tfe_tensor_f64 *tangents[] = {da, NULL};  // no tangent for b
/// tfe_status_t status;
/// tfe_tensor_f64 *dc = tfe_einsum_frule_f64(
///     "ij,jk->ik", primals, 2, tangents, &status);
/// tfe_tensor_f64_release(dc);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_einsum_frule_f64(
    subscripts: *const c_char,
    primals: *const *const TfeTensorF64,
    num_operands: usize,
    tangents: *const *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if subscripts.is_null() || primals.is_null() || tangents.is_null() {
            return Err(TFE_INVALID_ARGUMENT);
        }

        let subs = CStr::from_ptr(subscripts)
            .to_str()
            .map_err(|_| TFE_INVALID_ARGUMENT)?;

        let primal_ptrs = std::slice::from_raw_parts(primals, num_operands);
        let primal_refs: Vec<&Tensor<f64>> = primal_ptrs
            .iter()
            .map(|&p| {
                if p.is_null() {
                    Err(TFE_INVALID_ARGUMENT)
                } else {
                    Ok(handle_to_ref(p))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let tangent_ptrs = std::slice::from_raw_parts(tangents, num_operands);
        let tangent_refs: Vec<Option<&Tensor<f64>>> = tangent_ptrs
            .iter()
            .map(|&p| {
                if p.is_null() {
                    None
                } else {
                    Some(handle_to_ref(p))
                }
            })
            .collect();

        let mut ctx = CpuContext::new(1);
        einsum_frule::<f64, Standard<f64>, CpuBackend>(&mut ctx, subs, &primal_refs, &tangent_refs)
            .map(|t| tensor_to_handle(t))
            .map_err(|e| map_device_error(&e))
    }));

    finalize_ptr(result, status)
}

// ============================================================================
// SVD
// ============================================================================

/// Compute the SVD of a tensor.
///
/// Decomposes the tensor into `U * diag(S) * Vt` after matricizing
/// according to `left`/`right` dimension indices. Returns the three
/// factors via output pointers. The caller must release each.
///
/// Set `max_rank` to 0 for no rank limit.
/// Set `cutoff` to a negative value for no cutoff.
///
/// # Safety
///
/// - `tensor` must be a valid, non-null tensor pointer.
/// - `left` must point to `left_len` valid `usize` values.
/// - `right` must point to `right_len` valid `usize` values.
/// - `u_out`, `s_out`, `vt_out` must be valid, non-null pointers to
///   `*mut TfeTensorF64`.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// size_t left[] = {0};
/// size_t right[] = {1};
/// tfe_tensor_f64 *u, *s, *vt;
/// tfe_status_t status;
/// tfe_svd_f64(a, left, 1, right, 1, 0, -1.0, &u, &s, &vt, &status);
/// assert(status == TFE_SUCCESS);
/// tfe_tensor_f64_release(u);
/// tfe_tensor_f64_release(s);
/// tfe_tensor_f64_release(vt);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_svd_f64(
    tensor: *const TfeTensorF64,
    left: *const usize,
    left_len: usize,
    right: *const usize,
    right_len: usize,
    max_rank: usize,
    cutoff: f64,
    u_out: *mut *mut TfeTensorF64,
    s_out: *mut *mut TfeTensorF64,
    vt_out: *mut *mut TfeTensorF64,
    status: *mut tfe_status_t,
) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if tensor.is_null() || u_out.is_null() || s_out.is_null() || vt_out.is_null() {
            return Err(TFE_INVALID_ARGUMENT);
        }
        if left.is_null() && left_len > 0 {
            return Err(TFE_INVALID_ARGUMENT);
        }
        if right.is_null() && right_len > 0 {
            return Err(TFE_INVALID_ARGUMENT);
        }

        let t = handle_to_ref(tensor);
        let left_indices = if left_len > 0 {
            std::slice::from_raw_parts(left, left_len)
        } else {
            &[]
        };
        let right_indices = if right_len > 0 {
            std::slice::from_raw_parts(right, right_len)
        } else {
            &[]
        };

        let (matrix, left_dims, right_dims) = matricize(t, left_indices, right_indices)?;

        let opts = build_svd_options(max_rank, cutoff);
        let result = svd(&matrix, opts.as_ref()).map_err(|e| map_device_error(&e))?;

        // Reshape U from [m, k] to [left_dims..., k]
        let k = result.s.len();
        let mut u_dims: Vec<usize> = left_dims;
        u_dims.push(k);
        let u_reshaped = result
            .u
            .reshape(&u_dims)
            .map_err(|e| map_device_error(&e))?
            .contiguous(MemoryOrder::ColumnMajor);

        // Reshape Vt from [k, n] to [k, right_dims...]
        let mut vt_dims: Vec<usize> = vec![k];
        vt_dims.extend_from_slice(&right_dims);
        let vt_reshaped = result
            .vt
            .reshape(&vt_dims)
            .map_err(|e| map_device_error(&e))?
            .contiguous(MemoryOrder::ColumnMajor);

        *u_out = tensor_to_handle(u_reshaped);
        *s_out = tensor_to_handle(result.s);
        *vt_out = tensor_to_handle(vt_reshaped);
        Ok(())
    }));

    finalize_void(result, status)
}

/// Reverse-mode rule (VJP) for SVD.
///
/// Computes the gradient of the input tensor given cotangents for
/// U, S, and Vt. Any cotangent may be null (zero cotangent).
///
/// # Safety
///
/// - `tensor` must be a valid, non-null tensor pointer.
/// - `left` must point to `left_len` valid `usize` values.
/// - `right` must point to `right_len` valid `usize` values.
/// - `cotangent_u`, `cotangent_s`, `cotangent_vt` may each be null.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// size_t left[] = {0};
/// size_t right[] = {1};
/// tfe_status_t status;
/// // Only need gradient through singular values
/// tfe_tensor_f64 *grad = tfe_svd_rrule_f64(
///     a, left, 1, right, 1, 0, -1.0,
///     NULL,    // no cotangent for U
///     cot_s,   // cotangent for S
///     NULL,    // no cotangent for Vt
///     &status);
/// tfe_tensor_f64_release(grad);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_svd_rrule_f64(
    tensor: *const TfeTensorF64,
    left: *const usize,
    left_len: usize,
    right: *const usize,
    right_len: usize,
    max_rank: usize,
    cutoff: f64,
    cotangent_u: *const TfeTensorF64,
    cotangent_s: *const TfeTensorF64,
    cotangent_vt: *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if tensor.is_null() {
            return Err(TFE_INVALID_ARGUMENT);
        }
        if left.is_null() && left_len > 0 {
            return Err(TFE_INVALID_ARGUMENT);
        }
        if right.is_null() && right_len > 0 {
            return Err(TFE_INVALID_ARGUMENT);
        }

        let t = handle_to_ref(tensor);
        let left_indices = if left_len > 0 {
            std::slice::from_raw_parts(left, left_len)
        } else {
            &[]
        };
        let right_indices = if right_len > 0 {
            std::slice::from_raw_parts(right, right_len)
        } else {
            &[]
        };

        let original_dims = t.dims().to_vec();
        let (matrix, left_dims, right_dims) = matricize(t, left_indices, right_indices)?;

        let mut inferred_k: Option<usize> = None;
        let cot_u = if cotangent_u.is_null() {
            None
        } else {
            let (u_mat, k) = u_cotangent_to_matrix(handle_to_ref(cotangent_u), &left_dims)?;
            inferred_k = Some(k);
            Some(u_mat)
        };
        let cot_s = if cotangent_s.is_null() {
            None
        } else {
            let cot_s_ref = handle_to_ref(cotangent_s);
            let k = validate_s_cotangent(cot_s_ref)?;
            if let Some(prev) = inferred_k {
                if prev != k {
                    return Err(TFE_SHAPE_MISMATCH);
                }
            } else {
                inferred_k = Some(k);
            }
            Some(cot_s_ref.clone())
        };
        let cot_vt = if cotangent_vt.is_null() {
            None
        } else {
            let (vt_mat, k) = vt_cotangent_to_matrix(handle_to_ref(cotangent_vt), &right_dims)?;
            if let Some(prev) = inferred_k {
                if prev != k {
                    return Err(TFE_SHAPE_MISMATCH);
                }
            } else {
                inferred_k = Some(k);
            }
            Some(vt_mat)
        };
        let _ = inferred_k;

        let cotangent = SvdCotangent {
            u: cot_u,
            s: cot_s,
            vt: cot_vt,
        };

        let opts = build_svd_options(max_rank, cutoff);
        let grad_matrix =
            svd_rrule(&matrix, &cotangent, opts.as_ref()).map_err(|e| map_ad_error(&e))?;
        let grad = grad_matrix_to_public(
            grad_matrix,
            &original_dims,
            left_indices,
            right_indices,
            &left_dims,
            &right_dims,
        )?;

        Ok(tensor_to_handle(grad))
    }));

    finalize_ptr(result, status)
}

/// Forward-mode rule (JVP) for SVD.
///
/// Computes tangents for U, S, Vt given an input tangent.
/// The `tangent` parameter may be null (zero tangent).
///
/// # Safety
///
/// - `tensor` must be a valid, non-null tensor pointer.
/// - `left` must point to `left_len` valid `usize` values.
/// - `right` must point to `right_len` valid `usize` values.
/// - `tangent` may be null (zero tangent).
/// - `u_out`, `s_out`, `vt_out` must be valid, non-null pointers to
///   `*mut TfeTensorF64`.
/// - `status` must be a valid, non-null pointer.
///
/// # Examples (C)
///
/// ```c
/// size_t left[] = {0};
/// size_t right[] = {1};
/// tfe_tensor_f64 *du, *ds, *dvt;
/// tfe_status_t status;
/// tfe_svd_frule_f64(
///     a, left, 1, right, 1, 0, -1.0,
///     da, &du, &ds, &dvt, &status);
/// tfe_tensor_f64_release(du);
/// tfe_tensor_f64_release(ds);
/// tfe_tensor_f64_release(dvt);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_svd_frule_f64(
    tensor: *const TfeTensorF64,
    left: *const usize,
    left_len: usize,
    right: *const usize,
    right_len: usize,
    max_rank: usize,
    cutoff: f64,
    tangent: *const TfeTensorF64,
    u_out: *mut *mut TfeTensorF64,
    s_out: *mut *mut TfeTensorF64,
    vt_out: *mut *mut TfeTensorF64,
    status: *mut tfe_status_t,
) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if tensor.is_null() || u_out.is_null() || s_out.is_null() || vt_out.is_null() {
            return Err(TFE_INVALID_ARGUMENT);
        }
        if left.is_null() && left_len > 0 {
            return Err(TFE_INVALID_ARGUMENT);
        }
        if right.is_null() && right_len > 0 {
            return Err(TFE_INVALID_ARGUMENT);
        }

        let t = handle_to_ref(tensor);
        let left_indices = if left_len > 0 {
            std::slice::from_raw_parts(left, left_len)
        } else {
            &[]
        };
        let right_indices = if right_len > 0 {
            std::slice::from_raw_parts(right, right_len)
        } else {
            &[]
        };

        let (matrix, left_dims, right_dims) = matricize(t, left_indices, right_indices)?;

        let tang = if tangent.is_null() {
            Tensor::<f64>::zeros(
                matrix.dims(),
                LogicalMemorySpace::MainMemory,
                MemoryOrder::ColumnMajor,
            )
        } else {
            let tang_tensor = handle_to_ref(tangent);
            let (tang_matrix, _, _) = matricize(tang_tensor, left_indices, right_indices)?;
            tang_matrix
        };

        let opts = build_svd_options(max_rank, cutoff);
        let (_primal, tangent_result) =
            svd_frule(&matrix, &tang, opts.as_ref()).map_err(|e| map_ad_error(&e))?;

        let u_public = u_matrix_to_public(tangent_result.u, &left_dims)?;
        let vt_public = vt_matrix_to_public(tangent_result.vt, &right_dims)?;

        *u_out = tensor_to_handle(u_public);
        *s_out = tensor_to_handle(tangent_result.s);
        *vt_out = tensor_to_handle(vt_public);
        Ok(())
    }));

    finalize_void(result, status)
}
