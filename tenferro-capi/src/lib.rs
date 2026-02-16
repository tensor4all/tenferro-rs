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

use std::os::raw::{c_char, c_void};

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
    _data: *const f64,
    _len: usize,
    _shape: *const usize,
    _ndim: usize,
    _status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    todo!()
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
    _shape: *const usize,
    _ndim: usize,
    _status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    todo!()
}

/// Deep-copy a tensor.
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
    _tensor: *const TfeTensorF64,
    _status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    todo!()
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
pub unsafe extern "C" fn tfe_tensor_f64_release(_tensor: *mut TfeTensorF64) {
    todo!()
}

/// Return the number of dimensions (rank) of the tensor.
///
/// # Safety
///
/// `tensor` must be a valid, non-null tensor pointer.
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_ndim(_tensor: *const TfeTensorF64) -> usize {
    todo!()
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
///
/// # Examples (C)
///
/// ```c
/// size_t ndim = tfe_tensor_f64_ndim(t);
/// size_t *shape = malloc(ndim * sizeof(size_t));
/// tfe_tensor_f64_shape(t, shape);
/// // shape now contains the dimensions
/// free(shape);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_shape(
    _tensor: *const TfeTensorF64,
    _out_shape: *mut usize,
) {
    todo!()
}

/// Return the total number of elements in the tensor.
///
/// # Safety
///
/// `tensor` must be a valid, non-null tensor pointer.
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_len(_tensor: *const TfeTensorF64) -> usize {
    todo!()
}

/// Return a pointer to the tensor's raw data buffer.
///
/// The pointer is valid until `tfe_tensor_f64_release` is called on
/// the tensor.
///
/// # Safety
///
/// `tensor` must be a valid, non-null tensor pointer. The returned
/// pointer must not be used after `tfe_tensor_f64_release(tensor)`.
///
/// # Examples (C)
///
/// ```c
/// const double *ptr = tfe_tensor_f64_data(t);
/// size_t n = tfe_tensor_f64_len(t);
/// for (size_t i = 0; i < n; i++) {
///     printf("%f ", ptr[i]);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tensor_f64_data(_tensor: *const TfeTensorF64) -> *const f64 {
    todo!()
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
    _subscripts: *const c_char,
    _operands: *const *const TfeTensorF64,
    _num_operands: usize,
    _status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    todo!()
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
    _subscripts: *const c_char,
    _operands: *const *const TfeTensorF64,
    _num_operands: usize,
    _cotangent: *const TfeTensorF64,
    _grads_out: *mut *mut TfeTensorF64,
    _status: *mut tfe_status_t,
) {
    todo!()
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
    _subscripts: *const c_char,
    _primals: *const *const TfeTensorF64,
    _num_operands: usize,
    _tangents: *const *const TfeTensorF64,
    _status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    todo!()
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
    _tensor: *const TfeTensorF64,
    _left: *const usize,
    _left_len: usize,
    _right: *const usize,
    _right_len: usize,
    _max_rank: usize,
    _cutoff: f64,
    _u_out: *mut *mut TfeTensorF64,
    _s_out: *mut *mut TfeTensorF64,
    _vt_out: *mut *mut TfeTensorF64,
    _status: *mut tfe_status_t,
) {
    todo!()
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
    _tensor: *const TfeTensorF64,
    _left: *const usize,
    _left_len: usize,
    _right: *const usize,
    _right_len: usize,
    _max_rank: usize,
    _cutoff: f64,
    _cotangent_u: *const TfeTensorF64,
    _cotangent_s: *const TfeTensorF64,
    _cotangent_vt: *const TfeTensorF64,
    _status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    todo!()
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
    _tensor: *const TfeTensorF64,
    _left: *const usize,
    _left_len: usize,
    _right: *const usize,
    _right_len: usize,
    _max_rank: usize,
    _cutoff: f64,
    _tangent: *const TfeTensorF64,
    _u_out: *mut *mut TfeTensorF64,
    _s_out: *mut *mut TfeTensorF64,
    _vt_out: *mut *mut TfeTensorF64,
    _status: *mut tfe_status_t,
) {
    todo!()
}
