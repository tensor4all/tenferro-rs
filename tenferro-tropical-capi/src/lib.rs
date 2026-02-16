//! C-API (FFI) for tropical semiring tensor operations.
//!
//! Extends [`tenferro_capi`] with tropical einsum functions for three
//! semirings: MaxPlus (⊕=max, ⊗=+), MinPlus (⊕=min, ⊗=+), and
//! MaxMul (⊕=max, ⊗=×).
//!
//! # Design
//!
//! - **Reuses [`TfeTensorF64`]** handles from `tenferro-capi`. Since
//!   `MaxPlus<f64>` is `#[repr(transparent)]`, it has the same memory
//!   layout as `f64`. Tropical functions accept f64 tensor handles,
//!   internally wrap data as `MaxPlus<f64>` (or `MinPlus`/`MaxMul`),
//!   perform the tropical einsum, and unwrap the result back to f64.
//! - **No new handle type** — the algebra is selected by the function
//!   name, not the tensor type.
//! - **Naming convention**: `tfe_tropical_einsum_<algebra>_f64`.
//!
//! # Linking
//!
//! This crate produces a separate shared library from `tenferro-capi`.
//! C/Julia/Python consumers load both:
//!
//! ```c
//! // Core tensor operations
//! tfe_tensor_f64 *t = tfe_tensor_f64_from_data(...);
//!
//! // Tropical einsum (from tenferro-tropical-capi)
//! const tfe_tensor_f64 *ops[] = {a, b};
//! tfe_tensor_f64 *c = tfe_tropical_einsum_maxplus_f64("ij,jk->ik", ops, 2, &status);
//! ```
//!
//! # Example (C pseudocode)
//!
//! ```c
//! #include "tenferro.h"
//! #include "tenferro_tropical.h"
//!
//! tfe_status_t status;
//! size_t shape[] = {3, 4};
//! double data_a[12] = { /* ... */ };
//! double data_b[12] = { /* ... */ };
//!
//! tfe_tensor_f64 *a = tfe_tensor_f64_from_data(data_a, 12, shape, 2, &status);
//! tfe_tensor_f64 *b = tfe_tensor_f64_from_data(data_b, 12, shape, 2, &status);
//!
//! // MaxPlus tropical einsum: C[i,k] = max_j (A[i,j] + B[j,k])
//! const tfe_tensor_f64 *ops[] = {a, b};
//! tfe_tensor_f64 *c = tfe_tropical_einsum_maxplus_f64("ij,jk->ik", ops, 2, &status);
//! assert(status == TFE_SUCCESS);
//!
//! tfe_tensor_f64_release(c);
//! tfe_tensor_f64_release(b);
//! tfe_tensor_f64_release(a);
//! ```

#![allow(clippy::missing_safety_doc)]
#![allow(non_camel_case_types)]

use std::os::raw::c_char;

use tenferro_capi::{tfe_status_t, TfeTensorF64};

// ============================================================================
// MaxPlus einsum
// ============================================================================

/// Execute tropical einsum under MaxPlus algebra (⊕=max, ⊗=+).
///
/// Accepts standard `TfeTensorF64` handles. Data is interpreted as
/// `MaxPlus<f64>` internally: addition becomes `max`, multiplication
/// becomes ordinary `+`.
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
/// // C[i,k] = max_j (A[i,j] + B[j,k])
/// tfe_tensor_f64 *c = tfe_tropical_einsum_maxplus_f64("ij,jk->ik", ops, 2, &status);
/// tfe_tensor_f64_release(c);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tropical_einsum_maxplus_f64(
    _subscripts: *const c_char,
    _operands: *const *const TfeTensorF64,
    _num_operands: usize,
    _status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    todo!()
}

/// Reverse-mode rule (VJP) for MaxPlus tropical einsum.
///
/// Computes one gradient tensor per input operand given the output
/// cotangent. The caller must provide `grads_out` as a pre-allocated
/// array of `num_operands` pointers.
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
/// tfe_tensor_f64 *grads[2];
/// tfe_status_t status;
/// const tfe_tensor_f64 *ops[] = {a, b};
/// tfe_tropical_einsum_rrule_maxplus_f64("ij,jk->ik", ops, 2, grad_c, grads, &status);
/// tfe_tensor_f64_release(grads[0]);
/// tfe_tensor_f64_release(grads[1]);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tropical_einsum_rrule_maxplus_f64(
    _subscripts: *const c_char,
    _operands: *const *const TfeTensorF64,
    _num_operands: usize,
    _cotangent: *const TfeTensorF64,
    _grads_out: *mut *mut TfeTensorF64,
    _status: *mut tfe_status_t,
) {
    todo!()
}

/// Forward-mode rule (JVP) for MaxPlus tropical einsum.
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
/// const tfe_tensor_f64 *tangents[] = {da, NULL};
/// tfe_status_t status;
/// tfe_tensor_f64 *dc = tfe_tropical_einsum_frule_maxplus_f64(
///     "ij,jk->ik", primals, 2, tangents, &status);
/// tfe_tensor_f64_release(dc);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tropical_einsum_frule_maxplus_f64(
    _subscripts: *const c_char,
    _primals: *const *const TfeTensorF64,
    _num_operands: usize,
    _tangents: *const *const TfeTensorF64,
    _status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    todo!()
}

// ============================================================================
// MinPlus einsum
// ============================================================================

/// Execute tropical einsum under MinPlus algebra (⊕=min, ⊗=+).
///
/// Same interface as [`tfe_tropical_einsum_maxplus_f64`] but uses
/// min-plus semantics: addition becomes `min`, multiplication becomes `+`.
///
/// # Safety
///
/// Same as [`tfe_tropical_einsum_maxplus_f64`].
///
/// # Examples (C)
///
/// ```c
/// const tfe_tensor_f64 *ops[] = {a, b};
/// tfe_status_t status;
/// // C[i,k] = min_j (A[i,j] + B[j,k])
/// tfe_tensor_f64 *c = tfe_tropical_einsum_minplus_f64("ij,jk->ik", ops, 2, &status);
/// tfe_tensor_f64_release(c);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tropical_einsum_minplus_f64(
    _subscripts: *const c_char,
    _operands: *const *const TfeTensorF64,
    _num_operands: usize,
    _status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    todo!()
}

/// Reverse-mode rule (VJP) for MinPlus tropical einsum.
///
/// # Safety
///
/// Same as [`tfe_tropical_einsum_rrule_maxplus_f64`].
#[no_mangle]
pub unsafe extern "C" fn tfe_tropical_einsum_rrule_minplus_f64(
    _subscripts: *const c_char,
    _operands: *const *const TfeTensorF64,
    _num_operands: usize,
    _cotangent: *const TfeTensorF64,
    _grads_out: *mut *mut TfeTensorF64,
    _status: *mut tfe_status_t,
) {
    todo!()
}

/// Forward-mode rule (JVP) for MinPlus tropical einsum.
///
/// # Safety
///
/// Same as [`tfe_tropical_einsum_frule_maxplus_f64`].
#[no_mangle]
pub unsafe extern "C" fn tfe_tropical_einsum_frule_minplus_f64(
    _subscripts: *const c_char,
    _primals: *const *const TfeTensorF64,
    _num_operands: usize,
    _tangents: *const *const TfeTensorF64,
    _status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    todo!()
}

// ============================================================================
// MaxMul einsum
// ============================================================================

/// Execute tropical einsum under MaxMul algebra (⊕=max, ⊗=×).
///
/// Same interface as [`tfe_tropical_einsum_maxplus_f64`] but uses
/// max-times semantics: addition becomes `max`, multiplication becomes `×`.
///
/// # Safety
///
/// Same as [`tfe_tropical_einsum_maxplus_f64`].
///
/// # Examples (C)
///
/// ```c
/// const tfe_tensor_f64 *ops[] = {a, b};
/// tfe_status_t status;
/// // C[i,k] = max_j (A[i,j] * B[j,k])
/// tfe_tensor_f64 *c = tfe_tropical_einsum_maxmul_f64("ij,jk->ik", ops, 2, &status);
/// tfe_tensor_f64_release(c);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_tropical_einsum_maxmul_f64(
    _subscripts: *const c_char,
    _operands: *const *const TfeTensorF64,
    _num_operands: usize,
    _status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    todo!()
}

/// Reverse-mode rule (VJP) for MaxMul tropical einsum.
///
/// # Safety
///
/// Same as [`tfe_tropical_einsum_rrule_maxplus_f64`].
#[no_mangle]
pub unsafe extern "C" fn tfe_tropical_einsum_rrule_maxmul_f64(
    _subscripts: *const c_char,
    _operands: *const *const TfeTensorF64,
    _num_operands: usize,
    _cotangent: *const TfeTensorF64,
    _grads_out: *mut *mut TfeTensorF64,
    _status: *mut tfe_status_t,
) {
    todo!()
}

/// Forward-mode rule (JVP) for MaxMul tropical einsum.
///
/// # Safety
///
/// Same as [`tfe_tropical_einsum_frule_maxplus_f64`].
#[no_mangle]
pub unsafe extern "C" fn tfe_tropical_einsum_frule_maxmul_f64(
    _subscripts: *const c_char,
    _primals: *const *const TfeTensorF64,
    _num_operands: usize,
    _tangents: *const *const TfeTensorF64,
    _status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    todo!()
}
