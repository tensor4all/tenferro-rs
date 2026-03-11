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
//! # Header generation
//!
//! Generate the extension header from a workspace checkout with:
//!
//! ```text
//! cbindgen \
//!   --config extension/tenferro-tropical-capi/cbindgen.toml \
//!   --crate tenferro-tropical-capi \
//!   --output tenferro_tropical.h
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
use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_capi::{
    tfe_status_t, TfeTensorF64, TFE_INTERNAL_ERROR, TFE_INVALID_ARGUMENT, TFE_SHAPE_MISMATCH,
    TFE_SUCCESS,
};
use tenferro_device::Error as DeviceError;
use tenferro_einsum::{einsum, EinsumBackend};
use tenferro_prims::{CpuBackend, CpuContext, TensorSemiringCore, TensorSemiringFastPath};
use tenferro_tensor::Tensor;
use tenferro_tropical::ad::{
    extract_inner, promote_to_tropical, tropical_einsum_frule, tropical_einsum_rrule,
    TropicalScalar,
};
use tenferro_tropical::{MaxMul, MaxMulAlgebra, MaxPlus, MaxPlusAlgebra, MinPlus, MinPlusAlgebra};

/// Finalize a `catch_unwind` result for functions returning a pointer via status.
unsafe fn finalize_ptr(
    result: std::thread::Result<std::result::Result<*mut TfeTensorF64, tfe_status_t>>,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    match result {
        Ok(Ok(ptr)) => {
            if !status.is_null() {
                *status = TFE_SUCCESS;
            }
            ptr
        }
        Ok(Err(code)) => {
            if !status.is_null() {
                *status = code;
            }
            std::ptr::null_mut()
        }
        Err(_panic) => {
            if !status.is_null() {
                *status = TFE_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
    }
}

/// Finalize a `catch_unwind` result for functions returning void via status.
unsafe fn finalize_void(
    result: std::thread::Result<std::result::Result<(), tfe_status_t>>,
    status: *mut tfe_status_t,
) {
    match result {
        Ok(Ok(())) => {
            if !status.is_null() {
                *status = TFE_SUCCESS;
            }
        }
        Ok(Err(code)) => {
            if !status.is_null() {
                *status = code;
            }
        }
        Err(_panic) => {
            if !status.is_null() {
                *status = TFE_INTERNAL_ERROR;
            }
        }
    }
}

fn tensor_to_handle(tensor: Tensor<f64>) -> *mut TfeTensorF64 {
    Box::into_raw(Box::new(tensor)) as *mut TfeTensorF64
}

unsafe fn handle_to_ref<'a>(handle: *const TfeTensorF64) -> &'a Tensor<f64> {
    &*(handle as *const Tensor<f64>)
}

fn map_device_error(err: &DeviceError) -> tfe_status_t {
    match err {
        DeviceError::ShapeMismatch { .. } | DeviceError::RankMismatch { .. } => TFE_SHAPE_MISMATCH,
        DeviceError::InvalidArgument(_)
        | DeviceError::StrideError(_)
        | DeviceError::CrossMemorySpaceOperation { .. } => TFE_INVALID_ARGUMENT,
        DeviceError::DeviceError(_) | DeviceError::NoCompatibleComputeDevice { .. } => {
            TFE_INTERNAL_ERROR
        }
    }
}

unsafe fn parse_subscripts<'a>(
    subscripts: *const c_char,
) -> std::result::Result<&'a str, tfe_status_t> {
    if subscripts.is_null() {
        return Err(TFE_INVALID_ARGUMENT);
    }
    std::ffi::CStr::from_ptr(subscripts)
        .to_str()
        .map_err(|_| TFE_INVALID_ARGUMENT)
}

unsafe fn collect_operand_handles<'a>(
    operands: *const *const TfeTensorF64,
    num_operands: usize,
) -> std::result::Result<Vec<&'a Tensor<f64>>, tfe_status_t> {
    if operands.is_null() {
        return Err(TFE_INVALID_ARGUMENT);
    }
    std::slice::from_raw_parts(operands, num_operands)
        .iter()
        .map(|&ptr| {
            if ptr.is_null() {
                Err(TFE_INVALID_ARGUMENT)
            } else {
                Ok(handle_to_ref(ptr))
            }
        })
        .collect()
}

unsafe fn collect_optional_tangent_handles<'a>(
    tangents: *const *const TfeTensorF64,
    num_operands: usize,
) -> std::result::Result<Vec<Option<&'a Tensor<f64>>>, tfe_status_t> {
    if tangents.is_null() {
        return Err(TFE_INVALID_ARGUMENT);
    }
    Ok(std::slice::from_raw_parts(tangents, num_operands)
        .iter()
        .map(|&ptr| {
            if ptr.is_null() {
                None
            } else {
                Some(handle_to_ref(ptr))
            }
        })
        .collect())
}

unsafe fn tropical_einsum_impl<T, Alg>(
    subscripts: *const c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
) -> std::result::Result<*mut TfeTensorF64, tfe_status_t>
where
    T: TropicalScalar<Inner = f64> + tenferro_algebra::HasAlgebra<Algebra = Alg>,
    Alg: tenferro_algebra::Semiring<Scalar = T>,
    CpuBackend: EinsumBackend<Alg>
        + TensorSemiringCore<Alg, Context = CpuContext>
        + TensorSemiringFastPath<
            Alg,
            Context = CpuContext,
            Plan = <CpuBackend as TensorSemiringCore<Alg>>::Plan,
        >,
{
    let subscripts = parse_subscripts(subscripts)?;
    let operands = collect_operand_handles(operands, num_operands)?;
    let tropical_operands: Vec<Tensor<T>> = operands
        .iter()
        .map(|tensor| promote_to_tropical::<T>(*tensor).map_err(|err| map_device_error(&err)))
        .collect::<std::result::Result<_, _>>()?;
    let tropical_refs: Vec<&Tensor<T>> = tropical_operands.iter().collect();
    let mut ctx = CpuContext::new(1);
    let output = einsum::<Alg, CpuBackend>(&mut ctx, subscripts, &tropical_refs, None)
        .map_err(|err| map_device_error(&err))?;
    let output = extract_inner::<T>(&output).map_err(|err| map_device_error(&err))?;
    Ok(tensor_to_handle(output))
}

unsafe fn tropical_einsum_rrule_impl<T, Alg>(
    subscripts: *const c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
    cotangent: *const TfeTensorF64,
    grads_out: *mut *mut TfeTensorF64,
) -> std::result::Result<(), tfe_status_t>
where
    T: TropicalScalar<Inner = f64> + tenferro_algebra::HasAlgebra<Algebra = Alg>,
    Alg: tenferro_algebra::Semiring<Scalar = T>,
    CpuBackend: EinsumBackend<Alg>
        + TensorSemiringCore<Alg, Context = CpuContext>
        + TensorSemiringFastPath<
            Alg,
            Context = CpuContext,
            Plan = <CpuBackend as TensorSemiringCore<Alg>>::Plan,
        >,
{
    if cotangent.is_null() || grads_out.is_null() {
        return Err(TFE_INVALID_ARGUMENT);
    }

    let subscripts = parse_subscripts(subscripts)?;
    let operands = collect_operand_handles(operands, num_operands)?;
    let tropical_operands: Vec<Tensor<T>> = operands
        .iter()
        .map(|tensor| promote_to_tropical::<T>(*tensor).map_err(|err| map_device_error(&err)))
        .collect::<std::result::Result<_, _>>()?;
    let tropical_refs: Vec<&Tensor<T>> = tropical_operands.iter().collect();
    let cotangent = handle_to_ref(cotangent);
    let mut ctx = CpuContext::new(1);
    let grads = tropical_einsum_rrule::<T, Alg, CpuBackend>(
        &mut ctx,
        subscripts,
        &tropical_refs,
        cotangent,
    )
    .map_err(|err| map_device_error(&err))?;

    if grads.len() != num_operands {
        return Err(TFE_INTERNAL_ERROR);
    }

    let out = std::slice::from_raw_parts_mut(grads_out, num_operands);
    for (slot, grad) in out.iter_mut().zip(grads.into_iter()) {
        *slot = tensor_to_handle(grad);
    }
    Ok(())
}

unsafe fn tropical_einsum_frule_impl<T, Alg>(
    subscripts: *const c_char,
    primals: *const *const TfeTensorF64,
    num_operands: usize,
    tangents: *const *const TfeTensorF64,
) -> std::result::Result<*mut TfeTensorF64, tfe_status_t>
where
    T: TropicalScalar<Inner = f64> + tenferro_algebra::HasAlgebra<Algebra = Alg>,
    Alg: tenferro_algebra::Semiring<Scalar = T>,
    CpuBackend: EinsumBackend<Alg>
        + TensorSemiringCore<Alg, Context = CpuContext>
        + TensorSemiringFastPath<
            Alg,
            Context = CpuContext,
            Plan = <CpuBackend as TensorSemiringCore<Alg>>::Plan,
        >,
{
    let subscripts = parse_subscripts(subscripts)?;
    let primals = collect_operand_handles(primals, num_operands)?;
    let tangents = collect_optional_tangent_handles(tangents, num_operands)?;
    let tropical_primals: Vec<Tensor<T>> = primals
        .iter()
        .map(|tensor| promote_to_tropical::<T>(*tensor).map_err(|err| map_device_error(&err)))
        .collect::<std::result::Result<_, _>>()?;
    let tropical_refs: Vec<&Tensor<T>> = tropical_primals.iter().collect();
    let mut ctx = CpuContext::new(1);
    let output = tropical_einsum_frule::<T, Alg, CpuBackend>(
        &mut ctx,
        subscripts,
        &tropical_refs,
        &tangents,
    )
    .map_err(|err| map_device_error(&err))?;
    Ok(tensor_to_handle(output))
}

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
    subscripts: *const c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<*mut TfeTensorF64, tfe_status_t> {
            tropical_einsum_impl::<MaxPlus<f64>, MaxPlusAlgebra<f64>>(
                subscripts,
                operands,
                num_operands,
            )
        },
    ));
    finalize_ptr(result, status)
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
    subscripts: *const c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
    cotangent: *const TfeTensorF64,
    grads_out: *mut *mut TfeTensorF64,
    status: *mut tfe_status_t,
) {
    let result = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<(), tfe_status_t> {
            tropical_einsum_rrule_impl::<MaxPlus<f64>, MaxPlusAlgebra<f64>>(
                subscripts,
                operands,
                num_operands,
                cotangent,
                grads_out,
            )
        },
    ));
    finalize_void(result, status)
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
    subscripts: *const c_char,
    primals: *const *const TfeTensorF64,
    num_operands: usize,
    tangents: *const *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<*mut TfeTensorF64, tfe_status_t> {
            tropical_einsum_frule_impl::<MaxPlus<f64>, MaxPlusAlgebra<f64>>(
                subscripts,
                primals,
                num_operands,
                tangents,
            )
        },
    ));
    finalize_ptr(result, status)
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
    subscripts: *const c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<*mut TfeTensorF64, tfe_status_t> {
            tropical_einsum_impl::<MinPlus<f64>, MinPlusAlgebra<f64>>(
                subscripts,
                operands,
                num_operands,
            )
        },
    ));
    finalize_ptr(result, status)
}

/// Reverse-mode rule (VJP) for MinPlus tropical einsum.
///
/// # Safety
///
/// Same as [`tfe_tropical_einsum_rrule_maxplus_f64`].
#[no_mangle]
pub unsafe extern "C" fn tfe_tropical_einsum_rrule_minplus_f64(
    subscripts: *const c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
    cotangent: *const TfeTensorF64,
    grads_out: *mut *mut TfeTensorF64,
    status: *mut tfe_status_t,
) {
    let result = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<(), tfe_status_t> {
            tropical_einsum_rrule_impl::<MinPlus<f64>, MinPlusAlgebra<f64>>(
                subscripts,
                operands,
                num_operands,
                cotangent,
                grads_out,
            )
        },
    ));
    finalize_void(result, status)
}

/// Forward-mode rule (JVP) for MinPlus tropical einsum.
///
/// # Safety
///
/// Same as [`tfe_tropical_einsum_frule_maxplus_f64`].
#[no_mangle]
pub unsafe extern "C" fn tfe_tropical_einsum_frule_minplus_f64(
    subscripts: *const c_char,
    primals: *const *const TfeTensorF64,
    num_operands: usize,
    tangents: *const *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<*mut TfeTensorF64, tfe_status_t> {
            tropical_einsum_frule_impl::<MinPlus<f64>, MinPlusAlgebra<f64>>(
                subscripts,
                primals,
                num_operands,
                tangents,
            )
        },
    ));
    finalize_ptr(result, status)
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
    subscripts: *const c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<*mut TfeTensorF64, tfe_status_t> {
            tropical_einsum_impl::<MaxMul<f64>, MaxMulAlgebra<f64>>(
                subscripts,
                operands,
                num_operands,
            )
        },
    ));
    finalize_ptr(result, status)
}

/// Reverse-mode rule (VJP) for MaxMul tropical einsum.
///
/// # Safety
///
/// Same as [`tfe_tropical_einsum_rrule_maxplus_f64`].
#[no_mangle]
pub unsafe extern "C" fn tfe_tropical_einsum_rrule_maxmul_f64(
    subscripts: *const c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
    cotangent: *const TfeTensorF64,
    grads_out: *mut *mut TfeTensorF64,
    status: *mut tfe_status_t,
) {
    let result = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<(), tfe_status_t> {
            tropical_einsum_rrule_impl::<MaxMul<f64>, MaxMulAlgebra<f64>>(
                subscripts,
                operands,
                num_operands,
                cotangent,
                grads_out,
            )
        },
    ));
    finalize_void(result, status)
}

/// Forward-mode rule (JVP) for MaxMul tropical einsum.
///
/// # Safety
///
/// Same as [`tfe_tropical_einsum_frule_maxplus_f64`].
#[no_mangle]
pub unsafe extern "C" fn tfe_tropical_einsum_frule_maxmul_f64(
    subscripts: *const c_char,
    primals: *const *const TfeTensorF64,
    num_operands: usize,
    tangents: *const *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<*mut TfeTensorF64, tfe_status_t> {
            tropical_einsum_frule_impl::<MaxMul<f64>, MaxMulAlgebra<f64>>(
                subscripts,
                primals,
                num_operands,
                tangents,
            )
        },
    ));
    finalize_ptr(result, status)
}

#[cfg(test)]
mod tests;
