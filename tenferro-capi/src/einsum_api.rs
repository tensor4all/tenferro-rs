use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_algebra::Standard;
use tenferro_einsum::{einsum, einsum_frule, einsum_rrule};
use tenferro_prims::CpuBackend;

use crate::ffi_utils::{cpu_context, read_c_str, read_optional_tensor_refs, read_tensor_refs};
use crate::handle::{ensure_col_major, handle_to_ref, tensor_to_handle, TfeTensorF64};
use crate::status::{
    finalize_ptr, finalize_void, map_device_error, tfe_status_t, TFE_INVALID_ARGUMENT,
};

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
/// tfe_tensor_f64_release(c);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_einsum_f64(
    subscripts: *const std::os::raw::c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let subs = read_c_str(subscripts, "einsum subscripts")?;
        let ops = read_tensor_refs(operands, num_operands, "einsum operands")?;

        let mut ctx = cpu_context()?;
        einsum::<Standard<f64>, CpuBackend>(&mut ctx, subs, &ops, None)
            .and_then(|t| ensure_col_major(&mut ctx, t))
            .map(tensor_to_handle)
            .map_err(|e| map_device_error(&e))
    }));

    finalize_ptr(result, status)
}

/// Reverse-mode rule (VJP) for einsum.
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
/// tfe_einsum_rrule_f64("ij,jk->ik", ops, 2, grad_c, grads, &status);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_einsum_rrule_f64(
    subscripts: *const std::os::raw::c_char,
    operands: *const *const TfeTensorF64,
    num_operands: usize,
    cotangent: *const TfeTensorF64,
    grads_out: *mut *mut TfeTensorF64,
    status: *mut tfe_status_t,
) {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if cotangent.is_null() || grads_out.is_null() {
            return Err(TFE_INVALID_ARGUMENT);
        }

        let subs = read_c_str(subscripts, "einsum_rrule subscripts")?;
        let ops = read_tensor_refs(operands, num_operands, "einsum_rrule operands")?;
        let cot = handle_to_ref(cotangent);

        let mut ctx = cpu_context()?;
        let grads = einsum_rrule::<Standard<f64>, CpuBackend>(&mut ctx, subs, &ops, cot)
            .map_err(|e| map_device_error(&e))?;

        let out_slice = std::slice::from_raw_parts_mut(grads_out, num_operands);
        for (i, g) in grads.into_iter().enumerate() {
            let g = ensure_col_major(&mut ctx, g).map_err(|e| map_device_error(&e))?;
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
/// const tfe_tensor_f64 *tangents[] = {da, NULL};
/// tfe_status_t status;
/// tfe_tensor_f64 *dc = tfe_einsum_frule_f64("ij,jk->ik", primals, 2, tangents, &status);
/// tfe_tensor_f64_release(dc);
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_einsum_frule_f64(
    subscripts: *const std::os::raw::c_char,
    primals: *const *const TfeTensorF64,
    num_operands: usize,
    tangents: *const *const TfeTensorF64,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let subs = read_c_str(subscripts, "einsum_frule subscripts")?;
        let primal_refs = read_tensor_refs(primals, num_operands, "einsum_frule primals")?;
        let tangent_refs =
            read_optional_tensor_refs(tangents, num_operands, "einsum_frule tangents")?;

        let mut ctx = cpu_context()?;
        einsum_frule::<Standard<f64>, CpuBackend>(&mut ctx, subs, &primal_refs, &tangent_refs)
            .and_then(|t| ensure_col_major(&mut ctx, t))
            .map(tensor_to_handle)
            .map_err(|e| map_device_error(&e))
    }));

    finalize_ptr(result, status)
}
