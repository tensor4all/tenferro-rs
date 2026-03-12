use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_device::LogicalMemorySpace;
use tenferro_linalg::{svd, svd_frule, svd_rrule, SvdCotangent, SvdOptions};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::ffi_utils::read_usize_slice;
use crate::handle::{handle_to_ref, tensor_to_handle, TfeTensorF64};
use crate::status::{
    finalize_ptr, finalize_void, map_ad_error, map_device_error, tfe_status_t,
    TFE_INVALID_ARGUMENT, TFE_SHAPE_MISMATCH,
};

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

fn matricize(
    tensor: &Tensor<f64>,
    left: &[usize],
    right: &[usize],
) -> Result<(Tensor<f64>, Vec<usize>, Vec<usize>), tfe_status_t> {
    let dims = tensor.dims();
    let mut seen = vec![false; dims.len()];

    for &l in left {
        if l >= dims.len() || seen[l] {
            return Err(TFE_INVALID_ARGUMENT);
        }
        seen[l] = true;
    }
    for &r in right {
        if r >= dims.len() || seen[r] {
            return Err(TFE_INVALID_ARGUMENT);
        }
        seen[r] = true;
    }
    if left.len() + right.len() != dims.len() || seen.iter().any(|&v| !v) {
        return Err(TFE_INVALID_ARGUMENT);
    }

    let left_dims: Vec<usize> = left.iter().map(|&i| dims[i]).collect();
    let right_dims: Vec<usize> = right.iter().map(|&i| dims[i]).collect();
    let m: usize = left_dims.iter().product();
    let n: usize = right_dims.iter().product();

    let mut perm: Vec<usize> = Vec::with_capacity(dims.len());
    perm.extend_from_slice(left);
    perm.extend_from_slice(right);

    let permuted = tensor
        .permute(&perm)
        .map_err(|_| TFE_INVALID_ARGUMENT)?
        .contiguous(MemoryOrder::ColumnMajor)
        .reshape(&[m, n])
        .map_err(|_| TFE_SHAPE_MISMATCH)?
        .contiguous(MemoryOrder::ColumnMajor);

    Ok((permuted, left_dims, right_dims))
}

fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

fn u_cotangent_to_matrix(
    cot_u: &Tensor<f64>,
    left_dims: &[usize],
) -> Result<(Tensor<f64>, usize), tfe_status_t> {
    let u_dims = cot_u.dims();
    if u_dims.len() != left_dims.len() + 1 || &u_dims[..left_dims.len()] != left_dims {
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

fn vt_cotangent_to_matrix(
    cot_vt: &Tensor<f64>,
    right_dims: &[usize],
) -> Result<(Tensor<f64>, usize), tfe_status_t> {
    let vt_dims = cot_vt.dims();
    if vt_dims.len() != right_dims.len() + 1 || &vt_dims[1..] != right_dims {
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

fn validate_s_cotangent(cot_s: &Tensor<f64>) -> Result<usize, tfe_status_t> {
    let dims = cot_s.dims();
    if dims.len() != 1 {
        return Err(TFE_SHAPE_MISMATCH);
    }
    Ok(dims[0])
}

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
        .map_err(|_| TFE_INVALID_ARGUMENT)
        .map(|t| t.contiguous(MemoryOrder::ColumnMajor))
}

/// Compute the SVD of a tensor after matricizing by `left` and `right`.
///
/// # Safety
///
/// - `tensor` must be valid and non-null.
/// - `left` and `right` must point to valid index arrays.
/// - `u_out`, `s_out`, `vt_out` must be valid, non-null pointers.
/// - `status` must be valid.
///
/// # Examples (C)
///
/// ```c
/// size_t left[] = {0};
/// size_t right[] = {1};
/// tfe_tensor_f64 *u, *s, *vt;
/// tfe_status_t status;
/// tfe_svd_f64(a, left, 1, right, 1, 0, -1.0, &u, &s, &vt, &status);
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
        let t = handle_to_ref(tensor);
        let left_indices = read_usize_slice(left, left_len, "svd left indices")?;
        let right_indices = read_usize_slice(right, right_len, "svd right indices")?;

        let (matrix, left_dims, right_dims) = matricize(t, left_indices, right_indices)?;

        let opts = build_svd_options(max_rank, cutoff);
        let mut ctx = CpuContext::new(1);
        let result = svd(&mut ctx, &matrix, opts.as_ref()).map_err(|e| map_device_error(&e))?;

        let k = result.s.len();
        let mut u_dims: Vec<usize> = left_dims;
        u_dims.push(k);
        let u_reshaped = result
            .u
            .reshape(&u_dims)
            .map_err(|e| map_device_error(&e))?
            .contiguous(MemoryOrder::ColumnMajor);

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
/// # Safety
///
/// - `tensor` must be valid and non-null.
/// - `left` and `right` must point to valid index arrays.
/// - `status` must be valid.
///
/// # Examples (C)
///
/// ```c
/// tfe_tensor_f64 *grad = tfe_svd_rrule_f64(a, left, 1, right, 1, 0, -1.0, NULL, cot_s, NULL, &status);
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

        let t = handle_to_ref(tensor);
        let left_indices = read_usize_slice(left, left_len, "svd_rrule left indices")?;
        let right_indices = read_usize_slice(right, right_len, "svd_rrule right indices")?;
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
        let mut ctx = CpuContext::new(1);
        let grad_matrix = svd_rrule(&mut ctx, &matrix, &cotangent, opts.as_ref())
            .map_err(|e| map_ad_error(&e))?;
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
/// # Safety
///
/// - `tensor` must be valid and non-null.
/// - `left` and `right` must point to valid index arrays.
/// - `u_out`, `s_out`, `vt_out` must be valid, non-null pointers.
/// - `status` must be valid.
///
/// # Examples (C)
///
/// ```c
/// tfe_svd_frule_f64(a, left, 1, right, 1, 0, -1.0, da, &du, &ds, &dvt, &status);
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

        let t = handle_to_ref(tensor);
        let left_indices = read_usize_slice(left, left_len, "svd_frule left indices")?;
        let right_indices = read_usize_slice(right, right_len, "svd_frule right indices")?;
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
        let mut ctx = CpuContext::new(1);
        let (_primal, tangent_result) =
            svd_frule(&mut ctx, &matrix, &tang, opts.as_ref()).map_err(|e| map_ad_error(&e))?;

        let u_public = u_matrix_to_public(tangent_result.u, &left_dims)?;
        let vt_public = vt_matrix_to_public(tangent_result.vt, &right_dims)?;

        *u_out = tensor_to_handle(u_public);
        *s_out = tensor_to_handle(tangent_result.s);
        *vt_out = tensor_to_handle(vt_public);
        Ok(())
    }));

    finalize_void(result, status)
}
