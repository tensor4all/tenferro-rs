//! Tests for tenferro-capi: C-API tensor lifecycle, einsum, SVD.
//!
//! TDD approach: tests written to verify correctness based on docs/design/capi.md.

use tenferro_capi::*;
use tenferro_linalg::backend::CpuTensorLinalgContext;
use tenferro_linalg::{svd_frule, svd_rrule, SvdCotangent};
use tenferro_tensor::{MemoryOrder, Tensor};

fn approx_eq_slice(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {} != {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "mismatch at {i}: actual={a}, expected={e}, |diff|={}",
            (a - e).abs()
        );
    }
}

unsafe fn handle_dims_data(
    t: *const TfeTensorF64,
    status: &mut tfe_status_t,
) -> (Vec<usize>, Vec<f64>) {
    let ndim = tfe_tensor_f64_ndim(t, status);
    assert_eq!(*status, TFE_SUCCESS);
    let mut dims = vec![0usize; ndim];
    tfe_tensor_f64_shape(t, dims.as_mut_ptr(), status);
    assert_eq!(*status, TFE_SUCCESS);
    let len = tfe_tensor_f64_len(t, status);
    assert_eq!(*status, TFE_SUCCESS);
    let ptr = tfe_tensor_f64_data(t, status);
    assert_eq!(*status, TFE_SUCCESS);
    let data = std::slice::from_raw_parts(ptr, len).to_vec();
    (dims, data)
}

fn matrixize_for_test(
    tensor: &Tensor<f64>,
    left: &[usize],
    right: &[usize],
) -> (Tensor<f64>, Vec<usize>, Vec<usize>) {
    let dims = tensor.dims();
    let left_dims: Vec<usize> = left.iter().map(|&i| dims[i]).collect();
    let right_dims: Vec<usize> = right.iter().map(|&i| dims[i]).collect();
    let m: usize = left_dims.iter().product();
    let n: usize = right_dims.iter().product();

    let mut perm = Vec::with_capacity(dims.len());
    perm.extend_from_slice(left);
    perm.extend_from_slice(right);
    let matrix = tensor
        .permute(&perm)
        .unwrap()
        .contiguous(MemoryOrder::ColumnMajor)
        .reshape(&[m, n])
        .unwrap()
        .contiguous(MemoryOrder::ColumnMajor);
    (matrix, left_dims, right_dims)
}

fn unmatrixize_grad_for_test(
    grad_matrix: Tensor<f64>,
    left: &[usize],
    right: &[usize],
    left_dims: &[usize],
    right_dims: &[usize],
) -> Tensor<f64> {
    let mut perm_dims = left_dims.to_vec();
    perm_dims.extend_from_slice(right_dims);
    let reshaped = grad_matrix.reshape(&perm_dims).unwrap();

    let mut perm = Vec::with_capacity(left.len() + right.len());
    perm.extend_from_slice(left);
    perm.extend_from_slice(right);
    let mut inv = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    reshaped
        .permute(&inv)
        .unwrap()
        .contiguous(MemoryOrder::ColumnMajor)
}

// ============================================================================
// Phase 1: Tensor lifecycle tests
// ============================================================================

#[test]
fn from_data_round_trip() {
    unsafe {
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2_usize, 3];
        let mut status: tfe_status_t = -999;

        let t = tfe_tensor_f64_from_data(
            data.as_ptr(),
            data.len(),
            shape.as_ptr(),
            shape.len(),
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);
        assert!(!t.is_null());

        // Query metadata
        assert_eq!(tfe_tensor_f64_ndim(t, &mut status), 2);
        assert_eq!(status, TFE_SUCCESS);
        assert_eq!(tfe_tensor_f64_len(t, &mut status), 6);
        assert_eq!(status, TFE_SUCCESS);

        let mut out_shape = [0_usize; 2];
        tfe_tensor_f64_shape(t, out_shape.as_mut_ptr(), &mut status);
        assert_eq!(status, TFE_SUCCESS);
        assert_eq!(out_shape, [2, 3]);

        // Query data
        let ptr = tfe_tensor_f64_data(t, &mut status);
        assert_eq!(status, TFE_SUCCESS);
        assert!(!ptr.is_null());
        let slice = std::slice::from_raw_parts(ptr, 6);
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        tfe_tensor_f64_release(t);
    }
}

#[test]
fn zeros_creates_zero_filled() {
    unsafe {
        let shape = [3_usize, 4];
        let mut status: tfe_status_t = -999;

        let t = tfe_tensor_f64_zeros(shape.as_ptr(), shape.len(), &mut status);
        assert_eq!(status, TFE_SUCCESS);
        assert!(!t.is_null());

        assert_eq!(tfe_tensor_f64_ndim(t, &mut status), 2);
        assert_eq!(tfe_tensor_f64_len(t, &mut status), 12);

        let ptr = tfe_tensor_f64_data(t, &mut status);
        let slice = std::slice::from_raw_parts(ptr, 12);
        assert!(slice.iter().all(|&x| x == 0.0));

        tfe_tensor_f64_release(t);
    }
}

#[test]
fn clone_creates_independent_copy() {
    unsafe {
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let shape = [2_usize, 2];
        let mut status: tfe_status_t = -999;

        let t = tfe_tensor_f64_from_data(
            data.as_ptr(),
            data.len(),
            shape.as_ptr(),
            shape.len(),
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);

        let t2 = tfe_tensor_f64_clone(t, &mut status);
        assert_eq!(status, TFE_SUCCESS);
        assert!(!t2.is_null());

        // Both should have same data
        let ptr1 = tfe_tensor_f64_data(t, &mut status);
        let ptr2 = tfe_tensor_f64_data(t2, &mut status);
        let s1 = std::slice::from_raw_parts(ptr1, 4);
        let s2 = std::slice::from_raw_parts(ptr2, 4);
        assert_eq!(s1, s2);

        // But different addresses (independent copy)
        assert_ne!(ptr1, ptr2);

        tfe_tensor_f64_release(t);
        tfe_tensor_f64_release(t2);
    }
}

#[test]
fn release_null_is_noop() {
    unsafe {
        // Should not crash
        tfe_tensor_f64_release(std::ptr::null_mut());
    }
}

#[test]
fn from_data_null_data_returns_error() {
    unsafe {
        let shape = [2_usize, 3];
        let mut status: tfe_status_t = -999;

        let t = tfe_tensor_f64_from_data(
            std::ptr::null(),
            6,
            shape.as_ptr(),
            shape.len(),
            &mut status,
        );
        assert_eq!(status, TFE_INVALID_ARGUMENT);
        assert!(t.is_null());
    }
}

#[test]
fn from_data_null_shape_returns_error() {
    unsafe {
        let data = [1.0_f64; 6];
        let mut status: tfe_status_t = -999;

        let t = tfe_tensor_f64_from_data(data.as_ptr(), 6, std::ptr::null(), 2, &mut status);
        assert_eq!(status, TFE_INVALID_ARGUMENT);
        assert!(t.is_null());
    }
}

#[test]
fn from_data_mismatched_len_returns_error() {
    unsafe {
        let data = [1.0_f64; 5]; // 5 elements
        let shape = [2_usize, 3]; // needs 6
        let mut status: tfe_status_t = -999;

        let t = tfe_tensor_f64_from_data(
            data.as_ptr(),
            data.len(),
            shape.as_ptr(),
            shape.len(),
            &mut status,
        );
        assert_ne!(status, TFE_SUCCESS);
        assert!(t.is_null());
    }
}

#[test]
fn scalar_tensor() {
    unsafe {
        let data = [42.0_f64];
        let mut status: tfe_status_t = -999;

        // Scalar = 0-dim tensor
        let t = tfe_tensor_f64_from_data(
            data.as_ptr(),
            1,
            std::ptr::null(), // no shape for scalar
            0,
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);
        assert!(!t.is_null());

        assert_eq!(tfe_tensor_f64_ndim(t, &mut status), 0);
        assert_eq!(tfe_tensor_f64_len(t, &mut status), 1);

        let ptr = tfe_tensor_f64_data(t, &mut status);
        assert_eq!(*ptr, 42.0);

        tfe_tensor_f64_release(t);
    }
}

// ============================================================================
// Query function null guard tests
// ============================================================================

#[test]
fn ndim_null_tensor_returns_error() {
    unsafe {
        let mut status: tfe_status_t = -999;
        let n = tfe_tensor_f64_ndim(std::ptr::null(), &mut status);
        assert_eq!(status, TFE_INVALID_ARGUMENT);
        assert_eq!(n, 0);
    }
}

#[test]
fn shape_null_tensor_returns_error() {
    unsafe {
        let mut status: tfe_status_t = -999;
        let mut out = [0_usize; 2];
        tfe_tensor_f64_shape(std::ptr::null(), out.as_mut_ptr(), &mut status);
        assert_eq!(status, TFE_INVALID_ARGUMENT);
    }
}

#[test]
fn len_null_tensor_returns_error() {
    unsafe {
        let mut status: tfe_status_t = -999;
        let n = tfe_tensor_f64_len(std::ptr::null(), &mut status);
        assert_eq!(status, TFE_INVALID_ARGUMENT);
        assert_eq!(n, 0);
    }
}

#[test]
fn data_null_tensor_returns_error() {
    unsafe {
        let mut status: tfe_status_t = -999;
        let ptr = tfe_tensor_f64_data(std::ptr::null(), &mut status);
        assert_eq!(status, TFE_INVALID_ARGUMENT);
        assert!(ptr.is_null());
    }
}

// ============================================================================
// Last-error message API tests
// ============================================================================

#[test]
fn last_error_null_out_len_returns_error() {
    unsafe {
        let status = tfe_last_error_message(std::ptr::null_mut(), 0, std::ptr::null_mut());
        assert_eq!(status, TFE_INVALID_ARGUMENT);
    }
}

#[test]
fn last_error_query_length_only() {
    unsafe {
        // Trigger an error to populate last-error
        let mut status: tfe_status_t = -999;
        let _ = tfe_tensor_f64_from_data(std::ptr::null(), 6, std::ptr::null(), 2, &mut status);
        assert_ne!(status, TFE_SUCCESS);

        // Query length (buf=NULL)
        let mut out_len: usize = 0;
        let s = tfe_last_error_message(std::ptr::null_mut(), 0, &mut out_len);
        assert_eq!(s, TFE_SUCCESS);
        assert!(out_len > 0); // includes null terminator
    }
}

#[test]
fn last_error_buffer_too_small() {
    unsafe {
        // Trigger an error that goes through map_device_error (shape mismatch)
        let data = [1.0_f64; 5]; // 5 elements
        let shape = [2_usize, 3]; // needs 6
        let mut status: tfe_status_t = -999;
        let _ = tfe_tensor_f64_from_data(
            data.as_ptr(),
            data.len(),
            shape.as_ptr(),
            shape.len(),
            &mut status,
        );
        assert_ne!(status, TFE_SUCCESS);

        // Query required length
        let mut out_len: usize = 0;
        tfe_last_error_message(std::ptr::null_mut(), 0, &mut out_len);
        assert!(out_len > 1);

        // Provide a too-small buffer
        let mut buf = vec![0u8; 1];
        let s = tfe_last_error_message(buf.as_mut_ptr(), buf.len(), &mut out_len);
        assert_eq!(s, TFE_BUFFER_TOO_SMALL);
        assert!(out_len > 1);
    }
}

#[test]
fn last_error_round_trip() {
    unsafe {
        // Trigger shape mismatch error via from_data
        let data = [1.0_f64; 5]; // 5 elements
        let shape = [2_usize, 3]; // needs 6
        let mut status: tfe_status_t = -999;
        let _ = tfe_tensor_f64_from_data(
            data.as_ptr(),
            data.len(),
            shape.as_ptr(),
            shape.len(),
            &mut status,
        );
        assert_ne!(status, TFE_SUCCESS);

        // Query length
        let mut out_len: usize = 0;
        tfe_last_error_message(std::ptr::null_mut(), 0, &mut out_len);

        if out_len > 0 {
            // Read the message
            let mut buf = vec![0u8; out_len];
            let s = tfe_last_error_message(buf.as_mut_ptr(), buf.len(), &mut out_len);
            assert_eq!(s, TFE_SUCCESS);

            // Should be a valid null-terminated string
            let msg = std::ffi::CStr::from_ptr(buf.as_ptr() as *const i8)
                .to_str()
                .unwrap();
            assert!(!msg.is_empty());
        }
    }
}

// ============================================================================
// Phase 3: Einsum tests
// ============================================================================

#[test]
fn einsum_matmul() {
    unsafe {
        // A = [[1, 2], [3, 4]]  (2x2 col-major: [1, 3, 2, 4])
        // B = [[5, 6], [7, 8]]  (2x2 col-major: [5, 7, 6, 8])
        // C = A * B = [[19, 22], [43, 50]]  (col-major: [19, 43, 22, 50])
        let a_data = [1.0_f64, 3.0, 2.0, 4.0];
        let b_data = [5.0_f64, 7.0, 6.0, 8.0];
        let shape = [2_usize, 2];
        let mut status: tfe_status_t = -999;

        let a = tfe_tensor_f64_from_data(a_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);
        assert_eq!(status, TFE_SUCCESS);

        let b = tfe_tensor_f64_from_data(b_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);
        assert_eq!(status, TFE_SUCCESS);

        let subscripts = b"ij,jk->ik\0";
        let ops = [a as *const TfeTensorF64, b as *const TfeTensorF64];

        let c = tfe_einsum_f64(
            subscripts.as_ptr() as *const i8,
            ops.as_ptr(),
            2,
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);
        assert!(!c.is_null());

        assert_eq!(tfe_tensor_f64_len(c, &mut status), 4);
        let ptr = tfe_tensor_f64_data(c, &mut status);
        let result = std::slice::from_raw_parts(ptr, 4);

        // Column-major: C = [[19, 22], [43, 50]] -> [19, 43, 22, 50]
        assert!((result[0] - 19.0).abs() < 1e-10);
        assert!((result[1] - 43.0).abs() < 1e-10);
        assert!((result[2] - 22.0).abs() < 1e-10);
        assert!((result[3] - 50.0).abs() < 1e-10);

        tfe_tensor_f64_release(c);
        tfe_tensor_f64_release(b as *mut _);
        tfe_tensor_f64_release(a);
    }
}

#[test]
fn einsum_trace() {
    unsafe {
        // A = [[1, 2], [3, 4]], trace = 1 + 4 = 5
        let a_data = [1.0_f64, 3.0, 2.0, 4.0]; // col-major
        let shape = [2_usize, 2];
        let mut status: tfe_status_t = -999;

        let a = tfe_tensor_f64_from_data(a_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);

        let subscripts = b"ii->\0";
        let ops = [a as *const TfeTensorF64];

        let c = tfe_einsum_f64(
            subscripts.as_ptr() as *const i8,
            ops.as_ptr(),
            1,
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);
        assert!(!c.is_null());

        assert_eq!(tfe_tensor_f64_len(c, &mut status), 1);
        let ptr = tfe_tensor_f64_data(c, &mut status);
        assert!(((*ptr) - 5.0).abs() < 1e-10);

        tfe_tensor_f64_release(c);
        tfe_tensor_f64_release(a);
    }
}

#[test]
fn einsum_rrule_matmul() {
    unsafe {
        // A (2x2), B (2x2), einsum "ij,jk->ik"
        // cotangent = ones(2x2)
        let a_data = [1.0_f64, 3.0, 2.0, 4.0];
        let b_data = [5.0_f64, 7.0, 6.0, 8.0];
        let cot_data = [1.0_f64, 1.0, 1.0, 1.0];
        let shape = [2_usize, 2];
        let mut status: tfe_status_t = -999;

        let a = tfe_tensor_f64_from_data(a_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);
        let b = tfe_tensor_f64_from_data(b_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);
        let cot = tfe_tensor_f64_from_data(cot_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);

        let subscripts = b"ij,jk->ik\0";
        let ops = [a as *const TfeTensorF64, b as *const TfeTensorF64];
        let mut grads = [std::ptr::null_mut::<TfeTensorF64>(); 2];

        tfe_einsum_rrule_f64(
            subscripts.as_ptr() as *const i8,
            ops.as_ptr(),
            2,
            cot as *const TfeTensorF64,
            grads.as_mut_ptr(),
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);
        assert!(!grads[0].is_null());
        assert!(!grads[1].is_null());

        // grad_A shape should be (2, 2)
        assert_eq!(tfe_tensor_f64_len(grads[0] as *const _, &mut status), 4);
        // grad_B shape should be (2, 2)
        assert_eq!(tfe_tensor_f64_len(grads[1] as *const _, &mut status), 4);

        // Numeric oracle:
        // grad_A = cot * B^T, grad_B = A^T * cot, with cot = ones(2x2).
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]].
        // grad_A = [[11,15],[11,15]]  -> col-major [11,11,15,15]
        // grad_B = [[4,4],[6,6]]      -> col-major [4,6,4,6]
        let grad_a_ptr = tfe_tensor_f64_data(grads[0] as *const _, &mut status);
        assert_eq!(status, TFE_SUCCESS);
        let grad_b_ptr = tfe_tensor_f64_data(grads[1] as *const _, &mut status);
        assert_eq!(status, TFE_SUCCESS);
        let grad_a = std::slice::from_raw_parts(grad_a_ptr, 4);
        let grad_b = std::slice::from_raw_parts(grad_b_ptr, 4);
        approx_eq_slice(grad_a, &[11.0, 11.0, 15.0, 15.0], 1e-12);
        approx_eq_slice(grad_b, &[4.0, 6.0, 4.0, 6.0], 1e-12);

        tfe_tensor_f64_release(grads[0]);
        tfe_tensor_f64_release(grads[1]);
        tfe_tensor_f64_release(cot as *mut _);
        tfe_tensor_f64_release(b as *mut _);
        tfe_tensor_f64_release(a);
    }
}

#[test]
fn einsum_frule_matmul() {
    unsafe {
        // A (2x2), B (2x2), einsum "ij,jk->ik"
        // tangent for A, none for B
        let a_data = [1.0_f64, 3.0, 2.0, 4.0];
        let b_data = [5.0_f64, 7.0, 6.0, 8.0];
        let da_data = [1.0_f64, 0.0, 0.0, 0.0]; // tangent for A
        let shape = [2_usize, 2];
        let mut status: tfe_status_t = -999;

        let a = tfe_tensor_f64_from_data(a_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);
        let b = tfe_tensor_f64_from_data(b_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);
        let da = tfe_tensor_f64_from_data(da_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);

        let subscripts = b"ij,jk->ik\0";
        let primals = [a as *const TfeTensorF64, b as *const TfeTensorF64];
        let tangents = [da as *const TfeTensorF64, std::ptr::null::<TfeTensorF64>()];

        let dc = tfe_einsum_frule_f64(
            subscripts.as_ptr() as *const i8,
            primals.as_ptr(),
            2,
            tangents.as_ptr(),
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);
        assert!(!dc.is_null());

        // dc = dA * B = [[1,0],[0,0]] * [[5,6],[7,8]] = [[5,6],[0,0]]
        // col-major: [5, 0, 6, 0]
        let ptr = tfe_tensor_f64_data(dc, &mut status);
        let result = std::slice::from_raw_parts(ptr, 4);
        assert!((result[0] - 5.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] - 6.0).abs() < 1e-10);
        assert!((result[3] - 0.0).abs() < 1e-10);

        tfe_tensor_f64_release(dc);
        tfe_tensor_f64_release(da as *mut _);
        tfe_tensor_f64_release(b as *mut _);
        tfe_tensor_f64_release(a);
    }
}

// ============================================================================
// Phase 3: SVD tests
// ============================================================================

#[test]
fn svd_reconstruction() {
    unsafe {
        // A = [[1, 2], [3, 4]]  (col-major: [1, 3, 2, 4])
        let a_data = [1.0_f64, 3.0, 2.0, 4.0];
        let shape = [2_usize, 2];
        let mut status: tfe_status_t = -999;

        let a = tfe_tensor_f64_from_data(a_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);
        assert_eq!(status, TFE_SUCCESS);

        let left = [0_usize];
        let right = [1_usize];
        let mut u: *mut TfeTensorF64 = std::ptr::null_mut();
        let mut s: *mut TfeTensorF64 = std::ptr::null_mut();
        let mut vt: *mut TfeTensorF64 = std::ptr::null_mut();

        tfe_svd_f64(
            a as *const _,
            left.as_ptr(),
            1,
            right.as_ptr(),
            1,
            0,
            -1.0,
            &mut u,
            &mut s,
            &mut vt,
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);
        assert!(!u.is_null());
        assert!(!s.is_null());
        assert!(!vt.is_null());

        // U: 2x2, S: 2, Vt: 2x2
        assert_eq!(tfe_tensor_f64_ndim(u as *const _, &mut status), 2);
        assert_eq!(tfe_tensor_f64_ndim(vt as *const _, &mut status), 2);

        // Reconstruct: U * diag(S) * Vt ~= A
        let u_ptr = tfe_tensor_f64_data(u as *const _, &mut status);
        let s_ptr = tfe_tensor_f64_data(s as *const _, &mut status);
        let vt_ptr = tfe_tensor_f64_data(vt as *const _, &mut status);
        let s_len = tfe_tensor_f64_len(s as *const _, &mut status);

        let u_slice = std::slice::from_raw_parts(u_ptr, 4);
        let s_slice = std::slice::from_raw_parts(s_ptr, s_len);
        let vt_slice = std::slice::from_raw_parts(vt_ptr, 4);

        // U * diag(S) * Vt (2x2 matmul)
        let k = s_len;
        let m = 2;
        let n = 2;
        let mut reconstructed = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for kk in 0..k {
                    // U is m x k col-major, Vt is k x n col-major
                    sum += u_slice[i + kk * m] * s_slice[kk] * vt_slice[kk + j * k];
                }
                reconstructed[i + j * m] = sum;
            }
        }

        for i in 0..4 {
            assert!(
                (reconstructed[i] - a_data[i]).abs() < 1e-10,
                "reconstruction mismatch at {}: {} vs {}",
                i,
                reconstructed[i],
                a_data[i]
            );
        }

        tfe_tensor_f64_release(vt);
        tfe_tensor_f64_release(s);
        tfe_tensor_f64_release(u);
        tfe_tensor_f64_release(a);
    }
}

#[test]
fn svd_rrule_null_left_with_nonzero_len_returns_error() {
    unsafe {
        let a_data = [1.0_f64, 3.0, 2.0, 4.0];
        let shape = [2_usize, 2];
        let mut status: tfe_status_t = -999;

        let a = tfe_tensor_f64_from_data(a_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);
        assert_eq!(status, TFE_SUCCESS);

        let right = [1_usize];

        // null left pointer with left_len=1 should fail
        let result = tfe_svd_rrule_f64(
            a as *const _,
            std::ptr::null(),
            1,
            right.as_ptr(),
            1,
            0,
            -1.0,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            &mut status,
        );
        assert_eq!(status, TFE_INVALID_ARGUMENT);
        assert!(result.is_null());

        tfe_tensor_f64_release(a);
    }
}

#[test]
fn svd_rrule_null_right_with_nonzero_len_returns_error() {
    unsafe {
        let a_data = [1.0_f64, 3.0, 2.0, 4.0];
        let shape = [2_usize, 2];
        let mut status: tfe_status_t = -999;

        let a = tfe_tensor_f64_from_data(a_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);
        assert_eq!(status, TFE_SUCCESS);

        let left = [0_usize];

        // null right pointer with right_len=1 should fail
        let result = tfe_svd_rrule_f64(
            a as *const _,
            left.as_ptr(),
            1,
            std::ptr::null(),
            1,
            0,
            -1.0,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            &mut status,
        );
        assert_eq!(status, TFE_INVALID_ARGUMENT);
        assert!(result.is_null());

        tfe_tensor_f64_release(a);
    }
}

#[test]
fn svd_frule_null_left_with_nonzero_len_returns_error() {
    unsafe {
        let a_data = [1.0_f64, 3.0, 2.0, 4.0];
        let shape = [2_usize, 2];
        let mut status: tfe_status_t = -999;

        let a = tfe_tensor_f64_from_data(a_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);
        assert_eq!(status, TFE_SUCCESS);

        let right = [1_usize];
        let mut u: *mut TfeTensorF64 = std::ptr::null_mut();
        let mut s: *mut TfeTensorF64 = std::ptr::null_mut();
        let mut vt: *mut TfeTensorF64 = std::ptr::null_mut();

        // null left pointer with left_len=1 should fail
        tfe_svd_frule_f64(
            a as *const _,
            std::ptr::null(),
            1,
            right.as_ptr(),
            1,
            0,
            -1.0,
            std::ptr::null(),
            &mut u,
            &mut s,
            &mut vt,
            &mut status,
        );
        assert_eq!(status, TFE_INVALID_ARGUMENT);

        tfe_tensor_f64_release(a);
    }
}

#[test]
fn svd_frule_null_right_with_nonzero_len_returns_error() {
    unsafe {
        let a_data = [1.0_f64, 3.0, 2.0, 4.0];
        let shape = [2_usize, 2];
        let mut status: tfe_status_t = -999;

        let a = tfe_tensor_f64_from_data(a_data.as_ptr(), 4, shape.as_ptr(), 2, &mut status);
        assert_eq!(status, TFE_SUCCESS);

        let left = [0_usize];
        let mut u: *mut TfeTensorF64 = std::ptr::null_mut();
        let mut s: *mut TfeTensorF64 = std::ptr::null_mut();
        let mut vt: *mut TfeTensorF64 = std::ptr::null_mut();

        // null right pointer with right_len=1 should fail
        tfe_svd_frule_f64(
            a as *const _,
            left.as_ptr(),
            1,
            std::ptr::null(),
            1,
            0,
            -1.0,
            std::ptr::null(),
            &mut u,
            &mut s,
            &mut vt,
            &mut status,
        );
        assert_eq!(status, TFE_INVALID_ARGUMENT);

        tfe_tensor_f64_release(a);
    }
}

#[test]
fn svd_null_tensor_returns_error() {
    unsafe {
        let left = [0_usize];
        let right = [1_usize];
        let mut u: *mut TfeTensorF64 = std::ptr::null_mut();
        let mut s: *mut TfeTensorF64 = std::ptr::null_mut();
        let mut vt: *mut TfeTensorF64 = std::ptr::null_mut();
        let mut status: tfe_status_t = -999;

        tfe_svd_f64(
            std::ptr::null(),
            left.as_ptr(),
            1,
            right.as_ptr(),
            1,
            0,
            -1.0,
            &mut u,
            &mut s,
            &mut vt,
            &mut status,
        );
        assert_eq!(status, TFE_INVALID_ARGUMENT);
    }
}

#[test]
fn svd_rrule_rank3_permuted_axes_matches_rust_oracle() {
    unsafe {
        let shape = [2_usize, 3, 2];
        let left = [2_usize, 0];
        let right = [1_usize];
        let mut status: tfe_status_t = -999;

        let a_data: Vec<f64> = (0..12).map(|i| (i + 1) as f64).collect();
        let cot_u_data: Vec<f64> = (0..12).map(|i| 0.1 * (i as f64 + 1.0)).collect();
        let cot_s_data = [0.3_f64, -0.2, 0.5];
        let cot_vt_data: Vec<f64> = (0..9).map(|i| -0.05 * (i as f64 + 1.0)).collect();

        let a = tfe_tensor_f64_from_data(
            a_data.as_ptr(),
            a_data.len(),
            shape.as_ptr(),
            3,
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);
        let cot_u_shape = [2_usize, 2, 3]; // [left_dims..., k]
        let cot_vt_shape = [3_usize, 3]; // [k, right_dims...]
        let cot_s_shape = [3_usize];
        let cot_u = tfe_tensor_f64_from_data(
            cot_u_data.as_ptr(),
            cot_u_data.len(),
            cot_u_shape.as_ptr(),
            cot_u_shape.len(),
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);
        let cot_s = tfe_tensor_f64_from_data(
            cot_s_data.as_ptr(),
            cot_s_data.len(),
            cot_s_shape.as_ptr(),
            cot_s_shape.len(),
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);
        let cot_vt = tfe_tensor_f64_from_data(
            cot_vt_data.as_ptr(),
            cot_vt_data.len(),
            cot_vt_shape.as_ptr(),
            cot_vt_shape.len(),
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);

        let grad = tfe_svd_rrule_f64(
            a as *const _,
            left.as_ptr(),
            left.len(),
            right.as_ptr(),
            right.len(),
            0,
            -1.0,
            cot_u as *const _,
            cot_s as *const _,
            cot_vt as *const _,
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);
        assert!(!grad.is_null());

        let (grad_dims, grad_data) = handle_dims_data(grad as *const _, &mut status);
        assert_eq!(grad_dims, shape);

        let t = Tensor::from_slice(&a_data, &shape, MemoryOrder::ColumnMajor).unwrap();
        let (matrix, left_dims, right_dims) = matrixize_for_test(&t, &left, &right);
        let cot_u_public =
            Tensor::from_slice(&cot_u_data, &cot_u_shape, MemoryOrder::ColumnMajor).unwrap();
        let cot_s_public =
            Tensor::from_slice(&cot_s_data, &cot_s_shape, MemoryOrder::ColumnMajor).unwrap();
        let cot_vt_public =
            Tensor::from_slice(&cot_vt_data, &cot_vt_shape, MemoryOrder::ColumnMajor).unwrap();

        let m: usize = left_dims.iter().product();
        let n: usize = right_dims.iter().product();
        let cot_u_mat = cot_u_public.reshape(&[m, cot_s_data.len()]).unwrap();
        let cot_vt_mat = cot_vt_public.reshape(&[cot_s_data.len(), n]).unwrap();
        let cot = SvdCotangent {
            u: Some(cot_u_mat),
            s: Some(cot_s_public),
            vt: Some(cot_vt_mat),
        };
        let mut ctx = CpuTensorLinalgContext::new();
        let grad_matrix = svd_rrule(&mut ctx, &matrix, &cot, None).unwrap();
        let grad_expected =
            unmatrixize_grad_for_test(grad_matrix, &left, &right, &left_dims, &right_dims);
        let expected_data = grad_expected
            .buffer()
            .as_slice()
            .expect("CPU tensor")
            .to_vec();
        approx_eq_slice(&grad_data, &expected_data, 1e-10);

        tfe_tensor_f64_release(grad);
        tfe_tensor_f64_release(cot_vt);
        tfe_tensor_f64_release(cot_s);
        tfe_tensor_f64_release(cot_u);
        tfe_tensor_f64_release(a);
    }
}

#[test]
fn svd_frule_rank3_permuted_axes_matches_rust_oracle() {
    unsafe {
        let shape = [2_usize, 3, 2];
        let left = [2_usize, 0];
        let right = [1_usize];
        let mut status: tfe_status_t = -999;

        let a_data: Vec<f64> = (0..12).map(|i| 0.2 * (i as f64 + 1.0)).collect();
        let da_data: Vec<f64> = (0..12).map(|i| -0.1 * (i as f64 + 1.0)).collect();
        let a = tfe_tensor_f64_from_data(
            a_data.as_ptr(),
            a_data.len(),
            shape.as_ptr(),
            3,
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);
        let da = tfe_tensor_f64_from_data(
            da_data.as_ptr(),
            da_data.len(),
            shape.as_ptr(),
            3,
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);

        let mut du: *mut TfeTensorF64 = std::ptr::null_mut();
        let mut ds: *mut TfeTensorF64 = std::ptr::null_mut();
        let mut dvt: *mut TfeTensorF64 = std::ptr::null_mut();
        tfe_svd_frule_f64(
            a as *const _,
            left.as_ptr(),
            left.len(),
            right.as_ptr(),
            right.len(),
            0,
            -1.0,
            da as *const _,
            &mut du,
            &mut ds,
            &mut dvt,
            &mut status,
        );
        assert_eq!(status, TFE_SUCCESS);
        assert!(!du.is_null());
        assert!(!ds.is_null());
        assert!(!dvt.is_null());

        let (du_dims, du_data) = handle_dims_data(du as *const _, &mut status);
        let (ds_dims, ds_data) = handle_dims_data(ds as *const _, &mut status);
        let (dvt_dims, dvt_data) = handle_dims_data(dvt as *const _, &mut status);
        assert_eq!(du_dims, vec![2, 2, 3]); // [left_dims..., k]
        assert_eq!(ds_dims, vec![3]); // [k]
        assert_eq!(dvt_dims, vec![3, 3]); // [k, right_dims...]

        let t = Tensor::from_slice(&a_data, &shape, MemoryOrder::ColumnMajor).unwrap();
        let dt = Tensor::from_slice(&da_data, &shape, MemoryOrder::ColumnMajor).unwrap();
        let (matrix, left_dims, right_dims) = matrixize_for_test(&t, &left, &right);
        let (tang_matrix, _, _) = matrixize_for_test(&dt, &left, &right);
        let mut ctx = CpuTensorLinalgContext::new();
        let (_primal, tangent_result) = svd_frule(&mut ctx, &matrix, &tang_matrix, None).unwrap();

        let k = tangent_result.s.len();
        let mut du_expected_dims = left_dims.clone();
        du_expected_dims.push(k);
        let du_expected = tangent_result
            .u
            .reshape(&du_expected_dims)
            .unwrap()
            .contiguous(MemoryOrder::ColumnMajor);
        let mut dvt_expected_dims = vec![k];
        dvt_expected_dims.extend_from_slice(&right_dims);
        let dvt_expected = tangent_result
            .vt
            .reshape(&dvt_expected_dims)
            .unwrap()
            .contiguous(MemoryOrder::ColumnMajor);
        let du_expected_data = du_expected.buffer().as_slice().expect("CPU tensor");
        let ds_expected_data = tangent_result.s.buffer().as_slice().expect("CPU tensor");
        let dvt_expected_data = dvt_expected.buffer().as_slice().expect("CPU tensor");

        approx_eq_slice(&du_data, du_expected_data, 1e-10);
        approx_eq_slice(&ds_data, ds_expected_data, 1e-10);
        approx_eq_slice(&dvt_data, dvt_expected_data, 1e-10);

        tfe_tensor_f64_release(dvt);
        tfe_tensor_f64_release(ds);
        tfe_tensor_f64_release(du);
        tfe_tensor_f64_release(da);
        tfe_tensor_f64_release(a);
    }
}
