mod organization;

use tenferro_capi::{
    tfe_status_t, tfe_tensor_f64_release, TfeTensorF64, TFE_INTERNAL_ERROR, TFE_SUCCESS,
};
use tenferro_tensor::{MemoryOrder, Tensor};

use super::*;

type TropicalRruleFn = unsafe extern "C" fn(
    *const std::ffi::c_char,
    *const *const TfeTensorF64,
    usize,
    *const TfeTensorF64,
    *mut *mut TfeTensorF64,
    *mut tfe_status_t,
);

fn tensor_handle(data: &[f64], dims: &[usize]) -> *mut TfeTensorF64 {
    tensor_to_handle(Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap())
}

unsafe fn read_tensor(handle: *const TfeTensorF64) -> Vec<f64> {
    handle_to_ref(handle).buffer().as_slice().unwrap().to_vec()
}

fn assert_rrule_prefers_smallest_linear_index(
    rrule: TropicalRruleFn,
    a_data: &[f64],
    b_data: &[f64],
    expected_da: &[f64],
    expected_db: &[f64],
) {
    let a = tensor_handle(a_data, &[2]);
    let b = tensor_handle(b_data, &[2]);
    let cot = tensor_handle(&[1.0], &[]);
    let ops = [a as *const TfeTensorF64, b as *const TfeTensorF64];
    let mut grads = [std::ptr::null_mut(); 2];
    let mut status = TFE_INTERNAL_ERROR;

    unsafe {
        rrule(
            c"i,i->".as_ptr(),
            ops.as_ptr(),
            2,
            cot,
            grads.as_mut_ptr(),
            &mut status,
        )
    };
    assert_eq!(status, TFE_SUCCESS);
    assert_eq!(unsafe { read_tensor(grads[0]) }, expected_da);
    assert_eq!(unsafe { read_tensor(grads[1]) }, expected_db);

    unsafe {
        tfe_tensor_f64_release(grads[0]);
        tfe_tensor_f64_release(grads[1]);
        tfe_tensor_f64_release(cot);
        tfe_tensor_f64_release(b);
        tfe_tensor_f64_release(a);
    }
}

#[test]
fn maxplus_entrypoints_produce_expected_values() {
    let a = tensor_handle(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor_handle(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let da = tensor_handle(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let cot = tensor_handle(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let ops = [a as *const TfeTensorF64, b as *const TfeTensorF64];
    let tangents = [da as *const TfeTensorF64, std::ptr::null()];
    let mut grads = [std::ptr::null_mut(); 2];
    let mut status = TFE_INTERNAL_ERROR;

    let output = unsafe {
        tfe_tropical_einsum_maxplus_f64(c"ij,jk->ik".as_ptr(), ops.as_ptr(), 2, &mut status)
    };
    assert_eq!(status, TFE_SUCCESS);
    assert_eq!(unsafe { read_tensor(output) }, vec![9.0, 10.0, 11.0, 12.0]);

    unsafe {
        tfe_tropical_einsum_rrule_maxplus_f64(
            c"ij,jk->ik".as_ptr(),
            ops.as_ptr(),
            2,
            cot,
            grads.as_mut_ptr(),
            &mut status,
        )
    };
    assert_eq!(status, TFE_SUCCESS);
    assert_eq!(unsafe { read_tensor(grads[0]) }, vec![0.0, 0.0, 2.0, 2.0]);
    assert_eq!(unsafe { read_tensor(grads[1]) }, vec![0.0, 2.0, 0.0, 2.0]);

    let tangent = unsafe {
        tfe_tropical_einsum_frule_maxplus_f64(
            c"ij,jk->ik".as_ptr(),
            ops.as_ptr(),
            2,
            tangents.as_ptr(),
            &mut status,
        )
    };
    assert_eq!(status, TFE_SUCCESS);
    assert_eq!(unsafe { read_tensor(tangent) }, vec![3.0, 4.0, 3.0, 4.0]);

    unsafe {
        tfe_tensor_f64_release(tangent);
        tfe_tensor_f64_release(grads[0]);
        tfe_tensor_f64_release(grads[1]);
        tfe_tensor_f64_release(output);
        tfe_tensor_f64_release(cot);
        tfe_tensor_f64_release(da);
        tfe_tensor_f64_release(b);
        tfe_tensor_f64_release(a);
    }
}

#[test]
fn minplus_entrypoints_produce_expected_values() {
    let a = tensor_handle(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor_handle(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let da = tensor_handle(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let cot = tensor_handle(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let ops = [a as *const TfeTensorF64, b as *const TfeTensorF64];
    let tangents = [da as *const TfeTensorF64, std::ptr::null()];
    let mut grads = [std::ptr::null_mut(); 2];
    let mut status = TFE_INTERNAL_ERROR;

    let output = unsafe {
        tfe_tropical_einsum_minplus_f64(c"ij,jk->ik".as_ptr(), ops.as_ptr(), 2, &mut status)
    };
    assert_eq!(status, TFE_SUCCESS);
    assert_eq!(unsafe { read_tensor(output) }, vec![6.0, 7.0, 8.0, 9.0]);

    unsafe {
        tfe_tropical_einsum_rrule_minplus_f64(
            c"ij,jk->ik".as_ptr(),
            ops.as_ptr(),
            2,
            cot,
            grads.as_mut_ptr(),
            &mut status,
        )
    };
    assert_eq!(status, TFE_SUCCESS);
    assert_eq!(unsafe { read_tensor(grads[0]) }, vec![2.0, 2.0, 0.0, 0.0]);
    assert_eq!(unsafe { read_tensor(grads[1]) }, vec![2.0, 0.0, 2.0, 0.0]);

    let tangent = unsafe {
        tfe_tropical_einsum_frule_minplus_f64(
            c"ij,jk->ik".as_ptr(),
            ops.as_ptr(),
            2,
            tangents.as_ptr(),
            &mut status,
        )
    };
    assert_eq!(status, TFE_SUCCESS);
    assert_eq!(unsafe { read_tensor(tangent) }, vec![1.0, 2.0, 1.0, 2.0]);

    unsafe {
        tfe_tensor_f64_release(tangent);
        tfe_tensor_f64_release(grads[0]);
        tfe_tensor_f64_release(grads[1]);
        tfe_tensor_f64_release(output);
        tfe_tensor_f64_release(cot);
        tfe_tensor_f64_release(da);
        tfe_tensor_f64_release(b);
        tfe_tensor_f64_release(a);
    }
}

#[test]
fn maxmul_entrypoints_produce_expected_values() {
    let a = tensor_handle(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor_handle(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let da = tensor_handle(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let cot = tensor_handle(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
    let ops = [a as *const TfeTensorF64, b as *const TfeTensorF64];
    let tangents = [da as *const TfeTensorF64, std::ptr::null()];
    let mut grads = [std::ptr::null_mut(); 2];
    let mut status = TFE_INTERNAL_ERROR;

    let output = unsafe {
        tfe_tropical_einsum_maxmul_f64(c"ij,jk->ik".as_ptr(), ops.as_ptr(), 2, &mut status)
    };
    assert_eq!(status, TFE_SUCCESS);
    assert_eq!(unsafe { read_tensor(output) }, vec![18.0, 24.0, 24.0, 32.0]);

    unsafe {
        tfe_tropical_einsum_rrule_maxmul_f64(
            c"ij,jk->ik".as_ptr(),
            ops.as_ptr(),
            2,
            cot,
            grads.as_mut_ptr(),
            &mut status,
        )
    };
    assert_eq!(status, TFE_SUCCESS);
    assert_eq!(unsafe { read_tensor(grads[0]) }, vec![0.0, 0.0, 14.0, 14.0]);
    assert_eq!(unsafe { read_tensor(grads[1]) }, vec![0.0, 7.0, 0.0, 7.0]);

    let tangent = unsafe {
        tfe_tropical_einsum_frule_maxmul_f64(
            c"ij,jk->ik".as_ptr(),
            ops.as_ptr(),
            2,
            tangents.as_ptr(),
            &mut status,
        )
    };
    assert_eq!(status, TFE_SUCCESS);
    assert_eq!(
        unsafe { read_tensor(tangent) },
        vec![18.0, 24.0, 24.0, 32.0]
    );

    unsafe {
        tfe_tensor_f64_release(tangent);
        tfe_tensor_f64_release(grads[0]);
        tfe_tensor_f64_release(grads[1]);
        tfe_tensor_f64_release(output);
        tfe_tensor_f64_release(cot);
        tfe_tensor_f64_release(da);
        tfe_tensor_f64_release(b);
        tfe_tensor_f64_release(a);
    }
}

#[test]
fn maxplus_rrule_tie_prefers_smallest_linear_index() {
    assert_rrule_prefers_smallest_linear_index(
        tfe_tropical_einsum_rrule_maxplus_f64,
        &[1.0, 1.0],
        &[2.0, 2.0],
        &[1.0, 0.0],
        &[1.0, 0.0],
    );
}

#[test]
fn minplus_rrule_tie_prefers_smallest_linear_index() {
    assert_rrule_prefers_smallest_linear_index(
        tfe_tropical_einsum_rrule_minplus_f64,
        &[1.0, 1.0],
        &[2.0, 2.0],
        &[1.0, 0.0],
        &[1.0, 0.0],
    );
}

#[test]
fn maxmul_rrule_tie_prefers_smallest_linear_index() {
    assert_rrule_prefers_smallest_linear_index(
        tfe_tropical_einsum_rrule_maxmul_f64,
        &[2.0, 2.0],
        &[3.0, 3.0],
        &[3.0, 0.0],
        &[2.0, 0.0],
    );
}
