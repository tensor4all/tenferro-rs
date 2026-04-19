#![cfg(all(feature = "cpu-blas", feature = "provider-inject"))]

use std::ffi::c_char;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Once;

use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::inject::{register_blas_gemm_fn_ptrs, BlasGemmFnPtrSet};
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorBackend, TypedTensor};

static REGISTER_ONCE: Once = Once::new();
static DGEMM_CALLS: AtomicUsize = AtomicUsize::new(0);

fn register_test_ptrs_once() {
    REGISTER_ONCE.call_once(|| unsafe {
        register_blas_gemm_fn_ptrs(BlasGemmFnPtrSet {
            dgemm: Some(test_dgemm),
            ..BlasGemmFnPtrSet::new()
        });
    });
}

unsafe extern "C" fn test_dgemm(
    _transa: *const c_char,
    _transb: *const c_char,
    m: *const cblas_inject::blasint,
    n: *const cblas_inject::blasint,
    k: *const cblas_inject::blasint,
    alpha: *const f64,
    a: *const f64,
    lda: *const cblas_inject::blasint,
    b: *const f64,
    ldb: *const cblas_inject::blasint,
    beta: *const f64,
    c: *mut f64,
    ldc: *const cblas_inject::blasint,
) {
    DGEMM_CALLS.fetch_add(1, Ordering::SeqCst);
    let m = unsafe { *m as usize };
    let n = unsafe { *n as usize };
    let k = unsafe { *k as usize };
    let alpha = unsafe { *alpha };
    let beta = unsafe { *beta };
    let lda = unsafe { *lda as usize };
    let ldb = unsafe { *ldb as usize };
    let ldc = unsafe { *ldc as usize };

    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for p in 0..k {
                let av = unsafe { *a.add(i + p * lda) };
                let bv = unsafe { *b.add(p + j * ldb) };
                sum += av * bv;
            }
            let c_ptr = unsafe { c.add(i + j * ldc) };
            unsafe {
                *c_ptr = alpha * sum + beta * *c_ptr;
            }
        }
    }
}

#[test]
fn provider_inject_dot_general_uses_registered_blas() {
    register_test_ptrs_once();
    DGEMM_CALLS.store(0, Ordering::SeqCst);

    let a = Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]));
    let b = Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![5.0, 7.0, 6.0, 8.0]));

    let mut backend = CpuBackend::new();
    let c = backend.dot_general(
        &a,
        &b,
        &DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    );

    assert_eq!(DGEMM_CALLS.load(Ordering::SeqCst), 1);
    match c {
        Ok(Tensor::F64(inner)) => assert_eq!(inner.host_data(), &[19.0, 43.0, 22.0, 50.0]),
        _ => panic!("expected f64 tensor"),
    }
}
