#![cfg(all(feature = "linalg-lapack", feature = "provider-inject"))]

use std::ffi::c_char;
use std::ops::{Add, Mul};
use std::sync::Once;

use num_traits::Zero;
use tenferro_linalg::backend::{BlasLapackBackend, LinalgBackend};
use tenferro_linalg::inject::{register_blas_lapack_fn_ptrs, BlasLapackFnPtrSet};

static REGISTER_ONCE: Once = Once::new();

fn register_test_ptrs_once() {
    REGISTER_ONCE.call_once(|| unsafe {
        register_blas_lapack_fn_ptrs(BlasLapackFnPtrSet {
            sgemm: Some(test_sgemm),
            dgemm: Some(test_dgemm),
            cgemm: Some(test_cgemm),
            zgemm: Some(test_zgemm),
            dgesvd: Some(test_dgesvd),
            ..BlasLapackFnPtrSet::new()
        });
    });
}

unsafe fn gemm_no_trans<T>(
    m: usize,
    n: usize,
    k: usize,
    alpha: T,
    a: *const T,
    lda: usize,
    b: *const T,
    ldb: usize,
    beta: T,
    c: *mut T,
    ldc: usize,
) where
    T: Copy + Zero + Add<Output = T> + Mul<Output = T>,
{
    for j in 0..n {
        for i in 0..m {
            let mut sum = T::zero();
            for p in 0..k {
                let av = *a.add(i + p * lda);
                let bv = *b.add(p + j * ldb);
                sum = sum + av * bv;
            }
            let c_ptr = c.add(i + j * ldc);
            *c_ptr = alpha * sum + beta * *c_ptr;
        }
    }
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
    unsafe {
        gemm_no_trans(
            *m as usize,
            *n as usize,
            *k as usize,
            *alpha,
            a,
            *lda as usize,
            b,
            *ldb as usize,
            *beta,
            c,
            *ldc as usize,
        )
    }
}

unsafe extern "C" fn test_sgemm(
    _transa: *const c_char,
    _transb: *const c_char,
    m: *const cblas_inject::blasint,
    n: *const cblas_inject::blasint,
    k: *const cblas_inject::blasint,
    alpha: *const f32,
    a: *const f32,
    lda: *const cblas_inject::blasint,
    b: *const f32,
    ldb: *const cblas_inject::blasint,
    beta: *const f32,
    c: *mut f32,
    ldc: *const cblas_inject::blasint,
) {
    unsafe {
        gemm_no_trans(
            *m as usize,
            *n as usize,
            *k as usize,
            *alpha,
            a,
            *lda as usize,
            b,
            *ldb as usize,
            *beta,
            c,
            *ldc as usize,
        )
    }
}

unsafe extern "C" fn test_cgemm(
    _transa: *const c_char,
    _transb: *const c_char,
    m: *const cblas_inject::blasint,
    n: *const cblas_inject::blasint,
    k: *const cblas_inject::blasint,
    alpha: *const num_complex::Complex32,
    a: *const num_complex::Complex32,
    lda: *const cblas_inject::blasint,
    b: *const num_complex::Complex32,
    ldb: *const cblas_inject::blasint,
    beta: *const num_complex::Complex32,
    c: *mut num_complex::Complex32,
    ldc: *const cblas_inject::blasint,
) {
    unsafe {
        gemm_no_trans(
            *m as usize,
            *n as usize,
            *k as usize,
            *alpha,
            a,
            *lda as usize,
            b,
            *ldb as usize,
            *beta,
            c,
            *ldc as usize,
        )
    }
}

unsafe extern "C" fn test_zgemm(
    _transa: *const c_char,
    _transb: *const c_char,
    m: *const cblas_inject::blasint,
    n: *const cblas_inject::blasint,
    k: *const cblas_inject::blasint,
    alpha: *const num_complex::Complex64,
    a: *const num_complex::Complex64,
    lda: *const cblas_inject::blasint,
    b: *const num_complex::Complex64,
    ldb: *const cblas_inject::blasint,
    beta: *const num_complex::Complex64,
    c: *mut num_complex::Complex64,
    ldc: *const cblas_inject::blasint,
) {
    unsafe {
        gemm_no_trans(
            *m as usize,
            *n as usize,
            *k as usize,
            *alpha,
            a,
            *lda as usize,
            b,
            *ldb as usize,
            *beta,
            c,
            *ldc as usize,
        )
    }
}

unsafe extern "C" fn test_dgesvd(
    _jobu: *const c_char,
    _jobvt: *const c_char,
    m: *const lapack_inject::lapackint,
    n: *const lapack_inject::lapackint,
    _a: *mut f64,
    _lda: *const lapack_inject::lapackint,
    s: *mut f64,
    u: *mut f64,
    ldu: *const lapack_inject::lapackint,
    vt: *mut f64,
    ldvt: *const lapack_inject::lapackint,
    work: *mut f64,
    lwork: *const lapack_inject::lapackint,
    info: *mut lapack_inject::lapackint,
) {
    unsafe {
        *info = 0;
        if *lwork == -1 {
            *work = 1.0;
            return;
        }

        let m_usize = *m as usize;
        let n_usize = *n as usize;
        let k_usize = m_usize.min(n_usize);
        let ldu_usize = *ldu as usize;
        let ldvt_usize = *ldvt as usize;

        for i in 0..k_usize {
            *s.add(i) = 1.0;
        }

        for j in 0..k_usize {
            for i in 0..m_usize {
                *u.add(i + j * ldu_usize) = if i == j { 1.0 } else { 0.0 };
            }
        }

        for j in 0..n_usize {
            for i in 0..k_usize {
                *vt.add(i + j * ldvt_usize) = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
}

#[test]
fn provider_inject_mat_mul_f64() {
    register_test_ptrs_once();

    let mut backend = BlasLapackBackend::new();
    let a = [1.0_f64, 3.0, 2.0, 4.0];
    let b = [5.0_f64, 7.0, 6.0, 8.0];
    let mut c = [0.0_f64; 4];

    backend.mat_mul(&a, 2, 2, &b, 2, &mut c).unwrap();

    assert_eq!(c, [19.0, 43.0, 22.0, 50.0]);
}

#[test]
fn provider_inject_thin_svd_f64() {
    register_test_ptrs_once();

    let mut backend = BlasLapackBackend::new();
    let a = [1.0_f64, 0.0, 0.0, 1.0];
    let mut u = [0.0_f64; 4];
    let mut s = [0.0_f64; 2];
    let mut vt = [0.0_f64; 4];

    backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).unwrap();

    assert_eq!(s, [1.0, 1.0]);
    assert_eq!(u, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(vt, [1.0, 0.0, 0.0, 1.0]);
}
