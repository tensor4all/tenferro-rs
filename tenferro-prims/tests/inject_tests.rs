#![cfg(all(feature = "gemm-blas", feature = "provider-inject"))]

use std::ffi::c_char;
use std::ops::{Add, Mul};
use std::sync::Once;

use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_prims::inject::{register_blas_gemm_fn_ptrs, BlasGemmFnPtrSet};
use tenferro_prims::{CpuBackend, CpuContext, SemiringCoreDescriptor, TensorSemiringCore};
use tenferro_tensor::{MemoryOrder, Tensor};

static REGISTER_ONCE: Once = Once::new();

fn register_test_ptrs_once() {
    REGISTER_ONCE.call_once(|| unsafe {
        register_blas_gemm_fn_ptrs(BlasGemmFnPtrSet {
            sgemm: Some(test_sgemm),
            dgemm: Some(test_dgemm),
            cgemm: Some(test_cgemm),
            zgemm: Some(test_zgemm),
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
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const cblas_inject::blasint,
    b: *const Complex32,
    ldb: *const cblas_inject::blasint,
    beta: *const Complex32,
    c: *mut Complex32,
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
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const cblas_inject::blasint,
    b: *const Complex64,
    ldb: *const cblas_inject::blasint,
    beta: *const Complex64,
    c: *mut Complex64,
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

fn run_batched_gemm<T>(alpha: T, beta: T, a_data: &[T], b_data: &[T], expected: &[T])
where
    T: tenferro_algebra::Scalar + Copy + PartialEq + std::fmt::Debug,
{
    register_test_ptrs_once();

    let mut ctx = CpuContext::new(1);
    let desc = SemiringCoreDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 2,
    };
    let shapes: &[&[usize]] = &[&[2, 2], &[2, 2], &[2, 2]];

    let plan =
        <CpuBackend as TensorSemiringCore<Standard<T>>>::plan(&mut ctx, &desc, shapes).unwrap();

    let a = Tensor::from_slice(a_data, &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let b = Tensor::from_slice(b_data, &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let mut c = Tensor::<T>::zeros(
        &[2, 2],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    <CpuBackend as TensorSemiringCore<Standard<T>>>::execute(
        &mut ctx,
        &plan,
        alpha,
        &[&a, &b],
        beta,
        &mut c,
    )
    .unwrap();

    let out = c.buffer().as_slice().unwrap();
    assert_eq!(&out[..expected.len()], expected);
}

#[test]
fn provider_inject_batched_gemm_f64() {
    let a = [1.0_f64, 3.0, 2.0, 4.0];
    let b = [5.0_f64, 7.0, 6.0, 8.0];
    let expected = [19.0_f64, 43.0, 22.0, 50.0];
    run_batched_gemm(1.0, 0.0, &a, &b, &expected);
}

#[test]
fn provider_inject_batched_gemm_f32() {
    let a = [1.0_f32, 3.0, 2.0, 4.0];
    let b = [5.0_f32, 7.0, 6.0, 8.0];
    let expected = [19.0_f32, 43.0, 22.0, 50.0];
    run_batched_gemm(1.0, 0.0, &a, &b, &expected);
}

#[test]
fn provider_inject_batched_gemm_c64() {
    let a = [
        Complex64::new(1.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    let b = [
        Complex64::new(5.0, 0.0),
        Complex64::new(7.0, 0.0),
        Complex64::new(6.0, 0.0),
        Complex64::new(8.0, 0.0),
    ];
    let expected = [
        Complex64::new(19.0, 0.0),
        Complex64::new(43.0, 0.0),
        Complex64::new(22.0, 0.0),
        Complex64::new(50.0, 0.0),
    ];
    run_batched_gemm(
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        &a,
        &b,
        &expected,
    );
}

#[test]
fn provider_inject_batched_gemm_c32() {
    let a = [
        Complex32::new(1.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(4.0, 0.0),
    ];
    let b = [
        Complex32::new(5.0, 0.0),
        Complex32::new(7.0, 0.0),
        Complex32::new(6.0, 0.0),
        Complex32::new(8.0, 0.0),
    ];
    let expected = [
        Complex32::new(19.0, 0.0),
        Complex32::new(43.0, 0.0),
        Complex32::new(22.0, 0.0),
        Complex32::new(50.0, 0.0),
    ];
    run_batched_gemm(
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 0.0),
        &a,
        &b,
        &expected,
    );
}
