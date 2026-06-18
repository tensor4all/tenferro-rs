#![cfg(all(feature = "cpu-blas", feature = "provider-inject"))]

use std::ffi::c_char;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use tenferro_cpu::inject::{
    register_blas_gemm_provider_ptrs, register_lapack_provider_ptrs, BlasGemmProviderPtrSet,
    LapackProviderPtrSet, ProviderAbi, ProviderRegistrationError,
};
use tenferro_cpu::CpuBackend;
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorDot, TypedTensor};

static TEST_LOCK: Mutex<()> = Mutex::new(());

// --- GEMM ILP64 counters and fake ----------------------------------------

static DGEMM_ILP64_CALLS: AtomicUsize = AtomicUsize::new(0);
static DGEMM_ILP64_REGISTERED: AtomicBool = AtomicBool::new(false);

unsafe extern "C" fn test_dgemm_ilp64(
    transa: *const c_char,
    transb: *const c_char,
    m: *const i64,
    n: *const i64,
    k: *const i64,
    alpha: *const f64,
    a: *const f64,
    lda: *const i64,
    b: *const f64,
    ldb: *const i64,
    beta: *const f64,
    c: *mut f64,
    ldc: *const i64,
) {
    DGEMM_ILP64_CALLS.fetch_add(1, Ordering::SeqCst);
    let m = unsafe { *m as usize };
    let n = unsafe { *n as usize };
    let k = unsafe { *k as usize };
    let alpha = unsafe { *alpha };
    let beta = unsafe { *beta };
    let lda = unsafe { *lda as usize };
    let ldb = unsafe { *ldb as usize };
    let ldc = unsafe { *ldc as usize };
    let transa = unsafe { *transa as u8 as char };
    let transb = unsafe { *transb as u8 as char };

    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for p in 0..k {
                let av = match transa {
                    'N' | 'n' => unsafe { *a.add(i + p * lda) },
                    'T' | 't' | 'C' | 'c' => unsafe { *a.add(p + i * lda) },
                    _ => panic!("unexpected transa flag {transa}"),
                };
                let bv = match transb {
                    'N' | 'n' => unsafe { *b.add(p + j * ldb) },
                    'T' | 't' | 'C' | 'c' => unsafe { *b.add(j + p * ldb) },
                    _ => panic!("unexpected transb flag {transb}"),
                };
                sum += av * bv;
            }
            let c_ptr = unsafe { c.add(i + j * ldc) };
            unsafe {
                *c_ptr = alpha * sum + beta * *c_ptr;
            }
        }
    }
}

// --- LAPACK full-pivot LU ILP64 counters and fakes -----------------------

static DGETC2_ILP64_CALLS: AtomicUsize = AtomicUsize::new(0);
static DGESC2_ILP64_CALLS: AtomicUsize = AtomicUsize::new(0);
static DGETC2_ILP64_REGISTERED: AtomicBool = AtomicBool::new(false);
static DGESC2_ILP64_REGISTERED: AtomicBool = AtomicBool::new(false);

unsafe extern "C" fn test_dgetc2_ilp64(
    n: *const i64,
    _a: *mut f64,
    lda: *const i64,
    ipiv: *mut i64,
    jpiv: *mut i64,
    info: *mut i64,
) {
    let _ = lda;
    DGETC2_ILP64_CALLS.fetch_add(1, Ordering::SeqCst);
    let n = unsafe { *n as usize };
    for index in 0..n {
        let one_based = (index + 1) as i64;
        unsafe {
            *ipiv.add(index) = one_based;
            *jpiv.add(index) = one_based;
        }
    }
    unsafe {
        *info = 0;
    }
}

unsafe extern "C" fn test_dgesc2_ilp64(
    _n: *const i64,
    _a: *const f64,
    _lda: *const i64,
    _rhs: *mut f64,
    _ipiv: *const i64,
    _jpiv: *const i64,
    scale: *mut f64,
) {
    DGESC2_ILP64_CALLS.fetch_add(1, Ordering::SeqCst);
    unsafe {
        *scale = 1.0;
    }
}

// --- LAPACK QR ILP64 counters and fakes ----------------------------------

static DGEQRF_ILP64_CALLS: AtomicUsize = AtomicUsize::new(0);
static DORGQR_ILP64_CALLS: AtomicUsize = AtomicUsize::new(0);
static DGEQRF_ILP64_REGISTERED: AtomicBool = AtomicBool::new(false);
static DORGQR_ILP64_REGISTERED: AtomicBool = AtomicBool::new(false);

unsafe extern "C" fn test_dgeqrf_ilp64(
    m: *const i64,
    n: *const i64,
    _a: *mut f64,
    lda: *const i64,
    tau: *mut f64,
    work: *mut f64,
    lwork: *const i64,
    info: *mut i64,
) {
    let _ = lda;
    DGEQRF_ILP64_CALLS.fetch_add(1, Ordering::SeqCst);

    let lwork_val = unsafe { *lwork };
    if lwork_val == -1 {
        unsafe {
            *work = 1.0;
            *info = 0;
        }
        return;
    }

    let m_val = unsafe { *m as usize };
    let n_val = unsafe { *n as usize };
    let k_val = m_val.min(n_val);
    for i in 0..k_val {
        unsafe {
            *tau.add(i) = 0.0;
        }
    }
    unsafe {
        *info = 0;
    }
}

unsafe extern "C" fn test_dorgqr_ilp64(
    _m: *const i64,
    _n: *const i64,
    _k: *const i64,
    _a: *mut f64,
    _lda: *const i64,
    _tau: *const f64,
    work: *mut f64,
    lwork: *const i64,
    info: *mut i64,
) {
    DORGQR_ILP64_CALLS.fetch_add(1, Ordering::SeqCst);

    let lwork_val = unsafe { *lwork };
    if lwork_val == -1 {
        unsafe {
            *work = 1.0;
            *info = 0;
        }
        return;
    }

    unsafe {
        *info = 0;
    }
}

// --- Test: ILP64 GEMM provider reaches LP64 consumer ---------------------

#[test]
fn ilp64_gemm_provider_reaches_lp64_consumer() {
    let _guard = TEST_LOCK.lock().expect("test lock poisoned");

    if !DGEMM_ILP64_REGISTERED.swap(true, Ordering::SeqCst) {
        unsafe {
            register_blas_gemm_provider_ptrs(
                ProviderAbi::Ilp64,
                BlasGemmProviderPtrSet {
                    dgemm: Some(test_dgemm_ilp64 as *const std::ffi::c_void),
                    ..BlasGemmProviderPtrSet::new()
                },
            )
            .expect("dgemm ilp64 registration should succeed");
        }
    }

    DGEMM_ILP64_CALLS.store(0, Ordering::SeqCst);

    let a =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]).unwrap());
    let b =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![5.0, 7.0, 6.0, 8.0]).unwrap());

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

    assert_eq!(DGEMM_ILP64_CALLS.load(Ordering::SeqCst), 1);
    match c {
        Ok(Tensor::F64(inner)) => assert_eq!(inner.host_data().unwrap(), &[19.0, 43.0, 22.0, 50.0]),
        _ => panic!("expected f64 tensor"),
    }
}

// --- Test: ILP64 LAPACK full-pivot LU provider bridges integer arrays ----

#[test]
fn ilp64_lapack_full_piv_lu_bridges_integer_arrays() {
    let _guard = TEST_LOCK.lock().expect("test lock poisoned");

    if !DGETC2_ILP64_REGISTERED.swap(true, Ordering::SeqCst) {
        unsafe {
            register_lapack_provider_ptrs(
                ProviderAbi::Ilp64,
                LapackProviderPtrSet {
                    dgetc2: Some(test_dgetc2_ilp64 as *const std::ffi::c_void),
                    dgesc2: Some(test_dgesc2_ilp64 as *const std::ffi::c_void),
                    ..LapackProviderPtrSet::new()
                },
            )
            .expect("ilp64 full-piv-lu registration should succeed");
        }
        DGESC2_ILP64_REGISTERED.store(true, Ordering::SeqCst);
    }

    DGETC2_ILP64_CALLS.store(0, Ordering::SeqCst);
    DGESC2_ILP64_CALLS.store(0, Ordering::SeqCst);

    let a =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![4.0, 8.0]).unwrap());

    let mut backend = CpuBackend::new();
    let x = backend.full_piv_lu_solve(&a, &b, false);

    assert_eq!(DGETC2_ILP64_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(DGESC2_ILP64_CALLS.load(Ordering::SeqCst), 1);
    match x {
        Ok(Tensor::F64(inner)) => assert_eq!(inner.host_data().unwrap(), &[4.0, 8.0]),
        _ => panic!("expected f64 tensor"),
    }
}

// --- Test: ILP64 LAPACK workspace-query routine (QR) ---------------------

#[test]
fn ilp64_lapack_qr_workspace_query() {
    let _guard = TEST_LOCK.lock().expect("test lock poisoned");

    if !DGEQRF_ILP64_REGISTERED.swap(true, Ordering::SeqCst) {
        unsafe {
            register_lapack_provider_ptrs(
                ProviderAbi::Ilp64,
                LapackProviderPtrSet {
                    dgeqrf: Some(test_dgeqrf_ilp64 as *const std::ffi::c_void),
                    dorgqr: Some(test_dorgqr_ilp64 as *const std::ffi::c_void),
                    ..LapackProviderPtrSet::new()
                },
            )
            .expect("ilp64 qr registration should succeed");
        }
        DORGQR_ILP64_REGISTERED.store(true, Ordering::SeqCst);
    }

    DGEQRF_ILP64_CALLS.store(0, Ordering::SeqCst);
    DORGQR_ILP64_CALLS.store(0, Ordering::SeqCst);

    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0, 2.0]).unwrap());

    let mut backend = CpuBackend::new();
    let result = backend.qr(&a);

    assert!(result.is_ok(), "qr should succeed, got {:?}", result.err());
    assert_eq!(DGEQRF_ILP64_CALLS.load(Ordering::SeqCst), 2);
    assert_eq!(DORGQR_ILP64_CALLS.load(Ordering::SeqCst), 2);
}

// --- Test: null pointer error for BLAS dgemm -----------------------------

#[test]
fn blas_gemm_null_pointer_error() {
    let _guard = TEST_LOCK.lock().expect("test lock poisoned");

    let err = unsafe {
        register_blas_gemm_provider_ptrs(
            ProviderAbi::Ilp64,
            BlasGemmProviderPtrSet {
                dgemm: Some(std::ptr::null()),
                ..BlasGemmProviderPtrSet::new()
            },
        )
    }
    .unwrap_err();

    assert_eq!(
        err,
        ProviderRegistrationError::NullPointer { symbol: "dgemm" }
    );
}

// --- Test: null pointer error for LAPACK dpotrf --------------------------

#[test]
fn lapack_null_pointer_error() {
    let _guard = TEST_LOCK.lock().expect("test lock poisoned");

    let err = unsafe {
        register_lapack_provider_ptrs(
            ProviderAbi::Ilp64,
            LapackProviderPtrSet {
                dpotrf: Some(std::ptr::null()),
                ..LapackProviderPtrSet::new()
            },
        )
    }
    .unwrap_err();

    assert_eq!(
        err,
        ProviderRegistrationError::NullPointer { symbol: "dpotrf" }
    );
}
