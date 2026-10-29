#![cfg(all(feature = "blas", feature = "provider-inject"))]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use tenferro_cpu::inject::{
    register_blas_gemm_provider_ptrs, register_lapack_provider_ptrs, BlasGemmProviderPtrSet,
    LapackProviderPtrSet, ProviderAbi, ProviderRegistrationError,
};
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::{Tensor, TypedTensor};

static TEST_LOCK: Mutex<()> = Mutex::new(());

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

    let a = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0, 2.0]).unwrap(),
    );

    let mut backend = tenferro_cpu::CpuEngine::serial();
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
