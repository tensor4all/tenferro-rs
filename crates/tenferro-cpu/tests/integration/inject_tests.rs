#![cfg(all(feature = "cpu-blas", feature = "provider-inject"))]

use std::ffi::{c_char, c_void};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once};

use tenferro_cpu::inject::{
    register_blas_gemm_provider_ptrs, register_lapack_provider_ptrs, BlasGemmProviderPtrSet,
    LapackProviderPtrSet, ProviderAbi,
};
use tenferro_cpu::{
    discover_cpu_topology, CpuBackend, CpuBackendError, CpuBackendKind, CpuDomainExecutor,
    CpuDomainExecutorCapabilities, CpuDomainExecutorError, CpuExecutorAffinity,
    CpuExecutorReentrancy, CpuExecutorShutdown, CpuInnerParallelism, CpuPlacementGuarantee,
    CpuProviderBundle, CpuProviderBundleInstallError, CpuProviderDomainError, CpuProviderSlot,
    ExternalCpuDomain, ResolvedCpuPlacement, ScopedCpuJob, ScopedCpuJobs,
};
use tenferro_tensor::{CpuDomainId, DotGeneralConfig, Tensor, TensorDot, TypedTensor};

static REGISTER_ONCE: Once = Once::new();
static TEST_LOCK: Mutex<()> = Mutex::new(());
static DGEMM_CALLS: AtomicUsize = AtomicUsize::new(0);
static DGETC2_CALLS: AtomicUsize = AtomicUsize::new(0);
static DGESC2_CALLS: AtomicUsize = AtomicUsize::new(0);
static DGETRF_CALLS: AtomicUsize = AtomicUsize::new(0);
static DGETRS_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct InlineExternalExecutor;

impl CpuDomainExecutor for InlineExternalExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        CpuDomainExecutorCapabilities {
            worker_count: NonZeroUsize::new(1).unwrap(),
            outer_parallelism: false,
            inner_parallelism: CpuInnerParallelism::None,
            reentrancy: CpuExecutorReentrancy::Rejected,
            affinity: CpuExecutorAffinity::CallerDeclaredUnverified,
            shutdown: CpuExecutorShutdown::CallerOwned,
        }
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        for index in 0..jobs.len() {
            jobs.run(index)?;
        }
        Ok(())
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        job.run()
    }
}

fn all_allowed_external_domain(id: CpuDomainId) -> ExternalCpuDomain {
    let topology = discover_cpu_topology().expect("test host should expose CPU topology");
    ExternalCpuDomain::new(
        id,
        ResolvedCpuPlacement::AllAllowed {
            cpus: topology.allowed_cpus().clone(),
        },
        Arc::new(InlineExternalExecutor),
        NonZeroUsize::new(1).unwrap(),
        CpuPlacementGuarantee::ExactDeclared,
    )
    .unwrap()
}

fn register_test_ptrs_once() {
    REGISTER_ONCE.call_once(|| unsafe {
        register_blas_gemm_provider_ptrs(
            ProviderAbi::Lp64,
            BlasGemmProviderPtrSet {
                dgemm: Some(test_dgemm as *const c_void),
                ..BlasGemmProviderPtrSet::new()
            },
        )
        .expect("test dgemm registration should succeed");
        register_lapack_provider_ptrs(
            ProviderAbi::Lp64,
            LapackProviderPtrSet {
                dgetc2: Some(test_dgetc2 as *const c_void),
                dgesc2: Some(test_dgesc2 as *const c_void),
                dgetrf: Some(test_dgetrf as *const c_void),
                dgetrs: Some(test_dgetrs as *const c_void),
                ..LapackProviderPtrSet::new()
            },
        )
        .expect("test LAPACK registration should succeed");
    });
}

unsafe extern "C" fn test_dgemm(
    transa: *const c_char,
    transb: *const c_char,
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
    let transa = unsafe { *transa as u8 as char };
    let transb = unsafe { *transb as u8 as char };

    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for p in 0..k {
                let av = match transa {
                    'N' | 'n' => unsafe { *a.add(i + p * lda) },
                    'T' | 't' | 'C' | 'c' => unsafe { *a.add(p + i * lda) },
                    _ => return,
                };
                let bv = match transb {
                    'N' | 'n' => unsafe { *b.add(p + j * ldb) },
                    'T' | 't' | 'C' | 'c' => unsafe { *b.add(j + p * ldb) },
                    _ => return,
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

unsafe extern "C" fn test_dgetc2(
    n: *const lapack_inject::lapackint,
    _a: *mut f64,
    _lda: *const lapack_inject::lapackint,
    ipiv: *mut lapack_inject::lapackint,
    jpiv: *mut lapack_inject::lapackint,
    info: *mut lapack_inject::lapackint,
) {
    DGETC2_CALLS.fetch_add(1, Ordering::SeqCst);
    let n = unsafe { *n as usize };
    for index in 0..n {
        let one_based = (index + 1) as lapack_inject::lapackint;
        unsafe {
            *ipiv.add(index) = one_based;
            *jpiv.add(index) = one_based;
        }
    }
    unsafe {
        *info = 0;
    }
}

unsafe extern "C" fn test_dgesc2(
    _n: *const lapack_inject::lapackint,
    _a: *const f64,
    _lda: *const lapack_inject::lapackint,
    _rhs: *mut f64,
    _ipiv: *const lapack_inject::lapackint,
    _jpiv: *const lapack_inject::lapackint,
    scale: *mut f64,
) {
    DGESC2_CALLS.fetch_add(1, Ordering::SeqCst);
    unsafe {
        *scale = 1.0;
    }
}

unsafe extern "C" fn test_dgetrf(
    m: *const lapack_inject::lapackint,
    n: *const lapack_inject::lapackint,
    _a: *mut f64,
    _lda: *const lapack_inject::lapackint,
    ipiv: *mut lapack_inject::lapackint,
    info: *mut lapack_inject::lapackint,
) {
    DGETRF_CALLS.fetch_add(1, Ordering::SeqCst);
    let k = unsafe { (*m).min(*n) as usize };
    for index in 0..k {
        unsafe {
            *ipiv.add(index) = (index + 1) as lapack_inject::lapackint;
        }
    }
    unsafe {
        *info = 0;
    }
}

unsafe extern "C" fn test_dgetrs(
    _trans: *const c_char,
    _n: *const lapack_inject::lapackint,
    _nrhs: *const lapack_inject::lapackint,
    _a: *const f64,
    _lda: *const lapack_inject::lapackint,
    _ipiv: *const lapack_inject::lapackint,
    _b: *mut f64,
    _ldb: *const lapack_inject::lapackint,
    info: *mut lapack_inject::lapackint,
) {
    DGETRS_CALLS.fetch_add(1, Ordering::SeqCst);
    unsafe {
        *info = 0;
    }
}

#[test]
fn external_managed_constructor_validates_blas_or_custom_bundle_atomically() {
    let standard_error = CpuBackend::from_external_managed_domains(
        CpuDomainId::new(40),
        [all_allowed_external_domain(CpuDomainId::new(40))],
    )
    .unwrap_err();
    let CpuBackendError::Tensor(tensor_error) = &standard_error else {
        panic!("uncontrolled standard BLAS should retain the tensor error wrapper");
    };
    let install_error = std::error::Error::source(tensor_error)
        .and_then(|source| source.downcast_ref::<CpuProviderBundleInstallError>())
        .expect("standard BLAS rejection should retain the typed install source");
    assert!(matches!(
        install_error,
        CpuProviderBundleInstallError::IncompatibleDomain {
            domain_id,
            provider: CpuProviderSlot::Gemm,
            source: CpuProviderDomainError::ThreadCountNotEnforceable { .. },
        } if *domain_id == CpuDomainId::new(40)
    ));

    // This fixture uses the native provider only as a controlled capability
    // descriptor. Applications can supply their own provider implementations
    // through the same public constructor even when BLAS is the compiled kind.
    let controlled = CpuProviderBundle::builder(CpuBackendKind::Faer)
        .build()
        .unwrap();
    let backend = CpuBackend::from_external_managed_domains_with_provider_bundle(
        CpuDomainId::new(41),
        [all_allowed_external_domain(CpuDomainId::new(41))],
        controlled.clone(),
    )
    .unwrap();
    assert!(backend.provider_bundle().shares_identity_with(&controlled));
}

#[test]
fn provider_inject_dot_general_uses_registered_blas() {
    let _guard = TEST_LOCK
        .lock()
        .expect("provider-inject test lock poisoned");
    register_test_ptrs_once();
    DGEMM_CALLS.store(0, Ordering::SeqCst);

    let a =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]).unwrap());
    let b =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![5.0, 7.0, 6.0, 8.0]).unwrap());

    let mut backend = CpuBackend::with_kind(CpuBackendKind::Blas).unwrap();
    assert_eq!(backend.kind(), CpuBackendKind::Blas);
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
        Ok(Tensor::F64(inner)) => assert_eq!(inner.host_data().unwrap(), &[19.0, 43.0, 22.0, 50.0]),
        _ => panic!("expected f64 tensor"),
    }
}

#[test]
fn provider_inject_dot_general_singleton_contract_uses_registered_blas() {
    let _guard = TEST_LOCK
        .lock()
        .expect("provider-inject test lock poisoned");
    register_test_ptrs_once();
    DGEMM_CALLS.store(0, Ordering::SeqCst);

    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 2], vec![1.0, 2.0]).unwrap());
    let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 2], vec![3.0, 4.0]).unwrap());

    let mut backend = CpuBackend::with_kind(CpuBackendKind::Blas).unwrap();
    let c = backend.dot_general(
        &a,
        &b,
        &DotGeneralConfig {
            lhs_contracting_dims: vec![0],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    );

    assert_eq!(DGEMM_CALLS.load(Ordering::SeqCst), 1);
    match c {
        Ok(Tensor::F64(inner)) => assert_eq!(inner.host_data().unwrap(), &[3.0, 6.0, 4.0, 8.0]),
        _ => panic!("expected f64 tensor"),
    }
}

#[test]
fn provider_inject_dot_general_rhs_singleton_contract_uses_registered_blas() {
    let _guard = TEST_LOCK
        .lock()
        .expect("provider-inject test lock poisoned");
    register_test_ptrs_once();
    DGEMM_CALLS.store(0, Ordering::SeqCst);

    let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![1], vec![2.0]).unwrap());
    let b = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 1, 2], vec![3.0, 4.0, 5.0, 6.0]).unwrap(),
    );

    let mut backend = CpuBackend::with_kind(CpuBackendKind::Blas).unwrap();
    let c = backend.dot_general(
        &a,
        &b,
        &DotGeneralConfig {
            lhs_contracting_dims: vec![0],
            rhs_contracting_dims: vec![1],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    );

    assert_eq!(DGEMM_CALLS.load(Ordering::SeqCst), 1);
    match c {
        Ok(Tensor::F64(inner)) => assert_eq!(inner.host_data().unwrap(), &[6.0, 8.0, 10.0, 12.0]),
        _ => panic!("expected f64 tensor"),
    }
}
