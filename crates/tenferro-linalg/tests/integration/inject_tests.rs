#![cfg(all(feature = "blas", feature = "provider-inject"))]

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, Once};

use tenferro_cpu::inject::{
    register_blas_gemm_provider_ptrs, register_lapack_provider_ptrs, BlasGemmProviderPtrSet,
    LapackProviderPtrSet, ProviderAbi,
};
use tenferro_linalg::{LinalgBackend, QrOptions, RankRevealingQrOptions};
use tenferro_tensor::{Tensor, TypedTensor};

static REGISTER_ONCE: Once = Once::new();
static TEST_LOCK: Mutex<()> = Mutex::new(());
static DGEMM_CALLS: AtomicUsize = AtomicUsize::new(0);
static DGETC2_CALLS: AtomicUsize = AtomicUsize::new(0);
static DGESC2_CALLS: AtomicUsize = AtomicUsize::new(0);
static DGETRF_CALLS: AtomicUsize = AtomicUsize::new(0);
/// The CPU set of the thread that entered the registered `dgetrf` last (`None` = not entered or
/// unreadable).
static DGETRF_CALLER_CPUS: Mutex<Option<Vec<usize>>> = Mutex::new(None);
static DGETRS_CALLS: AtomicUsize = AtomicUsize::new(0);
static DGESVD_CALLS: AtomicUsize = AtomicUsize::new(0);
static DSYEVD_CALLS: AtomicUsize = AtomicUsize::new(0);
static DGEQP3_CALLS: AtomicUsize = AtomicUsize::new(0);
static DORGQR_CALLS: AtomicUsize = AtomicUsize::new(0);
static DORMQR_CALLS: AtomicUsize = AtomicUsize::new(0);

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
                dgesvd: Some(test_dgesvd as *const c_void),
                dsyevd: Some(test_dsyevd as *const c_void),
                dgeqp3: Some(test_dgeqp3 as *const c_void),
                dorgqr: Some(test_dorgqr as *const c_void),
                dormqr: Some(test_dormqr as *const c_void),
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
    *DGETRF_CALLER_CPUS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = allowed_cpus();
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
    DGESVD_CALLS.fetch_add(1, Ordering::SeqCst);
    if unsafe { *lwork } == -1 {
        unsafe {
            *work = 16.0;
            *info = 0;
        }
        return;
    }

    let m = unsafe { *m as usize };
    let n = unsafe { *n as usize };
    let k = m.min(n);
    for index in 0..k {
        unsafe {
            *s.add(index) = if index == 0 { 3.0 } else { 2.0 };
        }
    }

    let ldu = unsafe { *ldu as usize };
    for col in 0..k {
        for row in 0..m {
            unsafe {
                *u.add(row + col * ldu) = if row == col { 1.0 } else { 0.0 };
            }
        }
    }

    let ldvt = unsafe { *ldvt as usize };
    for col in 0..n {
        for row in 0..k {
            unsafe {
                *vt.add(row + col * ldvt) = if row == col { 1.0 } else { 0.0 };
            }
        }
    }

    unsafe {
        *info = 0;
    }
}

/// The calling thread's affinity set, from `Cpus_allowed_list` in `/proc/thread-self/status`.
///
/// `None` where that file is unavailable or unparsable.
fn allowed_cpus() -> Option<Vec<usize>> {
    let status = std::fs::read_to_string("/proc/thread-self/status").ok()?;
    let list = status
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?
        .trim();
    let mut cpus = Vec::new();
    for range in list.split(',') {
        let (lo, hi) = range.split_once('-').unwrap_or((range, range));
        cpus.extend(lo.parse::<usize>().ok()?..=hi.parse::<usize>().ok()?);
    }
    Some(cpus)
}

/// Opt-out for hosts that cannot place a two-member team. A skip is never silent: without this
/// variable such a host fails the test, so a green result means the widening was observed, unless
/// it was explicitly skipped (`TENFERRO_SKIP_AFFINITY_WIDENING_TEST=1`).
const SKIP_AFFINITY_TEST: &str = "TENFERRO_SKIP_AFFINITY_WIDENING_TEST";

// #2553: threadteam 0.5 pins the thread that builds a `Team` (the engine's leader) to `{S[0]}`
// for the team's life. A vendor library creates its own thread team from the calling thread, so a
// LAPACK call made from that one-CPU leader confined MKL's whole OpenMP team to one CPU (an 11-89x
// collapse at 4 threads). Every LAPACK call must run with the leader widened back to `S`.
// Deterministic: it reads the caller's CPU set inside the registered `dgetrf`, no timing, and
// asserts the leader's transition `{S[0]}` -> `S` (inside the vendor call) -> `{S[0]}`.
//
// Linux only: threadteam pins only on Linux, so elsewhere the leader is never narrowed and there is
// no widening to observe.
#[cfg(target_os = "linux")]
#[test]
fn provider_inject_lapack_call_runs_on_the_team_leaders_full_cpu_set() {
    // Observe under the lock, assert after releasing it, so a failure here cannot poison the
    // lock for the other provider-inject tests.
    let observed = {
        let _guard = TEST_LOCK
            .lock()
            .expect("provider-inject test lock poisoned");
        register_test_ptrs_once();
        let wide = allowed_cpus().expect("Linux exposes /proc/thread-self/status");
        if wide.len() < 2 {
            let reason = format!(
                "the caller's affinity set {wide:?} has fewer than 2 CPUs, so threadteam places no \
                 team and the vendor-call widening cannot be exercised"
            );
            if std::env::var_os(SKIP_AFFINITY_TEST).is_some() {
                eprintln!("SKIPPED, widening NOT verified: {reason} ({SKIP_AFFINITY_TEST} is set)");
                return;
            }
            drop(_guard);
            panic!(
                "cannot verify #2553: {reason}. Run on 2+ CPUs, or set {SKIP_AFFINITY_TEST}=1 to \
                 skip explicitly"
            );
        }
        // The thread that builds the engine's team is its placed leader, so build it here.
        let pool = tenferro_cpu::Pool::builder()
            .threads(2)
            .build()
            .expect("two-worker pool");
        let mut backend = tenferro_cpu::CpuEngine::new(pool);
        let has_team = backend.has_team();
        let pinned_before = allowed_cpus();

        let a = Tensor::from_typed::<f64>(
            TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        );
        let b = Tensor::from_typed::<f64>(
            TypedTensor::from_vec_col_major(vec![2, 1], vec![4.0, 8.0]).unwrap(),
        );
        DGETRF_CALLS.store(0, Ordering::SeqCst);
        *DGETRF_CALLER_CPUS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        let solved = backend.solve(&a, &b).is_ok();
        let calls = DGETRF_CALLS.load(Ordering::SeqCst);
        let inside = DGETRF_CALLER_CPUS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        let pinned_after = allowed_cpus();
        drop(backend);
        let released = allowed_cpus();
        (
            wide,
            has_team,
            pinned_before,
            solved,
            calls,
            inside,
            pinned_after,
            released,
        )
    };
    let (wide, has_team, pinned_before, solved, calls, inside, pinned_after, released) = observed;
    eprintln!(
        "affinity: S = {wide:?}; leader before = {pinned_before:?}; inside dgetrf = {inside:?}; \
         after = {pinned_after:?}; engine dropped = {released:?}"
    );

    assert!(has_team, "a two-worker engine holds a team");
    let home = vec![wide[0]];
    assert_eq!(
        pinned_before.as_ref(),
        Some(&home),
        "precondition: the team's creator is pinned to S[0] between operations"
    );
    assert!(solved, "solve through the registered provider");
    assert_eq!(calls, 1, "the registered dgetrf ran exactly once");
    assert_eq!(
        inside.as_ref(),
        Some(&wide),
        "the LAPACK call must run on the leader's full CPU set S, not on S[0] alone"
    );
    assert_eq!(
        pinned_after.as_ref(),
        Some(&home),
        "the leader is pinned to S[0] again once the vendor call returns"
    );
    assert_eq!(
        released.as_ref(),
        Some(&wide),
        "dropping the team gives the mask back"
    );
}

// Complete-pivoting LU runs on the shared tlinalg kernel in every build, so a registered
// `?getc2`/`?gesc2` is never called.
#[test]
fn provider_inject_full_piv_lu_solve_uses_the_shared_kernel() {
    let _guard = TEST_LOCK
        .lock()
        .expect("provider-inject test lock poisoned");
    register_test_ptrs_once();
    DGETC2_CALLS.store(0, Ordering::SeqCst);
    DGESC2_CALLS.store(0, Ordering::SeqCst);

    let a = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
    );
    let b = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2, 1], vec![4.0, 8.0]).unwrap(),
    );

    let mut backend = tenferro_cpu::CpuEngine::serial();
    let x = backend.full_piv_lu_solve(&a, &b, false);

    assert_eq!(DGETC2_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(DGESC2_CALLS.load(Ordering::SeqCst), 0);
    match x {
        Ok(tensor) => assert_eq!(
            tensor
                .into_typed::<f64>()
                .expect("expected f64 tensor")
                .host_data()
                .unwrap(),
            &[4.0, 8.0]
        ),
        _ => panic!("expected f64 tensor"),
    }
}

#[test]
fn provider_inject_solve_uses_registered_lapack_getrf_getrs() {
    let _guard = TEST_LOCK
        .lock()
        .expect("provider-inject test lock poisoned");
    register_test_ptrs_once();
    DGETRF_CALLS.store(0, Ordering::SeqCst);
    DGETRS_CALLS.store(0, Ordering::SeqCst);

    let a = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
    );
    let b = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2, 1], vec![4.0, 8.0]).unwrap(),
    );

    let mut backend = tenferro_cpu::CpuEngine::serial();
    let x = backend.solve(&a, &b);

    assert_eq!(DGETRF_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(DGETRS_CALLS.load(Ordering::SeqCst), 1);
    match x {
        Ok(tensor) => assert_eq!(
            tensor
                .into_typed::<f64>()
                .expect("expected f64 tensor")
                .host_data()
                .unwrap(),
            &[4.0, 8.0]
        ),
        _ => panic!("expected f64 tensor"),
    }
}

#[test]
fn provider_inject_svd_uses_registered_lapack_gesvd() {
    let _guard = TEST_LOCK
        .lock()
        .expect("provider-inject test lock poisoned");
    register_test_ptrs_once();
    DGESVD_CALLS.store(0, Ordering::SeqCst);

    let a = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2, 2], vec![3.0, 0.0, 0.0, 2.0]).unwrap(),
    );

    let mut backend = tenferro_cpu::CpuEngine::serial();
    let outputs = backend.svd(&a).expect("provider-inject SVD should run");

    assert_eq!(DGESVD_CALLS.load(Ordering::SeqCst), 2);
    assert_eq!(outputs.len(), 3);
    assert_eq!(
        outputs[1]
            .as_typed::<f64>()
            .expect("expected f64 singular values")
            .host_data()
            .unwrap(),
        &[3.0, 2.0]
    );
}

type Lint = lapack_inject::lapackint;

/// `?syevd` for a diagonal input with ascending diagonal: eigenvalues are the diagonal and
/// the eigenvectors the identity. A query (`lwork` or `liwork` of -1) returns size 1.
unsafe extern "C" fn test_dsyevd(
    _jobz: *const c_char,
    _uplo: *const c_char,
    n: *const Lint,
    a: *mut f64,
    lda: *const Lint,
    w: *mut f64,
    work: *mut f64,
    lwork: *const Lint,
    iwork: *mut Lint,
    liwork: *const Lint,
    info: *mut Lint,
) {
    unsafe {
        *info = 0;
        if *lwork == -1 || *liwork == -1 {
            *work = 1.0;
            *iwork = 1;
            return;
        }
        DSYEVD_CALLS.fetch_add(1, Ordering::SeqCst);
        let (n, lda) = (*n as usize, *lda as usize);
        for j in 0..n {
            *w.add(j) = *a.add(j + j * lda);
            for i in 0..n {
                *a.add(i + j * lda) = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
}

/// `?geqp3` for an upper-triangular input whose columns are already in decreasing-norm
/// order: identity permutation, zero reflectors (`tau = 0`), `R` is the input.
unsafe extern "C" fn test_dgeqp3(
    _m: *const Lint,
    n: *const Lint,
    _a: *mut f64,
    _lda: *const Lint,
    jpvt: *mut Lint,
    tau: *mut f64,
    work: *mut f64,
    lwork: *const Lint,
    info: *mut Lint,
) {
    unsafe {
        *info = 0;
        if *lwork == -1 {
            *work = 1.0;
            return;
        }
        DGEQP3_CALLS.fetch_add(1, Ordering::SeqCst);
        for j in 0..*n as usize {
            *jpvt.add(j) = (j + 1) as Lint;
            *tau.add(j) = 0.0;
        }
    }
}

/// `?orgqr` with zero reflectors: `Q` is the leading identity columns.
unsafe extern "C" fn test_dorgqr(
    m: *const Lint,
    n: *const Lint,
    _k: *const Lint,
    a: *mut f64,
    lda: *const Lint,
    _tau: *const f64,
    work: *mut f64,
    lwork: *const Lint,
    info: *mut Lint,
) {
    unsafe {
        *info = 0;
        if *lwork == -1 {
            *work = 1.0;
            return;
        }
        DORGQR_CALLS.fetch_add(1, Ordering::SeqCst);
        let (m, n, lda) = (*m as usize, *n as usize, *lda as usize);
        for j in 0..n {
            for i in 0..m {
                *a.add(i + j * lda) = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
}

/// `?ormqr` with zero reflectors: every `H(i)` is the identity, so `C` is unchanged.
unsafe extern "C" fn test_dormqr(
    _side: *const c_char,
    _trans: *const c_char,
    _m: *const Lint,
    _n: *const Lint,
    _k: *const Lint,
    _a: *const f64,
    _lda: *const Lint,
    _tau: *const f64,
    _c: *mut f64,
    _ldc: *const Lint,
    work: *mut f64,
    lwork: *const Lint,
    info: *mut Lint,
) {
    unsafe {
        *info = 0;
        if *lwork == -1 {
            *work = 1.0;
            return;
        }
        DORMQR_CALLS.fetch_add(1, Ordering::SeqCst);
    }
}

fn f64_matrix(rows: usize, cols: usize, col_major: Vec<f64>) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![rows, cols], col_major).unwrap())
}

fn f64_data(tensor: &Tensor) -> Vec<f64> {
    tensor
        .as_typed::<f64>()
        .expect("expected f64 tensor")
        .host_data()
        .unwrap()
        .to_vec()
}

#[test]
fn provider_inject_real_eigh_uses_registered_lapack_syevd() {
    let _guard = TEST_LOCK
        .lock()
        .expect("provider-inject test lock poisoned");
    register_test_ptrs_once();
    DSYEVD_CALLS.store(0, Ordering::SeqCst);

    let a = f64_matrix(2, 2, vec![1.0, 0.0, 0.0, 3.0]);
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let outputs = backend.eigh(&a).expect("provider-inject eigh should run");

    assert_eq!(DSYEVD_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(f64_data(&outputs[0]), [1.0, 3.0]);
    assert_eq!(f64_data(&outputs[1]), [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn provider_inject_rank_revealing_qr_uses_registered_lapack_geqp3() {
    let _guard = TEST_LOCK
        .lock()
        .expect("provider-inject test lock poisoned");
    register_test_ptrs_once();
    DGEQP3_CALLS.store(0, Ordering::SeqCst);
    DORGQR_CALLS.store(0, Ordering::SeqCst);

    let a = f64_matrix(2, 2, vec![3.0, 0.0, 0.0, 1.0]);
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let outputs = backend
        .rank_revealing_qr(&a, RankRevealingQrOptions::default())
        .expect("provider-inject rank-revealing QR should run");

    assert_eq!(DGEQP3_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(DORGQR_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(f64_data(&outputs[0]), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(f64_data(&outputs[1]), [3.0, 0.0, 0.0, 1.0]);
}

#[test]
fn provider_inject_householder_q_columns_use_registered_lapack_ormqr() {
    let _guard = TEST_LOCK
        .lock()
        .expect("provider-inject test lock poisoned");
    register_test_ptrs_once();
    DORMQR_CALLS.store(0, Ordering::SeqCst);

    // Zero Householder coefficients: Q is the identity.
    let packed = f64_matrix(2, 2, vec![3.0, 0.0, 0.0, 1.0]);
    let coeff = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![0.0, 0.0]).unwrap(),
    );
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let q = backend
        .householder_qr_q_columns(&packed, &coeff, 0..2, QrOptions::default())
        .expect("provider-inject Householder Q columns should run");

    assert_eq!(DORMQR_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(f64_data(&q), [1.0, 0.0, 0.0, 1.0]);
}
