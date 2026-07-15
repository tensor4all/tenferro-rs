use super::*;

use tenferro_tensor::{ErrorKind, ValidationKind};

#[test]
fn cpu_context_from_env_respects_rayon_num_threads() {
    with_rayon_num_threads(Some("3"), || {
        let ctx = CpuContext::from_env();
        assert_eq!(ctx.num_threads(), 3);
    });
}

#[test]
fn cpu_context_from_env_falls_back_to_affinity_when_rayon_num_threads_is_absent() {
    with_rayon_num_threads(None, || {
        let ctx = CpuContext::from_env();
        assert_eq!(ctx.num_threads(), crate::available_parallelism());
    });
}

#[test]
fn cpu_context_from_env_falls_back_to_single_threaded_when_rayon_num_threads_is_invalid() {
    with_rayon_num_threads(Some("not-a-number"), || {
        let ctx = CpuContext::from_env();
        assert_eq!(ctx.num_threads(), 1);
    });
}

#[test]
fn cpu_context_try_from_env_rejects_invalid_rayon_num_threads() {
    with_rayon_num_threads(Some("not-a-number"), || {
        assert!(CpuContext::try_from_env().is_err());
    });
}

#[cfg(unix)]
#[test]
fn cpu_context_try_from_env_rejects_non_unicode_rayon_num_threads() {
    use std::os::unix::ffi::OsStringExt;

    let _guard = RayonNumThreadsEnvGuard::new(None);
    std::env::set_var("RAYON_NUM_THREADS", OsString::from_vec(vec![0xff]));

    let err = CpuContext::try_from_env().unwrap_err();

    assert!(matches!(
        &err,
        Error::Extension {
            op: "CpuContext::try_from_env",
            family: "cpu",
            kind: ErrorKind::Validation(ValidationKind::InvalidArgument),
            ..
        }
    ));
    assert!(std::error::Error::source(&err).is_some());
}

#[test]
fn cpu_context_try_from_env_rejects_zero_rayon_num_threads() {
    with_rayon_num_threads(Some("0"), || {
        let err = match CpuContext::try_from_env() {
            Ok(_) => panic!("expected zero RAYON_NUM_THREADS to be rejected"),
            Err(err) => err,
        };
        assert!(format!("{err}").contains("CpuContext::try_from_env"));
        assert!(format!("{err}").contains("thread count must be at least 1"));
    });
}

#[test]
fn cpu_context_with_threads_zero_returns_error() {
    assert!(CpuContext::with_threads(0).is_err());
}

#[test]
fn cpu_backend_new_matches_context_from_env() {
    with_rayon_num_threads(Some("2"), || {
        let backend = CpuBackend::new();
        assert_eq!(backend.num_threads(), 2);
    });
}

#[test]
fn cpu_backend_new_falls_back_to_affinity_when_rayon_num_threads_is_absent() {
    with_rayon_num_threads(None, || {
        let backend = CpuBackend::new();
        assert_eq!(backend.num_threads(), crate::available_parallelism());
    });
}

#[test]
fn cpu_backend_try_new_propagates_invalid_rayon_num_threads() {
    with_rayon_num_threads(Some("not-a-number"), || {
        assert!(CpuBackend::try_new().is_err());
    });
}

#[test]
fn test_with_backend_session_runs_compiled_ops() {
    let mut backend = CpuBackend::with_threads(2).unwrap();
    let result = backend.with_backend_session(|session| {
        session
            .add(
                &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap()),
                &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap()),
            )
            .unwrap()
    });
    assert_eq!(get_f64(&result, &[0]), 4.0);
    assert_eq!(get_f64(&result, &[1]), 6.0);
}

#[test]
fn cpu_context_install_enters_owned_pool() {
    let ctx = CpuContext::with_threads(2).unwrap();
    let seen_threads = ctx.install(rayon::current_num_threads);
    assert_eq!(seen_threads, 2);
}

#[test]
fn cpu_install_accepts_send_state() {
    let ctx = CpuContext::with_threads(2).unwrap();
    let state = Arc::new(41usize);
    let seen = ctx.install(|| *state + 1);
    assert_eq!(seen, 42);

    let backend = CpuBackend::with_threads(2).unwrap();
    let state = Arc::new(20usize);
    let seen = backend.install(|| *state + 2);
    assert_eq!(seen, 22);
}

#[test]
fn cpu_backend_exec_session_uses_default_provider_scope() {
    let mut backend = CpuBackend::with_threads(2).unwrap();
    backend.with_backend_session(|_| {
        #[cfg(feature = "cpu-blas")]
        assert!(rayon::current_thread_index().is_none());

        #[cfg(all(not(feature = "cpu-blas"), feature = "cpu-faer"))]
        assert_eq!(rayon::current_num_threads(), 2);
    });
}

#[test]
fn cpu_backend_shared_context() {
    let ctx = Arc::new(CpuContext::with_threads(3).unwrap());
    let b1 = CpuBackend::from_context(ctx.clone());
    let b2 = CpuBackend::from_context(ctx);
    assert_eq!(b1.context_id_for_test(), b2.context_id_for_test());
}

#[test]
fn cpu_affinity_available_parallelism_reports_positive_count() {
    assert!(crate::available_parallelism() >= 1);
}

#[test]
fn cpu_backend_from_context_shares_runtime_owner() {
    let ctx = Arc::new(CpuContext::with_threads(3).unwrap());
    let b1 = CpuBackend::from_context(ctx.clone());
    let b2 = CpuBackend::from_context(ctx);
    assert_eq!(b1.num_threads(), 3);
    assert_eq!(b2.num_threads(), 3);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn cpu_context_faer_policy_is_seq_for_one_thread() {
    let ctx = CpuContext::with_threads(1).unwrap();
    assert!(matches!(ctx.faer_par(), faer::Par::Seq));
}

#[cfg(feature = "cpu-faer")]
#[test]
fn cpu_context_faer_policy_matches_configured_workers_outside_pool() {
    let ctx = CpuContext::with_threads(2).unwrap();
    assert_eq!(ctx.faer_par().degree(), 2);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn cpu_context_faer_policy_matches_configured_workers_inside_context_pool() {
    let ctx = CpuContext::with_threads(2).unwrap();
    let par = ctx.install(|| ctx.faer_par());
    assert_eq!(par.degree(), 2);
}

#[cfg(feature = "cpu-faer")]
#[test]
fn cpu_context_faer_policy_ignores_a_different_ambient_pool_size() {
    let ctx = CpuContext::with_threads(2).unwrap();
    let ambient = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    assert_eq!(ambient.install(|| ctx.faer_par().degree()), 2);
}

#[test]
fn performance_notes_match_current_cpu_threading_contract() {
    let notes = include_str!("../../../../../docs/performance/tt-inner-product-overhead.md");
    assert!(
        !notes.contains("The faer backend is therefore run without a tenferro-owned Rayon pool")
            && !notes.contains("maps multi-threaded execution to `Par::rayon(n)`")
            && !notes.contains("The global-Rayon columns became the production policy"),
        "performance notes must describe CpuContext::install plus Par::rayon(0), not the stale global-Rayon policy"
    );
}

#[test]
fn cpu_context_with_threads_reports_requested_size() {
    let ctx = CpuContext::with_threads(2).unwrap();
    assert_eq!(ctx.num_threads(), 2);
}

#[test]
fn cpu_context_install_executes_closure() {
    let ctx = CpuContext::with_threads(1).unwrap();
    let seen = ctx.install(|| 1 + 1);
    assert_eq!(seen, 2);
}

fn env_lock() -> MutexGuard<'static, ()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct RayonNumThreadsEnvGuard {
    _lock: MutexGuard<'static, ()>,
    prev: Option<OsString>,
}

impl RayonNumThreadsEnvGuard {
    fn new(value: Option<&str>) -> Self {
        let lock = env_lock();
        let prev = std::env::var_os("RAYON_NUM_THREADS");

        match value {
            Some(value) => std::env::set_var("RAYON_NUM_THREADS", value),
            None => std::env::remove_var("RAYON_NUM_THREADS"),
        }

        Self { _lock: lock, prev }
    }
}

impl Drop for RayonNumThreadsEnvGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(value) => std::env::set_var("RAYON_NUM_THREADS", value),
            None => std::env::remove_var("RAYON_NUM_THREADS"),
        }
    }
}

fn with_rayon_num_threads<T>(value: Option<&str>, f: impl FnOnce() -> T) -> T {
    let _guard = RayonNumThreadsEnvGuard::new(value);
    f()
}
