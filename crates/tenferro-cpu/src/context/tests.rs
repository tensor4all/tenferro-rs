use super::{CpuContext, CpuContextError, DEFAULT_WORKER_STACK_BYTES};
#[cfg(target_os = "linux")]
use crate::affinity::current_cpu;
use crate::affinity::{CpuAffinityError, ThreadAffinity};
use crate::arbiter::worker_execution_scope_registered;
use crate::domain_executor::{indexed_jobs, scoped_job};
#[cfg(target_os = "linux")]
use crate::process_cpu_affinity;
use crate::{
    CpuDomainExecutor, CpuDomainExecutorError, CpuExecutorAffinity, CpuExecutorReentrancy,
    CpuExecutorShutdown, CpuId, CpuInnerParallelism, CpuSet, Error, ScopedCpuJob,
};
#[cfg(target_os = "linux")]
use rayon::prelude::*;
#[cfg(target_os = "linux")]
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn managed_context_reports_verified_rayon_capabilities() {
    let ctx = CpuContext::with_threads(2).unwrap();
    let caps = CpuDomainExecutor::capabilities(&ctx);
    assert_eq!(caps.worker_count.get(), 2);
    assert!(caps.outer_parallelism);
    assert_eq!(caps.inner_parallelism, CpuInnerParallelism::Rayon);
    assert_eq!(caps.reentrancy, CpuExecutorReentrancy::SameExecutor);
    assert_eq!(caps.affinity, CpuExecutorAffinity::None);
    assert_eq!(caps.shutdown, CpuExecutorShutdown::TenferroOwned);
}

#[test]
fn direct_context_reports_inline_capabilities_and_runs_inline() {
    let ctx = CpuContext::with_threads(1).unwrap();
    let caps = CpuDomainExecutor::capabilities(&ctx);
    assert_eq!(caps.worker_count.get(), 1);
    assert!(!caps.outer_parallelism);
    assert_eq!(caps.inner_parallelism, CpuInnerParallelism::None);

    let caller = std::thread::current().id();
    let submitted = AtomicUsize::new(0);
    let empty_jobs = indexed_jobs(0, |_| panic!("empty submission must not run a job"));
    CpuDomainExecutor::submit(&ctx, &empty_jobs).unwrap();
    let jobs = indexed_jobs(3, |_| {
        assert_eq!(std::thread::current().id(), caller);
        submitted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    });
    CpuDomainExecutor::submit(&ctx, &jobs).unwrap();

    let installed = AtomicUsize::new(0);
    let mut job = scoped_job(|| {
        assert_eq!(std::thread::current().id(), caller);
        installed.fetch_add(1, Ordering::Relaxed);
    });
    CpuDomainExecutor::install(&ctx, &mut job).unwrap();

    assert_eq!(submitted.load(Ordering::Relaxed), 3);
    assert_eq!(installed.load(Ordering::Relaxed), 1);
}

#[test]
fn pinned_context_reports_verified_affinity_with_test_setter() {
    let cpus = CpuSet::new([CpuId::new(17)]).unwrap();
    let ctx = CpuContext::with_pinned_cpus_using(cpus, 1, ExactAffinitySetter).unwrap();
    let caps = CpuDomainExecutor::capabilities(&ctx);

    assert_eq!(caps.worker_count.get(), 1);
    assert!(!caps.outer_parallelism);
    assert_eq!(caps.inner_parallelism, CpuInnerParallelism::Rayon);
    assert_eq!(caps.affinity, CpuExecutorAffinity::TenferroDomainVerified);
}

#[test]
fn pooled_submit_runs_every_index_once_on_the_selected_context() {
    let ctx = CpuContext::with_threads(2).unwrap();
    let calls = (0..32).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
    let jobs = indexed_jobs(calls.len(), |index| {
        assert!(ctx.owns_current_worker_for_test());
        calls[index].fetch_add(1, Ordering::Relaxed);
        Ok(())
    });
    let executor: &dyn CpuDomainExecutor = &ctx;

    executor.submit(&jobs).unwrap();

    assert!(calls.iter().all(|calls| calls.load(Ordering::Relaxed) == 1));
}

#[test]
fn pooled_submit_leaves_a_foreign_rayon_pool_for_the_selected_context() {
    let ctx = CpuContext::with_threads(2).unwrap();
    let foreign_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let calls = AtomicUsize::new(0);
    let jobs = indexed_jobs(32, |_| {
        assert!(ctx.owns_current_worker_for_test());
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    });
    let executor: &dyn CpuDomainExecutor = &ctx;

    foreign_pool
        .install(|| {
            assert!(!ctx.owns_current_worker_for_test());
            executor.submit(&jobs)
        })
        .unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 32);
}

#[test]
fn trait_install_reenters_the_matching_context_without_a_second_pool_entry() {
    let ctx = CpuContext::with_threads(2).unwrap();
    let calls = AtomicUsize::new(0);
    let mut job = scoped_job(|| {
        assert!(ctx.owns_current_worker_for_test());
        calls.fetch_add(1, Ordering::Relaxed);
    });

    ctx.install(|| {
        assert!(ctx.owns_current_worker_for_test());
        let executor: &dyn CpuDomainExecutor = &ctx;
        assert_eq!(
            executor.capabilities().reentrancy,
            CpuExecutorReentrancy::SameExecutor
        );
        executor.install(&mut job)
    })
    .unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn managed_context_preserves_job_error_categories() {
    let ctx = CpuContext::with_threads(2).unwrap();
    let jobs = indexed_jobs(8, |_| {
        Err(CpuDomainExecutorError::Cancellation {
            message: "test indexed cancellation".to_string(),
        })
    });

    assert_eq!(
        CpuDomainExecutor::submit(&ctx, &jobs),
        Err(CpuDomainExecutorError::Cancellation {
            message: "test indexed cancellation".to_string(),
        })
    );

    let mut job = AdmissionFailingJob;
    assert_eq!(
        CpuDomainExecutor::install(&ctx, &mut job),
        Err(CpuDomainExecutorError::Admission {
            message: "test install admission".to_string(),
        })
    );
}

#[test]
fn with_threads_rejects_zero() {
    assert!(CpuContext::with_threads(0).is_err());
}

#[test]
fn context_constructor_registers_every_rayon_worker_execution_scope() {
    for threads in [2, 4] {
        let ctx = CpuContext::with_threads(threads).unwrap();
        let registered = ctx
            .pool
            .as_ref()
            .unwrap()
            .broadcast(|_| worker_execution_scope_registered());
        assert_eq!(registered, vec![true; threads]);
    }
}

#[cfg(target_os = "linux")]
#[test]
fn pinned_context_reports_only_assigned_cpus() {
    let allowed = process_cpu_affinity().unwrap();
    let selected = CpuSet::new(allowed.as_slice().iter().take(2).copied()).unwrap();
    let ctx = CpuContext::with_pinned_cpus(selected.clone(), selected.len()).unwrap();
    let observed = ctx.install(|| {
        (0..4096usize)
            .into_par_iter()
            .map(|_| current_cpu().unwrap())
            .collect::<BTreeSet<_>>()
    });

    assert!(observed.iter().all(|cpu| selected.contains(*cpu)));
    assert_eq!(ctx.pinned_cpus(), Some(&selected));
}

#[cfg(target_os = "linux")]
#[test]
fn pinned_workers_are_confined_to_the_whole_domain_cpu_set() {
    let allowed = process_cpu_affinity().unwrap();
    let selected = CpuSet::new(allowed.as_slice().iter().take(2).copied()).unwrap();
    let workers = selected.len();
    let ctx = CpuContext::with_pinned_cpus(selected.clone(), workers).unwrap();

    let observed = ctx
        .pool
        .as_ref()
        .unwrap()
        .broadcast(|_| process_cpu_affinity());

    assert_eq!(observed.len(), workers);
    assert!(observed.iter().all(|mask| mask.as_ref() == Some(&selected)));
}

#[cfg(target_os = "linux")]
#[test]
fn provider_style_thread_from_a_worker_inherits_the_domain_cpu_set() {
    // A BLAS/LAPACK provider creates its own thread team, and those threads
    // inherit the creating thread's mask. Confining a worker to one CPU would
    // therefore confine the provider's whole team, so each worker must carry the
    // complete domain set.
    let allowed = process_cpu_affinity().unwrap();
    let selected = CpuSet::new(allowed.as_slice().iter().take(2).copied()).unwrap();
    let ctx = CpuContext::with_pinned_cpus(selected.clone(), selected.len()).unwrap();

    let inherited = ctx.pool.as_ref().unwrap().broadcast(|_| {
        std::thread::spawn(|| process_cpu_affinity().unwrap())
            .join()
            .unwrap()
    });

    assert!(inherited.iter().all(|mask| mask == &selected));
}

#[cfg(target_os = "linux")]
#[test]
fn pinned_single_worker_context_still_enters_a_real_rayon_pool() {
    let allowed = process_cpu_affinity().unwrap();
    let selected = CpuSet::new(allowed.as_slice().iter().take(1).copied()).unwrap();
    let ctx = CpuContext::with_pinned_cpus(selected, 1).unwrap();

    assert!(ctx.install(|| rayon::current_thread_index().is_some()));
}

#[test]
fn pin_failure_aborts_context_construction() {
    let result = CpuContext::with_pinned_cpus_using(
        CpuSet::new([CpuId::new(0)]).unwrap(),
        1,
        FailingAffinitySetter,
    );

    assert!(matches!(
        result,
        Err(CpuContextError::WorkerAffinity { worker: 0, .. })
    ));
}

#[test]
fn pinned_context_rejects_invalid_worker_counts() {
    let cpus = CpuSet::new([CpuId::new(0)]).unwrap();
    assert!(matches!(
        CpuContext::with_pinned_cpus_using(cpus.clone(), 0, FailingAffinitySetter),
        Err(CpuContextError::InvalidThreadCount)
    ));
    assert!(matches!(
        CpuContext::with_pinned_cpus_using(cpus, 2, FailingAffinitySetter),
        Err(CpuContextError::TooManyWorkers {
            workers: 2,
            cpus: 1
        })
    ));
}

/// Recursion that keeps about one MiB of its own frame per level.
///
/// Provider implementations recurse with large private frames (a `NUM_THREADS=64`
/// OpenBLAS build reserves roughly 541 KiB per level of its threaded LU), so this
/// shape reproduces the failure the worker stack size exists to prevent: on the
/// `std::thread` default of 2 MiB a dozen levels abort the process.
#[inline(never)]
fn consume_one_mib_frames(depth: usize) -> usize {
    let mut frame = [0u8; 1 << 20];
    frame.fill(depth as u8);
    let observed = std::hint::black_box(&frame)[depth & 0xff] ^ frame[(1 << 20) - 1];
    std::hint::black_box(observed);
    if depth == 0 {
        0
    } else {
        1 + consume_one_mib_frames(depth - 1)
    }
}

#[test]
fn context_defaults_to_the_documented_worker_stack() {
    let ctx = CpuContext::with_threads(2).unwrap();
    assert_eq!(ctx.worker_stack_bytes(), DEFAULT_WORKER_STACK_BYTES);
}

#[test]
fn context_worker_stack_is_configurable_per_context() {
    let ctx = CpuContext::with_threads_and_worker_stack(2, 8 << 20).unwrap();
    assert_eq!(ctx.worker_stack_bytes(), 8 << 20);
    assert_eq!(ctx.install(|| 1 + 1), 2);
}

#[test]
fn context_rejects_unusable_worker_stacks() {
    for bytes in [0usize, 1, 1024] {
        assert!(
            matches!(
                CpuContext::with_threads_and_worker_stack(2, bytes),
                Err(Error::Validation { .. })
            ),
            "worker stack {bytes} should be rejected"
        );
    }
}

#[test]
fn worker_pool_runs_recursion_beyond_the_std_default_stack() {
    let ctx = CpuContext::with_threads(2).unwrap();
    assert_eq!(ctx.install(|| consume_one_mib_frames(12)), 12);
}

#[test]
fn pinned_worker_pool_runs_recursion_beyond_the_std_default_stack() {
    // The pinned path installs a custom spawn handler, which must forward the
    // Rayon stack size to the OS thread itself.
    let cpus = CpuSet::new([CpuId::new(17)]).unwrap();
    let ctx = CpuContext::with_pinned_cpus_and_worker_stack(cpus, 1, 16 << 20, ExactAffinitySetter)
        .unwrap();
    assert_eq!(ctx.worker_stack_bytes(), 16 << 20);
    assert_eq!(ctx.install(|| consume_one_mib_frames(12)), 12);
}

#[derive(Clone)]
struct FailingAffinitySetter;

impl ThreadAffinity for FailingAffinitySetter {
    fn confine_current(&self, _cpus: &CpuSet) -> Result<CpuSet, CpuAffinityError> {
        Err(CpuAffinityError::UnsupportedPlatform)
    }
}

#[derive(Clone)]
struct ExactAffinitySetter;

impl ThreadAffinity for ExactAffinitySetter {
    fn confine_current(&self, cpus: &CpuSet) -> Result<CpuSet, CpuAffinityError> {
        Ok(cpus.clone())
    }
}

struct AdmissionFailingJob;

impl ScopedCpuJob for AdmissionFailingJob {
    fn run(&mut self) -> Result<(), CpuDomainExecutorError> {
        Err(CpuDomainExecutorError::Admission {
            message: "test install admission".to_string(),
        })
    }
}
