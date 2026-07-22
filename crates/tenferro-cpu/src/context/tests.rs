use super::{select_worker_cpus, CpuContext, CpuContextError};
#[cfg(target_os = "linux")]
use crate::affinity::current_cpu;
use crate::affinity::{CpuAffinityError, ThreadAffinity};
use crate::arbiter::worker_execution_scope_registered;
use crate::domain_executor::{indexed_jobs, scoped_job};
#[cfg(target_os = "linux")]
use crate::process_cpu_affinity;
use crate::{
    CpuDomainExecutor, CpuDomainExecutorError, CpuExecutorAffinity, CpuExecutorReentrancy,
    CpuExecutorShutdown, CpuId, CpuInnerParallelism, CpuSet, ScopedCpuJob,
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
    assert_eq!(caps.affinity, CpuExecutorAffinity::TenferroPinnedVerified);
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
        Err(CpuContextError::WorkerPinning { worker: 0, .. })
    ));
}

#[test]
fn pinned_context_rejects_zero_workers() {
    let cpus = CpuSet::new([CpuId::new(0)]).unwrap();
    assert!(matches!(
        CpuContext::with_pinned_cpus_using(cpus, 0, FailingAffinitySetter),
        Err(CpuContextError::InvalidThreadCount)
    ));
}

#[test]
fn worker_assignment_spreads_a_reduced_budget_across_the_domain() {
    let cpus = CpuSet::new((0..8).map(CpuId::new)).unwrap();

    assert_eq!(
        select_worker_cpus(&cpus, 4),
        vec![CpuId::new(0), CpuId::new(2), CpuId::new(4), CpuId::new(7)]
    );
    assert_eq!(select_worker_cpus(&cpus, 1), vec![CpuId::new(4)]);
}

#[test]
fn worker_assignment_repeats_a_smaller_exact_cpu_set() {
    let cpus = CpuSet::new([CpuId::new(2), CpuId::new(7)]).unwrap();

    assert_eq!(
        select_worker_cpus(&cpus, 5),
        vec![
            CpuId::new(2),
            CpuId::new(7),
            CpuId::new(2),
            CpuId::new(7),
            CpuId::new(2),
        ]
    );
    let context = CpuContext::with_pinned_cpus_using(cpus, 5, ExactAffinitySetter).unwrap();
    assert_eq!(context.num_threads(), 5);
}

#[cfg(target_os = "linux")]
#[test]
fn oversubscribed_pinned_context_audits_every_worker_at_a_barrier() {
    use std::sync::{Arc, Barrier};

    let allowed = process_cpu_affinity().unwrap();
    let selected = CpuSet::new([allowed.as_slice()[0]]).unwrap();
    let workers = 4;
    let context = CpuContext::with_pinned_cpus(selected.clone(), workers).unwrap();
    let barrier = Arc::new(Barrier::new(workers));
    let observations = context.install(|| {
        rayon::broadcast(|broadcast| {
            barrier.wait();
            (broadcast.index(), current_cpu().unwrap())
        })
    });

    assert_eq!(
        observations
            .iter()
            .map(|(worker, _)| *worker)
            .collect::<BTreeSet<_>>(),
        (0..workers).collect()
    );
    assert!(observations.iter().all(|(_, cpu)| selected.contains(*cpu)));
}

#[derive(Clone)]
struct FailingAffinitySetter;

impl ThreadAffinity for FailingAffinitySetter {
    fn pin_current(&self, _cpu: CpuId) -> Result<CpuSet, CpuAffinityError> {
        Err(CpuAffinityError::UnsupportedPlatform)
    }
}

#[derive(Clone)]
struct ExactAffinitySetter;

impl ThreadAffinity for ExactAffinitySetter {
    fn pin_current(&self, cpu: CpuId) -> Result<CpuSet, CpuAffinityError> {
        Ok(CpuSet::new([cpu]).unwrap())
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
