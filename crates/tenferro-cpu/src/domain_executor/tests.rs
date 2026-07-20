use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

#[derive(Debug)]
struct InlineExecutor;

impl InlineExecutor {
    const fn new() -> Self {
        Self
    }
}

impl CpuDomainExecutor for InlineExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        CpuDomainExecutorCapabilities {
            worker_count: NonZeroUsize::new(1).unwrap(),
            outer_parallelism: true,
            inner_parallelism: CpuInnerParallelism::None,
            reentrancy: CpuExecutorReentrancy::Rejected,
            affinity: CpuExecutorAffinity::None,
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

#[test]
fn executor_is_object_safe_and_accepts_borrowed_jobs() {
    let executor = InlineExecutor::new();
    let object: &dyn CpuDomainExecutor = &executor;
    let input = 41usize;
    let mut output = 0usize;
    {
        let mut job = scoped_job(|| output = input + 1);
        object.install(&mut job).unwrap();
    }
    assert_eq!(output, 42);
}

#[test]
fn outer_submission_is_synchronous_and_indexed() {
    let executor = InlineExecutor::new();
    let seen = [AtomicUsize::new(0), AtomicUsize::new(0)];
    let jobs = indexed_jobs(2, |index| {
        seen[index].fetch_add(1, Ordering::Relaxed);
        Ok(())
    });

    executor.submit(&jobs).unwrap();

    assert_eq!(seen.map(|value| value.load(Ordering::Relaxed)), [1, 1]);
}

#[test]
fn zero_job_submission_is_empty_and_runs_nothing() {
    let executor = InlineExecutor::new();
    let calls = AtomicUsize::new(0);
    let jobs = indexed_jobs(0, |_| {
        calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    });

    assert!(jobs.is_empty());
    executor.submit(&jobs).unwrap();

    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn executor_capabilities_preserve_each_declared_axis() {
    let executor = InlineExecutor::new();
    let capabilities = executor.capabilities();

    assert_eq!(capabilities.worker_count.get(), 1);
    assert!(capabilities.outer_parallelism);
    assert_eq!(capabilities.inner_parallelism, CpuInnerParallelism::None);
    assert_eq!(capabilities.reentrancy, CpuExecutorReentrancy::Rejected);
    assert_eq!(capabilities.affinity, CpuExecutorAffinity::None);
    assert_eq!(capabilities.shutdown, CpuExecutorShutdown::CallerOwned);

    let supported = CpuDomainExecutorCapabilities {
        worker_count: NonZeroUsize::new(2).unwrap(),
        outer_parallelism: true,
        inner_parallelism: CpuInnerParallelism::Rayon,
        reentrancy: CpuExecutorReentrancy::SameExecutor,
        affinity: CpuExecutorAffinity::TenferroPinnedVerified,
        shutdown: CpuExecutorShutdown::TenferroOwned,
    };
    assert_eq!(
        supported.affinity,
        CpuExecutorAffinity::TenferroPinnedVerified
    );
    assert_ne!(
        supported.affinity,
        CpuExecutorAffinity::CallerDeclaredUnverified
    );
}

#[test]
fn executor_error_categories_remain_distinct_and_matchable() {
    let admission = CpuDomainExecutorError::Admission {
        message: "domain is busy".to_string(),
    };
    let scheduling = CpuDomainExecutorError::Scheduling {
        message: "worker unavailable".to_string(),
    };
    let cancellation = CpuDomainExecutorError::Cancellation {
        message: "request cancelled".to_string(),
    };
    let panic_bridge = CpuDomainExecutorError::PanicBridge {
        message: "worker panicked".to_string(),
    };

    assert!(matches!(
        admission,
        CpuDomainExecutorError::Admission { .. }
    ));
    assert!(matches!(
        scheduling,
        CpuDomainExecutorError::Scheduling { .. }
    ));
    assert!(matches!(
        cancellation,
        CpuDomainExecutorError::Cancellation { .. }
    ));
    assert!(matches!(
        panic_bridge,
        CpuDomainExecutorError::PanicBridge { .. }
    ));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SentinelOperationError;

#[test]
fn arbitrary_operation_error_survives_executor_dispatch_unchanged() {
    let executor = InlineExecutor::new();

    let operation_result = install_scoped(&executor, || {
        Err::<usize, SentinelOperationError>(SentinelOperationError)
    })
    .unwrap();

    assert_eq!(operation_result, Err(SentinelOperationError));
}
