use super::{CpuContext, CpuContextError, DEFAULT_WORKER_STACK_BYTES};
#[cfg(target_os = "linux")]
use crate::affinity::current_cpu;
use crate::affinity::{CpuAffinityError, ThreadAffinity};
#[cfg(target_os = "linux")]
use crate::process_cpu_affinity;
use crate::{CpuId, CpuSet, Error};
#[cfg(target_os = "linux")]
use rayon::prelude::*;
#[cfg(target_os = "linux")]
use std::collections::BTreeSet;

#[test]
fn with_threads_rejects_zero() {
    assert!(CpuContext::with_threads(0).is_err());
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
