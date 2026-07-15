use super::{select_worker_cpus, CpuContext, CpuContextError};
#[cfg(target_os = "linux")]
use crate::affinity::current_cpu;
use crate::affinity::ThreadAffinity;
use crate::arbiter::worker_execution_scope_registered;
#[cfg(target_os = "linux")]
use crate::process_cpu_affinity;
use crate::{CpuId, CpuSet};
use rayon::prelude::*;
#[cfg(target_os = "linux")]
use std::collections::BTreeSet;

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

#[test]
fn worker_assignment_spreads_a_reduced_budget_across_the_domain() {
    let cpus = CpuSet::new((0..8).map(CpuId::new)).unwrap();

    assert_eq!(
        select_worker_cpus(&cpus, 4),
        vec![CpuId::new(0), CpuId::new(2), CpuId::new(4), CpuId::new(7)]
    );
    assert_eq!(select_worker_cpus(&cpus, 1), vec![CpuId::new(4)]);
}

#[derive(Clone)]
struct FailingAffinitySetter;

impl ThreadAffinity for FailingAffinitySetter {
    fn pin_current(&self, _cpu: CpuId) -> Result<CpuSet, String> {
        Err("forced test failure".to_owned())
    }
}
