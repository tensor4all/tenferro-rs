//! `CpuExecutionContext::rayon_pool`: providers that run kernels on
//! tenferro's pool with the given budget (tensor4all/tprims-rs phase 1e).
use std::sync::Arc;

use tenferro_cpu::{with_cpu_exec_session, CpuBackend, CpuDomainExecutor, RayonCpuDomainExecutor};
use tenferro_tensor::BackendSessionHost;

#[test]
fn inner_context_exposes_the_pool_it_runs_on() {
    let mut backend = CpuBackend::with_threads(2).unwrap();
    let seen = backend
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |cpu| {
                cpu.with_linalg_pool(|context, _| {
                    let pool = context.rayon_pool().expect("inner two-thread context");
                    Ok((
                        pool.current_num_threads(),
                        pool.current_thread_index().is_some(),
                        context.thread_budget().get(),
                    ))
                })
            })
            .expect("a CPU backend session")
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        seen,
        (2, true, 2),
        "(pool threads, on a worker of it, budget)"
    );
}

#[test]
fn sequential_contexts_expose_no_pool() {
    let mut one = CpuBackend::with_threads(1).unwrap();
    let none = one
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |cpu| {
                cpu.with_linalg_pool(|context, _| Ok(context.rayon_pool().is_none()))
            })
            .expect("a CPU backend session")
        })
        .unwrap()
        .unwrap();
    assert!(none, "a one-thread backend has no pool");

    let mut two = CpuBackend::with_threads(2).unwrap();
    let lanes = two
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |cpu| {
                cpu.with_linalg_pool(|context, _| {
                    let seen = std::sync::Mutex::new(Vec::new());
                    context.with_outer_lanes(0..2, |_, lane| {
                        seen.lock().unwrap().push(lane.rayon_pool().is_none());
                    });
                    Ok(seen.into_inner().unwrap())
                })
            })
            .expect("a CPU backend session")
        })
        .unwrap()
        .unwrap();
    assert_eq!(lanes, vec![true, true], "outer lanes are sequential");
}

#[test]
fn rayon_domain_executor_reports_its_pool() {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    let executor = RayonCpuDomainExecutor::new(Arc::clone(&pool));
    let seen = executor.rayon_pool().expect("a Rayon executor");
    assert!(std::ptr::eq(seen, &*pool));
}
