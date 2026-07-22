use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::{Arc, Barrier};
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use tenferro_ad::EagerRuntime;
use tenferro_cpu::{
    process_cpu_affinity, CpuBackend, CpuContext, CpuPlacement, CpuPlacementGuarantee, CpuSet,
    ExternalCpuDomain, NumaNodeId, ResolvedCpuPlacement,
};
use tenferro_tensor::{CpuDomainId, DotGeneralConfig, Tensor};

fn tensor(shape: Vec<usize>, salt: usize) -> Tensor {
    let len = shape.iter().product();
    Tensor::from_vec_col_major(
        shape,
        (0..len)
            .map(|index| ((index * 19 + salt * 11) % 103) as f64 / 41.0 - 1.0)
            .collect(),
    )
    .unwrap()
}

fn current_cpu() -> usize {
    unsafe extern "C" {
        fn sched_getcpu() -> std::ffi::c_int;
    }
    // SAFETY: sched_getcpu has no arguments or preconditions.
    let cpu = unsafe { sched_getcpu() };
    usize::try_from(cpu).expect("Phase 2E affinity audit requires sched_getcpu")
}

fn maybe_write_affinity(key: &str, backend: &CpuBackend, ownership: &str, budget: usize) {
    if std::env::var("TENFERRO_PHASE2E_AFFINITY_ROW").as_deref() != Ok(key) {
        return;
    }
    let path = std::env::var_os("TENFERRO_PHASE2E_AFFINITY_FILE")
        .expect("Phase 2E selected row requires an affinity artifact path");
    let barrier = Arc::new(Barrier::new(budget));
    let observations = backend.install(Box::new(move || {
        rayon::broadcast(|worker| {
            barrier.wait();
            [worker.index(), current_cpu()]
        })
    }));
    let info = backend.execution_info();
    let declared = info.domain_cpus().as_usize_vec();
    if ownership != "external-advisory" {
        assert!(observations.iter().all(|item| declared.contains(&item[1])));
    }
    let guarantee = if ownership == "external-advisory" {
        "AdvisoryDeclared"
    } else {
        "ExactDeclared"
    };
    std::fs::write(
        path,
        format!(
            "{{\"key\":{key:?},\"ownership\":{ownership:?},\"guarantee\":{guarantee:?},\"budget\":{budget},\"worker_count\":{},\"declared_cpus\":{declared:?},\"observations\":{observations:?}}}\n",
            info.worker_count(),
        ),
    )
    .unwrap();
}

fn phase2e_characterization(c: &mut Criterion) {
    let native_lhs = tensor(vec![65_536], 1);
    let native_rhs = tensor(vec![65_536], 2);
    let dot_lhs = tensor(vec![128, 128], 3);
    let dot_rhs = tensor(vec![128, 128], 4);
    let dot_config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    for ownership in ["managed-exact", "external-exact", "external-advisory"] {
        for budget in [1, 2, 4] {
            let backend = phase2e_backend(ownership, budget);
            let native_key = format!("{ownership}/budget-{budget}/E-N");
            let dot_key = format!("{ownership}/budget-{budget}/E-D");
            maybe_write_affinity(&native_key, &backend, ownership, budget);
            maybe_write_affinity(&dot_key, &backend, ownership, budget);
            let placement = backend.placement();
            let runtime = EagerRuntime::with_cpu_backend(backend);
            let mut placed = runtime.on_cpu(placement).unwrap();
            c.bench_function(&format!("phase2e/{native_key}"), |b| {
                b.iter(|| {
                    black_box(
                        placed
                            .with_eager_session(|session| {
                                session
                                    .add(&native_lhs, &native_rhs)
                                    .map_err(tenferro_ad::Error::from)
                            })
                            .unwrap(),
                    )
                });
            });
            c.bench_function(&format!("phase2e/{dot_key}"), |b| {
                b.iter(|| {
                    black_box(
                        placed
                            .with_eager_session(|session| {
                                session
                                    .dot_general(&dot_lhs, &dot_rhs, &dot_config)
                                    .map_err(tenferro_ad::Error::from)
                            })
                            .unwrap(),
                    )
                });
            });
        }
    }
}

fn phase2e_backend(ownership: &str, budget: usize) -> CpuBackend {
    if ownership == "managed-exact" {
        let coordinator = CpuBackend::with_threads(budget).unwrap();
        let node = coordinator
            .topology()
            .nodes()
            .first()
            .expect("managed-exact latency requires one usable NUMA node");
        return coordinator
            .for_placement(CpuPlacement::NumaNode(node.id()))
            .unwrap();
    }
    let allowed = process_cpu_affinity().expect("Phase 2E benchmark needs process affinity");
    let id = CpuDomainId::new(
        0x2eba + budget as u64 + u64::from(u8::from(ownership == "external-exact")),
    );
    let exact = ownership == "external-exact";
    let selected = CpuSet::new(allowed.as_slice().iter().copied().take(budget)).unwrap();
    assert_eq!(
        selected.len(),
        budget,
        "real latency requires B allowed CPUs"
    );
    let placement = if exact {
        ResolvedCpuPlacement::NumaNode {
            id: NumaNodeId::new(0x2e),
            cpus: selected.clone(),
        }
    } else {
        ResolvedCpuPlacement::AllAllowed {
            cpus: allowed.clone(),
        }
    };
    let context = if exact {
        CpuContext::with_pinned_cpus(selected, budget).unwrap()
    } else {
        CpuContext::with_threads(budget).unwrap()
    };
    let domain = ExternalCpuDomain::new(
        id,
        placement,
        Arc::new(context),
        NonZeroUsize::new(budget).unwrap(),
        if exact {
            CpuPlacementGuarantee::ExactDeclared
        } else {
            CpuPlacementGuarantee::AdvisoryDeclared
        },
    )
    .unwrap();
    CpuBackend::from_external_managed_domains(id, [domain]).unwrap()
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(100)
        .confidence_level(0.95);
    targets = phase2e_characterization
}
criterion_main!(benches);
