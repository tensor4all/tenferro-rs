use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::{Arc, Barrier};
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use tenferro_cpu::{
    available_parallelism, process_cpu_affinity, CpuBackend, CpuBackendKind, CpuContext,
    CpuPlacement, CpuPlacementGuarantee, CpuSet, ExternalCpuDomain, NumaNodeId,
    ResolvedCpuPlacement,
};
use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, BackendSessionHost, ContractionScalar, CpuDomainId,
    DotGeneralAccumulation, DotGeneralConfig, Tensor, TensorDot, TensorElementwise, TensorRead,
    TensorWrite,
};

const MATRIX_SIZES: [usize; 3] = [64, 256, 512];

fn matrix(size: usize, salt: usize) -> Tensor {
    let shape = [size, size];
    let len = shape.iter().product();
    Tensor::from_vec_col_major(
        shape.to_vec(),
        (0..len)
            .map(|index| ((index * 29 + salt * 17) % 251) as f64 / 97.0 - 1.0)
            .collect(),
    )
    .expect("NUMA benchmark matrix should be valid")
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

fn run_session_workload(backend: &mut CpuBackend, input: &Tensor) -> Tensor {
    backend
        .with_backend_session(|exec| {
            let squared = exec.mul(input, input)?;
            exec.dot_general(
                &squared,
                input,
                &DotGeneralConfig {
                    lhs_contracting_dims: vec![1],
                    rhs_contracting_dims: vec![0],
                    lhs_batch_dims: vec![],
                    rhs_batch_dims: vec![],
                },
            )
        })
        .expect("NUMA benchmark session workload should succeed")
}

fn print_metadata(label: &str, backend: &CpuBackend, size: usize) {
    let info = backend.execution_info();
    eprintln!(
        "numa_execution case={label} allowed={:?} topology={:?} requested={:?} resolved={:?} mode={:?} kind={:?} provider={} workers={} shape={:?}",
        info.topology().allowed_cpus().as_usize_vec(),
        info.topology(),
        info.requested_placement(),
        info.resolved_placement(),
        info.execution_mode(),
        info.backend_kind(),
        info.provider_diagnostic(),
        info.worker_count(),
        [size, size],
    );
}

fn bench_configuration(c: &mut Criterion, size: usize, threads: usize) {
    let coordinator = CpuBackend::with_threads_and_kind(threads, CpuBackendKind::Faer)
        .expect("NUMA benchmark requires the cpu-faer feature");
    let all_allowed = coordinator
        .for_placement(CpuPlacement::AllAllowed)
        .expect("faer AllAllowed placement should resolve");
    let input = matrix(size, 1);
    let case = format!("numa_execution/{size}x{size}/threads_{threads}");
    print_metadata(&format!("{case}/all_allowed"), &all_allowed, size);

    let mut all_backend = all_allowed;
    c.bench_function(&format!("{case}/all_allowed"), |b| {
        b.iter(|| black_box(run_session_workload(&mut all_backend, &input)));
    });

    #[cfg(feature = "cpu-blas")]
    {
        let provider = CpuBackend::with_threads_and_kind(threads, CpuBackendKind::Blas)
            .expect("compiled BLAS provider should construct");
        print_metadata(
            &format!("{case}/provider_default_exclusive"),
            &provider,
            size,
        );
        let mut provider_backend = provider;
        c.bench_function(&format!("{case}/provider_default_exclusive"), |b| {
            b.iter(|| black_box(run_session_workload(&mut provider_backend, &input)));
        });
    }

    #[cfg(not(feature = "cpu-blas"))]
    eprintln!("{case}/provider_default_exclusive skipped: compile cpu-blas with a linked provider");

    let nodes = coordinator.topology().nodes();
    if nodes.len() < 2 {
        eprintln!(
            "{case}/disjoint_nodes_concurrent skipped: need at least two process-visible NUMA nodes; allowed={:?} topology={:?}",
            coordinator.topology().allowed_cpus().as_usize_vec(),
            coordinator.topology(),
        );
        return;
    }

    let node0 = coordinator
        .for_placement(CpuPlacement::NumaNode(nodes[0].id()))
        .expect("first reported NUMA node should resolve");
    let node1 = coordinator
        .for_placement(CpuPlacement::NumaNode(nodes[1].id()))
        .expect("second reported NUMA node should resolve");
    print_metadata(
        &format!("{case}/disjoint_nodes_concurrent/first"),
        &node0,
        size,
    );
    print_metadata(
        &format!("{case}/disjoint_nodes_concurrent/second"),
        &node1,
        size,
    );

    let mut node0_backend = node0;
    let mut node1_backend = node1;
    c.bench_function(&format!("{case}/disjoint_nodes_concurrent"), |b| {
        b.iter(|| {
            std::thread::scope(|scope| {
                let first = scope.spawn(|| run_session_workload(&mut node0_backend, &input));
                let second = scope.spawn(|| run_session_workload(&mut node1_backend, &input));
                black_box(first.join().expect("first NUMA benchmark worker panicked"));
                black_box(
                    second
                        .join()
                        .expect("second NUMA benchmark worker panicked"),
                );
            });
        });
    });
}

fn bench_numa_execution(c: &mut Criterion) {
    let default_threads = available_parallelism();
    let mut thread_budgets = vec![1, default_threads];
    thread_budgets.sort_unstable();
    thread_budgets.dedup();

    for size in MATRIX_SIZES {
        for &threads in &thread_budgets {
            bench_configuration(c, size, threads);
        }
    }
}

fn bench_phase2e_rows(c: &mut Criterion) {
    let native_lhs = Tensor::from_vec_col_major(
        vec![65_536],
        (0..65_536)
            .map(|index| (index % 97) as f64 / 31.0)
            .collect(),
    )
    .unwrap();
    let native_rhs = Tensor::from_vec_col_major(
        vec![65_536],
        (0..65_536)
            .map(|index| ((index * 19 + 7) % 101) as f64 / 37.0 - 1.0)
            .collect(),
    )
    .unwrap();
    let dot_config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    for ownership in ["managed-exact", "external-exact", "external-advisory"] {
        for budget in [1, 2, 4] {
            let mut native = phase2e_backend(ownership, budget);
            let native_key = format!("{ownership}/budget-{budget}/D-N");
            maybe_write_affinity(&native_key, &native, ownership, budget);
            c.bench_function(&format!("phase2e/{native_key}"), |b| {
                b.iter(|| black_box(native.add(&native_lhs, &native_rhs).unwrap()))
            });

            let dot_lhs = matrix(128, 2);
            let dot_rhs = matrix(128, 3);
            let mut dot = phase2e_backend(ownership, budget);
            let dot_key = format!("{ownership}/budget-{budget}/D-D");
            maybe_write_affinity(&dot_key, &dot, ownership, budget);
            c.bench_function(&format!("phase2e/{dot_key}"), |b| {
                b.iter(|| black_box(dot.dot_general(&dot_lhs, &dot_rhs, &dot_config).unwrap()))
            });

            let jobs_len = 2 * budget + 1;
            let matrix_len = 64 * 64;
            let grouped_lhs = Tensor::from_vec_col_major(
                vec![jobs_len * matrix_len],
                (0..jobs_len * matrix_len)
                    .map(|index| ((index * 31 + 5) % 257) as f64 / 113.0 - 0.75)
                    .collect(),
            )
            .unwrap();
            let grouped_rhs = Tensor::from_vec_col_major(
                vec![jobs_len * matrix_len],
                (0..jobs_len * matrix_len)
                    .map(|index| ((index * 37 + 11) % 263) as f64 / 127.0 - 0.5)
                    .collect(),
            )
            .unwrap();
            let mut grouped_out = Tensor::from_vec_col_major(
                vec![jobs_len * matrix_len],
                vec![0.0_f64; jobs_len * matrix_len],
            )
            .unwrap();
            let jobs: Vec<_> = (0..jobs_len)
                .map(|index| {
                    let offset = index * matrix_len;
                    GroupedGemmJob::new(offset, offset, offset, 64, 64, 64)
                })
                .collect();
            let accumulation = DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::F64(1.0),
                beta: ContractionScalar::F64(0.0),
            };
            let config = GroupedGemmConfig::new(&jobs, accumulation);
            let mut grouped = phase2e_backend(ownership, budget);
            let grouped_key = format!("{ownership}/budget-{budget}/G-O");
            maybe_write_affinity(&grouped_key, &grouped, ownership, budget);
            let mut cache = <CpuBackend as BackendRuntimeCache>::RuntimeCache::default();
            c.bench_function(&format!("phase2e/{grouped_key}"), |b| {
                b.iter(|| {
                    BackendCachedDot::grouped_gemm_cached(
                        &mut grouped,
                        &mut cache,
                        None,
                        TensorRead::from_tensor(&grouped_lhs),
                        TensorRead::from_tensor(&grouped_rhs),
                        &config,
                        TensorWrite::from_tensor(&mut grouped_out),
                    )
                    .unwrap();
                    black_box(grouped_out.as_slice::<f64>().unwrap()[0])
                });
            });
        }
    }
}

fn phase2e_backend(ownership: &str, budget: usize) -> CpuBackend {
    if ownership == "managed-exact" {
        let coordinator = CpuBackend::with_threads_and_kind(budget, CpuBackendKind::Faer).unwrap();
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
        0x2eb0 + budget as u64 + u64::from(u8::from(ownership == "external-exact")),
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
    targets = bench_numa_execution, bench_phase2e_rows
}
criterion_main!(benches);
