use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use tenferro_cpu::{available_parallelism, CpuBackend, CpuBackendKind, CpuPlacement};
use tenferro_tensor::{BackendSessionHost, DotGeneralConfig, Tensor};

const MATRIX_SIZES: [usize; 3] = [64, 256, 512];

fn matrix(size: usize) -> Tensor {
    let shape = [size, size];
    let len = shape.iter().product();
    Tensor::from_vec_col_major(shape.to_vec(), vec![1.0_f64; len])
        .expect("NUMA benchmark matrix should be valid")
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
    let input = matrix(size);
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

criterion_group!(benches, bench_numa_execution);
criterion_main!(benches);
