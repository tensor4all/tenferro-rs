use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuPlacement};
use tenferro_tensor::{BackendSessionHost, DotGeneralConfig, Tensor};

const MATRIX_SHAPE: [usize; 2] = [512, 512];

fn matrix() -> Tensor {
    let len = MATRIX_SHAPE.iter().product();
    Tensor::from_vec_col_major(MATRIX_SHAPE.to_vec(), vec![1.0_f64; len])
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

fn print_metadata(label: &str, backend: &CpuBackend) {
    let info = backend.execution_info();
    eprintln!(
        "numa_execution case={label} allowed={:?} topology={:?} requested={:?} resolved={:?} kind={:?} provider={} workers={} shape={:?}",
        backend.topology().allowed_cpus().as_usize_vec(),
        backend.topology(),
        info.requested_placement(),
        info.resolved_placement(),
        info.backend_kind(),
        info.provider_diagnostic(),
        backend.num_threads(),
        MATRIX_SHAPE,
    );
}

fn bench_numa_execution(c: &mut Criterion) {
    let coordinator = CpuBackend::with_kind(CpuBackendKind::Faer)
        .expect("NUMA benchmark requires the cpu-faer feature");
    let nodes = coordinator.topology().nodes();
    if nodes.len() < 2 {
        eprintln!(
            "numa_execution skipped: need at least two process-visible NUMA nodes; allowed={:?} topology={:?}",
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
    let all_allowed = coordinator
        .for_placement(CpuPlacement::AllAllowed)
        .expect("faer AllAllowed placement should resolve");
    let input = matrix();

    print_metadata("disjoint_nodes_concurrent/first", &node0);
    print_metadata("disjoint_nodes_concurrent/second", &node1);
    print_metadata("all_allowed", &all_allowed);

    let mut node0_backend = node0;
    let mut node1_backend = node1;
    c.bench_function("numa_execution/disjoint_nodes_concurrent", |b| {
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

    let mut all_backend = all_allowed;
    c.bench_function("numa_execution/all_allowed", |b| {
        b.iter(|| black_box(run_session_workload(&mut all_backend, &input)));
    });

    #[cfg(feature = "cpu-blas")]
    {
        let provider = CpuBackend::with_kind(CpuBackendKind::Blas)
            .expect("compiled BLAS provider should construct");
        print_metadata("provider_default_exclusive", &provider);
        let mut provider_backend = provider;
        c.bench_function("numa_execution/provider_default_exclusive", |b| {
            b.iter(|| black_box(run_session_workload(&mut provider_backend, &input)));
        });
    }

    #[cfg(not(feature = "cpu-blas"))]
    eprintln!(
        "numa_execution provider_default_exclusive skipped: compile cpu-blas with a linked provider"
    );
}

criterion_group!(benches, bench_numa_execution);
criterion_main!(benches);
