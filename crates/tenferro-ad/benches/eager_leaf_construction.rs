//! Eager leaf construction cost (#1704).
//!
//! Frozen one-thread protocol: the runtime is built with
//! `tenferro_cpu::CpuEngine::with_threads(1)` and the effective worker count is asserted, so
//! a run cannot silently measure an all-core configuration. The paired
//! baseline/candidate comparison and the acceptance gate are recorded in
//! `docs/worklogs/issue-1704-eager-leaf-session.md`.

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_tensor::Tensor;

fn runtime() -> Arc<EagerRuntime> {
    let backend = tenferro_cpu::CpuEngine::serial();
    assert_eq!(
        backend.num_threads(),
        1,
        "the #1704 experiment requires exactly one CPU worker"
    );
    EagerRuntime::new().expect("eager runtime")
}

fn leaf(runtime: &Arc<EagerRuntime>, dims: &[usize]) {
    let len = dims.iter().product();
    let tensor =
        Tensor::from_vec_col_major(dims.to_vec(), vec![1.0_f64; len]).expect("valid tensor");
    let value = EagerTensor::from_tensor_in(tensor, Arc::clone(runtime)).expect("leaf");
    black_box(value);
}

fn bench_leaf_construction(c: &mut Criterion) {
    let runtime = runtime();
    let mut group = c.benchmark_group("eager_leaf_construction");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(2));

    group.bench_function("input_tensor_8", |b| {
        b.iter(|| {
            let tensor =
                Tensor::from_vec_col_major(vec![2, 2, 2], vec![1.0_f64; 8]).expect("valid tensor");
            black_box(tensor);
        })
    });
    group.bench_function("from_tensor_in_8", |b| {
        b.iter(|| leaf(&runtime, &[2, 2, 2]))
    });
    group.bench_function("from_tensor_in_256", |b| {
        b.iter(|| leaf(&runtime, &[4, 4, 4, 4]))
    });
    group.bench_function("requires_grad_in_8", |b| {
        b.iter(|| {
            let tensor =
                Tensor::from_vec_col_major(vec![2, 2, 2], vec![1.0_f64; 8]).expect("valid tensor");
            let value = EagerTensor::requires_grad_in(tensor, Arc::clone(&runtime)).expect("leaf");
            black_box(value);
        })
    });
    group.finish();
}

criterion_group!(benches, bench_leaf_construction);
criterion_main!(benches);
