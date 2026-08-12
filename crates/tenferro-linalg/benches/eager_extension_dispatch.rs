//! Eager extension single-op dispatch latency: no-AD vs eager-AD forward.
//!
//! This benchmark isolates the per-call eager extension execution cost that
//! issue #1664 root-caused (SemanticProgram build + compile + run_compiled per
//! call, or per-call module install). It is the pre/post gate for issue #1665.

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_linalg::EagerTensorLinalgExt;
use tenferro_tensor::Tensor;

fn runtime() -> Arc<EagerRuntime> {
    EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).expect("cpu backend")).unwrap()
}

fn tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(shape, data).expect("valid tensor")
}

/// Column-major 2x2 matrix from row-major `[[m00, m01], [m10, m11]]`.
fn mat2(m00: f64, m01: f64, m10: f64, m11: f64) -> Tensor {
    tensor(vec![2, 2], vec![m00, m10, m01, m11])
}

fn eager_untracked_from(tensor: Tensor, rt: &Arc<EagerRuntime>) -> EagerTensor {
    EagerTensor::from_tensor_in(tensor, Arc::clone(rt)).expect("eager tensor")
}

fn eager_tracked_from(tensor: Tensor, rt: &Arc<EagerRuntime>) -> EagerTensor {
    EagerTensor::requires_grad_in(tensor, Arc::clone(rt)).expect("eager tensor")
}

/// Materialize and consume the scalar result so lazy work actually runs.
fn consume(output: EagerTensor) {
    let t = output.to_tensor().expect("materialize");
    black_box(t.shape());
    black_box(t.as_slice::<f64>().expect("f64")[0]);
}

fn mat(rt: &Arc<EagerRuntime>, tracked: bool) -> EagerTensor {
    let t = mat2(1.0, 2.0, 3.0, 4.0);
    if tracked {
        eager_tracked_from(t, rt)
    } else {
        eager_untracked_from(t, rt)
    }
}

fn bench_group(c: &mut Criterion, tracked: bool, label: &str) {
    let rt = runtime();
    let mut group = c.benchmark_group(format!("eager_extension_dispatch/{label}"));

    // matmul (standard-op composite; reference row)
    {
        let a = mat(&rt, tracked);
        let b = mat(&rt, tracked);
        group.bench_function(BenchmarkId::new("matmul_2x2", "f64"), |bench| {
            bench.iter(|| {
                let out = black_box(&a).matmul(black_box(&b)).expect("matmul");
                consume(out);
            });
        });
    }

    // solve (extension op)
    {
        let a = eager_untracked_from(mat2(2.0, 0.0, 0.0, 3.0), &rt);
        let b = eager_untracked_from(tensor(vec![2, 1], vec![1.0, 1.0]), &rt);
        group.bench_function(BenchmarkId::new("solve_2x2", "f64"), |bench| {
            bench.iter(|| {
                let out = black_box(&a).solve(black_box(&b)).expect("solve");
                consume(out);
            });
        });
    }

    // svd (extension op)
    {
        let a = eager_untracked_from(mat2(1.0, 0.0, 0.0, 2.0), &rt);
        group.bench_function(BenchmarkId::new("svd_2x2", "f64"), |bench| {
            bench.iter(|| {
                let (u, _s, _vt) = black_box(&a).svd().expect("svd");
                consume(u);
            });
        });
    }

    // eigh (extension op)
    {
        let a = eager_untracked_from(mat2(4.0, 1.0, 1.0, 3.0), &rt);
        group.bench_function(BenchmarkId::new("eigh_2x2", "f64"), |bench| {
            bench.iter(|| {
                let (w, _v) = black_box(&a).eigh().expect("eigh");
                consume(w);
            });
        });
    }

    group.finish();
}

fn bench_no_ad(c: &mut Criterion) {
    bench_group(c, false, "no_ad");
}

fn bench_eager_ad_forward(c: &mut Criterion) {
    bench_group(c, true, "eager_ad_forward");
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5))
        .sample_size(100)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_no_ad, bench_eager_ad_forward
}
criterion_main!(benches);
