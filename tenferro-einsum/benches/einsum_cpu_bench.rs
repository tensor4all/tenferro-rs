use std::env;
use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_einsum::eager_tensor::einsum;
use tenferro_runtime::Tensor;

const BATCHES: &[usize] = &[16, 64, 256];
const BATCHED_SMALL_SIZES: &[usize] = &[2, 4, 8, 16];

fn bench_threads() -> usize {
    env::var("TENFERRO_BENCH_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or(1)
}

fn cpu_ctx(threads: usize) -> Arc<EagerRuntime> {
    EagerRuntime::with_cpu_backend(CpuBackend::with_threads(threads))
}

fn f64_tensor(shape: Vec<usize>, seed: usize) -> Tensor {
    let len = shape.iter().product();
    let data = (0..len)
        .map(|idx| ((idx * 17 + seed * 31 + 7) % 997) as f64 / 997.0 - 0.5)
        .collect();
    Tensor::from_vec_col_major(shape, data)
}

fn c64_tensor(shape: Vec<usize>, seed: usize) -> Tensor {
    let len = shape.iter().product();
    let data = (0..len)
        .map(|idx| {
            let re = ((idx * 17 + seed * 31 + 7) % 997) as f64 / 997.0 - 0.5;
            let im = ((idx * 23 + seed * 19 + 11) % 991) as f64 / 991.0 - 0.5;
            Complex64::new(re, im)
        })
        .collect();
    Tensor::from_vec_col_major(shape, data)
}

fn eager(ctx: &Arc<EagerRuntime>, tensor: Tensor) -> EagerTensor {
    EagerTensor::from_tensor_in(tensor, Arc::clone(ctx))
}

fn consume_f64(tensor: &EagerTensor) {
    let data = tensor.data();
    black_box(data.shape());
    black_box(data.as_slice::<f64>().expect("f64 tensor")[0]);
}

fn consume_c64(tensor: &EagerTensor) {
    let data = tensor.data();
    black_box(data.shape());
    black_box(data.as_slice::<Complex64>().expect("c64 tensor")[0]);
}

fn bench_batched_einsum(c: &mut Criterion) {
    let threads = bench_threads();
    let ctx = cpu_ctx(threads);
    let mut group = c.benchmark_group(format!(
        "tenferro_einsum_cpu/batched_rightmost_batch/threads_{threads}"
    ));

    for &batch in BATCHES {
        for &n in BATCHED_SMALL_SIZES {
            let a = eager(&ctx, f64_tensor(vec![n, n, batch], 1));
            let b = eager(&ctx, f64_tensor(vec![n, n, batch], 2));
            let params = format!("n_{n}_batch_{batch}");
            group.bench_function(BenchmarkId::new("f64_ikb_knb_to_inb", params), |bench| {
                bench.iter(|| {
                    let out = einsum(&[black_box(&a), black_box(&b)], "ikb,knb->inb")
                        .expect("batched einsum should succeed");
                    consume_f64(&out);
                });
            });
        }
    }

    group.finish();
}

fn bench_einsum_patterns(c: &mut Criterion) {
    let threads = bench_threads();
    let ctx = cpu_ctx(threads);
    let mut group = c.benchmark_group(format!("tenferro_einsum_cpu/patterns/threads_{threads}"));

    let a = eager(&ctx, f64_tensor(vec![64, 64], 1));
    let b = eager(&ctx, f64_tensor(vec![64, 64], 2));
    let c_tensor = eager(&ctx, f64_tensor(vec![64, 64], 3));
    group.bench_function("f64_binary_ij_jk_to_ik", |bench| {
        bench.iter(|| {
            let out = einsum(&[black_box(&a), black_box(&b)], "ij,jk->ik")
                .expect("binary einsum should succeed");
            consume_f64(&out);
        });
    });
    group.bench_function("f64_chain_ij_jk_kl_to_il", |bench| {
        bench.iter(|| {
            let out = einsum(
                &[black_box(&a), black_box(&b), black_box(&c_tensor)],
                "ij,jk,kl->il",
            )
            .expect("chain einsum should succeed");
            consume_f64(&out);
        });
    });

    let x = eager(&ctx, f64_tensor(vec![8, 16, 8], 4));
    let y = eager(&ctx, f64_tensor(vec![16, 8, 8], 5));
    group.bench_function("f64_multiedge_ijk_jkl_to_il", |bench| {
        bench.iter(|| {
            let out = einsum(&[black_box(&x), black_box(&y)], "ijk,jkl->il")
                .expect("multi-edge einsum should succeed");
            consume_f64(&out);
        });
    });

    let a = eager(&ctx, c64_tensor(vec![32, 32], 6));
    let b = eager(&ctx, c64_tensor(vec![32, 32], 7));
    group.bench_function("c64_binary_ij_jk_to_ik", |bench| {
        bench.iter(|| {
            let out = einsum(&[black_box(&a), black_box(&b)], "ij,jk->ik")
                .expect("c64 binary einsum should succeed");
            consume_c64(&out);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_batched_einsum, bench_einsum_patterns);
criterion_main!(benches);
