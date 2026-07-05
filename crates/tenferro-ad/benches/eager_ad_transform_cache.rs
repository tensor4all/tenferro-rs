use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;

const LEN: usize = 1024;

struct Fixture {
    ctx: Arc<EagerRuntime>,
    x: EagerTensor,
    loss: EagerTensor,
}

fn fixture() -> Fixture {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let data = (0..LEN)
        .map(|index| (index as f64 + 1.0) / LEN as f64)
        .collect::<Vec<_>>();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![LEN], data).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let x2 = x.mul(&x).unwrap();
    let x3 = x2.mul(&x).unwrap();
    let loss = x3.reduce_sum(&[0]).unwrap();
    Fixture { ctx, x, loss }
}

fn run_backward(fixture: &Fixture) {
    fixture.ctx.clear_grads().unwrap();
    fixture.loss.backward().unwrap();
    let grad = fixture.x.grad().unwrap().unwrap();
    black_box(grad.as_slice::<f64>().unwrap()[0]);
}

fn bench_eager_ad_transform_cache(c: &mut Criterion) {
    let cold = fixture();
    let warm = fixture();
    run_backward(&warm);
    assert!(warm.ctx.cache_stats().unwrap().ad_transforms.entries > 0);

    let mut group = c.benchmark_group("eager_ad_transform_cache/same_tape");
    group.bench_function("cold_clear_cache_each_backward", |bench| {
        bench.iter(|| {
            cold.ctx.clear_caches().unwrap();
            run_backward(black_box(&cold));
        });
    });
    group.bench_function("warm_reuse_cached_linearization", |bench| {
        bench.iter(|| run_backward(black_box(&warm)));
    });
    group.finish();
}

criterion_group!(benches, bench_eager_ad_transform_cache);
criterion_main!(benches);
