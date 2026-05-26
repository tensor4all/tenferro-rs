use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tenferro_tensor::{
    buffer_pool::BufferPool,
    cpu::{CpuBackend, CpuContext},
    TensorBackend,
};

fn bench_cpu_context_entry_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_context_entry_overhead/one_thread");

    group.bench_function("ctx_install_inline_empty", |b| {
        let ctx = CpuContext::with_threads(1);
        b.iter(|| ctx.install(|| black_box(1usize)));
    });

    group.bench_function("backend_install_inline_empty", |b| {
        let backend = CpuBackend::with_threads(1);
        b.iter(|| backend.install(|| black_box(1usize)));
    });

    group.bench_function("buffer_pool_take_restore_only", |b| {
        let mut buffers = BufferPool::new();
        b.iter(|| {
            let taken = std::mem::take(black_box(&mut buffers));
            buffers = black_box(taken);
        });
    });

    group.bench_function("ctx_install_inline_with_buffer_pool", |b| {
        let ctx = CpuContext::with_threads(1);
        let mut buffers = BufferPool::new();
        b.iter(|| {
            let mut taken = std::mem::take(black_box(&mut buffers));
            let (result, returned) = ctx.install(|| {
                black_box(&mut taken);
                (black_box(1usize), taken)
            });
            buffers = black_box(returned);
            black_box(result);
        });
    });

    group.bench_function("ctx_install_inline_with_buffer_pool_and_cache", |b| {
        let ctx = CpuContext::with_threads(1);
        let mut buffers = BufferPool::new();
        let mut cache = <CpuBackend as TensorBackend>::RuntimeCache::default();
        b.iter(|| {
            let mut taken = std::mem::take(black_box(&mut buffers));
            let (result, returned) = ctx.install(|| {
                black_box(&mut taken);
                black_box(&mut cache);
                (black_box(1usize), taken)
            });
            buffers = black_box(returned);
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_cpu_context_entry_overhead);
criterion_main!(benches);
