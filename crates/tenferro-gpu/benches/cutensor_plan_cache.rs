use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_gpu::{
    cuda::gpu_available, cuda::upload_tensor, cuda::CudaBackend, cuda::CudaDeviceId,
};
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorDot, TensorScalar};

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    }
}

fn matrix<T>(rows: usize, cols: usize) -> Tensor
where
    T: TensorScalar + From<f32>,
{
    let len = rows * cols;
    let data = (0..len)
        .map(|index| T::from(((index % 97) as f32 + 1.0) * 0.001))
        .collect();
    Tensor::from_vec_col_major(vec![rows, cols], data).unwrap()
}

fn bench_case<T>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
    rows: usize,
    inner: usize,
    cols: usize,
) where
    T: TensorScalar + From<f32> + 'static,
{
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let lhs = matrix::<T>(rows, inner);
    let rhs = matrix::<T>(inner, cols);
    let lhs = upload_tensor(backend.runtime(), &lhs).unwrap();
    let rhs = upload_tensor(backend.runtime(), &rhs).unwrap();
    let config = matmul_config();

    let _ = backend.dot_general(&lhs, &rhs, &config).unwrap();
    backend.runtime().synchronize().unwrap();

    group.bench_function(
        BenchmarkId::new(format!("{label}/cold_clear_each_iter"), rows),
        |bench| {
            bench.iter(|| {
                backend.clear_cuda_extension_cache().unwrap();
                let out = backend
                    .dot_general(black_box(&lhs), black_box(&rhs), black_box(&config))
                    .unwrap();
                backend.runtime().synchronize().unwrap();
                black_box(out);
            });
        },
    );

    let _ = backend.dot_general(&lhs, &rhs, &config).unwrap();
    backend.runtime().synchronize().unwrap();
    group.bench_function(
        BenchmarkId::new(format!("{label}/warm_cache"), rows),
        |bench| {
            bench.iter(|| {
                let out = backend
                    .dot_general(black_box(&lhs), black_box(&rhs), black_box(&config))
                    .unwrap();
                backend.runtime().synchronize().unwrap();
                black_box(out);
            });
        },
    );
}

fn cutensor_plan_cache(c: &mut Criterion) {
    if !gpu_available() {
        eprintln!("skipping cuTENSOR plan-cache benchmark: no CUDA device available");
        return;
    }

    let mut group = c.benchmark_group("cutensor_plan_cache");
    for &(rows, inner, cols) in &[(64, 64, 64), (256, 256, 256)] {
        bench_case::<f32>(&mut group, "f32", rows, inner, cols);
        bench_case::<f64>(&mut group, "f64", rows, inner, cols);
    }
    group.finish();
}

criterion_group!(benches, cutensor_plan_cache);
criterion_main!(benches);
