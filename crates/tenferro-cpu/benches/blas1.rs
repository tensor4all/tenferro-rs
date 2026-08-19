use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{BackendSessionHost, ContractionScalar, Tensor, TensorRead, TensorWrite};

fn bench_axpby(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_blas1_axpby");
    for threads in [1, 4] {
        for len in [1_024, 65_536] {
            let x = Tensor::from_vec_col_major(vec![len], vec![1.0_f64; len]).unwrap();
            let mut fused = Tensor::from_vec_col_major(vec![len], vec![2.0_f64; len]).unwrap();
            let x_ref = x.as_slice::<f64>().unwrap().to_vec();
            let mut manual = vec![2.0_f64; len];
            let mut backend = CpuBackend::with_threads(threads).unwrap();

            group.bench_with_input(
                BenchmarkId::new(format!("fused_threads_{threads}"), len),
                &len,
                |bench, _| {
                    bench.iter(|| {
                        backend
                            .with_backend_session(|session| {
                                session.axpby_read_into_accum(
                                    ContractionScalar::F64(black_box(0.5)),
                                    TensorRead::from_tensor(black_box(&x)),
                                    ContractionScalar::F64(black_box(0.5)),
                                    TensorWrite::from_tensor(black_box(&mut fused)),
                                )
                            })
                            .unwrap();
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("manual_threads_{threads}"), len),
                &len,
                |bench, _| {
                    bench.iter(|| {
                        for (dst, src) in manual.iter_mut().zip(&x_ref) {
                            *dst = black_box(0.5) * *src + black_box(0.5) * *dst;
                        }
                        black_box(&manual);
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_axpby);
criterion_main!(benches);
