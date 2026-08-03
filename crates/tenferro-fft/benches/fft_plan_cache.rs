use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_complex::Complex64;
use tenferro_cpu::{with_cpu_exec_session, CpuBackend, CpuExecSession};
use tenferro_fft::{FftExecutor, FftNorm, TensorFftExt};
use tenferro_tensor::{BackendSessionHost, Tensor};

const CASES: &[(usize, usize)] = &[(256, 1), (1024, 16)];

fn with_cpu_session<R>(
    backend: &mut CpuBackend,
    f: impl for<'a> FnOnce(&'a mut CpuExecSession<'a>) -> R + Send,
) -> R
where
    R: Send,
{
    backend.with_backend_session(|session| {
        with_cpu_exec_session(session, f).expect("CpuBackend must expose a CPU execution session")
    })
}

fn c64_input(fft_len: usize, lanes: usize) -> Tensor {
    let len = fft_len * lanes;
    let values = (0..len)
        .map(|index| {
            let real = ((index * 17 + 3) % 257) as f64 / 257.0 - 0.5;
            let imag = ((index * 29 + 5) % 263) as f64 / 263.0 - 0.5;
            Complex64::new(real, imag)
        })
        .collect::<Vec<_>>();
    Tensor::from_vec_col_major(vec![fft_len, lanes], values).unwrap()
}

fn f64_input(fft_len: usize, lanes: usize) -> Tensor {
    let len = fft_len * lanes;
    let values = (0..len)
        .map(|index| ((index * 31 + 7) % 251) as f64 / 251.0 - 0.5)
        .collect::<Vec<_>>();
    Tensor::from_vec_col_major(vec![fft_len, lanes], values).unwrap()
}

fn bench_c64_fft_plan_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_plan_cache/c64_fft_axis0");
    for &(fft_len, lanes) in CASES {
        let input = c64_input(fft_len, lanes);
        let id = format!("{fft_len}x{lanes}");
        group.throughput(Throughput::Elements((fft_len * lanes) as u64));

        group.bench_function(BenchmarkId::new("direct_one_shot", &id), |bench| {
            let mut backend = CpuBackend::new();
            bench.iter(|| {
                let output = with_cpu_session(&mut backend, |session| {
                    black_box(&input).fft(None, 0, FftNorm::Backward, session)
                })
                .unwrap();
                black_box(output);
            });
        });

        group.bench_function(BenchmarkId::new("executor_warm_cache", &id), |bench| {
            let mut backend = CpuBackend::new();
            let mut executor = FftExecutor::default();
            with_cpu_session(&mut backend, |session| {
                executor.fft(&input, None, 0, FftNorm::Backward, session)
            })
            .unwrap();
            assert_eq!(executor.cache_stats().entries, 1);

            bench.iter(|| {
                let output = with_cpu_session(&mut backend, |session| {
                    executor.fft(black_box(&input), None, 0, FftNorm::Backward, session)
                })
                .unwrap();
                black_box(output);
            });
        });
    }
    group.finish();
}

fn bench_f64_rfft_plan_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_plan_cache/f64_rfft_axis0");
    for &(fft_len, lanes) in CASES {
        let input = f64_input(fft_len, lanes);
        let id = format!("{fft_len}x{lanes}");
        group.throughput(Throughput::Elements((fft_len * lanes) as u64));

        group.bench_function(BenchmarkId::new("direct_one_shot", &id), |bench| {
            let mut backend = CpuBackend::new();
            bench.iter(|| {
                let output = with_cpu_session(&mut backend, |session| {
                    black_box(&input).rfft(None, 0, FftNorm::Backward, session)
                })
                .unwrap();
                black_box(output);
            });
        });

        group.bench_function(BenchmarkId::new("executor_warm_cache", &id), |bench| {
            let mut backend = CpuBackend::new();
            let mut executor = FftExecutor::default();
            with_cpu_session(&mut backend, |session| {
                executor.rfft(&input, None, 0, FftNorm::Backward, session)
            })
            .unwrap();
            assert_eq!(executor.cache_stats().entries, 1);

            bench.iter(|| {
                let output = with_cpu_session(&mut backend, |session| {
                    executor.rfft(black_box(&input), None, 0, FftNorm::Backward, session)
                })
                .unwrap();
                black_box(output);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_c64_fft_plan_cache, bench_f64_rfft_plan_cache);
criterion_main!(benches);
