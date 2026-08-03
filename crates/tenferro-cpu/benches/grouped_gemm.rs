use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, ContractionScalar, DotGeneralAccumulation,
    DotGeneralConfig, Tensor, TensorDot, TensorRead, TensorView, TensorWrite,
};

struct GroupedFixture {
    lhs: Tensor,
    rhs: Tensor,
    out: Tensor,
    jobs: Vec<GroupedGemmJob>,
}

fn deterministic_data(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|idx| ((idx * 17 + seed * 31 + 7) % 101) as f64 / 101.0 - 0.5)
        .collect()
}

fn fixture_from_shapes(shapes: &[(usize, usize, usize)]) -> GroupedFixture {
    let mut lhs_len = 0usize;
    let mut rhs_len = 0usize;
    let mut out_len = 0usize;
    let mut jobs = Vec::with_capacity(shapes.len());
    for &(rows, contracted, cols) in shapes {
        jobs.push(GroupedGemmJob::new(
            out_len, lhs_len, rhs_len, rows, contracted, cols,
        ));
        lhs_len += rows * contracted;
        rhs_len += contracted * cols;
        out_len += rows * cols;
    }
    GroupedFixture {
        lhs: Tensor::from_vec_col_major(vec![lhs_len], deterministic_data(lhs_len, 1)).unwrap(),
        rhs: Tensor::from_vec_col_major(vec![rhs_len], deterministic_data(rhs_len, 2)).unwrap(),
        out: Tensor::from_vec_col_major(vec![out_len], deterministic_data(out_len, 3)).unwrap(),
        jobs,
    }
}

fn grouped_config(fixture: &GroupedFixture, beta: f64) -> GroupedGemmConfig<'_> {
    GroupedGemmConfig::new(
        &fixture.jobs,
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(1.0),
            beta: ContractionScalar::F64(beta),
        },
    )
}

fn run_grouped_into(
    backend: &mut CpuBackend,
    cache: &mut <CpuBackend as BackendRuntimeCache>::RuntimeCache,
    fixture: &GroupedFixture,
    config: &GroupedGemmConfig<'_>,
    out: &mut Tensor,
) {
    BackendCachedDot::grouped_gemm_cached(
        backend,
        cache,
        Some(0),
        TensorRead::from_tensor(&fixture.lhs),
        TensorRead::from_tensor(&fixture.rhs),
        config,
        TensorWrite::from_tensor(out),
    )
    .unwrap();
}

fn f64_view(tensor: &Tensor) -> TensorView<'_> {
    match tensor {
        Tensor::F64(tensor) => TensorView::F64(tensor.as_view()),
        _ => unreachable!("grouped GEMM benchmark fixtures are F64"),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_grouped_views_into(
    backend: &mut CpuBackend,
    cache: &mut <CpuBackend as BackendRuntimeCache>::RuntimeCache,
    lhs: &TensorView<'_>,
    rhs: &TensorView<'_>,
    config: &GroupedGemmConfig<'_>,
    out: &mut Tensor,
) {
    BackendCachedDot::grouped_gemm_cached(
        backend,
        cache,
        Some(0),
        TensorRead::from_view(lhs.clone()),
        TensorRead::from_view(rhs.clone()),
        config,
        TensorWrite::from_tensor(out),
    )
    .unwrap();
}

fn run_grouped(backend: &mut CpuBackend, fixture: &GroupedFixture) -> Tensor {
    let mut cache = <CpuBackend as BackendRuntimeCache>::RuntimeCache::default();
    let mut out = fixture.out.duplicate().unwrap();
    let config = grouped_config(fixture, 1.0);
    run_grouped_into(backend, &mut cache, fixture, &config, &mut out);
    out
}

fn run_sequential(backend: &mut CpuBackend, fixture: &GroupedFixture) -> Tensor {
    let mut out = fixture.out.duplicate().unwrap();
    let dot_config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: Vec::new(),
        rhs_batch_dims: Vec::new(),
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: ContractionScalar::F64(1.0),
        beta: ContractionScalar::F64(1.0),
    };
    let lhs = fixture.lhs.as_slice::<f64>().unwrap();
    let rhs = fixture.rhs.as_slice::<f64>().unwrap();
    for job in &fixture.jobs {
        let lhs_view = tenferro_tensor::TypedTensorView::from_slice(
            vec![job.rows(), job.contracted()],
            vec![1, job.rows() as isize],
            job.lhs_offset() as isize,
            lhs,
        )
        .unwrap();
        let rhs_view = tenferro_tensor::TypedTensorView::from_slice(
            vec![job.contracted(), job.cols()],
            vec![1, job.contracted() as isize],
            job.rhs_offset() as isize,
            rhs,
        )
        .unwrap();
        let mut out_view = match &mut out {
            Tensor::F64(tensor) => tensor.as_view_mut(),
            _ => unreachable!("fixture output is F64"),
        };
        let out_storage = out_view.host_storage_mut().unwrap();
        let out_matrix = tenferro_tensor::TypedTensorViewMut::from_slice(
            vec![job.rows(), job.cols()],
            vec![1, job.rows() as isize],
            job.out_offset() as isize,
            out_storage,
        )
        .unwrap();
        backend
            .dot_general_read_into_accum(
                TensorRead::from_view(tenferro_tensor::TensorView::F64(lhs_view)),
                TensorRead::from_view(tenferro_tensor::TensorView::F64(rhs_view)),
                &dot_config,
                accumulation,
                TensorWrite::from_view(tenferro_tensor::TensorViewMut::F64(out_matrix)),
            )
            .unwrap();
    }
    out
}

fn bench_grouped_gemm(c: &mut Criterion) {
    let cases = [
        ("uniform_small", vec![(8, 8, 8); 64]),
        (
            "mixed_large_small",
            std::iter::once((64, 64, 64))
                .chain(std::iter::repeat_n((8, 8, 8), 31))
                .collect(),
        ),
        ("medium_blocks", vec![(32, 32, 32); 16]),
    ];
    let mut group = c.benchmark_group("grouped_gemm");
    for (name, shapes) in cases {
        let fixture = fixture_from_shapes(&shapes);
        group.bench_with_input(BenchmarkId::new("grouped", name), &fixture, |b, fixture| {
            let mut backend = CpuBackend::new();
            b.iter(|| black_box(run_grouped(&mut backend, fixture)));
        });
        group.bench_with_input(
            BenchmarkId::new("sequential_n_call", name),
            &fixture,
            |b, fixture| {
                let mut backend = CpuBackend::new();
                b.iter(|| black_box(run_sequential(&mut backend, fixture)));
            },
        );
    }
    group.finish();
}

fn bench_grouped_gemm_steady_state(c: &mut Criterion) {
    const PROFILE_WARMUP_ITERATIONS: usize = 2_000;
    const CALLS_PER_ITERATION: usize = 6;

    let fixture = fixture_from_shapes(&[(4, 4, 4); 8]);
    let config = grouped_config(&fixture, 0.0);
    let lhs = f64_view(&fixture.lhs);
    let rhs = f64_view(&fixture.rhs);
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let mut cache = <CpuBackend as BackendRuntimeCache>::RuntimeCache::default();
    let mut out = fixture.out.duplicate().unwrap();

    // Match the steady-state profile that motivated issue #1385: warm the
    // backend, then issue six small grouped-GEMM calls per measured iteration.
    for _ in 0..PROFILE_WARMUP_ITERATIONS {
        for _ in 0..CALLS_PER_ITERATION {
            run_grouped_views_into(&mut backend, &mut cache, &lhs, &rhs, &config, &mut out);
        }
    }

    c.bench_function("grouped_gemm/steady_state/six_calls_8x4x4", |b| {
        b.iter(|| {
            for _ in 0..CALLS_PER_ITERATION {
                run_grouped_views_into(
                    black_box(&mut backend),
                    black_box(&mut cache),
                    black_box(&lhs),
                    black_box(&rhs),
                    black_box(&config),
                    black_box(&mut out),
                );
            }
            black_box(out.as_slice::<f64>().unwrap()[0]);
        });
    });
}

criterion_group!(benches, bench_grouped_gemm, bench_grouped_gemm_steady_state);
criterion_main!(benches);
