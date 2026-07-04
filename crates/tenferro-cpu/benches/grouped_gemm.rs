use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, ContractionScalar, DotGeneralAccumulation,
    DotGeneralConfig, Tensor, TensorDot, TensorRead, TensorWrite,
};

#[derive(Clone)]
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

fn run_grouped(backend: &mut CpuBackend, fixture: &GroupedFixture) -> Tensor {
    let mut cache = <CpuBackend as BackendRuntimeCache>::RuntimeCache::default();
    let mut out = fixture.out.clone();
    let config = GroupedGemmConfig::new(
        &fixture.jobs,
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(1.0),
            beta: ContractionScalar::F64(1.0),
        },
    );
    BackendCachedDot::grouped_gemm_cached(
        backend,
        &mut cache,
        Some(0),
        TensorRead::from_tensor(&fixture.lhs),
        TensorRead::from_tensor(&fixture.rhs),
        &config,
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap();
    out
}

fn run_sequential(backend: &mut CpuBackend, fixture: &GroupedFixture) -> Tensor {
    let mut out = fixture.out.clone();
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

criterion_group!(benches, bench_grouped_gemm);
criterion_main!(benches);
