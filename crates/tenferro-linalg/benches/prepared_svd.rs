use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_linalg::{LinalgBackend, PreparedSvdBackendExt, SvdOptions, SvdOutputWrites};
use tenferro_tensor::{DType, Tensor, TensorRead, TensorView, TensorWrite, TypedTensor};

const SHAPES: &[[usize; 2]] = &[
    [4, 4],
    [8, 4],
    [4, 8],
    [16, 16],
    [32, 16],
    [16, 32],
    [64, 64],
];

fn matrix([m, n]: [usize; 2]) -> TypedTensor<f64> {
    let data = (0..n)
        .flat_map(|col| {
            (0..m).map(move |row| {
                let diagonal = if row == col { 2.0 } else { 0.0 };
                diagonal + ((row * 17 + col * 31 + 1) % 97) as f64 / 97.0
            })
        })
        .collect();
    TypedTensor::from_vec_col_major(vec![m, n], data).unwrap()
}

fn outputs([m, n]: [usize; 2]) -> (Tensor, Tensor, Tensor) {
    let k = m.min(n);
    (
        Tensor::from_vec_col_major(vec![m, k], vec![0.0_f64; m * k]).unwrap(),
        Tensor::from_vec_col_major(vec![k], vec![0.0_f64; k]).unwrap(),
        Tensor::from_vec_col_major(vec![k, n], vec![0.0_f64; k * n]).unwrap(),
    )
}

fn bench_prepared_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("faer_f64_compact_svd");
    for &shape in SHAPES {
        let input = matrix(shape);

        let mut owned_backend = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap();
        let owned_id = BenchmarkId::new("owned_svd_read", format!("{}x{}", shape[0], shape[1]));
        group.bench_with_input(owned_id, &shape, |b, _| {
            b.iter(|| {
                let result = owned_backend
                    .svd_read(TensorView::F64(input.as_view()))
                    .unwrap();
                black_box(result);
            });
        });

        let mut prepared_backend =
            CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap();
        let plan = prepared_backend
            .prepare_svd(shape, DType::F64, SvdOptions::default())
            .unwrap();
        let mut workspace = plan.allocate_workspace(&mut prepared_backend).unwrap();
        let (mut u, mut s, mut vt) = outputs(shape);
        plan.execute_into(
            &mut prepared_backend,
            &mut workspace,
            TensorRead::from_view(TensorView::F64(input.as_view())),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut u),
                TensorWrite::from_tensor(&mut s),
                TensorWrite::from_tensor(&mut vt),
            ),
        )
        .unwrap();
        let prepared_id = BenchmarkId::new(
            "prepared_execute_into",
            format!("{}x{}", shape[0], shape[1]),
        );
        group.bench_with_input(prepared_id, &shape, |b, _| {
            b.iter(|| {
                plan.execute_into(
                    &mut prepared_backend,
                    &mut workspace,
                    TensorRead::from_view(TensorView::F64(input.as_view())),
                    SvdOutputWrites::new(
                        TensorWrite::from_tensor(&mut u),
                        TensorWrite::from_tensor(&mut s),
                        TensorWrite::from_tensor(&mut vt),
                    ),
                )
                .unwrap();
                black_box((&u, &s, &vt));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_prepared_svd);
criterion_main!(benches);
