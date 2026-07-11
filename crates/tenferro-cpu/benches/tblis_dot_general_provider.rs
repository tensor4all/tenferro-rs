use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, BenchmarkGroup, BenchmarkId,
    Criterion,
};
#[cfg(feature = "cpu-tblis")]
use std::env;

#[cfg(feature = "cpu-tblis")]
use num_complex::Complex64;
#[cfg(feature = "cpu-tblis")]
use tenferro_cpu::{CpuBackend, CpuBackendKind, DotGeneralProvider};
#[cfg(feature = "cpu-tblis")]
use tenferro_tensor::{
    DotGeneralConfig, Tensor, TensorDot, TensorRead, TensorView, TypedTensorView,
};

#[cfg(feature = "cpu-tblis")]
const SIZES: &[usize] = &[32, 64, 128];
#[cfg(feature = "cpu-tblis")]
const HIGHER_RANK_NS: &[usize] = &[4, 8];

#[cfg(feature = "cpu-tblis")]
fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

#[cfg(feature = "cpu-tblis")]
fn rank4_contract_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![2, 3],
        rhs_contracting_dims: vec![0, 1],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

#[cfg(feature = "cpu-tblis")]
fn rank4_mixed_contract_axes_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1, 3],
        rhs_contracting_dims: vec![2, 1],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

#[cfg(feature = "cpu-tblis")]
fn rank5_batched_mixed_contract_axes_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![2, 4],
        rhs_contracting_dims: vec![3, 2],
        lhs_batch_dims: vec![0],
        rhs_batch_dims: vec![0],
    }
}

#[cfg(feature = "cpu-tblis")]
fn parse_size_list_env(var: &str, default: &[usize]) -> Vec<usize> {
    env::var(var)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|item| item.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| default.to_vec())
}

#[cfg(feature = "cpu-tblis")]
fn real_tensor(shape: &[usize], seed: usize) -> Tensor {
    let len = shape.iter().product();
    let data = (0..len)
        .map(|idx| ((idx * 17 + seed * 11 + 5) % 101) as f64 / 101.0 - 0.5)
        .collect();
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

#[cfg(feature = "cpu-tblis")]
fn real_data(shape: &[usize], seed: usize) -> Vec<f64> {
    let len = shape.iter().product();
    (0..len)
        .map(|idx| ((idx * 17 + seed * 11 + 5) % 101) as f64 / 101.0 - 0.5)
        .collect()
}

#[cfg(feature = "cpu-tblis")]
fn real_matrix(rows: usize, cols: usize, seed: usize) -> Tensor {
    real_tensor(&[rows, cols], seed)
}

#[cfg(feature = "cpu-tblis")]
fn complex_matrix(rows: usize, cols: usize, seed: usize) -> Tensor {
    let data = (0..rows * cols)
        .map(|idx| {
            let real = ((idx * 17 + seed * 11 + 5) % 101) as f64 / 101.0 - 0.5;
            let imag = ((idx * 29 + seed * 7 + 13) % 103) as f64 / 103.0 - 0.5;
            Complex64::new(real, imag)
        })
        .collect();
    Tensor::from_vec_col_major(vec![rows, cols], data).unwrap()
}

#[cfg(feature = "cpu-tblis")]
fn tblis_runtime_available(config: &DotGeneralConfig) -> bool {
    let lhs = real_matrix(2, 2, 1);
    let rhs = real_matrix(2, 2, 2);
    let mut backend = CpuBackend::with_kind(CpuBackendKind::default_compiled())
        .unwrap()
        .with_dot_general_provider(DotGeneralProvider::TblisRequired);
    match backend.dot_general(&lhs, &rhs, config) {
        Ok(_) => true,
        Err(err) => {
            eprintln!("skipping tblis_dot_general_provider: {err}");
            false
        }
    }
}

#[cfg(feature = "cpu-tblis")]
fn bench_owned_dot(
    group: &mut BenchmarkGroup<'_, WallTime>,
    provider: &str,
    kind: CpuBackendKind,
    dot_general_provider: DotGeneralProvider,
    case: &str,
    params: impl std::fmt::Display,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
) {
    group.bench_function(
        BenchmarkId::new(format!("{provider}_{case}"), params),
        |b| {
            let mut backend = CpuBackend::with_kind(kind)
                .unwrap()
                .with_dot_general_provider(dot_general_provider);
            b.iter(|| {
                let out = backend
                    .dot_general(black_box(lhs), black_box(rhs), black_box(config))
                    .unwrap();
                black_box(out.shape().to_vec());
            });
        },
    );
}

#[cfg(feature = "cpu-tblis")]
fn bench_owned_conj_dot(
    group: &mut BenchmarkGroup<'_, WallTime>,
    provider: &str,
    kind: CpuBackendKind,
    dot_general_provider: DotGeneralProvider,
    case: &str,
    params: impl std::fmt::Display,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
) {
    group.bench_function(
        BenchmarkId::new(format!("{provider}_{case}"), params),
        |b| {
            let mut backend = CpuBackend::with_kind(kind)
                .unwrap()
                .with_dot_general_provider(dot_general_provider);
            b.iter(|| {
                let out = backend
                    .dot_general_with_conj(
                        black_box(lhs),
                        black_box(rhs),
                        black_box(config),
                        true,
                        false,
                    )
                    .unwrap();
                black_box(out.shape().to_vec());
            });
        },
    );
}

#[cfg(feature = "cpu-tblis")]
fn bench_read_view_dot(
    group: &mut BenchmarkGroup<'_, WallTime>,
    provider: &str,
    kind: CpuBackendKind,
    dot_general_provider: DotGeneralProvider,
    case: &str,
    params: impl std::fmt::Display,
    lhs: StridedData<'_>,
    rhs: StridedData<'_>,
    config: &DotGeneralConfig,
) {
    group.bench_function(
        BenchmarkId::new(format!("{provider}_{case}"), params),
        |b| {
            let mut backend = CpuBackend::with_kind(kind)
                .unwrap()
                .with_dot_general_provider(dot_general_provider);
            b.iter(|| {
                let lhs_view =
                    TypedTensorView::from_slice(lhs.shape, lhs.strides, 0, black_box(lhs.data))
                        .unwrap();
                let rhs_view =
                    TypedTensorView::from_slice(rhs.shape, rhs.strides, 0, black_box(rhs.data))
                        .unwrap();
                let out = backend
                    .dot_general_read(
                        TensorRead::from_view(TensorView::F64(lhs_view)),
                        TensorRead::from_view(TensorView::F64(rhs_view)),
                        black_box(config),
                    )
                    .unwrap();
                black_box(out.shape().to_vec());
            });
        },
    );
}

#[cfg(feature = "cpu-tblis")]
#[derive(Clone, Copy)]
struct StridedData<'a> {
    shape: &'a [usize],
    strides: &'a [isize],
    data: &'a [f64],
}

#[cfg(feature = "cpu-tblis")]
fn bench_owned_all_providers(
    group: &mut BenchmarkGroup<'_, WallTime>,
    case: &str,
    params: impl std::fmt::Display + Clone,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
) {
    #[cfg(feature = "cpu-faer")]
    bench_owned_dot(
        group,
        "faer",
        CpuBackendKind::Faer,
        DotGeneralProvider::Base,
        case,
        params.clone(),
        lhs,
        rhs,
        config,
    );
    #[cfg(feature = "cpu-blas")]
    bench_owned_dot(
        group,
        "blas",
        CpuBackendKind::Blas,
        DotGeneralProvider::Base,
        case,
        params.clone(),
        lhs,
        rhs,
        config,
    );
    bench_owned_dot(
        group,
        "tblis",
        CpuBackendKind::default_compiled(),
        DotGeneralProvider::TblisIfAvailable,
        case,
        params,
        lhs,
        rhs,
        config,
    );
}

#[cfg(feature = "cpu-tblis")]
fn bench_read_view_all_providers(
    group: &mut BenchmarkGroup<'_, WallTime>,
    case: &str,
    params: impl std::fmt::Display + Clone,
    lhs: StridedData<'_>,
    rhs: StridedData<'_>,
    config: &DotGeneralConfig,
) {
    #[cfg(feature = "cpu-faer")]
    bench_read_view_dot(
        group,
        "faer",
        CpuBackendKind::Faer,
        DotGeneralProvider::Base,
        case,
        params.clone(),
        lhs,
        rhs,
        config,
    );
    #[cfg(feature = "cpu-blas")]
    bench_read_view_dot(
        group,
        "blas",
        CpuBackendKind::Blas,
        DotGeneralProvider::Base,
        case,
        params.clone(),
        lhs,
        rhs,
        config,
    );
    bench_read_view_dot(
        group,
        "tblis",
        CpuBackendKind::default_compiled(),
        DotGeneralProvider::TblisIfAvailable,
        case,
        params,
        lhs,
        rhs,
        config,
    );
}

#[cfg(feature = "cpu-tblis")]
fn bench_tblis_dot_general_provider(c: &mut Criterion) {
    let matmul_config = matmul_config();
    if !tblis_runtime_available(&matmul_config) {
        return;
    }

    let mut group = c.benchmark_group("tblis_dot_general_provider");
    let matmul_sizes = parse_size_list_env("TENFERRO_TBLIS_BENCH_MATMUL_SIZES", SIZES);
    let higher_rank_sizes =
        parse_size_list_env("TENFERRO_TBLIS_BENCH_HIGHER_RANK_NS", HIGHER_RANK_NS);

    for &n in &matmul_sizes {
        let real_lhs = real_matrix(n, n, 1);
        let real_rhs = real_matrix(n, n, 2);
        let complex_lhs = complex_matrix(n, n, 3);
        let complex_rhs = complex_matrix(n, n, 4);

        bench_owned_all_providers(
            &mut group,
            "f64_matrix_square_gemm",
            n,
            &real_lhs,
            &real_rhs,
            &matmul_config,
        );
        #[cfg(feature = "cpu-faer")]
        bench_owned_conj_dot(
            &mut group,
            "faer",
            CpuBackendKind::Faer,
            DotGeneralProvider::Base,
            "c64_matrix_square_gemm_lhs_conj",
            n,
            &complex_lhs,
            &complex_rhs,
            &matmul_config,
        );
        #[cfg(feature = "cpu-blas")]
        bench_owned_conj_dot(
            &mut group,
            "blas",
            CpuBackendKind::Blas,
            DotGeneralProvider::Base,
            "c64_matrix_square_gemm_lhs_conj",
            n,
            &complex_lhs,
            &complex_rhs,
            &matmul_config,
        );
        bench_owned_conj_dot(
            &mut group,
            "tblis",
            CpuBackendKind::default_compiled(),
            DotGeneralProvider::TblisIfAvailable,
            "c64_matrix_square_gemm_lhs_conj",
            n,
            &complex_lhs,
            &complex_rhs,
            &matmul_config,
        );
    }

    let rank4_config = rank4_contract_config();
    let rank4_mixed_config = rank4_mixed_contract_axes_config();
    let batched_mixed_config = rank5_batched_mixed_contract_axes_config();

    for &n in &higher_rank_sizes {
        let rank4_lhs = real_tensor(&[n, n, n, n], 11);
        let rank4_rhs = real_tensor(&[n, n, n, n], 12);
        bench_owned_all_providers(
            &mut group,
            "f64_rank4_packed_contract_axes",
            format!("n_{n}"),
            &rank4_lhs,
            &rank4_rhs,
            &rank4_config,
        );

        let mixed_lhs = real_tensor(&[n, n, n, n], 21);
        let mixed_rhs = real_tensor(&[n, n, n, n], 22);
        bench_owned_all_providers(
            &mut group,
            "f64_rank4_mixed_contract_axes",
            format!("n_{n}"),
            &mixed_lhs,
            &mixed_rhs,
            &rank4_mixed_config,
        );

        let batch = 4;
        let batched_lhs = real_tensor(&[batch, n, n, n, n], 31);
        let batched_rhs = real_tensor(&[batch, n, n, n, n], 32);
        bench_owned_all_providers(
            &mut group,
            "f64_rank5_batched_mixed_contract_axes",
            format!("batch_{batch}_n_{n}"),
            &batched_lhs,
            &batched_rhs,
            &batched_mixed_config,
        );
    }

    for &n in &higher_rank_sizes {
        let lhs_shape = [n, n, n, n];
        let rhs_shape = [n, n, n, n];
        let lhs_row_major_strides = [(n * n * n) as isize, (n * n) as isize, n as isize, 1];
        let rhs_row_major_strides = [(n * n * n) as isize, (n * n) as isize, n as isize, 1];
        let lhs_data = real_data(&lhs_shape, 41);
        let rhs_data = real_data(&rhs_shape, 42);
        bench_read_view_all_providers(
            &mut group,
            "f64_rank4_row_major_view_mixed_contract_axes",
            format!("n_{n}"),
            StridedData {
                shape: &lhs_shape,
                strides: &lhs_row_major_strides,
                data: &lhs_data,
            },
            StridedData {
                shape: &rhs_shape,
                strides: &rhs_row_major_strides,
                data: &rhs_data,
            },
            &rank4_mixed_config,
        );
    }

    group.finish();
}

#[cfg(not(feature = "cpu-tblis"))]
fn bench_tblis_dot_general_provider(_c: &mut Criterion) {}

criterion_group!(benches, bench_tblis_dot_general_provider);
criterion_main!(benches);
