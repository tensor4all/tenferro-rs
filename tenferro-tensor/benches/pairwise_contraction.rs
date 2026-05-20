use std::env;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;
use tenferro_tensor::{cpu::CpuBackend, DotGeneralConfig, Tensor, TensorBackend};

const PHYS_DIM: usize = 2;
const CHIS: &[usize] = &[1, 2, 4, 8, 16, 32];

#[derive(Clone, Copy)]
enum PairwiseCase {
    FirstSite,
    EnvBra,
    TmpKet,
    SiteUpdate,
}

impl PairwiseCase {
    fn name(self) -> &'static str {
        match self {
            Self::FirstSite => "first_site",
            Self::EnvBra => "env_bra",
            Self::TmpKet => "tmp_ket",
            Self::SiteUpdate => "site_update",
        }
    }
}

struct Fixtures {
    bra_first: Tensor,
    ket_first: Tensor,
    env: Tensor,
    bra_bulk: Tensor,
    tmp: Tensor,
    ket_bulk: Tensor,
}

struct Configs {
    first_site: DotGeneralConfig,
    env_bra: DotGeneralConfig,
    tmp_ket: DotGeneralConfig,
}

impl Configs {
    fn new() -> Self {
        Self {
            first_site: DotGeneralConfig {
                lhs_contracting_dims: vec![0],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
            env_bra: DotGeneralConfig {
                lhs_contracting_dims: vec![0],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
            tmp_ket: DotGeneralConfig {
                lhs_contracting_dims: vec![0, 1],
                rhs_contracting_dims: vec![0, 1],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        }
    }
}

fn complex_tensor(shape: &[usize], seed: usize) -> Tensor {
    let len = shape.iter().product();
    let data = (0..len)
        .map(|idx| {
            let real = ((idx * 17 + seed * 13 + 3) % 97) as f64 / 97.0 - 0.5;
            let imag = ((idx * 29 + seed * 7 + 5) % 89) as f64 / 89.0 - 0.5;
            Complex64::new(real, imag)
        })
        .collect();
    Tensor::from_vec(shape.to_vec(), data)
}

fn build_fixtures(phys_dim: usize, chi: usize) -> Fixtures {
    Fixtures {
        bra_first: complex_tensor(&[phys_dim, chi], 1),
        ket_first: complex_tensor(&[phys_dim, chi], 2),
        env: complex_tensor(&[chi, chi], 3),
        bra_bulk: complex_tensor(&[chi, phys_dim, chi], 4),
        tmp: complex_tensor(&[chi, phys_dim, chi], 5),
        ket_bulk: complex_tensor(&[chi, phys_dim, chi], 6),
    }
}

fn benchmark_threads() -> usize {
    env::var("TENFERRO_PAIRWISE_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or(1)
}

fn run_case_normal(
    backend: &mut CpuBackend,
    fixtures: &Fixtures,
    configs: &Configs,
    case: PairwiseCase,
) -> Tensor {
    match case {
        PairwiseCase::FirstSite => backend
            .dot_general_with_conj(
                &fixtures.bra_first,
                &fixtures.ket_first,
                &configs.first_site,
                true,
                false,
            )
            .expect("first-site contraction should succeed"),
        PairwiseCase::EnvBra => backend
            .dot_general_with_conj(
                &fixtures.env,
                &fixtures.bra_bulk,
                &configs.env_bra,
                false,
                true,
            )
            .expect("environment-bra contraction should succeed"),
        PairwiseCase::TmpKet => backend
            .dot_general_with_conj(
                &fixtures.tmp,
                &fixtures.ket_bulk,
                &configs.tmp_ket,
                false,
                false,
            )
            .expect("temporary-ket contraction should succeed"),
        PairwiseCase::SiteUpdate => {
            let tmp = backend
                .dot_general_with_conj(
                    &fixtures.env,
                    &fixtures.bra_bulk,
                    &configs.env_bra,
                    false,
                    true,
                )
                .expect("environment-bra contraction should succeed");
            backend
                .dot_general_with_conj(&tmp, &fixtures.ket_bulk, &configs.tmp_ket, false, false)
                .expect("temporary-ket contraction should succeed")
        }
    }
}

fn run_case_cached(
    backend: &mut CpuBackend,
    cache: &mut <CpuBackend as TensorBackend>::RuntimeCache,
    fixtures: &Fixtures,
    configs: &Configs,
    case: PairwiseCase,
) -> Tensor {
    match case {
        PairwiseCase::FirstSite => backend
            .dot_general_with_conj_cached(
                cache,
                Some(0),
                &fixtures.bra_first,
                &fixtures.ket_first,
                &configs.first_site,
                true,
                false,
            )
            .expect("first-site contraction should succeed"),
        PairwiseCase::EnvBra => backend
            .dot_general_with_conj_cached(
                cache,
                Some(0),
                &fixtures.env,
                &fixtures.bra_bulk,
                &configs.env_bra,
                false,
                true,
            )
            .expect("environment-bra contraction should succeed"),
        PairwiseCase::TmpKet => backend
            .dot_general_with_conj_cached(
                cache,
                Some(0),
                &fixtures.tmp,
                &fixtures.ket_bulk,
                &configs.tmp_ket,
                false,
                false,
            )
            .expect("temporary-ket contraction should succeed"),
        PairwiseCase::SiteUpdate => {
            let tmp = backend
                .dot_general_with_conj_cached(
                    cache,
                    Some(0),
                    &fixtures.env,
                    &fixtures.bra_bulk,
                    &configs.env_bra,
                    false,
                    true,
                )
                .expect("environment-bra contraction should succeed");
            backend
                .dot_general_with_conj_cached(
                    cache,
                    Some(1),
                    &tmp,
                    &fixtures.ket_bulk,
                    &configs.tmp_ket,
                    false,
                    false,
                )
                .expect("temporary-ket contraction should succeed")
        }
    }
}

fn bench_pairwise_contraction(c: &mut Criterion) {
    let cases = [
        PairwiseCase::FirstSite,
        PairwiseCase::EnvBra,
        PairwiseCase::TmpKet,
        PairwiseCase::SiteUpdate,
    ];
    let configs = Configs::new();
    let threads = benchmark_threads();
    let group_name = if threads == 1 {
        "pairwise_contraction/c64/one_thread".to_string()
    } else {
        format!("pairwise_contraction/c64/threads_{threads}")
    };
    let mut group = c.benchmark_group(group_name);

    for &chi in CHIS {
        let fixtures = build_fixtures(PHYS_DIM, chi);
        let params = format!("chi_{chi}_d_{PHYS_DIM}");

        for &case in &cases {
            let case_params = format!("{}/{}", case.name(), params);

            group.bench_function(BenchmarkId::new("normal_per_call", &case_params), |b| {
                let mut backend = CpuBackend::with_threads(threads);
                b.iter(|| {
                    let output = run_case_normal(
                        black_box(&mut backend),
                        black_box(&fixtures),
                        black_box(&configs),
                        black_box(case),
                    );
                    black_box(output.shape().to_vec());
                });
            });

            group.bench_function(
                BenchmarkId::new("cached_analysis_per_call", &case_params),
                |b| {
                    let mut backend = CpuBackend::with_threads(threads);
                    let mut cache = <CpuBackend as TensorBackend>::RuntimeCache::default();
                    b.iter(|| {
                        let output = run_case_cached(
                            black_box(&mut backend),
                            black_box(&mut cache),
                            black_box(&fixtures),
                            black_box(&configs),
                            black_box(case),
                        );
                        black_box(output.shape().to_vec());
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_pairwise_contraction);
criterion_main!(benches);
