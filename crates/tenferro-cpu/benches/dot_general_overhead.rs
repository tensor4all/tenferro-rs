use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_tensor::{DotGeneralConfig, Tensor};

/// Run `f` on `engine` (keeps the measured call shape of the former session entry).
fn run<R>(
    engine: &mut CpuEngine,
    f: impl FnOnce(&mut dyn tenferro_tensor::Engine) -> R,
) -> tenferro_tensor::Result<R> {
    Ok(f(engine))
}

/// An engine with `threads` workers (serial for one).
fn engine(threads: usize) -> CpuEngine {
    if threads <= 1 {
        CpuEngine::serial()
    } else {
        CpuEngine::new(
            tenferro_cpu::thread_pool_builder()
                .num_threads(threads)
                .build()
                .expect("benchmark pool"),
        )
    }
}

const L: usize = 32;
const PHYS_DIM: usize = 2;
const CHIS: &[usize] = &[4, 8, 16, 32, 64];

struct MpsFixture {
    bra_tensors: Vec<Tensor>,
    ket_tensors: Vec<Tensor>,
}

struct LocalPathConfigs {
    env_bra: DotGeneralConfig,
    tmp_ket: DotGeneralConfig,
}

impl LocalPathConfigs {
    fn new() -> Self {
        Self {
            env_bra: DotGeneralConfig {
                lhs_contracting_dims: [0].as_slice().into(),
                rhs_contracting_dims: [0].as_slice().into(),
                lhs_batch_dims: [].as_slice().into(),
                rhs_batch_dims: [].as_slice().into(),
            },
            tmp_ket: DotGeneralConfig {
                lhs_contracting_dims: [0, 1].as_slice().into(),
                rhs_contracting_dims: [0, 1].as_slice().into(),
                lhs_batch_dims: [].as_slice().into(),
                rhs_batch_dims: [].as_slice().into(),
            },
        }
    }
}

fn mps_shapes(sites: usize, phys_dim: usize, bond_dim: usize) -> Vec<Vec<usize>> {
    (0..sites)
        .map(|site| {
            let left = if site == 0 { 1 } else { bond_dim };
            let right = if site + 1 == sites { 1 } else { bond_dim };
            vec![left, phys_dim, right]
        })
        .collect()
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
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn build_mps_fixture(sites: usize, phys_dim: usize, bond_dim: usize) -> MpsFixture {
    let shapes = mps_shapes(sites, phys_dim, bond_dim);
    let bra_tensors = shapes
        .iter()
        .enumerate()
        .map(|(site, shape)| complex_tensor(shape, site + 1))
        .collect();
    let ket_tensors = shapes
        .iter()
        .enumerate()
        .map(|(site, shape)| complex_tensor(shape, sites + site + 1))
        .collect();

    MpsFixture {
        bra_tensors,
        ket_tensors,
    }
}

fn scalar_one() -> Tensor {
    Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(1.0, 0.0)]).unwrap()
}

fn inner_fresh_backend_cache(
    backend: &mut CpuEngine,
    bra: &[Tensor],
    ket: &[Tensor],
    configs: &LocalPathConfigs,
) -> Tensor {
    let mut env = scalar_one();
    for (bra_core, ket_core) in bra.iter().zip(ket) {
        let tmp = run(backend, |__s| {
            __s.dot_general_with_conj(&env, bra_core, &configs.env_bra, false, true)
        })
        .unwrap()
        .expect("environment and conjugated bra contraction should succeed");
        env = run(backend, |__s| {
            __s.dot_general_with_conj(&tmp, ket_core, &configs.tmp_ket, false, false)
        })
        .unwrap()
        .expect("normal ket contraction should succeed");
    }
    env
}

fn inner_persistent_backend_cache(
    backend: &mut CpuEngine,
    _cache: &mut tenferro_cpu::CpuPlanCache,
    bra: &[Tensor],
    ket: &[Tensor],
    configs: &LocalPathConfigs,
) -> Tensor {
    let mut env = scalar_one();
    for (site, (bra_core, ket_core)) in bra.iter().zip(ket).enumerate() {
        let tmp = run(backend, |__s| {
            __s.dot_general_with_conj_cached(
                Some(2 * site),
                &env,
                bra_core,
                &configs.env_bra,
                false,
                true,
            )
        })
        .unwrap()
        .expect("environment and conjugated bra contraction should succeed");
        env = run(backend, |__s| {
            __s.dot_general_with_conj_cached(
                Some(2 * site + 1),
                &tmp,
                ket_core,
                &configs.tmp_ket,
                false,
                false,
            )
        })
        .unwrap()
        .expect("normal ket contraction should succeed");
    }
    env
}

fn inner_single_exec_session(
    backend: &mut CpuEngine,
    _cache: &mut tenferro_cpu::CpuPlanCache,
    bra: &[Tensor],
    ket: &[Tensor],
    configs: &LocalPathConfigs,
) -> Tensor {
    let mut env = scalar_one();
    run(backend, |exec| {
        for (site, (bra_core, ket_core)) in bra.iter().zip(ket).enumerate() {
            let tmp = exec.dot_general_with_conj_cached(
                Some(2 * site),
                &env,
                bra_core,
                &configs.env_bra,
                false,
                true,
            )?;
            env = exec.dot_general_with_conj_cached(
                Some(2 * site + 1),
                &tmp,
                ket_core,
                &configs.tmp_ket,
                false,
                false,
            )?;
        }
        Ok::<(), tenferro_tensor::Error>(())
    })
    .unwrap()
    .expect("single exec session inner product should succeed");
    env
}

fn bench_dot_general_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_general_overhead/c64/one_thread");
    for &chi in CHIS {
        let fixture = build_mps_fixture(L, PHYS_DIM, chi);
        let configs = LocalPathConfigs::new();
        let params = format!("L_{L}_chi_{chi}_d_{PHYS_DIM}");

        group.bench_function(BenchmarkId::new("fresh_backend_cache", &params), |b| {
            b.iter(|| {
                let mut backend = engine(1);
                let output = inner_fresh_backend_cache(
                    black_box(&mut backend),
                    black_box(&fixture.bra_tensors),
                    black_box(&fixture.ket_tensors),
                    black_box(&configs),
                );
                black_box(output.shape().to_vec());
            });
        });

        group.bench_function(BenchmarkId::new("persistent_backend_cache", &params), |b| {
            b.iter_batched(
                || (engine(1), tenferro_cpu::CpuPlanCache::default()),
                |(mut backend, mut cache)| {
                    let output = inner_persistent_backend_cache(
                        black_box(&mut backend),
                        black_box(&mut cache),
                        black_box(&fixture.bra_tensors),
                        black_box(&fixture.ket_tensors),
                        black_box(&configs),
                    );
                    black_box(output.shape().to_vec());
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_function(BenchmarkId::new("single_exec_session", &params), |b| {
            b.iter_batched(
                || (engine(1), tenferro_cpu::CpuPlanCache::default()),
                |(mut backend, mut cache)| {
                    let output = inner_single_exec_session(
                        black_box(&mut backend),
                        black_box(&mut cache),
                        black_box(&fixture.bra_tensors),
                        black_box(&fixture.ket_tensors),
                        black_box(&configs),
                    );
                    black_box(output.shape().to_vec());
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dot_general_overhead);
criterion_main!(benches);
