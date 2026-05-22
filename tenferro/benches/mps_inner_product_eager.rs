use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;
use tenferro::{CpuBackend, DotGeneralConfig, EagerRuntime, EagerTensor, Tensor, TypedTensor};

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
    Tensor::C64(TypedTensor::from_vec(shape.to_vec(), data))
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

fn eager_mps_tensors(ctx: &Arc<EagerRuntime>, tensors: &[Tensor]) -> Vec<EagerTensor> {
    tensors
        .iter()
        .cloned()
        .map(|tensor| EagerTensor::from_tensor_in(tensor, ctx.clone()))
        .collect()
}

fn eager_inner_product_local_path(
    ctx: &Arc<EagerRuntime>,
    bra: &[EagerTensor],
    ket: &[EagerTensor],
    configs: &LocalPathConfigs,
) -> EagerTensor {
    let mut env = EagerTensor::from_tensor_in(
        Tensor::from_vec(vec![1, 1], vec![Complex64::new(1.0, 0.0)]),
        ctx.clone(),
    );

    for (bra_core, ket_core) in bra.iter().zip(ket) {
        let tmp = env
            .dot_general_with_conj(bra_core, &configs.env_bra, false, true)
            .expect("environment and conjugated bra contraction should succeed");
        env = tmp
            .dot_general_with_conj(ket_core, &configs.tmp_ket, false, false)
            .expect("normal ket contraction should succeed");
    }

    env.reshape(&[])
        .expect("final 1x1 environment should reshape to scalar")
}

fn bench_mps_inner_product_eager(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_inner_product_eager/c64/one_thread");
    for &chi in CHIS {
        let fixture = build_mps_fixture(L, PHYS_DIM, chi);
        let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1));
        let bra = eager_mps_tensors(&ctx, &fixture.bra_tensors);
        let ket = eager_mps_tensors(&ctx, &fixture.ket_tensors);
        let configs = LocalPathConfigs::new();
        let params = format!("L_{L}_chi_{chi}_d_{PHYS_DIM}");

        let warmup_output = eager_inner_product_local_path(&ctx, &bra, &ket, &configs);
        black_box(warmup_output.data().shape());

        group.bench_function(BenchmarkId::new("eval_local_path", &params), |b| {
            b.iter(|| {
                let output = eager_inner_product_local_path(
                    black_box(&ctx),
                    black_box(&bra),
                    black_box(&ket),
                    black_box(&configs),
                );
                let value = output
                    .data()
                    .as_slice::<Complex64>()
                    .expect("scalar output")[0];
                black_box(value);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_mps_inner_product_eager);
criterion_main!(benches);
