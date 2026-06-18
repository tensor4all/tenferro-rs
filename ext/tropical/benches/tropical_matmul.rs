use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
use tenferro_tensor::Tensor;

const DIRECT_SIZES: &[(usize, usize, usize)] = &[(16, 16, 16), (32, 32, 32), (64, 32, 64)];
const FALLBACK_SIZES: &[(usize, usize, usize)] = &[(16, 16, 16), (32, 16, 32)];
const MULTI_CONTRACTED_SIZES: &[(usize, usize, usize, usize)] = &[(8, 8, 4, 4), (16, 16, 4, 4)];

fn f64_tensor(shape: Vec<usize>, seed: usize) -> Tensor {
    let len = shape.iter().product();
    let data = (0..len)
        .map(|idx| ((idx * 17 + seed * 31 + 7) % 997) as f64 / 997.0 - 0.5)
        .collect();
    Tensor::from_vec_col_major(shape, data).unwrap()
}

fn consume(result: &tenferro_ext_tropical::einsum::TropicalEinsumResult) {
    let value_checksum = result
        .output
        .as_slice::<f64>()
        .map(|values| values.iter().copied().sum::<f64>())
        .unwrap_or(0.0);
    let argmax_checksum = result
        .argmax
        .iter()
        .flat_map(|step| step.indices())
        .map(|&index| u64::from(index))
        .sum::<u64>();
    black_box((
        result.output.shape(),
        value_checksum,
        argmax_checksum,
        result.argmax.len(),
    ));
}

fn bench_direct_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenferro_ext_tropical/direct_matmul");

    for &(m, k, n) in DIRECT_SIZES {
        let lhs = f64_tensor(vec![m, k], 1);
        let rhs = f64_tensor(vec![k, n], 2);
        group.bench_function(
            BenchmarkId::from_parameter(format!("{m}x{k}x{n}")),
            |bench| {
                bench.iter(|| {
                    let result = tropical_einsum_with_argmax(
                        TropicalKind::MaxPlus,
                        &[black_box(&lhs), black_box(&rhs)],
                        "ij,jk->ik",
                    )
                    .expect("direct tropical matmul benchmark should succeed");
                    consume(&result);
                });
            },
        );
    }

    group.finish();
}

fn bench_permuted_fallback(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenferro_ext_tropical/permuted_fallback");

    for &(m, k, n) in FALLBACK_SIZES {
        let lhs = f64_tensor(vec![k, m], 3);
        let rhs = f64_tensor(vec![k, n], 4);
        group.bench_function(
            BenchmarkId::from_parameter(format!("{m}x{k}x{n}")),
            |bench| {
                bench.iter(|| {
                    let result = tropical_einsum_with_argmax(
                        TropicalKind::MaxPlus,
                        &[black_box(&lhs), black_box(&rhs)],
                        "ji,jk->ik",
                    )
                    .expect("permuted tropical fallback benchmark should succeed");
                    consume(&result);
                });
            },
        );
    }

    group.finish();
}

fn bench_multi_contracted_fallback(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenferro_ext_tropical/multi_contracted_fallback");

    for &(i, l, j, k) in MULTI_CONTRACTED_SIZES {
        let lhs = f64_tensor(vec![k, j, i], 5);
        let rhs = f64_tensor(vec![l, j, k], 6);
        group.bench_function(
            BenchmarkId::from_parameter(format!("i{i}_l{l}_j{j}_k{k}")),
            |bench| {
                bench.iter(|| {
                    let result = tropical_einsum_with_argmax(
                        TropicalKind::MaxPlus,
                        &[black_box(&lhs), black_box(&rhs)],
                        "kji,ljk->il",
                    )
                    .expect("multi-contracted tropical fallback benchmark should succeed");
                    consume(&result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_direct_matmul,
    bench_permuted_fallback,
    bench_multi_contracted_fallback
);
criterion_main!(benches);
