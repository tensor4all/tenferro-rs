use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_device::LogicalMemorySpace;
use tenferro_linalg::{
    matrix_exp, qr, solve, solve_rrule, svd, svd_rrule, SolveGrad, SvdCotangent, SvdResult,
};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

const COL: MemoryOrder = MemoryOrder::ColumnMajor;
const MEM: LogicalMemorySpace = LogicalMemorySpace::MainMemory;

fn matrix_tensor(
    rows: usize,
    cols: usize,
    batches: usize,
    scale: f64,
    diagonal_boost: f64,
) -> Tensor<f64> {
    let mut data = vec![0.0; rows * cols * batches];
    for b in 0..batches {
        for j in 0..cols {
            for i in 0..rows {
                let seed = (i * 37 + j * 17 + b * 13 + 1) % 97;
                let centered = (seed as f64 / 97.0) - 0.5;
                let mut value = centered * scale;
                if i == j {
                    value += diagonal_boost;
                }
                data[i + rows * (j + cols * b)] = value;
            }
        }
    }

    let dims = if batches == 1 {
        vec![rows, cols]
    } else {
        vec![rows, cols, batches]
    };
    Tensor::from_slice(&data, &dims, COL).expect("matrix tensor build must succeed")
}

fn rhs_tensor(n: usize, nrhs: usize, batches: usize, scale: f64) -> Tensor<f64> {
    let mut data = vec![0.0; n * nrhs * batches];
    for b in 0..batches {
        for j in 0..nrhs {
            for i in 0..n {
                let seed = (i * 19 + j * 11 + b * 7 + 3) % 101;
                let value = ((seed as f64 / 101.0) - 0.5) * scale;
                data[i + n * (j + nrhs * b)] = value;
            }
        }
    }

    let dims = if batches == 1 {
        vec![n, nrhs]
    } else {
        vec![n, nrhs, batches]
    };
    Tensor::from_slice(&data, &dims, COL).expect("rhs tensor build must succeed")
}

fn singular_value_cotangent(k: usize, batches: usize) -> Tensor<f64> {
    let mut data = vec![0.0; k * batches];
    for b in 0..batches {
        for i in 0..k {
            data[i + k * b] = 1.0 + (((i + b) % 7) as f64) * 0.01;
        }
    }
    let dims = if batches == 1 {
        vec![k]
    } else {
        vec![k, batches]
    };
    Tensor::from_slice(&data, &dims, COL).expect("svd cotangent tensor build must succeed")
}

fn bench_forward_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward/svd");
    let cases = [
        ("sq16", 16usize, 16usize, 1usize),
        ("sq64", 64usize, 64usize, 1usize),
        ("tall128x32", 128usize, 32usize, 1usize),
        ("wide32x128", 32usize, 128usize, 1usize),
        ("batched8x8x32", 8usize, 8usize, 32usize),
    ];

    for &(label, m, n, batches) in &cases {
        let diag = if m == n { m as f64 } else { 0.0 };
        let a = matrix_tensor(m, n, batches, 0.25, diag);
        let mut ctx = CpuContext::new(1);
        group.bench_with_input(BenchmarkId::new("shape", label), &label, |bench, _| {
            bench.iter(|| {
                let result: SvdResult<f64, f64> =
                    svd(&mut ctx, black_box(&a), None).expect("svd benchmark call failed");
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_forward_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward/qr");
    let cases = [
        ("sq16", 16usize, 16usize, 1usize),
        ("sq64", 64usize, 64usize, 1usize),
        ("tall128x32", 128usize, 32usize, 1usize),
        ("wide32x128", 32usize, 128usize, 1usize),
        ("batched8x8x32", 8usize, 8usize, 32usize),
    ];

    for &(label, m, n, batches) in &cases {
        let diag = if m == n { m as f64 } else { 0.0 };
        let a = matrix_tensor(m, n, batches, 0.25, diag);
        let mut ctx = CpuContext::new(1);
        group.bench_with_input(BenchmarkId::new("shape", label), &label, |bench, _| {
            bench.iter(|| {
                let result = qr(&mut ctx, black_box(&a)).expect("qr benchmark call failed");
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_forward_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward/solve");
    let cases = [
        ("sq16_rhs1", 16usize, 1usize, 1usize),
        ("sq64_rhs4", 64usize, 4usize, 1usize),
        ("batched8_rhs1_b32", 8usize, 1usize, 32usize),
    ];

    for &(label, n, nrhs, batches) in &cases {
        let a = matrix_tensor(n, n, batches, 0.05, n as f64);
        let b = rhs_tensor(n, nrhs, batches, 0.5);
        let mut ctx = CpuContext::new(1);
        group.bench_with_input(BenchmarkId::new("shape", label), &label, |bench, _| {
            bench.iter(|| {
                let result = solve(&mut ctx, black_box(&a), black_box(&b))
                    .expect("solve benchmark call failed");
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_forward_matrix_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward/matrix_exp");
    let cases = [
        ("sq16", 16usize, 1usize),
        ("sq64", 64usize, 1usize),
        ("batched8x8x32", 8usize, 32usize),
    ];

    for &(label, n, batches) in &cases {
        let a = matrix_tensor(n, n, batches, 0.01, 0.0);
        let mut ctx = CpuContext::new(1);
        group.bench_with_input(BenchmarkId::new("shape", label), &label, |bench, _| {
            bench.iter(|| {
                let result =
                    matrix_exp(&mut ctx, black_box(&a)).expect("matrix_exp benchmark call failed");
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_ad_svd_rrule(c: &mut Criterion) {
    let mut group = c.benchmark_group("ad/svd_rrule");
    let cases = [
        ("sq16_s_only", 16usize, 16usize, 1usize),
        ("batched8x8x32_s_only", 8usize, 8usize, 32usize),
    ];

    for &(label, m, n, batches) in &cases {
        let a = matrix_tensor(m, n, batches, 0.2, m.min(n) as f64);
        let k = m.min(n);
        let cotangent = SvdCotangent {
            u: None,
            s: Some(singular_value_cotangent(k, batches)),
            vt: None,
        };
        let mut ctx = CpuContext::new(1);
        group.bench_with_input(BenchmarkId::new("shape", label), &label, |bench, _| {
            bench.iter(|| {
                let result = svd_rrule(&mut ctx, black_box(&a), black_box(&cotangent), None)
                    .expect("svd_rrule benchmark call failed");
                black_box(result);
            });
        });
    }
    group.finish();
}

fn bench_ad_solve_rrule(c: &mut Criterion) {
    let mut group = c.benchmark_group("ad/solve_rrule");
    let cases = [
        ("sq16_rhs1", 16usize, 1usize, 1usize),
        ("batched8_rhs1_b32", 8usize, 1usize, 32usize),
    ];

    for &(label, n, nrhs, batches) in &cases {
        let a = matrix_tensor(n, n, batches, 0.05, n as f64);
        let b = rhs_tensor(n, nrhs, batches, 0.5);
        let cotangent = Tensor::<f64>::ones(b.dims(), MEM, COL).unwrap();
        let mut ctx = CpuContext::new(1);
        group.bench_with_input(BenchmarkId::new("shape", label), &label, |bench, _| {
            bench.iter(|| {
                let result: SolveGrad<f64> = solve_rrule(
                    &mut ctx,
                    black_box(&a),
                    black_box(&b),
                    black_box(&cotangent),
                )
                .expect("solve_rrule benchmark call failed");
                black_box(result);
            });
        });
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(20)
}

criterion_group!(
    name = linalg_benches;
    config = criterion_config();
    targets = bench_forward_svd,
              bench_forward_qr,
              bench_forward_solve,
              bench_forward_matrix_exp,
              bench_ad_svd_rrule,
              bench_ad_solve_rrule
);
criterion_main!(linalg_benches);
