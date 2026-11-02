//! Fixed-overhead measurement for small contractions (tenferro-rs#2324).
//!
//! Reports raw microseconds, median of a fixed iteration count, on a single
//! pinned CPU. Not a criterion benchmark: the numbers here are quoted directly
//! in the issue/PR body, so the harness must be simple and stable.

use std::env;
use std::sync::Arc;
use std::time::Instant;

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_einsum::{EagerSessionEinsumExt, TensorEinsumExt};
use tenferro_runtime::Tensor;
use tenferro_tensor::{
    DType, DotGeneralAccumulation, DotGeneralConfig, TensorDot, TensorRead, TensorWrite,
    TypedTensor,
};

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default)
}

fn f64_tensor(shape: &[usize], seed: usize) -> Tensor {
    let len: usize = shape.iter().product();
    let data: Vec<f64> = (0..len)
        .map(|i| ((i * 17 + seed * 31 + 7) % 997) as f64 / 997.0 - 0.5)
        .collect();
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn zeros(shape: &[usize]) -> Tensor {
    TypedTensor::<f64>::zeros(shape.to_vec()).unwrap().into()
}

fn engine(threads: usize) -> CpuEngine {
    CpuEngine::new(Pool::builder().threads(threads).build().unwrap())
}

/// Median of `iters` samples of `f`, in microseconds.
fn median_us(iters: usize, warmup: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn row(name: &str, us: f64) {
    println!("{name:<46} {us:>10.3} us");
}

fn raw_faer_2x2(a: &[f64], b: &[f64], c: &mut [f64]) {
    use faer::{Accum, MatMut, MatRef, Par};
    let a_mat = MatRef::<f64>::from_column_major_slice(a, 2, 2);
    let b_mat = MatRef::<f64>::from_column_major_slice(b, 2, 2);
    let mut c_mat = MatMut::<f64>::from_column_major_slice_mut(c, 2, 2);
    faer::linalg::matmul::matmul(&mut c_mat, Accum::Replace, &a_mat, &b_mat, 1.0, Par::Seq);
}

fn contract_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [0].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    }
}

fn main() {
    let iters = env_usize("TENFERRO_BENCH_ITERS", 1000);
    let warmup = env_usize("TENFERRO_BENCH_WARMUP", 200);
    let threads = env_usize("TENFERRO_BENCH_THREADS", 1);
    let big_iters = env_usize("TENFERRO_BENCH_BIG_ITERS", 15);
    let n = env_usize("TENFERRO_BENCH_BIG_N", 1024);

    println!("# small_contraction_overhead threads={threads} iters={iters} warmup={warmup}");

    let ctx: Arc<EagerRuntime> = EagerRuntime::new().unwrap();
    let mut cpu = engine(threads);
    let mut eager_cpu = engine(threads);

    let cfg = contract_config();

    let a2 = f64_tensor(&[2, 2], 1);
    let b2 = f64_tensor(&[2, 2], 2);
    let a1 = f64_tensor(&[1, 1], 1);
    let b1 = f64_tensor(&[1, 1], 2);
    let ea2 = EagerTensor::from_tensor_in(a2.duplicate().unwrap(), Arc::clone(&ctx)).unwrap();
    let eb2 = EagerTensor::from_tensor_in(b2.duplicate().unwrap(), Arc::clone(&ctx)).unwrap();
    let ea1 = EagerTensor::from_tensor_in(a1.duplicate().unwrap(), Arc::clone(&ctx)).unwrap();
    let eb1 = EagerTensor::from_tensor_in(b1.duplicate().unwrap(), Arc::clone(&ctx)).unwrap();

    let a2v: Vec<f64> = a2.as_slice::<f64>().unwrap().to_vec();
    let b2v: Vec<f64> = b2.as_slice::<f64>().unwrap().to_vec();
    let mut c2v = vec![0.0f64; 4];

    // 1. Raw faer: the floor.
    row(
        "raw_faer_2x2",
        median_us(iters, warmup, || raw_faer_2x2(&a2v, &b2v, &mut c2v)),
    );

    // 2. Output allocation alone (2x2 f64 zeroed tensor).
    row(
        "alloc_output_2x2_f64",
        median_us(iters, warmup, || {
            std::hint::black_box(zeros(&[2, 2]));
        }),
    );

    // 3. dot_general_read_into with a preallocated output (no allocation).
    let mut out2 = zeros(&[2, 2]);
    row(
        "dot_general_read_into_2x2",
        median_us(iters, warmup, || {
            cpu.dot_general_read_into(
                TensorRead::from_tensor(&a2),
                TensorRead::from_tensor(&b2),
                &cfg,
                TensorWrite::from_tensor(&mut out2),
            )
            .unwrap();
        }),
    );

    // 4. dot_general_read (allocating).
    row(
        "dot_general_read_2x2",
        median_us(iters, warmup, || {
            let t = cpu
                .dot_general_read(
                    TensorRead::from_tensor(&a2),
                    TensorRead::from_tensor(&b2),
                    &cfg,
                )
                .unwrap();
            std::hint::black_box(t.shape());
        }),
    );

    // 5. dot_general_read_into_accum: explicit accumulation object cost.
    let mut out2b = zeros(&[2, 2]);
    let accum = DotGeneralAccumulation::overwrite(DType::F64).unwrap();
    row(
        "dot_general_read_into_accum_2x2",
        median_us(iters, warmup, || {
            cpu.dot_general_read_into_accum(
                TensorRead::from_tensor(&a2),
                TensorRead::from_tensor(&b2),
                &cfg,
                accum,
                TensorWrite::from_tensor(&mut out2b),
            )
            .unwrap();
        }),
    );

    // 6. Ablation: a 1x1 contraction does no measurable GEMM work, so it
    // isolates per-call plumbing from the arithmetic.
    let mut out1 = zeros(&[1, 1]);
    row(
        "dot_general_read_into_1x1 (plumbing floor)",
        median_us(iters, warmup, || {
            cpu.dot_general_read_into(
                TensorRead::from_tensor(&a1),
                TensorRead::from_tensor(&b1),
                &cfg,
                TensorWrite::from_tensor(&mut out1),
            )
            .unwrap();
        }),
    );
    row(
        "dot_general_read_1x1 (plumbing + alloc)",
        median_us(iters, warmup, || {
            let t = cpu
                .dot_general_read(
                    TensorRead::from_tensor(&a1),
                    TensorRead::from_tensor(&b1),
                    &cfg,
                )
                .unwrap();
            std::hint::black_box(t.shape());
        }),
    );

    // 7. Concrete einsum: plan build plus dispatch.
    row(
        "einsum_read_2x2 (concrete)",
        median_us(iters, warmup, || {
            let t = [&a2, &b2].einsum("ij,jk->ik", &mut cpu).unwrap();
            std::hint::black_box(t.shape());
        }),
    );
    row(
        "einsum_read_1x1 (concrete, plan floor)",
        median_us(iters, warmup, || {
            let t = [&a1, &b1].einsum("ij,jk->ik", &mut cpu).unwrap();
            std::hint::black_box(t.shape());
        }),
    );

    // 8. Eager engine-held einsum.
    row(
        "eager_einsum_2x2",
        median_us(iters, warmup, || {
            let t = ctx
                .with_eager_session(&mut eager_cpu, |s| s.einsum(&[&ea2, &eb2], "ij,jk->ik"))
                .unwrap();
            std::hint::black_box(t.shape());
        }),
    );
    row(
        "eager_einsum_1x1",
        median_us(iters, warmup, || {
            let t = ctx
                .with_eager_session(&mut eager_cpu, |s| s.einsum(&[&ea1, &eb1], "ij,jk->ik"))
                .unwrap();
            std::hint::black_box(t.shape());
        }),
    );
    row(
        "eager_dot_general_2x2",
        median_us(iters, warmup, || {
            let t = ctx
                .with_eager_session(&mut eager_cpu, |s| {
                    s.dot_general(&ea2, &eb2, contract_config())
                })
                .unwrap();
            std::hint::black_box(t.shape());
        }),
    );

    // 9. Large case: no regression allowed here.
    let abig = f64_tensor(&[n, n], 3);
    let bbig = f64_tensor(&[n, n], 4);
    let eabig = EagerTensor::from_tensor_in(abig.duplicate().unwrap(), Arc::clone(&ctx)).unwrap();
    let ebbig = EagerTensor::from_tensor_in(bbig.duplicate().unwrap(), Arc::clone(&ctx)).unwrap();
    row(
        &format!("dot_general_read_{n}x{n}"),
        median_us(big_iters, 3, || {
            let t = cpu
                .dot_general_read(
                    TensorRead::from_tensor(&abig),
                    TensorRead::from_tensor(&bbig),
                    &cfg,
                )
                .unwrap();
            std::hint::black_box(t.shape());
        }),
    );
    row(
        &format!("einsum_ij_jk_ik_{n}x{n}"),
        median_us(big_iters, 3, || {
            let t = ctx
                .with_eager_session(&mut eager_cpu, |s| s.einsum(&[&eabig, &ebbig], "ij,jk->ik"))
                .unwrap();
            std::hint::black_box(t.shape());
        }),
    );
}
