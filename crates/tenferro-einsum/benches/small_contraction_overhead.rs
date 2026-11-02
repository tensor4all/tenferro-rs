//! Fixed-overhead measurement for small contractions (tenferro-rs#2324).
//!
//! Reports raw microseconds, median of a fixed iteration count, on a
//! single pinned CPU. Not a criterion benchmark: the numbers here are quoted
//! directly in the issue/PR body, so the harness must be simple and stable.

use std::env;
use std::sync::Arc;
use std::time::Instant;

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_einsum::EagerSessionEinsumExt;
use tenferro_runtime::Tensor;
use tenferro_tensor::{
    BackendSessionHost, DType, DotGeneralAccumulation, DotGeneralConfig, TensorRead, TensorWrite,
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

/// Median of `iters` samples taken inside one held backend session, so
/// session-entry plumbing is excluded from the per-call number.
fn median_us_inside(
    iters: usize,
    warmup: usize,
    backend: &mut CpuBackend,
    mut f: impl FnMut(&mut dyn tenferro_tensor::BackendSession) -> Result<(), String> + Send,
) -> f64 {
    backend
        .with_backend_session(|s| {
            for _ in 0..warmup {
                f(s).unwrap();
            }
            let mut samples: Vec<f64> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                f(s).unwrap();
                samples.push(t.elapsed().as_secs_f64() * 1e6);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Ok::<f64, tenferro_tensor::Error>(samples[samples.len() / 2])
        })
        .unwrap()
        .unwrap()
}

/// Median of `iters` samples taken inside one held eager session.
fn median_us_eager_inside(
    iters: usize,
    warmup: usize,
    ctx: &Arc<EagerRuntime>,
    mut f: impl FnMut(&mut tenferro_ad::EagerSession<'_>) -> Result<(), String> + Send,
) -> f64 {
    ctx.with_eager_session(|s| {
        for _ in 0..warmup {
            f(s).unwrap();
        }
        let mut samples: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            f(s).unwrap();
            samples.push(t.elapsed().as_secs_f64() * 1e6);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok::<f64, tenferro_ad::Error>(samples[samples.len() / 2])
    })
    .unwrap()
    .unwrap()
}

fn row(name: &str, us: f64) {
    println!("{name:<50} {us:>9.3} us");
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
    let warmup = env_usize("TENFERRO_BENCH_WARMUP", 100);
    let threads = env_usize("TENFERRO_BENCH_THREADS", 1);

    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(threads).unwrap()).unwrap();
    let mut backend = CpuBackend::with_threads(threads).unwrap();

    let cfg_2x2 = contract_config();

    let a1g = f64_tensor(&[1, 1], 1);
    let b1g = f64_tensor(&[1, 1], 2);
    let ea1 = EagerTensor::from_tensor_in(a1g.duplicate().unwrap(), Arc::clone(&ctx)).unwrap();
    let eb1 = EagerTensor::from_tensor_in(b1g.duplicate().unwrap(), Arc::clone(&ctx)).unwrap();
    let a2 = f64_tensor(&[2, 2], 1);
    let b2 = f64_tensor(&[2, 2], 2);
    let ea2 = EagerTensor::from_tensor_in(a2.duplicate().unwrap(), Arc::clone(&ctx)).unwrap();
    let eb2 = EagerTensor::from_tensor_in(b2.duplicate().unwrap(), Arc::clone(&ctx)).unwrap();

    let a2v: Vec<f64> = a2.as_slice::<f64>().unwrap().to_vec();
    let b2v: Vec<f64> = b2.as_slice::<f64>().unwrap().to_vec();
    let mut c2v = vec![0.0f64; 4];

    let big_iters_early = env_usize("TENFERRO_BENCH_BIG_ITERS", 20);
    let n_early = env_usize("TENFERRO_BENCH_BIG_N", 1024);
    let abig_early = f64_tensor(&[n_early, n_early], 3);
    let bbig_early = f64_tensor(&[n_early, n_early], 4);

    println!("# small_contraction_overhead threads={threads} iters={iters} warmup={warmup}");

    // 0. Session-entry floor: what any engine-held call pays before any work.
    row(
        "with_eager_session(empty)",
        median_us(iters, warmup, || {
            ctx.with_eager_session(|_s| Ok::<(), tenferro_ad::Error>(()))
                .unwrap()
                .unwrap();
        }),
    );

    // 1. Raw faer: the floor.
    row(
        "raw_faer_2x2",
        median_us(iters, warmup, || raw_faer_2x2(&a2v, &b2v, &mut c2v)),
    );

    // 2. Session entry alone: how much of dot_general_read is pure plumbing.
    row(
        "with_backend_session(empty)",
        median_us(iters, warmup, || {
            backend
                .with_backend_session(|_s| Ok::<(), tenferro_tensor::Error>(()))
                .unwrap()
                .unwrap();
        }),
    );

    // 3. Output allocation alone (2x2 f64 zeroed tensor).
    row(
        "alloc_output_2x2_f64",
        median_us(iters, warmup, || {
            std::hint::black_box(zeros(&[2, 2]));
        }),
    );

    // 4. dot_general_read_into with a preallocated output (no allocation).
    let mut out2 = zeros(&[2, 2]);
    row(
        "dot_general_read_into_2x2",
        median_us(iters, warmup, || {
            backend
                .with_backend_session(|s| {
                    s.dot_general_read_into(
                        TensorRead::from_tensor(&a2),
                        TensorRead::from_tensor(&b2),
                        &cfg_2x2,
                        TensorWrite::from_tensor(&mut out2),
                    )
                })
                .unwrap()
                .unwrap();
        }),
    );

    // 5. dot_general_read (allocating) through a session entry.
    row(
        "dot_general_read_2x2 (with_backend_session)",
        median_us(iters, warmup, || {
            backend
                .with_backend_session(|s| {
                    s.dot_general_read(
                        TensorRead::from_tensor(&a2),
                        TensorRead::from_tensor(&b2),
                        &cfg_2x2,
                    )
                })
                .unwrap()
                .unwrap();
        }),
    );

    // 6. dot_general_read_into_accum: explicit accumulation object cost.
    let mut out2b = zeros(&[2, 2]);
    let accum = DotGeneralAccumulation::overwrite(DType::F64).unwrap();
    row(
        "dot_general_read_into_accum_2x2",
        median_us(iters, warmup, || {
            backend
                .with_backend_session(|s| {
                    s.dot_general_read_into_accum(
                        TensorRead::from_tensor(&a2),
                        TensorRead::from_tensor(&b2),
                        &cfg_2x2,
                        accum,
                        TensorWrite::from_tensor(&mut out2b),
                    )
                })
                .unwrap()
                .unwrap();
        }),
    );

    // 6b. Same calls but with the session held across all iterations, so the
    // measurement excludes session-entry plumbing and shows the per-call cost.
    row(
        "dot_general_read_2x2 (held session)",
        median_us_inside(iters, warmup, &mut backend, |s| {
            s.dot_general_read(
                TensorRead::from_tensor(&a2),
                TensorRead::from_tensor(&b2),
                &cfg_2x2,
            )
            .map(|t| {
                std::hint::black_box(t.shape());
            })
            .map_err(|e| e.to_string())
        }),
    );
    row(
        "dot_general_read_into_2x2 (held session)",
        median_us_inside(iters, warmup, &mut backend, |s| {
            s.dot_general_read_into(
                TensorRead::from_tensor(&a2),
                TensorRead::from_tensor(&b2),
                &cfg_2x2,
                TensorWrite::from_tensor(&mut out2),
            )
            .map_err(|e| e.to_string())
        }),
    );
    row(
        "einsum_read_2x2 (held session)",
        median_us_inside(iters, warmup, &mut backend, |s| {
            use tenferro_einsum::TensorEinsumExt;
            [&a2, &b2]
                .einsum("ij,jk->ik", s)
                .map(|t| {
                    std::hint::black_box(t.shape());
                })
                .map_err(|e| e.to_string())
        }),
    );
    row(
        "dot_general_read_1024x1024 (held session)",
        median_us_inside(big_iters_early, 3, &mut backend, |s| {
            s.dot_general_read(
                TensorRead::from_tensor(&abig_early),
                TensorRead::from_tensor(&bbig_early),
                &cfg_2x2,
            )
            .map(|t| {
                std::hint::black_box(t.shape());
            })
            .map_err(|e| e.to_string())
        }),
    );

    // 6c. Ablation: a 1x1 contraction does no measurable GEMM work, so it
    // isolates the per-call plumbing (validation, mode selection, plan
    // lookup, buffer checkout, output tagging) from the arithmetic.
    let a1 = f64_tensor(&[1, 1], 1);
    let b1 = f64_tensor(&[1, 1], 2);
    let mut out1 = zeros(&[1, 1]);
    row(
        "dot_general_read_into_1x1 (plumbing floor)",
        median_us_inside(iters, warmup, &mut backend, |s| {
            s.dot_general_read_into(
                TensorRead::from_tensor(&a1),
                TensorRead::from_tensor(&b1),
                &cfg_2x2,
                TensorWrite::from_tensor(&mut out1),
            )
            .map_err(|e| e.to_string())
        }),
    );
    row(
        "dot_general_read_1x1 (plumbing + alloc)",
        median_us_inside(iters, warmup, &mut backend, |s| {
            s.dot_general_read(
                TensorRead::from_tensor(&a1),
                TensorRead::from_tensor(&b1),
                &cfg_2x2,
            )
            .map(|t| {
                std::hint::black_box(t.shape());
            })
            .map_err(|e| e.to_string())
        }),
    );
    row(
        "einsum_read_1x1 (plan cost floor)",
        median_us_inside(iters, warmup, &mut backend, |s| {
            use tenferro_einsum::TensorEinsumExt;
            [&a1, &b1]
                .einsum("ij,jk->ik", s)
                .map(|t| {
                    std::hint::black_box(t.shape());
                })
                .map_err(|e| e.to_string())
        }),
    );

    // 7. einsum, engine held (the 14.8 us path from the issue).
    row(
        "einsum_ij_jk_ik_2x2 (with_eager_session)",
        median_us(iters, warmup, || {
            let out = ctx
                .with_eager_session(|s| s.einsum(&[&ea2, &eb2], "ij,jk->ik"))
                .unwrap()
                .unwrap();
            std::hint::black_box(out.value().unwrap().shape());
        }),
    );

    // 7b. The same two eager calls with the eager session held across all
    // iterations: separates eager-runtime per-call cost from session entry.
    row(
        "eager_einsum_2x2 (held eager session)",
        median_us_eager_inside(iters, warmup, &ctx, |s| {
            s.einsum(&[&ea2, &eb2], "ij,jk->ik")
                .map(|t| {
                    std::hint::black_box(t.shape());
                })
                .map_err(|e| e.to_string())
        }),
    );
    row(
        "eager_dot_general_2x2 (held eager session)",
        median_us_eager_inside(iters, warmup, &ctx, |s| {
            s.dot_general(&ea2, &eb2, contract_config())
                .map(|t| {
                    std::hint::black_box(t.shape());
                })
                .map_err(|e| e.to_string())
        }),
    );
    row(
        "eager_einsum_1x1 (held eager session)",
        median_us_eager_inside(iters, warmup, &ctx, |s| {
            s.einsum(&[&ea1, &eb1], "ij,jk->ik")
                .map(|t| {
                    std::hint::black_box(t.shape());
                })
                .map_err(|e| e.to_string())
        }),
    );

    // 8. einsum on concrete tensors inside an already-held backend session.
    row(
        "einsum_read_2x2 (with_backend_session)",
        median_us(iters, warmup, || {
            use tenferro_einsum::TensorEinsumExt;
            backend
                .with_backend_session(|s| [&a2, &b2].einsum("ij,jk->ik", s))
                .unwrap()
                .unwrap();
        }),
    );

    // Large case: no regression allowed here.
    let big_iters = big_iters_early;
    let n = n_early;
    let abig = &abig_early;
    let bbig = &bbig_early;
    let eabig = EagerTensor::from_tensor_in(abig.duplicate().unwrap(), Arc::clone(&ctx)).unwrap();
    let ebbig = EagerTensor::from_tensor_in(bbig.duplicate().unwrap(), Arc::clone(&ctx)).unwrap();
    let cfg_big = contract_config();
    row(
        &format!("dot_general_read_{n}x{n}"),
        median_us(big_iters, 3, || {
            backend
                .with_backend_session(|s| {
                    s.dot_general_read(
                        TensorRead::from_tensor(abig),
                        TensorRead::from_tensor(bbig),
                        &cfg_big,
                    )
                })
                .unwrap()
                .unwrap();
        }),
    );
    row(
        &format!("einsum_ij_jk_ik_{n}x{n}"),
        median_us(big_iters, 3, || {
            let out = ctx
                .with_eager_session(|s| s.einsum(&[&eabig, &ebbig], "ij,jk->ik"))
                .unwrap()
                .unwrap();
            std::hint::black_box(out.value().unwrap().shape());
        }),
    );
}
