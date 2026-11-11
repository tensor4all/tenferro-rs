//! Reproducer for tensor4all/tenferro-rs#2563, adapted from the reporter's
//! reproducer in that issue.
//!
//! `T(s) = sum_a t(kappa_a(K(s)))` for `K(s) = [[lam + d/2, s], [s, lam - d/2]]`
//! and `t(k) = -softplus(-beta (k - mu)) / beta`. `T` is smooth in `s` even at
//! `d = 0`, but the HVP `jvp(grad(T))` goes through `eigh`, whose pair weight is
//! `Delta / (Delta^2 + eps^2)`, and that weight is exactly zero at `Delta = 0`.
//! The run prints the AD value next to the closed-form `T''(0)`; no error and no
//! `NaN` is reported in either case.
//!
//! Run with:
//!
//! ```text
//! cargo run -p tenferro-linalg --features autodiff --example eigh_hvp_degenerate_pair
//! ```

use tenferro_ad::AdContext;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext, TracedTensor};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

const LAM: f64 = 2.3;
const MU: f64 = 2.0;
const BETA: f64 = 10.0;

fn run(t: &TracedTensor) -> Result<f64, BoxError> {
    let mut engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
    builder.install_extension_module(tenferro_linalg::extension_module::<CpuEngine>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    let runtime = builder.build()?;
    let program = GraphCompiler::new().compile(t)?;
    let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
    let mut out = runtime.run(&mut [&mut engine], &prepared, &[])?;
    Ok(out.remove(0).into_tensor(&mut engine)?.as_slice::<f64>()?[0])
}

/// `f(y) = t''(mu + y)` for `t(k) = -softplus(-beta (k - mu)) / beta`.
fn f(y: f64) -> f64 {
    let s = 1.0 / ((BETA * y).exp() + 1.0);
    -BETA * s * (1.0 - s)
}

/// Closed-form `T''(s)`, with the eigenvalue difference written through
/// `tanh x - tanh y = sinh(x - y) / (cosh x cosh y)` so that a small split does
/// not cancel.
fn exact(delta: f64, s: f64) -> f64 {
    let radius = ((delta / 2.0).powi(2) + s * s).sqrt();
    if radius == 0.0 {
        let half = BETA * (LAM - MU) / 2.0;
        return -BETA / (2.0 * half.cosh().powi(2));
    }
    let (upper, lower) = (LAM + radius - MU, LAM - radius - MU);
    let difference =
        -0.5 * (BETA * radius).sinh() / ((BETA * upper / 2.0).cosh() * (BETA * lower / 2.0).cosh());
    let ratio = s / radius;
    (1.0 / radius - s * s / (radius * radius * radius)) * difference
        + ratio * ratio * (f(upper) + f(lower))
}

fn hvp(delta: f64, s0: f64) -> Result<f64, BoxError> {
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules()?)?
        .build()?;
    let trace = TraceContext::new();
    let s = trace.from_vec_col_major(vec![1, 1], vec![s0])?;
    let offdiag = trace.from_vec_col_major(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0])?;
    let diag = trace.from_vec_col_major(
        vec![4, 1],
        vec![LAM + delta / 2.0, 0.0, 0.0, LAM - delta / 2.0],
    )?;
    let k = offdiag.matmul(&s)?.add(&diag)?.reshape([2, 2])?;
    let shift = trace.from_vec_col_major(vec![2], vec![BETA * MU; 2])?;
    let t = k
        .eigvalsh()?
        .scale_real(-BETA)?
        .add(&shift)?
        .softplus()?
        .reduce_sum(Some(&[0]))?
        .scale_real(-1.0 / BETA)?;
    let one = trace.from_vec_col_major(vec![1, 1], vec![1.0])?;
    let result = ad.jvp(&ad.grad(&t, &s)?, &s, &one)?;
    run(&result)
}

fn main() -> Result<(), BoxError> {
    println!("T(s) = tr t(K(s)), K(s) = [[lam + d/2, s], [s, lam - d/2]], t(k) = -softplus(-beta (k - mu))/beta");
    println!("lam = {LAM}, mu = {MU}, beta = {BETA}\n");

    for s0 in [0.0_f64, 1e-3] {
        let ad = hvp(0.0, s0)?;
        println!(
            "s = {s0:e}: AD T''(s) = {ad:+.6}, exact = {:+.6}",
            exact(0.0, s0)
        );
    }

    println!("\ncorner entry K[0][1] = s, s = 0, exact T''(0) as a function of d:");
    println!(
        "{:>10}  {:>12}  {:>12}  {:>10}",
        "d", "AD", "exact", "|error|"
    );
    for delta in [1e-1, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 0.0] {
        let ad = hvp(delta, 0.0)?;
        let expected = exact(delta, 0.0);
        println!(
            "{delta:>10e}  {ad:>+12.6}  {expected:>+12.6}  {:>10.2e}",
            (ad - expected).abs()
        );
    }
    Ok(())
}
