//! Where the eager `lstsq` rank check (#2583) gets its cutoff from.
//!
//! `EagerSessionLinalgExt::lstsq` refuses a rank-deficient `A` when some pivot
//! of the thin `R` satisfies `|R_ii| <= max(m, n) * eps * max_j |R_jj|`. This
//! example prints the distribution that cutoff has to separate: the pivot ratios
//! of the rank-deficient fixtures, the ratios of a full-rank corpus, and what the
//! pre-check solve (`QR` + `triangular_solve`, the whole of `lstsq` before that
//! check) returned on the rank-deficient ones. With `--time` it prints the cost
//! of that path, of the checked `lstsq`, and of the SVD-based alternatives.
//!
//! ```text
//! cargo run --release -p tenferro-linalg --features autodiff --example lstsq_rank_deficient_diag
//! cargo run --release -p tenferro-linalg --features autodiff --example lstsq_rank_deficient_diag -- --time
//! ```
//!
//! The 768x384 cases are the tenferro-benchmark `cpu/public_api` `lstsq` fixture,
//! reproduced here without that crate: `benchmark_value` over a column-major
//! buffer, seed 1 for `A` and seed 2 for the 16-column right-hand side. The
//! generator repeats a length-2048 pattern, so the 384 columns of that fixture
//! span at most 8 directions (they do span 8: `sv[8]/sv[0]` is roundoff-level).
//!
//! Timing caveat: the cost table is one thread of the default `native` (faer)
//! provider. A `blas` build hands the kernels to the vendor library, whose own
//! thread count the engine does not constrain.

use std::time::Instant;

use tenferro_ad::{DotGeneralConfig, EagerRuntime, EagerSession, EagerTensor, Tensor};
use tenferro_linalg::EagerSessionLinalgExt;

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// A column-major `rows x cols` `f64` matrix, built from a declared generator.
struct Fixture {
    label: String,
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Fixture {
    fn new(label: impl Into<String>, rows: usize, cols: usize, data: Vec<f64>) -> Self {
        Self {
            label: label.into(),
            rows,
            cols,
            data,
        }
    }

    fn filled(
        label: impl Into<String>,
        rows: usize,
        cols: usize,
        f: impl Fn(usize) -> f64,
    ) -> Self {
        Self::new(label, rows, cols, (0..rows * cols).map(f).collect())
    }

    /// `max(m, n) * eps`, the relative cutoff `lstsq` applies to the pivots.
    fn cutoff(&self) -> f64 {
        self.rows.max(self.cols) as f64 * f64::EPSILON
    }

    /// The right-hand side this diagnostic uses: the benchmark's 16-column
    /// buffer for the 768x384 fixture, a column of ones for the small ones.
    fn rhs(&self) -> Self {
        if self.cols == 384 {
            Self::filled("rhs", self.rows, 16, |i| benchmark_value(i, 2))
        } else {
            Self::filled("rhs", self.rows, 1, |_| 1.0)
        }
    }
}

/// The tenferro-benchmark `cpu/public_api` generator, seed included; the values
/// are multiples of `1/1024` in `[-1, 1)` and repeat with period 2048.
fn benchmark_value(i: usize, seed: u64) -> f64 {
    let x = (i as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(seed.wrapping_mul(1_442_695_040_888_963_407));
    ((x % 2048) as f64 - 1024.0) / 1024.0
}

/// splitmix64 in `[-1, 1)`: an ordinary well-conditioned dense matrix.
fn uniform_value(i: usize) -> f64 {
    let mut z = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

fn rank_deficient_fixtures() -> Vec<Fixture> {
    // Second column equal to the first: rank one, and the last pivot of `R`
    // comes out as exactly zero on this host.
    let duplicated = Fixture::new(
        "4x2 duplicated column",
        4,
        2,
        vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
    );
    // col3 = 0.5 col1 + 0.25 col2, formed in floating point, so the last pivot
    // is roundoff-level rather than an exact zero.
    let c0 = [1.0, 2.0, 3.0, 4.0, 5.0];
    let c1 = [0.1, 0.7, 0.3, 0.9, 0.2];
    let mut data = Vec::new();
    data.extend_from_slice(&c0);
    data.extend_from_slice(&c1);
    data.extend(c0.iter().zip(c1).map(|(x, y)| 0.5 * x + 0.25 * y));
    let dependent = Fixture::new("5x3 dependent column (fp)", 5, 3, data);
    let benchmark = Fixture::filled("768x384 benchmark (rank 8)", 768, 384, |i| {
        benchmark_value(i, 1)
    });
    vec![duplicated, dependent, benchmark]
}

fn full_rank_fixtures() -> Vec<Fixture> {
    let mut fixtures: Vec<Fixture> = [("4x2", 4, 2), ("5x3", 5, 3), ("64x32", 64, 32)]
        .into_iter()
        .map(|(label, rows, cols)| {
            Fixture::filled(format!("{label} uniform"), rows, cols, uniform_value)
        })
        .collect();
    fixtures.push(Fixture::filled("256x256 uniform", 256, 256, uniform_value));
    fixtures.push(Fixture::filled("768x384 uniform", 768, 384, uniform_value));
    // The full-rank input the regression test keeps: column scales 1 and 1e-8,
    // cond(A) ~ 1e8, still far above the cutoff.
    fixtures.push(Fixture::new(
        "4x2 column scales 1 / 1e-8",
        4,
        2,
        vec![1.0, 1.0, 1.0, 1.0, 0.0, 1e-8, 2e-8, 3e-8],
    ));
    fixtures
}

fn transpose_perm(rank: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(0, 1);
    perm
}

fn trailing_batch_dot_config(rank: usize) -> DotGeneralConfig {
    let batch: Vec<usize> = (2..rank).collect();
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: batch.as_slice().into(),
        rhs_batch_dims: batch.as_slice().into(),
    }
}

/// `|R_ii|` of the thin `A = QR`, one entry per column.
fn pivots(session: &mut EagerSession<'_>, a: &EagerTensor) -> tenferro_ad::Result<Vec<f64>> {
    let (_q, r) = session.qr(a)?;
    let diagonal = session.extract_diag(&r, 0, 1)?;
    let magnitudes = session.abs(&diagonal)?;
    let host = session.duplicate_value(&magnitudes)?;
    Ok(host.as_slice::<f64>()?.to_vec())
}

/// Pivot magnitudes divided by the largest one in the same matrix.
fn pivot_ratios(session: &mut EagerSession<'_>, a: &EagerTensor) -> tenferro_ad::Result<Vec<f64>> {
    let pivots = pivots(session, a)?;
    let scale = pivots.iter().fold(0.0_f64, |acc, &p| acc.max(p));
    Ok(pivots.iter().map(|&p| p / scale).collect())
}

/// Exactly what `lstsq` did before the rank check: thin `QR` and a triangular
/// solve, with no pivot inspection.
fn lstsq_without_check(
    session: &mut EagerSession<'_>,
    a: &EagerTensor,
    b: &EagerTensor,
) -> tenferro_ad::Result<EagerTensor> {
    let (q, r) = session.qr(a)?;
    let q_conj = session.conj(&q)?;
    let qh = session.transpose(&q_conj, &transpose_perm(q.shape().len()))?;
    let qh_b = session.dot_general(&qh, b, trailing_batch_dot_config(qh.shape().len()))?;
    session.triangular_solve(&r, &qh_b, true, false, false, false)
}

/// `max |A x - b|` through the session's own contraction and reduction.
fn max_abs_residual(
    session: &mut EagerSession<'_>,
    a: &EagerTensor,
    x: &EagerTensor,
    b: &EagerTensor,
) -> tenferro_ad::Result<f64> {
    let ax = session.dot_general(a, x, trailing_batch_dot_config(a.shape().len()))?;
    let difference = session.sub(&ax, b)?;
    let magnitude = session.abs(&difference)?;
    let largest = session.reduce_max(&magnitude, None)?;
    let host = session.duplicate_value(&largest)?;
    Ok(host.as_slice::<f64>()?[0])
}

fn to_eager(
    runtime: &std::sync::Arc<EagerRuntime>,
    fixture: &Fixture,
) -> tenferro_ad::Result<EagerTensor> {
    EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([fixture.rows, fixture.cols], fixture.data.clone())?,
        runtime.clone(),
    )
}

/// What the pre-check solve returned for one fixture.
fn pre_check_outcome(
    runtime: &std::sync::Arc<EagerRuntime>,
    engine: &mut tenferro_cpu::CpuEngine,
    fixture: &Fixture,
) -> tenferro_ad::Result<String> {
    let a = to_eager(runtime, fixture)?;
    let b = to_eager(runtime, &fixture.rhs())?;
    runtime.with_eager_session(engine, |session| {
        match lstsq_without_check(session, &a, &b) {
            Ok(x) => {
                let values = session.duplicate_value(&x)?.as_slice::<f64>()?.to_vec();
                // `f64::max` drops a NaN, so count the non-finite entries too.
                let largest = values.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
                let non_finite = values.iter().filter(|v| !v.is_finite()).count();
                let shown = if values.len() <= 8 {
                    format!("x = {values:?}")
                } else {
                    format!(
                        "finite max|x| = {largest:.3e}, {non_finite} non-finite of {} entries",
                        values.len()
                    )
                };
                let residual = max_abs_residual(session, &a, &x, &b)?;
                Ok(format!("{shown}, max|Ax-b| = {residual:.3e}"))
            }
            Err(error) => Ok(format!("error: {error}")),
        }
    })
}

/// Whether the checked `lstsq` accepts this fixture.
fn checked_outcome(
    runtime: &std::sync::Arc<EagerRuntime>,
    engine: &mut tenferro_cpu::CpuEngine,
    fixture: &Fixture,
) -> tenferro_ad::Result<String> {
    let a = to_eager(runtime, fixture)?;
    let b = to_eager(runtime, &fixture.rhs())?;
    runtime.with_eager_session(engine, |session| {
        Ok(session.lstsq(&a, &b).map_or_else(
            |error| format!("{error}"),
            |x| format!("Ok, shape {:?}", x.shape()),
        ))
    })
}

fn report(engine: &mut tenferro_cpu::CpuEngine) -> Outcome<()> {
    let runtime = EagerRuntime::new()?;
    println!(
        "engine: {} thread(s), f64, eps = {:.3e}",
        engine.num_threads(),
        f64::EPSILON
    );
    println!("cutoff on |R_ii| / max_j |R_jj|: max(m, n) * eps\n");

    println!("rank-deficient fixtures");
    for fixture in rank_deficient_fixtures() {
        let a = to_eager(&runtime, &fixture)?;
        let cutoff = fixture.cutoff();
        let ratios = runtime.with_eager_session(engine, |session| pivot_ratios(session, &a))?;
        let below: Vec<f64> = ratios.iter().copied().filter(|&r| r <= cutoff).collect();
        let smallest_kept = ratios
            .iter()
            .copied()
            .filter(|&r| r > cutoff)
            .fold(f64::INFINITY, f64::min);
        let largest_flagged = below.iter().copied().fold(0.0_f64, f64::max);
        // Also count against a bare `eps`, the cutoff without the `max(m, n)` factor.
        let below_eps = ratios.iter().filter(|&&r| r <= f64::EPSILON).count();
        println!(
            "  {:<30} n = {:<4} cutoff = {cutoff:.3e}  below cutoff {}/{}, below 1 * eps {}/{}",
            fixture.label,
            fixture.cols,
            below.len(),
            ratios.len(),
            below_eps,
            ratios.len()
        );
        println!(
            "      smallest ratio = {:.3e}, largest below cutoff = {:.3e} ({:.1} eps), smallest kept = {:.3e}",
            ratios.iter().copied().fold(f64::INFINITY, f64::min),
            largest_flagged,
            largest_flagged / f64::EPSILON,
            smallest_kept
        );
        println!(
            "      pre-check solve: {}",
            pre_check_outcome(&runtime, engine, &fixture)?
        );
        println!(
            "      lstsq with the check: {}\n",
            checked_outcome(&runtime, engine, &fixture)?
        );
    }

    println!("full-rank fixtures");
    for fixture in full_rank_fixtures() {
        let a = to_eager(&runtime, &fixture)?;
        let cutoff = fixture.cutoff();
        let ratios = runtime.with_eager_session(engine, |session| pivot_ratios(session, &a))?;
        let smallest = ratios.iter().copied().fold(f64::INFINITY, f64::min);
        let below = ratios.iter().filter(|&&r| r <= cutoff).count();
        println!(
            "  {:<30} n = {:<4} cutoff = {cutoff:.3e}  smallest ratio = {smallest:.3e}, below cutoff {below}/{}",
            fixture.label,
            fixture.cols,
            ratios.len()
        );
    }

    // The pre-check residual is not thread-invariant: the same bad pivot summed
    // in a different order gives a different garbage magnitude.
    println!("\npre-check residual, 768x384 benchmark fixture, rhs = 16");
    let fixture = Fixture::filled("768x384 benchmark", 768, 384, |i| benchmark_value(i, 1));
    for threads in [1, 4, 8] {
        let mut engine = if threads == 1 {
            tenferro_cpu::CpuEngine::serial()
        } else {
            tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().threads(threads).build()?)
        };
        // A pool whose workers could not be reserved silently runs serially.
        assert_eq!(engine.num_threads(), threads, "engine width");
        let a = to_eager(&runtime, &fixture)?;
        let b = to_eager(&runtime, &fixture.rhs())?;
        let (residual, checked) = runtime.with_eager_session(&mut engine, |session| {
            let x = lstsq_without_check(session, &a, &b)?;
            let residual = max_abs_residual(session, &a, &x, &b)?;
            let checked = session
                .lstsq(&a, &b)
                .map_or_else(|error| format!("{error}"), |_| "Ok".to_string());
            Ok::<_, tenferro_ad::Error>((residual, checked))
        })?;
        println!(
            "  {threads} thread(s): max|Ax-b| = {residual:.3e}; lstsq with the check: {checked}",
        );
    }
    Ok(())
}

/// Median of `reps` interleaved timings of each labelled path, in milliseconds.
///
/// The paths are timed round-robin rather than one at a time: the first path
/// timed in a session pays a warm-up cost that is a large share of a 2x2 call.
fn medians(reps: usize, mut run: impl FnMut(usize)) -> [f64; 4] {
    for label in 0..4 {
        run(label); // untimed warm-up of every path
    }
    let mut samples: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(reps));
    for _ in 0..reps {
        for (label, times) in samples.iter_mut().enumerate() {
            let start = Instant::now();
            run(label);
            times.push(start.elapsed().as_secs_f64() * 1e3);
        }
    }
    for times in &mut samples {
        times.sort_by(f64::total_cmp);
    }
    std::array::from_fn(|label| samples[label][reps / 2])
}

fn report_cost(engine: &mut tenferro_cpu::CpuEngine) -> Outcome<()> {
    const REPS: usize = 21;
    let runtime = EagerRuntime::new()?;
    let cases = [
        Fixture::new("2x2 full rank", 2, 2, vec![1.0, 2.0, 3.0, 4.0]),
        rank_deficient_fixtures().remove(0),
        Fixture::filled("768x384 rhs=16 full rank", 768, 384, uniform_value),
        Fixture::filled("768x384 rhs=16 benchmark", 768, 384, |i| {
            benchmark_value(i, 1)
        }),
    ];
    println!(
        "serial cost, median of {REPS} interleaved runs, ms ({} thread(s), default `native` provider)",
        engine.num_threads()
    );
    println!(
        "{:<30} {:>10} {:>12} {:>9} {:>11}",
        "case", "pre-check", "lstsq(check)", "svd(A)", "pinv(A)@b"
    );
    for fixture in cases {
        let a = to_eager(&runtime, &fixture)?;
        let b = to_eager(&runtime, &fixture.rhs())?;
        let (accepted, medians) = runtime.with_eager_session(engine, |session| {
            let accepted = session.lstsq(&a, &b).is_ok();
            let medians = medians(REPS, |path| match path {
                0 => {
                    std::hint::black_box(lstsq_without_check(session, &a, &b).unwrap());
                }
                1 if accepted => {
                    std::hint::black_box(session.lstsq(&a, &b).unwrap());
                }
                1 => {}
                2 => {
                    std::hint::black_box(session.svd(&a).unwrap());
                }
                _ => {
                    let pinv = session.pinv(&a).unwrap();
                    std::hint::black_box(
                        session
                            .dot_general(&pinv, &b, trailing_batch_dot_config(2))
                            .unwrap(),
                    );
                }
            });
            Ok::<_, tenferro_ad::Error>((accepted, medians))
        })?;
        let show = |value: f64, accepted: bool| {
            if accepted {
                format!("{value:.4}")
            } else {
                "error".to_string()
            }
        };
        println!(
            "{:<30} {:>10} {:>12} {:>9} {:>11}",
            fixture.label,
            show(medians[0], true),
            show(medians[1], accepted),
            show(medians[2], true),
            show(medians[3], true)
        );
    }
    Ok(())
}

fn main() -> Outcome<()> {
    let mut engine = tenferro_cpu::CpuEngine::serial();
    if std::env::args().any(|argument| argument == "--time") {
        report_cost(&mut engine)
    } else {
        report(&mut engine)
    }
}
