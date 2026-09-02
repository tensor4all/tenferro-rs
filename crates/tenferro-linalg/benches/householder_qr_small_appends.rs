use std::env;
use std::fs;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use num_complex::Complex64;
use serde::Serialize;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::{with_cpu_exec_session, CpuBackend, CpuBackendKind, CpuExecSession};
use tenferro_linalg::{EagerTensorLinalgExt, HouseholderQr, QrGauge, QrOptions, TensorLinalgExt};
use tenferro_tensor::{BackendSessionHost, Tensor};

const INITIAL_RANK: usize = 5;
const BLOCK_WIDTH: usize = 3;
const APPENDS: usize = 9;
const FINAL_RANK: usize = INITIAL_RANK + BLOCK_WIDTH * APPENDS;
const SAMPLE_BATCH: usize = 64;
const CPU_CLOCK_WARMUP_MS: u64 = 50;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Lane {
    Append,
    R,
    QColumns,
    Complete,
    FreshSessionComplete,
    EagerComplete,
}

#[derive(Clone, Copy, Debug)]
enum BenchDType {
    F64,
    C64,
}

impl BenchDType {
    const fn name(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::C64 => "c64",
        }
    }
}

#[derive(Debug)]
struct Config {
    backend: CpuBackendKind,
    lane: Lane,
    dtype: BenchDType,
    rows: usize,
    warmups: usize,
    repetitions: usize,
    git_commit: String,
}

#[derive(Serialize)]
struct ThreadEnvironment {
    rayon_num_threads: Option<String>,
    openblas_num_threads: Option<String>,
    omp_num_threads: Option<String>,
    mkl_num_threads: Option<String>,
}

#[derive(Serialize)]
struct Record {
    schema: &'static str,
    git_commit: String,
    backend: &'static str,
    lane: Lane,
    dtype: &'static str,
    gauge: &'static str,
    rows: usize,
    initial_rank: usize,
    block_width: usize,
    appends: usize,
    final_rank: usize,
    sample_batch: usize,
    warmups: usize,
    repetitions: usize,
    cpu_affinity: Option<String>,
    cpu_frequency_mhz: Option<f64>,
    thread_environment: ThreadEnvironment,
    timings_ms: Vec<f64>,
    reconstruction_relative_error: f64,
    orthogonality_relative_error: f64,
}

fn main() {
    let config = parse_args().unwrap_or_else(|error| panic!("{error}"));
    let (initial, blocks, accumulated) = generate_inputs(config.rows, config.dtype).unwrap();
    let (timings_ms, q, r) = match config.lane {
        Lane::FreshSessionComplete => run_fresh_session(&config, &initial, &blocks),
        Lane::EagerComplete => run_eager(&config, initial, blocks),
        lane => run_concrete(&config, lane, &initial, &blocks),
    }
    .unwrap_or_else(|error| panic!("benchmark failed: {error}"));
    let (reconstruction_relative_error, orthogonality_relative_error) =
        errors(&q, &r, &accumulated, config.dtype).unwrap();
    let backend = match config.backend {
        CpuBackendKind::Faer => "faer",
        CpuBackendKind::Blas => "blas",
    };
    let record = Record {
        schema: "tenferro.householder-qr-small-appends.v1",
        git_commit: config.git_commit.clone(),
        backend,
        lane: config.lane,
        dtype: config.dtype.name(),
        gauge: "raw",
        rows: config.rows,
        initial_rank: INITIAL_RANK,
        block_width: BLOCK_WIDTH,
        appends: APPENDS,
        final_rank: FINAL_RANK,
        sample_batch: SAMPLE_BATCH,
        warmups: config.warmups,
        repetitions: config.repetitions,
        cpu_affinity: process_affinity(),
        cpu_frequency_mhz: pinned_cpu_frequency_mhz(),
        thread_environment: ThreadEnvironment {
            rayon_num_threads: env::var("RAYON_NUM_THREADS").ok(),
            openblas_num_threads: env::var("OPENBLAS_NUM_THREADS").ok(),
            omp_num_threads: env::var("OMP_NUM_THREADS").ok(),
            mkl_num_threads: env::var("MKL_NUM_THREADS").ok(),
        },
        timings_ms,
        reconstruction_relative_error,
        orthogonality_relative_error,
    };
    println!("{}", serde_json::to_string(&record).unwrap());
}

fn parse_args() -> Result<Config, String> {
    let mut backend = None;
    let mut lane = None;
    let mut rows = None;
    let mut dtype = BenchDType::F64;
    let mut warmups = 3;
    let mut repetitions = 10;
    let args = env::args()
        .skip(1)
        .filter(|argument| argument != "--bench")
        .collect::<Vec<_>>();
    let mut index = 0;
    while index < args.len() {
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("missing value after {}", args[index]))?;
        match args[index].as_str() {
            "--backend" => {
                backend = Some(match value.as_str() {
                    "faer" => CpuBackendKind::Faer,
                    "blas" => CpuBackendKind::Blas,
                    _ => return Err(format!("unknown backend {value:?}")),
                })
            }
            "--lane" => {
                lane = Some(match value.as_str() {
                    "append" => Lane::Append,
                    "r" => Lane::R,
                    "q-columns" => Lane::QColumns,
                    "complete" => Lane::Complete,
                    "fresh-session-complete" => Lane::FreshSessionComplete,
                    "eager-complete" => Lane::EagerComplete,
                    _ => return Err(format!("unknown lane {value:?}")),
                })
            }
            "--rows" => rows = Some(parse(value, "rows")?),
            "--dtype" => {
                dtype = match value.as_str() {
                    "f64" => BenchDType::F64,
                    "c64" => BenchDType::C64,
                    _ => return Err(format!("unknown dtype {value:?}")),
                }
            }
            "--warmups" => warmups = parse(value, "warmups")?,
            "--repetitions" => repetitions = parse(value, "repetitions")?,
            other => return Err(format!("unknown option {other:?}")),
        }
        index += 2;
    }
    let rows = rows.ok_or_else(|| "--rows is required".to_string())?;
    if ![64, 128, 256].contains(&rows) {
        return Err("rows must be one of 64, 128, or 256".into());
    }
    if repetitions == 0 {
        return Err("repetitions must be positive".into());
    }
    Ok(Config {
        backend: backend.ok_or_else(|| "--backend is required".to_string())?,
        lane: lane.ok_or_else(|| "--lane is required".to_string())?,
        dtype,
        rows,
        warmups,
        repetitions,
        git_commit: env::var("TENFERRO_BENCH_GIT_COMMIT")
            .map_err(|_| "TENFERRO_BENCH_GIT_COMMIT is required".to_string())?,
    })
}

fn parse<T: std::str::FromStr>(value: &str, field: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid {field}: {value:?}"))
}

fn generate_inputs(
    rows: usize,
    dtype: BenchDType,
) -> Result<(Tensor, Vec<Tensor>, Tensor), String> {
    match dtype {
        BenchDType::F64 => {
            generate_inputs_typed(rows, |real, _, diagonal| real + f64::from(diagonal))
        }
        BenchDType::C64 => generate_inputs_typed(rows, |real, imag, diagonal| {
            Complex64::new(real + f64::from(diagonal), imag)
        }),
    }
}

fn generate_inputs_typed<T>(
    rows: usize,
    value: impl Fn(f64, f64, bool) -> T,
) -> Result<(Tensor, Vec<Tensor>, Tensor), String>
where
    Tensor: From<tenferro_tensor::TypedTensor<T>>,
    T: tenferro_tensor::TensorScalar,
{
    let mut state = 7_u64;
    let mut values = Vec::with_capacity(rows * FINAL_RANK);
    let scale = (rows as f64).sqrt().recip();
    for column in 0..FINAL_RANK {
        for row in 0..rows {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let real = (((state >> 11) as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0) * scale;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let imag = (((state >> 11) as f64) / ((1_u64 << 53) as f64) * 2.0 - 1.0) * scale;
            values.push(value(real, imag, row == column));
        }
    }
    let initial_len = rows * INITIAL_RANK;
    let initial = Tensor::from(
        tenferro_tensor::TypedTensor::from_vec_col_major(
            vec![rows, INITIAL_RANK],
            values[..initial_len].to_vec(),
        )
        .map_err(to_string)?,
    );
    let blocks = (0..APPENDS)
        .map(|block| {
            let start = initial_len + block * rows * BLOCK_WIDTH;
            tenferro_tensor::TypedTensor::from_vec_col_major(
                vec![rows, BLOCK_WIDTH],
                values[start..start + rows * BLOCK_WIDTH].to_vec(),
            )
            .map(Tensor::from)
            .map_err(to_string)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let accumulated = Tensor::from(
        tenferro_tensor::TypedTensor::from_vec_col_major(vec![rows, FINAL_RANK], values)
            .map_err(to_string)?,
    );
    Ok((initial, blocks, accumulated))
}

fn run_concrete(
    config: &Config,
    lane: Lane,
    initial: &Tensor,
    blocks: &[Tensor],
) -> Result<(Vec<f64>, Tensor, Tensor), String> {
    let mut backend =
        CpuBackend::with_threads_and_kind(1, config.backend).map_err(|error| error.to_string())?;
    backend.with_backend_session(|session| {
        with_cpu_exec_session(session, |session| {
            let total = config.warmups + config.repetitions;
            let mut timings = Vec::with_capacity(config.repetitions);
            for iteration in 0..total {
                let states = (0..SAMPLE_BATCH)
                    .map(|_| initial.householder_qr(session).map_err(to_string))
                    .collect::<Result<Vec<_>, _>>()?;
                warm_cpu_clock();
                let elapsed = if matches!(lane, Lane::Complete) {
                    let start = Instant::now();
                    for state in states {
                        black_box(complete_concrete(state, blocks, session)?);
                    }
                    start.elapsed()
                } else {
                    let mut elapsed = std::time::Duration::ZERO;
                    for state in states {
                        elapsed += attributed_concrete(state, blocks, session, lane)?;
                    }
                    elapsed
                }
                .as_secs_f64()
                    * 1.0e3;
                if iteration >= config.warmups {
                    timings.push(elapsed);
                }
            }
            let state = initial.householder_qr(session).map_err(to_string)?;
            let final_state = append_sequence(state, blocks, session)?;
            let q = final_state
                .q_columns(0..FINAL_RANK, raw_options(), session)
                .map_err(to_string)?;
            let r = final_state.r(raw_options(), session).map_err(to_string)?;
            Ok((timings, q, r))
        })
        .ok_or_else(|| "CPU execution session unavailable".to_string())?
    })
}

fn append_sequence(
    mut state: HouseholderQr<Tensor>,
    blocks: &[Tensor],
    session: &mut CpuExecSession<'_>,
) -> Result<HouseholderQr<Tensor>, String> {
    for block in blocks {
        state = state.append_columns(block, session).map_err(to_string)?;
    }
    Ok(state)
}

fn attributed_concrete(
    mut state: HouseholderQr<Tensor>,
    blocks: &[Tensor],
    session: &mut CpuExecSession<'_>,
    lane: Lane,
) -> Result<std::time::Duration, String> {
    let mut elapsed = std::time::Duration::ZERO;
    for (append, block) in blocks.iter().enumerate() {
        let start = Instant::now();
        state = state.append_columns(block, session).map_err(to_string)?;
        if matches!(lane, Lane::Append) {
            elapsed += start.elapsed();
        }

        let start = Instant::now();
        black_box(state.r(raw_options(), session).map_err(to_string)?);
        if matches!(lane, Lane::R) {
            elapsed += start.elapsed();
        }

        let column = INITIAL_RANK + append * BLOCK_WIDTH;
        let start = Instant::now();
        black_box(
            state
                .q_columns(column..column + BLOCK_WIDTH, raw_options(), session)
                .map_err(to_string)?,
        );
        if matches!(lane, Lane::QColumns) {
            elapsed += start.elapsed();
        }
    }
    black_box(state);
    Ok(elapsed)
}

fn complete_concrete(
    mut state: HouseholderQr<Tensor>,
    blocks: &[Tensor],
    session: &mut CpuExecSession<'_>,
) -> Result<HouseholderQr<Tensor>, String> {
    for (append, block) in blocks.iter().enumerate() {
        state = state.append_columns(block, session).map_err(to_string)?;
        black_box(state.r(raw_options(), session).map_err(to_string)?);
        let start = INITIAL_RANK + append * BLOCK_WIDTH;
        black_box(
            state
                .q_columns(start..start + BLOCK_WIDTH, raw_options(), session)
                .map_err(to_string)?,
        );
    }
    Ok(state)
}

fn run_fresh_session(
    config: &Config,
    initial: &Tensor,
    blocks: &[Tensor],
) -> Result<(Vec<f64>, Tensor, Tensor), String> {
    let mut backend = CpuBackend::with_threads_and_kind(1, config.backend).map_err(to_string)?;
    let total = config.warmups + config.repetitions;
    let mut timings = Vec::with_capacity(config.repetitions);
    for iteration in 0..total {
        let states = (0..SAMPLE_BATCH)
            .map(|_| backend.with_backend_session(|session| initial.householder_qr(session)))
            .collect::<Result<Vec<_>, _>>()
            .map_err(to_string)?;
        warm_cpu_clock();
        let start = Instant::now();
        for mut state in states {
            for (append, block) in blocks.iter().enumerate() {
                state = backend
                    .with_backend_session(|session| state.append_columns(block, session))
                    .map_err(to_string)?;
                black_box(
                    backend
                        .with_backend_session(|session| state.r(raw_options(), session))
                        .map_err(to_string)?,
                );
                let start = INITIAL_RANK + append * BLOCK_WIDTH;
                black_box(
                    backend
                        .with_backend_session(|session| {
                            state.q_columns(start..start + BLOCK_WIDTH, raw_options(), session)
                        })
                        .map_err(to_string)?,
                );
            }
        }
        let elapsed = start.elapsed().as_secs_f64() * 1.0e3;
        if iteration >= config.warmups {
            timings.push(elapsed);
        }
    }
    let mut state = backend
        .with_backend_session(|session| initial.householder_qr(session))
        .map_err(to_string)?;
    for block in blocks {
        state = backend
            .with_backend_session(|session| state.append_columns(block, session))
            .map_err(to_string)?;
    }
    let q = backend
        .with_backend_session(|session| state.q_columns(0..FINAL_RANK, raw_options(), session))
        .map_err(to_string)?;
    let r = backend
        .with_backend_session(|session| state.r(raw_options(), session))
        .map_err(to_string)?;
    Ok((timings, q, r))
}

fn run_eager(
    config: &Config,
    initial: Tensor,
    blocks: Vec<Tensor>,
) -> Result<(Vec<f64>, Tensor, Tensor), String> {
    let runtime = EagerRuntime::with_cpu_backend(
        CpuBackend::with_threads_and_kind(1, config.backend).map_err(to_string)?,
    )
    .map_err(to_string)?;
    let initial = EagerTensor::from_tensor_in(initial, Arc::clone(&runtime)).map_err(to_string)?;
    let blocks = blocks
        .into_iter()
        .map(|block| EagerTensor::from_tensor_in(block, Arc::clone(&runtime)).map_err(to_string))
        .collect::<Result<Vec<_>, _>>()?;
    let initial_state = initial.householder_qr().map_err(to_string)?;
    let total = config.warmups + config.repetitions;
    let mut timings = Vec::with_capacity(config.repetitions);
    for iteration in 0..total {
        let states = (0..SAMPLE_BATCH)
            .map(|_| initial_state.clone())
            .collect::<Vec<_>>();
        warm_cpu_clock();
        let start = Instant::now();
        for state in states {
            black_box(complete_eager(state, &blocks)?);
        }
        let elapsed = start.elapsed().as_secs_f64() * 1.0e3;
        if iteration >= config.warmups {
            timings.push(elapsed);
        }
    }
    let state = complete_eager(initial_state, &blocks)?;
    let q = state
        .q_columns(0..FINAL_RANK, raw_options())
        .map_err(to_string)?
        .to_tensor()
        .map_err(to_string)?;
    let r = state
        .r(raw_options())
        .map_err(to_string)?
        .to_tensor()
        .map_err(to_string)?;
    Ok((timings, q, r))
}

fn complete_eager(
    mut state: HouseholderQr<EagerTensor>,
    blocks: &[EagerTensor],
) -> Result<HouseholderQr<EagerTensor>, String> {
    for (append, block) in blocks.iter().enumerate() {
        state = state.append_columns(block).map_err(to_string)?;
        black_box(state.r(raw_options()).map_err(to_string)?);
        let start = INITIAL_RANK + append * BLOCK_WIDTH;
        black_box(
            state
                .q_columns(start..start + BLOCK_WIDTH, raw_options())
                .map_err(to_string)?,
        );
    }
    Ok(state)
}

fn warm_cpu_clock() {
    let start = Instant::now();
    let duration = std::time::Duration::from_millis(CPU_CLOCK_WARMUP_MS);
    let mut state = 1_u64;
    while start.elapsed() < duration {
        state = black_box(
            state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1),
        );
    }
    black_box(state);
}

fn process_affinity() -> Option<String> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:\t"))
        .map(str::to_owned)
}

fn pinned_cpu_frequency_mhz() -> Option<f64> {
    let cpu = process_affinity()?.parse::<usize>().ok()?;
    fs::read_to_string(format!(
        "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
    ))
    .ok()?
    .trim()
    .parse::<f64>()
    .ok()
    .map(|khz| khz / 1_000.0)
}

fn raw_options() -> QrOptions {
    QrOptions::default().gauge(QrGauge::Raw)
}

fn to_string(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn errors(q: &Tensor, r: &Tensor, a: &Tensor, dtype: BenchDType) -> Result<(f64, f64), String> {
    match dtype {
        BenchDType::F64 => errors_f64(q, r, a),
        BenchDType::C64 => errors_c64(q, r, a),
    }
}

fn errors_f64(q: &Tensor, r: &Tensor, a: &Tensor) -> Result<(f64, f64), String> {
    let q = q.as_slice::<f64>().map_err(to_string)?;
    let r = r.as_slice::<f64>().map_err(to_string)?;
    let a = a.as_slice::<f64>().map_err(to_string)?;
    let rows = a.len() / FINAL_RANK;
    let mut residual_sq = 0.0;
    let mut norm_sq = 0.0;
    for col in 0..FINAL_RANK {
        for row in 0..rows {
            let actual = (0..FINAL_RANK)
                .map(|inner| q[row + inner * rows] * r[inner + col * FINAL_RANK])
                .sum::<f64>();
            let expected = a[row + col * rows];
            residual_sq += (actual - expected).powi(2);
            norm_sq += expected.powi(2);
        }
    }
    let mut orthogonality_sq = 0.0;
    for left in 0..FINAL_RANK {
        for right in 0..FINAL_RANK {
            let actual = (0..rows)
                .map(|row| q[row + left * rows] * q[row + right * rows])
                .sum::<f64>();
            let expected = f64::from(left == right);
            orthogonality_sq += (actual - expected).powi(2);
        }
    }
    Ok((
        residual_sq.sqrt() / norm_sq.sqrt().max(f64::MIN_POSITIVE),
        orthogonality_sq.sqrt() / (FINAL_RANK as f64).sqrt(),
    ))
}

fn errors_c64(q: &Tensor, r: &Tensor, a: &Tensor) -> Result<(f64, f64), String> {
    let q = q.as_slice::<Complex64>().map_err(to_string)?;
    let r = r.as_slice::<Complex64>().map_err(to_string)?;
    let a = a.as_slice::<Complex64>().map_err(to_string)?;
    let rows = a.len() / FINAL_RANK;
    let mut residual_sq = 0.0;
    let mut norm_sq = 0.0;
    for col in 0..FINAL_RANK {
        for row in 0..rows {
            let actual = (0..FINAL_RANK)
                .map(|inner| q[row + inner * rows] * r[inner + col * FINAL_RANK])
                .sum::<Complex64>();
            let expected = a[row + col * rows];
            residual_sq += (actual - expected).norm_sqr();
            norm_sq += expected.norm_sqr();
        }
    }
    let mut orthogonality_sq = 0.0;
    for left in 0..FINAL_RANK {
        for right in 0..FINAL_RANK {
            let actual = (0..rows)
                .map(|row| q[row + left * rows].conj() * q[row + right * rows])
                .sum::<Complex64>();
            let expected = Complex64::new(f64::from(left == right), 0.0);
            orthogonality_sq += (actual - expected).norm_sqr();
        }
    }
    Ok((
        residual_sq.sqrt() / norm_sq.sqrt().max(f64::MIN_POSITIVE),
        orthogonality_sq.sqrt() / (FINAL_RANK as f64).sqrt(),
    ))
}
