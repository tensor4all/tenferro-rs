use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use serde::Serialize;
use tenferro_cpu::{with_cpu_exec_session, CpuBackend, CpuBackendKind, CpuExecSession};
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::{
    download_tensor, gpu_available, upload_tensor, with_cuda_exec_session, CudaBackend,
    CudaDeviceId, CudaExecSession,
};
use tenferro_linalg::{HouseholderQr, LinalgBackend, QrGauge, QrOptions, TensorLinalgExt};
use tenferro_tensor::{BackendSessionHost, DotGeneralConfig, Tensor, TensorRead};

const SOURCE_COMMIT: &str = "da0775a208006352f6e5eab18bc6bb09ca39a1f6";
const SAMPLE_BATCH: usize = 4;
const CPU_CLOCK_WARMUP_MS: u64 = 50;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Algorithm {
    Compact,
    Bcgs2,
    FullQr,
}

#[derive(Debug)]
struct Config {
    backend: String,
    algorithm: Algorithm,
    rows: usize,
    initial_rank: usize,
    block_width: usize,
    max_rank: usize,
    warmups: usize,
    repetitions: usize,
    seed: u64,
}

#[derive(Serialize)]
struct Record {
    schema: &'static str,
    source_commit: &'static str,
    backend: String,
    algorithm: Algorithm,
    rows: usize,
    initial_rank: usize,
    block_width: usize,
    max_rank: usize,
    appended_blocks: usize,
    final_rank: usize,
    warmups: usize,
    repetitions: usize,
    sample_batch: usize,
    cpu_frequency_mhz: Option<f64>,
    cpu_affinity: Option<String>,
    timings_ms: Vec<f64>,
    reconstruction_relative_error: f64,
    orthogonality_relative_error: f64,
    r_relative_error: f64,
}

trait BenchSession: LinalgBackend {
    fn benchmark_synchronize(&mut self) -> tenferro_tensor::Result<()>;
}

impl BenchSession for CpuExecSession<'_> {
    fn benchmark_synchronize(&mut self) -> tenferro_tensor::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl BenchSession for CudaExecSession<'_> {
    fn benchmark_synchronize(&mut self) -> tenferro_tensor::Result<()> {
        self.runtime().synchronize()
    }
}

fn main() {
    let config = parse_args().unwrap_or_else(|error| panic!("{error}"));
    let (initial_host, blocks_host, accumulated_host) = generate_inputs(&config).unwrap();
    let record = match config.backend.as_str() {
        "faer" => run_cpu(
            CpuBackendKind::Faer,
            &config,
            &initial_host,
            &blocks_host,
            &accumulated_host,
        ),
        "blas" => run_cpu(
            CpuBackendKind::Blas,
            &config,
            &initial_host,
            &blocks_host,
            &accumulated_host,
        ),
        "cuda" => run_cuda(&config, &initial_host, &blocks_host, &accumulated_host),
        other => Err(format!("unsupported backend {other:?}")),
    }
    .unwrap_or_else(|error| panic!("benchmark failed: {error}"));
    println!("{}", serde_json::to_string(&record).unwrap());
}

fn parse_args() -> Result<Config, String> {
    let mut backend = None;
    let mut algorithm = None;
    let mut bond: Option<usize> = None;
    let mut rows: Option<usize> = None;
    let mut initial_rank = 2usize;
    let mut block_width = 3usize;
    let mut max_rank = 32usize;
    let mut warmups = 3usize;
    let mut repetitions = None;
    let mut seed = 7u64;
    let args = env::args()
        .skip(1)
        .filter(|argument| argument != "--bench")
        .collect::<Vec<_>>();
    let mut index = 0usize;
    while index < args.len() {
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("missing value after {}", args[index]))?;
        match args[index].as_str() {
            "--backend" => backend = Some(value.clone()),
            "--algorithm" => {
                algorithm = Some(match value.as_str() {
                    "compact" => Algorithm::Compact,
                    "bcgs2" => Algorithm::Bcgs2,
                    "full-qr" => Algorithm::FullQr,
                    _ => return Err(format!("unknown algorithm {value:?}")),
                });
            }
            "--bond" => bond = Some(parse(value, "bond")?),
            "--rows" => rows = Some(parse(value, "rows")?),
            "--initial-rank" => initial_rank = parse(value, "initial-rank")?,
            "--block-width" => block_width = parse(value, "block-width")?,
            "--max-rank" => max_rank = parse(value, "max-rank")?,
            "--warmups" => warmups = parse(value, "warmups")?,
            "--repetitions" => repetitions = Some(parse(value, "repetitions")?),
            "--seed" => seed = parse(value, "seed")?,
            other => return Err(format!("unknown option {other:?}")),
        }
        index += 2;
    }
    let rows = match (rows, bond) {
        (Some(rows), None) => rows,
        (None, Some(bond)) => bond
            .checked_mul(bond)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| "SRC-derived row count overflowed".to_string())?,
        (Some(_), Some(_)) => return Err("pass either --rows or --bond, not both".into()),
        (None, None) => return Err("one of --rows or --bond is required".into()),
    };
    if initial_rank == 0 || block_width == 0 || initial_rank > max_rank || max_rank > rows {
        return Err("invalid rank/block configuration".into());
    }
    let blocks = (max_rank - initial_rank) / block_width;
    if blocks == 0 {
        return Err("case must contain at least one full append block".into());
    }
    let repetitions = repetitions.unwrap_or(if rows >= 32768 { 3 } else { 5 });
    if repetitions == 0 {
        return Err("repetitions must be positive".into());
    }
    warmups
        .checked_add(repetitions)
        .ok_or_else(|| "warmup/repetition count overflowed".to_string())?;
    Ok(Config {
        backend: backend.ok_or_else(|| "--backend is required".to_string())?,
        algorithm: algorithm.ok_or_else(|| "--algorithm is required".to_string())?,
        rows,
        initial_rank,
        block_width,
        max_rank,
        warmups,
        repetitions,
        seed,
    })
}

fn parse<T: std::str::FromStr>(value: &str, field: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid {field}: {value:?}"))
}

fn generate_inputs(config: &Config) -> Result<(Tensor, Vec<Tensor>, Tensor), String> {
    let blocks = (config.max_rank - config.initial_rank) / config.block_width;
    let final_rank = config.initial_rank + blocks * config.block_width;
    let elements = config
        .rows
        .checked_mul(final_rank)
        .ok_or_else(|| "input element count overflowed".to_string())?;
    let mut state = config.seed;
    let mut values = Vec::with_capacity(elements);
    let scale = (config.rows as f64).sqrt().recip();
    for column in 0..final_rank {
        for row in 0..config.rows {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            let diagonal = if row == column { 1.0 } else { 0.0 };
            values.push((2.0 * unit - 1.0) * scale + diagonal);
        }
    }
    let initial_len = config.rows * config.initial_rank;
    let initial = Tensor::from_vec_col_major(
        vec![config.rows, config.initial_rank],
        values[..initial_len].to_vec(),
    )
    .map_err(|error| error.to_string())?;
    let mut appended = Vec::with_capacity(blocks);
    for block in 0..blocks {
        let start = initial_len + block * config.rows * config.block_width;
        let end = start + config.rows * config.block_width;
        appended.push(
            Tensor::from_vec_col_major(
                vec![config.rows, config.block_width],
                values[start..end].to_vec(),
            )
            .map_err(|error| error.to_string())?,
        );
    }
    let accumulated = Tensor::from_vec_col_major(vec![config.rows, final_rank], values)
        .map_err(|error| error.to_string())?;
    Ok((initial, appended, accumulated))
}

fn run_cpu(
    kind: CpuBackendKind,
    config: &Config,
    initial: &Tensor,
    blocks: &[Tensor],
    accumulated: &Tensor,
) -> Result<Record, String> {
    let mut backend =
        CpuBackend::with_threads_and_kind(1, kind).map_err(|error| error.to_string())?;
    backend.with_backend_session(|session| {
        with_cpu_exec_session(session, |session| {
            run_session(config, session, initial, blocks, accumulated)
        })
        .ok_or_else(|| "CPU execution session unavailable".to_string())?
    })
}

#[cfg(feature = "cuda")]
fn run_cuda(
    config: &Config,
    initial: &Tensor,
    blocks: &[Tensor],
    accumulated: &Tensor,
) -> Result<Record, String> {
    if !gpu_available() {
        return Err("CUDA backend requested but no GPU is available".into());
    }
    let mut backend =
        CudaBackend::new(CudaDeviceId::from_ordinal(0)).map_err(|error| error.to_string())?;
    let initial = upload_tensor(backend.runtime(), initial).map_err(|error| error.to_string())?;
    let blocks = blocks
        .iter()
        .map(|block| upload_tensor(backend.runtime(), block).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let accumulated_device =
        upload_tensor(backend.runtime(), accumulated).map_err(|error| error.to_string())?;
    let (mut record, q, r, reference_r) = backend.with_backend_session(|session| {
        with_cuda_exec_session(session, |session| {
            run_session_outputs(config, session, &initial, &blocks, &accumulated_device)
        })
        .ok_or_else(|| "CUDA execution session unavailable".to_string())?
    })?;
    let q = download_tensor(backend.runtime(), &q).map_err(|error| error.to_string())?;
    let r = download_tensor(backend.runtime(), &r).map_err(|error| error.to_string())?;
    let reference_r =
        download_tensor(backend.runtime(), &reference_r).map_err(|error| error.to_string())?;
    fill_errors(&mut record, &q, &r, accumulated, &reference_r)?;
    Ok(record)
}

#[cfg(not(feature = "cuda"))]
fn run_cuda(
    _config: &Config,
    _initial: &Tensor,
    _blocks: &[Tensor],
    _accumulated: &Tensor,
) -> Result<Record, String> {
    Err("rebuild with --features cuda for --backend cuda".into())
}

fn run_session<B: BenchSession>(
    config: &Config,
    session: &mut B,
    initial: &Tensor,
    blocks: &[Tensor],
    accumulated: &Tensor,
) -> Result<Record, String> {
    let (mut record, q, r, reference_r) =
        run_session_outputs(config, session, initial, blocks, accumulated)?;
    fill_errors(&mut record, &q, &r, accumulated, &reference_r)?;
    Ok(record)
}

fn run_session_outputs<B: BenchSession>(
    config: &Config,
    session: &mut B,
    initial: &Tensor,
    blocks: &[Tensor],
    accumulated: &Tensor,
) -> Result<(Record, Tensor, Tensor, Tensor), String> {
    let total = config
        .warmups
        .checked_add(config.repetitions)
        .ok_or_else(|| "warmup/repetition count overflowed".to_string())?;
    let mut timings_ms = Vec::with_capacity(config.repetitions);
    let mut final_factors = None;
    for iteration in 0..total {
        let mut prepared_batch = Vec::with_capacity(SAMPLE_BATCH);
        for _ in 0..SAMPLE_BATCH {
            prepared_batch.push(prepare(config.algorithm, session, initial)?);
        }
        warm_cpu_clock();
        session
            .benchmark_synchronize()
            .map_err(|error| error.to_string())?;
        let start = Instant::now();
        for prepared in prepared_batch {
            let factors = append_sequence(config.algorithm, session, prepared, blocks)?;
            black_box(&factors);
            final_factors = Some(factors);
        }
        session
            .benchmark_synchronize()
            .map_err(|error| error.to_string())?;
        let elapsed = start.elapsed().as_secs_f64() * 1.0e3;
        if iteration >= config.warmups {
            timings_ms.push(elapsed);
        }
    }
    let cpu_frequency_mhz = cpu0_frequency_mhz();
    let cpu_affinity = process_affinity();
    let (q, r) = canonical_factors(session, final_factors.unwrap())?;
    let reference = session
        .qr_with_options(
            accumulated,
            QrOptions::default().gauge(QrGauge::PositiveDiagonal),
        )
        .map_err(|error| error.to_string())?;
    let reference_r = reference
        .into_iter()
        .nth(1)
        .ok_or_else(|| "one-shot QR did not return R".to_string())?;
    let final_rank = config.initial_rank + blocks.len() * config.block_width;
    Ok((
        Record {
            schema: "tenferro.incremental-householder-qr-benchmark.v1",
            source_commit: SOURCE_COMMIT,
            backend: config.backend.clone(),
            algorithm: config.algorithm,
            rows: config.rows,
            initial_rank: config.initial_rank,
            block_width: config.block_width,
            max_rank: config.max_rank,
            appended_blocks: blocks.len(),
            final_rank,
            warmups: config.warmups,
            repetitions: config.repetitions,
            sample_batch: SAMPLE_BATCH,
            cpu_frequency_mhz,
            cpu_affinity,
            timings_ms,
            reconstruction_relative_error: f64::NAN,
            orthogonality_relative_error: f64::NAN,
            r_relative_error: f64::NAN,
        },
        q,
        r,
        reference_r,
    ))
}

fn warm_cpu_clock() {
    let start = Instant::now();
    let duration = std::time::Duration::from_millis(CPU_CLOCK_WARMUP_MS);
    let mut state = 1u64;
    while start.elapsed() < duration {
        state = black_box(
            state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1),
        );
    }
    black_box(state);
}

fn cpu0_frequency_mhz() -> Option<f64> {
    fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()
        .map(|khz| khz / 1_000.0)
}

fn process_affinity() -> Option<String> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:\t"))
        .map(str::to_owned)
}

enum Prepared {
    Compact(HouseholderQr<Tensor>),
    Bcgs2 { q: Tensor, r: Tensor },
    Full { accumulated: Tensor },
}

fn prepare<B: BenchSession>(
    algorithm: Algorithm,
    session: &mut B,
    initial: &Tensor,
) -> Result<Prepared, String> {
    match algorithm {
        Algorithm::Compact => initial
            .householder_qr(session)
            .map(Prepared::Compact)
            .map_err(|error| error.to_string()),
        Algorithm::Bcgs2 => {
            let (q, r) = pair(session.qr(initial).map_err(|error| error.to_string())?)?;
            Ok(Prepared::Bcgs2 { q, r })
        }
        Algorithm::FullQr => Ok(Prepared::Full {
            accumulated: session
                .to_contiguous_read(TensorRead::from_tensor(initial))
                .map_err(|error| error.to_string())?,
        }),
    }
}

fn append_sequence<B: BenchSession>(
    algorithm: Algorithm,
    session: &mut B,
    prepared: Prepared,
    blocks: &[Tensor],
) -> Result<Prepared, String> {
    match (algorithm, prepared) {
        (Algorithm::Compact, Prepared::Compact(mut state)) => {
            for block in blocks {
                state = state
                    .append_columns(black_box(block), session)
                    .map_err(|error| error.to_string())?;
            }
            Ok(Prepared::Compact(state))
        }
        (Algorithm::Bcgs2, Prepared::Bcgs2 { mut q, mut r }) => {
            for block in blocks {
                (q, r) = bcgs2_append(session, &q, &r, black_box(block))?;
            }
            Ok(Prepared::Bcgs2 { q, r })
        }
        (Algorithm::FullQr, Prepared::Full { mut accumulated }) => {
            for block in blocks {
                accumulated = session
                    .concatenate(&[&accumulated, black_box(block)], 1)
                    .map_err(|error| error.to_string())?;
                let _ = session
                    .qr(&accumulated)
                    .map_err(|error| error.to_string())?;
            }
            Ok(Prepared::Full { accumulated })
        }
        _ => Err("algorithm/state mismatch".into()),
    }
}

fn canonical_factors<B: BenchSession>(
    session: &mut B,
    prepared: Prepared,
) -> Result<(Tensor, Tensor), String> {
    let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    match prepared {
        Prepared::Compact(state) => Ok((
            state
                .q_columns(
                    0..state
                        .r(QrOptions::default(), session)
                        .map_err(|e| e.to_string())?
                        .shape()[0],
                    options,
                    session,
                )
                .map_err(|error| error.to_string())?,
            state
                .r(options, session)
                .map_err(|error| error.to_string())?,
        )),
        Prepared::Bcgs2 { q, r } => {
            let state = HouseholderQr::<Tensor>::from_factors(&q, &r, session)
                .map_err(|error| error.to_string())?;
            let rank = r.shape()[0];
            Ok((
                state
                    .q_columns(0..rank, options, session)
                    .map_err(|e| e.to_string())?,
                state.r(options, session).map_err(|e| e.to_string())?,
            ))
        }
        Prepared::Full { accumulated } => pair(
            session
                .qr_with_options(&accumulated, options)
                .map_err(|e| e.to_string())?,
        ),
    }
}

fn bcgs2_append<B: BenchSession>(
    session: &mut B,
    q: &Tensor,
    r: &Tensor,
    block: &Tensor,
) -> Result<(Tensor, Tensor), String> {
    let qh = session
        .transpose(q, &[1, 0])
        .map_err(|error| error.to_string())?;
    let first = matmul(session, &qh, block)?;
    let first_reconstruction = matmul(session, q, &first)?;
    let first_residual = session
        .sub(block, &first_reconstruction)
        .map_err(|error| error.to_string())?;
    let correction = matmul(session, &qh, &first_residual)?;
    let correction_reconstruction = matmul(session, q, &correction)?;
    let residual = session
        .sub(&first_residual, &correction_reconstruction)
        .map_err(|error| error.to_string())?;
    let projection = session
        .add(&first, &correction)
        .map_err(|error| error.to_string())?;
    let (appended_q, appended_r) = pair(session.qr(&residual).map_err(|error| error.to_string())?)?;
    let new_q = session
        .concatenate(&[q, &appended_q], 1)
        .map_err(|error| error.to_string())?;
    let scalar = session
        .reduce_sum(&projection, &[0, 1])
        .map_err(|error| error.to_string())?;
    let zero = session
        .sub(&scalar, &scalar)
        .map_err(|error| error.to_string())?;
    let bottom_left = session
        .broadcast_in_dim(&zero, &[appended_r.shape()[0], r.shape()[1]], &[])
        .map_err(|error| error.to_string())?;
    let top = session
        .concatenate(&[r, &projection], 1)
        .map_err(|error| error.to_string())?;
    let bottom = session
        .concatenate(&[&bottom_left, &appended_r], 1)
        .map_err(|error| error.to_string())?;
    let new_r = session
        .concatenate(&[&top, &bottom], 0)
        .map_err(|error| error.to_string())?;
    Ok((new_q, new_r))
}

fn matmul<B: BenchSession>(session: &mut B, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, String> {
    session
        .dot_general(
            lhs,
            rhs,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .map_err(|error| error.to_string())
}

fn pair(outputs: Vec<Tensor>) -> Result<(Tensor, Tensor), String> {
    let mut outputs = outputs.into_iter();
    match (outputs.next(), outputs.next(), outputs.next()) {
        (Some(q), Some(r), None) => Ok((q, r)),
        _ => Err("QR returned an unexpected output count".into()),
    }
}

fn fill_errors(
    record: &mut Record,
    q: &Tensor,
    r: &Tensor,
    accumulated: &Tensor,
    reference_r: &Tensor,
) -> Result<(), String> {
    let q = q.as_slice::<f64>().map_err(|error| error.to_string())?;
    let r = r.as_slice::<f64>().map_err(|error| error.to_string())?;
    let a = accumulated
        .as_slice::<f64>()
        .map_err(|error| error.to_string())?;
    let reference_r = reference_r
        .as_slice::<f64>()
        .map_err(|error| error.to_string())?;
    let m = record.rows;
    let k = record.final_rank;
    let n = record.final_rank;
    let mut residual_sq = 0.0;
    let mut norm_sq = 0.0;
    for col in 0..n {
        for row in 0..m {
            let mut value = 0.0;
            for inner in 0..k {
                value += q[row + inner * m] * r[inner + col * k];
            }
            let expected = a[row + col * m];
            residual_sq += (value - expected).powi(2);
            norm_sq += expected.powi(2);
        }
    }
    let mut orthogonality_sq = 0.0;
    for left in 0..k {
        for right in 0..k {
            let mut dot = 0.0;
            for row in 0..m {
                dot += q[row + left * m] * q[row + right * m];
            }
            let expected = if left == right { 1.0 } else { 0.0 };
            orthogonality_sq += (dot - expected).powi(2);
        }
    }
    let r_diff = r
        .iter()
        .zip(reference_r)
        .map(|(lhs, rhs)| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt();
    let r_norm = reference_r
        .iter()
        .map(|value| value.powi(2))
        .sum::<f64>()
        .sqrt();
    record.reconstruction_relative_error =
        residual_sq.sqrt() / norm_sq.sqrt().max(f64::MIN_POSITIVE);
    record.orthogonality_relative_error = orthogonality_sq.sqrt() / (k as f64).sqrt();
    record.r_relative_error = r_diff / r_norm.max(f64::MIN_POSITIVE);
    Ok(())
}
