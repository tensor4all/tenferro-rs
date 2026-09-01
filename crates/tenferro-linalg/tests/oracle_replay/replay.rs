use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use num_complex::{Complex32, Complex64};
use serde_json::Value;
use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_linalg::{HouseholderQr, QrGauge, QrOptions, TracedTensorLinalgExt};
use tenferro_runtime::{DType, DotGeneralConfig, GraphCompiler, Runtime, Tensor, TracedTensor};

use super::db::{self, CaseRecord, ComparisonTolerance, DbTensor};
use super::support::{self, RecordSupport};

#[derive(Clone)]
struct NamedTensor {
    name: String,
    tensor: TracedTensor,
}

struct CaseExecution {
    inputs: BTreeMap<String, TracedTensor>,
    outputs: Vec<NamedTensor>,
}

#[derive(Clone, Copy)]
enum DerivativeKind {
    Jvp,
    Vjp,
    Hvp,
}

#[derive(Debug, Default)]
pub struct ReplayRunSummary {
    pub total_records: usize,
    pub supported_success_records: usize,
    pub expected_error_records: usize,
    pub unsupported_records: usize,
    pub skipped_by_filter_records: usize,
    pub replayed_success_records: usize,
    pub replayed_expected_error_records: usize,
    pub parallel_jobs: usize,
}

#[derive(Clone, Copy)]
struct ReplayConfig {
    include_hvp: bool,
}

#[derive(Clone)]
struct ReplayTask {
    record: CaseRecord,
    kind: ReplayTaskKind,
}

#[derive(Clone, Copy)]
enum ReplayTaskKind {
    Supported,
    ExpectedError,
}

#[derive(Clone, Copy)]
enum ReplayTaskOutcome {
    Supported,
    ExpectedError,
}

impl ReplayConfig {
    fn all_derivatives() -> Self {
        Self { include_hvp: true }
    }

    fn from_env() -> Self {
        Self {
            include_hvp: !env_flag("ORACLE_REPLAY_SKIP_HVP"),
        }
    }
}

pub(super) fn replay_case_id(op: &str, family: &str, case_id: &str) -> Result<(), String> {
    let root = db::default_oracle_db_root().ok_or_else(|| {
        "vendored tensor-ad-oracles root not found; set TENSOR_AD_ORACLES_ROOT".to_string()
    })?;
    let path = root.join("cases").join(op).join(format!("{family}.jsonl"));
    let record = load_case_id(&path, case_id)?;
    replay_case_with_config(&record, ReplayConfig::all_derivatives())
}

pub(super) fn replay_expected_error_case_id(
    op: &str,
    family: &str,
    case_id: &str,
) -> Result<(), String> {
    let root = db::default_oracle_db_root().ok_or_else(|| {
        "vendored tensor-ad-oracles root not found; set TENSOR_AD_ORACLES_ROOT".to_string()
    })?;
    let path = root.join("cases").join(op).join(format!("{family}.jsonl"));
    let record = load_case_id(&path, case_id)?;
    replay_expected_error_case(&record)
}

pub(super) fn replay_supported_cases_from_env() -> Result<Option<ReplayRunSummary>, String> {
    if !env_flag("RUN_ORACLE_REPLAY") {
        return Ok(None);
    }

    let root = db::default_oracle_db_root().ok_or_else(|| {
        "vendored tensor-ad-oracles root not found; set TENSOR_AD_ORACLES_ROOT".to_string()
    })?;
    let op_filter = env::var("ORACLE_REPLAY_OP").ok();
    let family_filter = env::var("ORACLE_REPLAY_FAMILY").ok();
    let case_id_filter = env::var("ORACLE_REPLAY_CASE_ID").ok();
    let limit = env::var("ORACLE_REPLAY_LIMIT")
        .ok()
        .map(|raw| {
            raw.parse::<usize>()
                .map_err(|err| format!("invalid ORACLE_REPLAY_LIMIT={raw}: {err}"))
        })
        .transpose()?;
    let config = ReplayConfig::from_env();
    let mut tasks = Vec::new();
    let mut summary = ReplayRunSummary::default();

    'files: for path in db::case_files(&root)? {
        for record in db::load_case_records(&path)? {
            summary.total_records += 1;
            let support = support::classify_record(&record);
            match support {
                RecordSupport::Supported(_) => summary.supported_success_records += 1,
                RecordSupport::ExpectedError(_) => summary.expected_error_records += 1,
                RecordSupport::Unsupported { .. } => {
                    summary.unsupported_records += 1;
                    continue;
                }
            }

            if !record_matches_filters(
                &record,
                op_filter.as_deref(),
                family_filter.as_deref(),
                case_id_filter.as_deref(),
            ) {
                summary.skipped_by_filter_records += 1;
                continue;
            }

            match support {
                RecordSupport::Supported(_) => tasks.push(ReplayTask {
                    record,
                    kind: ReplayTaskKind::Supported,
                }),
                RecordSupport::ExpectedError(_) => tasks.push(ReplayTask {
                    record,
                    kind: ReplayTaskKind::ExpectedError,
                }),
                RecordSupport::Unsupported { .. } => unreachable!(),
            }

            if limit.is_some_and(|limit| tasks.len() >= limit) {
                break 'files;
            }
        }
    }

    summary.parallel_jobs = replay_worker_count(tasks.len())?;
    for outcome in replay_tasks(&tasks, config, summary.parallel_jobs)? {
        match outcome {
            ReplayTaskOutcome::Supported => summary.replayed_success_records += 1,
            ReplayTaskOutcome::ExpectedError => summary.replayed_expected_error_records += 1,
        }
    }

    Ok(Some(summary))
}

fn replay_tasks(
    tasks: &[ReplayTask],
    config: ReplayConfig,
    jobs: usize,
) -> Result<Vec<ReplayTaskOutcome>, String> {
    if tasks.is_empty() {
        return Ok(Vec::new());
    }
    if jobs <= 1 || tasks.len() <= 1 {
        return tasks
            .iter()
            .map(|task| task.replay(config))
            .collect::<Result<Vec<_>, _>>();
    }

    let next_task = AtomicUsize::new(0);
    let results = Mutex::new(vec![None; tasks.len()]);
    // Keep replay-level parallelism on plain OS threads. CPU backend execution
    // uses Rayon internally and treats managed Rayon scopes as owned execution
    // contexts, so wrapping whole records in a Rayon pool would trip re-entry
    // protection rather than testing numerical replay.
    let spawn_result = std::thread::scope(|scope| {
        for worker_index in 0..jobs {
            let next_task = &next_task;
            let results = &results;
            std::thread::Builder::new()
                .name(format!("oracle-replay-{worker_index}"))
                .spawn_scoped(scope, move || loop {
                    let task_index = next_task.fetch_add(1, Ordering::Relaxed);
                    let Some(task) = tasks.get(task_index) else {
                        break;
                    };
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        task.replay(config)
                    }))
                    .unwrap_or_else(|payload| {
                        Err(format!(
                            "{}: oracle replay worker panicked: {}",
                            task.record.case_id,
                            panic_payload_to_string(payload)
                        ))
                    });
                    match results.lock() {
                        Ok(mut results) => results[task_index] = Some(result),
                        Err(poisoned) => {
                            let mut results = poisoned.into_inner();
                            results[task_index] =
                                Some(Err("oracle replay result buffer was poisoned".to_string()));
                            break;
                        }
                    }
                })
                .map_err(|err| format!("failed to spawn oracle replay worker: {err}"))?;
        }
        Ok::<(), String>(())
    });
    spawn_result?;
    results
        .into_inner()
        .map_err(|_| "oracle replay result buffer was poisoned".to_string())?
        .into_iter()
        .enumerate()
        .map(|(index, result)| {
            result.unwrap_or_else(|| {
                Err(format!(
                    "oracle replay worker did not report result for task {index}"
                ))
            })
        })
        .collect()
}

impl ReplayTask {
    fn replay(&self, config: ReplayConfig) -> Result<ReplayTaskOutcome, String> {
        match self.kind {
            ReplayTaskKind::Supported => {
                replay_case_with_config(&self.record, config)
                    .map_err(|err| format!("{}: {err}", self.record.case_id))?;
                Ok(ReplayTaskOutcome::Supported)
            }
            ReplayTaskKind::ExpectedError => {
                replay_expected_error_case(&self.record)
                    .map_err(|err| format!("{}: {err}", self.record.case_id))?;
                Ok(ReplayTaskOutcome::ExpectedError)
            }
        }
    }
}

fn replay_worker_count(task_count: usize) -> Result<usize, String> {
    let configured = parse_replay_jobs_env(env::var("ORACLE_REPLAY_JOBS").ok().as_deref())?
        .unwrap_or_else(default_replay_jobs);
    Ok(configured.max(1).min(task_count.max(1)))
}

fn default_replay_jobs() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

fn parse_replay_jobs_env(raw: Option<&str>) -> Result<Option<usize>, String> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let jobs = raw
        .trim()
        .parse::<usize>()
        .map_err(|err| format!("invalid ORACLE_REPLAY_JOBS={raw}: {err}"))?;
    if jobs == 0 {
        return Err("ORACLE_REPLAY_JOBS must be greater than zero".to_string());
    }
    Ok(Some(jobs))
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn load_case_id(path: &Path, case_id: &str) -> Result<CaseRecord, String> {
    db::load_case_records(path)?
        .into_iter()
        .find(|record| record.case_id == case_id)
        .ok_or_else(|| format!("case {case_id} not found in {}", path.display()))
}

fn replay_case_with_config(case: &CaseRecord, config: ReplayConfig) -> Result<(), String> {
    match support::classify_record(case) {
        RecordSupport::Supported(_) => {}
        RecordSupport::ExpectedError(_) => {
            return Err(format!("{} is an expected-error record", case.case_id));
        }
        RecordSupport::Unsupported { reason } => {
            return Err(format!("{} is unsupported: {reason}", case.case_id));
        }
    }

    let execution = dispatch_case(case)?;
    let outputs = apply_observable(case, execution.outputs)?;
    let runtime = cpu_runtime_with_linalg()?;
    let ad = ad_context_with_linalg()?;

    for probe in &case.probes {
        let first_order = required_tolerance(case.comparison.first_order(), "first_order", case)?;
        let directions = decode_named_tensors(&probe.direction)?;
        let cotangents = decode_named_tensors(&probe.cotangent)?;

        let jvp_outputs = build_jvp_outputs(
            &ad,
            case,
            &outputs,
            &execution.inputs,
            &directions,
            &probe.pytorch_ref.jvp,
        )
        .map_err(|err| format!("probe {} build JVP: {err}", probe.probe_id))?;
        let jvp_results = eval_named_tensors(&runtime, &jvp_outputs)
            .map_err(|err| format!("probe {} eval JVP: {err}", probe.probe_id))?;
        compare_named_results(
            case,
            DerivativeKind::Jvp,
            &jvp_results,
            &jvp_outputs,
            &probe.pytorch_ref.jvp,
            first_order,
            &format!("probe {} JVP", probe.probe_id),
        )?;

        let scalar = cotangent_scalar(case, &outputs, &cotangents, &probe.probe_id)
            .map_err(|err| format!("probe {} build cotangent scalar: {err}", probe.probe_id))?;
        let vjp_outputs =
            build_grad_outputs(&ad, &scalar, &execution.inputs, &probe.pytorch_ref.vjp)
                .map_err(|err| format!("probe {} build VJP: {err}", probe.probe_id))?;
        let vjp_results = eval_named_tensors(&runtime, &vjp_outputs)
            .map_err(|err| format!("probe {} eval VJP: {err}", probe.probe_id))?;
        compare_named_results(
            case,
            DerivativeKind::Vjp,
            &vjp_results,
            &vjp_outputs,
            &probe.pytorch_ref.vjp,
            first_order,
            &format!("probe {} VJP", probe.probe_id),
        )?;

        if config.include_hvp {
            let Some(expected_hvp) = &probe.pytorch_ref.hvp else {
                continue;
            };
            let second_order =
                required_tolerance(case.comparison.second_order(), "second_order", case)?;
            let hvp_outputs = build_hvp_outputs(
                &ad,
                &vjp_outputs,
                &execution.inputs,
                &directions,
                expected_hvp,
            )
            .map_err(|err| format!("probe {} build HVP: {err}", probe.probe_id))?;
            let hvp_results = eval_named_tensors(&runtime, &hvp_outputs)
                .map_err(|err| format!("probe {} eval HVP: {err}", probe.probe_id))?;
            compare_named_results(
                case,
                DerivativeKind::Hvp,
                &hvp_results,
                &hvp_outputs,
                expected_hvp,
                second_order,
                &format!("probe {} HVP", probe.probe_id),
            )?;
        }
    }

    Ok(())
}

fn replay_expected_error_case(case: &CaseRecord) -> Result<(), String> {
    match support::classify_record(case) {
        RecordSupport::ExpectedError(support::ExpectedErrorKind::GaugeIllDefined) => {}
        RecordSupport::Supported(kind) => {
            return Err(format!(
                "{} is supported success record {kind:?}, not expected-error",
                case.case_id
            ));
        }
        RecordSupport::Unsupported { reason } => {
            return Err(format!("{} is unsupported: {reason}", case.case_id));
        }
    }

    let error = case
        .comparison
        .error()
        .ok_or_else(|| format!("{}: expected error comparison schema", case.case_id))?;
    if error.kind != "expect_error" {
        return Err(format!(
            "{}: expected error comparison kind expect_error, got {}",
            case.case_id, error.kind
        ));
    }
    if error.reason_code != "gauge_ill_defined" {
        return Err(format!(
            "{}: expected gauge_ill_defined reason, got {}",
            case.case_id, error.reason_code
        ));
    }
    if !case.probes.is_empty() {
        return Err(format!(
            "{}: expected-error records should not carry derivative probes",
            case.case_id
        ));
    }
    Ok(())
}

fn record_matches_filters(
    record: &CaseRecord,
    op: Option<&str>,
    family: Option<&str>,
    case_id: Option<&str>,
) -> bool {
    op.is_none_or(|expected| record.op == expected)
        && family.is_none_or(|expected| record.family == expected)
        && case_id.is_none_or(|expected| record.case_id == expected)
}

fn env_flag(name: &str) -> bool {
    env::var(name).is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes"))
}

fn ad_context_with_linalg() -> Result<AdContext, String> {
    AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().map_err(to_string)?)
        .map_err(to_string)?
        .build()
        .map_err(to_string)
}

fn cpu_runtime_with_linalg() -> Result<Runtime, String> {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).map_err(to_string)?)
        .map_err(to_string)?;
    builder
        .install_extension_module(
            tenferro_linalg::extension_module::<CpuBackend>(
                tenferro_cpu::runtime_engine_id().map_err(to_string)?,
            )
            .map_err(to_string)?,
        )
        .map_err(to_string)?;
    builder.build().map_err(to_string)
}

fn dispatch_case(case: &CaseRecord) -> Result<CaseExecution, String> {
    let inputs = decode_case_inputs(case)?;
    if case.op == "incremental_householder_qr" {
        return dispatch_incremental_householder_qr(case, inputs);
    }
    let a = required_input(&inputs, "a", case)?;
    match case.op.as_str() {
        "solve" | "solve_ex" => {
            let b = required_input(&inputs, "b", case)?;
            let rhs_core_rank = rhs_core_rank(a, b);
            let b_tf = oracle_to_tenferro(b, rhs_core_rank)?;
            let b_solve = promote_vector_rhs_for_solve(&b_tf, rhs_core_rank)?;
            let solution = oracle_to_tenferro(a, 2)?
                .solve(&b_solve)
                .map_err(to_string)?;
            let solution = demote_vector_solution_after_solve(&solution, rhs_core_rank)?;
            let output_name = if case.op == "solve_ex" {
                "output_0"
            } else {
                "value"
            };
            Ok(CaseExecution {
                inputs,
                outputs: vec![NamedTensor {
                    name: output_name.to_string(),
                    tensor: tenferro_to_oracle(&solution, rhs_core_rank)?,
                }],
            })
        }
        "solve_triangular" => {
            let b = required_input(&inputs, "b", case)?;
            let rhs_core_rank = rhs_core_rank(a, b);
            let left_side = bool_kwarg(case, "left")?.unwrap_or(true);
            let lower = !bool_kwarg(case, "upper")?.unwrap_or(false);
            let unit_diagonal = bool_kwarg(case, "unitriangular")?.unwrap_or(false);
            let b_tf = oracle_to_tenferro(b, rhs_core_rank)?;
            let b_solve = promote_vector_rhs_for_triangular_solve(&b_tf, rhs_core_rank, left_side)?;
            let solution = oracle_to_tenferro(a, 2)?
                .triangular_solve(&b_solve, left_side, lower, false, unit_diagonal)
                .map_err(to_string)?;
            let solution =
                demote_vector_solution_after_triangular_solve(&solution, rhs_core_rank, left_side)?;
            Ok(CaseExecution {
                inputs,
                outputs: vec![NamedTensor {
                    name: "value".to_string(),
                    tensor: tenferro_to_oracle(&solution, rhs_core_rank)?,
                }],
            })
        }
        "cholesky" | "cholesky_ex" => {
            let upper = bool_kwarg(case, "upper")?.unwrap_or(false);
            let a_tf = hermitian_wrapper_tenferro(&oracle_to_tenferro(a, 2)?)?;
            let factor = a_tf.cholesky().map_err(to_string)?;
            let factor = if upper {
                adjoint_tenferro_matrix_axes(&factor)?
            } else {
                factor
            };
            let output_name = if case.op == "cholesky_ex" {
                "output_0"
            } else {
                "value"
            };
            single_output(inputs, output_name, tenferro_to_oracle(&factor, 2)?)
        }
        "qr" => {
            let a_tf = oracle_to_tenferro(a, 2)?;
            let (q_tf, r_tf) = a_tf.qr().map_err(to_string)?;
            Ok(CaseExecution {
                inputs,
                outputs: vec![
                    named("output_0", tenferro_to_oracle(&q_tf, 2)?),
                    named("output_1", tenferro_to_oracle(&r_tf, 2)?),
                ],
            })
        }
        "slogdet" => {
            let a_tf = oracle_to_tenferro(a, 2)?;
            let (sign_tf, logabsdet_tf) = a_tf.slogdet().map_err(to_string)?;
            Ok(CaseExecution {
                inputs,
                outputs: vec![
                    named("output_0", tenferro_to_oracle(&sign_tf, 0)?),
                    named("output_1", tenferro_to_oracle(&logabsdet_tf, 0)?),
                ],
            })
        }
        "det" => {
            let a_tf = oracle_to_tenferro(a, 2)?;
            single_output(
                inputs,
                "value",
                tenferro_to_oracle(&a_tf.det().map_err(to_string)?, 0)?,
            )
        }
        "inv" | "inv_ex" => {
            let a_tf = oracle_to_tenferro(a, 2)?;
            let output_name = if case.op == "inv_ex" {
                "output_0"
            } else {
                "value"
            };
            single_output(
                inputs,
                output_name,
                tenferro_to_oracle(&a_tf.inv().map_err(to_string)?, 2)?,
            )
        }
        "lu" => {
            if bool_kwarg(case, "pivot")? != Some(true) {
                return Err(format!(
                    "{}: only pivot=true LU records are replayed",
                    case.case_id
                ));
            }
            let a_tf = oracle_to_tenferro(a, 2)?;
            let (p_tf, l_tf, u_tf, _parity_tf) = a_tf.lu().map_err(to_string)?;
            Ok(CaseExecution {
                inputs,
                outputs: vec![
                    named("output_0", tenferro_to_oracle(&p_tf, 2)?),
                    named("output_1", tenferro_to_oracle(&l_tf, 2)?),
                    named("output_2", tenferro_to_oracle(&u_tf, 2)?),
                ],
            })
        }
        "full_pivot_lu" => {
            let a_tf = oracle_to_tenferro(a, 2)?;
            let (_p_tf, l_tf, u_tf, _q_tf, _parity_tf) = a_tf.full_piv_lu().map_err(to_string)?;
            Ok(CaseExecution {
                inputs,
                outputs: vec![
                    named("l", tenferro_to_oracle(&l_tf, 2)?),
                    named("u", tenferro_to_oracle(&u_tf, 2)?),
                ],
            })
        }
        "svd" => {
            let a_tf = oracle_to_tenferro(a, 2)?;
            let (u_tf, s_tf, vh_tf) = a_tf.svd().map_err(to_string)?;
            svd_observable_execution(case, inputs, &u_tf, &s_tf, &vh_tf)
        }
        "eigh" => {
            let a_tf = hermitian_wrapper_tenferro(&oracle_to_tenferro(a, 2)?)?;
            let (values_tf, vectors_tf) = a_tf.eigh().map_err(to_string)?;
            Ok(CaseExecution {
                inputs,
                outputs: vec![
                    named("values", tenferro_to_oracle(&values_tf, 1)?),
                    named(
                        "vectors",
                        tenferro_to_oracle(&vectors_tf.abs().map_err(to_string)?, 2)?,
                    ),
                ],
            })
        }
        "eig" => {
            let a_tf = oracle_to_tenferro(a, 2)?;
            let (values_tf, vectors_tf) = a_tf.eig().map_err(to_string)?;
            Ok(CaseExecution {
                inputs,
                outputs: vec![
                    named("values", tenferro_to_oracle(&values_tf, 1)?),
                    named(
                        "vectors",
                        tenferro_to_oracle(&vectors_tf.abs().map_err(to_string)?, 2)?,
                    ),
                ],
            })
        }
        "pinv" => {
            let a_tf = oracle_to_tenferro(a, 2)?;
            let inverse = match optional_f64_kwarg(case, "rtol")? {
                Some(rtol) => a_tf.pinv_with_rtol(rtol).map_err(to_string)?,
                None => a_tf.pinv().map_err(to_string)?,
            };
            single_output(inputs, "value", tenferro_to_oracle(&inverse, 2)?)
        }
        "norm" | "vector_norm" | "matrix_norm" => {
            let (ord, dim, keepdim) = norm_arguments(case, a.rank)?;
            let dim_ref = dim.as_deref();
            let value = a.norm(ord, dim_ref, keepdim).map_err(to_string)?;
            single_output(inputs, "value", value)
        }
        "lstsq_grad_oriented" => {
            let b = required_input(&inputs, "b", case)?;
            let rhs_core_rank = rhs_core_rank(a, b);
            let a_tf = oracle_to_tenferro(a, 2)?;
            let b_tf = oracle_to_tenferro(b, rhs_core_rank)?;
            let b_lstsq = promote_vector_rhs_for_solve(&b_tf, rhs_core_rank)?;
            let solution_lstsq = a_tf.lstsq(&b_lstsq).map_err(to_string)?;
            let residuals = lstsq_residuals(&a_tf, &b_lstsq, &solution_lstsq, rhs_core_rank)?;
            let solution = demote_vector_solution_after_solve(&solution_lstsq, rhs_core_rank)?;
            Ok(CaseExecution {
                inputs,
                outputs: vec![
                    named("output_0", tenferro_to_oracle(&solution, rhs_core_rank)?),
                    named("output_1", tenferro_to_oracle(&residuals, 1)?),
                ],
            })
        }
        other => Err(format!(
            "{}: replay dispatch for op {other} is not implemented yet",
            case.case_id
        )),
    }
}

fn dispatch_incremental_householder_qr(
    case: &CaseRecord,
    inputs: BTreeMap<String, TracedTensor>,
) -> Result<CaseExecution, String> {
    let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    let state = match case.family.as_str() {
        "factor_qr" | "selected_q_columns" | "r" => {
            let a = oracle_to_tenferro(required_input(&inputs, "a", case)?, 2)?;
            a.householder_qr().map_err(to_string)?
        }
        "append_qr" => {
            let a = oracle_to_tenferro(required_input(&inputs, "a", case)?, 2)?;
            let b = oracle_to_tenferro(required_input(&inputs, "b", case)?, 2)?;
            a.householder_qr()
                .map_err(to_string)?
                .append_columns(&b)
                .map_err(to_string)?
        }
        "from_factors_qr" => {
            let q = oracle_to_tenferro(required_input(&inputs, "q", case)?, 2)?;
            let r = oracle_to_tenferro(required_input(&inputs, "r", case)?, 2)?;
            HouseholderQr::<TracedTensor>::from_factors(&q, &r).map_err(to_string)?
        }
        other => {
            return Err(format!(
                "{}: incremental Householder QR family {other} is not implemented",
                case.case_id
            ));
        }
    };

    let outputs = match case.family.as_str() {
        "selected_q_columns" => {
            let start = required_usize_kwarg(case, "start")?;
            let end = required_usize_kwarg(case, "end")?;
            let q = state.q_columns(start..end, options).map_err(to_string)?;
            vec![named("q", tenferro_to_oracle(&q, 2)?)]
        }
        "r" => {
            let r = state.r(options).map_err(to_string)?;
            vec![named("r", tenferro_to_oracle(&r, 2)?)]
        }
        "factor_qr" | "append_qr" | "from_factors_qr" => {
            let width = incremental_qr_thin_width(case)?;
            let q = state.q_columns(0..width, options).map_err(to_string)?;
            let r = state.r(options).map_err(to_string)?;
            vec![
                named("q", tenferro_to_oracle(&q, 2)?),
                named("r", tenferro_to_oracle(&r, 2)?),
            ]
        }
        _ => unreachable!("family was validated while constructing the state"),
    };
    Ok(CaseExecution { inputs, outputs })
}

fn incremental_qr_thin_width(case: &CaseRecord) -> Result<usize, String> {
    let shape = |name: &str| {
        case.inputs
            .get(name)
            .map(|tensor| tensor.shape.as_slice())
            .ok_or_else(|| format!("{}: missing input {name}", case.case_id))
    };
    let (rows, cols) = match case.family.as_str() {
        "factor_qr" => {
            let a = shape("a")?;
            (a[0], a[1])
        }
        "append_qr" => {
            let a = shape("a")?;
            let b = shape("b")?;
            (a[0], a[1] + b[1])
        }
        "from_factors_qr" => {
            let q = shape("q")?;
            let r = shape("r")?;
            (q[0], r[1])
        }
        other => return Err(format!("{}: no thin-Q width for {other}", case.case_id)),
    };
    Ok(rows.min(cols))
}

fn svd_observable_execution(
    case: &CaseRecord,
    inputs: BTreeMap<String, TracedTensor>,
    u_tf: &TracedTensor,
    s_tf: &TracedTensor,
    vh_tf: &TracedTensor,
) -> Result<CaseExecution, String> {
    let outputs = match case.observable.kind.as_str() {
        "svd_s" => vec![named("s", tenferro_to_oracle(s_tf, 1)?)],
        "svd_u_abs" => vec![named(
            "u",
            tenferro_to_oracle(&u_tf.abs().map_err(to_string)?, 2)?,
        )],
        "svd_vh_abs" => vec![
            named("s", tenferro_to_oracle(s_tf, 1)?),
            named(
                "vh",
                tenferro_to_oracle(&vh_tf.abs().map_err(to_string)?, 2)?,
            ),
        ],
        "svd_uvh_product" => {
            let uvh_tf = matmul_tenferro_matrix_axes(u_tf, vh_tf)?;
            vec![
                named("s", tenferro_to_oracle(s_tf, 1)?),
                named("uvh", tenferro_to_oracle(&uvh_tf, 2)?),
            ]
        }
        other => {
            return Err(format!(
                "{}: SVD observable {other} is not implemented",
                case.case_id
            ));
        }
    };
    Ok(CaseExecution { inputs, outputs })
}

fn decode_case_inputs(case: &CaseRecord) -> Result<BTreeMap<String, TracedTensor>, String> {
    let mut inputs = BTreeMap::new();
    for (name, tensor_data) in &case.inputs {
        inputs.insert(name.clone(), decode_traced_tensor(tensor_data)?);
    }
    Ok(inputs)
}

fn decode_named_tensors(
    tensors: &BTreeMap<String, DbTensor>,
) -> Result<BTreeMap<String, TracedTensor>, String> {
    tensors
        .iter()
        .map(|(name, tensor)| Ok((name.clone(), decode_traced_tensor(tensor)?)))
        .collect()
}

fn decode_traced_tensor(tensor: &DbTensor) -> Result<TracedTensor, String> {
    TracedTensor::from_tensor_concrete_shape(decode_tensor(tensor)?).map_err(to_string)
}

fn decode_tensor(tensor: &DbTensor) -> Result<Tensor, String> {
    match tensor.dtype.as_str() {
        "float32" => Ok(Tensor::from_vec_col_major(
            tensor.shape.clone(),
            tensor_data_as_col_major::<f32>(tensor)?,
        )
        .map_err(to_string)?),
        "float64" => Ok(Tensor::from_vec_col_major(
            tensor.shape.clone(),
            tensor_data_as_col_major::<f64>(tensor)?,
        )
        .map_err(to_string)?),
        "complex64" => Ok(Tensor::from_vec_col_major(
            tensor.shape.clone(),
            complex_tensor_data_as_col_major::<f32>(tensor)?,
        )
        .map_err(to_string)?),
        "complex128" => Ok(Tensor::from_vec_col_major(
            tensor.shape.clone(),
            complex_tensor_data_as_col_major::<f64>(tensor)?,
        )
        .map_err(to_string)?),
        other => Err(format!("unsupported oracle tensor dtype {other}")),
    }
}

fn tensor_data_as_col_major<T>(tensor: &DbTensor) -> Result<Vec<T>, String>
where
    T: FloatDecode + Clone,
{
    let values = tensor
        .data
        .iter()
        .map(T::decode_value)
        .collect::<Result<Vec<_>, _>>()?;
    validate_data_len(tensor, values.len(), 1)?;
    tensor_data_ordered_as_col_major(tensor, values)
}

fn complex_tensor_data_as_col_major<T>(
    tensor: &DbTensor,
) -> Result<Vec<num_complex::Complex<T>>, String>
where
    T: FloatDecode + Clone,
{
    let values = tensor
        .data
        .iter()
        .map(|value| {
            let pair = value
                .as_array()
                .ok_or_else(|| format!("complex {} entry must be [re, im]", tensor.dtype))?;
            if pair.len() != 2 {
                return Err(format!("complex {} entry must be [re, im]", tensor.dtype));
            }
            Ok(num_complex::Complex::new(
                T::decode_value(&pair[0])?,
                T::decode_value(&pair[1])?,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_data_len(tensor, values.len(), 1)?;
    tensor_data_ordered_as_col_major(tensor, values)
}

trait FloatDecode: Sized {
    fn decode_value(value: &serde_json::Value) -> Result<Self, String>;
}

impl FloatDecode for f32 {
    fn decode_value(value: &serde_json::Value) -> Result<Self, String> {
        f64::decode_value(value).map(|value| value as f32)
    }
}

impl FloatDecode for f64 {
    fn decode_value(value: &serde_json::Value) -> Result<Self, String> {
        match value {
            serde_json::Value::Number(number) => number
                .as_f64()
                .ok_or_else(|| format!("expected finite JSON number, got {value}")),
            serde_json::Value::String(text) if text == "NaN" => Ok(f64::NAN),
            serde_json::Value::String(text) if text == "Infinity" => Ok(f64::INFINITY),
            serde_json::Value::String(text) if text == "-Infinity" => Ok(f64::NEG_INFINITY),
            _ => Err(format!("expected numeric JSON value, got {value}")),
        }
    }
}

fn validate_data_len(tensor: &DbTensor, actual_len: usize, block: usize) -> Result<(), String> {
    let expected = checked_product(&tensor.shape)?
        .checked_mul(block)
        .ok_or_else(|| format!("{} data length overflow", tensor.dtype))?;
    if actual_len != expected {
        return Err(format!(
            "{} data length {} does not match shape product {}",
            tensor.dtype, actual_len, expected
        ));
    }
    Ok(())
}

fn tensor_data_ordered_as_col_major<T: Clone>(
    tensor: &DbTensor,
    values: Vec<T>,
) -> Result<Vec<T>, String> {
    match tensor.order.as_str() {
        "row_major" => row_major_to_col_major(&values, &tensor.shape),
        "col_major" => Ok(values),
        other => Err(format!("unsupported oracle tensor storage order {other}")),
    }
}

fn row_major_to_col_major<T: Clone>(values: &[T], shape: &[usize]) -> Result<Vec<T>, String> {
    let total = checked_product(shape)?;
    if values.len() != total {
        return Err(format!(
            "oracle tensor value length mismatch: expected {total}, got {}",
            values.len()
        ));
    }
    if total == 0 {
        return Ok(Vec::new());
    }
    let rank = shape.len();
    if rank == 0 {
        return Ok(values.to_vec());
    }

    let mut row_strides = vec![1usize; rank];
    let mut col_strides = vec![1usize; rank];
    for index in (0..rank.saturating_sub(1)).rev() {
        row_strides[index] = row_strides[index + 1]
            .checked_mul(shape[index + 1])
            .ok_or_else(|| "row-major stride overflow".to_string())?;
    }
    for index in 1..rank {
        col_strides[index] = col_strides[index - 1]
            .checked_mul(shape[index - 1])
            .ok_or_else(|| "column-major stride overflow".to_string())?;
    }

    let mut result = values.to_vec();
    for (row_index, value) in values.iter().enumerate() {
        let mut remaining = row_index;
        let mut col_index = 0usize;
        for dim in 0..rank {
            let coord = remaining / row_strides[dim];
            remaining %= row_strides[dim];
            col_index = col_index
                .checked_add(
                    coord
                        .checked_mul(col_strides[dim])
                        .ok_or_else(|| "column-major offset overflow".to_string())?,
                )
                .ok_or_else(|| "column-major offset overflow".to_string())?;
        }
        result[col_index] = value.clone();
    }
    Ok(result)
}

fn apply_observable(
    case: &CaseRecord,
    outputs: Vec<NamedTensor>,
) -> Result<Vec<NamedTensor>, String> {
    match case.observable.kind.as_str() {
        "identity" => Ok(outputs),
        "svd_s"
        | "svd_u_abs"
        | "svd_vh_abs"
        | "svd_uvh_product"
        | "eigh_values_vectors_abs"
        | "eig_values_vectors_abs" => Ok(outputs),
        other => Err(format!(
            "{}: observable {other} is not implemented yet",
            case.case_id
        )),
    }
}

fn build_jvp_outputs(
    ad: &AdContext,
    _case: &CaseRecord,
    outputs: &[NamedTensor],
    inputs: &BTreeMap<String, TracedTensor>,
    directions: &BTreeMap<String, TracedTensor>,
    expected: &BTreeMap<String, DbTensor>,
) -> Result<Vec<NamedTensor>, String> {
    let mut tangents = Vec::new();
    for output in outputs {
        if !expected.contains_key(&output.name) {
            continue;
        }
        let expected_output = expected.get(&output.name).expect("expected output");
        if checked_product(&expected_output.shape)? == 0 {
            tangents.push(NamedTensor {
                name: output.name.clone(),
                tensor: zero_traced_tensor(output.tensor.dtype, expected_output.shape.clone())?,
            });
            continue;
        }
        let tangent =
            directional_jvp(ad, &output.tensor, inputs, directions)?.unwrap_or_else(|| {
                zero_traced_tensor(output.tensor.dtype, expected_output.shape.clone())
                    .expect("zero fallback should build")
            });
        tangents.push(NamedTensor {
            name: output.name.clone(),
            tensor: tangent,
        });
    }
    Ok(tangents)
}

fn build_grad_outputs(
    ad: &AdContext,
    scalar: &TracedTensor,
    inputs: &BTreeMap<String, TracedTensor>,
    expected: &BTreeMap<String, DbTensor>,
) -> Result<Vec<NamedTensor>, String> {
    let mut gradients = Vec::with_capacity(inputs.len());
    for (name, input) in inputs {
        if expected
            .get(name)
            .is_some_and(|tensor| checked_product(&tensor.shape).is_ok_and(|len| len == 0))
        {
            let expected_shape = expected.get(name).expect("expected gradient").shape.clone();
            gradients.push(NamedTensor {
                name: name.clone(),
                tensor: zero_traced_tensor(input.dtype, expected_shape)?,
            });
            continue;
        }
        let gradient = ad
            .grad_optional(scalar, input)
            .map_err(|err| format!("failed to build VJP for input {name}: {err}"))?
            .unwrap_or_else(|| {
                zero_traced_tensor(input.dtype, tensor_shape(input))
                    .expect("zero fallback should build")
            });
        gradients.push(NamedTensor {
            name: name.clone(),
            tensor: gradient,
        });
    }
    Ok(gradients)
}

fn build_hvp_outputs(
    ad: &AdContext,
    gradients: &[NamedTensor],
    inputs: &BTreeMap<String, TracedTensor>,
    directions: &BTreeMap<String, TracedTensor>,
    expected: &BTreeMap<String, DbTensor>,
) -> Result<Vec<NamedTensor>, String> {
    let mut hvps = Vec::with_capacity(gradients.len());
    for gradient in gradients {
        let fallback_shape = inputs
            .get(&gradient.name)
            .map(tensor_shape)
            .ok_or_else(|| format!("missing primal input {}", gradient.name))?;
        if expected
            .get(&gradient.name)
            .is_some_and(|tensor| checked_product(&tensor.shape).is_ok_and(|len| len == 0))
        {
            let expected_shape = expected
                .get(&gradient.name)
                .expect("expected HVP")
                .shape
                .clone();
            hvps.push(NamedTensor {
                name: gradient.name.clone(),
                tensor: zero_traced_tensor(gradient.tensor.dtype, expected_shape)?,
            });
            continue;
        }
        let tangent =
            directional_jvp(ad, &gradient.tensor, inputs, directions)?.unwrap_or_else(|| {
                zero_traced_tensor(gradient.tensor.dtype, fallback_shape)
                    .expect("zero fallback should build")
            });
        hvps.push(NamedTensor {
            name: gradient.name.clone(),
            tensor: tangent,
        });
    }
    Ok(hvps)
}

fn directional_jvp(
    ad: &AdContext,
    output: &TracedTensor,
    inputs: &BTreeMap<String, TracedTensor>,
    directions: &BTreeMap<String, TracedTensor>,
) -> Result<Option<TracedTensor>, String> {
    let mut tangent_sum: Option<TracedTensor> = None;
    for (name, input) in inputs {
        let Some(direction) = directions.get(name) else {
            continue;
        };
        let Some(tangent) = ad
            .jvp_optional(output, input, direction)
            .map_err(|err| format!("failed to build JVP for input {name}: {err}"))?
        else {
            continue;
        };
        tangent_sum = Some(match tangent_sum {
            Some(current) => (&current + &tangent).map_err(to_string)?,
            None => tangent,
        });
    }
    Ok(tangent_sum)
}

fn cotangent_scalar(
    case: &CaseRecord,
    outputs: &[NamedTensor],
    cotangents: &BTreeMap<String, TracedTensor>,
    probe_id: &str,
) -> Result<TracedTensor, String> {
    let mut scalar_terms = Vec::new();
    for output in outputs {
        let Some(cotangent) = cotangents.get(&output.name) else {
            continue;
        };
        let cotangent = align_cotangent_dtype(&output.tensor, cotangent)?;
        let cotangent = match output.tensor.dtype {
            DType::C32 | DType::C64 => cotangent.conj().map_err(to_string)?,
            DType::F32 | DType::F64 => cotangent,
            _ => continue,
        };
        let product = (&output.tensor * &cotangent).map_err(to_string)?;
        let axes: Vec<usize> = (0..product.rank).collect();
        scalar_terms.push(product.reduce_sum(Some(&axes)).map_err(to_string)?);
    }

    let mut iter = scalar_terms.into_iter();
    let Some(mut scalar) = iter.next() else {
        return Err(format!(
            "{} probe {probe_id}: no outputs to build cotangent scalar",
            case.case_id
        ));
    };
    for term in iter {
        scalar = (&scalar + &term).map_err(to_string)?;
    }
    Ok(scalar)
}

fn eval_named_tensors(runtime: &Runtime, outputs: &[NamedTensor]) -> Result<Vec<Tensor>, String> {
    let traced = outputs
        .iter()
        .map(|output| &output.tensor)
        .collect::<Vec<_>>();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&traced).map_err(to_string)?;
    runtime.run_compiled(&program, &[]).map_err(to_string)
}

fn compare_named_results(
    _case: &CaseRecord,
    _derivative: DerivativeKind,
    actual: &[Tensor],
    outputs: &[NamedTensor],
    expected: &BTreeMap<String, DbTensor>,
    tolerance: &ComparisonTolerance,
    context: &str,
) -> Result<(), String> {
    let actual_names = outputs
        .iter()
        .map(|output| output.name.as_str())
        .collect::<BTreeSet<_>>();
    let expected_names = expected.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if actual_names != expected_names {
        return Err(format!(
            "{context}: output name mismatch: actual {:?} vs expected {:?}",
            actual_names, expected_names
        ));
    }
    if actual.len() != outputs.len() {
        return Err(format!(
            "{context}: output count mismatch: actual {} vs expected {}",
            actual.len(),
            outputs.len()
        ));
    }
    for (tensor, output) in actual.iter().zip(outputs.iter()) {
        let expected_tensor = expected.get(&output.name).ok_or_else(|| {
            format!(
                "{context}: missing expected tensor for output {}",
                output.name
            )
        })?;
        compare_tensor(tensor, expected_tensor, tolerance.rtol, tolerance.atol)
            .map_err(|err| format!("{context} output {}: {err}", output.name))?;
    }
    Ok(())
}

fn compare_tensor(
    actual: &Tensor,
    expected: &DbTensor,
    rtol: f64,
    atol: f64,
) -> Result<(), String> {
    if actual.shape() != expected.shape.as_slice() {
        return Err(format!(
            "shape mismatch: actual {:?} vs expected {:?}",
            actual.shape(),
            expected.shape
        ));
    }
    match actual {
        Tensor::F32(_) => compare_real_slice(
            actual.as_slice::<f32>().map_err(to_string)?,
            &tensor_data_as_col_major::<f32>(expected)?,
            rtol,
            atol,
        ),
        Tensor::F64(_) => compare_real_slice(
            actual.as_slice::<f64>().map_err(to_string)?,
            &tensor_data_as_col_major::<f64>(expected)?,
            rtol,
            atol,
        ),
        Tensor::C32(_) => compare_complex_slice(
            actual.as_slice::<Complex32>().map_err(to_string)?,
            &complex_tensor_data_as_col_major::<f32>(expected)?,
            rtol,
            atol,
        ),
        Tensor::C64(_) => compare_complex_slice(
            actual.as_slice::<Complex64>().map_err(to_string)?,
            &complex_tensor_data_as_col_major::<f64>(expected)?,
            rtol,
            atol,
        ),
        _ => Err(format!(
            "unsupported actual tensor dtype {:?}",
            actual.dtype()
        )),
    }
}

trait CompareReal {
    fn to_f64(self) -> f64;
}

impl CompareReal for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl CompareReal for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}

fn compare_real_slice<T: CompareReal + Copy>(
    actual: &[T],
    expected: &[T],
    rtol: f64,
    atol: f64,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "length mismatch: actual {} vs expected {}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate()
    {
        let actual_value = actual_value.to_f64();
        let expected_value = expected_value.to_f64();
        if actual_value == expected_value || (actual_value.is_nan() && expected_value.is_nan()) {
            continue;
        }
        let diff = (actual_value - expected_value).abs();
        let limit = atol + rtol * expected_value.abs();
        if diff > limit {
            return Err(format!(
                "mismatch at flat index {index}: actual={actual_value}, expected={expected_value}, diff={diff}, limit={limit}"
            ));
        }
    }
    Ok(())
}

trait CompareComplex {
    fn re_f64(self) -> f64;
    fn im_f64(self) -> f64;
}

impl CompareComplex for Complex32 {
    fn re_f64(self) -> f64 {
        f64::from(self.re)
    }

    fn im_f64(self) -> f64 {
        f64::from(self.im)
    }
}

impl CompareComplex for Complex64 {
    fn re_f64(self) -> f64 {
        self.re
    }

    fn im_f64(self) -> f64 {
        self.im
    }
}

fn compare_complex_slice<T: CompareComplex + Copy>(
    actual: &[T],
    expected: &[T],
    rtol: f64,
    atol: f64,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "length mismatch: actual {} vs expected {}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate()
    {
        let actual_re = actual_value.re_f64();
        let actual_im = actual_value.im_f64();
        let expected_re = expected_value.re_f64();
        let expected_im = expected_value.im_f64();
        if (actual_re == expected_re && actual_im == expected_im)
            || ((actual_re.is_nan() || actual_im.is_nan())
                && (expected_re.is_nan() || expected_im.is_nan()))
        {
            continue;
        }
        let diff = ((actual_re - expected_re).powi(2) + (actual_im - expected_im).powi(2)).sqrt();
        let expected_norm = (expected_re.powi(2) + expected_im.powi(2)).sqrt();
        let limit = atol + rtol * expected_norm;
        if diff > limit {
            return Err(format!(
                "mismatch at flat index {index}: actual=({actual_re}, {actual_im}), expected=({expected_re}, {expected_im}), diff={diff}, limit={limit}"
            ));
        }
    }
    Ok(())
}

fn required_tolerance<'a>(
    tolerance: Option<&'a ComparisonTolerance>,
    label: &str,
    case: &CaseRecord,
) -> Result<&'a ComparisonTolerance, String> {
    let tolerance =
        tolerance.ok_or_else(|| format!("{}: missing {label} tolerance", case.case_id))?;
    if tolerance.kind != "allclose" {
        return Err(format!(
            "{}: unsupported {label} comparison kind {}",
            case.case_id, tolerance.kind
        ));
    }
    Ok(tolerance)
}

fn required_input<'a>(
    inputs: &'a BTreeMap<String, TracedTensor>,
    name: &str,
    case: &CaseRecord,
) -> Result<&'a TracedTensor, String> {
    inputs
        .get(name)
        .ok_or_else(|| format!("{}: missing input {name}", case.case_id))
}

fn oracle_to_tenferro(tensor: &TracedTensor, core_rank: usize) -> Result<TracedTensor, String> {
    if tensor.rank <= core_rank {
        return Ok(tensor.clone());
    }
    let rank = tensor.rank;
    let split = rank - core_rank;
    let mut perm = Vec::with_capacity(rank);
    perm.extend(split..rank);
    perm.extend(0..split);
    tensor.transpose(&perm).map_err(to_string)
}

fn tenferro_to_oracle(tensor: &TracedTensor, core_rank: usize) -> Result<TracedTensor, String> {
    if tensor.rank <= core_rank {
        return Ok(tensor.clone());
    }
    let rank = tensor.rank;
    let mut perm = Vec::with_capacity(rank);
    perm.extend(core_rank..rank);
    perm.extend(0..core_rank);
    tensor.transpose(&perm).map_err(to_string)
}

fn rhs_core_rank(a: &TracedTensor, b: &TracedTensor) -> usize {
    if b.rank + 1 == a.rank {
        1
    } else {
        2
    }
}

fn promote_vector_rhs_for_solve(
    rhs: &TracedTensor,
    rhs_core_rank: usize,
) -> Result<TracedTensor, String> {
    if rhs_core_rank != 1 {
        return Ok(rhs.clone());
    }
    let mut shape = tensor_shape(rhs);
    shape.insert(1, 1);
    rhs.reshape(&shape).map_err(to_string)
}

fn demote_vector_solution_after_solve(
    solution: &TracedTensor,
    rhs_core_rank: usize,
) -> Result<TracedTensor, String> {
    if rhs_core_rank != 1 {
        return Ok(solution.clone());
    }
    let mut shape = tensor_shape(solution);
    if shape.get(1) != Some(&1) {
        return Err(format!(
            "solve vector RHS expected singleton solution axis 1, got shape {shape:?}"
        ));
    }
    shape.remove(1);
    solution.reshape(&shape).map_err(to_string)
}

fn lstsq_residuals(
    a: &TracedTensor,
    b: &TracedTensor,
    solution: &TracedTensor,
    rhs_core_rank: usize,
) -> Result<TracedTensor, String> {
    let shape = tensor_shape(a);
    let (m, n) = (shape[0], shape[1]);
    if m <= n {
        return zero_traced_tensor(solution.dtype, vec![0]);
    }

    let fitted = matmul_tenferro_matrix_axes(a, solution)?;
    let residual = fitted.sub(b).map_err(to_string)?;
    let squared = residual.mul(&residual).map_err(to_string)?;
    let residuals = squared.reduce_sum(Some(&[0])).map_err(to_string)?;
    demote_vector_lstsq_residuals(&residuals, rhs_core_rank)
}

fn demote_vector_lstsq_residuals(
    residuals: &TracedTensor,
    rhs_core_rank: usize,
) -> Result<TracedTensor, String> {
    if rhs_core_rank != 1 {
        return Ok(residuals.clone());
    }
    let mut shape = tensor_shape(residuals);
    if shape.first() != Some(&1) {
        return Err(format!(
            "lstsq vector RHS expected singleton residual axis 0, got shape {shape:?}"
        ));
    }
    shape.remove(0);
    residuals.reshape(&shape).map_err(to_string)
}

fn promote_vector_rhs_for_triangular_solve(
    rhs: &TracedTensor,
    rhs_core_rank: usize,
    left_side: bool,
) -> Result<TracedTensor, String> {
    if rhs_core_rank != 1 {
        return Ok(rhs.clone());
    }
    let mut shape = tensor_shape(rhs);
    if left_side {
        shape.insert(1, 1);
    } else {
        shape.insert(0, 1);
    }
    rhs.reshape(&shape).map_err(to_string)
}

fn demote_vector_solution_after_triangular_solve(
    solution: &TracedTensor,
    rhs_core_rank: usize,
    left_side: bool,
) -> Result<TracedTensor, String> {
    if rhs_core_rank != 1 {
        return Ok(solution.clone());
    }
    let mut shape = tensor_shape(solution);
    let axis = if left_side { 1 } else { 0 };
    if shape.get(axis) != Some(&1) {
        return Err(format!(
            "triangular solve vector RHS expected singleton solution axis {axis}, got shape {shape:?}"
        ));
    }
    shape.remove(axis);
    solution.reshape(&shape).map_err(to_string)
}

fn single_output(
    inputs: BTreeMap<String, TracedTensor>,
    name: &str,
    tensor: TracedTensor,
) -> Result<CaseExecution, String> {
    Ok(CaseExecution {
        inputs,
        outputs: vec![named(name, tensor)],
    })
}

fn named(name: &str, tensor: TracedTensor) -> NamedTensor {
    NamedTensor {
        name: name.to_string(),
        tensor,
    }
}

fn swap_tenferro_matrix_axes(tensor: &TracedTensor) -> Result<TracedTensor, String> {
    if tensor.rank < 2 {
        return Ok(tensor.clone());
    }
    let mut perm: Vec<usize> = (0..tensor.rank).collect();
    perm.swap(0, 1);
    tensor.transpose(&perm).map_err(to_string)
}

fn adjoint_tenferro_matrix_axes(tensor: &TracedTensor) -> Result<TracedTensor, String> {
    swap_tenferro_matrix_axes(&tensor.conj().map_err(to_string)?)
}

fn hermitian_wrapper_tenferro(tensor: &TracedTensor) -> Result<TracedTensor, String> {
    let tensor_h = adjoint_tenferro_matrix_axes(tensor)?;
    (tensor + &tensor_h).map_err(to_string)
}

fn required_usize_kwarg(case: &CaseRecord, key: &str) -> Result<usize, String> {
    let value = case
        .op_kwargs
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{}: missing or invalid usize kwarg {key}", case.case_id))?;
    usize::try_from(value).map_err(|_| format!("{}: kwarg {key} exceeds usize", case.case_id))
}

fn bool_kwarg(case: &CaseRecord, key: &str) -> Result<Option<bool>, String> {
    match case.op_kwargs.get(key) {
        Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Bool(flag)) => Ok(Some(*flag)),
        Some(other) => Err(format!(
            "{}: expected boolean kwarg {key}, got {other}",
            case.case_id
        )),
        None => Ok(None),
    }
}

fn optional_f64_kwarg(case: &CaseRecord, key: &str) -> Result<Option<f64>, String> {
    match case.op_kwargs.get(key) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => Ok(Some(value_as_f64(value).ok_or_else(|| {
            format!(
                "{}: expected numeric kwarg {key}, got {value}",
                case.case_id
            )
        })?)),
    }
}

type NormReplayArguments = (Option<f64>, Option<Vec<usize>>, bool);

fn norm_arguments(case: &CaseRecord, rank: usize) -> Result<NormReplayArguments, String> {
    match case.op.as_str() {
        "norm" => norm_arguments_for_norm(case, rank),
        "vector_norm" => norm_arguments_for_vector_norm(case, rank),
        "matrix_norm" => norm_arguments_for_matrix_norm(case, rank),
        other => Err(format!("{}: {other} is not a norm op", case.case_id)),
    }
}

fn norm_arguments_for_norm(case: &CaseRecord, rank: usize) -> Result<NormReplayArguments, String> {
    let keepdim = bool_kwarg(case, "keepdim")?.unwrap_or(false);
    let dim = case
        .op_kwargs
        .get("dim")
        .map(|value| axes_value(value, rank, &case.case_id))
        .transpose()?;
    let ord = match case.op_kwargs.get("ord") {
        Some(value) => norm_ord_value(value, &case.case_id)?,
        None => case
            .op_args
            .first()
            .map(|value| norm_ord_value(value, &case.case_id))
            .transpose()?
            .flatten(),
    };
    Ok((ord, dim, keepdim))
}

fn norm_arguments_for_vector_norm(
    case: &CaseRecord,
    rank: usize,
) -> Result<NormReplayArguments, String> {
    let keepdim = bool_kwarg(case, "keepdim")?.unwrap_or(false);
    let dim = case
        .op_kwargs
        .get("dim")
        .map(|value| axes_value(value, rank, &case.case_id))
        .transpose()?
        .unwrap_or_else(|| vec![0]);
    let ord = case
        .op_kwargs
        .get("ord")
        .map(|value| norm_ord_value(value, &case.case_id))
        .transpose()?
        .flatten();
    Ok((ord, Some(dim), keepdim))
}

fn norm_arguments_for_matrix_norm(
    case: &CaseRecord,
    rank: usize,
) -> Result<NormReplayArguments, String> {
    if case.op_args.len() != 3 {
        return Err(format!(
            "{}: matrix_norm replay expects [ord, dim, keepdim] op_args",
            case.case_id
        ));
    }
    let ord = norm_ord_value(&case.op_args[0], &case.case_id)?;
    let dim = axes_value(&case.op_args[1], rank, &case.case_id)?;
    let keepdim = case.op_args[2].as_bool().ok_or_else(|| {
        format!(
            "{}: matrix_norm keepdim must be boolean, got {}",
            case.case_id, case.op_args[2]
        )
    })?;
    Ok((ord, Some(dim), keepdim))
}

fn norm_ord_value(value: &Value, case_id: &str) -> Result<Option<f64>, String> {
    match value {
        Value::Null => Ok(None),
        Value::String(text) if text == "fro" => Ok(None),
        Value::String(text) if text == "nuc" => Err(format!(
            "{case_id}: nuclear norm is not supported by the current tenferro norm adapter"
        )),
        Value::String(_) | Value::Number(_) => value_as_f64(value)
            .map(Some)
            .ok_or_else(|| format!("{case_id}: invalid norm ord {value}")),
        other => Err(format!("{case_id}: invalid norm ord {other}")),
    }
}

fn axes_value(value: &Value, rank: usize, case_id: &str) -> Result<Vec<usize>, String> {
    match value {
        Value::Number(_) => Ok(vec![axis_value(value, rank, case_id)?]),
        Value::Array(values) => values
            .iter()
            .map(|value| axis_value(value, rank, case_id))
            .collect(),
        other => Err(format!("{case_id}: invalid axis list {other}")),
    }
}

fn axis_value(value: &Value, rank: usize, case_id: &str) -> Result<usize, String> {
    let axis = value
        .as_i64()
        .ok_or_else(|| format!("{case_id}: axis must be an integer, got {value}"))?;
    let rank_i64 = i64::try_from(rank).map_err(|_| format!("{case_id}: rank overflow"))?;
    let normalized = if axis < 0 { rank_i64 + axis } else { axis };
    if !(0..rank_i64).contains(&normalized) {
        return Err(format!(
            "{case_id}: axis {axis} is out of bounds for rank {rank}"
        ));
    }
    usize::try_from(normalized).map_err(|_| format!("{case_id}: axis overflow"))
}

fn value_as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) if text == "Infinity" => Some(f64::INFINITY),
        Value::String(text) if text == "-Infinity" => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

fn align_cotangent_dtype(
    output: &TracedTensor,
    cotangent: &TracedTensor,
) -> Result<TracedTensor, String> {
    if output.dtype == cotangent.dtype {
        return Ok(cotangent.clone());
    }
    Err(format!(
        "cotangent dtype {:?} does not match output dtype {:?}",
        cotangent.dtype, output.dtype
    ))
}

fn zero_traced_tensor(dtype: DType, shape: Vec<usize>) -> Result<TracedTensor, String> {
    let len = checked_product(&shape)?;
    let tensor = match dtype {
        DType::F32 => Tensor::from_vec_col_major(shape, vec![0.0_f32; len]).map_err(to_string)?,
        DType::F64 => Tensor::from_vec_col_major(shape, vec![0.0_f64; len]).map_err(to_string)?,
        DType::C32 => Tensor::from_vec_col_major(shape, vec![Complex32::new(0.0, 0.0); len])
            .map_err(to_string)?,
        DType::C64 => Tensor::from_vec_col_major(shape, vec![Complex64::new(0.0, 0.0); len])
            .map_err(to_string)?,
        other => return Err(format!("unsupported zero tensor dtype {other:?}")),
    };
    TracedTensor::from_tensor_concrete_shape(tensor).map_err(to_string)
}

fn tensor_shape(tensor: &TracedTensor) -> Vec<usize> {
    tensor
        .try_concrete_shape()
        .expect("oracle replay tensors should have concrete shape")
}

fn checked_product(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| format!("shape product overflow for {shape:?}"))
    })
}

fn matmul_tenferro_matrix_axes(
    lhs: &TracedTensor,
    rhs: &TracedTensor,
) -> Result<TracedTensor, String> {
    let rank = lhs.rank;
    let batch_dims: Vec<usize> = (2..rank).collect();
    lhs.dot_general(
        rhs,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: batch_dims.clone(),
            rhs_batch_dims: batch_dims,
        },
    )
    .map_err(to_string)
}

fn to_string(error: impl std::fmt::Display) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::parse_replay_jobs_env;

    #[test]
    fn replay_jobs_env_parser_accepts_missing_and_positive_values() {
        assert_eq!(parse_replay_jobs_env(None).unwrap(), None);
        assert_eq!(parse_replay_jobs_env(Some("1")).unwrap(), Some(1));
        assert_eq!(parse_replay_jobs_env(Some("48")).unwrap(), Some(48));
    }

    #[test]
    fn replay_jobs_env_parser_rejects_zero_and_invalid_values() {
        assert!(parse_replay_jobs_env(Some("0")).is_err());
        assert!(parse_replay_jobs_env(Some("many")).is_err());
    }
}
