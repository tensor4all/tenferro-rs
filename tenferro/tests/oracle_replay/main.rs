#![cfg(feature = "autodiff")]

mod compare;
mod db;
mod decode;
mod dispatch;
mod observable;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::{env, num::ParseIntError};

use num_complex::{Complex32, Complex64};
use tenferro::{
    CpuBackend, DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor, TypedTensor,
};

use crate::compare::compare_tensor;
use crate::db::{oracle_cases_dir, try_discover_case_files, try_load_cases};
use crate::decode::{try_decode_tensor, CaseRecord, Probe, TensorData, Tolerance};
use crate::dispatch::{dispatch_case, DispatchResult, NamedTensor};
use crate::observable::apply_observable;

const MAX_ORACLE_REPLAY_SHARDS: usize = 16;
const MIN_CASES_PER_ORACLE_REPLAY_SHARD: usize = 2_048;

#[derive(Default)]
struct ReplayStats {
    files: usize,
    cases: usize,
    passed: usize,
    skipped_dtype: usize,
    skipped_unimplemented_cases: usize,
    skipped_unimplemented_ops: BTreeSet<String>,
    expected_error: usize,
    failed: Vec<String>,
}

enum CaseOutcome {
    Passed,
    SkippedDType,
    SkippedUnimplemented(String),
    ExpectedError,
}

#[derive(Clone, Copy)]
enum DerivativeKind {
    Jvp,
    Vjp,
    Hvp,
}

struct LoadedCaseFile {
    cases: Vec<CaseRecord>,
}

struct ManifestEntry {
    file_index: usize,
    case_index: usize,
}

struct ReplayManifest {
    loaded_files: Vec<LoadedCaseFile>,
    entries: Vec<ManifestEntry>,
    load_failures: Vec<String>,
}

#[test]
#[ignore = "replaced by sharded tests for nextest parallelism"]
fn oracle_replay_all() {
    run_oracle_replay(0, Some(1));
}

macro_rules! oracle_replay_shard_tests {
    ($(($name:ident, $index:expr)),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                run_oracle_replay($index, None);
            }
        )*
    };
}

oracle_replay_shard_tests!(
    (oracle_replay_shard_0, 0),
    (oracle_replay_shard_1, 1),
    (oracle_replay_shard_2, 2),
    (oracle_replay_shard_3, 3),
    (oracle_replay_shard_4, 4),
    (oracle_replay_shard_5, 5),
    (oracle_replay_shard_6, 6),
    (oracle_replay_shard_7, 7),
    (oracle_replay_shard_8, 8),
    (oracle_replay_shard_9, 9),
    (oracle_replay_shard_10, 10),
    (oracle_replay_shard_11, 11),
    (oracle_replay_shard_12, 12),
    (oracle_replay_shard_13, 13),
    (oracle_replay_shard_14, 14),
    (oracle_replay_shard_15, 15),
);

/// Oracle replay shards are skipped locally unless explicitly opted in, to
/// keep `cargo nextest run --workspace` responsive on dev machines. CI
/// (`CI=true`, set automatically by GitHub Actions on every runner) runs the
/// full shard set. For local runs, set `RUN_ORACLE_REPLAY=1`.
fn oracle_replay_enabled() -> bool {
    env::var("CI").ok().as_deref() == Some("true") || env::var("RUN_ORACLE_REPLAY").is_ok()
}

fn run_oracle_replay(registered_shard_index: usize, forced_shard_count: Option<usize>) {
    if !oracle_replay_enabled() {
        eprintln!(
            "oracle_replay_shard_{registered_shard_index}: skipped locally; \
             set CI=true or RUN_ORACLE_REPLAY=1 to run"
        );
        return;
    }

    assert!(
        registered_shard_index < MAX_ORACLE_REPLAY_SHARDS,
        "invalid registered shard index {registered_shard_index}"
    );

    let root = oracle_cases_dir();
    let case_files = match try_discover_case_files(&root) {
        Ok(files) => files,
        Err(err) => panic!("{err}"),
    };

    let mut stats = ReplayStats {
        files: case_files.len(),
        ..ReplayStats::default()
    };
    let op_filter = env::var("ORACLE_REPLAY_OP").ok();
    let case_limit = match parse_case_limit(env::var("ORACLE_REPLAY_CASE_LIMIT").ok()) {
        Ok(limit) => limit,
        Err(err) => panic!("{err}"),
    };

    let entry_count_estimate = count_replay_entries(&case_files, op_filter.as_deref(), case_limit);

    let active_shard_count = match forced_shard_count {
        Some(count) => count.min(entry_count_estimate.max(1)),
        None => match resolve_active_shard_count(entry_count_estimate) {
            Ok(count) => count,
            Err(err) => panic!("{err}"),
        },
    };
    if registered_shard_index >= active_shard_count {
        return;
    }

    let manifest = build_replay_manifest(&case_files, op_filter.as_deref(), case_limit);
    stats.failed.extend(manifest.load_failures);

    let backend_threads = oracle_replay_backend_threads(active_shard_count);

    for (entry_index, manifest_entry) in manifest.entries.iter().enumerate() {
        if !entry_in_active_shard(entry_index, registered_shard_index, active_shard_count) {
            continue;
        }
        let case =
            &manifest.loaded_files[manifest_entry.file_index].cases[manifest_entry.case_index];
        stats.cases += 1;
        let mut compiler = GraphCompiler::new();
        let mut executor = GraphExecutor::new(CpuBackend::with_threads(backend_threads));
        match replay_case(case, &mut compiler, &mut executor) {
            Ok(CaseOutcome::Passed) => stats.passed += 1,
            Ok(CaseOutcome::SkippedDType) => stats.skipped_dtype += 1,
            Ok(CaseOutcome::ExpectedError) => stats.expected_error += 1,
            Ok(CaseOutcome::SkippedUnimplemented(op)) => {
                stats.skipped_unimplemented_cases += 1;
                stats.skipped_unimplemented_ops.insert(op);
            }
            Err(err) => stats.failed.push(format!("{}: {err}", case.case_id)),
        }
    }

    print_summary(
        &stats,
        registered_shard_index,
        active_shard_count,
        backend_threads,
    );
    assert!(
        stats.failed.is_empty(),
        "oracle replay had {} failures",
        stats.failed.len()
    );
}

fn parse_case_limit(value: Option<String>) -> Result<usize, String> {
    match value {
        Some(value) => value.parse::<usize>().map_err(|err: ParseIntError| {
            format!("invalid ORACLE_REPLAY_CASE_LIMIT {value}: {err}")
        }),
        None => Ok(usize::MAX),
    }
}

#[test]
fn adaptive_shard_count_override_validates_bounds() {
    assert_eq!(
        parse_shard_count_override(Some("4".to_string()), 16).expect("valid override"),
        Some(4)
    );
    assert!(parse_shard_count_override(Some("0".to_string()), 16)
        .expect_err("zero override must fail")
        .contains("must be between 1 and 16"));
    assert!(parse_shard_count_override(Some("17".to_string()), 16)
        .expect_err("too-large override must fail")
        .contains("must be between 1 and 16"));
}

#[test]
fn adaptive_auto_shard_count_clamps_to_limits() {
    assert_eq!(compute_auto_shard_count(8, 9_572, 16, 2_048), 4);
    assert_eq!(compute_auto_shard_count(64, 9_572, 16, 2_048), 4);
    assert_eq!(compute_auto_shard_count(8, 600, 16, 2_048), 1);
    assert_eq!(compute_auto_shard_count(8, 0, 16, 2_048), 1);
}

#[test]
fn adaptive_partition_covers_each_entry_once() {
    let weights = vec![3usize, 3, 3, 3, 3, 3, 3];
    let ranges = partition_weighted_indices(&weights, 3);
    let mut owners = vec![0usize; weights.len()];

    for (shard_index, range) in ranges.iter().enumerate() {
        for entry_index in range.clone() {
            owners[entry_index] += 1;
            assert!(entry_index < weights.len(), "entry index out of bounds");
        }
        assert!(
            range.start <= range.end,
            "invalid range for shard {shard_index}"
        );
    }

    assert!(owners.into_iter().all(|count| count == 1));
}

#[test]
fn adaptive_partition_balances_weighted_fixture() {
    let weights = vec![8usize, 1, 8, 1, 8, 1];
    let ranges = partition_weighted_indices(&weights, 3);
    let shard_weights: Vec<usize> = ranges
        .iter()
        .map(|range| weights[range.clone()].iter().sum())
        .collect();
    assert_eq!(shard_weights, vec![9, 9, 9]);
}

#[test]
fn adaptive_entry_assignment_covers_each_index_once() {
    let shard_count = 5usize;
    for entry_index in 0..97usize {
        let owners: Vec<usize> = (0..shard_count)
            .filter(|&shard_index| entry_in_active_shard(entry_index, shard_index, shard_count))
            .collect();
        assert_eq!(
            owners,
            vec![entry_index % shard_count],
            "entry {entry_index}"
        );
    }
}

#[test]
fn adaptive_entry_assignment_balances_prefix_within_one() {
    let shard_count = 5usize;
    let total_entries = 97usize;
    let counts: Vec<usize> = (0..shard_count)
        .map(|shard_index| {
            (0..total_entries)
                .filter(|&entry_index| entry_in_active_shard(entry_index, shard_index, shard_count))
                .count()
        })
        .collect();
    let min = *counts.iter().min().expect("non-empty shard counts");
    let max = *counts.iter().max().expect("non-empty shard counts");
    assert!(max - min <= 1, "counts: {counts:?}");
}

fn entry_in_active_shard(entry_index: usize, shard_index: usize, shard_count: usize) -> bool {
    assert!(shard_count > 0, "shard_count must be > 0");
    assert!(
        shard_index < shard_count,
        "invalid shard {shard_index}/{shard_count}"
    );
    entry_index % shard_count == shard_index
}

fn parse_shard_count_override(
    value: Option<String>,
    max_supported: usize,
) -> Result<Option<usize>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let parsed = value.parse::<usize>().map_err(|err: ParseIntError| {
        format!("invalid ORACLE_REPLAY_SHARD_COUNT {value}: {err}")
    })?;
    if !(1..=max_supported).contains(&parsed) {
        return Err(format!(
            "ORACLE_REPLAY_SHARD_COUNT must be between 1 and {max_supported}, got {parsed}"
        ));
    }
    Ok(Some(parsed))
}

fn compute_auto_shard_count(
    available_parallelism: usize,
    total_cases: usize,
    max_supported: usize,
    min_cases_per_shard: usize,
) -> usize {
    let available_parallelism = available_parallelism.max(1);
    let total_cases_cap = total_cases.max(1);
    let max_supported = max_supported.max(1);
    let min_cases_per_shard = min_cases_per_shard.max(1);
    let by_case_budget = std::cmp::max(1, total_cases / min_cases_per_shard);

    available_parallelism
        .min(total_cases_cap)
        .min(max_supported)
        .min(by_case_budget)
}

fn resolve_active_shard_count(total_cases: usize) -> Result<usize, String> {
    let override_count = parse_shard_count_override(
        env::var("ORACLE_REPLAY_SHARD_COUNT").ok(),
        MAX_ORACLE_REPLAY_SHARDS,
    )?;
    if let Some(count) = override_count {
        return Ok(count.min(total_cases.max(1)));
    }

    let available_parallelism = std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1);
    Ok(compute_auto_shard_count(
        available_parallelism,
        total_cases,
        MAX_ORACLE_REPLAY_SHARDS,
        MIN_CASES_PER_ORACLE_REPLAY_SHARD,
    ))
}

fn partition_weighted_indices(weights: &[usize], shard_count: usize) -> Vec<Range<usize>> {
    assert!(shard_count > 0, "shard_count must be > 0");
    if weights.is_empty() {
        return (0..shard_count).map(|_| 0..0).collect();
    }

    let total_weight: usize = weights.iter().sum();
    let mut ranges = Vec::with_capacity(shard_count);
    let mut start = 0usize;
    let mut cumulative_weight = 0usize;

    for shard_index in 0..shard_count {
        if shard_index == shard_count - 1 {
            ranges.push(start..weights.len());
            break;
        }
        if start >= weights.len() {
            ranges.push(weights.len()..weights.len());
            continue;
        }

        let remaining_shards = shard_count - shard_index;
        let remaining_entries = weights.len() - start;
        if remaining_entries <= remaining_shards {
            cumulative_weight += weights[start];
            ranges.push(start..start + 1);
            start += 1;
            continue;
        }

        let target_weight = (total_weight * (shard_index + 1)).div_ceil(shard_count);
        let max_end = weights.len() - (remaining_shards - 1);
        let mut end = start;
        while end < max_end && cumulative_weight < target_weight {
            cumulative_weight += weights[end];
            end += 1;
        }
        if end == start {
            cumulative_weight += weights[end];
            end += 1;
        }
        ranges.push(start..end);
        start = end;
    }

    while ranges.len() < shard_count {
        ranges.push(weights.len()..weights.len());
    }
    ranges
}

fn count_replay_entries(
    case_files: &[(String, std::path::PathBuf)],
    op_filter: Option<&str>,
    case_limit: usize,
) -> usize {
    let mut total = 0usize;
    for (op_name, path) in case_files {
        if op_filter.is_some_and(|filter| op_name != filter) {
            continue;
        }
        let Ok(contents) = std::fs::read_to_string(path) else {
            continue;
        };
        for line in contents.lines() {
            if line.trim().is_empty() {
                continue;
            }
            total += 1;
            if total >= case_limit {
                return total;
            }
        }
    }
    total
}

fn build_replay_manifest(
    case_files: &[(String, std::path::PathBuf)],
    op_filter: Option<&str>,
    case_limit: usize,
) -> ReplayManifest {
    let mut loaded_files = Vec::new();
    let mut entries = Vec::new();
    let mut load_failures = Vec::new();

    for (op_name, path) in case_files {
        if op_filter.is_some_and(|filter| op_name != filter) {
            continue;
        }
        let cases = match try_load_cases(path) {
            Ok(cases) => cases,
            Err(err) => {
                load_failures.push(err);
                continue;
            }
        };

        let file_index = loaded_files.len();
        let case_count = cases.len();
        loaded_files.push(LoadedCaseFile { cases });

        for case_index in 0..case_count {
            if entries.len() >= case_limit {
                break;
            }
            let case = &loaded_files[file_index].cases[case_index];
            if op_filter.is_some_and(|filter| case.op != filter) {
                continue;
            }
            entries.push(ManifestEntry {
                file_index,
                case_index,
            });
        }
        if entries.len() >= case_limit {
            break;
        }
    }

    ReplayManifest {
        loaded_files,
        entries,
        load_failures,
    }
}

fn oracle_replay_backend_threads(shard_count: usize) -> usize {
    let available = std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1);
    if shard_count <= 1 {
        return available;
    }
    std::cmp::max(1, available / shard_count)
}

fn replay_case(
    case: &CaseRecord,
    compiler: &mut GraphCompiler,
    executor: &mut GraphExecutor<CpuBackend>,
) -> Result<CaseOutcome, String> {
    if case.expected_behavior == "error" {
        return Ok(CaseOutcome::ExpectedError);
    }
    if case.dtype != "float64" {
        return Ok(CaseOutcome::SkippedDType);
    }
    if case.op == "eig" && case.observable.kind == "eig_values_vectors_abs" {
        return Ok(CaseOutcome::SkippedUnimplemented(
            "eig vector derivatives".to_string(),
        ));
    }

    let execution = match dispatch_case(case)? {
        DispatchResult::Executed(execution) => execution,
        DispatchResult::SkippedUnimplemented(op) => {
            return Ok(CaseOutcome::SkippedUnimplemented(op));
        }
    };
    let outputs = apply_observable(&case.observable.kind, execution.outputs, compiler)?;

    for probe in &case.probes {
        replay_probe(case, probe, &execution.inputs, &outputs, executor)?;
    }

    Ok(CaseOutcome::Passed)
}

fn replay_probe(
    case: &CaseRecord,
    probe: &Probe,
    inputs: &BTreeMap<String, TracedTensor>,
    outputs: &[NamedTensor],
    executor: &mut GraphExecutor<CpuBackend>,
) -> Result<(), String> {
    let first_order = required_tolerance(
        case.comparison.first_order.as_ref(),
        "first_order",
        &case.case_id,
    )?;
    let direction_tensors = decode_named_tensors(&probe.direction)?;
    let cotangent_tensors = decode_named_tensors(&probe.cotangent)?;

    let mut jvp_outputs = build_jvp_outputs(
        case,
        outputs,
        inputs,
        &direction_tensors,
        &probe.pytorch_ref.jvp,
    )?;
    let jvp_results = eval_named_tensors(executor, &mut jvp_outputs)?;
    compare_named_results(
        case,
        DerivativeKind::Jvp,
        &jvp_results,
        &jvp_outputs,
        &probe.pytorch_ref.jvp,
        first_order,
        &format!("probe {} JVP", probe.probe_id),
    )?;

    let scalar = cotangent_scalar(case, outputs, &cotangent_tensors, &probe.probe_id)?;
    let mut vjp_outputs = build_grad_outputs(&scalar, inputs)?;
    let vjp_results = eval_named_tensors(executor, &mut vjp_outputs)?;
    compare_named_results(
        case,
        DerivativeKind::Vjp,
        &vjp_results,
        &vjp_outputs,
        &probe.pytorch_ref.vjp,
        first_order,
        &format!("probe {} VJP", probe.probe_id),
    )?;

    if !probe.pytorch_ref.hvp.is_empty() && supports_derivative(case, DerivativeKind::Hvp) {
        let second_order = required_tolerance(
            case.comparison.second_order.as_ref(),
            "second_order",
            &case.case_id,
        )?;
        let mut hvp_outputs = build_hvp_outputs(&vjp_outputs, inputs, &direction_tensors)?;
        let hvp_results = eval_named_tensors(executor, &mut hvp_outputs)?;
        compare_named_results(
            case,
            DerivativeKind::Hvp,
            &hvp_results,
            &hvp_outputs,
            &probe.pytorch_ref.hvp,
            second_order,
            &format!("probe {} HVP", probe.probe_id),
        )?;
    }

    Ok(())
}

fn required_tolerance<'a>(
    tolerance: Option<&'a Tolerance>,
    label: &str,
    case_id: &str,
) -> Result<&'a Tolerance, String> {
    let tolerance =
        tolerance.ok_or_else(|| format!("{case_id}: missing {label} comparison tolerance"))?;
    if tolerance.kind != "allclose" {
        return Err(format!(
            "{case_id}: unsupported {label} comparison kind {}",
            tolerance.kind
        ));
    }
    Ok(tolerance)
}

fn decode_named_tensors(
    tensors: &HashMap<String, TensorData>,
) -> Result<BTreeMap<String, TracedTensor>, String> {
    let mut decoded = BTreeMap::new();
    for (name, tensor_data) in tensors {
        let tensor = try_decode_tensor(tensor_data)?
            .ok_or_else(|| format!("tensor {name} has unsupported dtype {}", tensor_data.dtype))?;
        decoded.insert(
            name.clone(),
            TracedTensor::from_tensor_concrete_shape(tensor),
        );
    }
    Ok(decoded)
}

fn build_jvp_outputs(
    case: &CaseRecord,
    outputs: &[NamedTensor],
    inputs: &BTreeMap<String, TracedTensor>,
    directions: &BTreeMap<String, TracedTensor>,
    expected: &HashMap<String, TensorData>,
) -> Result<Vec<NamedTensor>, String> {
    let mut tangents = Vec::with_capacity(outputs.len());
    for output in outputs {
        if !expected.contains_key(&output.name)
            || !supports_output(case, DerivativeKind::Jvp, &output.name)
        {
            continue;
        }
        let tangent =
            try_directional_jvp(&output.tensor, inputs, directions)?.unwrap_or_else(|| {
                zero_traced_tensor(
                    output.tensor.dtype,
                    expected
                        .get(&output.name)
                        .expect("expected JVP reference tensor")
                        .shape
                        .clone(),
                )
            });
        tangents.push(NamedTensor {
            name: output.name.clone(),
            tensor: tangent,
        });
    }
    Ok(tangents)
}

fn build_grad_outputs(
    scalar: &TracedTensor,
    inputs: &BTreeMap<String, TracedTensor>,
) -> Result<Vec<NamedTensor>, String> {
    let mut gradients = Vec::with_capacity(inputs.len());
    for (name, input) in inputs {
        let gradient = match scalar
            .try_grad(input)
            .map_err(|err| format!("failed to build VJP for input {name}: {err}"))?
        {
            Some(gradient) => gradient,
            None => zero_traced_tensor(
                input.dtype,
                input
                    .data
                    .as_ref()
                    .expect("expected concrete input shape")
                    .shape()
                    .to_vec(),
            ),
        };
        gradients.push(NamedTensor {
            name: name.clone(),
            tensor: gradient,
        });
    }
    Ok(gradients)
}

fn build_hvp_outputs(
    gradients: &[NamedTensor],
    inputs: &BTreeMap<String, TracedTensor>,
    directions: &BTreeMap<String, TracedTensor>,
) -> Result<Vec<NamedTensor>, String> {
    let mut hvps = Vec::with_capacity(gradients.len());
    for gradient in gradients {
        let fallback_shape = inputs
            .get(&gradient.name)
            .and_then(|input| input.data.as_ref().map(|data| data.shape().to_vec()))
            .expect("expected concrete input shape");
        hvps.push(NamedTensor {
            name: gradient.name.clone(),
            tensor: match try_directional_jvp(&gradient.tensor, inputs, directions)? {
                Some(tangent) => tangent,
                None => zero_traced_tensor(gradient.tensor.dtype, fallback_shape),
            },
        });
    }
    Ok(hvps)
}

fn try_directional_jvp(
    output: &TracedTensor,
    inputs: &BTreeMap<String, TracedTensor>,
    directions: &BTreeMap<String, TracedTensor>,
) -> Result<Option<TracedTensor>, String> {
    let mut tangent_sum: Option<TracedTensor> = None;
    for (name, input) in inputs {
        let Some(direction) = directions.get(name) else {
            continue;
        };
        let Some(tangent) = output.try_jvp(input, direction) else {
            continue;
        };
        tangent_sum = Some(match tangent_sum {
            Some(current) => &current + &tangent,
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
    let mut scalar_terms = Vec::with_capacity(outputs.len());
    for output in outputs {
        if !supports_output(case, DerivativeKind::Vjp, &output.name) {
            continue;
        }
        let Some(cotangent) = cotangents.get(&output.name) else {
            continue;
        };
        let aligned_cotangent = align_cotangent_dtype(&output.tensor, cotangent)?;
        let aligned_cotangent = match output.tensor.dtype {
            DType::C32 | DType::C64 => aligned_cotangent.conj(),
            DType::F32 | DType::F64 => aligned_cotangent,
            DType::I32 | DType::I64 | DType::Bool => continue,
        };
        let axes: Vec<usize> = (0..output.tensor.rank).collect();
        scalar_terms.push((&output.tensor * &aligned_cotangent).reduce_sum(&axes));
    }

    let mut iter = scalar_terms.into_iter();
    let Some(mut scalar) = iter.next() else {
        return Err(format!(
            "probe {probe_id}: no outputs to build cotangent scalar"
        ));
    };
    for term in iter {
        scalar = &scalar + &term;
    }
    Ok(scalar)
}

fn eval_named_tensors(
    executor: &mut GraphExecutor<CpuBackend>,
    outputs: &mut [NamedTensor],
) -> Result<Vec<Tensor>, String> {
    let traced: Vec<&TracedTensor> = outputs.iter().map(|output| &output.tensor).collect();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_many(&traced)
        .map_err(|err| err.to_string())?;
    executor.run_many(&program).map_err(|err| err.to_string())
}

fn compare_named_results(
    case: &CaseRecord,
    derivative: DerivativeKind,
    actual: &[Tensor],
    outputs: &[NamedTensor],
    expected: &HashMap<String, TensorData>,
    tolerance: &Tolerance,
    context: &str,
) -> Result<(), String> {
    let actual_names: BTreeSet<&str> = outputs.iter().map(|output| output.name.as_str()).collect();
    let expected_names: BTreeSet<&str> = expected
        .keys()
        .filter(|name| supports_output(case, derivative, name))
        .map(String::as_str)
        .collect();
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

fn supports_derivative(case: &CaseRecord, derivative: DerivativeKind) -> bool {
    !matches!(
        (case.op.as_str(), derivative),
        ("eig", DerivativeKind::Hvp) | ("eigvals", DerivativeKind::Hvp)
    )
}

fn supports_output(case: &CaseRecord, derivative: DerivativeKind, name: &str) -> bool {
    !matches!(
        (case.op.as_str(), derivative, name),
        ("eig", DerivativeKind::Jvp, "vectors") | ("eig", DerivativeKind::Vjp, "vectors")
    )
}

fn print_summary(
    stats: &ReplayStats,
    shard_index: usize,
    shard_count: usize,
    backend_threads: usize,
) {
    eprintln!("oracle replay summary:");
    eprintln!("  shard: {}/{}", shard_index + 1, shard_count);
    eprintln!("  backend threads: {}", backend_threads);
    eprintln!("  selector: entry % {} == {}", shard_count, shard_index);
    eprintln!("  files: {}", stats.files);
    eprintln!("  cases: {}", stats.cases);
    eprintln!("  passed: {}", stats.passed);
    eprintln!("  skipped dtype: {}", stats.skipped_dtype);
    eprintln!(
        "  skipped unimplemented cases: {}",
        stats.skipped_unimplemented_cases
    );
    eprintln!(
        "  skipped unimplemented ops: {}",
        stats.skipped_unimplemented_ops.len()
    );
    if !stats.skipped_unimplemented_ops.is_empty() {
        eprintln!(
            "  unimplemented op names: {}",
            stats
                .skipped_unimplemented_ops
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    eprintln!("  expected error: {}", stats.expected_error);
    eprintln!("  failed: {}", stats.failed.len());
    for failure in &stats.failed {
        eprintln!("    {failure}");
    }
}

fn zero_traced_tensor(dtype: DType, shape: Vec<usize>) -> TracedTensor {
    let tensor = match dtype {
        DType::F32 => Tensor::F32(TypedTensor::<f32>::zeros(shape)),
        DType::F64 => Tensor::F64(TypedTensor::<f64>::zeros(shape)),
        DType::I32 => Tensor::I32(TypedTensor::<i32>::zeros(shape)),
        DType::I64 => Tensor::I64(TypedTensor::<i64>::zeros(shape)),
        DType::Bool => {
            let n_elements = shape.iter().product();
            Tensor::Bool(TypedTensor::from_vec_col_major(
                shape,
                vec![false; n_elements],
            ))
        }
        DType::C32 => Tensor::C32(TypedTensor::<Complex32>::zeros(shape)),
        DType::C64 => Tensor::C64(TypedTensor::<Complex64>::zeros(shape)),
    };
    TracedTensor::from_tensor_concrete_shape(tensor)
}

fn align_cotangent_dtype(
    output: &TracedTensor,
    cotangent: &TracedTensor,
) -> Result<TracedTensor, String> {
    Ok(cotangent.convert(output.dtype))
}
