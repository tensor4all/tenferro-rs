mod compare;
mod db;
mod decode;
mod dispatch;
mod observable;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::{env, num::ParseIntError};

use tenferro::engine::Engine;
use tenferro::traced::eval_all;
use tenferro::{CpuBackend, Tensor, TracedTensor, TypedTensor};

use crate::compare::compare_tensor;
use crate::db::{oracle_cases_dir, try_discover_case_files, try_load_cases};
use crate::decode::{try_decode_tensor, CaseRecord, Probe, TensorData, Tolerance};
use crate::dispatch::{dispatch_case, DispatchResult, NamedTensor};
use crate::observable::apply_observable;

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

#[test]
fn oracle_replay_all() {
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

    for (op_name, path) in &case_files {
        if op_filter.as_deref().is_some_and(|filter| op_name != filter) {
            continue;
        }
        let cases = match try_load_cases(path) {
            Ok(cases) => cases,
            Err(err) => {
                stats.failed.push(err);
                continue;
            }
        };

        for case in &cases {
            if op_filter.as_deref().is_some_and(|filter| case.op != filter) {
                continue;
            }
            if stats.cases >= case_limit {
                break;
            }
            stats.cases += 1;
            let mut engine = Engine::new(CpuBackend::new());
            match replay_case(case, &mut engine) {
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
        if stats.cases >= case_limit {
            break;
        }
    }

    print_summary(&stats);
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

fn replay_case(case: &CaseRecord, engine: &mut Engine<CpuBackend>) -> Result<CaseOutcome, String> {
    if case.expected_behavior == "error" {
        return Ok(CaseOutcome::ExpectedError);
    }
    if case.dtype != "float64" {
        return Ok(CaseOutcome::SkippedDType);
    }

    let execution = match dispatch_case(case)? {
        DispatchResult::Executed(execution) => execution,
        DispatchResult::SkippedUnimplemented(op) => {
            return Ok(CaseOutcome::SkippedUnimplemented(op));
        }
    };
    let outputs = apply_observable(&case.observable.kind, execution.outputs, engine)?;

    for probe in &case.probes {
        replay_probe(case, probe, &execution.inputs, &outputs, engine)?;
    }

    Ok(CaseOutcome::Passed)
}

fn replay_probe(
    case: &CaseRecord,
    probe: &Probe,
    inputs: &BTreeMap<String, TracedTensor>,
    outputs: &[NamedTensor],
    engine: &mut Engine<CpuBackend>,
) -> Result<(), String> {
    let first_order = required_tolerance(
        case.comparison.first_order.as_ref(),
        "first_order",
        &case.case_id,
    )?;
    let direction_tensors = decode_named_tensors(&probe.direction)?;
    let cotangent_tensors = decode_named_tensors(&probe.cotangent)?;

    let mut jvp_outputs =
        build_jvp_outputs(outputs, inputs, &direction_tensors, &probe.pytorch_ref.jvp)?;
    let jvp_results = eval_named_tensors(engine, &mut jvp_outputs)?;
    compare_named_results(
        &jvp_results,
        &jvp_outputs,
        &probe.pytorch_ref.jvp,
        first_order,
        &format!("probe {} JVP", probe.probe_id),
    )?;

    let scalar = cotangent_scalar(outputs, &cotangent_tensors, &probe.probe_id)?;
    let mut vjp_outputs = build_grad_outputs(&scalar, inputs)?;
    let vjp_results = eval_named_tensors(engine, &mut vjp_outputs)?;
    compare_named_results(
        &vjp_results,
        &vjp_outputs,
        &probe.pytorch_ref.vjp,
        first_order,
        &format!("probe {} VJP", probe.probe_id),
    )?;

    if !probe.pytorch_ref.hvp.is_empty() {
        let second_order = required_tolerance(
            case.comparison.second_order.as_ref(),
            "second_order",
            &case.case_id,
        )?;
        let mut hvp_outputs = build_hvp_outputs(&vjp_outputs, inputs, &direction_tensors)?;
        let hvp_results = eval_named_tensors(engine, &mut hvp_outputs)?;
        compare_named_results(
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
            .ok_or_else(|| format!("tensor {name} is not float64"))?;
        decoded.insert(name.clone(), TracedTensor::from_tensor(tensor));
    }
    Ok(decoded)
}

fn build_jvp_outputs(
    outputs: &[NamedTensor],
    inputs: &BTreeMap<String, TracedTensor>,
    directions: &BTreeMap<String, TracedTensor>,
    expected: &HashMap<String, TensorData>,
) -> Result<Vec<NamedTensor>, String> {
    let mut tangents = Vec::with_capacity(outputs.len());
    for output in outputs {
        if !expected.contains_key(&output.name) {
            continue;
        }
        let tangent = try_directional_jvp(&output.tensor, inputs, directions)?
            .unwrap_or_else(|| output.tensor.scale_real(0.0));
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
            None => TracedTensor::from_tensor(Tensor::F64(TypedTensor::zeros(input.shape.clone()))),
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
        hvps.push(NamedTensor {
            name: gradient.name.clone(),
            tensor: directional_jvp(&gradient.tensor, inputs, directions)?,
        });
    }
    Ok(hvps)
}

fn directional_jvp(
    output: &TracedTensor,
    inputs: &BTreeMap<String, TracedTensor>,
    directions: &BTreeMap<String, TracedTensor>,
) -> Result<TracedTensor, String> {
    Ok(match try_directional_jvp(output, inputs, directions)? {
        Some(tangent) => tangent,
        None => TracedTensor::from_tensor(Tensor::F64(TypedTensor::zeros(output.shape.clone()))),
    })
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
    outputs: &[NamedTensor],
    cotangents: &BTreeMap<String, TracedTensor>,
    probe_id: &str,
) -> Result<TracedTensor, String> {
    let mut scalar_terms = Vec::with_capacity(outputs.len());
    for output in outputs {
        let Some(cotangent) = cotangents.get(&output.name) else {
            continue;
        };
        let axes: Vec<usize> = (0..output.tensor.shape.len()).collect();
        scalar_terms.push((&output.tensor * cotangent).reduce_sum(&axes));
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
    engine: &mut Engine<CpuBackend>,
    outputs: &mut [NamedTensor],
) -> Result<Vec<Tensor>, String> {
    let mut traced: Vec<&mut TracedTensor> = outputs
        .iter_mut()
        .map(|output| &mut output.tensor)
        .collect();
    eval_all(engine, traced.as_mut_slice()).map_err(|err| err.to_string())
}

fn compare_named_results(
    actual: &[Tensor],
    outputs: &[NamedTensor],
    expected: &HashMap<String, TensorData>,
    tolerance: &Tolerance,
    context: &str,
) -> Result<(), String> {
    let actual_names: BTreeSet<&str> = outputs.iter().map(|output| output.name.as_str()).collect();
    let expected_names: BTreeSet<&str> = expected.keys().map(String::as_str).collect();
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

fn print_summary(stats: &ReplayStats) {
    eprintln!("oracle replay summary:");
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
