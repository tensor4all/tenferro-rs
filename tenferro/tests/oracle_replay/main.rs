mod compare;
mod db;
mod decode;
mod dispatch;
mod observable;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::{env, num::ParseIntError};

use num_complex::{Complex32, Complex64};
use tenferro::engine::Engine;
use tenferro::traced::eval_all;
use tenferro::{CpuBackend, DType, Tensor, TracedTensor, TypedTensor};

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

#[derive(Clone, Copy)]
enum DerivativeKind {
    Jvp,
    Vjp,
    Hvp,
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

#[test]
fn oracle_replay_norm_case_048() {
    let root = oracle_cases_dir();
    let path = root.join("norm").join("identity.jsonl");
    let cases = try_load_cases(&path).expect("load norm oracle cases");
    let case = cases
        .iter()
        .find(|case| case.case_id == "norm_f64_identity_048")
        .expect("find norm_f64_identity_048");

    let execution = match dispatch_case(case).expect("dispatch case") {
        DispatchResult::Executed(execution) => execution,
        DispatchResult::SkippedUnimplemented(op) => panic!("unexpected skip for {op}"),
    };
    let mut engine = Engine::new(CpuBackend::new());
    let outputs = apply_observable(&case.observable.kind, execution.outputs, &mut engine)
        .expect("apply observable");
    let probe = &case.probes[0];
    let cotangent_tensors = decode_named_tensors(&probe.cotangent).expect("decode cotangent");
    let scalar = cotangent_scalar(case, &outputs, &cotangent_tensors, &probe.probe_id)
        .expect("build cotangent scalar");
    let input = execution.inputs.get("a").expect("input a");
    let maybe_grad = scalar.try_grad(input).expect("try_grad");
    assert!(maybe_grad.is_some(), "try_grad returned None");
    let manual_input = TracedTensor::from_tensor(Tensor::F64(TypedTensor::from_vec(
        vec![5, 5],
        vec![
            -4.826984902407649,
            -4.146041530864057,
            5.576059452216908,
            -3.063683029231029,
            -3.8432258180800494,
            -7.582430495695129,
            -5.4215280659972755,
            -8.315684908389088,
            -3.342174545322517,
            -3.0355148286483775,
            -0.6046891126565539,
            -4.784169829877467,
            4.177597026003685,
            -5.439777184204883,
            -2.146076312776824,
            -1.5411692216498662,
            -4.150805063878843,
            2.047386382824099,
            4.480518929058965,
            -2.6718482427688133,
            -1.9719097652168658,
            7.380839390984031,
            -4.076012721760325,
            2.685009210157367,
            -7.7232222137058715,
        ],
    )));
    let manual_cotangent =
        TracedTensor::from_tensor(Tensor::F64(TypedTensor::from_vec(vec![], vec![1.0])));
    let manual_output = tenferro::norm(
        &manual_input.clone(),
        Some(f64::NEG_INFINITY),
        Some(&[0, 1]),
        false,
    );
    let manual_axes: Vec<usize> = (0..manual_output.rank).collect();
    let manual_scalar = (&manual_output * &manual_cotangent).reduce_sum(&manual_axes);
    let manual_grad = manual_scalar.grad(&manual_input).expect("manual grad");
    let manual_actual = eval_named_tensors(
        &mut Engine::new(CpuBackend::new()),
        &mut [NamedTensor {
            name: "a".to_string(),
            tensor: manual_grad,
        }],
    )
    .expect("eval manual grad");
    compare_tensor(
        &manual_actual[0],
        probe.pytorch_ref.vjp.get("a").expect("expected manual vjp"),
        case.comparison
            .first_order
            .as_ref()
            .expect("first order tolerance")
            .rtol,
        case.comparison
            .first_order
            .as_ref()
            .expect("first order tolerance")
            .atol,
    )
    .expect("compare manual grad");
    let direct_grad = scalar.grad(input).expect("direct grad");
    let direct_actual = eval_named_tensors(
        &mut Engine::new(CpuBackend::new()),
        &mut [NamedTensor {
            name: "a".to_string(),
            tensor: direct_grad,
        }],
    )
    .expect("eval direct grad");
    compare_tensor(
        &direct_actual[0],
        probe.pytorch_ref.vjp.get("a").expect("expected direct vjp"),
        case.comparison
            .first_order
            .as_ref()
            .expect("first order tolerance")
            .rtol,
        case.comparison
            .first_order
            .as_ref()
            .expect("first order tolerance")
            .atol,
    )
    .expect("compare direct grad");

    let mut gradients = build_grad_outputs(&scalar, &execution.inputs).expect("build gradients");
    let actual = eval_named_tensors(&mut engine, &mut gradients).expect("eval gradients");
    compare_named_results(
        case,
        DerivativeKind::Vjp,
        &actual,
        &gradients,
        &probe.pytorch_ref.vjp,
        case.comparison
            .first_order
            .as_ref()
            .expect("first order tolerance"),
        "probe p0 VJP",
    )
    .expect("compare gradients");
}

#[test]
fn oracle_manual_norm_case_048() {
    let root = oracle_cases_dir();
    let path = root.join("norm").join("identity.jsonl");
    let cases = try_load_cases(&path).expect("load norm oracle cases");
    let case = cases
        .iter()
        .find(|case| case.case_id == "norm_f64_identity_048")
        .expect("find norm_f64_identity_048");
    let probe = &case.probes[0];
    let tolerance = case
        .comparison
        .first_order
        .as_ref()
        .expect("first order tolerance");

    let manual_input = TracedTensor::from_tensor(Tensor::F64(TypedTensor::from_vec(
        vec![5, 5],
        vec![
            -4.826984902407649,
            -4.146041530864057,
            5.576059452216908,
            -3.063683029231029,
            -3.8432258180800494,
            -7.582430495695129,
            -5.4215280659972755,
            -8.315684908389088,
            -3.342174545322517,
            -3.0355148286483775,
            -0.6046891126565539,
            -4.784169829877467,
            4.177597026003685,
            -5.439777184204883,
            -2.146076312776824,
            -1.5411692216498662,
            -4.150805063878843,
            2.047386382824099,
            4.480518929058965,
            -2.6718482427688133,
            -1.9719097652168658,
            7.380839390984031,
            -4.076012721760325,
            2.685009210157367,
            -7.7232222137058715,
        ],
    )));
    let manual_cotangent =
        TracedTensor::from_tensor(Tensor::F64(TypedTensor::from_vec(vec![], vec![1.0])));
    let manual_output = tenferro::norm(
        &manual_input.clone(),
        Some(f64::NEG_INFINITY),
        Some(&[0, 1]),
        false,
    );
    let manual_axes: Vec<usize> = (0..manual_output.rank).collect();
    let manual_scalar = (&manual_output * &manual_cotangent).reduce_sum(&manual_axes);
    let manual_grad = manual_scalar.grad(&manual_input).expect("manual grad");
    let actual = eval_named_tensors(
        &mut Engine::new(CpuBackend::new()),
        &mut [NamedTensor {
            name: "a".to_string(),
            tensor: manual_grad,
        }],
    )
    .expect("eval manual grad");
    compare_tensor(
        &actual[0],
        probe.pytorch_ref.vjp.get("a").expect("expected manual vjp"),
        tolerance.rtol,
        tolerance.atol,
    )
    .expect("compare manual grad");
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

    let mut jvp_outputs = build_jvp_outputs(
        case,
        outputs,
        inputs,
        &direction_tensors,
        &probe.pytorch_ref.jvp,
    )?;
    let jvp_results = eval_named_tensors(engine, &mut jvp_outputs)?;
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
    let vjp_results = eval_named_tensors(engine, &mut vjp_outputs)?;
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
        let hvp_results = eval_named_tensors(engine, &mut hvp_outputs)?;
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
        decoded.insert(name.clone(), TracedTensor::from_tensor(tensor));
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

fn zero_traced_tensor(dtype: DType, shape: Vec<usize>) -> TracedTensor {
    let tensor = match dtype {
        DType::F32 => Tensor::F32(TypedTensor::<f32>::zeros(shape)),
        DType::F64 => Tensor::F64(TypedTensor::<f64>::zeros(shape)),
        DType::C32 => Tensor::C32(TypedTensor::<Complex32>::zeros(shape)),
        DType::C64 => Tensor::C64(TypedTensor::<Complex64>::zeros(shape)),
    };
    TracedTensor::from_tensor(tensor)
}

fn align_cotangent_dtype(
    output: &TracedTensor,
    cotangent: &TracedTensor,
) -> Result<TracedTensor, String> {
    Ok(cotangent.convert(output.dtype))
}
