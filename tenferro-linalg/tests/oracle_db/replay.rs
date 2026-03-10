use std::collections::BTreeMap;

use tenferro_linalg::{
    cholesky_frule, cholesky_rrule, eigen, eigen_frule, eigen_rrule, pinv_frule, pinv_rrule,
    qr_frule, qr_rrule, solve_frule, solve_rrule, svd, svd_frule, svd_rrule, EigenCotangent,
    QrCotangent, SolveGrad, SvdCotangent,
};
use tenferro_prims::CpuContext;
use tenferro_tensor::Tensor;

use crate::db::{
    case_files, default_oracle_db_root, load_case_records, CaseRecord, DbTensor, ProbeRecord,
};
use crate::decode::{
    batched_matmul, batched_transpose, compare_tensor_maps, decode_f64_tensor_with_core_rank,
    elementwise_sign_mul, tensor_add, tensor_data_col_major, tensor_map_inner_product,
};
use crate::hvp::{central_diff_tensor_maps, perturb_input_map};
use crate::support::{classify_record, ExpectedErrorKind, RecordSupport, ReplayKind};

#[derive(Debug)]
pub struct ReplaySummary {
    pub validated_records: usize,
    pub validated_hvp_records: usize,
    pub expected_error_case_ids: Vec<String>,
    pub unsupported_records: usize,
    pub failures: Vec<String>,
}

pub fn run_database_replay() -> ReplaySummary {
    let mut summary = ReplaySummary {
        validated_records: 0,
        validated_hvp_records: 0,
        expected_error_case_ids: Vec::new(),
        unsupported_records: 0,
        failures: Vec::new(),
    };

    let Some(root) = default_oracle_db_root() else {
        summary
            .failures
            .push("vendored tensor-ad-oracles root not found".to_string());
        return summary;
    };

    let files = match case_files(&root) {
        Ok(files) => files,
        Err(err) => {
            summary.failures.push(err);
            return summary;
        }
    };

    for path in files {
        let records = match load_case_records(&path) {
            Ok(records) => records,
            Err(err) => {
                summary.failures.push(err);
                continue;
            }
        };
        for record in records {
            match replay_case(&record) {
                Ok(ReplayOutcome::Validated { hvp_checked }) => {
                    summary.validated_records += 1;
                    if hvp_checked {
                        summary.validated_hvp_records += 1;
                    }
                }
                Ok(ReplayOutcome::ExpectedError) => {
                    summary.expected_error_case_ids.push(record.case_id.clone());
                }
                Ok(ReplayOutcome::Unsupported) => summary.unsupported_records += 1,
                Err(err) => summary.failures.push(format!("{}: {err}", record.case_id)),
            }
        }
    }

    summary.expected_error_case_ids.sort();
    summary
}

enum ReplayOutcome {
    Validated { hvp_checked: bool },
    ExpectedError,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TriangularOrientation {
    Lower,
    Upper,
}

fn replay_case(record: &CaseRecord) -> Result<ReplayOutcome, String> {
    match classify_record(record) {
        RecordSupport::Supported(kind) => {
            if record.dtype != "float64" {
                return Err(format!("unsupported success dtype {}", record.dtype));
            }
            let hvp_checked = match kind {
                ReplayKind::SolveIdentity => replay_solve(record),
                ReplayKind::CholeskyIdentity => replay_cholesky(record),
                ReplayKind::QrIdentity => replay_qr(record),
                ReplayKind::SvdUAbs
                | ReplayKind::SvdS
                | ReplayKind::SvdVhAbs
                | ReplayKind::SvdUvhProduct => replay_svd(record),
                ReplayKind::EighValuesVectorsAbs => replay_eigen(record),
                ReplayKind::PinvSingularIdentity => replay_pinv_singular(record),
            }?;
            Ok(ReplayOutcome::Validated { hvp_checked })
        }
        RecordSupport::ExpectedError(ExpectedErrorKind::GaugeIllDefined) => {
            Ok(ReplayOutcome::ExpectedError)
        }
        RecordSupport::Unsupported { .. } => Ok(ReplayOutcome::Unsupported),
        RecordSupport::Unknown => Err(format!(
            "unclassified oracle family {}/{}/{} ({})",
            record.op, record.family, record.observable.kind, record.expected_behavior
        )),
    }
}

fn comparison(record: &CaseRecord) -> Result<(f64, f64), String> {
    let comparison = record
        .comparison
        .first_order()
        .ok_or_else(|| format!("missing first-order comparison for {}", record.case_id))?;
    if comparison.kind != "allclose" {
        return Err(format!("unsupported comparison kind {}", comparison.kind));
    }
    Ok((comparison.rtol, comparison.atol))
}

#[allow(dead_code)]
fn second_order_comparison(record: &CaseRecord) -> Result<(f64, f64), String> {
    let comparison = record
        .comparison
        .second_order()
        .ok_or_else(|| format!("missing second-order comparison for {}", record.case_id))?;
    if comparison.kind != "allclose" {
        return Err(format!(
            "unsupported second-order comparison kind {}",
            comparison.kind
        ));
    }
    Ok((comparison.rtol, comparison.atol))
}

fn validate_hvp<F>(
    label: &str,
    record: &CaseRecord,
    base_inputs: &BTreeMap<String, Tensor<f64>>,
    direction: &BTreeMap<String, Tensor<f64>>,
    probe: &ProbeRecord,
    eval_grad: F,
) -> Result<bool, String>
where
    F: Fn(&BTreeMap<String, Tensor<f64>>) -> Result<BTreeMap<String, Tensor<f64>>, String>,
{
    let Some(expected_torch_hvp) = probe.pytorch_ref.hvp.as_ref() else {
        return Ok(false);
    };
    let expected_fd_hvp = probe
        .fd_ref
        .hvp
        .as_ref()
        .ok_or_else(|| format!("missing fd_ref.hvp for {}", record.case_id))?;
    let expected_hvp_torch = decode_input_map_like(record, expected_torch_hvp)?;
    let expected_hvp_fd = decode_input_map_like(record, expected_fd_hvp)?;
    let step = probe.fd_ref.step;
    let plus_inputs = perturb_input_map(base_inputs, direction, step)?;
    let minus_inputs = perturb_input_map(base_inputs, direction, -step)?;
    let grad_plus = eval_grad(&plus_inputs)?;
    let grad_minus = eval_grad(&minus_inputs)?;
    let actual_hvp = central_diff_tensor_maps(&grad_plus, &grad_minus, step)?;
    let (rtol, atol) = second_order_comparison(record)?;
    compare_tensor_maps(
        &format!("{label}.hvp.fd"),
        &expected_hvp_fd,
        &actual_hvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps(
        &format!("{label}.hvp.torch"),
        &expected_hvp_torch,
        &actual_hvp,
        rtol,
        atol,
    )?;
    Ok(true)
}

fn solve_rhs_core_rank(a: &DbTensor, b: &DbTensor) -> usize {
    if b.shape.len() + 1 == a.shape.len() {
        1
    } else {
        2
    }
}

fn decode_inputs(record: &CaseRecord) -> Result<BTreeMap<String, Tensor<f64>>, String> {
    let mut inputs = BTreeMap::new();
    for (name, tensor) in &record.inputs {
        let core_rank = match (record.op.as_str(), name.as_str()) {
            ("solve", "b") => solve_rhs_core_rank(record.inputs.get("a").unwrap(), tensor),
            _ => {
                if tensor.shape.len() <= 1 {
                    1
                } else {
                    2
                }
            }
        };
        inputs.insert(
            name.clone(),
            decode_f64_tensor_with_core_rank(tensor, core_rank)?,
        );
    }
    Ok(inputs)
}

fn decode_input_map_like(
    record: &CaseRecord,
    encoded: &BTreeMap<String, DbTensor>,
) -> Result<BTreeMap<String, Tensor<f64>>, String> {
    let mut out = BTreeMap::new();
    for (name, tensor) in encoded {
        let core_rank = match (record.op.as_str(), name.as_str()) {
            ("solve", "b") => solve_rhs_core_rank(record.inputs.get("a").unwrap(), tensor),
            _ => {
                if tensor.shape.len() <= 1 {
                    1
                } else {
                    2
                }
            }
        };
        out.insert(
            name.clone(),
            decode_f64_tensor_with_core_rank(tensor, core_rank)?,
        );
    }
    Ok(out)
}

fn observable_core_rank(record: &CaseRecord, key: &str) -> Result<usize, String> {
    match (record.op.as_str(), record.family.as_str(), key) {
        ("solve", "identity", "value") => Ok(solve_rhs_core_rank(
            record.inputs.get("a").unwrap(),
            record.inputs.get("b").unwrap(),
        )),
        ("cholesky", "identity", "value") => Ok(2),
        ("qr", "identity", "output_0") | ("qr", "identity", "output_1") => Ok(2),
        ("svd", "u_abs", "u") => Ok(2),
        ("svd", "s", "s") => Ok(1),
        ("svd", "vh_abs", "s") => Ok(1),
        ("svd", "vh_abs", "vh") => Ok(2),
        ("svd", "uvh_product", "s") => Ok(1),
        ("svd", "uvh_product", "uvh") => Ok(2),
        ("eigh", "values_vectors_abs", "values") => Ok(1),
        ("eigh", "values_vectors_abs", "vectors") => Ok(2),
        ("pinv_singular", "identity", "value") => Ok(2),
        _ => Err(format!(
            "unsupported observable tensor key {key} for {}/{}",
            record.op, record.family
        )),
    }
}

fn decode_observable_map(
    record: &CaseRecord,
    encoded: &BTreeMap<String, DbTensor>,
) -> Result<BTreeMap<String, Tensor<f64>>, String> {
    let mut out = BTreeMap::new();
    for (name, tensor) in encoded {
        let core_rank = observable_core_rank(record, name)?;
        out.insert(
            name.clone(),
            decode_f64_tensor_with_core_rank(tensor, core_rank)?,
        );
    }
    Ok(out)
}

fn probe(record: &CaseRecord) -> Result<&ProbeRecord, String> {
    record
        .probes
        .first()
        .ok_or_else(|| format!("missing probe for {}", record.case_id))
}

fn check_adjoint_identity(
    record: &CaseRecord,
    cotangent: &BTreeMap<String, Tensor<f64>>,
    jvp: &BTreeMap<String, Tensor<f64>>,
    vjp: &BTreeMap<String, Tensor<f64>>,
    direction: &BTreeMap<String, Tensor<f64>>,
) -> Result<(), String> {
    let (rtol, atol) = comparison(record)?;
    let lhs = tensor_map_inner_product(cotangent, jvp)?;
    let rhs = tensor_map_inner_product(vjp, direction)?;
    let allowed = atol + rtol * lhs.abs();
    if (lhs - rhs).abs() > allowed {
        return Err(format!(
            "adjoint identity mismatch: lhs={lhs}, rhs={rhs}, allowed={allowed}"
        ));
    }
    Ok(())
}

fn apply_hermitian_wrapper(tensor: &Tensor<f64>) -> Result<Tensor<f64>, String> {
    let transpose = batched_transpose(tensor)?;
    Ok(tensor_add(tensor, &transpose))
}

fn replay_solve(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like(record, &probe.direction)?;
    let cotangent = decode_observable_map(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;

    let mut ctx = CpuContext::new(1);
    let (_x, dx) = solve_frule(
        &mut ctx,
        inputs.get("a").unwrap(),
        inputs.get("b").unwrap(),
        direction.get("a").unwrap(),
        direction.get("b").unwrap(),
    )
    .map_err(|err| format!("solve_frule failed: {err}"))?;
    let grad = solve_rrule(
        &mut ctx,
        inputs.get("a").unwrap(),
        inputs.get("b").unwrap(),
        cotangent.get("value").unwrap(),
    )
    .map_err(|err| format!("solve_rrule failed: {err}"))?;

    let actual_jvp = BTreeMap::from([(String::from("value"), dx)]);
    let actual_vjp = BTreeMap::from([(String::from("a"), grad.a), (String::from("b"), grad.b)]);
    compare_tensor_maps("solve.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps(
        "solve.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("solve.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp("solve", record, &inputs, &direction, probe, |perturbed| {
        let mut ctx = CpuContext::new(1);
        let grad = solve_rrule(
            &mut ctx,
            perturbed.get("a").unwrap(),
            perturbed.get("b").unwrap(),
            cotangent.get("value").unwrap(),
        )
        .map_err(|err| format!("solve_rrule failed during HVP replay: {err}"))?;
        Ok(BTreeMap::from([
            (String::from("a"), grad.a),
            (String::from("b"), grad.b),
        ]))
    })?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn replay_cholesky(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like(record, &probe.direction)?;
    let cotangent = decode_observable_map(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let orientation = infer_triangular_orientation(
        expected_jvp_fd
            .get("value")
            .ok_or_else(|| format!("missing cholesky fd_ref value for {}", record.case_id))?,
    )?;
    let wrapped_a = apply_hermitian_wrapper(inputs.get("a").unwrap())?;
    let wrapped_da = apply_hermitian_wrapper(direction.get("a").unwrap())?;

    let mut ctx = CpuContext::new(1);
    let (_l, dl) = cholesky_frule(&mut ctx, &wrapped_a, &wrapped_da)
        .map_err(|err| format!("cholesky_frule failed: {err}"))?;
    let raw_cotangent = match orientation {
        TriangularOrientation::Lower => cotangent.get("value").unwrap().clone(),
        TriangularOrientation::Upper => batched_transpose(cotangent.get("value").unwrap())?,
    };
    let grad = cholesky_rrule(&mut ctx, &wrapped_a, &raw_cotangent)
        .map_err(|err| format!("cholesky_rrule failed: {err}"))?;

    let actual_value = match orientation {
        TriangularOrientation::Lower => dl,
        TriangularOrientation::Upper => batched_transpose(&dl)?,
    };
    let actual_jvp = BTreeMap::from([(String::from("value"), actual_value)]);
    let actual_vjp = BTreeMap::from([(String::from("a"), apply_hermitian_wrapper(&grad)?)]);
    compare_tensor_maps("cholesky.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps(
        "cholesky.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("cholesky.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp(
        "cholesky",
        record,
        &inputs,
        &direction,
        probe,
        |perturbed| {
            let wrapped = apply_hermitian_wrapper(perturbed.get("a").unwrap())?;
            let mut ctx = CpuContext::new(1);
            let grad = cholesky_rrule(&mut ctx, &wrapped, &raw_cotangent)
                .map_err(|err| format!("cholesky_rrule failed during HVP replay: {err}"))?;
            Ok(BTreeMap::from([(
                String::from("a"),
                apply_hermitian_wrapper(&grad)?,
            )]))
        },
    )?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn infer_triangular_orientation(tensor: &Tensor<f64>) -> Result<TriangularOrientation, String> {
    let dims = tensor.dims();
    if dims.len() < 2 || dims[0] != dims[1] {
        return Err(format!(
            "triangular observable must be square, got dims {:?}",
            dims
        ));
    }
    let n = dims[0];
    if n <= 1 {
        return Ok(TriangularOrientation::Lower);
    }

    let values = tensor_data_col_major(tensor);
    let bc = if dims.len() <= 2 {
        1
    } else {
        dims[2..].iter().product()
    };

    let mut upper_norm = 0.0f64;
    let mut lower_norm = 0.0f64;
    for batch in 0..bc {
        let base = batch * n * n;
        for j in 0..n {
            for i in 0..n {
                let value = values[base + i + j * n].abs();
                if i < j {
                    upper_norm += value;
                } else if i > j {
                    lower_norm += value;
                }
            }
        }
    }

    let tol = 1e-12;
    if upper_norm <= tol || lower_norm <= tol {
        return Ok(if lower_norm <= tol {
            TriangularOrientation::Upper
        } else {
            TriangularOrientation::Lower
        });
    }
    Err(format!(
        "failed to infer triangular orientation: upper_norm={upper_norm}, lower_norm={lower_norm}"
    ))
}

fn replay_qr(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like(record, &probe.direction)?;
    let cotangent = decode_observable_map(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;

    let mut ctx = CpuContext::new(1);
    let (_qr, dqr) = qr_frule(
        &mut ctx,
        inputs.get("a").unwrap(),
        direction.get("a").unwrap(),
    )
    .map_err(|err| format!("qr_frule failed: {err}"))?;
    let grad = qr_rrule(
        &mut ctx,
        inputs.get("a").unwrap(),
        &QrCotangent {
            q: Some(cotangent.get("output_0").unwrap().clone()),
            r: Some(cotangent.get("output_1").unwrap().clone()),
        },
    )
    .map_err(|err| format!("qr_rrule failed: {err}"))?;

    let actual_jvp = BTreeMap::from([
        (String::from("output_0"), dqr.q),
        (String::from("output_1"), dqr.r),
    ]);
    let actual_vjp = BTreeMap::from([(String::from("a"), grad)]);
    compare_tensor_maps("qr.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps("qr.jvp.torch", &expected_jvp_torch, &actual_jvp, rtol, atol)?;
    compare_tensor_maps("qr.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp("qr", record, &inputs, &direction, probe, |perturbed| {
        let mut ctx = CpuContext::new(1);
        let grad = qr_rrule(
            &mut ctx,
            perturbed.get("a").unwrap(),
            &QrCotangent {
                q: Some(cotangent.get("output_0").unwrap().clone()),
                r: Some(cotangent.get("output_1").unwrap().clone()),
            },
        )
        .map_err(|err| format!("qr_rrule failed during HVP replay: {err}"))?;
        Ok(BTreeMap::from([(String::from("a"), grad)]))
    })?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn svd_observable_jvp(
    family: &str,
    primal_u: &Tensor<f64>,
    primal_vt: &Tensor<f64>,
    du: &Tensor<f64>,
    ds: &Tensor<f64>,
    dvt: &Tensor<f64>,
) -> Result<BTreeMap<String, Tensor<f64>>, String> {
    match family {
        "u_abs" => Ok(BTreeMap::from([(
            String::from("u"),
            elementwise_sign_mul(primal_u, du),
        )])),
        "s" => Ok(BTreeMap::from([(String::from("s"), ds.clone())])),
        "vh_abs" => Ok(BTreeMap::from([
            (String::from("s"), ds.clone()),
            (String::from("vh"), elementwise_sign_mul(primal_vt, dvt)),
        ])),
        "uvh_product" => Ok(BTreeMap::from([
            (String::from("s"), ds.clone()),
            (
                String::from("uvh"),
                tensor_add(
                    &batched_matmul(du, primal_vt)?,
                    &batched_matmul(primal_u, dvt)?,
                ),
            ),
        ])),
        _ => Err(format!("unsupported svd family {family}")),
    }
}

fn svd_observable_cotangent(
    family: &str,
    primal_u: &Tensor<f64>,
    primal_vt: &Tensor<f64>,
    observable_cotangent: &BTreeMap<String, Tensor<f64>>,
) -> Result<SvdCotangent<f64>, String> {
    match family {
        "u_abs" => Ok(SvdCotangent {
            u: Some(elementwise_sign_mul(
                primal_u,
                observable_cotangent.get("u").unwrap(),
            )),
            s: None,
            vt: None,
        }),
        "s" => Ok(SvdCotangent {
            u: None,
            s: Some(observable_cotangent.get("s").unwrap().clone()),
            vt: None,
        }),
        "vh_abs" => Ok(SvdCotangent {
            u: None,
            s: Some(observable_cotangent.get("s").unwrap().clone()),
            vt: Some(elementwise_sign_mul(
                primal_vt,
                observable_cotangent.get("vh").unwrap(),
            )),
        }),
        "uvh_product" => {
            let cot_uvh = observable_cotangent.get("uvh").unwrap();
            let vt_t = batched_transpose(primal_vt)?;
            let u_t = batched_transpose(primal_u)?;
            Ok(SvdCotangent {
                u: Some(batched_matmul(cot_uvh, &vt_t)?),
                s: Some(observable_cotangent.get("s").unwrap().clone()),
                vt: Some(batched_matmul(&u_t, cot_uvh)?),
            })
        }
        _ => Err(format!("unsupported svd family {family}")),
    }
}

fn replay_svd(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like(record, &probe.direction)?;
    let cotangent = decode_observable_map(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;

    let mut ctx = CpuContext::new(1);
    let (result, dresult) = svd_frule(
        &mut ctx,
        inputs.get("a").unwrap(),
        direction.get("a").unwrap(),
        None,
    )
    .map_err(|err| format!("svd_frule failed: {err}"))?;
    let cotangent_raw =
        svd_observable_cotangent(record.family.as_str(), &result.u, &result.vt, &cotangent)?;
    let grad = svd_rrule(&mut ctx, inputs.get("a").unwrap(), &cotangent_raw, None)
        .map_err(|err| format!("svd_rrule failed: {err}"))?;

    let actual_jvp = svd_observable_jvp(
        record.family.as_str(),
        &result.u,
        &result.vt,
        &dresult.u,
        &dresult.s,
        &dresult.vt,
    )?;
    let actual_vjp = BTreeMap::from([(String::from("a"), grad)]);

    compare_tensor_maps("svd.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps(
        "svd.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("svd.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp("svd", record, &inputs, &direction, probe, |perturbed| {
        let mut ctx = CpuContext::new(1);
        let primal = svd(&mut ctx, perturbed.get("a").unwrap(), None)
            .map_err(|err| format!("svd failed during HVP replay: {err}"))?;
        let cotangent_raw =
            svd_observable_cotangent(record.family.as_str(), &primal.u, &primal.vt, &cotangent)?;
        let grad = svd_rrule(&mut ctx, perturbed.get("a").unwrap(), &cotangent_raw, None)
            .map_err(|err| format!("svd_rrule failed during HVP replay: {err}"))?;
        Ok(BTreeMap::from([(String::from("a"), grad)]))
    })?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn replay_eigen(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like(record, &probe.direction)?;
    let cotangent = decode_observable_map(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let wrapped_a = apply_hermitian_wrapper(inputs.get("a").unwrap())?;
    let wrapped_da = apply_hermitian_wrapper(direction.get("a").unwrap())?;

    let mut ctx = CpuContext::new(1);
    let (result, dresult) = eigen_frule(&mut ctx, &wrapped_a, &wrapped_da)
        .map_err(|err| format!("eigen_frule failed: {err}"))?;
    let grad = eigen_rrule(
        &mut ctx,
        &wrapped_a,
        &EigenCotangent {
            values: Some(cotangent.get("values").unwrap().clone()),
            vectors: Some(elementwise_sign_mul(
                &result.vectors,
                cotangent.get("vectors").unwrap(),
            )),
        },
    )
    .map_err(|err| format!("eigen_rrule failed: {err}"))?;

    let actual_jvp = BTreeMap::from([
        (String::from("values"), dresult.values),
        (
            String::from("vectors"),
            elementwise_sign_mul(&result.vectors, &dresult.vectors),
        ),
    ]);
    let actual_vjp = BTreeMap::from([(String::from("a"), apply_hermitian_wrapper(&grad)?)]);
    compare_tensor_maps("eigen.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps(
        "eigen.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("eigen.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp("eigen", record, &inputs, &direction, probe, |perturbed| {
        let wrapped = apply_hermitian_wrapper(perturbed.get("a").unwrap())?;
        let mut ctx = CpuContext::new(1);
        let primal = eigen(&mut ctx, &wrapped)
            .map_err(|err| format!("eigen failed during HVP replay: {err}"))?;
        let grad = eigen_rrule(
            &mut ctx,
            &wrapped,
            &EigenCotangent {
                values: Some(cotangent.get("values").unwrap().clone()),
                vectors: Some(elementwise_sign_mul(
                    &primal.vectors,
                    cotangent.get("vectors").unwrap(),
                )),
            },
        )
        .map_err(|err| format!("eigen_rrule failed during HVP replay: {err}"))?;
        Ok(BTreeMap::from([(
            String::from("a"),
            apply_hermitian_wrapper(&grad)?,
        )]))
    })?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn pinv_factor_product(a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, String> {
    let bt = batched_transpose(b)?;
    batched_matmul(a, &bt)
}

fn pinv_factor_pullback(
    grad_matrix: &Tensor<f64>,
    a: &Tensor<f64>,
    b: &Tensor<f64>,
) -> Result<SolveGrad<f64>, String> {
    let grad_t = batched_transpose(grad_matrix)?;
    Ok(SolveGrad {
        a: batched_matmul(grad_matrix, b)?,
        b: batched_matmul(&grad_t, a)?,
    })
}

fn replay_pinv_singular(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs(record)?;
    let a = inputs.get("a").unwrap();
    let b = inputs.get("b").unwrap();
    let probe = probe(record)?;
    let direction = decode_input_map_like(record, &probe.direction)?;
    let cotangent = decode_observable_map(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;

    let matrix = pinv_factor_product(a, b)?;
    let dmatrix = tensor_add(
        &pinv_factor_product(direction.get("a").unwrap(), b)?,
        &pinv_factor_product(a, direction.get("b").unwrap())?,
    );

    let mut ctx = CpuContext::new(1);
    let (_ap, dap) = pinv_frule(&mut ctx, &matrix, &dmatrix, None)
        .map_err(|err| format!("pinv_frule failed: {err}"))?;
    let grad_matrix = pinv_rrule(&mut ctx, &matrix, cotangent.get("value").unwrap(), None)
        .map_err(|err| format!("pinv_rrule failed: {err}"))?;
    let grad_factors = pinv_factor_pullback(&grad_matrix, a, b)?;

    let actual_jvp = BTreeMap::from([(String::from("value"), dap)]);
    let actual_vjp = BTreeMap::from([
        (String::from("a"), grad_factors.a),
        (String::from("b"), grad_factors.b),
    ]);
    compare_tensor_maps("pinv.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps(
        "pinv.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("pinv.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp("pinv", record, &inputs, &direction, probe, |perturbed| {
        let pa = perturbed.get("a").unwrap();
        let pb = perturbed.get("b").unwrap();
        let matrix = pinv_factor_product(pa, pb)?;
        let mut ctx = CpuContext::new(1);
        let grad_matrix = pinv_rrule(&mut ctx, &matrix, cotangent.get("value").unwrap(), None)
            .map_err(|err| format!("pinv_rrule failed during HVP replay: {err}"))?;
        let grad_factors = pinv_factor_pullback(&grad_matrix, pa, pb)?;
        Ok(BTreeMap::from([
            (String::from("a"), grad_factors.a),
            (String::from("b"), grad_factors.b),
        ]))
    })?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}
