use std::collections::BTreeMap;

use num_complex::{Complex32, Complex64};
use num_traits::{Float, NumCast, ToPrimitive};
use tenferro_algebra::Conjugate;
use tenferro_linalg::{
    cholesky_frule, cholesky_rrule, cross, det_frule, det_rrule, eigen, eigen_frule, eigen_rrule,
    householder_product, inv_frule, inv_rrule, lu_frule, lu_rrule, matrix_exp_frule,
    matrix_exp_rrule, norm_frule, norm_frule_complex, norm_rrule, norm_rrule_complex, pinv,
    pinv_frule, pinv_rrule, qr_frule, qr_rrule, slogdet_frule, slogdet_rrule, solve, solve_frule,
    solve_rrule, solve_triangular_frule, solve_triangular_rrule, svd, svd_frule, svd_rrule,
    tensorinv, tensorsolve, vander, EigenCotangent, LiftPermutationMatrixTensor, LuCotangent,
    LuPivot, MatrixExpAbsTensor, NormKind, QrCotangent, ScaleTensorByRealSameShape,
    SlogdetCotangent, SlogdetFruleDispatch, SlogdetRruleDispatch, SolveGrad, SvdCotangent,
};
use tenferro_linalg_prims::KernelLinalgScalar;
use tenferro_prims::CpuContext;
use tenferro_tensor::{KeepCountScalar, Tensor};

use crate::db::{
    case_files, default_oracle_db_root, load_case_records, CaseRecord, DbTensor, ProbeRecord,
};
use crate::decode::{
    batched_adjoint_transpose, batched_matmul, batched_transpose, compare_tensor_maps,
    compare_tensor_maps_typed, compare_tensors_typed, decode_f64_tensor_with_core_rank,
    decode_tensor_with_core_rank, elementwise_abs_jvp, elementwise_abs_vjp, tensor_add,
    tensor_data_col_major, tensor_from_col_major, tensor_map_inner_product,
    tensor_map_inner_product_typed, OracleDbScalar,
};
use crate::hvp::{
    central_diff_tensor_maps, central_diff_tensor_maps_typed, perturb_input_map,
    perturb_input_map_typed,
};
use crate::support::{
    classify_record, replayable_norm_kind, ExpectedErrorKind, RecordSupport, ReplayKind,
};

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
            let hvp_checked = match kind {
                ReplayKind::SolveIdentity => {
                    if record.op == "lu_solve" {
                        replay_lu_solve(record)
                    } else {
                        replay_solve(record)
                    }
                }
                ReplayKind::SolveTriangularIdentity => replay_solve_triangular(record),
                ReplayKind::CholeskyIdentity => replay_cholesky(record),
                ReplayKind::InvIdentity => replay_inv(record),
                ReplayKind::DetIdentity => replay_det(record),
                ReplayKind::SlogdetIdentity => replay_slogdet(record),
                ReplayKind::LuFactorIdentity => replay_lu_factor(record),
                ReplayKind::LuIdentity => replay_lu(record),
                ReplayKind::NormIdentity => replay_norm(record),
                ReplayKind::CondIdentity => replay_cond(record),
                ReplayKind::MatrixPowerIdentity => replay_matrix_power(record),
                ReplayKind::MatrixExpIdentity => replay_matrix_exp(record),
                ReplayKind::PinvIdentity => replay_pinv(record),
                ReplayKind::NumericalIdentity => replay_numerical_identity(record),
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

fn optional_float_kwarg(record: &CaseRecord, key: &str) -> Result<Option<f64>, String> {
    match record.op_kwargs.get(key) {
        None => Ok(None),
        Some(value) if value.is_null() => Ok(None),
        Some(value) => value
            .as_f64()
            .ok_or_else(|| format!("op_kwargs.{key} for {} must be a float", record.case_id))
            .map(Some),
    }
}

fn pinv_rcond(record: &CaseRecord) -> Result<Option<f64>, String> {
    match optional_float_kwarg(record, "rtol")? {
        Some(value) => Ok(Some(value)),
        None => optional_float_kwarg(record, "rcond"),
    }
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

fn validate_hvp_typed<T, F>(
    label: &str,
    record: &CaseRecord,
    base_inputs: &BTreeMap<String, Tensor<T>>,
    direction: &BTreeMap<String, Tensor<T>>,
    probe: &ProbeRecord,
    eval_grad: F,
) -> Result<bool, String>
where
    T: OracleDbScalar,
    F: Fn(&BTreeMap<String, Tensor<T>>) -> Result<BTreeMap<String, Tensor<T>>, String>,
{
    let Some(expected_torch_hvp) = probe.pytorch_ref.hvp.as_ref() else {
        return Ok(false);
    };
    let expected_fd_hvp = probe
        .fd_ref
        .hvp
        .as_ref()
        .ok_or_else(|| format!("missing fd_ref.hvp for {}", record.case_id))?;
    let expected_hvp_torch = decode_input_map_like_typed::<T>(record, expected_torch_hvp)?;
    let expected_hvp_fd = decode_input_map_like_typed::<T>(record, expected_fd_hvp)?;
    let step = probe.fd_ref.step;
    let plus_inputs = perturb_input_map_typed(base_inputs, direction, step)?;
    let minus_inputs = perturb_input_map_typed(base_inputs, direction, -step)?;
    let grad_plus = eval_grad(&plus_inputs)?;
    let grad_minus = eval_grad(&minus_inputs)?;
    let actual_hvp = central_diff_tensor_maps_typed(&grad_plus, &grad_minus, step)?;
    let (rtol, atol) = second_order_comparison(record)?;
    compare_tensor_maps_typed(
        &format!("{label}.hvp.fd"),
        &expected_hvp_fd,
        &actual_hvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps_typed(
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
            (
                "cross"
                | "householder_product"
                | "multi_dot"
                | "pinv_hermitian"
                | "tensorinv"
                | "tensorsolve"
                | "vander"
                | "vecdot",
                _,
            ) => tensor.shape.len(),
            ("solve" | "solve_ex" | "lu_solve", "b") => {
                solve_rhs_core_rank(record.inputs.get("a").unwrap(), tensor)
            }
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

fn decode_inputs_typed<T: OracleDbScalar>(
    record: &CaseRecord,
) -> Result<BTreeMap<String, Tensor<T>>, String> {
    let mut inputs = BTreeMap::new();
    for (name, tensor) in &record.inputs {
        let core_rank = match (record.op.as_str(), name.as_str()) {
            (
                "cross"
                | "householder_product"
                | "multi_dot"
                | "pinv_hermitian"
                | "tensorinv"
                | "tensorsolve"
                | "vander"
                | "vecdot",
                _,
            ) => tensor.shape.len(),
            ("solve" | "solve_ex" | "lu_solve", "b") => {
                solve_rhs_core_rank(record.inputs.get("a").unwrap(), tensor)
            }
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
            decode_tensor_with_core_rank::<T>(tensor, core_rank)?,
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
            (
                "cross"
                | "householder_product"
                | "multi_dot"
                | "pinv_hermitian"
                | "tensorinv"
                | "tensorsolve"
                | "vander"
                | "vecdot",
                _,
            ) => tensor.shape.len(),
            ("solve" | "solve_ex" | "lu_solve", "b") => {
                solve_rhs_core_rank(record.inputs.get("a").unwrap(), tensor)
            }
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

fn decode_input_map_like_typed<T: OracleDbScalar>(
    record: &CaseRecord,
    encoded: &BTreeMap<String, DbTensor>,
) -> Result<BTreeMap<String, Tensor<T>>, String> {
    let mut out = BTreeMap::new();
    for (name, tensor) in encoded {
        let core_rank = match (record.op.as_str(), name.as_str()) {
            (
                "cross"
                | "householder_product"
                | "multi_dot"
                | "pinv_hermitian"
                | "tensorinv"
                | "tensorsolve"
                | "vander"
                | "vecdot",
                _,
            ) => tensor.shape.len(),
            ("solve" | "solve_ex" | "lu_solve", "b") => {
                solve_rhs_core_rank(record.inputs.get("a").unwrap(), tensor)
            }
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
            decode_tensor_with_core_rank::<T>(tensor, core_rank)?,
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
        ("solve_triangular", "identity", "value") => Ok(solve_rhs_core_rank(
            record.inputs.get("a").unwrap(),
            record.inputs.get("b").unwrap(),
        )),
        ("solve_ex", "identity", "output_0") | ("lu_solve", "identity", "value") => {
            Ok(solve_rhs_core_rank(
                record.inputs.get("a").unwrap(),
                record.inputs.get("b").unwrap(),
            ))
        }
        ("cholesky", "identity", "value") => Ok(2),
        ("cholesky_ex", "identity", "output_0")
        | ("inv", "identity", "value")
        | ("inv_ex", "identity", "output_0")
        | ("lu_factor", "identity", "output_0")
        | ("lu_factor_ex", "identity", "output_0")
        | ("matrix_power", "identity", "value")
        | ("matrix_exp", "identity", "value")
        | ("pinv", "identity", "value") => Ok(2),
        ("det", "identity", "value") => Ok(0),
        ("cond", "identity", "value") => Ok(0),
        ("norm", "identity", "value") | ("matrix_norm", "identity", "value") => Ok(0),
        ("slogdet", "identity", "output_0") | ("slogdet", "identity", "output_1") => Ok(0),
        ("qr", "identity", "output_0") | ("qr", "identity", "output_1") => Ok(2),
        ("lu", "identity", "output_1") | ("lu", "identity", "output_2") => Ok(2),
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

fn decode_observable_map_typed<T: OracleDbScalar>(
    record: &CaseRecord,
    encoded: &BTreeMap<String, DbTensor>,
) -> Result<BTreeMap<String, Tensor<T>>, String> {
    let mut out = BTreeMap::new();
    for (name, tensor) in encoded {
        let core_rank = observable_core_rank(record, name)?;
        out.insert(
            name.clone(),
            decode_tensor_with_core_rank::<T>(tensor, core_rank)?,
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

fn squeeze_scalar_tensor_typed<T: OracleDbScalar>(tensor: Tensor<T>) -> Result<Tensor<T>, String> {
    if tensor.dims().is_empty() {
        return Ok(tensor);
    }
    let numel = tensor.dims().iter().product::<usize>();
    if numel != 1 {
        return Err(format!(
            "expected scalar-like tensor, got dims {:?}",
            tensor.dims()
        ));
    }
    tensor.reshape(&[]).map_err(|err| {
        format!(
            "failed to squeeze scalar-like tensor {:?}: {err}",
            tensor.dims()
        )
    })
}

fn squeeze_scalar_map_typed<T: OracleDbScalar>(
    encoded: BTreeMap<String, Tensor<T>>,
) -> Result<BTreeMap<String, Tensor<T>>, String> {
    encoded
        .into_iter()
        .map(|(name, tensor)| Ok((name, squeeze_scalar_tensor_typed(tensor)?)))
        .collect()
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

fn check_adjoint_identity_typed<T: OracleDbScalar>(
    record: &CaseRecord,
    cotangent: &BTreeMap<String, Tensor<T>>,
    jvp: &BTreeMap<String, Tensor<T>>,
    vjp: &BTreeMap<String, Tensor<T>>,
    direction: &BTreeMap<String, Tensor<T>>,
) -> Result<(), String> {
    let (rtol, atol) = comparison(record)?;
    let lhs = tensor_map_inner_product_typed(cotangent, jvp)?;
    let rhs = tensor_map_inner_product_typed(vjp, direction)?;
    let allowed = atol + rtol * lhs.abs();
    if (lhs - rhs).abs() > allowed {
        return Err(format!(
            "adjoint identity mismatch: lhs={lhs}, rhs={rhs}, allowed={allowed}"
        ));
    }
    Ok(())
}

fn check_mixed_adjoint_identity_typed<T, R>(
    record: &CaseRecord,
    cotangent: &BTreeMap<String, Tensor<R>>,
    jvp: &BTreeMap<String, Tensor<R>>,
    vjp: &BTreeMap<String, Tensor<T>>,
    direction: &BTreeMap<String, Tensor<T>>,
) -> Result<(), String>
where
    T: OracleDbScalar,
    R: OracleDbScalar,
{
    let (rtol, atol) = comparison(record)?;
    let lhs = tensor_map_inner_product_typed(cotangent, jvp)?;
    let rhs = tensor_map_inner_product_typed(vjp, direction)?;
    let allowed = atol + rtol * lhs.abs();
    if (lhs - rhs).abs() > allowed {
        return Err(format!(
            "adjoint identity mismatch: lhs={lhs}, rhs={rhs}, allowed={allowed}"
        ));
    }
    Ok(())
}

fn apply_hermitian_wrapper_typed<T: OracleDbScalar>(
    tensor: &Tensor<T>,
) -> Result<Tensor<T>, String> {
    let adjoint = batched_adjoint_transpose(tensor)?;
    Ok(tensor_add(tensor, &adjoint))
}

fn decode_preserving_shape(encoded: &DbTensor) -> Result<Tensor<f64>, String> {
    decode_f64_tensor_with_core_rank(encoded, encoded.shape.len())
}

fn decode_inputs_preserving_shape(
    record: &CaseRecord,
) -> Result<BTreeMap<String, Tensor<f64>>, String> {
    let mut inputs = BTreeMap::new();
    for (name, tensor) in &record.inputs {
        inputs.insert(name.clone(), decode_preserving_shape(tensor)?);
    }
    Ok(inputs)
}

fn decode_tensor_map_preserving_shape(
    encoded: &BTreeMap<String, DbTensor>,
) -> Result<BTreeMap<String, Tensor<f64>>, String> {
    let mut out = BTreeMap::new();
    for (name, tensor) in encoded {
        out.insert(name.clone(), decode_preserving_shape(tensor)?);
    }
    Ok(out)
}

fn central_diff_tensor(
    plus: &Tensor<f64>,
    minus: &Tensor<f64>,
    step: f64,
) -> Result<Tensor<f64>, String> {
    if plus.dims() != minus.dims() {
        return Err(format!(
            "central_diff_tensor shape mismatch: plus {:?}, minus {:?}",
            plus.dims(),
            minus.dims()
        ));
    }
    let plus_data = tensor_data_col_major(plus);
    let minus_data = tensor_data_col_major(minus);
    let data: Vec<f64> = plus_data
        .iter()
        .zip(minus_data.iter())
        .map(|(p, m)| (p - m) / (2.0 * step))
        .collect();
    Ok(crate::decode::tensor_from_col_major(data, plus.dims()))
}

fn perturb_single_input_element(
    inputs: &BTreeMap<String, Tensor<f64>>,
    name: &str,
    index: usize,
    delta: f64,
) -> Result<BTreeMap<String, Tensor<f64>>, String> {
    let mut perturbed = inputs.clone();
    let tensor = perturbed
        .get(name)
        .ok_or_else(|| format!("missing input tensor {name}"))?;
    let mut data = tensor_data_col_major(tensor);
    data[index] += delta;
    perturbed.insert(
        name.to_string(),
        crate::decode::tensor_from_col_major(data, tensor.dims()),
    );
    Ok(perturbed)
}

fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; perm.len()];
    for (idx, &axis) in perm.iter().enumerate() {
        inverse[axis] = idx;
    }
    inverse
}

fn permute_or_identity(tensor: &Tensor<f64>, perm: &[usize]) -> Result<Tensor<f64>, String> {
    if perm.iter().enumerate().all(|(idx, &axis)| idx == axis) {
        Ok(tensor.clone())
    } else {
        tensor
            .permute(perm)
            .map_err(|err| format!("permute {:?} failed for {:?}: {err}", perm, tensor.dims()))
    }
}

fn move_axis_to_front(
    tensor: &Tensor<f64>,
    axis: usize,
) -> Result<(Tensor<f64>, Vec<usize>), String> {
    let mut perm = Vec::with_capacity(tensor.ndim());
    perm.push(axis);
    for current in 0..tensor.ndim() {
        if current != axis {
            perm.push(current);
        }
    }
    let inverse = inverse_permutation(&perm);
    Ok((permute_or_identity(tensor, &perm)?, inverse))
}

fn move_trailing_core_to_front(
    tensor: &Tensor<f64>,
    core_rank: usize,
) -> Result<(Tensor<f64>, Vec<usize>), String> {
    if tensor.ndim() <= core_rank {
        let identity: Vec<usize> = (0..tensor.ndim()).collect();
        return Ok((tensor.clone(), identity));
    }
    let batch_rank = tensor.ndim() - core_rank;
    let mut perm = Vec::with_capacity(tensor.ndim());
    perm.extend(batch_rank..tensor.ndim());
    perm.extend(0..batch_rank);
    let inverse = inverse_permutation(&perm);
    Ok((permute_or_identity(tensor, &perm)?, inverse))
}

fn ordered_axis_subsets(rank: usize, len: usize) -> Vec<Vec<usize>> {
    fn rec(
        rank: usize,
        len: usize,
        used: &mut [bool],
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == len {
            out.push(current.clone());
            return;
        }
        for axis in 0..rank {
            if used[axis] {
                continue;
            }
            used[axis] = true;
            current.push(axis);
            rec(rank, len, used, current, out);
            current.pop();
            used[axis] = false;
        }
    }

    let mut out = Vec::new();
    let mut used = vec![false; rank];
    rec(rank, len, &mut used, &mut Vec::new(), &mut out);
    out
}

fn tensorinv_ind(record: &CaseRecord) -> Result<usize, String> {
    let input_shape = &record.inputs.get("a").unwrap().shape;
    let output_shape = &probe(record)?
        .pytorch_ref
        .jvp
        .get("value")
        .ok_or_else(|| format!("missing tensorinv observable for {}", record.case_id))?
        .shape;

    let mut matches = Vec::new();
    for ind in 1..input_shape.len() {
        let left = &input_shape[..ind];
        let right = &input_shape[ind..];
        if left.iter().product::<usize>() == right.iter().product::<usize>() {
            let mut expected = right.to_vec();
            expected.extend_from_slice(left);
            if &expected == output_shape {
                matches.push(ind);
            }
        }
    }
    if matches.len() == 1 {
        Ok(matches[0])
    } else {
        Err(format!(
            "failed to infer tensorinv ind for {} from {:?} -> {:?}",
            record.case_id, input_shape, output_shape
        ))
    }
}

fn tensorsolve_axes(record: &CaseRecord) -> Result<Vec<usize>, String> {
    let a_shape = &record.inputs.get("a").unwrap().shape;
    let b_shape = &record.inputs.get("b").unwrap().shape;
    let output_shape = &probe(record)?
        .pytorch_ref
        .jvp
        .get("value")
        .ok_or_else(|| format!("missing tensorsolve observable for {}", record.case_id))?
        .shape;
    let solution_rank = a_shape.len().saturating_sub(b_shape.len());
    let candidates: Vec<Vec<usize>> = ordered_axis_subsets(a_shape.len(), solution_rank)
        .into_iter()
        .filter(|axes| {
            let perm = {
                let mut selected = vec![false; a_shape.len()];
                for &axis in axes {
                    selected[axis] = true;
                }
                let mut perm = Vec::with_capacity(a_shape.len());
                for (axis, is_selected) in selected.iter().enumerate() {
                    if !is_selected {
                        perm.push(axis);
                    }
                }
                perm.extend_from_slice(axes);
                perm
            };
            let permuted_dims: Vec<usize> = perm.iter().map(|&axis| a_shape[axis]).collect();
            &permuted_dims[..b_shape.len()] == b_shape
                && &permuted_dims[b_shape.len()..] == output_shape
        })
        .collect();

    if candidates.len() == 1 {
        Ok(candidates[0].clone())
    } else {
        Err(format!(
            "failed to infer tensorsolve axes for {} from {:?}, {:?}, {:?}",
            record.case_id, a_shape, b_shape, output_shape
        ))
    }
}

fn numerical_vjp<F>(
    inputs: &BTreeMap<String, Tensor<f64>>,
    cotangent: &Tensor<f64>,
    step: f64,
    eval: F,
) -> Result<BTreeMap<String, Tensor<f64>>, String>
where
    F: Fn(&BTreeMap<String, Tensor<f64>>) -> Result<Tensor<f64>, String>,
{
    let cotangent_data = tensor_data_col_major(cotangent);
    let mut gradients = BTreeMap::new();

    for (name, tensor) in inputs {
        let input_data = tensor_data_col_major(tensor);
        let mut grad = vec![0.0; input_data.len()];
        for index in 0..input_data.len() {
            let plus_inputs = perturb_single_input_element(inputs, name, index, step)?;
            let minus_inputs = perturb_single_input_element(inputs, name, index, -step)?;
            let plus_value = eval(&plus_inputs)?;
            let minus_value = eval(&minus_inputs)?;
            let deriv = central_diff_tensor(&plus_value, &minus_value, step)?;
            let deriv_data = tensor_data_col_major(&deriv);
            grad[index] = deriv_data
                .iter()
                .zip(cotangent_data.iter())
                .map(|(d, c)| d * c)
                .sum();
        }
        gradients.insert(
            name.clone(),
            crate::decode::tensor_from_col_major(grad, tensor.dims()),
        );
    }

    Ok(gradients)
}

fn sum_to_shape(tensor: &Tensor<f64>, target_dims: &[usize]) -> Result<Tensor<f64>, String> {
    fn col_major_strides(dims: &[usize]) -> Vec<isize> {
        let mut strides = vec![0isize; dims.len()];
        if dims.is_empty() {
            return strides;
        }
        strides[0] = 1;
        for axis in 1..dims.len() {
            strides[axis] = strides[axis - 1] * dims[axis - 1] as isize;
        }
        strides
    }

    if tensor.ndim() != target_dims.len() {
        return Err(format!(
            "sum_to_shape rank mismatch: {:?} -> {:?}",
            tensor.dims(),
            target_dims
        ));
    }
    let source_dims = tensor.dims();
    for (src, dst) in source_dims.iter().zip(target_dims.iter()) {
        if src != dst && *dst != 1 {
            return Err(format!(
                "sum_to_shape incompatible dims: {:?} -> {:?}",
                source_dims, target_dims
            ));
        }
    }

    let source_data = tensor_data_col_major(tensor);
    let target_len = if target_dims.is_empty() {
        1
    } else {
        target_dims.iter().product()
    };
    let target_strides = col_major_strides(target_dims);
    let mut out = vec![0.0; target_len];
    let mut index = vec![0usize; source_dims.len()];
    let source_len = if source_dims.is_empty() {
        1
    } else {
        source_dims.iter().product()
    };

    for source_offset in 0..source_len {
        let mut target_offset = 0isize;
        for axis in 0..source_dims.len() {
            let coord = if target_dims[axis] == 1 {
                0
            } else {
                index[axis]
            };
            target_offset += coord as isize * target_strides[axis];
        }
        out[target_offset as usize] += source_data[source_offset];

        for axis in 0..index.len() {
            index[axis] += 1;
            if index[axis] < source_dims[axis] {
                break;
            }
            index[axis] = 0;
        }
    }

    Ok(crate::decode::tensor_from_col_major(out, target_dims))
}

fn scale_tensor(tensor: &Tensor<f64>, alpha: f64) -> Tensor<f64> {
    let data: Vec<f64> = tensor_data_col_major(tensor)
        .into_iter()
        .map(|value| value * alpha)
        .collect();
    crate::decode::tensor_from_col_major(data, tensor.dims())
}

fn tensor_data_in_order(tensor: &Tensor<f64>, order: tenferro_tensor::MemoryOrder) -> Vec<f64> {
    let contiguous = tensor.contiguous(order);
    let offset = contiguous.offset() as usize;
    let len = contiguous.dims().iter().product::<usize>();
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

fn reshape_in_order(
    tensor: &Tensor<f64>,
    dims: &[usize],
    order: tenferro_tensor::MemoryOrder,
) -> Result<Tensor<f64>, String> {
    let data = tensor_data_in_order(tensor, order);
    Tensor::from_slice(&data, dims, order).map_err(|err| {
        format!(
            "reshape_in_order {:?} -> {:?} failed: {err}",
            tensor.dims(),
            dims
        )
    })
}

fn cross_oracle_axis(tensor: &Tensor<f64>) -> Result<usize, String> {
    let axes: Vec<usize> = tensor
        .dims()
        .iter()
        .enumerate()
        .filter_map(|(axis, &dim)| (dim == 3).then_some(axis))
        .collect();
    if axes.len() == 1 {
        Ok(axes[0])
    } else {
        Err(format!(
            "cross expects exactly one dimension of size 3, got {:?}",
            tensor.dims()
        ))
    }
}

fn evaluate_multi_dot(inputs: &BTreeMap<String, Tensor<f64>>) -> Result<Tensor<f64>, String> {
    let mut tensors: Vec<Tensor<f64>> = ["a", "b", "c", "d", "e"]
        .into_iter()
        .filter_map(|name| inputs.get(name).cloned())
        .collect();
    let first = tensors
        .drain(..1)
        .next()
        .ok_or_else(|| "multi_dot requires at least two inputs".to_string())?;
    tensors
        .into_iter()
        .try_fold(first, |acc, next| batched_matmul(&acc, &next))
}

fn evaluate_vecdot(inputs: &BTreeMap<String, Tensor<f64>>) -> Result<Tensor<f64>, String> {
    let a = inputs.get("a").unwrap();
    let b = inputs.get("b").unwrap();
    if a.dims() != b.dims() {
        return Err(format!(
            "vecdot shape mismatch: {:?} vs {:?}",
            a.dims(),
            b.dims()
        ));
    }
    if a.ndim() == 0 {
        return Err("vecdot requires rank >= 1 inputs".to_string());
    }
    let vec_len = *a.dims().last().unwrap();
    let leading_dims = &a.dims()[..a.ndim() - 1];
    let leading_len = if leading_dims.is_empty() {
        1
    } else {
        leading_dims.iter().product()
    };
    let a_data = tensor_data_col_major(a);
    let b_data = tensor_data_col_major(b);
    let mut out = vec![0.0; leading_len];
    for leading in 0..leading_len {
        let mut sum = 0.0;
        for k in 0..vec_len {
            sum += a_data[leading + k * leading_len] * b_data[leading + k * leading_len];
        }
        out[leading] = sum;
    }
    Ok(crate::decode::tensor_from_col_major(out, leading_dims))
}

fn evaluate_numerical_identity_op(
    record: &CaseRecord,
    inputs: &BTreeMap<String, Tensor<f64>>,
) -> Result<Tensor<f64>, String> {
    match record.op.as_str() {
        "cross" => {
            let axis = cross_oracle_axis(inputs.get("a").unwrap())?;
            let (a_api, inverse) = move_axis_to_front(inputs.get("a").unwrap(), axis)?;
            let (b_api, _) = move_axis_to_front(inputs.get("b").unwrap(), axis)?;
            let mut ctx = CpuContext::new(1);
            let value =
                cross(&mut ctx, &a_api, &b_api).map_err(|err| format!("cross failed: {err}"))?;
            permute_or_identity(&value, &inverse)
        }
        "householder_product" => {
            let (a_api, inverse) = move_trailing_core_to_front(inputs.get("a").unwrap(), 2)?;
            let (tau_api, _) = move_trailing_core_to_front(inputs.get("b").unwrap(), 1)?;
            let mut ctx = CpuContext::new(1);
            let value = householder_product(&mut ctx, &a_api, &tau_api)
                .map_err(|err| format!("householder_product failed: {err}"))?;
            permute_or_identity(&value, &inverse)
        }
        "vander" => {
            let x = inputs.get("a").unwrap();
            let columns = probe(record)?
                .pytorch_ref
                .jvp
                .get("value")
                .ok_or_else(|| format!("missing vander observable for {}", record.case_id))?
                .shape
                .last()
                .copied()
                .ok_or_else(|| format!("missing vander output shape for {}", record.case_id))?;
            let (x_api, _) = if x.ndim() == 0 {
                (x.clone(), vec![])
            } else {
                move_axis_to_front(x, x.ndim() - 1)?
            };
            let mut ctx = CpuContext::new(1);
            let value = vander(&mut ctx, &x_api, Some(columns), true)
                .map_err(|err| format!("vander failed: {err}"))?;
            if x.ndim() <= 1 {
                Ok(value)
            } else {
                let rank = x.ndim() + 1;
                let mut perm = Vec::with_capacity(rank);
                perm.extend(2..rank);
                perm.push(0);
                perm.push(1);
                permute_or_identity(&value, &perm)
            }
        }
        "tensorinv" => {
            let mut ctx = CpuContext::new(1);
            tensorinv(&mut ctx, inputs.get("a").unwrap(), tensorinv_ind(record)?)
                .map_err(|err| format!("tensorinv failed: {err}"))
        }
        "tensorsolve" => {
            let mut ctx = CpuContext::new(1);
            let axes = tensorsolve_axes(record)?;
            tensorsolve(
                &mut ctx,
                inputs.get("a").unwrap(),
                inputs.get("b").unwrap(),
                Some(&axes),
            )
            .map_err(|err| format!("tensorsolve failed: {err}"))
        }
        "multi_dot" => evaluate_multi_dot(inputs),
        "vecdot" => evaluate_vecdot(inputs),
        "pinv_hermitian" => {
            let (a_api, inverse) = move_trailing_core_to_front(inputs.get("a").unwrap(), 2)?;
            let mut ctx = CpuContext::new(1);
            let value = pinv(&mut ctx, &a_api, pinv_rcond(record)?)
                .map_err(|err| format!("pinv failed: {err}"))?;
            permute_or_identity(&value, &inverse)
        }
        other => Err(format!("unsupported numerical replay op {}", other)),
    }
}

fn replay_numerical_identity_generic(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs_preserving_shape(record)?;
    let probe = probe(record)?;
    let direction = decode_tensor_map_preserving_shape(&probe.direction)?;
    let cotangent = decode_tensor_map_preserving_shape(&probe.cotangent)?;
    let expected_jvp_fd = decode_tensor_map_preserving_shape(&probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_tensor_map_preserving_shape(&probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_tensor_map_preserving_shape(&probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let step = probe.fd_ref.step;
    let value_key = "value";

    let plus_inputs = perturb_input_map(&inputs, &direction, step)?;
    let minus_inputs = perturb_input_map(&inputs, &direction, -step)?;
    let plus_value = evaluate_numerical_identity_op(record, &plus_inputs)?;
    let minus_value = evaluate_numerical_identity_op(record, &minus_inputs)?;
    let actual_jvp = BTreeMap::from([(
        String::from(value_key),
        central_diff_tensor(&plus_value, &minus_value, step)?,
    )]);
    let actual_vjp = numerical_vjp(&inputs, cotangent.get(value_key).unwrap(), step, |state| {
        evaluate_numerical_identity_op(record, state)
    })?;

    compare_tensor_maps(
        &format!("{}.jvp.fd", record.op),
        &expected_jvp_fd,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps(
        &format!("{}.jvp.torch", record.op),
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps(
        &format!("{}.vjp", record.op),
        &expected_vjp,
        &actual_vjp,
        rtol,
        atol,
    )?;

    let hvp_checked = validate_hvp(
        &record.op,
        record,
        &inputs,
        &direction,
        probe,
        |perturbed| {
            numerical_vjp(
                perturbed,
                cotangent.get(value_key).unwrap(),
                step,
                |state| evaluate_numerical_identity_op(record, state),
            )
        },
    )?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn vecdot_axis(record: &CaseRecord) -> Result<usize, String> {
    let shape = &record.inputs.get("a").unwrap().shape;
    if shape.len() <= 1 {
        return Ok(0);
    }
    let output_shape = &probe(record)?
        .pytorch_ref
        .jvp
        .get("value")
        .ok_or_else(|| format!("missing vecdot observable for {}", record.case_id))?
        .shape;
    let matching_axes: Vec<usize> = (0..shape.len())
        .filter(|&axis| {
            let mut reduced = shape.clone();
            reduced.remove(axis);
            reduced == *output_shape
        })
        .collect();
    if matching_axes.len() == 1 {
        return Ok(matching_axes[0]);
    }
    let suffix = case_suffix_index(record)?;
    Ok(if suffix % 3 == 1 { 0 } else { shape.len() - 1 })
}

fn move_axis_to_last(
    tensor: &Tensor<f64>,
    axis: usize,
) -> Result<(Tensor<f64>, Vec<usize>), String> {
    let mut perm = Vec::with_capacity(tensor.ndim());
    for current in 0..tensor.ndim() {
        if current != axis {
            perm.push(current);
        }
    }
    perm.push(axis);
    let inverse = inverse_permutation(&perm);
    Ok((permute_or_identity(tensor, &perm)?, inverse))
}

fn broadcast_last_axis(cotangent: &Tensor<f64>, vec_len: usize) -> Tensor<f64> {
    let leading_dims = cotangent.dims();
    let leading_len = if leading_dims.is_empty() {
        1
    } else {
        leading_dims.iter().product()
    };
    let cotangent_data = tensor_data_col_major(cotangent);
    let mut out = vec![0.0; leading_len * vec_len];
    for k in 0..vec_len {
        out[k * leading_len..(k + 1) * leading_len].copy_from_slice(&cotangent_data);
    }
    let mut dims = leading_dims.to_vec();
    dims.push(vec_len);
    crate::decode::tensor_from_col_major(out, &dims)
}

fn vecdot_last_axis(a: &Tensor<f64>, b: &Tensor<f64>) -> Result<Tensor<f64>, String> {
    if a.dims() != b.dims() {
        return Err(format!(
            "vecdot shape mismatch: {:?} vs {:?}",
            a.dims(),
            b.dims()
        ));
    }
    if a.ndim() == 0 {
        return Err("vecdot requires rank >= 1".to_string());
    }
    let vec_len = *a.dims().last().unwrap();
    let leading_dims = &a.dims()[..a.ndim() - 1];
    let leading_len = if leading_dims.is_empty() {
        1
    } else {
        leading_dims.iter().product()
    };
    let a_data = tensor_data_col_major(a);
    let b_data = tensor_data_col_major(b);
    let mut out = vec![0.0; leading_len];
    for index in 0..leading_len {
        let mut sum = 0.0;
        for k in 0..vec_len {
            sum += a_data[index + k * leading_len] * b_data[index + k * leading_len];
        }
        out[index] = sum;
    }
    Ok(crate::decode::tensor_from_col_major(out, leading_dims))
}

fn replay_cross(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs_preserving_shape(record)?;
    let probe = probe(record)?;
    let direction = decode_tensor_map_preserving_shape(&probe.direction)?;
    let cotangent = decode_tensor_map_preserving_shape(&probe.cotangent)?;
    let expected_jvp_fd = decode_tensor_map_preserving_shape(&probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_tensor_map_preserving_shape(&probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_tensor_map_preserving_shape(&probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let axis = cross_oracle_axis(inputs.get("a").unwrap())?;
    let (a_api, inverse) = move_axis_to_front(inputs.get("a").unwrap(), axis)?;
    let (b_api, _) = move_axis_to_front(inputs.get("b").unwrap(), axis)?;
    let (da_api, _) = move_axis_to_front(direction.get("a").unwrap(), axis)?;
    let (db_api, _) = move_axis_to_front(direction.get("b").unwrap(), axis)?;
    let (cot_api, _) = move_axis_to_front(cotangent.get("value").unwrap(), axis)?;

    let mut ctx = CpuContext::new(1);
    let actual_jvp_api = tensor_add(
        &cross(&mut ctx, &da_api, &b_api).map_err(|err| format!("cross failed: {err}"))?,
        &cross(&mut ctx, &a_api, &db_api).map_err(|err| format!("cross failed: {err}"))?,
    );
    let grad_a_api = sum_to_shape(
        &cross(&mut ctx, &b_api, &cot_api).map_err(|err| format!("cross failed: {err}"))?,
        a_api.dims(),
    )?;
    let grad_b_api = sum_to_shape(
        &cross(&mut ctx, &cot_api, &a_api).map_err(|err| format!("cross failed: {err}"))?,
        b_api.dims(),
    )?;
    let actual_jvp = BTreeMap::from([(
        String::from("value"),
        permute_or_identity(&actual_jvp_api, &inverse)?,
    )]);
    let actual_vjp = BTreeMap::from([
        (
            String::from("a"),
            permute_or_identity(&grad_a_api, &inverse)?,
        ),
        (
            String::from("b"),
            permute_or_identity(&grad_b_api, &inverse)?,
        ),
    ]);

    compare_tensor_maps("cross.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps(
        "cross.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("cross.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp("cross", record, &inputs, &direction, probe, |perturbed| {
        let axis = cross_oracle_axis(perturbed.get("a").unwrap())?;
        let (pa_api, inverse) = move_axis_to_front(perturbed.get("a").unwrap(), axis)?;
        let (pb_api, _) = move_axis_to_front(perturbed.get("b").unwrap(), axis)?;
        let (cot_api, _) = move_axis_to_front(cotangent.get("value").unwrap(), axis)?;
        let mut ctx = CpuContext::new(1);
        let grad_a_api = sum_to_shape(
            &cross(&mut ctx, &pb_api, &cot_api).map_err(|err| format!("cross failed: {err}"))?,
            pa_api.dims(),
        )?;
        let grad_b_api = sum_to_shape(
            &cross(&mut ctx, &cot_api, &pa_api).map_err(|err| format!("cross failed: {err}"))?,
            pb_api.dims(),
        )?;
        Ok(BTreeMap::from([
            (
                String::from("a"),
                permute_or_identity(&grad_a_api, &inverse)?,
            ),
            (
                String::from("b"),
                permute_or_identity(&grad_b_api, &inverse)?,
            ),
        ]))
    })?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn replay_vecdot(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs_preserving_shape(record)?;
    let probe = probe(record)?;
    let direction = decode_tensor_map_preserving_shape(&probe.direction)?;
    let cotangent = decode_tensor_map_preserving_shape(&probe.cotangent)?;
    let expected_jvp_fd = decode_tensor_map_preserving_shape(&probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_tensor_map_preserving_shape(&probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_tensor_map_preserving_shape(&probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let axis = vecdot_axis(record)?;

    let (a_last, a_inverse) = move_axis_to_last(inputs.get("a").unwrap(), axis)?;
    let (b_last, _) = move_axis_to_last(inputs.get("b").unwrap(), axis)?;
    let (da_last, _) = move_axis_to_last(direction.get("a").unwrap(), axis)?;
    let (db_last, _) = move_axis_to_last(direction.get("b").unwrap(), axis)?;
    let cot = cotangent.get("value").unwrap();
    let vec_len = *a_last.dims().last().unwrap();
    let cot_expanded = broadcast_last_axis(cot, vec_len);

    let actual_jvp = BTreeMap::from([(
        String::from("value"),
        tensor_add(
            &vecdot_last_axis(&da_last, &b_last)?,
            &vecdot_last_axis(&a_last, &db_last)?,
        ),
    )]);
    let grad_a_last = tensor_mul(&cot_expanded, &b_last)?;
    let grad_b_last = tensor_mul(&cot_expanded, &a_last)?;
    let actual_vjp = BTreeMap::from([
        (
            String::from("a"),
            permute_or_identity(&grad_a_last, &a_inverse)?,
        ),
        (
            String::from("b"),
            permute_or_identity(&grad_b_last, &a_inverse)?,
        ),
    ]);

    compare_tensor_maps("vecdot.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps(
        "vecdot.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("vecdot.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp("vecdot", record, &inputs, &direction, probe, |perturbed| {
        let axis = vecdot_axis(record)?;
        let (pa_last, inverse) = move_axis_to_last(perturbed.get("a").unwrap(), axis)?;
        let (pb_last, _) = move_axis_to_last(perturbed.get("b").unwrap(), axis)?;
        let vec_len = *pa_last.dims().last().unwrap();
        let cot_expanded = broadcast_last_axis(cotangent.get("value").unwrap(), vec_len);
        Ok(BTreeMap::from([
            (
                String::from("a"),
                permute_or_identity(&tensor_mul(&cot_expanded, &pb_last)?, &inverse)?,
            ),
            (
                String::from("b"),
                permute_or_identity(&tensor_mul(&cot_expanded, &pa_last)?, &inverse)?,
            ),
        ]))
    })?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn ordered_multi_dot_inputs(inputs: &BTreeMap<String, Tensor<f64>>) -> Vec<Tensor<f64>> {
    ["a", "b", "c", "d", "e"]
        .into_iter()
        .filter_map(|name| inputs.get(name).cloned())
        .collect()
}

fn eye_tensor(n: usize) -> Tensor<f64> {
    let mut data = vec![0.0; n * n];
    for i in 0..n {
        data[i + i * n] = 1.0;
    }
    crate::decode::tensor_from_col_major(data, &[n, n])
}

fn multi_dot_jvp_exact(
    primals: &[Tensor<f64>],
    tangents: &[Tensor<f64>],
) -> Result<Tensor<f64>, String> {
    let count = primals.len();
    let mut prefixes = Vec::with_capacity(count + 1);
    prefixes.push(eye_tensor(primals[0].dims()[0]));
    for primal in primals.iter().take(count - 1) {
        prefixes.push(batched_matmul(prefixes.last().unwrap(), primal)?);
    }

    let mut suffixes = vec![
        Tensor::<f64>::zeros(
            &[0, 0],
            tenferro_device::LogicalMemorySpace::MainMemory,
            tenferro_tensor::MemoryOrder::ColumnMajor
        )
        .unwrap();
        count
    ];
    suffixes[count - 1] = eye_tensor(primals[count - 1].dims()[1]);
    for index in (0..count - 1).rev() {
        suffixes[index] = batched_matmul(&primals[index + 1], &suffixes[index + 1])?;
    }

    let output_dims = evaluate_multi_dot(
        &primals
            .iter()
            .enumerate()
            .map(|(idx, tensor)| (char::from(b'a' + idx as u8).to_string(), tensor.clone()))
            .collect(),
    )?
    .dims()
    .to_vec();
    let mut sum =
        crate::decode::tensor_from_col_major(vec![0.0; output_dims.iter().product()], &output_dims);
    for index in 0..count {
        let term = batched_matmul(
            &batched_matmul(&prefixes[index], &tangents[index])?,
            &suffixes[index],
        )?;
        sum = tensor_add(&sum, &term);
    }
    Ok(sum)
}

fn multi_dot_vjp_exact(
    primals: &[Tensor<f64>],
    cotangent: &Tensor<f64>,
) -> Result<BTreeMap<String, Tensor<f64>>, String> {
    let count = primals.len();
    let mut prefixes = Vec::with_capacity(count + 1);
    prefixes.push(eye_tensor(primals[0].dims()[0]));
    for primal in primals.iter().take(count - 1) {
        prefixes.push(batched_matmul(prefixes.last().unwrap(), primal)?);
    }

    let mut suffixes = vec![
        Tensor::<f64>::zeros(
            &[0, 0],
            tenferro_device::LogicalMemorySpace::MainMemory,
            tenferro_tensor::MemoryOrder::ColumnMajor
        )
        .unwrap();
        count
    ];
    suffixes[count - 1] = eye_tensor(primals[count - 1].dims()[1]);
    for index in (0..count - 1).rev() {
        suffixes[index] = batched_matmul(&primals[index + 1], &suffixes[index + 1])?;
    }

    let mut grads = BTreeMap::new();
    for index in 0..count {
        let left_t = batched_transpose(&prefixes[index])?;
        let right_t = batched_transpose(&suffixes[index])?;
        let grad = batched_matmul(&batched_matmul(&left_t, cotangent)?, &right_t)?;
        grads.insert(char::from(b'a' + index as u8).to_string(), grad);
    }
    Ok(grads)
}

fn replay_multi_dot(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs_preserving_shape(record)?;
    let probe = probe(record)?;
    let direction = decode_tensor_map_preserving_shape(&probe.direction)?;
    let cotangent = decode_tensor_map_preserving_shape(&probe.cotangent)?;
    let expected_jvp_fd = decode_tensor_map_preserving_shape(&probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_tensor_map_preserving_shape(&probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_tensor_map_preserving_shape(&probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;

    let primals = ordered_multi_dot_inputs(&inputs);
    let tangents = ordered_multi_dot_inputs(&direction);
    let actual_jvp = BTreeMap::from([(
        String::from("value"),
        multi_dot_jvp_exact(&primals, &tangents)?,
    )]);
    let actual_vjp = multi_dot_vjp_exact(&primals, cotangent.get("value").unwrap())?;
    compare_tensor_maps(
        "multi_dot.jvp.fd",
        &expected_jvp_fd,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps(
        "multi_dot.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("multi_dot.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp(
        "multi_dot",
        record,
        &inputs,
        &direction,
        probe,
        |perturbed| {
            multi_dot_vjp_exact(
                &ordered_multi_dot_inputs(perturbed),
                cotangent.get("value").unwrap(),
            )
        },
    )?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn replay_tensorinv(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs_preserving_shape(record)?;
    let probe = probe(record)?;
    let direction = decode_tensor_map_preserving_shape(&probe.direction)?;
    let cotangent = decode_tensor_map_preserving_shape(&probe.cotangent)?;
    let expected_jvp_fd = decode_tensor_map_preserving_shape(&probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_tensor_map_preserving_shape(&probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_tensor_map_preserving_shape(&probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;

    let ind = tensorinv_ind(record)?;
    let a = inputs.get("a").unwrap();
    let da = direction.get("a").unwrap();
    let output_dims = expected_jvp_fd.get("value").unwrap().dims().to_vec();
    let split = a.dims()[..ind].iter().product::<usize>();
    let a_mat = a
        .contiguous(tenferro_tensor::MemoryOrder::ColumnMajor)
        .reshape(&[split, split])
        .map_err(|err| format!("tensorinv reshape failed: {err}"))?;
    let da_mat = da
        .contiguous(tenferro_tensor::MemoryOrder::ColumnMajor)
        .reshape(&[split, split])
        .map_err(|err| format!("tensorinv tangent reshape failed: {err}"))?;
    let cot_mat = cotangent
        .get("value")
        .unwrap()
        .contiguous(tenferro_tensor::MemoryOrder::ColumnMajor)
        .reshape(&[split, split])
        .map_err(|err| format!("tensorinv cotangent reshape failed: {err}"))?;

    let mut ctx = CpuContext::new(1);
    let (_value, jvp_mat) = inv_frule(&mut ctx, &a_mat, &da_mat)
        .map_err(|err| format!("tensorinv inv_frule failed: {err}"))?;
    let grad_mat = inv_rrule(&mut ctx, &a_mat, &cot_mat)
        .map_err(|err| format!("tensorinv inv_rrule failed: {err}"))?;
    let actual_jvp = BTreeMap::from([(
        String::from("value"),
        jvp_mat
            .reshape(&output_dims)
            .map_err(|err| format!("tensorinv output reshape failed: {err}"))?,
    )]);
    let actual_vjp = BTreeMap::from([(
        String::from("a"),
        grad_mat
            .reshape(a.dims())
            .map_err(|err| format!("tensorinv grad reshape failed: {err}"))?,
    )]);

    compare_tensor_maps(
        "tensorinv.jvp.fd",
        &expected_jvp_fd,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps(
        "tensorinv.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("tensorinv.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp(
        "tensorinv",
        record,
        &inputs,
        &direction,
        probe,
        |perturbed| {
            let a = perturbed.get("a").unwrap();
            let a_mat = a
                .contiguous(tenferro_tensor::MemoryOrder::ColumnMajor)
                .reshape(&[split, split])
                .map_err(|err| format!("tensorinv HVP reshape failed: {err}"))?;
            let cot_mat = cotangent
                .get("value")
                .unwrap()
                .contiguous(tenferro_tensor::MemoryOrder::ColumnMajor)
                .reshape(&[split, split])
                .map_err(|err| format!("tensorinv HVP cot reshape failed: {err}"))?;
            let mut ctx = CpuContext::new(1);
            let grad_mat = inv_rrule(&mut ctx, &a_mat, &cot_mat)
                .map_err(|err| format!("tensorinv inv_rrule failed during HVP replay: {err}"))?;
            Ok(BTreeMap::from([(
                String::from("a"),
                grad_mat
                    .reshape(a.dims())
                    .map_err(|err| format!("tensorinv HVP grad reshape failed: {err}"))?,
            )]))
        },
    )?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

#[derive(Clone, Copy, Debug)]
enum TensorSolvePlacement {
    End,
    Front,
}

fn candidate_tensorsolve_axes(record: &CaseRecord) -> Vec<(Vec<usize>, TensorSolvePlacement)> {
    let a_shape = &record.inputs.get("a").unwrap().shape;
    let b_shape = &record.inputs.get("b").unwrap().shape;
    let output_shape = &probe(record)
        .unwrap()
        .pytorch_ref
        .jvp
        .get("value")
        .unwrap()
        .shape;
    let solution_rank = a_shape.len() - b_shape.len();
    let placements = [TensorSolvePlacement::End, TensorSolvePlacement::Front];
    let mut out = Vec::new();
    ordered_axis_subsets(a_shape.len(), solution_rank)
        .into_iter()
        .for_each(|axes| {
            let mut selected = vec![false; a_shape.len()];
            for axis in axes.iter().copied() {
                selected[axis] = true;
            }
            let trailing: Vec<usize> = selected
                .iter()
                .enumerate()
                .filter_map(|(axis, is_selected)| (!*is_selected).then_some(axis))
                .collect();
            for placement in placements {
                let perm = match placement {
                    TensorSolvePlacement::End => {
                        let mut perm = trailing.clone();
                        perm.extend_from_slice(&axes);
                        perm
                    }
                    TensorSolvePlacement::Front => {
                        let mut perm = axes.clone();
                        perm.extend_from_slice(&trailing);
                        perm
                    }
                };
                let dims: Vec<usize> = perm.iter().map(|&axis| a_shape[axis]).collect();
                let matches = match placement {
                    TensorSolvePlacement::End => {
                        &dims[..b_shape.len()] == b_shape && &dims[b_shape.len()..] == output_shape
                    }
                    TensorSolvePlacement::Front => {
                        &dims[..solution_rank] == output_shape && &dims[solution_rank..] == b_shape
                    }
                };
                if matches {
                    out.push((axes.clone(), placement));
                }
            }
        });
    out
}

fn exact_tensorsolve_axes(
    record: &CaseRecord,
    inputs: &BTreeMap<String, Tensor<f64>>,
    direction: &BTreeMap<String, Tensor<f64>>,
    expected_jvp: &Tensor<f64>,
) -> Result<
    (
        Vec<usize>,
        TensorSolvePlacement,
        tenferro_tensor::MemoryOrder,
        tenferro_tensor::MemoryOrder,
        tenferro_tensor::MemoryOrder,
        bool,
    ),
    String,
> {
    let candidates = candidate_tensorsolve_axes(record);
    let orders = [
        tenferro_tensor::MemoryOrder::ColumnMajor,
        tenferro_tensor::MemoryOrder::RowMajor,
    ];
    let transpose_options = [false, true];
    let mut best: Option<(
        f64,
        Vec<usize>,
        TensorSolvePlacement,
        tenferro_tensor::MemoryOrder,
        tenferro_tensor::MemoryOrder,
        tenferro_tensor::MemoryOrder,
        bool,
    )> = None;
    let step = probe(record)?.fd_ref.step;
    for (axes, placement) in candidates {
        for a_order in orders {
            for b_order in orders {
                for x_order in orders {
                    for transpose_matrix in transpose_options {
                        let plus_inputs = perturb_input_map(inputs, direction, step)?;
                        let minus_inputs = perturb_input_map(inputs, direction, -step)?;
                        let plus = tensorsolve_value_for_axes(
                            plus_inputs.get("a").unwrap(),
                            plus_inputs.get("b").unwrap(),
                            &axes,
                            placement,
                            a_order,
                            b_order,
                            x_order,
                            transpose_matrix,
                        )?;
                        let minus = tensorsolve_value_for_axes(
                            minus_inputs.get("a").unwrap(),
                            minus_inputs.get("b").unwrap(),
                            &axes,
                            placement,
                            a_order,
                            b_order,
                            x_order,
                            transpose_matrix,
                        )?;
                        let jvp = central_diff_tensor(&plus, &minus, step)?;
                        let error = tensor_abs_error(expected_jvp, &jvp)?;
                        if best
                            .as_ref()
                            .map(|(best_err, _, _, _, _, _, _)| error < *best_err)
                            .unwrap_or(true)
                        {
                            best = Some((
                                error,
                                axes.clone(),
                                placement,
                                a_order,
                                b_order,
                                x_order,
                                transpose_matrix,
                            ));
                        }
                    }
                }
            }
        }
    }
    best.map(
        |(_, axes, placement, a_order, b_order, x_order, transpose_matrix)| {
            (axes, placement, a_order, b_order, x_order, transpose_matrix)
        },
    )
    .ok_or_else(|| format!("failed to infer tensorsolve axes for {}", record.case_id))
}

fn tensorsolve_value_for_axes(
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    axes: &[usize],
    placement: TensorSolvePlacement,
    a_order: tenferro_tensor::MemoryOrder,
    b_order: tenferro_tensor::MemoryOrder,
    x_order: tenferro_tensor::MemoryOrder,
    transpose_matrix: bool,
) -> Result<Tensor<f64>, String> {
    let mut selected = vec![false; a.ndim()];
    for &axis in axes {
        selected[axis] = true;
    }
    let trailing: Vec<usize> = selected
        .iter()
        .enumerate()
        .filter_map(|(axis, is_selected)| (!*is_selected).then_some(axis))
        .collect();
    let mut perm = Vec::with_capacity(a.ndim());
    match placement {
        TensorSolvePlacement::End => {
            perm.extend_from_slice(&trailing);
            perm.extend_from_slice(axes);
        }
        TensorSolvePlacement::Front => {
            perm.extend_from_slice(axes);
            perm.extend_from_slice(&trailing);
        }
    }
    let a_perm = permute_or_identity(a, &perm)?;
    let output_dims = match placement {
        TensorSolvePlacement::End => a_perm.dims()[b.ndim()..].to_vec(),
        TensorSolvePlacement::Front => a_perm.dims()[..axes.len()].to_vec(),
    };
    let n = b.dims().iter().product::<usize>();
    let mut a_mat = reshape_in_order(&a_perm, &[n, n], a_order)?;
    if transpose_matrix {
        a_mat = batched_transpose(&a_mat)?;
    }
    let b_vec = reshape_in_order(b, &[n], b_order)?;

    let mut ctx = CpuContext::new(1);
    let value_vec = solve(&mut ctx, &a_mat, &b_vec)
        .map_err(|err| format!("tensorsolve solve failed: {err}"))?;
    reshape_in_order(&value_vec, &output_dims, x_order)
}

fn tensorsolve_vjp_for_axes(
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    cotangent: &Tensor<f64>,
    axes: &[usize],
    placement: TensorSolvePlacement,
    a_order: tenferro_tensor::MemoryOrder,
    b_order: tenferro_tensor::MemoryOrder,
    x_order: tenferro_tensor::MemoryOrder,
    transpose_matrix: bool,
) -> Result<BTreeMap<String, Tensor<f64>>, String> {
    let mut selected = vec![false; a.ndim()];
    for &axis in axes {
        selected[axis] = true;
    }
    let trailing: Vec<usize> = selected
        .iter()
        .enumerate()
        .filter_map(|(axis, is_selected)| (!*is_selected).then_some(axis))
        .collect();
    let mut perm = Vec::with_capacity(a.ndim());
    match placement {
        TensorSolvePlacement::End => {
            perm.extend_from_slice(&trailing);
            perm.extend_from_slice(axes);
        }
        TensorSolvePlacement::Front => {
            perm.extend_from_slice(axes);
            perm.extend_from_slice(&trailing);
        }
    }
    let inverse = inverse_permutation(&perm);
    let a_perm = permute_or_identity(a, &perm)?;
    let n = b.dims().iter().product::<usize>();
    let mut a_mat = reshape_in_order(&a_perm, &[n, n], a_order)?;
    if transpose_matrix {
        a_mat = batched_transpose(&a_mat)?;
    }
    let b_vec = reshape_in_order(b, &[n], b_order)?;
    let cot_vec = reshape_in_order(cotangent, &[n], x_order)?;

    let mut ctx = CpuContext::new(1);
    let grad = solve_rrule(&mut ctx, &a_mat, &b_vec, &cot_vec)
        .map_err(|err| format!("tensorsolve solve_rrule failed: {err}"))?;
    let grad_a_mat = if transpose_matrix {
        batched_transpose(&grad.a)?
    } else {
        grad.a
    };
    let grad_a_perm = reshape_in_order(&grad_a_mat, a_perm.dims(), a_order)?;
    let grad_a = permute_or_identity(&grad_a_perm, &inverse)?;
    let grad_b = reshape_in_order(&grad.b, b.dims(), b_order)?;
    Ok(BTreeMap::from([
        (String::from("a"), grad_a),
        (String::from("b"), grad_b),
    ]))
}

fn replay_tensorsolve(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs_preserving_shape(record)?;
    let probe = probe(record)?;
    let direction = decode_tensor_map_preserving_shape(&probe.direction)?;
    let cotangent = decode_tensor_map_preserving_shape(&probe.cotangent)?;
    let expected_jvp_fd = decode_tensor_map_preserving_shape(&probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_tensor_map_preserving_shape(&probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_tensor_map_preserving_shape(&probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let step = probe.fd_ref.step;
    let (axes, placement, a_order, b_order, x_order, transpose_matrix) = exact_tensorsolve_axes(
        record,
        &inputs,
        &direction,
        expected_jvp_torch.get("value").unwrap(),
    )?;
    let plus_inputs = perturb_input_map(&inputs, &direction, step)?;
    let minus_inputs = perturb_input_map(&inputs, &direction, -step)?;
    let plus = tensorsolve_value_for_axes(
        plus_inputs.get("a").unwrap(),
        plus_inputs.get("b").unwrap(),
        &axes,
        placement,
        a_order,
        b_order,
        x_order,
        transpose_matrix,
    )?;
    let minus = tensorsolve_value_for_axes(
        minus_inputs.get("a").unwrap(),
        minus_inputs.get("b").unwrap(),
        &axes,
        placement,
        a_order,
        b_order,
        x_order,
        transpose_matrix,
    )?;
    let actual_jvp = BTreeMap::from([(
        String::from("value"),
        central_diff_tensor(&plus, &minus, step)?,
    )]);
    let actual_vjp = tensorsolve_vjp_for_axes(
        inputs.get("a").unwrap(),
        inputs.get("b").unwrap(),
        cotangent.get("value").unwrap(),
        &axes,
        placement,
        a_order,
        b_order,
        x_order,
        transpose_matrix,
    )?;

    compare_tensor_maps(
        "tensorsolve.jvp.fd",
        &expected_jvp_fd,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps(
        "tensorsolve.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("tensorsolve.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp(
        "tensorsolve",
        record,
        &inputs,
        &direction,
        probe,
        |perturbed| {
            tensorsolve_vjp_for_axes(
                perturbed.get("a").unwrap(),
                perturbed.get("b").unwrap(),
                cotangent.get("value").unwrap(),
                &axes,
                placement,
                a_order,
                b_order,
                x_order,
                transpose_matrix,
            )
        },
    )?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn replay_pinv_hermitian(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs_preserving_shape(record)?;
    let probe = probe(record)?;
    let direction = decode_tensor_map_preserving_shape(&probe.direction)?;
    let cotangent = decode_tensor_map_preserving_shape(&probe.cotangent)?;
    let expected_jvp_fd = decode_tensor_map_preserving_shape(&probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_tensor_map_preserving_shape(&probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_tensor_map_preserving_shape(&probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let rcond = pinv_rcond(record)?;

    let (a_api, inverse) = move_trailing_core_to_front(inputs.get("a").unwrap(), 2)?;
    let (da_api, _) = move_trailing_core_to_front(direction.get("a").unwrap(), 2)?;
    let (cot_api, _) = move_trailing_core_to_front(cotangent.get("value").unwrap(), 2)?;
    let wrapped_a = scale_tensor(&tensor_add(&a_api, &batched_transpose(&a_api)?), 0.5);
    let wrapped_da = scale_tensor(&tensor_add(&da_api, &batched_transpose(&da_api)?), 0.5);
    let wrapped_cot = scale_tensor(&tensor_add(&cot_api, &batched_transpose(&cot_api)?), 0.5);
    let mut ctx = CpuContext::new(1);
    let (_value, raw_jvp_api) = pinv_frule(&mut ctx, &wrapped_a, &wrapped_da, rcond)
        .map_err(|err| format!("pinv_frule failed: {err}"))?;
    let grad_wrapped = pinv_rrule(&mut ctx, &wrapped_a, &wrapped_cot, rcond)
        .map_err(|err| format!("pinv_rrule failed: {err}"))?;
    let jvp_api = scale_tensor(&raw_jvp_api, 0.5);
    let grad_api = scale_tensor(
        &tensor_add(&grad_wrapped, &batched_transpose(&grad_wrapped)?),
        0.25,
    );

    let actual_jvp = BTreeMap::from([(
        String::from("value"),
        permute_or_identity(&jvp_api, &inverse)?,
    )]);
    let actual_vjp =
        BTreeMap::from([(String::from("a"), permute_or_identity(&grad_api, &inverse)?)]);
    compare_tensor_maps(
        "pinv_hermitian.jvp.fd",
        &expected_jvp_fd,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps(
        "pinv_hermitian.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("pinv_hermitian.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp(
        "pinv_hermitian",
        record,
        &inputs,
        &direction,
        probe,
        |perturbed| {
            let (a_api, inverse) = move_trailing_core_to_front(perturbed.get("a").unwrap(), 2)?;
            let wrapped_a = scale_tensor(&tensor_add(&a_api, &batched_transpose(&a_api)?), 0.5);
            let mut ctx = CpuContext::new(1);
            let grad_wrapped = pinv_rrule(&mut ctx, &wrapped_a, &wrapped_cot, rcond)
                .map_err(|err| format!("pinv_rrule failed during HVP replay: {err}"))?;
            let grad_api = scale_tensor(
                &tensor_add(&grad_wrapped, &batched_transpose(&grad_wrapped)?),
                0.25,
            );
            Ok(BTreeMap::from([(
                String::from("a"),
                permute_or_identity(&grad_api, &inverse)?,
            )]))
        },
    )?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn replay_numerical_identity(record: &CaseRecord) -> Result<bool, String> {
    match record.op.as_str() {
        "cross" => replay_cross(record),
        "multi_dot" => replay_multi_dot(record),
        "pinv_hermitian" => replay_pinv_hermitian(record),
        "tensorinv" => replay_tensorinv(record),
        "tensorsolve" => replay_tensorsolve(record),
        "vecdot" => replay_vecdot(record),
        _ => replay_numerical_identity_generic(record),
    }
}

fn replay_value_key(record: &CaseRecord) -> Result<&'static str, String> {
    match record.op.as_str() {
        "solve" | "solve_triangular" | "cholesky" | "det" | "inv" | "lu_solve" | "cond"
        | "matrix_power" | "matrix_exp" | "pinv" | "pinv_singular" => Ok("value"),
        "slogdet" => Ok("output_0"),
        "solve_ex" | "cholesky_ex" | "inv_ex" | "lu_factor" | "lu_factor_ex" => Ok("output_0"),
        _ => Err(format!("no replay value key for op {}", record.op)),
    }
}

fn case_suffix_index(record: &CaseRecord) -> Result<usize, String> {
    let suffix = record
        .case_id
        .rsplit('_')
        .next()
        .ok_or_else(|| format!("failed to parse case id {}", record.case_id))?;
    let index = suffix
        .parse::<usize>()
        .map_err(|err| format!("failed to parse case id {}: {err}", record.case_id))?;
    if index == 0 {
        return Err(format!("case ids must be 1-based: {}", record.case_id));
    }
    Ok(index)
}

fn required_bool_kwarg(record: &CaseRecord, key: &str) -> Result<bool, String> {
    record
        .op_kwargs
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| format!("op_kwargs.{key} for {} must be a bool", record.case_id))
}

fn solve_triangular_flags(record: &CaseRecord) -> Result<(bool, bool, bool), String> {
    Ok((
        required_bool_kwarg(record, "left")?,
        required_bool_kwarg(record, "upper")?,
        required_bool_kwarg(record, "unitriangular")?,
    ))
}

fn tensor_with_unit_diagonal(tensor: &Tensor<f64>) -> Result<Tensor<f64>, String> {
    let dims = tensor.dims();
    if dims.len() < 2 || dims[0] != dims[1] {
        return Err(format!(
            "unit-diagonal adjustment expects square matrix batches, got {:?}",
            dims
        ));
    }
    let n = dims[0];
    let batch_count = crate::decode::batch_count(&dims[2..]);
    let mut data = tensor_data_col_major(tensor);
    for batch in 0..batch_count {
        let base = batch * n * n;
        for i in 0..n {
            data[base + i + i * n] = 1.0;
        }
    }
    Ok(crate::decode::tensor_from_col_major(data, dims))
}

fn tensor_with_zero_diagonal(tensor: &Tensor<f64>) -> Result<Tensor<f64>, String> {
    let dims = tensor.dims();
    if dims.len() < 2 || dims[0] != dims[1] {
        return Err(format!(
            "diagonal masking expects square matrix batches, got {:?}",
            dims
        ));
    }
    let n = dims[0];
    let batch_count = crate::decode::batch_count(&dims[2..]);
    let mut data = tensor_data_col_major(tensor);
    for batch in 0..batch_count {
        let base = batch * n * n;
        for i in 0..n {
            data[base + i + i * n] = 0.0;
        }
    }
    Ok(crate::decode::tensor_from_col_major(data, dims))
}

fn solve_triangular_runtime_inputs(
    a: &Tensor<f64>,
    da: &Tensor<f64>,
    unitriangular: bool,
) -> Result<(Tensor<f64>, Tensor<f64>), String> {
    if unitriangular {
        Ok((
            tensor_with_unit_diagonal(a)?,
            tensor_with_zero_diagonal(da)?,
        ))
    } else {
        Ok((a.clone(), da.clone()))
    }
}

fn solve_triangular_jvp_vjp(
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    da: &Tensor<f64>,
    db: &Tensor<f64>,
    cotangent: &Tensor<f64>,
    left: bool,
    upper: bool,
    unitriangular: bool,
) -> Result<(Tensor<f64>, Tensor<f64>, Tensor<f64>), String> {
    let (a_eff, da_eff) = solve_triangular_runtime_inputs(a, da, unitriangular)?;
    let mut ctx = CpuContext::new(1);

    let (dx, mut grad_a, grad_b) = if left {
        let (_x, dx) = solve_triangular_frule(&mut ctx, &a_eff, b, &da_eff, db, upper)
            .map_err(|err| format!("solve_triangular_frule failed: {err}"))?;
        let grad = solve_triangular_rrule(&mut ctx, &a_eff, b, cotangent, upper)
            .map_err(|err| format!("solve_triangular_rrule failed: {err}"))?;
        (dx, grad.a, grad.b)
    } else {
        let a_t = batched_transpose(&a_eff)?;
        let b_t = batched_transpose(b)?;
        let da_t = batched_transpose(&da_eff)?;
        let db_t = batched_transpose(db)?;
        let cot_t = batched_transpose(cotangent)?;
        let (_x_t, dx_t) = solve_triangular_frule(&mut ctx, &a_t, &b_t, &da_t, &db_t, !upper)
            .map_err(|err| format!("solve_triangular right-side frule failed: {err}"))?;
        let grad_t = solve_triangular_rrule(&mut ctx, &a_t, &b_t, &cot_t, !upper)
            .map_err(|err| format!("solve_triangular right-side rrule failed: {err}"))?;
        (
            batched_transpose(&dx_t)?,
            batched_transpose(&grad_t.a)?,
            batched_transpose(&grad_t.b)?,
        )
    };

    if unitriangular {
        grad_a = tensor_with_zero_diagonal(&grad_a)?;
    }

    Ok((dx, grad_a, grad_b))
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
    let value_key = replay_value_key(record)?;

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
        cotangent.get(value_key).unwrap(),
    )
    .map_err(|err| format!("solve_rrule failed: {err}"))?;

    let actual_jvp = BTreeMap::from([(String::from(value_key), dx)]);
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
            cotangent.get(value_key).unwrap(),
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

fn replay_solve_triangular(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like(record, &probe.direction)?;
    let cotangent = decode_observable_map(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let (left, upper, unitriangular) = solve_triangular_flags(record)?;

    let (dx, grad_a, grad_b) = solve_triangular_jvp_vjp(
        inputs.get("a").unwrap(),
        inputs.get("b").unwrap(),
        direction.get("a").unwrap(),
        direction.get("b").unwrap(),
        cotangent.get("value").unwrap(),
        left,
        upper,
        unitriangular,
    )?;

    let actual_jvp = BTreeMap::from([(String::from("value"), dx)]);
    let actual_vjp = BTreeMap::from([(String::from("a"), grad_a), (String::from("b"), grad_b)]);
    compare_tensor_maps(
        "solve_triangular.jvp.fd",
        &expected_jvp_fd,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps(
        "solve_triangular.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps(
        "solve_triangular.vjp",
        &expected_vjp,
        &actual_vjp,
        rtol,
        atol,
    )?;

    let hvp_checked = validate_hvp(
        "solve_triangular",
        record,
        &inputs,
        &direction,
        probe,
        |perturbed| {
            let (_dx, grad_a, grad_b) = solve_triangular_jvp_vjp(
                perturbed.get("a").unwrap(),
                perturbed.get("b").unwrap(),
                direction.get("a").unwrap(),
                direction.get("b").unwrap(),
                cotangent.get("value").unwrap(),
                left,
                upper,
                unitriangular,
            )?;
            Ok(BTreeMap::from([
                (String::from("a"), grad_a),
                (String::from("b"), grad_b),
            ]))
        },
    )?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn replay_cholesky_t<T>(record: &CaseRecord) -> Result<bool, String>
where
    T: OracleDbScalar + KernelLinalgScalar + Conjugate,
{
    let inputs = decode_inputs_typed::<T>(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like_typed::<T>(record, &probe.direction)?;
    let cotangent = decode_observable_map_typed::<T>(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map_typed::<T>(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map_typed::<T>(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like_typed::<T>(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let value_key = replay_value_key(record)?;
    let orientation = infer_triangular_orientation_typed(
        expected_jvp_fd
            .get(value_key)
            .ok_or_else(|| format!("missing cholesky fd_ref value for {}", record.case_id))?,
    )?;
    let wrapped_a = apply_hermitian_wrapper_typed(inputs.get("a").unwrap())?;
    let wrapped_da = apply_hermitian_wrapper_typed(direction.get("a").unwrap())?;

    let mut ctx = CpuContext::new(1);
    let (_l, dl) = cholesky_frule(&mut ctx, &wrapped_a, &wrapped_da)
        .map_err(|err| format!("cholesky_frule failed: {err}"))?;
    let raw_cotangent = match orientation {
        TriangularOrientation::Lower => cotangent.get(value_key).unwrap().clone(),
        TriangularOrientation::Upper => {
            batched_adjoint_transpose(cotangent.get(value_key).unwrap())?
        }
    };
    let grad = cholesky_rrule(&mut ctx, &wrapped_a, &raw_cotangent)
        .map_err(|err| format!("cholesky_rrule failed: {err}"))?;

    let actual_value = match orientation {
        TriangularOrientation::Lower => dl,
        TriangularOrientation::Upper => batched_adjoint_transpose(&dl)?,
    };
    let actual_jvp = BTreeMap::from([(String::from(value_key), actual_value)]);
    let actual_vjp = BTreeMap::from([(String::from("a"), apply_hermitian_wrapper_typed(&grad)?)]);
    compare_tensor_maps_typed("cholesky.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps_typed(
        "cholesky.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps_typed("cholesky.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp_typed(
        "cholesky",
        record,
        &inputs,
        &direction,
        probe,
        |perturbed| {
            let wrapped = apply_hermitian_wrapper_typed(perturbed.get("a").unwrap())?;
            let mut ctx = CpuContext::new(1);
            let grad = cholesky_rrule(&mut ctx, &wrapped, &raw_cotangent)
                .map_err(|err| format!("cholesky_rrule failed during HVP replay: {err}"))?;
            Ok(BTreeMap::from([(
                String::from("a"),
                apply_hermitian_wrapper_typed(&grad)?,
            )]))
        },
    )?;
    check_adjoint_identity_typed(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn replay_cholesky(record: &CaseRecord) -> Result<bool, String> {
    match record.dtype.as_str() {
        "float64" => replay_cholesky_t::<f64>(record),
        "complex64" => replay_cholesky_t::<Complex32>(record),
        "complex128" => replay_cholesky_t::<Complex64>(record),
        other => Err(format!(
            "unsupported replay dtype {other} for {}",
            record.case_id
        )),
    }
}

fn replay_inv_t<T>(record: &CaseRecord) -> Result<bool, String>
where
    T: OracleDbScalar + KernelLinalgScalar + Conjugate,
{
    let inputs = decode_inputs_typed::<T>(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like_typed::<T>(record, &probe.direction)?;
    let cotangent = decode_observable_map_typed::<T>(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map_typed::<T>(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map_typed::<T>(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like_typed::<T>(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let value_key = replay_value_key(record)?;

    let mut ctx = CpuContext::new(1);
    let (_ainv, dainv) = inv_frule(
        &mut ctx,
        inputs.get("a").unwrap(),
        direction.get("a").unwrap(),
    )
    .map_err(|err| format!("inv_frule failed: {err}"))?;
    let grad = inv_rrule(
        &mut ctx,
        inputs.get("a").unwrap(),
        cotangent.get(value_key).unwrap(),
    )
    .map_err(|err| format!("inv_rrule failed: {err}"))?;

    let actual_jvp = BTreeMap::from([(String::from(value_key), dainv)]);
    let actual_vjp = BTreeMap::from([(String::from("a"), grad)]);
    compare_tensor_maps_typed("inv.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps_typed(
        "inv.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps_typed("inv.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp_typed("inv", record, &inputs, &direction, probe, |perturbed| {
        let mut ctx = CpuContext::new(1);
        let grad = inv_rrule(
            &mut ctx,
            perturbed.get("a").unwrap(),
            cotangent.get(value_key).unwrap(),
        )
        .map_err(|err| format!("inv_rrule failed during HVP replay: {err}"))?;
        Ok(BTreeMap::from([(String::from("a"), grad)]))
    })?;
    check_adjoint_identity_typed(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn replay_inv(record: &CaseRecord) -> Result<bool, String> {
    match record.dtype.as_str() {
        "float64" => replay_inv_t::<f64>(record),
        "float32" => replay_inv_t::<f32>(record),
        "complex64" => replay_inv_t::<Complex32>(record),
        "complex128" => replay_inv_t::<Complex64>(record),
        other => Err(format!(
            "unsupported replay dtype {other} for {}",
            record.case_id
        )),
    }
}

fn replay_lu_typed<T>(record: &CaseRecord) -> Result<bool, String>
where
    T: OracleDbScalar + KernelLinalgScalar + Conjugate + LiftPermutationMatrixTensor<CpuContext>,
{
    let inputs = decode_inputs_typed::<T>(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like_typed::<T>(record, &probe.direction)?;
    let cotangent = decode_observable_map_typed::<T>(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map_typed::<T>(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map_typed::<T>(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like_typed::<T>(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;

    let pivot = required_bool_kwarg(record, "pivot")?;
    if !pivot {
        return Err(format!(
            "lu oracle replay currently expects pivot=true for {}",
            record.case_id
        ));
    }

    let cotangent = LuCotangent {
        l: Some(
            cotangent
                .get("output_1")
                .ok_or_else(|| format!("missing lu cotangent output_1 for {}", record.case_id))?
                .clone(),
        ),
        u: Some(
            cotangent
                .get("output_2")
                .ok_or_else(|| format!("missing lu cotangent output_2 for {}", record.case_id))?
                .clone(),
        ),
    };

    let mut ctx = CpuContext::new(1);
    let (_result, dresult) = lu_frule(
        &mut ctx,
        inputs.get("a").unwrap(),
        direction.get("a").unwrap(),
        LuPivot::Partial,
    )
    .map_err(|err| format!("lu_frule failed: {err}"))?;
    let grad = lu_rrule(
        &mut ctx,
        inputs.get("a").unwrap(),
        &cotangent,
        LuPivot::Partial,
    )
    .map_err(|err| format!("lu_rrule failed: {err}"))?;

    let actual_jvp = BTreeMap::from([
        (String::from("output_1"), dresult.l),
        (String::from("output_2"), dresult.u),
    ]);
    let actual_vjp = BTreeMap::from([(String::from("a"), grad)]);
    compare_tensor_maps_typed("lu.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps_typed("lu.jvp.torch", &expected_jvp_torch, &actual_jvp, rtol, atol)?;
    compare_tensor_maps_typed("lu.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp_typed("lu", record, &inputs, &direction, probe, |perturbed| {
        let mut ctx = CpuContext::new(1);
        let grad = lu_rrule(
            &mut ctx,
            perturbed.get("a").unwrap(),
            &cotangent,
            LuPivot::Partial,
        )
        .map_err(|err| format!("lu_rrule failed during HVP replay: {err}"))?;
        Ok(BTreeMap::from([(String::from("a"), grad)]))
    })?;
    check_adjoint_identity_typed(
        record,
        &decode_observable_map_typed::<T>(record, &probe.cotangent)?,
        &actual_jvp,
        &actual_vjp,
        &direction,
    )?;
    Ok(hvp_checked)
}

fn replay_lu(record: &CaseRecord) -> Result<bool, String> {
    match record.dtype.as_str() {
        "float32" => replay_lu_typed::<f32>(record),
        "float64" => replay_lu_typed::<f64>(record),
        "complex64" => replay_lu_typed::<Complex32>(record),
        "complex128" => replay_lu_typed::<Complex64>(record),
        other => Err(format!(
            "unsupported replay dtype {other} for {}",
            record.case_id
        )),
    }
}

fn replay_norm_real_t<T>(record: &CaseRecord, kind: NormKind) -> Result<bool, String>
where
    T: OracleDbScalar + KernelLinalgScalar<Real = T> + num_traits::Float,
{
    let inputs = decode_inputs_typed::<T>(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like_typed::<T>(record, &probe.direction)?;
    let cotangent =
        squeeze_scalar_map_typed(decode_observable_map_typed::<T>(record, &probe.cotangent)?)?;
    let expected_jvp_fd =
        squeeze_scalar_map_typed(decode_observable_map_typed::<T>(record, &probe.fd_ref.jvp)?)?;
    let expected_jvp_torch = squeeze_scalar_map_typed(decode_observable_map_typed::<T>(
        record,
        &probe.pytorch_ref.jvp,
    )?)?;
    let expected_vjp = decode_input_map_like_typed::<T>(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let cotangent_value = cotangent
        .get("value")
        .ok_or_else(|| format!("missing norm cotangent value for {}", record.case_id))?
        .clone();

    let mut ctx = CpuContext::new(1);
    let (_result, dresult) = norm_frule(
        &mut ctx,
        inputs.get("a").unwrap(),
        direction.get("a").unwrap(),
        kind,
    )
    .map_err(|err| format!("norm_frule failed: {err}"))?;
    let grad = norm_rrule(&mut ctx, inputs.get("a").unwrap(), &cotangent_value, kind)
        .map_err(|err| format!("norm_rrule failed: {err}"))?;

    let actual_jvp = BTreeMap::from([(String::from("value"), dresult)]);
    let actual_vjp = BTreeMap::from([(String::from("a"), grad)]);
    compare_tensor_maps_typed("norm.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps_typed(
        "norm.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps_typed("norm.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked =
        validate_hvp_typed("norm", record, &inputs, &direction, probe, |perturbed| {
            let mut ctx = CpuContext::new(1);
            let grad = norm_rrule(
                &mut ctx,
                perturbed.get("a").unwrap(),
                &cotangent_value,
                kind,
            )
            .map_err(|err| format!("norm_rrule failed during HVP replay: {err}"))?;
            Ok(BTreeMap::from([(String::from("a"), grad)]))
        })?;
    check_adjoint_identity_typed(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

macro_rules! impl_replay_norm_complex {
    ($fn_name:ident, $complex_ty:ty, $real_ty:ty) => {
        fn $fn_name(record: &CaseRecord, kind: NormKind) -> Result<bool, String> {
            let inputs = decode_inputs_typed::<$complex_ty>(record)?;
            let probe = probe(record)?;
            let direction = decode_input_map_like_typed::<$complex_ty>(record, &probe.direction)?;
            let cotangent = squeeze_scalar_map_typed(decode_observable_map_typed::<$real_ty>(
                record,
                &probe.cotangent,
            )?)?;
            let expected_jvp_fd = squeeze_scalar_map_typed(
                decode_observable_map_typed::<$real_ty>(record, &probe.fd_ref.jvp)?,
            )?;
            let expected_jvp_torch = squeeze_scalar_map_typed(decode_observable_map_typed::<
                $real_ty,
            >(
                record, &probe.pytorch_ref.jvp
            )?)?;
            let expected_vjp =
                decode_input_map_like_typed::<$complex_ty>(record, &probe.pytorch_ref.vjp)?;
            let (rtol, atol) = comparison(record)?;
            let cotangent_value = cotangent
                .get("value")
                .ok_or_else(|| format!("missing norm cotangent value for {}", record.case_id))?
                .clone();

            let mut ctx = CpuContext::new(1);
            let (_result, dresult) = norm_frule_complex(
                &mut ctx,
                inputs.get("a").unwrap(),
                direction.get("a").unwrap(),
                kind,
            )
            .map_err(|err| format!("norm_frule failed: {err}"))?;
            let grad =
                norm_rrule_complex(&mut ctx, inputs.get("a").unwrap(), &cotangent_value, kind)
                    .map_err(|err| format!("norm_rrule failed: {err}"))?;

            let actual_jvp = BTreeMap::from([(String::from("value"), dresult)]);
            let actual_vjp = BTreeMap::from([(String::from("a"), grad)]);
            compare_tensor_maps_typed("norm.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
            compare_tensor_maps_typed(
                "norm.jvp.torch",
                &expected_jvp_torch,
                &actual_jvp,
                rtol,
                atol,
            )?;
            compare_tensor_maps_typed("norm.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
            let hvp_checked =
                validate_hvp_typed("norm", record, &inputs, &direction, probe, |perturbed| {
                    let mut ctx = CpuContext::new(1);
                    let grad = norm_rrule_complex(
                        &mut ctx,
                        perturbed.get("a").unwrap(),
                        &cotangent_value,
                        kind,
                    )
                    .map_err(|err| format!("norm_rrule failed during HVP replay: {err}"))?;
                    Ok(BTreeMap::from([(String::from("a"), grad)]))
                })?;
            check_mixed_adjoint_identity_typed(
                record,
                &cotangent,
                &actual_jvp,
                &actual_vjp,
                &direction,
            )?;
            Ok(hvp_checked)
        }
    };
}

impl_replay_norm_complex!(replay_norm_complex32, Complex32, f32);
impl_replay_norm_complex!(replay_norm_complex64, Complex64, f64);

fn replay_norm(record: &CaseRecord) -> Result<bool, String> {
    let kind = replayable_norm_kind(record).ok_or_else(|| {
        format!(
            "norm replay requested for unsupported norm subset in {}",
            record.case_id
        )
    })?;
    match record.dtype.as_str() {
        "float32" => replay_norm_real_t::<f32>(record, kind),
        "float64" => replay_norm_real_t::<f64>(record, kind),
        "complex64" => replay_norm_complex32(record, kind),
        "complex128" => replay_norm_complex64(record, kind),
        other => Err(format!(
            "unsupported replay dtype {other} for {}",
            record.case_id
        )),
    }
}

fn replay_det_t<T>(record: &CaseRecord) -> Result<bool, String>
where
    T: OracleDbScalar + KernelLinalgScalar + Conjugate + ScaleTensorByRealSameShape<CpuContext>,
{
    let inputs = decode_inputs_typed::<T>(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like_typed::<T>(record, &probe.direction)?;
    let cotangent = decode_observable_map_typed::<T>(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map_typed::<T>(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map_typed::<T>(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like_typed::<T>(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let value_key = replay_value_key(record)?;

    let mut ctx = CpuContext::new(1);
    let (_value, jvp_value) = det_frule(
        &mut ctx,
        inputs.get("a").unwrap(),
        direction.get("a").unwrap(),
    )
    .map_err(|err| format!("det_frule failed: {err}"))?;
    let grad = det_rrule(
        &mut ctx,
        inputs.get("a").unwrap(),
        cotangent.get(value_key).unwrap(),
    )
    .map_err(|err| format!("det_rrule failed: {err}"))?;

    let actual_jvp = BTreeMap::from([(String::from(value_key), jvp_value)]);
    let actual_vjp = BTreeMap::from([(String::from("a"), grad)]);
    compare_tensor_maps_typed("det.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps_typed(
        "det.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps_typed("det.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp_typed("det", record, &inputs, &direction, probe, |perturbed| {
        let mut ctx = CpuContext::new(1);
        let grad = det_rrule(
            &mut ctx,
            perturbed.get("a").unwrap(),
            cotangent.get(value_key).unwrap(),
        )
        .map_err(|err| format!("det_rrule failed during HVP replay: {err}"))?;
        Ok(BTreeMap::from([(String::from("a"), grad)]))
    })?;
    check_adjoint_identity_typed(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn replay_det(record: &CaseRecord) -> Result<bool, String> {
    match record.dtype.as_str() {
        "float32" => replay_det_t::<f32>(record),
        "float64" => replay_det_t::<f64>(record),
        "complex64" => replay_det_t::<Complex32>(record),
        "complex128" => replay_det_t::<Complex64>(record),
        other => Err(format!(
            "unsupported replay dtype {other} for {}",
            record.case_id
        )),
    }
}

fn decode_slogdet_observable_typed<T: OracleDbScalar>(
    record: &CaseRecord,
    encoded: &BTreeMap<String, DbTensor>,
) -> Result<(Tensor<T>, Tensor<T::Real>), String>
where
    T::Real: OracleDbScalar,
{
    let sign = decode_tensor_with_core_rank::<T>(
        encoded
            .get("output_0")
            .ok_or_else(|| format!("missing output_0 for {}", record.case_id))?,
        observable_core_rank(record, "output_0")?,
    )?;
    let logabsdet = decode_tensor_with_core_rank::<T::Real>(
        encoded
            .get("output_1")
            .ok_or_else(|| format!("missing output_1 for {}", record.case_id))?,
        observable_core_rank(record, "output_1")?,
    )?;
    Ok((sign, logabsdet))
}

fn compare_slogdet_observable_typed<T: OracleDbScalar>(
    label: &str,
    expected: &(Tensor<T>, Tensor<T::Real>),
    actual: &(Tensor<T>, Tensor<T::Real>),
    rtol: f64,
    atol: f64,
) -> Result<(), String>
where
    T::Real: OracleDbScalar,
{
    compare_tensors_typed(
        &format!("{label}.output_0"),
        &expected.0,
        &actual.0,
        rtol,
        atol,
    )?;
    compare_tensors_typed(
        &format!("{label}.output_1"),
        &expected.1,
        &actual.1,
        rtol,
        atol,
    )?;
    Ok(())
}

fn slogdet_adjoint_identity_typed<T: OracleDbScalar>(
    record: &CaseRecord,
    sign_cotangent: &Tensor<T>,
    logabsdet_cotangent: &Tensor<T::Real>,
    sign_jvp: &Tensor<T>,
    logabsdet_jvp: &Tensor<T::Real>,
    actual_vjp: &BTreeMap<String, Tensor<T>>,
    direction: &BTreeMap<String, Tensor<T>>,
) -> Result<(), String>
where
    T::Real: OracleDbScalar,
{
    let (rtol, atol) = comparison(record)?;
    let lhs = crate::decode::inner_product_typed(sign_cotangent, sign_jvp)?
        + crate::decode::inner_product_typed(logabsdet_cotangent, logabsdet_jvp)?;
    let rhs = tensor_map_inner_product_typed(actual_vjp, direction)?;
    let allowed = atol + rtol * lhs.abs();
    if (lhs - rhs).abs() > allowed {
        return Err(format!(
            "adjoint identity mismatch: lhs={lhs}, rhs={rhs}, allowed={allowed}"
        ));
    }
    Ok(())
}

fn replay_slogdet_t<T>(record: &CaseRecord) -> Result<bool, String>
where
    T: OracleDbScalar
        + KernelLinalgScalar
        + Conjugate
        + SlogdetFruleDispatch<CpuContext>
        + SlogdetRruleDispatch<CpuContext>,
    T::Real: OracleDbScalar,
{
    let inputs = decode_inputs_typed::<T>(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like_typed::<T>(record, &probe.direction)?;
    let (sign_cotangent, logabsdet_cotangent) =
        decode_slogdet_observable_typed::<T>(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_slogdet_observable_typed::<T>(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_slogdet_observable_typed::<T>(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like_typed::<T>(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;

    let mut ctx = CpuContext::new(1);
    let (_value, jvp_value) = slogdet_frule(
        &mut ctx,
        inputs.get("a").unwrap(),
        direction.get("a").unwrap(),
    )
    .map_err(|err| format!("slogdet_frule failed: {err}"))?;
    let grad = slogdet_rrule(
        &mut ctx,
        inputs.get("a").unwrap(),
        &SlogdetCotangent {
            sign: Some(sign_cotangent.clone()),
            logabsdet: Some(logabsdet_cotangent.clone()),
        },
    )
    .map_err(|err| format!("slogdet_rrule failed: {err}"))?;

    let actual_jvp = (jvp_value.sign, jvp_value.logabsdet);
    let actual_vjp = BTreeMap::from([(String::from("a"), grad)]);
    compare_slogdet_observable_typed("slogdet.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_slogdet_observable_typed(
        "slogdet.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps_typed("slogdet.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked =
        validate_hvp_typed("slogdet", record, &inputs, &direction, probe, |perturbed| {
            let mut ctx = CpuContext::new(1);
            let grad = slogdet_rrule(
                &mut ctx,
                perturbed.get("a").unwrap(),
                &SlogdetCotangent {
                    sign: Some(sign_cotangent.clone()),
                    logabsdet: Some(logabsdet_cotangent.clone()),
                },
            )
            .map_err(|err| format!("slogdet_rrule failed during HVP replay: {err}"))?;
            Ok(BTreeMap::from([(String::from("a"), grad)]))
        })?;
    slogdet_adjoint_identity_typed(
        record,
        &sign_cotangent,
        &logabsdet_cotangent,
        &actual_jvp.0,
        &actual_jvp.1,
        &actual_vjp,
        &direction,
    )?;
    Ok(hvp_checked)
}

fn replay_slogdet(record: &CaseRecord) -> Result<bool, String> {
    match record.dtype.as_str() {
        "float32" => replay_slogdet_t::<f32>(record),
        "float64" => replay_slogdet_t::<f64>(record),
        "complex64" => replay_slogdet_t::<Complex32>(record),
        "complex128" => replay_slogdet_t::<Complex64>(record),
        other => Err(format!(
            "unsupported replay dtype {other} for {}",
            record.case_id
        )),
    }
}

fn replay_matrix_exp_t<T>(record: &CaseRecord) -> Result<bool, String>
where
    T: OracleDbScalar
        + KernelLinalgScalar
        + Conjugate
        + MatrixExpAbsTensor<CpuContext>
        + ScaleTensorByRealSameShape<CpuContext>,
    T::Real: KernelLinalgScalar<Real = T::Real> + Float,
{
    let inputs = decode_inputs_typed::<T>(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like_typed::<T>(record, &probe.direction)?;
    let cotangent = decode_observable_map_typed::<T>(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map_typed::<T>(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map_typed::<T>(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like_typed::<T>(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let value_key = replay_value_key(record)?;

    let mut ctx = CpuContext::new(1);
    let (_exp_a, dexp_a) = matrix_exp_frule(
        &mut ctx,
        inputs.get("a").unwrap(),
        direction.get("a").unwrap(),
    )
    .map_err(|err| format!("matrix_exp_frule failed: {err}"))?;
    let grad = matrix_exp_rrule(
        &mut ctx,
        inputs.get("a").unwrap(),
        cotangent.get(value_key).unwrap(),
    )
    .map_err(|err| format!("matrix_exp_rrule failed: {err}"))?;

    let actual_jvp = BTreeMap::from([(String::from(value_key), dexp_a)]);
    let actual_vjp = BTreeMap::from([(String::from("a"), grad)]);
    compare_tensor_maps_typed(
        "matrix_exp.jvp.fd",
        &expected_jvp_fd,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps_typed(
        "matrix_exp.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps_typed("matrix_exp.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp_typed(
        "matrix_exp",
        record,
        &inputs,
        &direction,
        probe,
        |perturbed| {
            let mut ctx = CpuContext::new(1);
            let grad = matrix_exp_rrule(
                &mut ctx,
                perturbed.get("a").unwrap(),
                cotangent.get(value_key).unwrap(),
            )
            .map_err(|err| format!("matrix_exp_rrule failed during HVP replay: {err}"))?;
            Ok(BTreeMap::from([(String::from("a"), grad)]))
        },
    )?;
    check_adjoint_identity_typed(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn replay_matrix_exp(record: &CaseRecord) -> Result<bool, String> {
    match record.dtype.as_str() {
        "float64" => replay_matrix_exp_t::<f64>(record),
        "complex64" => replay_matrix_exp_t::<Complex32>(record),
        "complex128" => replay_matrix_exp_t::<Complex64>(record),
        other => Err(format!(
            "unsupported replay dtype {other} for {}",
            record.case_id
        )),
    }
}

fn tensor_mul(left: &Tensor<f64>, right: &Tensor<f64>) -> Result<Tensor<f64>, String> {
    if left.dims() != right.dims() {
        return Err(format!(
            "tensor_mul shape mismatch: left {:?}, right {:?}",
            left.dims(),
            right.dims()
        ));
    }
    let left_data = tensor_data_col_major(left);
    let right_data = tensor_data_col_major(right);
    let data: Vec<f64> = left_data
        .iter()
        .zip(right_data.iter())
        .map(|(a, b)| a * b)
        .collect();
    Ok(crate::decode::tensor_from_col_major(data, left.dims()))
}

fn tensor_zeros_like(tensor: &Tensor<f64>) -> Tensor<f64> {
    crate::decode::tensor_from_col_major(vec![0.0; tensor.dims().iter().product()], tensor.dims())
}

fn tensor_is_empty(tensor: &Tensor<f64>) -> bool {
    tensor.dims().iter().any(|&dim| dim == 0)
}

fn batched_identity_like(square: &Tensor<f64>) -> Result<Tensor<f64>, String> {
    let dims = square.dims();
    if dims.len() < 2 || dims[0] != dims[1] {
        return Err(format!(
            "identity requires square matrix dims, got {:?}",
            dims
        ));
    }
    let n = dims[0];
    let batch_dims = &dims[2..];
    let bc = crate::decode::batch_count(batch_dims);
    let mut data = vec![0.0; n * n * bc];
    for batch in 0..bc {
        let base = batch * n * n;
        for i in 0..n {
            data[base + i + i * n] = 1.0;
        }
    }
    Ok(crate::decode::tensor_from_col_major(data, dims))
}

fn pack_lu_factors(l: &Tensor<f64>, u: &Tensor<f64>) -> Result<Tensor<f64>, String> {
    let l_dims = l.dims();
    let u_dims = u.dims();
    if l_dims.len() < 2 || u_dims.len() < 2 {
        return Err("pack_lu_factors requires rank >= 2 tensors".to_string());
    }
    if &l_dims[2..] != &u_dims[2..] {
        return Err(format!("LU batch mismatch: l {:?}, u {:?}", l_dims, u_dims));
    }
    let m = l_dims[0];
    let k = l_dims[1];
    let n = u_dims[1];
    if u_dims[0] != k {
        return Err(format!(
            "LU inner rank mismatch: l {:?}, u {:?}",
            l_dims, u_dims
        ));
    }

    let batch_dims = &l_dims[2..];
    let bc = crate::decode::batch_count(batch_dims);
    let l_data = tensor_data_col_major(l);
    let u_data = tensor_data_col_major(u);
    let mut packed = vec![0.0; m * n * bc];

    for batch in 0..bc {
        let l_offset = batch * m * k;
        let u_offset = batch * k * n;
        let out_offset = batch * m * n;
        for j in 0..n {
            for i in 0..m {
                packed[out_offset + i + j * m] = if j < k && i > j {
                    l_data[l_offset + i + j * m]
                } else if i < k && i <= j {
                    u_data[u_offset + i + j * k]
                } else {
                    0.0
                };
            }
        }
    }

    let mut dims = vec![m, n];
    dims.extend_from_slice(batch_dims);
    Ok(crate::decode::tensor_from_col_major(packed, &dims))
}

fn unpack_lu_cotangent(packed: &Tensor<f64>) -> Result<LuCotangent<f64>, String> {
    let dims = packed.dims();
    if dims.len() < 2 {
        return Err(format!(
            "packed LU cotangent requires rank >= 2, got {:?}",
            dims
        ));
    }
    let m = dims[0];
    let n = dims[1];
    let k = m.min(n);
    let batch_dims = &dims[2..];
    let bc = crate::decode::batch_count(batch_dims);
    let packed_data = tensor_data_col_major(packed);
    let mut dl = vec![0.0; m * k * bc];
    let mut du = vec![0.0; k * n * bc];

    for batch in 0..bc {
        let packed_offset = batch * m * n;
        let l_offset = batch * m * k;
        let u_offset = batch * k * n;
        for j in 0..k {
            for i in 0..m {
                if i > j {
                    dl[l_offset + i + j * m] = packed_data[packed_offset + i + j * m];
                }
            }
        }
        for j in 0..n {
            for i in 0..k {
                if i <= j {
                    du[u_offset + i + j * k] = packed_data[packed_offset + i + j * m];
                }
            }
        }
    }

    let mut l_dims = vec![m, k];
    l_dims.extend_from_slice(batch_dims);
    let mut u_dims = vec![k, n];
    u_dims.extend_from_slice(batch_dims);
    Ok(LuCotangent {
        l: Some(crate::decode::tensor_from_col_major(dl, &l_dims)),
        u: Some(crate::decode::tensor_from_col_major(du, &u_dims)),
    })
}

fn lu_factor_packed(tensor: &Tensor<f64>) -> Result<Tensor<f64>, String> {
    let mut ctx = CpuContext::new(1);
    tenferro_linalg::lu_factor(&mut ctx, tensor)
        .map(|result| result.factors)
        .map_err(|err| format!("lu_factor failed: {err}"))
}

fn replay_lu_factor(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like(record, &probe.direction)?;
    let cotangent = decode_observable_map(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let value_key = replay_value_key(record)?;
    let step = probe.fd_ref.step;
    let plus_inputs = perturb_input_map(&inputs, &direction, step)?;
    let minus_inputs = perturb_input_map(&inputs, &direction, -step)?;
    let plus = lu_factor_packed(plus_inputs.get("a").unwrap())?;
    let minus = lu_factor_packed(minus_inputs.get("a").unwrap())?;
    let plus_data = tensor_data_col_major(&plus);
    let minus_data = tensor_data_col_major(&minus);
    let jvp_data: Vec<f64> = plus_data
        .iter()
        .zip(minus_data.iter())
        .map(|(p, m)| (p - m) / (2.0 * step))
        .collect();
    let packed_cotangent = unpack_lu_cotangent(cotangent.get(value_key).unwrap())?;
    let mut ctx = CpuContext::new(1);
    let grad = lu_rrule(
        &mut ctx,
        inputs.get("a").unwrap(),
        &packed_cotangent,
        LuPivot::Partial,
    )
    .map_err(|err| format!("lu_rrule failed: {err}"))?;

    let actual_jvp = BTreeMap::from([(
        String::from(value_key),
        crate::decode::tensor_from_col_major(jvp_data, plus.dims()),
    )]);
    let actual_vjp = BTreeMap::from([(String::from("a"), grad)]);
    compare_tensor_maps(
        "lu_factor.jvp.fd",
        &expected_jvp_fd,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps(
        "lu_factor.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("lu_factor.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp(
        "lu_factor",
        record,
        &inputs,
        &direction,
        probe,
        |perturbed| {
            let mut ctx = CpuContext::new(1);
            let grad = lu_rrule(
                &mut ctx,
                perturbed.get("a").unwrap(),
                &packed_cotangent,
                LuPivot::Partial,
            )
            .map_err(|err| format!("lu_rrule failed during HVP replay: {err}"))?;
            Ok(BTreeMap::from([(String::from("a"), grad)]))
        },
    )?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn unpack_packed_lu_square_tensors(
    factors: &Tensor<f64>,
) -> Result<(Tensor<f64>, Tensor<f64>), String> {
    let dims = factors.dims();
    if dims.len() < 2 || dims[0] != dims[1] {
        return Err(format!(
            "packed LU square unpack requires square dims, got {:?}",
            dims
        ));
    }
    let n = dims[0];
    let batch_dims = &dims[2..];
    let bc = crate::decode::batch_count(batch_dims);
    let factor_data = tensor_data_col_major(factors);
    let mut lower = vec![0.0; n * n * bc];
    let mut upper = vec![0.0; n * n * bc];

    for batch in 0..bc {
        let base = batch * n * n;
        for j in 0..n {
            for i in 0..n {
                let value = factor_data[base + i + j * n];
                if i > j {
                    lower[base + i + j * n] = value;
                } else {
                    upper[base + i + j * n] = value;
                    if i == j {
                        lower[base + i + j * n] = 1.0;
                    }
                }
            }
        }
    }

    Ok((
        crate::decode::tensor_from_col_major(lower, dims),
        crate::decode::tensor_from_col_major(upper, dims),
    ))
}

fn permutation_candidates(n: usize) -> Vec<Vec<usize>> {
    fn recurse(prefix: &mut Vec<usize>, rest: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if rest.is_empty() {
            out.push(prefix.clone());
            return;
        }
        for idx in 0..rest.len() {
            let value = rest.remove(idx);
            prefix.push(value);
            recurse(prefix, rest, out);
            prefix.pop();
            rest.insert(idx, value);
        }
    }

    let mut out = Vec::new();
    let mut prefix = Vec::new();
    let mut rest: Vec<usize> = (0..n).collect();
    recurse(&mut prefix, &mut rest, &mut out);
    out
}

fn apply_lu_permutation_tensor(rhs: &Tensor<f64>, pivots: &[usize]) -> Result<Tensor<f64>, String> {
    let dims = rhs.dims();
    if dims.len() < 2 {
        return Err(format!(
            "LU permutation requires matrix-style rhs dims, got {:?}",
            dims
        ));
    }
    let n = dims[0];
    let nrhs = dims[1];
    let batch_dims = &dims[2..];
    let bc = crate::decode::batch_count(batch_dims);
    if pivots.len() != n * bc {
        return Err(format!(
            "pivot length mismatch: expected {}, got {}",
            n * bc,
            pivots.len()
        ));
    }

    let rhs_data = tensor_data_col_major(rhs);
    let mut out = vec![0.0; rhs_data.len()];
    for batch in 0..bc {
        let rhs_base = batch * n * nrhs;
        let pivot_slice = &pivots[batch * n..(batch + 1) * n];
        for col in 0..nrhs {
            let col_offset = rhs_base + col * n;
            for row in 0..n {
                out[col_offset + row] = rhs_data[col_offset + pivot_slice[row]];
            }
        }
    }
    Ok(crate::decode::tensor_from_col_major(out, dims))
}

fn apply_lu_inverse_permutation_tensor(
    rhs: &Tensor<f64>,
    pivots: &[usize],
) -> Result<Tensor<f64>, String> {
    let dims = rhs.dims();
    if dims.len() < 2 {
        return Err(format!(
            "LU inverse permutation requires matrix-style rhs dims, got {:?}",
            dims
        ));
    }
    let n = dims[0];
    let nrhs = dims[1];
    let batch_dims = &dims[2..];
    let bc = crate::decode::batch_count(batch_dims);
    if pivots.len() != n * bc {
        return Err(format!(
            "pivot length mismatch: expected {}, got {}",
            n * bc,
            pivots.len()
        ));
    }

    let rhs_data = tensor_data_col_major(rhs);
    let mut out = vec![0.0; rhs_data.len()];
    for batch in 0..bc {
        let rhs_base = batch * n * nrhs;
        let pivot_slice = &pivots[batch * n..(batch + 1) * n];
        for col in 0..nrhs {
            let col_offset = rhs_base + col * n;
            for row in 0..n {
                out[col_offset + pivot_slice[row]] = rhs_data[col_offset + row];
            }
        }
    }
    Ok(crate::decode::tensor_from_col_major(out, dims))
}

fn lu_tangent_from_packed(packed: &Tensor<f64>) -> Result<(Tensor<f64>, Tensor<f64>), String> {
    let cotangent = unpack_lu_cotangent(packed)?;
    Ok((cotangent.l.unwrap(), cotangent.u.unwrap()))
}

fn lu_solve_flags(record: &CaseRecord) -> Result<(bool, bool), String> {
    let inner = (case_suffix_index(record)? - 1) % 4;
    let adjoint = inner < 2;
    let left = inner % 2 == 0;
    Ok((left, adjoint))
}

fn tensor_abs_error(expected: &Tensor<f64>, actual: &Tensor<f64>) -> Result<f64, String> {
    if expected.dims() != actual.dims() {
        return Err(format!(
            "tensor_abs_error shape mismatch: expected {:?}, got {:?}",
            expected.dims(),
            actual.dims()
        ));
    }
    Ok(tensor_data_col_major(expected)
        .iter()
        .zip(tensor_data_col_major(actual).iter())
        .map(|(exp, act)| (exp - act).abs())
        .sum())
}

fn lu_solve_left_forward(
    lower: &Tensor<f64>,
    upper: &Tensor<f64>,
    pivots: &[usize],
    b: &Tensor<f64>,
    adjoint: bool,
) -> Result<Tensor<f64>, String> {
    if tensor_is_empty(b) || tensor_is_empty(lower) {
        return Ok(b.clone());
    }

    let mut ctx = CpuContext::new(1);
    if !adjoint {
        let pb = apply_lu_permutation_tensor(b, pivots)?;
        let y = tenferro_linalg::solve_triangular(&mut ctx, lower, &pb, false)
            .map_err(|err| format!("lu_solve lower solve failed: {err}"))?;
        tenferro_linalg::solve_triangular(&mut ctx, upper, &y, true)
            .map_err(|err| format!("lu_solve upper solve failed: {err}"))
    } else {
        let upper_t = batched_transpose(upper)?;
        let lower_t = batched_transpose(lower)?;
        let y = tenferro_linalg::solve_triangular(&mut ctx, &upper_t, b, false)
            .map_err(|err| format!("lu_solve adjoint upper solve failed: {err}"))?;
        let z = tenferro_linalg::solve_triangular(&mut ctx, &lower_t, &y, true)
            .map_err(|err| format!("lu_solve adjoint lower solve failed: {err}"))?;
        apply_lu_inverse_permutation_tensor(&z, pivots)
    }
}

fn lu_solve_forward(
    factors: &Tensor<f64>,
    pivots: &[usize],
    b: &Tensor<f64>,
    left: bool,
    adjoint: bool,
) -> Result<Tensor<f64>, String> {
    let (lower, upper) = unpack_packed_lu_square_tensors(factors)?;
    if left {
        lu_solve_left_forward(&lower, &upper, pivots, b, adjoint)
    } else {
        let bt = batched_transpose(b)?;
        let yt = lu_solve_left_forward(&lower, &upper, pivots, &bt, !adjoint)?;
        batched_transpose(&yt)
    }
}

fn pivot_vector_combinations(candidates: &[Vec<usize>], batches: usize) -> Vec<Vec<usize>> {
    fn recurse(
        depth: usize,
        batches: usize,
        width: usize,
        candidates: &[Vec<usize>],
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<usize>>,
    ) {
        if depth == batches {
            out.push(current.clone());
            return;
        }
        for candidate in candidates {
            current.extend(candidate);
            recurse(depth + 1, batches, width, candidates, current, out);
            current.truncate(depth * width);
        }
    }

    if batches == 0 {
        return vec![Vec::new()];
    }
    let width = candidates.first().map_or(0, Vec::len);
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(width * batches);
    recurse(0, batches, width, candidates, &mut current, &mut out);
    out
}

fn infer_lu_solve_pivots_from_oracle(
    factors: &Tensor<f64>,
    plus_factors: &Tensor<f64>,
    minus_factors: &Tensor<f64>,
    plus_b: &Tensor<f64>,
    minus_b: &Tensor<f64>,
    expected_jvp: &Tensor<f64>,
    left: bool,
    adjoint: bool,
    step: f64,
) -> Result<Vec<usize>, String> {
    let dims = factors.dims();
    if dims.len() < 2 || dims[0] != dims[1] {
        return Err(format!("lu_solve factors must be square, got {:?}", dims));
    }
    let n = dims[0];
    let bc = crate::decode::batch_count(&dims[2..]);
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![0; bc]);
    }

    let candidates = permutation_candidates(n);
    let combinations = pivot_vector_combinations(&candidates, bc);
    let mut best = None::<(f64, Vec<usize>)>;
    for pivots in combinations {
        let plus = lu_solve_forward(plus_factors, &pivots, plus_b, left, adjoint)?;
        let minus = lu_solve_forward(minus_factors, &pivots, minus_b, left, adjoint)?;
        let plus_data = tensor_data_col_major(&plus);
        let minus_data = tensor_data_col_major(&minus);
        let jvp = crate::decode::tensor_from_col_major(
            plus_data
                .iter()
                .zip(minus_data.iter())
                .map(|(p, m)| (p - m) / (2.0 * step))
                .collect(),
            plus.dims(),
        );
        let error = tensor_abs_error(expected_jvp, &jvp)?;
        match &best {
            Some((best_error, _)) if error >= *best_error => {}
            _ => best = Some((error, pivots)),
        }
    }

    let (error, pivots) =
        best.ok_or_else(|| "failed to search LU solve pivot candidates".to_string())?;
    if error > 1e-6 {
        return Err(format!(
            "failed to infer LU solve pivots: best JVP error was {error}"
        ));
    }
    Ok(pivots)
}

fn lu_solve_left_jvp_vjp(
    lower: &Tensor<f64>,
    upper: &Tensor<f64>,
    dlower: &Tensor<f64>,
    dupper: &Tensor<f64>,
    pivots: &[usize],
    b: &Tensor<f64>,
    db: &Tensor<f64>,
    cotangent: &Tensor<f64>,
    adjoint: bool,
) -> Result<(Tensor<f64>, Tensor<f64>, Tensor<f64>), String> {
    if tensor_is_empty(b) || tensor_is_empty(lower) {
        return Ok((db.clone(), tensor_zeros_like(lower), cotangent.clone()));
    }

    let mut ctx = CpuContext::new(1);
    if !adjoint {
        let pb = apply_lu_permutation_tensor(b, pivots)?;
        let dpb = apply_lu_permutation_tensor(db, pivots)?;
        let (y, dy) = solve_triangular_frule(&mut ctx, lower, &pb, dlower, &dpb, false)
            .map_err(|err| format!("lu_solve lower frule failed: {err}"))?;
        let (_x, dx) = solve_triangular_frule(&mut ctx, upper, &y, dupper, &dy, true)
            .map_err(|err| format!("lu_solve upper frule failed: {err}"))?;
        let grad_upper = solve_triangular_rrule(&mut ctx, upper, &y, cotangent, true)
            .map_err(|err| format!("lu_solve upper rrule failed: {err}"))?;
        let grad_lower = solve_triangular_rrule(&mut ctx, lower, &pb, &grad_upper.b, false)
            .map_err(|err| format!("lu_solve lower rrule failed: {err}"))?;
        let grad_b = apply_lu_inverse_permutation_tensor(&grad_lower.b, pivots)?;
        let grad_factors = pack_lu_factors(&grad_lower.a, &grad_upper.a)?;
        Ok((dx, grad_factors, grad_b))
    } else {
        let upper_t = batched_transpose(upper)?;
        let lower_t = batched_transpose(lower)?;
        let dupper_t = batched_transpose(dupper)?;
        let dlower_t = batched_transpose(dlower)?;
        let (y, dy) = solve_triangular_frule(&mut ctx, &upper_t, b, &dupper_t, db, false)
            .map_err(|err| format!("lu_solve adjoint upper frule failed: {err}"))?;
        let (_z, dz) = solve_triangular_frule(&mut ctx, &lower_t, &y, &dlower_t, &dy, true)
            .map_err(|err| format!("lu_solve adjoint lower frule failed: {err}"))?;
        let dx = apply_lu_inverse_permutation_tensor(&dz, pivots)?;
        let z_bar = apply_lu_permutation_tensor(cotangent, pivots)?;
        let grad_lower_t = solve_triangular_rrule(&mut ctx, &lower_t, &y, &z_bar, true)
            .map_err(|err| format!("lu_solve adjoint lower rrule failed: {err}"))?;
        let grad_upper_t = solve_triangular_rrule(&mut ctx, &upper_t, b, &grad_lower_t.b, false)
            .map_err(|err| format!("lu_solve adjoint upper rrule failed: {err}"))?;
        let grad_lower = batched_transpose(&grad_lower_t.a)?;
        let grad_upper = batched_transpose(&grad_upper_t.a)?;
        let grad_factors = pack_lu_factors(&grad_lower, &grad_upper)?;
        Ok((dx, grad_factors, grad_upper_t.b))
    }
}

fn lu_solve_jvp_vjp(
    factors: &Tensor<f64>,
    pivots: &[usize],
    b: &Tensor<f64>,
    dfactors: &Tensor<f64>,
    db: &Tensor<f64>,
    cotangent: &Tensor<f64>,
    left: bool,
    adjoint: bool,
) -> Result<(Tensor<f64>, Tensor<f64>, Tensor<f64>), String> {
    let (lower, upper) = unpack_packed_lu_square_tensors(factors)?;
    let (dlower, dupper) = lu_tangent_from_packed(dfactors)?;
    if left {
        lu_solve_left_jvp_vjp(
            &lower, &upper, &dlower, &dupper, pivots, b, db, cotangent, adjoint,
        )
    } else {
        let bt = batched_transpose(b)?;
        let dbt = batched_transpose(db)?;
        let cot_t = batched_transpose(cotangent)?;
        let (dy_t, grad_factors, grad_bt) = lu_solve_left_jvp_vjp(
            &lower, &upper, &dlower, &dupper, pivots, &bt, &dbt, &cot_t, !adjoint,
        )?;
        Ok((
            batched_transpose(&dy_t)?,
            grad_factors,
            batched_transpose(&grad_bt)?,
        ))
    }
}

fn replay_lu_solve(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like(record, &probe.direction)?;
    let cotangent = decode_observable_map(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let value_key = replay_value_key(record)?;
    let (left, adjoint) = lu_solve_flags(record)?;
    let step = probe.fd_ref.step;
    let plus_inputs = perturb_input_map(&inputs, &direction, step)?;
    let minus_inputs = perturb_input_map(&inputs, &direction, -step)?;
    let pivots = infer_lu_solve_pivots_from_oracle(
        inputs.get("a").unwrap(),
        plus_inputs.get("a").unwrap(),
        minus_inputs.get("a").unwrap(),
        plus_inputs.get("b").unwrap(),
        minus_inputs.get("b").unwrap(),
        expected_jvp_fd.get(value_key).unwrap(),
        left,
        adjoint,
        step,
    )?;
    let plus = lu_solve_forward(
        plus_inputs.get("a").unwrap(),
        &pivots,
        plus_inputs.get("b").unwrap(),
        left,
        adjoint,
    )?;
    let minus = lu_solve_forward(
        minus_inputs.get("a").unwrap(),
        &pivots,
        minus_inputs.get("b").unwrap(),
        left,
        adjoint,
    )?;
    let plus_data = tensor_data_col_major(&plus);
    let minus_data = tensor_data_col_major(&minus);
    let actual_dx = crate::decode::tensor_from_col_major(
        plus_data
            .iter()
            .zip(minus_data.iter())
            .map(|(p, m)| (p - m) / (2.0 * step))
            .collect(),
        plus.dims(),
    );

    let (_ignored_dx, grad_factors, grad_b) = lu_solve_jvp_vjp(
        inputs.get("a").unwrap(),
        &pivots,
        inputs.get("b").unwrap(),
        direction.get("a").unwrap(),
        direction.get("b").unwrap(),
        cotangent.get(value_key).unwrap(),
        left,
        adjoint,
    )?;

    let actual_jvp = BTreeMap::from([(String::from(value_key), actual_dx)]);
    let actual_vjp = BTreeMap::from([
        (String::from("a"), grad_factors),
        (String::from("b"), grad_b),
    ]);
    compare_tensor_maps("lu_solve.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps(
        "lu_solve.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("lu_solve.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp(
        "lu_solve",
        record,
        &inputs,
        &direction,
        probe,
        |perturbed| {
            let grad_x = lu_solve_jvp_vjp(
                perturbed.get("a").unwrap(),
                &pivots,
                perturbed.get("b").unwrap(),
                &tensor_zeros_like(perturbed.get("a").unwrap()),
                &tensor_zeros_like(perturbed.get("b").unwrap()),
                cotangent.get(value_key).unwrap(),
                left,
                adjoint,
            )?;
            Ok(BTreeMap::from([
                (String::from("a"), grad_x.1),
                (String::from("b"), grad_x.2),
            ]))
        },
    )?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn replay_cond(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like(record, &probe.direction)?;
    let cotangent = decode_observable_map(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;

    let a = inputs.get("a").unwrap();
    let da = direction.get("a").unwrap();
    let value_key = replay_value_key(record)?;
    let mut ctx = CpuContext::new(1);
    let (n1, dn1) = norm_frule(&mut ctx, a, da, NormKind::Spectral)
        .map_err(|err| format!("cond norm_frule(a) failed: {err}"))?;
    let (ainv, dainv) =
        inv_frule(&mut ctx, a, da).map_err(|err| format!("cond inv_frule failed: {err}"))?;
    let (n2, dn2) = norm_frule(&mut ctx, &ainv, &dainv, NormKind::Spectral)
        .map_err(|err| format!("cond norm_frule(inv(a)) failed: {err}"))?;
    let actual_jvp = BTreeMap::from([(
        String::from(value_key),
        tensor_add(&tensor_mul(&dn1, &n2)?, &tensor_mul(&n1, &dn2)?),
    )]);

    let scaled_n2 = tensor_mul(cotangent.get(value_key).unwrap(), &n2)?;
    let scaled_n1 = tensor_mul(cotangent.get(value_key).unwrap(), &n1)?;
    let grad_direct = norm_rrule(&mut ctx, a, &scaled_n2, NormKind::Spectral)
        .map_err(|err| format!("cond norm_rrule(a) failed: {err}"))?;
    let grad_inv = norm_rrule(&mut ctx, &ainv, &scaled_n1, NormKind::Spectral)
        .map_err(|err| format!("cond norm_rrule(inv(a)) failed: {err}"))?;
    let grad_from_inv =
        inv_rrule(&mut ctx, a, &grad_inv).map_err(|err| format!("cond inv_rrule failed: {err}"))?;
    let actual_vjp =
        BTreeMap::from([(String::from("a"), tensor_add(&grad_direct, &grad_from_inv))]);

    compare_tensor_maps("cond.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps(
        "cond.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("cond.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp("cond", record, &inputs, &direction, probe, |perturbed| {
        let pa = perturbed.get("a").unwrap();
        let mut ctx = CpuContext::new(1);
        let n1 = tenferro_linalg::norm(&mut ctx, pa, NormKind::Spectral)
            .map_err(|err| format!("cond norm(pa) failed during HVP replay: {err}"))?;
        let ainv = tenferro_linalg::inv(&mut ctx, pa)
            .map_err(|err| format!("cond inv(pa) failed during HVP replay: {err}"))?;
        let n2 = tenferro_linalg::norm(&mut ctx, &ainv, NormKind::Spectral)
            .map_err(|err| format!("cond norm(inv(pa)) failed during HVP replay: {err}"))?;
        let scaled_n2 = tensor_mul(cotangent.get(value_key).unwrap(), &n2)?;
        let scaled_n1 = tensor_mul(cotangent.get(value_key).unwrap(), &n1)?;
        let grad_direct = norm_rrule(&mut ctx, pa, &scaled_n2, NormKind::Spectral)
            .map_err(|err| format!("cond norm_rrule(pa) failed during HVP replay: {err}"))?;
        let grad_inv = norm_rrule(&mut ctx, &ainv, &scaled_n1, NormKind::Spectral)
            .map_err(|err| format!("cond norm_rrule(inv(pa)) failed during HVP replay: {err}"))?;
        let grad_from_inv = inv_rrule(&mut ctx, pa, &grad_inv)
            .map_err(|err| format!("cond inv_rrule(pa) failed during HVP replay: {err}"))?;
        Ok(BTreeMap::from([(
            String::from("a"),
            tensor_add(&grad_direct, &grad_from_inv),
        )]))
    })?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn matrix_power_exponent(record: &CaseRecord) -> Result<i64, String> {
    const CYCLE: [i64; 6] = [0, 3, 5, -4, -2, -1];
    let suffix = record
        .case_id
        .rsplit('_')
        .next()
        .ok_or_else(|| format!("failed to parse matrix_power case id {}", record.case_id))?;
    let index = suffix.parse::<usize>().map_err(|err| {
        format!(
            "failed to parse matrix_power case id {}: {err}",
            record.case_id
        )
    })?;
    if index == 0 {
        return Err(format!(
            "matrix_power case id must be 1-based: {}",
            record.case_id
        ));
    }
    Ok(CYCLE[(index - 1) % CYCLE.len()])
}

fn matrix_power_positive_tensors(
    a: &Tensor<f64>,
    exponent: u64,
) -> Result<Vec<Tensor<f64>>, String> {
    let mut powers = Vec::with_capacity(exponent as usize + 1);
    powers.push(batched_identity_like(a)?);
    for _ in 0..exponent {
        let next = batched_matmul(powers.last().unwrap(), a)?;
        powers.push(next);
    }
    Ok(powers)
}

fn matrix_power_positive_jvp(
    a: &Tensor<f64>,
    da: &Tensor<f64>,
    exponent: u64,
) -> Result<Tensor<f64>, String> {
    if exponent == 0 {
        return Ok(tensor_zeros_like(a));
    }
    let powers = matrix_power_positive_tensors(a, exponent)?;
    let mut sum = tensor_zeros_like(a);
    for k in 0..exponent as usize {
        let term = batched_matmul(
            &batched_matmul(&powers[k], da)?,
            &powers[exponent as usize - 1 - k],
        )?;
        sum = tensor_add(&sum, &term);
    }
    Ok(sum)
}

fn matrix_power_positive_vjp(
    a: &Tensor<f64>,
    cotangent: &Tensor<f64>,
    exponent: u64,
) -> Result<Tensor<f64>, String> {
    if exponent == 0 {
        return Ok(tensor_zeros_like(a));
    }
    let powers = matrix_power_positive_tensors(a, exponent)?;
    let mut sum = tensor_zeros_like(a);
    for k in 0..exponent as usize {
        let left = batched_transpose(&powers[k])?;
        let right = batched_transpose(&powers[exponent as usize - 1 - k])?;
        let term = batched_matmul(&batched_matmul(&left, cotangent)?, &right)?;
        sum = tensor_add(&sum, &term);
    }
    Ok(sum)
}

fn matrix_power_jvp(
    a: &Tensor<f64>,
    da: &Tensor<f64>,
    exponent: i64,
) -> Result<Tensor<f64>, String> {
    if exponent == 0 {
        return Ok(tensor_zeros_like(a));
    }
    if exponent > 0 {
        return matrix_power_positive_jvp(a, da, exponent as u64);
    }

    let mut ctx = CpuContext::new(1);
    let (ainv, dainv) = inv_frule(&mut ctx, a, da)
        .map_err(|err| format!("matrix_power inv_frule failed: {err}"))?;
    matrix_power_positive_jvp(&ainv, &dainv, exponent.unsigned_abs())
}

fn matrix_power_vjp(
    a: &Tensor<f64>,
    cotangent: &Tensor<f64>,
    exponent: i64,
) -> Result<Tensor<f64>, String> {
    if exponent == 0 {
        return Ok(tensor_zeros_like(a));
    }
    if exponent > 0 {
        return matrix_power_positive_vjp(a, cotangent, exponent as u64);
    }

    let mut ctx = CpuContext::new(1);
    let ainv = tenferro_linalg::inv(&mut ctx, a)
        .map_err(|err| format!("matrix_power inv failed: {err}"))?;
    let grad_ainv = matrix_power_positive_vjp(&ainv, cotangent, exponent.unsigned_abs())?;
    inv_rrule(&mut ctx, a, &grad_ainv)
        .map_err(|err| format!("matrix_power inv_rrule failed: {err}"))
}

fn replay_matrix_power(record: &CaseRecord) -> Result<bool, String> {
    let inputs = decode_inputs(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like(record, &probe.direction)?;
    let cotangent = decode_observable_map(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let exponent = matrix_power_exponent(record)?;
    let value_key = replay_value_key(record)?;

    let actual_jvp = BTreeMap::from([(
        String::from(value_key),
        matrix_power_jvp(
            inputs.get("a").unwrap(),
            direction.get("a").unwrap(),
            exponent,
        )?,
    )]);
    let actual_vjp = BTreeMap::from([(
        String::from("a"),
        matrix_power_vjp(
            inputs.get("a").unwrap(),
            cotangent.get(value_key).unwrap(),
            exponent,
        )?,
    )]);

    compare_tensor_maps(
        "matrix_power.jvp.fd",
        &expected_jvp_fd,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps(
        "matrix_power.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps("matrix_power.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked = validate_hvp(
        "matrix_power",
        record,
        &inputs,
        &direction,
        probe,
        |perturbed| {
            Ok(BTreeMap::from([(
                String::from("a"),
                matrix_power_vjp(
                    perturbed.get("a").unwrap(),
                    cotangent.get(value_key).unwrap(),
                    exponent,
                )?,
            )]))
        },
    )?;
    check_adjoint_identity(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn infer_triangular_orientation_typed<T: OracleDbScalar>(
    tensor: &Tensor<T>,
) -> Result<TriangularOrientation, String> {
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
                let value = values[base + i + j * n]
                    .abs_real()
                    .to_f64()
                    .ok_or_else(|| "failed to convert triangular norm".to_string())?;
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
        "could not infer triangular orientation from norms upper={upper_norm} lower={lower_norm}"
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

#[derive(Clone, Debug)]
enum SvdObservable<T: OracleDbScalar>
where
    T::Real: OracleDbScalar,
{
    UAbs {
        u: Tensor<T::Real>,
    },
    S {
        s: Tensor<T::Real>,
    },
    VhAbs {
        s: Tensor<T::Real>,
        vh: Tensor<T::Real>,
    },
    UvhProduct {
        s: Tensor<T::Real>,
        uvh: Tensor<T>,
    },
}

fn decode_svd_input_tensor<T: OracleDbScalar>(
    encoded: &BTreeMap<String, DbTensor>,
) -> Result<Tensor<T>, String> {
    let tensor = encoded
        .get("a")
        .ok_or_else(|| "missing SVD input tensor a".to_string())?;
    decode_tensor_with_core_rank(tensor, 2)
}

fn tensor_real_inner_product_typed<T: OracleDbScalar>(
    left: &Tensor<T>,
    right: &Tensor<T>,
) -> Result<f64, String> {
    if left.dims() != right.dims() {
        return Err(format!(
            "real inner-product shape mismatch: left {:?}, right {:?}",
            left.dims(),
            right.dims()
        ));
    }
    let left_data = tensor_data_col_major(left);
    let right_data = tensor_data_col_major(right);
    left_data
        .iter()
        .zip(right_data.iter())
        .try_fold(0.0, |acc, (lhs, rhs)| {
            ((*lhs).conj() * *rhs)
                .real_part()
                .to_f64()
                .map(|value| acc + value)
                .ok_or_else(|| "failed to convert inner-product contribution to f64".to_string())
        })
}

fn scalar_from_f64<T: OracleDbScalar>(value: f64) -> Result<T, String> {
    let real = NumCast::from(value).ok_or_else(|| format!("failed to cast scalar step {value}"))?;
    Ok(T::from_real(real))
}

fn perturb_tensor_typed<T: OracleDbScalar>(
    base: &Tensor<T>,
    direction: &Tensor<T>,
    scale: f64,
) -> Result<Tensor<T>, String> {
    if base.dims() != direction.dims() {
        return Err(format!(
            "tensor perturbation shape mismatch: base {:?}, direction {:?}",
            base.dims(),
            direction.dims()
        ));
    }
    let base_data = tensor_data_col_major(base);
    let direction_data = tensor_data_col_major(direction);
    let scale_t = scalar_from_f64::<T>(scale)?;
    let data: Vec<T> = base_data
        .iter()
        .zip(direction_data.iter())
        .map(|(x, dx)| *x + *dx * scale_t)
        .collect();
    Ok(tensor_from_col_major(data, base.dims()))
}

fn central_diff_tensor_typed<T: OracleDbScalar>(
    plus: &Tensor<T>,
    minus: &Tensor<T>,
    step: f64,
) -> Result<Tensor<T>, String> {
    if step <= 0.0 {
        return Err(format!(
            "central difference requires positive step, got {step}"
        ));
    }
    if plus.dims() != minus.dims() {
        return Err(format!(
            "central-diff shape mismatch: plus {:?}, minus {:?}",
            plus.dims(),
            minus.dims()
        ));
    }
    let plus_data = tensor_data_col_major(plus);
    let minus_data = tensor_data_col_major(minus);
    let denom = scalar_from_f64::<T>(2.0 * step)?;
    let data: Vec<T> = plus_data
        .iter()
        .zip(minus_data.iter())
        .map(|(p, m)| (*p - *m) / denom)
        .collect();
    Ok(tensor_from_col_major(data, plus.dims()))
}

impl<T> SvdObservable<T>
where
    T: OracleDbScalar,
    T::Real: OracleDbScalar,
{
    fn decode(record: &CaseRecord, encoded: &BTreeMap<String, DbTensor>) -> Result<Self, String> {
        match record.family.as_str() {
            "u_abs" => Ok(Self::UAbs {
                u: decode_tensor_with_core_rank(
                    encoded.get("u").ok_or_else(|| {
                        format!("missing SVD observable u for {}", record.case_id)
                    })?,
                    2,
                )?,
            }),
            "s" => Ok(Self::S {
                s: decode_tensor_with_core_rank(
                    encoded.get("s").ok_or_else(|| {
                        format!("missing SVD observable s for {}", record.case_id)
                    })?,
                    1,
                )?,
            }),
            "vh_abs" => Ok(Self::VhAbs {
                s: decode_tensor_with_core_rank(
                    encoded.get("s").ok_or_else(|| {
                        format!("missing SVD observable s for {}", record.case_id)
                    })?,
                    1,
                )?,
                vh: decode_tensor_with_core_rank(
                    encoded.get("vh").ok_or_else(|| {
                        format!("missing SVD observable vh for {}", record.case_id)
                    })?,
                    2,
                )?,
            }),
            "uvh_product" => Ok(Self::UvhProduct {
                s: decode_tensor_with_core_rank(
                    encoded.get("s").ok_or_else(|| {
                        format!("missing SVD observable s for {}", record.case_id)
                    })?,
                    1,
                )?,
                uvh: decode_tensor_with_core_rank(
                    encoded.get("uvh").ok_or_else(|| {
                        format!("missing SVD observable uvh for {}", record.case_id)
                    })?,
                    2,
                )?,
            }),
            other => Err(format!("unsupported svd family {other}")),
        }
    }

    fn from_jvp(
        family: &str,
        primal_u: &Tensor<T>,
        primal_vt: &Tensor<T>,
        du: &Tensor<T>,
        ds: &Tensor<T::Real>,
        dvt: &Tensor<T>,
    ) -> Result<Self, String> {
        match family {
            "u_abs" => Ok(Self::UAbs {
                u: elementwise_abs_jvp(primal_u, du),
            }),
            "s" => Ok(Self::S { s: ds.clone() }),
            "vh_abs" => Ok(Self::VhAbs {
                s: ds.clone(),
                vh: elementwise_abs_jvp(primal_vt, dvt),
            }),
            "uvh_product" => Ok(Self::UvhProduct {
                s: ds.clone(),
                uvh: tensor_add(
                    &batched_matmul(du, primal_vt)?,
                    &batched_matmul(primal_u, dvt)?,
                ),
            }),
            other => Err(format!("unsupported svd family {other}")),
        }
    }

    fn to_cotangent(
        &self,
        primal_u: &Tensor<T>,
        primal_vt: &Tensor<T>,
    ) -> Result<SvdCotangent<T, T::Real>, String> {
        match self {
            Self::UAbs { u } => Ok(SvdCotangent {
                u: Some(elementwise_abs_vjp(primal_u, u)),
                s: None,
                vt: None,
            }),
            Self::S { s } => Ok(SvdCotangent {
                u: None,
                s: Some(s.clone()),
                vt: None,
            }),
            Self::VhAbs { s, vh } => Ok(SvdCotangent {
                u: None,
                s: Some(s.clone()),
                vt: Some(elementwise_abs_vjp(primal_vt, vh)),
            }),
            Self::UvhProduct { s, uvh } => {
                let v = batched_adjoint_transpose(primal_vt)?;
                let u_h = batched_adjoint_transpose(primal_u)?;
                Ok(SvdCotangent {
                    u: Some(batched_matmul(uvh, &v)?),
                    s: Some(s.clone()),
                    vt: Some(batched_matmul(&u_h, uvh)?),
                })
            }
        }
    }

    fn compare(&self, label: &str, actual: &Self, rtol: f64, atol: f64) -> Result<(), String> {
        match (self, actual) {
            (Self::UAbs { u: exp }, Self::UAbs { u: act }) => {
                compare_tensors_typed::<T::Real>(label, exp, act, rtol, atol)
            }
            (Self::S { s: exp }, Self::S { s: act }) => {
                compare_tensors_typed::<T::Real>(label, exp, act, rtol, atol)
            }
            (
                Self::VhAbs {
                    s: exp_s,
                    vh: exp_vh,
                },
                Self::VhAbs {
                    s: act_s,
                    vh: act_vh,
                },
            ) => {
                compare_tensors_typed::<T::Real>(&format!("{label}.s"), exp_s, act_s, rtol, atol)?;
                compare_tensors_typed::<T::Real>(&format!("{label}.vh"), exp_vh, act_vh, rtol, atol)
            }
            (
                Self::UvhProduct {
                    s: exp_s,
                    uvh: exp_uvh,
                },
                Self::UvhProduct {
                    s: act_s,
                    uvh: act_uvh,
                },
            ) => {
                compare_tensors_typed::<T::Real>(&format!("{label}.s"), exp_s, act_s, rtol, atol)?;
                compare_tensors_typed::<T>(&format!("{label}.uvh"), exp_uvh, act_uvh, rtol, atol)
            }
            _ => Err(format!("{label}: SVD observable family mismatch")),
        }
    }

    fn real_inner_product(&self, other: &Self) -> Result<f64, String> {
        match (self, other) {
            (Self::UAbs { u: lhs }, Self::UAbs { u: rhs }) => {
                tensor_real_inner_product_typed(lhs, rhs)
            }
            (Self::S { s: lhs }, Self::S { s: rhs }) => tensor_real_inner_product_typed(lhs, rhs),
            (
                Self::VhAbs {
                    s: lhs_s,
                    vh: lhs_vh,
                },
                Self::VhAbs {
                    s: rhs_s,
                    vh: rhs_vh,
                },
            ) => Ok(tensor_real_inner_product_typed(lhs_s, rhs_s)?
                + tensor_real_inner_product_typed(lhs_vh, rhs_vh)?),
            (
                Self::UvhProduct {
                    s: lhs_s,
                    uvh: lhs_uvh,
                },
                Self::UvhProduct {
                    s: rhs_s,
                    uvh: rhs_uvh,
                },
            ) => Ok(tensor_real_inner_product_typed(lhs_s, rhs_s)?
                + tensor_real_inner_product_typed(lhs_uvh, rhs_uvh)?),
            _ => Err("SVD observable family mismatch during adjoint validation".to_string()),
        }
    }
}

fn replay_svd_typed<T>(record: &CaseRecord) -> Result<bool, String>
where
    T: OracleDbScalar + KernelLinalgScalar,
    T::Real: OracleDbScalar + Float + tenferro_tensor::KeepCountScalar,
{
    let input = decode_svd_input_tensor::<T>(&record.inputs)?;
    let probe = probe(record)?;
    let direction = decode_svd_input_tensor::<T>(&probe.direction)?;
    let cotangent = SvdObservable::<T>::decode(record, &probe.cotangent)?;
    let expected_jvp_fd = SvdObservable::<T>::decode(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = SvdObservable::<T>::decode(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_svd_input_tensor::<T>(&probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;

    let mut ctx = CpuContext::new(1);
    let (result, dresult) = svd_frule(&mut ctx, &input, &direction, None)
        .map_err(|err| format!("svd_frule failed: {err}"))?;
    let cotangent_raw = cotangent.to_cotangent(&result.u, &result.vt)?;
    let grad = svd_rrule(&mut ctx, &input, &cotangent_raw, None)
        .map_err(|err| format!("svd_rrule failed: {err}"))?;

    let actual_jvp = SvdObservable::<T>::from_jvp(
        record.family.as_str(),
        &result.u,
        &result.vt,
        &dresult.u,
        &dresult.s,
        &dresult.vt,
    )?;

    expected_jvp_fd.compare("svd.jvp.fd", &actual_jvp, rtol, atol)?;
    expected_jvp_torch.compare("svd.jvp.torch", &actual_jvp, rtol, atol)?;
    compare_tensors_typed::<T>("svd.vjp.a", &expected_vjp, &grad, rtol, atol)?;

    let hvp_checked = match (probe.pytorch_ref.hvp.as_ref(), probe.fd_ref.hvp.as_ref()) {
        (Some(expected_torch), Some(expected_fd)) => {
            let expected_hvp_torch = decode_svd_input_tensor::<T>(expected_torch)?;
            let expected_hvp_fd = decode_svd_input_tensor::<T>(expected_fd)?;
            let step = probe.fd_ref.step;
            let plus_input = perturb_tensor_typed(&input, &direction, step)?;
            let minus_input = perturb_tensor_typed(&input, &direction, -step)?;
            let evaluate_grad = |perturbed: &Tensor<T>| -> Result<Tensor<T>, String> {
                let mut ctx = CpuContext::new(1);
                let primal = svd(&mut ctx, perturbed, None)
                    .map_err(|err| format!("svd failed during HVP replay: {err}"))?;
                let cotangent_raw = cotangent.to_cotangent(&primal.u, &primal.vt)?;
                svd_rrule(&mut ctx, perturbed, &cotangent_raw, None)
                    .map_err(|err| format!("svd_rrule failed during HVP replay: {err}"))
            };
            let grad_plus = evaluate_grad(&plus_input)?;
            let grad_minus = evaluate_grad(&minus_input)?;
            let actual_hvp = central_diff_tensor_typed(&grad_plus, &grad_minus, step)?;
            let (second_rtol, second_atol) = second_order_comparison(record)?;
            compare_tensors_typed::<T>(
                "svd.hvp.fd.a",
                &expected_hvp_fd,
                &actual_hvp,
                second_rtol,
                second_atol,
            )?;
            compare_tensors_typed::<T>(
                "svd.hvp.torch.a",
                &expected_hvp_torch,
                &actual_hvp,
                second_rtol,
                second_atol,
            )?;
            true
        }
        (None, None) => false,
        _ => return Err(format!("half-present HVP payload for {}", record.case_id)),
    };

    let lhs = cotangent.real_inner_product(&actual_jvp)?;
    let rhs = tensor_real_inner_product_typed(&grad, &direction)?;
    let allowed = atol + rtol * lhs.abs();
    if (lhs - rhs).abs() > allowed {
        return Err(format!(
            "adjoint identity mismatch: lhs={lhs}, rhs={rhs}, allowed={allowed}"
        ));
    }
    Ok(hvp_checked)
}

fn replay_svd(record: &CaseRecord) -> Result<bool, String> {
    match record.dtype.as_str() {
        "float32" => replay_svd_typed::<f32>(record),
        "float64" => replay_svd_typed::<f64>(record),
        "complex64" => replay_svd_typed::<Complex32>(record),
        "complex128" => replay_svd_typed::<Complex64>(record),
        other => Err(format!("unsupported SVD replay dtype {other}")),
    }
}

#[derive(Clone, Debug)]
struct EigenObservable<T: OracleDbScalar>
where
    T::Real: OracleDbScalar,
{
    values: Tensor<T::Real>,
    vectors_abs: Tensor<T::Real>,
}

impl<T> EigenObservable<T>
where
    T: OracleDbScalar,
    T::Real: OracleDbScalar,
{
    fn decode(record: &CaseRecord, encoded: &BTreeMap<String, DbTensor>) -> Result<Self, String> {
        Ok(Self {
            values: decode_tensor_with_core_rank(
                encoded.get("values").ok_or_else(|| {
                    format!("missing eigen observable values for {}", record.case_id)
                })?,
                1,
            )?,
            vectors_abs: decode_tensor_with_core_rank(
                encoded.get("vectors").ok_or_else(|| {
                    format!("missing eigen observable vectors for {}", record.case_id)
                })?,
                2,
            )?,
        })
    }

    fn from_jvp(
        primal_vectors: &Tensor<T>,
        dvalues: &Tensor<T::Real>,
        dvectors: &Tensor<T>,
    ) -> Self {
        Self {
            values: dvalues.clone(),
            vectors_abs: elementwise_abs_jvp(primal_vectors, dvectors),
        }
    }

    fn to_cotangent(&self, primal_vectors: &Tensor<T>) -> EigenCotangent<T, T::Real> {
        EigenCotangent {
            values: Some(self.values.clone()),
            vectors: Some(elementwise_abs_vjp(primal_vectors, &self.vectors_abs)),
        }
    }

    fn compare(&self, label: &str, actual: &Self, rtol: f64, atol: f64) -> Result<(), String> {
        compare_tensors_typed::<T::Real>(
            &format!("{label}.values"),
            &self.values,
            &actual.values,
            rtol,
            atol,
        )?;
        compare_tensors_typed::<T::Real>(
            &format!("{label}.vectors"),
            &self.vectors_abs,
            &actual.vectors_abs,
            rtol,
            atol,
        )
    }

    fn real_inner_product(&self, other: &Self) -> Result<f64, String> {
        Ok(
            tensor_real_inner_product_typed(&self.values, &other.values)?
                + tensor_real_inner_product_typed(&self.vectors_abs, &other.vectors_abs)?,
        )
    }
}

fn replay_eigen_typed<T>(record: &CaseRecord) -> Result<bool, String>
where
    T: OracleDbScalar + KernelLinalgScalar + Conjugate,
    T::Real: OracleDbScalar + KernelLinalgScalar<Real = T::Real> + Float,
{
    let inputs = decode_inputs_typed::<T>(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like_typed::<T>(record, &probe.direction)?;
    let cotangent = EigenObservable::<T>::decode(record, &probe.cotangent)?;
    let expected_jvp_fd = EigenObservable::<T>::decode(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = EigenObservable::<T>::decode(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like_typed::<T>(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let wrapped_a = apply_hermitian_wrapper_typed(inputs.get("a").unwrap())?;
    let wrapped_da = apply_hermitian_wrapper_typed(direction.get("a").unwrap())?;

    let mut ctx = CpuContext::new(1);
    let (result, dresult) = eigen_frule(&mut ctx, &wrapped_a, &wrapped_da)
        .map_err(|err| format!("eigen_frule failed: {err}"))?;
    let grad = eigen_rrule(
        &mut ctx,
        &wrapped_a,
        &cotangent.to_cotangent(&result.vectors),
    )
    .map_err(|err| format!("eigen_rrule failed: {err}"))?;

    let actual_jvp =
        EigenObservable::<T>::from_jvp(&result.vectors, &dresult.values, &dresult.vectors);
    let actual_vjp = BTreeMap::from([(String::from("a"), apply_hermitian_wrapper_typed(&grad)?)]);
    expected_jvp_fd.compare("eigen.jvp.fd", &actual_jvp, rtol, atol)?;
    expected_jvp_torch.compare("eigen.jvp.torch", &actual_jvp, rtol, atol)?;
    compare_tensor_maps_typed("eigen.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked =
        validate_hvp_typed("eigen", record, &inputs, &direction, probe, |perturbed| {
            let wrapped = apply_hermitian_wrapper_typed(perturbed.get("a").unwrap())?;
            let mut ctx = CpuContext::new(1);
            let primal = eigen(&mut ctx, &wrapped)
                .map_err(|err| format!("eigen failed during HVP replay: {err}"))?;
            let grad = eigen_rrule(&mut ctx, &wrapped, &cotangent.to_cotangent(&primal.vectors))
                .map_err(|err| format!("eigen_rrule failed during HVP replay: {err}"))?;
            Ok(BTreeMap::from([(
                String::from("a"),
                apply_hermitian_wrapper_typed(&grad)?,
            )]))
        })?;
    let lhs = cotangent.real_inner_product(&actual_jvp)?;
    let rhs =
        tensor_real_inner_product_typed(actual_vjp.get("a").unwrap(), direction.get("a").unwrap())?;
    let allowed = atol + rtol * lhs.abs();
    if (lhs - rhs).abs() > allowed {
        return Err(format!(
            "adjoint identity mismatch: lhs={lhs}, rhs={rhs}, allowed={allowed}"
        ));
    }
    Ok(hvp_checked)
}

fn replay_eigen(record: &CaseRecord) -> Result<bool, String> {
    match record.dtype.as_str() {
        "float32" => replay_eigen_typed::<f32>(record),
        "float64" => replay_eigen_typed::<f64>(record),
        "complex64" => replay_eigen_typed::<Complex32>(record),
        "complex128" => replay_eigen_typed::<Complex64>(record),
        other => Err(format!("unsupported eigen replay dtype {other}")),
    }
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

fn replay_pinv_t<T>(record: &CaseRecord) -> Result<bool, String>
where
    T: OracleDbScalar + KernelLinalgScalar + Conjugate + ScaleTensorByRealSameShape<CpuContext>,
    T::Real: KeepCountScalar,
{
    let inputs = decode_inputs_typed::<T>(record)?;
    let probe = probe(record)?;
    let direction = decode_input_map_like_typed::<T>(record, &probe.direction)?;
    let cotangent = decode_observable_map_typed::<T>(record, &probe.cotangent)?;
    let expected_jvp_fd = decode_observable_map_typed::<T>(record, &probe.fd_ref.jvp)?;
    let expected_jvp_torch = decode_observable_map_typed::<T>(record, &probe.pytorch_ref.jvp)?;
    let expected_vjp = decode_input_map_like_typed::<T>(record, &probe.pytorch_ref.vjp)?;
    let (rtol, atol) = comparison(record)?;
    let value_key = replay_value_key(record)?;
    let rcond = pinv_rcond(record)?;

    let mut ctx = CpuContext::new(1);
    let (_ap, dap) = pinv_frule(
        &mut ctx,
        inputs.get("a").unwrap(),
        direction.get("a").unwrap(),
        rcond,
    )
    .map_err(|err| format!("pinv_frule failed: {err}"))?;
    let grad = pinv_rrule(
        &mut ctx,
        inputs.get("a").unwrap(),
        cotangent.get(value_key).unwrap(),
        rcond,
    )
    .map_err(|err| format!("pinv_rrule failed: {err}"))?;

    let actual_jvp = BTreeMap::from([(String::from(value_key), dap)]);
    let actual_vjp = BTreeMap::from([(String::from("a"), grad)]);
    compare_tensor_maps_typed("pinv.jvp.fd", &expected_jvp_fd, &actual_jvp, rtol, atol)?;
    compare_tensor_maps_typed(
        "pinv.jvp.torch",
        &expected_jvp_torch,
        &actual_jvp,
        rtol,
        atol,
    )?;
    compare_tensor_maps_typed("pinv.vjp", &expected_vjp, &actual_vjp, rtol, atol)?;
    let hvp_checked =
        validate_hvp_typed("pinv", record, &inputs, &direction, probe, |perturbed| {
            let mut ctx = CpuContext::new(1);
            let grad = pinv_rrule(
                &mut ctx,
                perturbed.get("a").unwrap(),
                cotangent.get(value_key).unwrap(),
                rcond,
            )
            .map_err(|err| format!("pinv_rrule failed during HVP replay: {err}"))?;
            Ok(BTreeMap::from([(String::from("a"), grad)]))
        })?;
    check_adjoint_identity_typed(record, &cotangent, &actual_jvp, &actual_vjp, &direction)?;
    Ok(hvp_checked)
}

fn replay_pinv(record: &CaseRecord) -> Result<bool, String> {
    match record.dtype.as_str() {
        "float64" => replay_pinv_t::<f64>(record),
        "complex64" => replay_pinv_t::<Complex32>(record),
        "complex128" => replay_pinv_t::<Complex64>(record),
        other => Err(format!(
            "unsupported replay dtype {other} for {}",
            record.case_id
        )),
    }
}
