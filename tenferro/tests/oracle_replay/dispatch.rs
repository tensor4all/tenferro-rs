use std::collections::BTreeMap;

use tenferro::{cholesky, eigh, qr, solve, svd, triangular_solve, TracedTensor};

use crate::decode::{try_decode_tensor, CaseRecord};

pub struct NamedTensor {
    pub name: String,
    pub tensor: TracedTensor,
}

pub struct CaseExecution {
    pub inputs: BTreeMap<String, TracedTensor>,
    pub outputs: Vec<NamedTensor>,
}

pub enum DispatchResult {
    Executed(CaseExecution),
    SkippedUnimplemented(String),
}

pub fn dispatch_case(case: &CaseRecord) -> Result<DispatchResult, String> {
    if !replay_enabled_op(case.op.as_str()) {
        return Ok(DispatchResult::SkippedUnimplemented(case.op.clone()));
    }

    let inputs = decode_inputs(case)?;

    let a = required_input(&inputs, "a", case)?.clone();

    let result = match case.op.as_str() {
        "abs" => single_output(inputs, "value", a.abs()),
        "neg" => single_output(inputs, "value", -&a),
        "exp" => single_output(inputs, "value", a.exp()),
        "expm1" => single_output(inputs, "value", a.expm1()),
        "log" => single_output(inputs, "value", a.log()),
        "log1p" => single_output(inputs, "value", a.log1p()),
        "sin" => single_output(inputs, "value", a.sin()),
        "cos" => single_output(inputs, "value", a.cos()),
        "tanh" => single_output(inputs, "value", a.tanh()),
        "sqrt" => single_output(inputs, "value", a.sqrt()),
        "rsqrt" => single_output(inputs, "value", a.rsqrt()),
        "sign" | "sgn" => single_output(inputs, "value", a.sign()),
        "conj" => single_output(inputs, "value", a.conj()),
        "conj_physical" => single_output(inputs, "value", a.clone()),
        "add" | "__radd__" => {
            let b = required_input(&inputs, "b", case)?.clone();
            let alpha = alpha_kwarg(case)?;
            single_output(inputs, "value", a.add(&b.scale_real(alpha)))
        }
        "mul" | "__rmul__" => {
            let b = required_input(&inputs, "b", case)?.clone();
            single_output(inputs, "value", a.mul(&b))
        }
        "sub" => {
            let b = required_input(&inputs, "b", case)?.clone();
            let alpha = alpha_kwarg(case)?;
            single_output(inputs, "value", a.add(&b.scale_real(-alpha)))
        }
        "rsub" | "__rsub__" => {
            let b = required_input(&inputs, "b", case)?.clone();
            let alpha = alpha_kwarg(case)?;
            single_output(inputs, "value", b.add(&a.scale_real(-alpha)))
        }
        "div_no_rounding_mode" | "true_divide" => {
            let b = required_input(&inputs, "b", case)?.clone();
            single_output(inputs, "value", a.div(&b))
        }
        "__rdiv__" => {
            let b = required_input(&inputs, "b", case)?.clone();
            single_output(inputs, "value", b.div(&a))
        }
        "pow" | "float_power" => {
            let b = required_input(&inputs, "b", case)?.clone();
            single_output(inputs, "value", a.pow(&b))
        }
        "__rpow__" => {
            let b = required_input(&inputs, "b", case)?.clone();
            single_output(inputs, "value", b.pow(&a))
        }
        "sum" => {
            let axes = sum_axes(case, a.shape.len())?;
            let keepdim = bool_kwarg(&case.op_kwargs, "keepdim")?.unwrap_or(false);
            let reduced = a.reduce_sum(&axes);
            if keepdim {
                let kept_shape = keepdim_shape(&a.shape, &axes);
                single_output(inputs, "value", reduced.reshape(&kept_shape))
            } else {
                single_output(inputs, "value", reduced)
            }
        }
        "svd" => {
            let a_tf = oracle_to_tenferro(&a, 2);
            let (u_tf, s_tf, vh_tf) = svd(&a_tf);
            DispatchResult::Executed(CaseExecution {
                inputs,
                outputs: vec![
                    named("u", tenferro_to_oracle(&u_tf, 2)),
                    named("s", tenferro_to_oracle(&s_tf, 1)),
                    named("vh", tenferro_to_oracle(&vh_tf, 2)),
                ],
            })
        }
        "qr" => {
            let a_tf = oracle_to_tenferro(&a, 2);
            let (q_tf, r_tf) = qr(&a_tf);
            DispatchResult::Executed(CaseExecution {
                inputs,
                outputs: vec![
                    named("output_0", tenferro_to_oracle(&q_tf, 2)),
                    named("output_1", tenferro_to_oracle(&r_tf, 2)),
                ],
            })
        }
        "eigh" => {
            let a_tf = hermitian_wrapper_tenferro(&oracle_to_tenferro(&a, 2));
            let (values_tf, vectors_tf) = eigh(&a_tf);
            DispatchResult::Executed(CaseExecution {
                inputs,
                outputs: vec![
                    named("values", tenferro_to_oracle(&values_tf, 1)),
                    named("vectors", tenferro_to_oracle(&vectors_tf, 2)),
                ],
            })
        }
        "cholesky" => {
            let a_tf = hermitian_wrapper_tenferro(&oracle_to_tenferro(&a, 2));
            let upper = bool_kwarg(&case.op_kwargs, "upper")?.unwrap_or(false);
            let factor_tf = if upper {
                swap_tenferro_matrix_axes(&cholesky(&a_tf))
            } else {
                cholesky(&a_tf)
            };
            single_output(inputs, "value", tenferro_to_oracle(&factor_tf, 2))
        }
        "solve" => {
            let b = required_input(&inputs, "b", case)?.clone();
            let rhs_matrix_rank = rhs_matrix_rank(&a, &b);
            let a_tf = oracle_to_tenferro(&a, 2);
            let b_tf = oracle_to_tenferro(&b, rhs_matrix_rank);
            let solution_tf = solve(&a_tf, &b_tf);
            single_output(
                inputs,
                "value",
                tenferro_to_oracle(&solution_tf, rhs_matrix_rank),
            )
        }
        "solve_ex" => {
            let b = required_input(&inputs, "b", case)?.clone();
            let rhs_matrix_rank = rhs_matrix_rank(&a, &b);
            let a_tf = oracle_to_tenferro(&a, 2);
            let b_tf = oracle_to_tenferro(&b, rhs_matrix_rank);
            let solution_tf = solve(&a_tf, &b_tf);
            single_output(
                inputs,
                "output_0",
                tenferro_to_oracle(&solution_tf, rhs_matrix_rank),
            )
        }
        "solve_triangular" => {
            let b = required_input(&inputs, "b", case)?.clone();
            let rhs_matrix_rank = rhs_matrix_rank(&a, &b);
            let left_side = bool_kwarg(&case.op_kwargs, "left")?.unwrap_or(true);
            let lower = !bool_kwarg(&case.op_kwargs, "upper")?.unwrap_or(false);
            let unit_diagonal = bool_kwarg(&case.op_kwargs, "unitriangular")?.unwrap_or(false);
            let a_tf = oracle_to_tenferro(&a, 2);
            let b_tf = oracle_to_tenferro(&b, rhs_matrix_rank);
            let solution_tf =
                triangular_solve(&a_tf, &b_tf, left_side, lower, false, unit_diagonal);
            single_output(
                inputs,
                "value",
                tenferro_to_oracle(&solution_tf, rhs_matrix_rank),
            )
        }
        other => DispatchResult::SkippedUnimplemented(other.to_string()),
    };

    Ok(result)
}

fn decode_inputs(case: &CaseRecord) -> Result<BTreeMap<String, TracedTensor>, String> {
    let mut inputs = BTreeMap::new();
    for (name, tensor_data) in &case.inputs {
        let tensor = try_decode_tensor(tensor_data)?
            .ok_or_else(|| format!("{}: input {name} is not float64", case.case_id))?;
        inputs.insert(name.clone(), TracedTensor::from_tensor(tensor));
    }
    Ok(inputs)
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

fn single_output(
    inputs: BTreeMap<String, TracedTensor>,
    name: &str,
    tensor: TracedTensor,
) -> DispatchResult {
    DispatchResult::Executed(CaseExecution {
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

fn replay_enabled_op(op: &str) -> bool {
    matches!(
        op,
        // unary elementwise
        "abs" | "neg"
            | "exp"
            | "expm1"
            | "log"
            | "log1p"
            | "sin"
            | "cos"
            | "tanh"
            | "sqrt"
            | "rsqrt"
            | "sign"
            | "sgn"
            | "conj"
            | "conj_physical"
            // binary elementwise
            | "add"
            | "__radd__"
            | "mul"
            | "__rmul__"
            | "sub"
            | "rsub"
            | "__rsub__"
            | "div_no_rounding_mode"
            | "true_divide"
            | "__rdiv__"
            | "pow"
            | "__rpow__"
            | "float_power"
            // reduction
            | "sum"
            // linalg
            | "svd"
            | "qr"
            | "eigh"
            | "cholesky"
            | "solve"
            | "solve_ex"
            | "solve_triangular"
    )
}

fn alpha_kwarg(case: &CaseRecord) -> Result<f64, String> {
    Ok(number_kwarg(&case.op_kwargs, "alpha")?.unwrap_or(1.0))
}

fn sum_axes(case: &CaseRecord, rank: usize) -> Result<Vec<usize>, String> {
    if rank == 0 {
        return Ok(Vec::new());
    }
    let Some(value) = case
        .op_kwargs
        .as_object()
        .and_then(|kwargs| kwargs.get("dim"))
    else {
        return Ok((0..rank).collect());
    };

    let mut axes = match value {
        serde_json::Value::Array(items) => {
            let mut axes = Vec::with_capacity(items.len());
            for item in items {
                axes.push(normalize_axis(as_i64(item)?, rank)?);
            }
            axes
        }
        _ => vec![normalize_axis(as_i64(value)?, rank)?],
    };

    axes.sort_unstable();
    axes.dedup();
    Ok(axes)
}

fn keepdim_shape(shape: &[usize], axes: &[usize]) -> Vec<usize> {
    let mut kept = shape.to_vec();
    for &axis in axes {
        kept[axis] = 1;
    }
    kept
}

fn oracle_to_tenferro(tensor: &TracedTensor, matrix_rank: usize) -> TracedTensor {
    if tensor.shape.len() <= matrix_rank {
        return tensor.clone();
    }
    let rank = tensor.shape.len();
    let split = rank - matrix_rank;
    let mut perm = Vec::with_capacity(rank);
    perm.extend(split..rank);
    perm.extend(0..split);
    tensor.transpose(&perm)
}

fn tenferro_to_oracle(tensor: &TracedTensor, matrix_rank: usize) -> TracedTensor {
    if tensor.shape.len() <= matrix_rank {
        return tensor.clone();
    }
    let rank = tensor.shape.len();
    let mut perm = Vec::with_capacity(rank);
    perm.extend(matrix_rank..rank);
    perm.extend(0..matrix_rank);
    tensor.transpose(&perm)
}

fn swap_tenferro_matrix_axes(tensor: &TracedTensor) -> TracedTensor {
    if tensor.shape.len() < 2 {
        return tensor.clone();
    }
    let mut perm: Vec<usize> = (0..tensor.shape.len()).collect();
    perm.swap(0, 1);
    tensor.transpose(&perm)
}

fn hermitian_wrapper_tenferro(tensor: &TracedTensor) -> TracedTensor {
    let tensor_h = swap_tenferro_matrix_axes(&tensor.conj());
    tensor + &tensor_h
}

fn rhs_matrix_rank(a: &TracedTensor, b: &TracedTensor) -> usize {
    if b.shape.len() + 1 == a.shape.len() {
        1
    } else {
        2
    }
}

fn bool_kwarg(value: &serde_json::Value, key: &str) -> Result<Option<bool>, String> {
    match value.as_object().and_then(|kwargs| kwargs.get(key)) {
        Some(serde_json::Value::Bool(flag)) => Ok(Some(*flag)),
        Some(other) => Err(format!("expected boolean kwarg {key}, got {other}")),
        None => Ok(None),
    }
}

fn number_kwarg(value: &serde_json::Value, key: &str) -> Result<Option<f64>, String> {
    match value.as_object().and_then(|kwargs| kwargs.get(key)) {
        Some(other) => Ok(Some(as_f64(other)?)),
        None => Ok(None),
    }
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize, String> {
    let rank_i64 = rank as i64;
    let normalized = if axis < 0 { rank_i64 + axis } else { axis };
    if !(0..rank_i64).contains(&normalized) {
        return Err(format!("axis {axis} is out of bounds for rank {rank}"));
    }
    Ok(normalized as usize)
}

fn as_i64(value: &serde_json::Value) -> Result<i64, String> {
    if let Some(number) = value.as_i64() {
        Ok(number)
    } else {
        Err(format!("expected integer JSON value, got {value}"))
    }
}

fn as_f64(value: &serde_json::Value) -> Result<f64, String> {
    if let Some(number) = value.as_f64() {
        Ok(number)
    } else {
        Err(format!("expected numeric JSON value, got {value}"))
    }
}
