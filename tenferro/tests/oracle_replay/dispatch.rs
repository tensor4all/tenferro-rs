use std::collections::BTreeMap;

use tenferro::traced_tensor::TracedTensor;

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
            let axes = sum_axes(case, a.rank)?;
            let keepdim = bool_kwarg(&case.op_kwargs, "keepdim")?.unwrap_or(false);
            let reduced = a.reduce_sum(&axes);
            if keepdim {
                let kept_shape = keepdim_shape(&concrete_shape(&a)?, &axes);
                single_output(inputs, "value", reduced.reshape(&kept_shape))
            } else {
                single_output(inputs, "value", reduced)
            }
        }
        other => DispatchResult::SkippedUnimplemented(other.to_string()),
    };

    Ok(result)
}

fn decode_inputs(case: &CaseRecord) -> Result<BTreeMap<String, TracedTensor>, String> {
    let mut inputs = BTreeMap::new();
    for (name, tensor_data) in &case.inputs {
        let tensor = try_decode_tensor(tensor_data)?.ok_or_else(|| {
            format!(
                "{}: input {name} has unsupported dtype {}",
                case.case_id, tensor_data.dtype
            )
        })?;
        inputs.insert(
            name.clone(),
            TracedTensor::from_tensor_concrete_shape(tensor),
        );
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

fn concrete_shape(tensor: &TracedTensor) -> Result<Vec<usize>, String> {
    tensor
        .data
        .as_ref()
        .map(|data| data.shape().to_vec())
        .ok_or_else(|| "expected concrete traced tensor shape".to_string())
}

fn bool_kwarg(value: &serde_json::Value, key: &str) -> Result<Option<bool>, String> {
    match value.as_object().and_then(|kwargs| kwargs.get(key)) {
        Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Bool(flag)) => Ok(Some(*flag)),
        Some(other) => Err(format!("expected boolean kwarg {key}, got {other}")),
        None => Ok(None),
    }
}

fn number_kwarg(value: &serde_json::Value, key: &str) -> Result<Option<f64>, String> {
    match value.as_object().and_then(|kwargs| kwargs.get(key)) {
        Some(serde_json::Value::Null) => Ok(None),
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
