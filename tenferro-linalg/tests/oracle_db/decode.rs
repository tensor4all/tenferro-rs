use std::collections::BTreeMap;

use num_complex::{Complex32, Complex64};
use num_traits::{Float, ToPrimitive, Zero};
use serde_json::Value;
use tenferro_linalg_prims::LinalgScalar;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::db::DbTensor;

pub trait OracleDbScalar: LinalgScalar + Copy {
    const ORACLE_DTYPE: &'static str;

    fn decode_json_value(value: &Value) -> Result<Self, String>;
}

fn decode_json_real<R: Float + ToPrimitive + num_traits::NumCast>(
    value: &Value,
) -> Result<R, String> {
    match value {
        Value::Number(num) => {
            let as_f64 = num
                .as_f64()
                .ok_or_else(|| "failed to decode numeric oracle payload".to_string())?;
            num_traits::NumCast::from(as_f64)
                .ok_or_else(|| format!("failed to cast oracle payload {as_f64}"))
        }
        Value::String(text) if text == "NaN" => Ok(R::nan()),
        Value::String(text) if text == "Infinity" => Ok(R::infinity()),
        Value::String(text) if text == "-Infinity" => Ok(R::neg_infinity()),
        _ => Err("expected numeric oracle payload".to_string()),
    }
}

impl OracleDbScalar for f64 {
    const ORACLE_DTYPE: &'static str = "float64";

    fn decode_json_value(value: &Value) -> Result<Self, String> {
        decode_json_real(value)
    }
}

impl OracleDbScalar for f32 {
    const ORACLE_DTYPE: &'static str = "float32";

    fn decode_json_value(value: &Value) -> Result<Self, String> {
        decode_json_real(value)
    }
}

impl OracleDbScalar for Complex64 {
    const ORACLE_DTYPE: &'static str = "complex128";

    fn decode_json_value(value: &Value) -> Result<Self, String> {
        let Value::Array(parts) = value else {
            return Err("expected [real, imag] complex payload".to_string());
        };
        if parts.len() != 2 {
            return Err(format!(
                "expected 2 complex payload entries, got {}",
                parts.len()
            ));
        }
        Ok(Self::from_parts(
            decode_json_real(&parts[0])?,
            decode_json_real(&parts[1])?,
        ))
    }
}

impl OracleDbScalar for Complex32 {
    const ORACLE_DTYPE: &'static str = "complex64";

    fn decode_json_value(value: &Value) -> Result<Self, String> {
        let Value::Array(parts) = value else {
            return Err("expected [real, imag] complex payload".to_string());
        };
        if parts.len() != 2 {
            return Err(format!(
                "expected 2 complex payload entries, got {}",
                parts.len()
            ));
        }
        Ok(Self::from_parts(
            decode_json_real(&parts[0])?,
            decode_json_real(&parts[1])?,
        ))
    }
}

fn move_core_dims_to_front<T: tenferro_algebra::Scalar>(
    tensor: Tensor<T>,
    core_rank: usize,
) -> Result<Tensor<T>, String> {
    let rank = tensor.ndim();
    if core_rank == 0 || rank <= core_rank {
        return Ok(tensor);
    }
    if core_rank > rank {
        return Err(format!(
            "core rank {core_rank} exceeds tensor rank {rank} for dims {:?}",
            tensor.dims()
        ));
    }
    let batch_rank = rank - core_rank;
    let mut perm = Vec::with_capacity(rank);
    perm.extend(batch_rank..rank);
    perm.extend(0..batch_rank);
    tensor
        .permute(&perm)
        .map_err(|err| format!("failed to permute tensor {:?}: {err}", tensor.dims()))
}

pub fn decode_tensor_with_core_rank<T: OracleDbScalar>(
    encoded: &DbTensor,
    core_rank: usize,
) -> Result<Tensor<T>, String> {
    if encoded.dtype != T::ORACLE_DTYPE {
        return Err(format!("unsupported tensor dtype {}", encoded.dtype));
    }
    if encoded.order != "row_major" {
        return Err(format!("unsupported tensor order {}", encoded.order));
    }
    let mut flat = Vec::with_capacity(encoded.data.len());
    for value in &encoded.data {
        flat.push(T::decode_json_value(value)?);
    }
    let stored = Tensor::from_slice(&flat, &encoded.shape, MemoryOrder::RowMajor)
        .map_err(|err| format!("failed to decode tensor: {err}"))?;
    move_core_dims_to_front(stored, core_rank)
}

pub fn decode_f64_tensor_with_core_rank(
    encoded: &DbTensor,
    core_rank: usize,
) -> Result<Tensor<f64>, String> {
    decode_tensor_with_core_rank(encoded, core_rank)
}

pub fn tensor_data_col_major<T: tenferro_algebra::Scalar + Copy>(tensor: &Tensor<T>) -> Vec<T> {
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.dims().iter().product::<usize>();
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

pub fn tensor_from_col_major<T: tenferro_algebra::Scalar>(
    data: Vec<T>,
    dims: &[usize],
) -> Tensor<T> {
    Tensor::from_slice(&data, dims, MemoryOrder::ColumnMajor).unwrap()
}

pub fn batch_count(batch_dims: &[usize]) -> usize {
    if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product()
    }
}

pub fn elementwise_abs_jvp<T: OracleDbScalar>(
    primal: &Tensor<T>,
    tangent: &Tensor<T>,
) -> Tensor<T::Real> {
    let primal_data = tensor_data_col_major(primal);
    let tangent_data = tensor_data_col_major(tangent);
    let data: Vec<T::Real> = primal_data
        .iter()
        .zip(tangent_data.iter())
        .map(|(z, dz)| {
            let mag = z.abs_real();
            if mag == T::Real::zero() {
                T::Real::zero()
            } else {
                ((*z).conj() * *dz).real_part() / mag
            }
        })
        .collect();
    tensor_from_col_major(data, primal.dims())
}

pub fn elementwise_abs_vjp<T: OracleDbScalar>(
    primal: &Tensor<T>,
    cotangent: &Tensor<T::Real>,
) -> Tensor<T> {
    let primal_data = tensor_data_col_major(primal);
    let cotangent_data = tensor_data_col_major(cotangent);
    let data: Vec<T> = primal_data
        .iter()
        .zip(cotangent_data.iter())
        .map(|(z, co)| {
            let mag = z.abs_real();
            if mag == T::Real::zero() {
                T::zero()
            } else {
                *z * T::from_real(*co / mag)
            }
        })
        .collect();
    tensor_from_col_major(data, primal.dims())
}

pub fn tensor_add<T: OracleDbScalar>(left: &Tensor<T>, right: &Tensor<T>) -> Tensor<T> {
    let left_data = tensor_data_col_major(left);
    let right_data = tensor_data_col_major(right);
    let data: Vec<T> = left_data
        .iter()
        .zip(right_data.iter())
        .map(|(a, b)| *a + *b)
        .collect();
    tensor_from_col_major(data, left.dims())
}

pub fn batched_matmul<T: OracleDbScalar>(
    left: &Tensor<T>,
    right: &Tensor<T>,
) -> Result<Tensor<T>, String> {
    let left_dims = left.dims();
    let right_dims = right.dims();
    if left_dims.len() < 2 || right_dims.len() < 2 {
        return Err("batched_matmul requires rank >= 2 tensors".to_string());
    }
    let m = left_dims[0];
    let k = left_dims[1];
    if right_dims[0] != k {
        return Err(format!(
            "matmul dimension mismatch: left {:?}, right {:?}",
            left_dims, right_dims
        ));
    }
    let n = right_dims[1];
    let left_batch = &left_dims[2..];
    let right_batch = &right_dims[2..];
    if left_batch != right_batch {
        return Err(format!(
            "matmul batch mismatch: left {:?}, right {:?}",
            left_dims, right_dims
        ));
    }

    let bc = batch_count(left_batch);
    let left_data = tensor_data_col_major(left);
    let right_data = tensor_data_col_major(right);
    let mut out = vec![T::zero(); m * n * bc];

    for batch in 0..bc {
        let left_offset = batch * m * k;
        let right_offset = batch * k * n;
        let out_offset = batch * m * n;
        for j in 0..n {
            for p in 0..k {
                let right_val = right_data[right_offset + p + j * k];
                for i in 0..m {
                    let out_index = out_offset + i + j * m;
                    out[out_index] =
                        out[out_index] + left_data[left_offset + i + p * m] * right_val;
                }
            }
        }
    }

    let mut dims = vec![m, n];
    dims.extend_from_slice(left_batch);
    Ok(tensor_from_col_major(out, &dims))
}

pub fn batched_transpose<T: OracleDbScalar>(tensor: &Tensor<T>) -> Result<Tensor<T>, String> {
    let dims = tensor.dims();
    if dims.len() < 2 {
        return Err("batched_transpose requires rank >= 2 tensor".to_string());
    }
    let m = dims[0];
    let n = dims[1];
    let batch_dims = &dims[2..];
    let bc = batch_count(batch_dims);
    let data = tensor_data_col_major(tensor);
    let mut out = vec![T::zero(); data.len()];

    for batch in 0..bc {
        let offset = batch * m * n;
        for j in 0..n {
            for i in 0..m {
                out[batch * n * m + j + i * n] = data[offset + i + j * m];
            }
        }
    }

    let mut out_dims = vec![n, m];
    out_dims.extend_from_slice(batch_dims);
    Ok(tensor_from_col_major(out, &out_dims))
}

pub fn batched_adjoint_transpose<T: OracleDbScalar>(
    tensor: &Tensor<T>,
) -> Result<Tensor<T>, String> {
    let dims = tensor.dims();
    if dims.len() < 2 {
        return Err("batched_adjoint_transpose requires rank >= 2 tensor".to_string());
    }
    let m = dims[0];
    let n = dims[1];
    let batch_dims = &dims[2..];
    let bc = batch_count(batch_dims);
    let data = tensor_data_col_major(tensor);
    let mut out = vec![T::zero(); data.len()];

    for batch in 0..bc {
        let offset = batch * m * n;
        for j in 0..n {
            for i in 0..m {
                out[batch * n * m + j + i * n] = data[offset + i + j * m].conj();
            }
        }
    }

    let mut out_dims = vec![n, m];
    out_dims.extend_from_slice(batch_dims);
    Ok(tensor_from_col_major(out, &out_dims))
}

pub fn inner_product(left: &Tensor<f64>, right: &Tensor<f64>) -> f64 {
    let left_data = tensor_data_col_major(left);
    let right_data = tensor_data_col_major(right);
    left_data
        .iter()
        .zip(right_data.iter())
        .map(|(a, b)| a * b)
        .sum()
}

pub fn inner_product_typed<T: OracleDbScalar>(
    left: &Tensor<T>,
    right: &Tensor<T>,
) -> Result<f64, String> {
    if left.dims() != right.dims() {
        return Err(format!(
            "typed inner product shape mismatch: left {:?}, right {:?}",
            left.dims(),
            right.dims()
        ));
    }
    let left_data = tensor_data_col_major(left);
    let right_data = tensor_data_col_major(right);
    let mut acc = 0.0f64;
    for (lhs, rhs) in left_data.iter().zip(right_data.iter()) {
        acc += ((*lhs).conj() * *rhs)
            .real_part()
            .to_f64()
            .ok_or_else(|| "failed to convert typed inner product to f64".to_string())?;
    }
    Ok(acc)
}

pub fn compare_tensors(
    label: &str,
    expected: &Tensor<f64>,
    actual: &Tensor<f64>,
    rtol: f64,
    atol: f64,
) -> Result<(), String> {
    if expected.dims() != actual.dims() {
        return Err(format!(
            "{label}: shape mismatch expected {:?} got {:?}",
            expected.dims(),
            actual.dims()
        ));
    }
    let expected_data = tensor_data_col_major(expected);
    let actual_data = tensor_data_col_major(actual);
    for (index, (exp, act)) in expected_data.iter().zip(actual_data.iter()).enumerate() {
        let allowed = atol + rtol * exp.abs();
        let diff = (exp - act).abs();
        if diff > allowed {
            return Err(format!(
                "{label}: mismatch at index {index}: expected={exp}, actual={act}, diff={diff}, allowed={allowed}"
            ));
        }
    }
    Ok(())
}

pub fn compare_tensors_typed<T: OracleDbScalar>(
    label: &str,
    expected: &Tensor<T>,
    actual: &Tensor<T>,
    rtol: f64,
    atol: f64,
) -> Result<(), String> {
    if expected.dims() != actual.dims() {
        return Err(format!(
            "{label}: shape mismatch expected {:?} got {:?}",
            expected.dims(),
            actual.dims()
        ));
    }
    let expected_data = tensor_data_col_major(expected);
    let actual_data = tensor_data_col_major(actual);
    for (index, (exp, act)) in expected_data.iter().zip(actual_data.iter()).enumerate() {
        let exp_abs = exp
            .abs_real()
            .to_f64()
            .ok_or_else(|| format!("{label}: failed to convert expected magnitude"))?;
        let act_diff = (*exp - *act)
            .abs_real()
            .to_f64()
            .ok_or_else(|| format!("{label}: failed to convert actual difference"))?;
        let allowed = atol + rtol * exp_abs;
        if act_diff > allowed {
            return Err(format!(
                "{label}: mismatch at index {index}: diff={act_diff}, allowed={allowed}, expected={exp:?}, actual={act:?}"
            ));
        }
    }
    Ok(())
}

pub fn compare_tensor_maps(
    label: &str,
    expected: &BTreeMap<String, Tensor<f64>>,
    actual: &BTreeMap<String, Tensor<f64>>,
    rtol: f64,
    atol: f64,
) -> Result<(), String> {
    if expected.keys().collect::<Vec<_>>() != actual.keys().collect::<Vec<_>>() {
        return Err(format!(
            "{label}: key mismatch expected {:?} got {:?}",
            expected.keys().collect::<Vec<_>>(),
            actual.keys().collect::<Vec<_>>()
        ));
    }
    for (name, expected_tensor) in expected {
        let actual_tensor = actual.get(name).unwrap();
        compare_tensors(
            &format!("{label}.{name}"),
            expected_tensor,
            actual_tensor,
            rtol,
            atol,
        )?;
    }
    Ok(())
}

pub fn compare_tensor_maps_typed<T: OracleDbScalar>(
    label: &str,
    expected: &BTreeMap<String, Tensor<T>>,
    actual: &BTreeMap<String, Tensor<T>>,
    rtol: f64,
    atol: f64,
) -> Result<(), String> {
    if expected.keys().collect::<Vec<_>>() != actual.keys().collect::<Vec<_>>() {
        return Err(format!(
            "{label}: key mismatch expected {:?} got {:?}",
            expected.keys().collect::<Vec<_>>(),
            actual.keys().collect::<Vec<_>>()
        ));
    }
    for (name, expected_tensor) in expected {
        let actual_tensor = actual.get(name).unwrap();
        compare_tensors_typed(
            &format!("{label}.{name}"),
            expected_tensor,
            actual_tensor,
            rtol,
            atol,
        )?;
    }
    Ok(())
}

pub fn tensor_map_inner_product(
    left: &BTreeMap<String, Tensor<f64>>,
    right: &BTreeMap<String, Tensor<f64>>,
) -> Result<f64, String> {
    if left.keys().collect::<Vec<_>>() != right.keys().collect::<Vec<_>>() {
        return Err("tensor-map inner product key mismatch".to_string());
    }
    Ok(left
        .iter()
        .map(|(name, tensor)| inner_product(tensor, right.get(name).unwrap()))
        .sum())
}

pub fn tensor_map_inner_product_typed<T: OracleDbScalar>(
    left: &BTreeMap<String, Tensor<T>>,
    right: &BTreeMap<String, Tensor<T>>,
) -> Result<f64, String> {
    if left.keys().collect::<Vec<_>>() != right.keys().collect::<Vec<_>>() {
        return Err("tensor-map inner product key mismatch".to_string());
    }
    let mut acc = 0.0f64;
    for (name, tensor) in left {
        acc += inner_product_typed(tensor, right.get(name).unwrap())?;
    }
    Ok(acc)
}
