use std::collections::BTreeMap;

use serde_json::Value;
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::db::DbTensor;

fn move_core_dims_to_front(tensor: Tensor<f64>, core_rank: usize) -> Result<Tensor<f64>, String> {
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

pub fn decode_f64_tensor_with_core_rank(
    encoded: &DbTensor,
    core_rank: usize,
) -> Result<Tensor<f64>, String> {
    if encoded.dtype != "float64" {
        return Err(format!("unsupported tensor dtype {}", encoded.dtype));
    }
    if encoded.order != "row_major" {
        return Err(format!("unsupported tensor order {}", encoded.order));
    }
    let mut flat = Vec::with_capacity(encoded.data.len());
    for value in &encoded.data {
        let number = match value {
            Value::Number(num) => num
                .as_f64()
                .ok_or_else(|| "failed to decode float64 value".to_string())?,
            _ => {
                return Err("expected float64 tensor payload to contain JSON numbers".to_string());
            }
        };
        flat.push(number);
    }
    let stored = Tensor::from_slice(&flat, &encoded.shape, MemoryOrder::RowMajor)
        .map_err(|err| format!("failed to decode tensor: {err}"))?;
    move_core_dims_to_front(stored, core_rank)
}

pub fn tensor_data_col_major(tensor: &Tensor<f64>) -> Vec<f64> {
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.dims().iter().product::<usize>();
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

pub fn tensor_from_col_major(data: Vec<f64>, dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(&data, dims, MemoryOrder::ColumnMajor).unwrap()
}

pub fn batch_count(batch_dims: &[usize]) -> usize {
    if batch_dims.is_empty() {
        1
    } else {
        batch_dims.iter().product()
    }
}

pub fn elementwise_sign_mul(primal: &Tensor<f64>, cotangent: &Tensor<f64>) -> Tensor<f64> {
    let primal_data = tensor_data_col_major(primal);
    let cotangent_data = tensor_data_col_major(cotangent);
    let data: Vec<f64> = primal_data
        .iter()
        .zip(cotangent_data.iter())
        .map(|(x, co)| if *x == 0.0 { 0.0 } else { x.signum() * co })
        .collect();
    tensor_from_col_major(data, primal.dims())
}

pub fn tensor_add(left: &Tensor<f64>, right: &Tensor<f64>) -> Tensor<f64> {
    let left_data = tensor_data_col_major(left);
    let right_data = tensor_data_col_major(right);
    let data: Vec<f64> = left_data
        .iter()
        .zip(right_data.iter())
        .map(|(a, b)| a + b)
        .collect();
    tensor_from_col_major(data, left.dims())
}

pub fn batched_matmul(left: &Tensor<f64>, right: &Tensor<f64>) -> Result<Tensor<f64>, String> {
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
    let mut out = vec![0.0; m * n * bc];

    for batch in 0..bc {
        let left_offset = batch * m * k;
        let right_offset = batch * k * n;
        let out_offset = batch * m * n;
        for j in 0..n {
            for p in 0..k {
                let right_val = right_data[right_offset + p + j * k];
                for i in 0..m {
                    out[out_offset + i + j * m] += left_data[left_offset + i + p * m] * right_val;
                }
            }
        }
    }

    let mut dims = vec![m, n];
    dims.extend_from_slice(left_batch);
    Ok(tensor_from_col_major(out, &dims))
}

pub fn batched_transpose(tensor: &Tensor<f64>) -> Result<Tensor<f64>, String> {
    let dims = tensor.dims();
    if dims.len() < 2 {
        return Err("batched_transpose requires rank >= 2 tensor".to_string());
    }
    let m = dims[0];
    let n = dims[1];
    let batch_dims = &dims[2..];
    let bc = batch_count(batch_dims);
    let data = tensor_data_col_major(tensor);
    let mut out = vec![0.0; data.len()];

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

pub fn inner_product(left: &Tensor<f64>, right: &Tensor<f64>) -> f64 {
    let left_data = tensor_data_col_major(left);
    let right_data = tensor_data_col_major(right);
    left_data
        .iter()
        .zip(right_data.iter())
        .map(|(a, b)| a * b)
        .sum()
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
