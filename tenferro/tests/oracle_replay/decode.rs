#![allow(dead_code)]

use std::collections::HashMap;

use num_complex::Complex64;
use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer};
use tenferro::{Tensor, TypedTensor};

#[derive(Deserialize)]
pub struct CaseRecord {
    pub case_id: String,
    pub op: String,
    pub dtype: String,
    pub family: String,
    pub expected_behavior: String,
    pub comparison: Comparison,
    pub inputs: HashMap<String, TensorData>,
    pub observable: Observable,
    #[serde(default)]
    pub op_args: Vec<OracleArg>,
    #[serde(default)]
    pub op_kwargs: serde_json::Value,
    pub probes: Vec<Probe>,
}

#[derive(Clone, Deserialize)]
#[serde(untagged)]
pub enum OracleArg {
    Null(()),
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<OracleArg>),
    Object(HashMap<String, OracleArg>),
}

#[derive(Clone, Deserialize)]
pub struct TensorData {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub order: String,
    #[serde(deserialize_with = "deserialize_tensor_numbers")]
    pub data: Vec<f64>,
}

#[derive(Default, Deserialize)]
pub struct Comparison {
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub reason_code: Option<String>,
    #[serde(default)]
    pub first_order: Option<Tolerance>,
    #[serde(default)]
    pub second_order: Option<Tolerance>,
}

#[derive(Clone, Deserialize)]
pub struct Tolerance {
    pub kind: String,
    pub rtol: f64,
    pub atol: f64,
}

#[derive(Deserialize)]
pub struct Observable {
    pub kind: String,
}

#[derive(Deserialize)]
pub struct Probe {
    pub probe_id: String,
    #[serde(default)]
    pub direction: HashMap<String, TensorData>,
    #[serde(default)]
    pub cotangent: HashMap<String, TensorData>,
    pub pytorch_ref: ReferenceData,
    #[serde(default)]
    pub fd_ref: Option<ReferenceData>,
}

#[derive(Default, Deserialize)]
pub struct ReferenceData {
    #[serde(default)]
    pub jvp: HashMap<String, TensorData>,
    #[serde(default)]
    pub vjp: HashMap<String, TensorData>,
    #[serde(default)]
    pub hvp: HashMap<String, TensorData>,
}

pub fn decode_tensor(td: &TensorData) -> Option<Tensor> {
    match try_decode_tensor(td) {
        Ok(Some(tensor)) => Some(tensor),
        Ok(None) | Err(_) => None,
    }
}

pub fn try_decode_tensor(td: &TensorData) -> Result<Option<Tensor>, String> {
    match td.dtype.as_str() {
        "float64" => {
            let data = tensor_data_as_col_major(td)?;
            Ok(Some(Tensor::F64(TypedTensor::from_vec_col_major(
                td.shape.clone(),
                data,
            ))))
        }
        "complex128" => {
            let data = complex_tensor_data_as_col_major(td)?;
            Ok(Some(Tensor::C64(TypedTensor::from_vec_col_major(
                td.shape.clone(),
                data,
            ))))
        }
        _ => Ok(None),
    }
}

pub fn tensor_data_as_col_major(td: &TensorData) -> Result<Vec<f64>, String> {
    if td.dtype != "float64" {
        return Err(format!("unsupported tensor dtype {}", td.dtype));
    }
    let total: usize = td.shape.iter().product();
    if td.data.len() != total {
        return Err(format!(
            "tensor data length {} does not match shape product {}",
            td.data.len(),
            total
        ));
    }
    match td.order.as_str() {
        "row_major" => Ok(row_major_to_column_major_blocks(&td.data, &td.shape, 1)),
        "col_major" => Ok(td.data.clone()),
        other => Err(format!("unsupported tensor storage order {other}")),
    }
}

pub fn complex_tensor_data_as_col_major(td: &TensorData) -> Result<Vec<Complex64>, String> {
    if td.dtype != "complex128" {
        return Err(format!("unsupported tensor dtype {}", td.dtype));
    }
    let total: usize = td.shape.iter().product();
    if td.data.len() != 2 * total {
        return Err(format!(
            "complex tensor data length {} does not match shape product {}",
            td.data.len(),
            total
        ));
    }

    let flat = match td.order.as_str() {
        "row_major" => row_major_to_column_major_blocks(&td.data, &td.shape, 2),
        "col_major" => td.data.clone(),
        other => return Err(format!("unsupported tensor storage order {other}")),
    };

    let mut out = Vec::with_capacity(total);
    for pair in flat.chunks_exact(2) {
        out.push(Complex64::new(pair[0], pair[1]));
    }
    Ok(out)
}

pub fn row_major_to_column_major(data: &[f64], shape: &[usize]) -> Vec<f64> {
    row_major_to_column_major_blocks(data, shape, 1)
}

pub fn row_major_to_column_major_blocks(data: &[f64], shape: &[usize], block: usize) -> Vec<f64> {
    let total: usize = shape.iter().product();
    if total == 0 {
        return Vec::new();
    }

    let rank = shape.len();
    let mut result = vec![0.0; total * block];
    let mut row_strides = vec![1usize; rank];
    let mut col_strides = vec![1usize; rank];

    for index in (0..rank.saturating_sub(1)).rev() {
        row_strides[index] = row_strides[index + 1] * shape[index + 1];
    }
    for index in 1..rank {
        col_strides[index] = col_strides[index - 1] * shape[index - 1];
    }

    for row_idx in 0..total {
        let mut remaining = row_idx;
        let mut col_idx = 0usize;
        for dim in 0..rank {
            let coord = remaining / row_strides[dim];
            remaining %= row_strides[dim];
            col_idx += coord * col_strides[dim];
        }
        let src = row_idx * block;
        let dst = col_idx * block;
        result[dst..dst + block].copy_from_slice(&data[src..src + block]);
    }

    result
}

fn deserialize_tensor_numbers<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    let Some(items) = value.as_array() else {
        return Err(D::Error::custom("tensor data must be a JSON array"));
    };

    let mut out = Vec::with_capacity(items.len());
    for item in items {
        if let Some(number) = item.as_f64() {
            out.push(number);
        } else if item.is_array() {
            let pair = item
                .as_array()
                .ok_or_else(|| D::Error::custom("complex tensor entry must be an array"))?;
            if pair.len() != 2 {
                return Err(D::Error::custom(
                    "complex tensor entries must be [re, im] pairs",
                ));
            }
            out.push(
                pair[0]
                    .as_f64()
                    .ok_or_else(|| D::Error::custom("complex real part must be numeric"))?,
            );
            out.push(
                pair[1]
                    .as_f64()
                    .ok_or_else(|| D::Error::custom("complex imaginary part must be numeric"))?,
            );
        } else {
            return Err(D::Error::custom(
                "tensor data elements must be numbers, null, or numeric tuples",
            ));
        }
    }
    Ok(out)
}
