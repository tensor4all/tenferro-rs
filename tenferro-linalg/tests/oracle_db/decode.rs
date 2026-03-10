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
        let Value::Number(number) = value else {
            return Err("expected float64 tensor payload to contain JSON numbers".to_string());
        };
        flat.push(
            number
                .as_f64()
                .ok_or_else(|| "failed to decode float64 value".to_string())?,
        );
    }

    let stored = Tensor::from_slice(&flat, &encoded.shape, MemoryOrder::RowMajor)
        .map_err(|err| format!("failed to decode tensor: {err}"))?;
    move_core_dims_to_front(stored, core_rank)
}
