use tenferro_tensor::Tensor;

use crate::error::{Error, Result};

pub(crate) fn round_real_to_i64(value: f64) -> i64 {
    if value.is_finite() {
        value.round() as i64
    } else {
        0
    }
}

pub fn dynamic_truncate_size(size_tensor: &Tensor, axis_extent: usize) -> Result<usize> {
    let value = scalar_size_value(size_tensor)?;
    let rounded = if value.is_finite() {
        value.round()
    } else {
        0.0
    };
    Ok(rounded.max(0.0).min(axis_extent as f64) as usize)
}

fn scalar_size_value(size_tensor: &Tensor) -> Result<f64> {
    if !size_tensor.shape().is_empty() {
        return Err(Error::Internal(format!(
            "DynamicTruncate size must be an f32, f64, or i64 scalar, got shape {:?}",
            size_tensor.shape()
        )));
    }

    match size_tensor {
        Tensor::F64(inner) => Ok(inner.host_data()[0]),
        Tensor::F32(inner) => Ok(inner.host_data()[0] as f64),
        Tensor::I64(inner) => Ok(inner.host_data()[0] as f64),
        _ => Err(Error::Internal(
            "DynamicTruncate size must be an f32, f64, or i64 scalar".into(),
        )),
    }
}
