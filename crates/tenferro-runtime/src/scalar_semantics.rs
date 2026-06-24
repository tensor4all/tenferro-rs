use tenferro_tensor::{DType, Tensor};

use crate::error::{Error, Result};

fn finite_real_scalar(op: &'static str, value: f64) -> Result<f64> {
    if !value.is_finite() {
        return Err(Error::InvalidGraphBuild {
            op,
            message: format!("real scalar value must be finite, got {value}"),
        });
    }
    Ok(value)
}

pub(crate) fn round_real_to_i64(value: f64) -> Result<i64> {
    round_real_to_i64_for_op("scale_real", value)
}

pub(crate) fn round_real_to_i64_for_op(op: &'static str, value: f64) -> Result<i64> {
    let rounded = finite_real_scalar(op, value)?.round();
    if rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return Err(Error::InvalidGraphBuild {
            op,
            message: format!("rounded real scalar {rounded} is out of i64 range"),
        });
    }
    Ok(rounded as i64)
}

pub(crate) fn round_real_to_i32_for_op(op: &'static str, value: f64) -> Result<i32> {
    let rounded = round_real_to_i64_for_op(op, value)?;
    i32::try_from(rounded).map_err(|_| Error::InvalidGraphBuild {
        op,
        message: format!("rounded real scalar {rounded} is out of i32 range"),
    })
}

pub(crate) fn bool_from_real_for_op(op: &'static str, value: f64) -> Result<bool> {
    Ok(finite_real_scalar(op, value)? != 0.0)
}

pub fn dynamic_truncate_size(size_tensor: &Tensor, axis_extent: usize) -> Result<usize> {
    let value = scalar_size_value(size_tensor)?;
    let rounded = finite_real_scalar("DynamicTruncate", value)?.round();
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
        Tensor::F64(inner) => scalar_host_value(inner.host_data()?, DType::F64),
        Tensor::F32(inner) => Ok(scalar_host_value(inner.host_data()?, DType::F32)? as f64),
        Tensor::I64(inner) => Ok(scalar_host_value(inner.host_data()?, DType::I64)? as f64),
        _ => Err(Error::Internal(
            "DynamicTruncate size must be an f32, f64, or i64 scalar".into(),
        )),
    }
}

fn scalar_host_value<T: Copy>(data: &[T], dtype: DType) -> Result<T> {
    data.first().copied().ok_or_else(|| {
        Error::Internal(format!(
            "DynamicTruncate size scalar had empty {dtype:?} host buffer"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        bool_from_real_for_op, dynamic_truncate_size, round_real_to_i32_for_op, round_real_to_i64,
        round_real_to_i64_for_op, scalar_host_value,
    };
    use tenferro_tensor::{DType, Tensor, TypedTensor};

    fn f64_scalar(value: f64) -> Tensor {
        Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![value]).unwrap())
    }

    fn f32_scalar(value: f32) -> Tensor {
        Tensor::F32(TypedTensor::from_vec_col_major(vec![], vec![value]).unwrap())
    }

    fn i64_scalar(value: i64) -> Tensor {
        Tensor::I64(TypedTensor::from_vec_col_major(vec![], vec![value]).unwrap())
    }

    #[test]
    fn real_scalar_conversions_round_and_validate_ranges() {
        assert_eq!(round_real_to_i64_for_op("test", 2.6).unwrap(), 3);
        assert_eq!(round_real_to_i64_for_op("test", -2.4).unwrap(), -2);
        assert_eq!(round_real_to_i32_for_op("test", 42.4).unwrap(), 42);

        let err = round_real_to_i64_for_op("test", i64::MAX as f64 * 2.0).unwrap_err();
        assert!(
            err.to_string().contains("out of i64 range"),
            "expected i64 range error, got {err:?}"
        );

        let err = round_real_to_i32_for_op("test", i32::MAX as f64 + 1024.0).unwrap_err();
        assert!(
            err.to_string().contains("out of i32 range"),
            "expected i32 range error, got {err:?}"
        );
    }

    #[test]
    fn bool_scalar_conversion_uses_nonzero_finite_values() {
        assert!(!bool_from_real_for_op("test", 0.0).unwrap());
        assert!(bool_from_real_for_op("test", -0.5).unwrap());
        assert!(bool_from_real_for_op("test", 1.0).unwrap());
        assert!(bool_from_real_for_op("test", f64::NAN).is_err());
    }

    #[test]
    fn dynamic_truncate_size_clamps_supported_scalar_dtypes() {
        assert_eq!(dynamic_truncate_size(&f64_scalar(2.6), 4).unwrap(), 3);
        assert_eq!(dynamic_truncate_size(&f32_scalar(-2.0), 4).unwrap(), 0);
        assert_eq!(dynamic_truncate_size(&i64_scalar(9), 4).unwrap(), 4);
    }

    #[test]
    fn dynamic_truncate_size_rejects_non_scalar_or_wrong_dtype() {
        let vector = Tensor::F64(TypedTensor::from_vec_col_major(vec![1], vec![1.0]).unwrap());
        let err = dynamic_truncate_size(&vector, 4).unwrap_err();
        assert!(
            err.to_string().contains("scalar"),
            "expected scalar-shape error, got {err:?}"
        );

        let bool_scalar =
            Tensor::Bool(TypedTensor::from_vec_col_major(vec![], vec![true]).unwrap());
        let err = dynamic_truncate_size(&bool_scalar, 4).unwrap_err();
        assert!(
            err.to_string().contains("f32, f64, or i64"),
            "expected dtype error, got {err:?}"
        );
    }

    #[test]
    fn dynamic_truncate_size_rejects_non_finite_scalars() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = dynamic_truncate_size(&f64_scalar(value), 4).unwrap_err();
            assert!(
                err.to_string().contains("finite"),
                "expected finite-value error, got {err:?}"
            );
        }
    }

    #[test]
    fn round_real_to_i64_rejects_non_finite_values() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(round_real_to_i64(value).is_err());
        }
    }

    #[test]
    fn scalar_host_value_rejects_empty_buffers() {
        let err = scalar_host_value::<f64>(&[], DType::F64).unwrap_err();
        assert!(
            err.to_string().contains("empty"),
            "expected empty-buffer error, got {err:?}"
        );
    }
}
