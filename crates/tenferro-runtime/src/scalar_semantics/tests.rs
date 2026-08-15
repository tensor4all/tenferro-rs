use super::{
    bool_from_real_for_op, dynamic_truncate_size, round_real_to_i32_for_op, round_real_to_i64,
    round_real_to_i64_for_op, scalar_host_value,
};
use crate::{Error, ErrorPhase};
use tenferro_tensor::{DType, ErrorKind, Tensor, TypedTensor};

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

    // Issue #1685: exactly 2^63 passes the old `> i64::MAX as f64`
    // guard (since `i64::MAX as f64 == 2^63`) and `as i64` saturated.
    let err = round_real_to_i64_for_op("test", 9_223_372_036_854_775_808.0).unwrap_err();
    assert!(
        err.to_string().contains("out of i64 range"),
        "exactly 2^63 must be rejected, got {err:?}"
    );
    assert_eq!(
        round_real_to_i64_for_op("test", 9_223_372_036_854_774_784.0).unwrap(),
        i64::MAX - 1023,
        "2^63 - 1024 (largest f64 below 2^63) must convert exactly"
    );
    assert_eq!(
        round_real_to_i64_for_op("test", -9_223_372_036_854_775_808.0).unwrap(),
        i64::MIN,
        "-2^63 == i64::MIN must remain valid"
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
fn dynamic_truncate_i64_routing_clamps_without_lossy_float_conversion() {
    assert_eq!(dynamic_truncate_size(&i64_scalar(-1), 4).unwrap(), 0);
    assert_eq!(dynamic_truncate_size(&i64_scalar(9), 4).unwrap(), 4);

    #[cfg(target_pointer_width = "64")]
    {
        const ABOVE_F64_INTEGER_PRECISION: i64 = (1_i64 << 53) + 1;
        let axis_extent = usize::try_from(ABOVE_F64_INTEGER_PRECISION + 1).unwrap();
        assert_eq!(
            dynamic_truncate_size(&i64_scalar(ABOVE_F64_INTEGER_PRECISION), axis_extent).unwrap(),
            usize::try_from(ABOVE_F64_INTEGER_PRECISION).unwrap()
        );
    }
}

#[test]
fn dynamic_truncate_size_rejects_non_scalar_or_wrong_dtype() {
    let vector = Tensor::F64(TypedTensor::from_vec_col_major(vec![1], vec![1.0]).unwrap());
    let err = dynamic_truncate_size(&vector, 4).unwrap_err();
    assert!(
        err.to_string().contains("scalar"),
        "expected scalar-shape error, got {err:?}"
    );

    let bool_scalar = Tensor::Bool(TypedTensor::from_vec_col_major(vec![], vec![true]).unwrap());
    let err = dynamic_truncate_size(&bool_scalar, 4).unwrap_err();
    assert!(matches!(
        &err,
        Error::Unsupported {
            op: "dynamic_truncate",
            phase: ErrorPhase::Execution,
            ..
        }
    ));
    assert_eq!(err.kind(), ErrorKind::Unsupported);
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
