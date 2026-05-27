use super::*;

#[test]
fn promote_same_returns_same() {
    assert_eq!(promote_dtype(DType::F64, DType::F64), DType::F64);
    assert_eq!(promote_dtype(DType::C64, DType::C64), DType::C64);
    assert_eq!(promote_dtype(DType::I64, DType::I64), DType::I64);
}

#[test]
fn promote_i64_to_float() {
    assert_eq!(promote_dtype(DType::I64, DType::F64), DType::F64);
    assert_eq!(promote_dtype(DType::F64, DType::I64), DType::F64);
    assert_eq!(promote_dtype(DType::I64, DType::F32), DType::F64);
    assert_eq!(promote_dtype(DType::F32, DType::I64), DType::F64);
}

#[test]
fn promote_i64_to_complex() {
    assert_eq!(promote_dtype(DType::I64, DType::C64), DType::C64);
    assert_eq!(promote_dtype(DType::C64, DType::I64), DType::C64);
    assert_eq!(promote_dtype(DType::I64, DType::C32), DType::C64);
    assert_eq!(promote_dtype(DType::C32, DType::I64), DType::C64);
}

#[test]
fn promote_float_to_wider_float() {
    assert_eq!(promote_dtype(DType::F32, DType::F64), DType::F64);
    assert_eq!(promote_dtype(DType::F64, DType::F32), DType::F64);
}

#[test]
fn promote_float_to_complex() {
    assert_eq!(promote_dtype(DType::F32, DType::C32), DType::C32);
    assert_eq!(promote_dtype(DType::F64, DType::C64), DType::C64);
    assert_eq!(promote_dtype(DType::F64, DType::C32), DType::C64);
    assert_eq!(promote_dtype(DType::F32, DType::C64), DType::C64);
}

#[test]
fn promote_complex_to_wider_complex() {
    assert_eq!(promote_dtype(DType::C32, DType::C64), DType::C64);
    assert_eq!(promote_dtype(DType::C64, DType::C32), DType::C64);
}

#[test]
fn promote_dtype_div_like_i64_to_f64() {
    assert_eq!(promote_dtype_div_like(DType::I64, DType::I64), DType::F64);
    assert_eq!(promote_dtype_div_like(DType::F64, DType::F64), DType::F64);
    assert_eq!(promote_dtype_div_like(DType::I64, DType::F64), DType::F64);
}

#[test]
fn promote_dtypes_fold() {
    assert_eq!(
        promote_dtypes([DType::I64, DType::F32, DType::C64]),
        DType::C64
    );
    assert_eq!(promote_dtypes([DType::F32, DType::F64]), DType::F64);
    assert_eq!(promote_dtypes([]), DType::F64); // empty -> F64 default
}

#[test]
fn promotion_rank_ordering() {
    assert!(promotion_rank(DType::I64) < promotion_rank(DType::F32));
    assert!(promotion_rank(DType::F32) < promotion_rank(DType::F64));
    assert!(promotion_rank(DType::F64) < promotion_rank(DType::C32));
    assert!(promotion_rank(DType::C32) < promotion_rank(DType::C64));
}
