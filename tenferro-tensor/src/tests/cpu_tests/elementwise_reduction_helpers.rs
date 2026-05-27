use super::*;

#[test]
fn test_direct_elementwise_helpers_cover_f32_c32_and_error_paths() {
    let lhs_f32 = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![8.0f32, -2.0]));
    let rhs_f32 = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![2.0f32, 5.0]));
    let pred_bool = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![false, true]));
    let lower_f32 = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![-1.0f32, -1.0],
    ));
    let upper_f32 = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 4.0]));

    let div_out = div(&lhs_f32, &rhs_f32).unwrap();
    assert_eq!(get_f32(&div_out, &[0]), 4.0);
    assert_eq!(get_f32(&div_out, &[1]), -0.4);

    let abs_out = abs(&lhs_f32).unwrap();
    assert_eq!(get_f32(&abs_out, &[0]), 8.0);
    assert_eq!(get_f32(&abs_out, &[1]), 2.0);

    let sign_out = sign(&lhs_f32).unwrap();
    assert_eq!(get_f32(&sign_out, &[0]), 1.0);
    assert_eq!(get_f32(&sign_out, &[1]), -1.0);

    let max_out = maximum(&lhs_f32, &rhs_f32).unwrap();
    assert_eq!(get_f32(&max_out, &[0]), 8.0);
    assert_eq!(get_f32(&max_out, &[1]), 5.0);

    let min_out = minimum(&lhs_f32, &rhs_f32).unwrap();
    assert_eq!(get_f32(&min_out, &[0]), 2.0);
    assert_eq!(get_f32(&min_out, &[1]), -2.0);

    let cmp_out = compare(&lhs_f32, &rhs_f32, &CompareDir::Gt).unwrap();
    assert!(get_bool(&cmp_out, &[0]));
    assert!(!get_bool(&cmp_out, &[1]));

    let select_out = select(&pred_bool, &lhs_f32, &rhs_f32).unwrap();
    assert_eq!(get_f32(&select_out, &[0]), 2.0);
    assert_eq!(get_f32(&select_out, &[1]), -2.0);

    let clamp_out = clamp(&lhs_f32, &lower_f32, &upper_f32).unwrap();
    assert_eq!(get_f32(&clamp_out, &[0]), 1.0);
    assert_eq!(get_f32(&clamp_out, &[1]), -1.0);

    let input_c32 = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(3.0, 4.0), Complex32::new(0.0, 0.0)],
    ));
    let lhs_c32 = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(3.0, 4.0), Complex32::new(1.0, 0.0)],
    ));
    let rhs_c32 = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 2.0)],
    ));
    let lower_c32 = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(0.5, 0.0), Complex32::new(0.5, 0.0)],
    ));
    let upper_c32 = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(4.0, 0.0), Complex32::new(2.0, 2.0)],
    ));

    let abs_c32 = abs(&input_c32).unwrap();
    assert_eq!(get_c32(&abs_c32, &[0]), Complex32::new(5.0, 0.0));
    assert_eq!(get_c32(&abs_c32, &[1]), Complex32::new(0.0, 0.0));

    let sign_c32 = sign(&input_c32).unwrap();
    assert_eq!(get_c32(&sign_c32, &[1]), Complex32::new(0.0, 0.0));

    let max_c32 = maximum(&lhs_c32, &rhs_c32).unwrap();
    assert_eq!(get_c32(&max_c32, &[0]), Complex32::new(3.0, 4.0));

    let min_c32 = minimum(&lhs_c32, &rhs_c32).unwrap();
    assert_eq!(get_c32(&min_c32, &[1]), Complex32::new(1.0, 0.0));

    let cmp_c32 = compare(&lhs_c32, &rhs_c32, &CompareDir::Eq).unwrap();
    assert!(!get_bool(&cmp_c32, &[0]));

    let select_c32 = select(&pred_bool, &lhs_c32, &rhs_c32).unwrap();
    assert_eq!(get_c32(&select_c32, &[0]), Complex32::new(1.0, 0.0));
    assert_eq!(get_c32(&select_c32, &[1]), Complex32::new(1.0, 0.0));

    let clamp_c32 = clamp(&lhs_c32, &lower_c32, &upper_c32).unwrap();
    assert_eq!(get_c32(&clamp_c32, &[0]), Complex32::new(4.0, 0.0));

    let scalar_f32 = Tensor::F32(TypedTensor::from_vec_col_major(vec![], vec![2.0f32]));
    let add_c32 = add(&scalar_f32, &rhs_c32).unwrap();
    assert_eq!(get_c32(&add_c32, &[0]), Complex32::new(3.0, 0.0));

    let mul_c32 = mul(&scalar_f32, &rhs_c32).unwrap();
    assert_eq!(get_c32(&mul_c32, &[1]), Complex32::new(0.0, 4.0));

    let scalar_div_c32 = div(&scalar_f32, &rhs_c32).unwrap();
    assert_eq!(get_c32(&scalar_div_c32, &[0]), Complex32::new(2.0, 0.0));
    assert_eq!(get_c32(&scalar_div_c32, &[1]), Complex32::new(0.0, -1.0));

    let c32_div_scalar = div(&rhs_c32, &scalar_f32).unwrap();
    assert_eq!(get_c32(&c32_div_scalar, &[0]), Complex32::new(0.5, 0.0));
    assert_eq!(get_c32(&c32_div_scalar, &[1]), Complex32::new(0.0, 1.0));

    assert!(matches!(
        div(
            &lhs_f32,
            &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]))
        ),
        Err(crate::Error::DTypeMismatch { op: "div", .. })
    ));
    assert!(matches!(
        clamp(
            &lhs_f32,
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![1], vec![0.0f32])),
            &upper_f32
        ),
        Err(crate::Error::ShapeMismatch { op: "clamp", .. })
    ));
}

#[test]
fn test_direct_elementwise_helpers_cover_f64_c64_dispatch_and_mismatch_paths() {
    let lhs_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.5f64, -3.0]));
    let rhs_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![2.0f64, 4.0]));
    let scalar_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![2.0f64]));
    let pred_bool = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![false, true]));
    let lower_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![0.0f64, -2.0]));
    let upper_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![2.0f64, 3.0]));
    let short_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![1], vec![1.0f64]));
    let lhs_i32 = Tensor::I32(TypedTensor::from_vec_col_major(vec![2], vec![1i32, 3]));
    let rhs_i32 = Tensor::I32(TypedTensor::from_vec_col_major(vec![2], vec![2i32, 3]));
    let lhs_i64 = Tensor::I64(TypedTensor::from_vec_col_major(vec![2], vec![5i64, -1]));
    let rhs_i64 = Tensor::I64(TypedTensor::from_vec_col_major(vec![2], vec![2i64, -1]));
    let lhs_bool = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, false]));
    let rhs_bool = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![false, false]));

    let add_out = add(&lhs_f64, &rhs_f64).unwrap();
    assert_eq!(get_f64(&add_out, &[0]), 3.5);
    assert_eq!(get_f64(&add_out, &[1]), 1.0);

    let mul_out = mul(&lhs_f64, &rhs_f64).unwrap();
    assert_eq!(get_f64(&mul_out, &[0]), 3.0);
    assert_eq!(get_f64(&mul_out, &[1]), -12.0);

    let div_out = div(
        &rhs_f64,
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![2.0, 2.0])),
    )
    .unwrap();
    assert_eq!(get_f64(&div_out, &[0]), 1.0);
    assert_eq!(get_f64(&div_out, &[1]), 2.0);

    let neg_out = neg(&lhs_f64).unwrap();
    assert_eq!(get_f64(&neg_out, &[0]), -1.5);
    assert_eq!(get_f64(&neg_out, &[1]), 3.0);

    let conj_out = conj(&lhs_f64).unwrap();
    assert_eq!(get_f64(&conj_out, &[0]), 1.5);
    assert_eq!(get_f64(&conj_out, &[1]), -3.0);

    let compare_out = compare(&lhs_f64, &rhs_f64, &CompareDir::Lt).unwrap();
    assert!(get_bool(&compare_out, &[0]));
    assert!(get_bool(&compare_out, &[1]));

    let select_out = select(&pred_bool, &lhs_f64, &rhs_f64).unwrap();
    assert_eq!(get_f64(&select_out, &[0]), 2.0);
    assert_eq!(get_f64(&select_out, &[1]), -3.0);

    assert!(get_bool(
        &compare(&lhs_i32, &rhs_i32, &CompareDir::Lt).unwrap(),
        &[0]
    ));
    assert!(get_bool(
        &compare(&lhs_i32, &rhs_i32, &CompareDir::Le).unwrap(),
        &[1]
    ));
    assert!(get_bool(
        &compare(&lhs_i64, &rhs_i64, &CompareDir::Gt).unwrap(),
        &[0]
    ));
    assert!(get_bool(
        &compare(&lhs_i64, &rhs_i64, &CompareDir::Ge).unwrap(),
        &[1]
    ));
    assert!(get_bool(
        &compare(&lhs_bool, &rhs_bool, &CompareDir::Eq).unwrap(),
        &[1]
    ));

    let select_i32 = select(&pred_bool, &lhs_i32, &rhs_i32).unwrap();
    assert_eq!(get_i32(&select_i32, &[0]), 2);
    assert_eq!(get_i32(&select_i32, &[1]), 3);
    let select_i64 = select(&pred_bool, &lhs_i64, &rhs_i64).unwrap();
    assert_eq!(get_i64(&select_i64, &[0]), 2);
    assert_eq!(get_i64(&select_i64, &[1]), -1);
    let select_bool = select(&pred_bool, &lhs_bool, &rhs_bool).unwrap();
    assert!(!get_bool(&select_bool, &[0]));
    assert!(!get_bool(&select_bool, &[1]));

    let clamp_out = clamp(&lhs_f64, &lower_f64, &upper_f64).unwrap();
    assert_eq!(get_f64(&clamp_out, &[0]), 1.5);
    assert_eq!(get_f64(&clamp_out, &[1]), -2.0);

    let lhs_c64 = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, 0.0)],
    ));
    let rhs_c64 = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 2.0)],
    ));
    let lower_c64 = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.0)],
    ));
    let upper_c64 = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(4.0, 0.0), Complex64::new(2.0, 2.0)],
    ));

    let add_left_scalar = add(&scalar_f64, &rhs_c64).unwrap();
    assert_c64_close(get_c64(&add_left_scalar, &[0]), Complex64::new(3.0, 0.0));
    let add_right_scalar = add(&lhs_c64, &scalar_f64).unwrap();
    assert_c64_close(get_c64(&add_right_scalar, &[1]), Complex64::new(3.0, 0.0));

    let mul_left_scalar = mul(&scalar_f64, &rhs_c64).unwrap();
    assert_c64_close(get_c64(&mul_left_scalar, &[1]), Complex64::new(0.0, 4.0));
    let mul_right_scalar = mul(&lhs_c64, &scalar_f64).unwrap();
    assert_c64_close(get_c64(&mul_right_scalar, &[0]), Complex64::new(6.0, 8.0));

    let div_left_scalar = div(&scalar_f64, &rhs_c64).unwrap();
    assert_c64_close(get_c64(&div_left_scalar, &[0]), Complex64::new(2.0, 0.0));
    assert_c64_close(get_c64(&div_left_scalar, &[1]), Complex64::new(0.0, -1.0));

    let div_right_scalar = div(&lhs_c64, &scalar_f64).unwrap();
    assert_c64_close(get_c64(&div_right_scalar, &[0]), Complex64::new(1.5, 2.0));
    assert_c64_close(get_c64(&div_right_scalar, &[1]), Complex64::new(0.5, 0.0));

    let div_c64 = div(
        &lhs_c64,
        &Tensor::C64(TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 1.0), Complex64::new(1.0, 0.0)],
        )),
    )
    .unwrap();
    assert_c64_close(get_c64(&div_c64, &[0]), Complex64::new(3.5, 0.5));
    assert_c64_close(get_c64(&div_c64, &[1]), Complex64::new(1.0, 0.0));

    let neg_c64 = neg(&lhs_c64).unwrap();
    assert_c64_close(get_c64(&neg_c64, &[0]), Complex64::new(-3.0, -4.0));
    let conj_c64 = conj(&lhs_c64).unwrap();
    assert_c64_close(get_c64(&conj_c64, &[0]), Complex64::new(3.0, -4.0));

    let compare_lt = compare(&lhs_c64, &rhs_c64, &CompareDir::Lt).unwrap();
    let compare_le = compare(&lhs_c64, &rhs_c64, &CompareDir::Le).unwrap();
    let compare_gt = compare(&lhs_c64, &rhs_c64, &CompareDir::Gt).unwrap();
    let compare_ge = compare(&lhs_c64, &rhs_c64, &CompareDir::Ge).unwrap();
    assert!(!get_bool(&compare_lt, &[0]));
    assert!(!get_bool(&compare_le, &[0]));
    assert!(get_bool(&compare_gt, &[0]));
    assert!(get_bool(&compare_ge, &[0]));

    let select_c64 = select(&pred_bool, &lhs_c64, &rhs_c64).unwrap();
    assert_c64_close(get_c64(&select_c64, &[0]), Complex64::new(1.0, 0.0));
    assert_c64_close(get_c64(&select_c64, &[1]), Complex64::new(1.0, 0.0));

    let clamp_c64 = clamp(&lhs_c64, &lower_c64, &upper_c64).unwrap();
    assert_c64_close(get_c64(&clamp_c64, &[0]), Complex64::new(4.0, 0.0));
    assert_c64_close(get_c64(&clamp_c64, &[1]), Complex64::new(1.0, 0.0));

    let lhs_f32 = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]));

    assert!(matches!(
        add(&lhs_f32, &rhs_f64),
        Err(crate::Error::DTypeMismatch { op: "add", .. })
    ));
    assert!(matches!(
        mul(&lhs_f32, &rhs_f64),
        Err(crate::Error::DTypeMismatch { op: "mul", .. })
    ));
    assert!(matches!(
        maximum(&lhs_f32, &rhs_f64),
        Err(crate::Error::DTypeMismatch { op: "maximum", .. })
    ));
    assert!(matches!(
        minimum(&lhs_f32, &rhs_f64),
        Err(crate::Error::DTypeMismatch { op: "minimum", .. })
    ));
    assert!(matches!(
        compare(&lhs_f32, &rhs_f64, &CompareDir::Eq),
        Err(crate::Error::DTypeMismatch { op: "compare", .. })
    ));
    assert!(matches!(
        select(&lhs_f32, &lhs_f32, &rhs_f64),
        Err(crate::Error::DTypeMismatch { op: "select", .. })
    ));
    assert!(matches!(
        clamp(&lhs_f32, &lhs_f32, &rhs_f64),
        Err(crate::Error::BackendFailure {
            op: "clamp",
            message,
        }) if message == "dtype mismatch"
    ));

    assert!(matches!(
        add(&lhs_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "add", .. })
    ));
    assert!(matches!(
        mul(&lhs_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "mul", .. })
    ));
    assert!(matches!(
        div(&lhs_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "div", .. })
    ));
    assert!(matches!(
        maximum(&lhs_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "maximum", .. })
    ));
    assert!(matches!(
        minimum(&lhs_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "minimum", .. })
    ));
    assert!(matches!(
        compare(&lhs_f64, &short_f64, &CompareDir::Eq),
        Err(crate::Error::ShapeMismatch { op: "compare", .. })
    ));
    assert!(matches!(
        select(&pred_bool, &short_f64, &rhs_f64),
        Err(crate::Error::ShapeMismatch { op: "select", .. })
    ));
    assert!(matches!(
        select(&pred_bool, &rhs_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "select", .. })
    ));
    assert!(matches!(
        clamp(&lhs_f64, &lower_f64, &short_f64),
        Err(crate::Error::ShapeMismatch { op: "clamp", .. })
    ));
}

#[test]
fn test_reduction_helpers_cover_complex_and_error_paths() {
    let complex = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, -1.0),
            Complex32::new(4.0, 2.0),
        ],
    ));
    let sum = reduce_sum(&complex, &[0]).unwrap();
    assert_eq!(get_c32(&sum, &[0]), Complex32::new(3.0, 1.0));
    assert_eq!(get_c32(&sum, &[1]), Complex32::new(7.0, 1.0));

    let prod = reduce_prod(&complex, &[]).unwrap();
    assert_eq!(prod.shape(), &[2, 2]);

    assert!(matches!(
        reduce_sum(
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0])),
            &[2]
        ),
        Err(crate::Error::AxisOutOfBounds {
            op: "reduce_sum",
            ..
        })
    ));
    assert!(matches!(
        reduce_prod(
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0])),
            &[0, 0]
        ),
        Err(crate::Error::DuplicateAxis {
            op: "reduce_prod",
            ..
        })
    ));
    assert!(matches!(
        reduce_max(&complex, &[0]),
        Err(crate::Error::BackendFailure {
            op: "reduce_max",
            ..
        })
    ));
    assert!(matches!(
        reduce_min(&complex, &[0]),
        Err(crate::Error::BackendFailure {
            op: "reduce_min",
            ..
        })
    ));
}

#[test]
fn test_structural_helpers_cover_f32_success_and_error_paths() {
    let matrix = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0f32, 2.0, 3.0, 4.0],
    ));
    let transposed = transpose(&matrix, &[1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(get_f32(&transposed, &[1, 0]), 3.0);

    let scalar = Tensor::F32(TypedTensor::from_vec_col_major(vec![], vec![5.0f32]));
    let broadcast = broadcast_in_dim(&scalar, &[2], &[]).unwrap();
    assert_eq!(get_f32(&broadcast, &[1]), 5.0);

    let diag = extract_diagonal(&matrix, 0, 1).unwrap();
    assert_eq!(get_f32(&diag, &[0]), 1.0);
    assert_eq!(get_f32(&diag, &[1]), 4.0);

    let embedded = embed_diagonal(
        &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![7.0f32, 8.0])),
        0,
        1,
    )
    .unwrap();
    assert_eq!(embedded.shape(), &[2, 2]);
    assert_eq!(get_f32(&embedded, &[1, 1]), 8.0);

    let lower = tril(&matrix, 0).unwrap();
    assert_eq!(get_f32(&lower, &[0, 1]), 0.0);
    let upper = triu(&matrix, 0).unwrap();
    assert_eq!(get_f32(&upper, &[1, 0]), 0.0);

    assert!(matches!(
        transpose(&matrix, &[0]),
        Err(crate::Error::RankMismatch {
            op: "transpose",
            ..
        })
    ));
    assert!(matches!(
        transpose(&matrix, &[0, 0]),
        Err(crate::Error::DuplicateAxis {
            op: "transpose",
            ..
        })
    ));
    assert!(matches!(
        broadcast_in_dim(
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0])),
            &[3, 2],
            &[0]
        ),
        Err(crate::Error::ShapeMismatch {
            op: "broadcast_in_dim",
            ..
        })
    ));
    assert!(matches!(
        extract_diagonal(&matrix, 1, 1),
        Err(crate::Error::DuplicateAxis {
            op: "extract_diagonal",
            ..
        })
    ));
    assert!(matches!(
        embed_diagonal(
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0])),
            0,
            2
        ),
        Err(crate::Error::AxisOutOfBounds {
            op: "embed_diagonal",
            ..
        })
    ));
    let vector = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]));
    assert!(matches!(
        tril(&vector, 0),
        Err(crate::Error::RankMismatch { op: "tril", .. })
    ));
    assert!(matches!(
        triu(&vector, 0),
        Err(crate::Error::RankMismatch { op: "triu", .. })
    ));
}
