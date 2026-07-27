use super::*;
use crate::sub;

#[test]
fn test_zeros_ones() {
    let z = TypedTensor::<f64>::zeros(vec![2, 3]).unwrap();
    assert_eq!(z.shape(), &[2, 3]);
    assert_eq!(z.n_elements(), 6);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(*z.get(&[i, j]).unwrap(), 0.0);
        }
    }

    let o = TypedTensor::<f64>::ones(vec![2, 3]).unwrap();
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(*o.get(&[i, j]).unwrap(), 1.0);
        }
    }
}

#[test]
fn test_from_vec_uses_column_major_indices() {
    let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    assert_eq!(*t.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(*t.get(&[1, 0]).unwrap(), 2.0);
    assert_eq!(*t.get(&[0, 1]).unwrap(), 3.0);
    assert_eq!(*t.get(&[1, 1]).unwrap(), 4.0);
    assert_eq!(*t.get(&[0, 2]).unwrap(), 5.0);
    assert_eq!(*t.get(&[1, 2]).unwrap(), 6.0);
}

#[test]
fn test_tensor_metadata() {
    let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0, 2.0]).unwrap());
    assert_eq!(t.shape(), &[2, 1]);
    assert_eq!(t.dtype(), DType::F64);
}

#[test]
fn test_reshape() {
    let t = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let r = reshape(&t, &[3, 2]).unwrap();
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(get_f64(&r, &[0, 0]), 1.0);
    assert_eq!(get_f64(&r, &[1, 0]), 2.0);
    assert_eq!(get_f64(&r, &[2, 0]), 3.0);
    assert_eq!(get_f64(&r, &[0, 1]), 4.0);
    assert_eq!(get_f64(&r, &[1, 1]), 5.0);
    assert_eq!(get_f64(&r, &[2, 1]), 6.0);
}

#[test]
fn test_add_mul() {
    let a =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let b = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
    );
    let sum = add(&a, &b).unwrap();
    let prod = mul(&a, &b).unwrap();

    assert_eq!(get_f64(&sum, &[0, 0]), 11.0);
    assert_eq!(get_f64(&sum, &[1, 0]), 22.0);
    assert_eq!(get_f64(&sum, &[0, 1]), 33.0);
    assert_eq!(get_f64(&sum, &[1, 1]), 44.0);

    assert_eq!(get_f64(&prod, &[0, 0]), 10.0);
    assert_eq!(get_f64(&prod, &[1, 0]), 40.0);
    assert_eq!(get_f64(&prod, &[0, 1]), 90.0);
    assert_eq!(get_f64(&prod, &[1, 1]), 160.0);
}

#[test]
fn test_add_mul_i64() {
    let a = Tensor::I64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1, 2, 3, 4]).unwrap());
    let b = Tensor::I64(TypedTensor::from_vec_col_major(vec![2, 2], vec![10, 20, 30, 40]).unwrap());
    let sum = add(&a, &b).unwrap();
    let prod = mul(&a, &b).unwrap();

    assert_eq!(get_i64(&sum, &[0, 0]), 11);
    assert_eq!(get_i64(&sum, &[1, 0]), 22);
    assert_eq!(get_i64(&sum, &[0, 1]), 33);
    assert_eq!(get_i64(&sum, &[1, 1]), 44);

    assert_eq!(get_i64(&prod, &[0, 0]), 10);
    assert_eq!(get_i64(&prod, &[1, 0]), 40);
    assert_eq!(get_i64(&prod, &[0, 1]), 90);
    assert_eq!(get_i64(&prod, &[1, 1]), 160);
}

#[test]
fn test_integer_add_mul_wrap_on_overflow() {
    let i32_lhs = Tensor::from_vec_col_major(vec![3], vec![i32::MAX, i32::MIN, 50_i32]).unwrap();
    let i32_rhs = Tensor::from_vec_col_major(vec![3], vec![1_i32, -1, i32::MAX]).unwrap();
    let sum = add(&i32_lhs, &i32_rhs).unwrap();
    let diff = sub(&i32_lhs, &i32_rhs).unwrap();
    let prod = mul(&i32_lhs, &i32_rhs).unwrap();

    assert_eq!(
        sum.as_slice::<i32>().unwrap(),
        &[i32::MIN, i32::MAX, -2_147_483_599]
    );
    assert_eq!(
        diff.as_slice::<i32>().unwrap(),
        &[2_147_483_646, -2_147_483_647, -2_147_483_597]
    );
    assert_eq!(prod.as_slice::<i32>().unwrap(), &[i32::MAX, i32::MIN, -50]);

    let i64_lhs = Tensor::from_vec_col_major(vec![2], vec![i64::MAX, i64::MIN]).unwrap();
    let i64_rhs = Tensor::from_vec_col_major(vec![2], vec![1_i64, -1]).unwrap();
    let sum = add(&i64_lhs, &i64_rhs).unwrap();
    let diff = sub(&i64_lhs, &i64_rhs).unwrap();
    let prod = mul(&i64_lhs, &i64_rhs).unwrap();

    assert_eq!(sum.as_slice::<i64>().unwrap(), &[i64::MIN, i64::MAX]);
    assert_eq!(
        diff.as_slice::<i64>().unwrap(),
        &[9_223_372_036_854_775_806, -9_223_372_036_854_775_807]
    );
    assert_eq!(prod.as_slice::<i64>().unwrap(), &[i64::MAX, i64::MIN]);
}

#[test]
fn test_integer_add_mul_read_views_wrap_on_overflow() {
    let lhs =
        TypedTensor::<i32>::from_vec_col_major(vec![3], vec![i32::MAX, i32::MIN, 50]).unwrap();
    let rhs = TypedTensor::<i32>::from_vec_col_major(vec![3], vec![1, -1, i32::MAX]).unwrap();
    let mut backend = CpuBackend::new();

    let sum = backend
        .add_read(
            TensorRead::from_view(TensorView::I32(lhs.as_view())),
            TensorRead::from_view(TensorView::I32(rhs.as_view())),
        )
        .unwrap();
    let diff = backend
        .sub_read(
            TensorRead::from_view(TensorView::I32(lhs.as_view())),
            TensorRead::from_view(TensorView::I32(rhs.as_view())),
        )
        .unwrap();
    let prod = backend
        .mul_read(
            TensorRead::from_view(TensorView::I32(lhs.as_view())),
            TensorRead::from_view(TensorView::I32(rhs.as_view())),
        )
        .unwrap();

    assert_eq!(
        sum.as_slice::<i32>().unwrap(),
        &[i32::MIN, i32::MAX, -2_147_483_599]
    );
    assert_eq!(
        diff.as_slice::<i32>().unwrap(),
        &[2_147_483_646, -2_147_483_647, -2_147_483_597]
    );
    assert_eq!(prod.as_slice::<i32>().unwrap(), &[i32::MAX, i32::MIN, -50]);
}

#[test]
fn test_integer_unary_ops_use_wrapping_semantics() {
    let i32_input = Tensor::from_vec_col_major(vec![4], vec![i32::MIN, -3, 0, 5]).unwrap();
    let neg_out = neg(&i32_input).unwrap();
    let abs_out = abs(&i32_input).unwrap();
    let sign_out = sign(&i32_input).unwrap();
    assert_eq!(neg_out.as_slice::<i32>().unwrap(), &[i32::MIN, 3, 0, -5]);
    assert_eq!(abs_out.as_slice::<i32>().unwrap(), &[i32::MIN, 3, 0, 5]);
    assert_eq!(sign_out.as_slice::<i32>().unwrap(), &[-1, -1, 0, 1]);

    let i64_input = Tensor::from_vec_col_major(vec![4], vec![i64::MIN, -2, 0, 7]).unwrap();
    let neg_out = neg(&i64_input).unwrap();
    let abs_out = abs(&i64_input).unwrap();
    let sign_out = sign(&i64_input).unwrap();
    assert_eq!(neg_out.as_slice::<i64>().unwrap(), &[i64::MIN, 2, 0, -7]);
    assert_eq!(abs_out.as_slice::<i64>().unwrap(), &[i64::MIN, 2, 0, 7]);
    assert_eq!(sign_out.as_slice::<i64>().unwrap(), &[-1, -1, 0, 1]);
}

#[test]
fn test_integer_unary_read_views_use_wrapping_semantics() {
    let input = TypedTensor::<i32>::from_vec_col_major(vec![4], vec![i32::MIN, -3, 0, 5]).unwrap();
    let read = || TensorRead::from_view(TensorView::I32(input.as_view()));
    let mut backend = CpuBackend::new();

    let neg_out = backend.neg_read(read()).unwrap();
    let abs_out = backend.abs_read(read()).unwrap();
    let sign_out = backend.sign_read(read()).unwrap();

    assert_eq!(neg_out.as_slice::<i32>().unwrap(), &[i32::MIN, 3, 0, -5]);
    assert_eq!(abs_out.as_slice::<i32>().unwrap(), &[i32::MIN, 3, 0, 5]);
    assert_eq!(sign_out.as_slice::<i32>().unwrap(), &[-1, -1, 0, 1]);
}

#[test]
fn test_integer_maximum_minimum_and_reductions() {
    let lhs = Tensor::from_vec_col_major(vec![4], vec![i32::MIN, -1, 7, i32::MAX]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![4], vec![0, -2, 8, i32::MIN]).unwrap();
    let max_out = maximum(&lhs, &rhs).unwrap();
    let min_out = minimum(&lhs, &rhs).unwrap();
    assert_eq!(max_out.as_slice::<i32>().unwrap(), &[0, -1, 8, i32::MAX]);
    assert_eq!(
        min_out.as_slice::<i32>().unwrap(),
        &[i32::MIN, -2, 7, i32::MIN]
    );

    let input = Tensor::from_vec_col_major(vec![2, 2], vec![i32::MIN, 1, i32::MAX, -5]).unwrap();
    let max_cols = reduce_max(&input, &[0]).unwrap();
    let min_cols = reduce_min(&input, &[0]).unwrap();
    assert_eq!(max_cols.as_slice::<i32>().unwrap(), &[1, i32::MAX]);
    assert_eq!(min_cols.as_slice::<i32>().unwrap(), &[i32::MIN, -5]);

    let input = Tensor::from_vec_col_major(vec![2, 2], vec![i64::MIN, 4, i64::MAX, -7]).unwrap();
    let max_cols = reduce_max(&input, &[0]).unwrap();
    let min_cols = reduce_min(&input, &[0]).unwrap();
    assert_eq!(max_cols.as_slice::<i64>().unwrap(), &[4, i64::MAX]);
    assert_eq!(min_cols.as_slice::<i64>().unwrap(), &[i64::MIN, -7]);
}

#[test]
fn test_integer_maximum_minimum_and_reduction_read_views() {
    let lhs =
        TypedTensor::<i32>::from_vec_col_major(vec![4], vec![i32::MIN, -1, 7, i32::MAX]).unwrap();
    let rhs = TypedTensor::<i32>::from_vec_col_major(vec![4], vec![0, -2, 8, i32::MIN]).unwrap();
    let mut backend = CpuBackend::new();

    let max_out = backend
        .maximum_read(
            TensorRead::from_view(TensorView::I32(lhs.as_view())),
            TensorRead::from_view(TensorView::I32(rhs.as_view())),
        )
        .unwrap();
    let min_out = backend
        .minimum_read(
            TensorRead::from_view(TensorView::I32(lhs.as_view())),
            TensorRead::from_view(TensorView::I32(rhs.as_view())),
        )
        .unwrap();
    assert_eq!(max_out.as_slice::<i32>().unwrap(), &[0, -1, 8, i32::MAX]);
    assert_eq!(
        min_out.as_slice::<i32>().unwrap(),
        &[i32::MIN, -2, 7, i32::MIN]
    );

    let input = TypedTensor::<i32>::from_vec_col_major(vec![2, 2], vec![i32::MIN, 1, i32::MAX, -5])
        .unwrap();
    let max_cols = backend
        .reduce_max_read(
            TensorRead::from_view(TensorView::I32(input.as_view())),
            &[0],
        )
        .unwrap();
    let min_cols = backend
        .reduce_min_read(
            TensorRead::from_view(TensorView::I32(input.as_view())),
            &[0],
        )
        .unwrap();
    assert_eq!(max_cols.as_slice::<i32>().unwrap(), &[1, i32::MAX]);
    assert_eq!(min_cols.as_slice::<i32>().unwrap(), &[i32::MIN, -5]);
}

#[test]
fn test_integer_div_rem_pow_contract() {
    let lhs =
        Tensor::from_vec_col_major(vec![6], vec![7_i32, -7, 7, -7, i32::MIN, i32::MAX]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![6], vec![3_i32, 3, -3, -3, -1, 2]).unwrap();

    let quotient = div(&lhs, &rhs).unwrap();
    let remainder = rem(&lhs, &rhs).unwrap();
    assert_eq!(
        quotient.as_slice::<i32>().unwrap(),
        &[2, -2, -2, 2, i32::MIN, i32::MAX / 2]
    );
    assert_eq!(
        remainder.as_slice::<i32>().unwrap(),
        &[1, -1, 1, -1, 0, i32::MAX % 2]
    );

    let base = Tensor::from_vec_col_major(vec![4], vec![2_i32, -2, i32::MAX, i32::MIN]).unwrap();
    let exp = Tensor::from_vec_col_major(vec![4], vec![3_i32, 3, 2, 1]).unwrap();
    let out = pow(&base, &exp).unwrap();
    assert_eq!(
        out.as_slice::<i32>().unwrap(),
        &[8, -8, i32::MAX.wrapping_pow(2), i32::MIN]
    );

    let lhs = Tensor::from_vec_col_major(vec![3], vec![i64::MIN, -9_i64, 9]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3], vec![-1_i64, 4, -4]).unwrap();
    let quotient = div(&lhs, &rhs).unwrap();
    let remainder = rem(&lhs, &rhs).unwrap();
    assert_eq!(quotient.as_slice::<i64>().unwrap(), &[i64::MIN, -2, -2]);
    assert_eq!(remainder.as_slice::<i64>().unwrap(), &[0, -1, 1]);

    let base = Tensor::from_vec_col_major(vec![3], vec![2_i64, -2, i64::MAX]).unwrap();
    let exp = Tensor::from_vec_col_major(vec![3], vec![63_i64, 63, 2]).unwrap();
    let out = pow(&base, &exp).unwrap();
    assert_eq!(
        out.as_slice::<i64>().unwrap(),
        &[i64::MIN, i64::MIN, i64::MAX.wrapping_pow(2)]
    );
}

#[test]
fn test_pow_accepts_rank_zero_operands() {
    let f64_tensor = Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 3.0, 4.0]).unwrap();
    let f64_exponent = Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let f64_base = Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();

    let tensor_base = pow(&f64_tensor, &f64_exponent).unwrap();
    assert_eq!(tensor_base.shape(), &[3]);
    assert_eq!(tensor_base.as_slice::<f64>().unwrap(), &[4.0, 9.0, 16.0]);

    let scalar_base = pow(&f64_base, &f64_tensor).unwrap();
    assert_eq!(scalar_base.shape(), &[3]);
    assert_eq!(scalar_base.as_slice::<f64>().unwrap(), &[4.0, 8.0, 16.0]);

    let i32_tensor = Tensor::from_vec_col_major(vec![3], vec![2_i32, 3, 4]).unwrap();
    let i32_exponent = Tensor::from_vec_col_major(vec![], vec![3_i32]).unwrap();
    let i32_base = Tensor::from_vec_col_major(vec![], vec![2_i32]).unwrap();

    let tensor_base = pow(&i32_tensor, &i32_exponent).unwrap();
    assert_eq!(tensor_base.shape(), &[3]);
    assert_eq!(tensor_base.as_slice::<i32>().unwrap(), &[8, 27, 64]);

    let scalar_base = pow(&i32_base, &i32_tensor).unwrap();
    assert_eq!(scalar_base.shape(), &[3]);
    assert_eq!(scalar_base.as_slice::<i32>().unwrap(), &[4, 8, 16]);
}

#[test]
fn test_pow_rank_zero_read_views_and_domain_contracts() {
    let tensor = TypedTensor::<f32>::from_vec_col_major(vec![3], vec![2.0, 3.0, 4.0]).unwrap();
    let exponent = TypedTensor::<f32>::from_vec_col_major(vec![], vec![2.0]).unwrap();
    let base = TypedTensor::<f32>::from_vec_col_major(vec![], vec![2.0]).unwrap();
    let mut backend = CpuBackend::new();

    let tensor_base = backend
        .pow_read(
            TensorRead::from_view(TensorView::F32(tensor.as_view())),
            TensorRead::from_view(TensorView::F32(exponent.as_view())),
        )
        .unwrap();
    assert_eq!(tensor_base.shape(), &[3]);
    assert_eq!(tensor_base.as_slice::<f32>().unwrap(), &[4.0, 9.0, 16.0]);

    let scalar_base = backend
        .pow_read(
            TensorRead::from_view(TensorView::F32(base.as_view())),
            TensorRead::from_view(TensorView::F32(tensor.as_view())),
        )
        .unwrap();
    assert_eq!(scalar_base.shape(), &[3]);
    assert_eq!(scalar_base.as_slice::<f32>().unwrap(), &[4.0, 8.0, 16.0]);

    let empty = Tensor::from_vec_col_major(vec![0, 2], Vec::<f64>::new()).unwrap();
    let scalar = Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let empty_out = pow(&empty, &scalar).unwrap();
    assert_eq!(empty_out.shape(), &[0, 2]);
    assert!(empty_out.as_slice::<f64>().unwrap().is_empty());

    for (base, exponent) in [
        (
            Tensor::from_vec_col_major(vec![2], vec![2_i64, 3]).unwrap(),
            Tensor::from_vec_col_major(vec![], vec![-1_i64]).unwrap(),
        ),
        (
            Tensor::from_vec_col_major(vec![], vec![2_i64]).unwrap(),
            Tensor::from_vec_col_major(vec![2], vec![2_i64, -1]).unwrap(),
        ),
    ] {
        assert!(matches!(
            pow(&base, &exponent),
            Err(Error::Extension {
                op: "pow",
                family: "cpu",
                kind: tenferro_tensor::ErrorKind::NumericalFailure,
                ..
            })
        ));
    }
}

#[test]
fn test_integer_div_rem_pow_read_views_contract() {
    let lhs =
        TypedTensor::<i32>::from_vec_col_major(vec![6], vec![7_i32, -7, 7, -7, i32::MIN, i32::MAX])
            .unwrap();
    let rhs =
        TypedTensor::<i32>::from_vec_col_major(vec![6], vec![3_i32, 3, -3, -3, -1, 2]).unwrap();
    let mut backend = CpuBackend::new();

    let quotient = backend
        .div_read(
            TensorRead::from_view(TensorView::I32(lhs.as_view())),
            TensorRead::from_view(TensorView::I32(rhs.as_view())),
        )
        .unwrap();
    let remainder = backend
        .rem_read(
            TensorRead::from_view(TensorView::I32(lhs.as_view())),
            TensorRead::from_view(TensorView::I32(rhs.as_view())),
        )
        .unwrap();
    assert_eq!(
        quotient.as_slice::<i32>().unwrap(),
        &[2, -2, -2, 2, i32::MIN, i32::MAX / 2]
    );
    assert_eq!(
        remainder.as_slice::<i32>().unwrap(),
        &[1, -1, 1, -1, 0, i32::MAX % 2]
    );

    let base = TypedTensor::<i32>::from_vec_col_major(vec![4], vec![2_i32, -2, i32::MAX, i32::MIN])
        .unwrap();
    let exp = TypedTensor::<i32>::from_vec_col_major(vec![4], vec![3_i32, 3, 2, 1]).unwrap();
    let out = backend
        .pow_read(
            TensorRead::from_view(TensorView::I32(base.as_view())),
            TensorRead::from_view(TensorView::I32(exp.as_view())),
        )
        .unwrap();
    assert_eq!(
        out.as_slice::<i32>().unwrap(),
        &[8, -8, i32::MAX.wrapping_pow(2), i32::MIN]
    );
}

#[test]
fn test_integer_div_rem_pow_domain_errors_are_structured() {
    let lhs = Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
    let zero_rhs = Tensor::from_vec_col_major(vec![2], vec![1_i32, 0]).unwrap();

    let err = div(&lhs, &zero_rhs).unwrap_err();
    assert!(matches!(
        err,
        Error::Extension {
            op: "div",
            family: "cpu",
            kind: tenferro_tensor::ErrorKind::NumericalFailure,
            ..
        }
    ));

    let err = rem(&lhs, &zero_rhs).unwrap_err();
    assert!(matches!(
        err,
        Error::Extension {
            op: "rem",
            family: "cpu",
            kind: tenferro_tensor::ErrorKind::NumericalFailure,
            ..
        }
    ));

    let base = Tensor::from_vec_col_major(vec![2], vec![2_i32, 3]).unwrap();
    let exp = Tensor::from_vec_col_major(vec![2], vec![2_i32, -1]).unwrap();
    let err = pow(&base, &exp).unwrap_err();
    assert!(matches!(
        err,
        Error::Extension {
            op: "pow",
            family: "cpu",
            kind: tenferro_tensor::ErrorKind::NumericalFailure,
            ..
        }
    ));
}

#[test]
fn float_div_rem_preserve_ieee_special_values() {
    let f32_lhs = Tensor::from_vec_col_major(
        vec![6],
        vec![1.0_f32, 1.0, 0.0, f32::NAN, f32::INFINITY, -0.0],
    )
    .unwrap();
    let f32_div_rhs =
        Tensor::from_vec_col_major(vec![6], vec![0.0_f32, -0.0, 0.0, 1.0, f32::INFINITY, 2.0])
            .unwrap();
    let f32_div = div(&f32_lhs, &f32_div_rhs).unwrap();
    let f32_div = f32_div.as_slice::<f32>().unwrap();
    assert!(f32_div[0].is_infinite() && f32_div[0].is_sign_positive());
    assert!(f32_div[1].is_infinite() && f32_div[1].is_sign_negative());
    assert!(f32_div[2].is_nan());
    assert!(f32_div[3].is_nan());
    assert!(f32_div[4].is_nan());
    assert_eq!(f32_div[5].to_bits(), (-0.0_f32).to_bits());

    let f32_rem_rhs =
        Tensor::from_vec_col_major(vec![6], vec![0.0_f32, -0.0, 2.0, 2.0, 2.0, 2.0]).unwrap();
    let f32_rem = rem(&f32_lhs, &f32_rem_rhs).unwrap();
    let f32_rem = f32_rem.as_slice::<f32>().unwrap();
    assert!(f32_rem[0].is_nan());
    assert!(f32_rem[1].is_nan());
    assert_eq!(f32_rem[2].to_bits(), 0.0_f32.to_bits());
    assert!(f32_rem[3].is_nan());
    assert!(f32_rem[4].is_nan());
    assert_eq!(f32_rem[5].to_bits(), (-0.0_f32).to_bits());

    let f64_lhs = Tensor::from_vec_col_major(
        vec![6],
        vec![1.0_f64, 1.0, 0.0, f64::NAN, f64::INFINITY, -0.0],
    )
    .unwrap();
    let f64_div_rhs =
        Tensor::from_vec_col_major(vec![6], vec![0.0_f64, -0.0, 0.0, 1.0, f64::INFINITY, 2.0])
            .unwrap();
    let f64_div = div(&f64_lhs, &f64_div_rhs).unwrap();
    let f64_div = f64_div.as_slice::<f64>().unwrap();
    assert!(f64_div[0].is_infinite() && f64_div[0].is_sign_positive());
    assert!(f64_div[1].is_infinite() && f64_div[1].is_sign_negative());
    assert!(f64_div[2].is_nan());
    assert!(f64_div[3].is_nan());
    assert!(f64_div[4].is_nan());
    assert_eq!(f64_div[5].to_bits(), (-0.0_f64).to_bits());

    let f64_rem_rhs =
        Tensor::from_vec_col_major(vec![6], vec![0.0_f64, -0.0, 2.0, 2.0, 2.0, 2.0]).unwrap();
    let f64_rem = rem(&f64_lhs, &f64_rem_rhs).unwrap();
    let f64_rem = f64_rem.as_slice::<f64>().unwrap();
    assert!(f64_rem[0].is_nan());
    assert!(f64_rem[1].is_nan());
    assert_eq!(f64_rem[2].to_bits(), 0.0_f64.to_bits());
    assert!(f64_rem[3].is_nan());
    assert!(f64_rem[4].is_nan());
    assert_eq!(f64_rem[5].to_bits(), (-0.0_f64).to_bits());

    for (zero_rhs, _expected_dtype) in [
        (
            Tensor::from_vec_col_major(vec![1], vec![0_i32]).unwrap(),
            DType::I32,
        ),
        (
            Tensor::from_vec_col_major(vec![1], vec![0_i64]).unwrap(),
            DType::I64,
        ),
    ] {
        let lhs = match zero_rhs.dtype() {
            DType::I32 => Tensor::from_vec_col_major(vec![1], vec![1_i32]).unwrap(),
            DType::I64 => Tensor::from_vec_col_major(vec![1], vec![1_i64]).unwrap(),
            _ => unreachable!(),
        };
        assert!(matches!(
            div(&lhs, &zero_rhs),
            Err(Error::Extension {
                op: "div",
                family: "cpu",
                kind: tenferro_tensor::ErrorKind::NumericalFailure,
                ..
            })
        ));
        assert!(matches!(
            rem(&lhs, &zero_rhs),
            Err(Error::Extension {
                op: "rem",
                family: "cpu",
                kind: tenferro_tensor::ErrorKind::NumericalFailure,
                ..
            })
        ));
    }
}

#[test]
fn test_float_rem_matches_rust_remainder_sign() {
    let lhs =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![4], vec![7.0, -7.0, 7.0, -7.0]).unwrap());
    let rhs =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![4], vec![3.0, 3.0, -3.0, -3.0]).unwrap());

    let out = rem(&lhs, &rhs).unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, -1.0, 1.0, -1.0]);
}

#[test]
fn test_add_mul_rank0_broadcast() {
    let scalar = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![2.0]).unwrap());
    let tensor =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());

    let scalar_plus_tensor = add(&scalar, &tensor).unwrap();
    let tensor_plus_scalar = add(&tensor, &scalar).unwrap();
    let scalar_times_tensor = mul(&scalar, &tensor).unwrap();
    let tensor_times_scalar = mul(&tensor, &scalar).unwrap();

    for actual in [&scalar_plus_tensor, &tensor_plus_scalar] {
        assert_eq!(actual.shape(), &[2, 2]);
        assert_eq!(get_f64(actual, &[0, 0]), 3.0);
        assert_eq!(get_f64(actual, &[1, 0]), 4.0);
        assert_eq!(get_f64(actual, &[0, 1]), 5.0);
        assert_eq!(get_f64(actual, &[1, 1]), 6.0);
    }

    for actual in [&scalar_times_tensor, &tensor_times_scalar] {
        assert_eq!(actual.shape(), &[2, 2]);
        assert_eq!(get_f64(actual, &[0, 0]), 2.0);
        assert_eq!(get_f64(actual, &[1, 0]), 4.0);
        assert_eq!(get_f64(actual, &[0, 1]), 6.0);
        assert_eq!(get_f64(actual, &[1, 1]), 8.0);
    }
}

#[test]
fn test_mul_rank0_real_scalar_broadcasts_over_complex_tensor() {
    let scalar = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![2.0]).unwrap());
    let tensor = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
        )
        .unwrap(),
    );

    let scalar_times_tensor = mul(&scalar, &tensor).unwrap();
    let tensor_times_scalar = mul(&tensor, &scalar).unwrap();

    for actual in [&scalar_times_tensor, &tensor_times_scalar] {
        assert_eq!(actual.shape(), &[2]);
        assert_c64_close(get_c64(actual, &[0]), Complex64::new(2.0, 4.0));
        assert_c64_close(get_c64(actual, &[1]), Complex64::new(-6.0, 1.0));
    }
}

#[test]
fn test_mul_rank0_complex_scalar_broadcasts_over_complex_tensor() {
    let scalar = Tensor::C64(
        TypedTensor::from_vec_col_major(vec![], vec![Complex64::new(2.0, -1.0)]).unwrap(),
    );
    let tensor = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
        )
        .unwrap(),
    );
    let output = mul(&scalar, &tensor).unwrap();

    assert_c64_close(get_c64(&output, &[0]), Complex64::new(4.0, 3.0));
    assert_c64_close(get_c64(&output, &[1]), Complex64::new(-5.5, 4.0));
}

#[test]
fn test_rank0_typed_tensor_behaves_like_scalar() {
    let mut zeros = TypedTensor::<f64>::zeros(vec![]).unwrap();
    assert_eq!(zeros.shape(), &[] as &[usize]);
    assert_eq!(zeros.n_elements(), 1);
    assert_eq!(zeros.linear_offset(&[]).unwrap(), 0);
    assert_eq!(zeros.get(&[]).unwrap(), &0.0);

    *zeros.get_mut(&[]).unwrap() = 2.5;
    assert_eq!(zeros.host_data().unwrap(), &[2.5]);

    let ones = TypedTensor::<f64>::ones(vec![]).unwrap();
    assert_eq!(ones.shape(), &[] as &[usize]);
    assert_eq!(ones.n_elements(), 1);
    assert_eq!(ones.get(&[]).unwrap(), &1.0);

    let scalar = TypedTensor::<f64>::from_vec_col_major(vec![], vec![7.0]).unwrap();
    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.n_elements(), 1);
    assert_eq!(scalar.linear_offset(&[]).unwrap(), 0);
    assert_eq!(scalar.get(&[]).unwrap(), &7.0);
}

#[test]
fn test_reduce_sum() {
    let t = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let r = reduce_sum(&t, &[0], &strided_kernel::ExecContext::serial()).unwrap();
    assert_eq!(r.shape(), &[3]);
    assert_eq!(get_f64(&r, &[0]), 3.0);
    assert_eq!(get_f64(&r, &[1]), 7.0);
    assert_eq!(get_f64(&r, &[2]), 11.0);

    let all = reduce_sum(&t, &[0, 1], &strided_kernel::ExecContext::serial()).unwrap();
    assert!(all.shape().is_empty());
    assert_eq!(get_f64(&all, &[]), 21.0);
}

#[test]
fn test_reduce_prod() {
    let t = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );

    let r = reduce_prod(&t, &[0], &strided_kernel::ExecContext::serial()).unwrap();
    assert_eq!(r.shape(), &[3]);
    assert_eq!(get_f64(&r, &[0]), 2.0);
    assert_eq!(get_f64(&r, &[1]), 12.0);
    assert_eq!(get_f64(&r, &[2]), 30.0);

    let all = reduce_prod(&t, &[0, 1], &strided_kernel::ExecContext::serial()).unwrap();
    assert!(all.shape().is_empty());
    assert_eq!(get_f64(&all, &[]), 720.0);
}

#[test]
fn test_integer_reduce_sum_prod_wrap_on_overflow() {
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![i32::MAX, 1, i32::MAX, 2]).unwrap();
    let sum = reduce_sum(&input, &[0], &strided_kernel::ExecContext::serial()).unwrap();
    assert_eq!(
        sum.as_slice::<i32>().unwrap(),
        &[i32::MIN, i32::MAX.wrapping_add(2)]
    );

    let input = Tensor::from_vec_col_major(vec![2, 2], vec![i32::MIN, -1, i32::MAX, 2]).unwrap();
    let prod = reduce_prod(&input, &[0], &strided_kernel::ExecContext::serial()).unwrap();
    assert_eq!(prod.as_slice::<i32>().unwrap(), &[i32::MIN, -2]);

    let input = Tensor::from_vec_col_major(vec![2], vec![i64::MAX, 2]).unwrap();
    let sum = reduce_sum(&input, &[0], &strided_kernel::ExecContext::serial()).unwrap();
    let prod = reduce_prod(&input, &[0], &strided_kernel::ExecContext::serial()).unwrap();
    assert_eq!(sum.as_slice::<i64>().unwrap(), &[i64::MIN.wrapping_add(1)]);
    assert_eq!(prod.as_slice::<i64>().unwrap(), &[-2]);
}

#[test]
fn test_integer_reduce_read_views_wrap_on_overflow() {
    let input =
        TypedTensor::<i32>::from_vec_col_major(vec![2, 2], vec![i32::MAX, 1, i32::MAX, 2]).unwrap();
    let mut backend = CpuBackend::new();

    let sum = backend
        .reduce_sum_read(
            TensorRead::from_view(TensorView::I32(input.as_view())),
            &[0],
        )
        .unwrap();
    assert_eq!(
        sum.as_slice::<i32>().unwrap(),
        &[i32::MIN, i32::MAX.wrapping_add(2)]
    );

    let input = TypedTensor::<i32>::from_vec_col_major(vec![2, 2], vec![i32::MIN, -1, i32::MAX, 2])
        .unwrap();
    let prod = backend
        .reduce_prod_read(
            TensorRead::from_view(TensorView::I32(input.as_view())),
            &[0],
        )
        .unwrap();
    assert_eq!(prod.as_slice::<i32>().unwrap(), &[i32::MIN, -2]);
}

#[test]
fn test_reduce_max_and_min() {
    let t = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );

    let max_cols = reduce_max(&t, &[0]).unwrap();
    assert_eq!(max_cols.shape(), &[3]);
    assert_eq!(get_f64(&max_cols, &[0]), 2.0);
    assert_eq!(get_f64(&max_cols, &[1]), 4.0);
    assert_eq!(get_f64(&max_cols, &[2]), 6.0);

    let max_all = reduce_max(&t, &[0, 1]).unwrap();
    assert!(max_all.shape().is_empty());
    assert_eq!(get_f64(&max_all, &[]), 6.0);

    let min_rows = reduce_min(&t, &[1]).unwrap();
    assert_eq!(min_rows.shape(), &[2]);
    assert_eq!(get_f64(&min_rows, &[0]), 1.0);
    assert_eq!(get_f64(&min_rows, &[1]), 2.0);

    let min_all = reduce_min(&t, &[0, 1]).unwrap();
    assert!(min_all.shape().is_empty());
    assert_eq!(get_f64(&min_all, &[]), 1.0);
}

#[test]
fn test_backend_reduce_prod_max_and_min_delegate_to_cpu_reduction_impls() {
    let t = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let mut backend = CpuBackend::new();

    let prod = backend.reduce_prod(&t, &[0]).unwrap();
    assert_eq!(prod.shape(), &[3]);
    assert_eq!(get_f64(&prod, &[0]), 2.0);
    assert_eq!(get_f64(&prod, &[1]), 12.0);
    assert_eq!(get_f64(&prod, &[2]), 30.0);

    let max = backend.reduce_max(&t, &[1]).unwrap();
    assert_eq!(max.shape(), &[2]);
    assert_eq!(get_f64(&max, &[0]), 5.0);
    assert_eq!(get_f64(&max, &[1]), 6.0);

    let min = backend.reduce_min(&t, &[0, 1]).unwrap();
    assert!(min.shape().is_empty());
    assert_eq!(get_f64(&min, &[]), 1.0);
}

#[test]
fn test_slice() {
    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![4, 4], (1..=16).map(|value| value as f64).collect())
            .unwrap(),
    );
    let mut backend = CpuBackend::new();
    let out = backend
        .slice(
            &input,
            &SliceConfig {
                starts: vec![1, 1],
                limits: vec![3, 3],
                strides: vec![1, 1],
            },
        )
        .unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(get_f64(&out, &[0, 0]), 6.0);
    assert_eq!(get_f64(&out, &[1, 0]), 7.0);
    assert_eq!(get_f64(&out, &[0, 1]), 10.0);
    assert_eq!(get_f64(&out, &[1, 1]), 11.0);
}

#[test]
fn test_reverse_axis_zero() {
    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let mut backend = CpuBackend::new();
    let out = backend.reverse(&input, &[0]).unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(get_f64(&out, &[0, 0]), 2.0);
    assert_eq!(get_f64(&out, &[1, 0]), 1.0);
    assert_eq!(get_f64(&out, &[0, 1]), 4.0);
    assert_eq!(get_f64(&out, &[1, 1]), 3.0);
    assert_eq!(get_f64(&out, &[0, 2]), 6.0);
    assert_eq!(get_f64(&out, &[1, 2]), 5.0);
}

#[test]
fn test_reverse_accepts_i64_data_tensor() {
    let input = Tensor::from_vec_col_major(vec![3], vec![1_i64, 2, 3]).unwrap();
    let mut backend = CpuBackend::new();

    let out = backend.reverse(&input, &[0]).unwrap();

    assert_eq!(out.dtype(), DType::I64);
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.as_slice::<i64>().unwrap(), [3, 2, 1].as_slice());
}

#[test]
fn tensor_index_select_trailing_axis_returns_expected_values() {
    let mut backend = CpuBackend::new();
    let input =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let out = input.index_select(-1, &[2, 0, 2], &mut backend).unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    assert_f64_close(get_f64(&out, &[0, 0]), 5.0);
    assert_f64_close(get_f64(&out, &[1, 0]), 6.0);
    assert_f64_close(get_f64(&out, &[0, 1]), 1.0);
    assert_f64_close(get_f64(&out, &[1, 1]), 2.0);
    assert_f64_close(get_f64(&out, &[0, 2]), 5.0);
    assert_f64_close(get_f64(&out, &[1, 2]), 6.0);
}

#[test]
fn tensor_index_select_rejects_invalid_axis_and_position() {
    let mut backend = CpuBackend::new();
    let input = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();

    let axis_err = input.index_select(-2, &[0], &mut backend).unwrap_err();
    assert!(axis_err.to_string().contains("index_select"));
    assert!(axis_err.to_string().contains("axis"));

    let position_err = input.index_select(0, &[3], &mut backend).unwrap_err();
    assert!(position_err.to_string().contains("index_select"));
    assert!(position_err.to_string().contains("position"));
}

#[test]
fn tensor_stack_trailing_axis_packs_scalars_vectors_and_matrices() {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap();
    let b = Tensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let scalars = Tensor::stack(&[&a, &b], -1, &mut backend).unwrap();
    assert_eq!(scalars.shape(), &[2]);
    assert_eq!(scalars.as_slice::<f64>().unwrap(), &[1.0, 2.0]);

    let v0 = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let v1 = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let vectors = Tensor::stack(&[&v0, &v1], -1, &mut backend).unwrap();
    assert_eq!(vectors.shape(), &[2, 2]);
    assert_f64_close(get_f64(&vectors, &[0, 0]), 1.0);
    assert_f64_close(get_f64(&vectors, &[1, 0]), 2.0);
    assert_f64_close(get_f64(&vectors, &[0, 1]), 3.0);
    assert_f64_close(get_f64(&vectors, &[1, 1]), 4.0);

    let m0 = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]).unwrap();
    let m1 = Tensor::from_vec_col_major(vec![2, 1], vec![3.0_f64, 4.0]).unwrap();
    let matrices = Tensor::stack(&[&m0, &m1], -1, &mut backend).unwrap();
    assert_eq!(matrices.shape(), &[2, 1, 2]);
    assert_f64_close(get_f64(&matrices, &[0, 0, 0]), 1.0);
    assert_f64_close(get_f64(&matrices, &[1, 0, 0]), 2.0);
    assert_f64_close(get_f64(&matrices, &[0, 0, 1]), 3.0);
    assert_f64_close(get_f64(&matrices, &[1, 0, 1]), 4.0);
}

#[test]
fn tensor_index_select_reuses_reclaimed_cpu_buffer() {
    let mut backend = CpuBackend::new();
    let reusable = Tensor::from_vec_col_major(vec![2, 3], vec![0.0_f64; 6]).unwrap();
    let expected_ptr = reusable.as_slice::<f64>().unwrap().as_ptr();
    backend.reclaim_buffer(reusable);

    let input =
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = input.index_select(-1, &[2, 0, 1], &mut backend).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap().as_ptr(), expected_ptr);
}

#[test]
fn tensor_stack_reuses_reclaimed_cpu_buffer() {
    let mut backend = CpuBackend::new();
    let reusable = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4]).unwrap();
    let expected_ptr = reusable.as_slice::<f64>().unwrap().as_ptr();
    backend.reclaim_buffer(reusable);

    let x0 = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let x1 = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let out = Tensor::stack(&[&x0, &x1], -1, &mut backend).unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap().as_ptr(), expected_ptr);
}

#[test]
fn test_reverse_axis_out_of_bounds_returns_error() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap());
    let mut backend = CpuBackend::new();

    let err = backend.reverse(&input, &[1]).unwrap_err();

    assert!(matches!(
        err,
        crate::Error::Validation {
            op: "reverse",
            source: tenferro_tensor::ValidationError::AxisOutOfBounds { axis: 1, rank: 1 },
        }
    ));
}

#[test]
fn test_gather_rejects_fractional_float_indices() {
    let operand = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap(),
    );
    let start_indices =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 1], vec![1.5]).unwrap());
    let mut backend = CpuBackend::new();

    let err = backend
        .gather(&operand, &start_indices, &simple_gather_config())
        .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::Validation {
            op: "index_tensor",
            ..
        }
    ));
}

#[test]
fn test_gather_rejects_complex_indices() {
    let operand = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap(),
    );
    let start_indices = Tensor::C64(
        TypedTensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(1.0, 0.0)]).unwrap(),
    );
    let mut backend = CpuBackend::new();

    let err = backend
        .gather(&operand, &start_indices, &simple_gather_config())
        .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::Validation {
            op: "index_tensor",
            ..
        }
    ));
}

#[test]
fn test_dynamic_slice_rejects_oversized_window() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
    let starts = Tensor::from_vec_col_major(vec![1], vec![0_i64]).unwrap();
    let mut backend = CpuBackend::new();

    let err = backend.dynamic_slice(&input, &starts, &[3]).unwrap_err();

    assert!(matches!(
        err,
        crate::Error::Validation {
            op: "dynamic_slice",
            ..
        }
    ));
}

#[test]
fn test_large_float_index_outside_exact_integer_range_returns_error() {
    let operand = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap(),
    );
    let start_indices = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![1, 1], vec![9_007_199_254_740_995.0f64]).unwrap(),
    );
    let mut backend = CpuBackend::new();

    let err = backend
        .gather(&operand, &start_indices, &simple_gather_config())
        .unwrap_err();

    assert!(matches!(
        err,
        crate::Error::Validation {
            op: "index_tensor",
            ..
        }
    ));
}

#[test]
fn test_invalid_slice_config_returns_error() {
    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let mut backend = CpuBackend::new();

    let err = backend
        .slice(
            &input,
            &SliceConfig {
                starts: vec![0, 0, 0],
                limits: vec![2, 2],
                strides: vec![1, 1],
            },
        )
        .unwrap_err();
    assert!(matches!(err, crate::Error::Validation { op: "slice", .. }));
}

#[test]
fn test_invalid_pad_config_returns_error() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
    let mut backend = CpuBackend::new();

    let err = backend
        .pad(
            &input,
            &PadConfig {
                edge_padding_low: vec![0],
                edge_padding_high: vec![0, 0],
                interior_padding: vec![0],
            },
        )
        .unwrap_err();
    assert!(matches!(err, crate::Error::Validation { op: "pad", .. }));
}

#[test]
fn test_gather_rejects_malformed_offset_dims() {
    let operand = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let start_indices = Tensor::from_vec_col_major(vec![3, 1], vec![0_i64, 1, 2]).unwrap();
    let mut backend = CpuBackend::new();
    let config = GatherConfig {
        offset_dims: vec![2],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, 2],
    };

    let err = backend
        .gather(&operand, &start_indices, &config)
        .unwrap_err();
    assert!(matches!(err, crate::Error::Validation { op: "gather", .. }));
}

#[test]
fn test_scatter_rejects_update_window_dim_out_of_bounds() {
    let operand = Tensor::F64(TypedTensor::zeros(vec![3, 3, 3]).unwrap());
    let scatter_indices =
        Tensor::from_vec_col_major(vec![3, 2], vec![0_i64, 0, 1, 1, 2, 2]).unwrap();
    let updates =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![3, 3, 3], vec![0.0; 27]).unwrap());
    let mut backend = CpuBackend::new();
    let config = ScatterConfig {
        update_window_dims: vec![0, 3],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![1, 2],
        index_vector_dim: 1,
    };

    let err = backend
        .scatter(&operand, &scatter_indices, &updates, &config)
        .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::Validation {
            op: "scatter",
            source: tenferro_tensor::ValidationError::AxisOutOfBounds { axis: 3, .. },
        }
    ));
}

#[test]
fn test_scatter_rejects_too_many_update_window_dims() {
    let operand = Tensor::F64(TypedTensor::zeros(vec![3, 3, 3]).unwrap());
    let scatter_indices =
        Tensor::from_vec_col_major(vec![3, 2], vec![0_i64, 0, 1, 1, 2, 2]).unwrap();
    let updates = Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![0.0; 3]).unwrap());
    let mut backend = CpuBackend::new();
    let config = ScatterConfig {
        update_window_dims: vec![0, 1],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![1, 2],
        index_vector_dim: 1,
    };

    let err = backend
        .scatter(&operand, &scatter_indices, &updates, &config)
        .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::Validation { op: "scatter", source }
        if source.to_string().contains("exceeds update rank")
    ));
}

#[test]
fn test_concatenate_axis_zero() {
    let lhs = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let rhs = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2, 3], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap(),
    );
    let mut backend = CpuBackend::new();
    let out = backend.concatenate(&[&lhs, &rhs], 0).unwrap();

    assert_eq!(out.shape(), &[4, 3]);
    assert_eq!(get_f64(&out, &[0, 0]), 1.0);
    assert_eq!(get_f64(&out, &[1, 0]), 2.0);
    assert_eq!(get_f64(&out, &[2, 0]), 7.0);
    assert_eq!(get_f64(&out, &[3, 0]), 8.0);
    assert_eq!(get_f64(&out, &[0, 1]), 3.0);
    assert_eq!(get_f64(&out, &[1, 1]), 4.0);
    assert_eq!(get_f64(&out, &[2, 1]), 9.0);
    assert_eq!(get_f64(&out, &[3, 1]), 10.0);
    assert_eq!(get_f64(&out, &[0, 2]), 5.0);
    assert_eq!(get_f64(&out, &[1, 2]), 6.0);
    assert_eq!(get_f64(&out, &[2, 2]), 11.0);
    assert_eq!(get_f64(&out, &[3, 2]), 12.0);
}
