use super::*;

fn assert_ordered_complex_error<T>(result: crate::Result<T>, op: &'static str) {
    assert!(matches!(
        result,
        Err(crate::Error::Unsupported {
            op: actual,
            message,
        }) if actual == op && message.contains("total order")
    ));
}

#[test]
fn elementwise_add_accepts_transposed_host_view_input() {
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = a.as_view().transpose_view([1, 0]).unwrap();
    let mut backend = CpuBackend::new();

    let out = backend
        .add_read(
            TensorRead::from_view(TensorView::F64(b.clone())),
            TensorRead::from_view(TensorView::F64(b)),
        )
        .unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 6.0, 4.0, 8.0]);
}

#[test]
fn elementwise_add_read_promotes_rank0_f64_view_with_c64_tensor() {
    let scalar = TypedTensor::<f64>::from_vec_col_major(vec![], vec![2.0]).unwrap();
    let rhs = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, -1.0), Complex64::new(-3.0, 0.5)],
        )
        .unwrap(),
    );
    let mut backend = CpuBackend::new();

    let out = backend
        .add_read(
            TensorRead::from_view(TensorView::F64(scalar.as_view())),
            TensorRead::from_tensor(&rhs),
        )
        .unwrap();

    assert_eq!(out.shape(), &[2]);
    assert_eq!(
        out.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(3.0, -1.0), Complex64::new(-1.0, 0.5)]
    );
}

#[test]
fn elementwise_add_read_promotes_c32_tensor_with_rank0_f32_view() {
    let lhs = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(1.0, -1.0), Complex32::new(-3.0, 0.5)],
        )
        .unwrap(),
    );
    let scalar = TypedTensor::<f32>::from_vec_col_major(vec![], vec![2.0]).unwrap();
    let mut backend = CpuBackend::new();

    let out = backend
        .add_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_view(TensorView::F32(scalar.as_view())),
        )
        .unwrap();

    assert_eq!(out.shape(), &[2]);
    assert_eq!(
        out.as_slice::<Complex32>().unwrap(),
        &[Complex32::new(3.0, -1.0), Complex32::new(-1.0, 0.5)]
    );
}

#[test]
fn reduce_sum_accepts_transposed_host_view_input() {
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = a.as_view().transpose_view([1, 0]).unwrap();
    let mut backend = CpuBackend::new();

    let out = backend
        .reduce_sum_read(TensorRead::from_view(TensorView::F64(b)), &[0])
        .unwrap();

    assert_eq!(out.shape(), &[2]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
}

#[test]
fn reduce_read_views_cover_dtype_and_validation_branches() {
    let mut backend = CpuBackend::new();

    let f32s =
        TypedTensor::<f32>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(
        backend
            .reduce_sum_read(TensorRead::from_view(TensorView::F32(f32s.as_view())), &[0])
            .unwrap()
            .as_slice::<f32>()
            .unwrap(),
        &[3.0, 7.0]
    );
    assert_eq!(
        backend
            .reduce_max_read(TensorRead::from_view(TensorView::F32(f32s.as_view())), &[0])
            .unwrap()
            .as_slice::<f32>()
            .unwrap(),
        &[2.0, 4.0]
    );

    let f64s =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(
        backend
            .reduce_min_read(TensorRead::from_view(TensorView::F64(f64s.as_view())), &[0])
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[1.0, 3.0]
    );

    let i32s = TypedTensor::<i32>::from_vec_col_major(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
    assert_eq!(
        backend
            .reduce_sum_read(TensorRead::from_view(TensorView::I32(i32s.as_view())), &[0])
            .unwrap()
            .as_slice::<i32>()
            .unwrap(),
        &[3, 7]
    );

    let i64s = TypedTensor::<i64>::from_vec_col_major(vec![2, 2], vec![1, 2, 3, 4]).unwrap();
    assert_eq!(
        backend
            .reduce_prod_read(TensorRead::from_view(TensorView::I64(i64s.as_view())), &[0])
            .unwrap()
            .as_slice::<i64>()
            .unwrap(),
        &[2, 12]
    );

    let c32s = TypedTensor::<Complex32>::from_vec_col_major(
        vec![2],
        vec![Complex32::new(1.0, 1.0), Complex32::new(2.0, -1.0)],
    )
    .unwrap();
    assert_eq!(
        backend
            .reduce_sum_read(TensorRead::from_view(TensorView::C32(c32s.as_view())), &[0])
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap(),
        &[Complex32::new(3.0, 0.0)]
    );

    let c64s = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
    )
    .unwrap();
    assert_eq!(
        backend
            .reduce_prod_read(TensorRead::from_view(TensorView::C64(c64s.as_view())), &[0])
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap(),
        &[Complex64::new(3.0, 1.0)]
    );

    let bools = TypedTensor::<bool>::from_vec_col_major(vec![2], vec![true, false]).unwrap();
    assert!(matches!(
        backend.reduce_sum_read(
            TensorRead::from_view(TensorView::Bool(bools.as_view())),
            &[0]
        ),
        Err(crate::Error::Unsupported {
            op: "reduce_sum",
            ..
        })
    ));
    assert!(matches!(
        backend.reduce_prod_read(
            TensorRead::from_view(TensorView::Bool(bools.as_view())),
            &[0]
        ),
        Err(crate::Error::Unsupported {
            op: "reduce_prod",
            ..
        })
    ));
    assert_eq!(
        backend
            .reduce_max_read(TensorRead::from_view(TensorView::I32(i32s.as_view())), &[0])
            .unwrap()
            .as_slice::<i32>()
            .unwrap(),
        &[2, 4]
    );
    assert!(matches!(
        backend.reduce_min_read(TensorRead::from_view(TensorView::C32(c32s.as_view())), &[0]),
        Err(crate::Error::Unsupported {
            op: "reduce_min",
            ..
        })
    ));
    assert!(matches!(
        backend.reduce_sum_read(TensorRead::from_view(TensorView::F64(f64s.as_view())), &[2]),
        Err(crate::Error::Validation {
            op: "reduce_sum",
            source: tenferro_tensor::ValidationError::AxisOutOfBounds { axis: 2, rank: 2 }
        })
    ));
    assert!(matches!(
        backend.reduce_prod_read(
            TensorRead::from_view(TensorView::F64(f64s.as_view())),
            &[0, 0]
        ),
        Err(crate::Error::Validation {
            op: "reduce_prod",
            source: tenferro_tensor::ValidationError::DuplicateAxis {
                axis: 0,
                role: "axes"
            }
        })
    ));
}

#[test]
fn reduce_read_empty_axes_materializes_views_for_all_dtypes() {
    let mut backend = CpuBackend::new();

    let f32s = TypedTensor::<f32>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    assert_eq!(
        backend
            .reduce_max_read(TensorRead::from_view(TensorView::F32(f32s.as_view())), &[])
            .unwrap()
            .as_slice::<f32>()
            .unwrap(),
        &[1.0, 2.0]
    );

    let f64s = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    assert_eq!(
        backend
            .reduce_min_read(TensorRead::from_view(TensorView::F64(f64s.as_view())), &[])
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[1.0, 2.0]
    );

    let i32s = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();
    assert_eq!(
        backend
            .reduce_max_read(TensorRead::from_view(TensorView::I32(i32s.as_view())), &[])
            .unwrap()
            .as_slice::<i32>()
            .unwrap(),
        &[1, 2]
    );

    let i64s = TypedTensor::<i64>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();
    assert_eq!(
        backend
            .reduce_min_read(TensorRead::from_view(TensorView::I64(i64s.as_view())), &[])
            .unwrap()
            .as_slice::<i64>()
            .unwrap(),
        &[1, 2]
    );

    let bools = TypedTensor::<bool>::from_vec_col_major(vec![2], vec![true, false]).unwrap();
    assert_eq!(
        backend
            .reduce_max_read(
                TensorRead::from_view(TensorView::Bool(bools.as_view())),
                &[]
            )
            .unwrap()
            .as_slice::<bool>()
            .unwrap(),
        &[true, false]
    );

    let c32s =
        TypedTensor::<Complex32>::from_vec_col_major(vec![1], vec![Complex32::new(1.0, -1.0)])
            .unwrap();
    assert_eq!(
        backend
            .reduce_min_read(TensorRead::from_view(TensorView::C32(c32s.as_view())), &[])
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap(),
        &[Complex32::new(1.0, -1.0)]
    );

    let c64s =
        TypedTensor::<Complex64>::from_vec_col_major(vec![1], vec![Complex64::new(2.0, 3.0)])
            .unwrap();
    assert_eq!(
        backend
            .reduce_max_read(TensorRead::from_view(TensorView::C64(c64s.as_view())), &[])
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap(),
        &[Complex64::new(2.0, 3.0)]
    );
}

#[test]
fn reduce_read_tensors_cover_host_dtype_dispatch() {
    let mut backend = CpuBackend::new();

    let f32s =
        Tensor::F32(TypedTensor::<f32>::from_vec_col_major(vec![2], vec![2.0, 3.0]).unwrap());
    assert_eq!(
        backend
            .reduce_sum_read(TensorRead::from_tensor(&f32s), &[0])
            .unwrap()
            .as_slice::<f32>()
            .unwrap(),
        &[5.0]
    );
    assert_eq!(
        backend
            .reduce_max_read(TensorRead::from_tensor(&f32s), &[0])
            .unwrap()
            .as_slice::<f32>()
            .unwrap(),
        &[3.0]
    );

    let f64s =
        Tensor::F64(TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 3.0]).unwrap());
    assert_eq!(
        backend
            .reduce_prod_read(TensorRead::from_tensor(&f64s), &[0])
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[6.0]
    );
    assert_eq!(
        backend
            .reduce_min_read(TensorRead::from_tensor(&f64s), &[0])
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[2.0]
    );

    let i32s = Tensor::I32(TypedTensor::<i32>::from_vec_col_major(vec![2], vec![2, 3]).unwrap());
    assert_eq!(
        backend
            .reduce_prod_read(TensorRead::from_tensor(&i32s), &[0])
            .unwrap()
            .as_slice::<i32>()
            .unwrap(),
        &[6]
    );
    assert_eq!(
        backend
            .reduce_max_read(TensorRead::from_tensor(&i32s), &[0])
            .unwrap()
            .as_slice::<i32>()
            .unwrap(),
        &[3]
    );

    let i64s = Tensor::I64(TypedTensor::<i64>::from_vec_col_major(vec![2], vec![2, 3]).unwrap());
    assert_eq!(
        backend
            .reduce_sum_read(TensorRead::from_tensor(&i64s), &[0])
            .unwrap()
            .as_slice::<i64>()
            .unwrap(),
        &[5]
    );
    assert_eq!(
        backend
            .reduce_min_read(TensorRead::from_tensor(&i64s), &[0])
            .unwrap()
            .as_slice::<i64>()
            .unwrap(),
        &[2]
    );

    let bools =
        Tensor::Bool(TypedTensor::<bool>::from_vec_col_major(vec![2], vec![true, false]).unwrap());
    assert!(matches!(
        backend.reduce_sum_read(TensorRead::from_tensor(&bools), &[0]),
        Err(crate::Error::Unsupported {
            op: "reduce_sum",
            ..
        })
    ));
    assert!(matches!(
        backend.reduce_prod_read(TensorRead::from_tensor(&bools), &[0]),
        Err(crate::Error::Unsupported {
            op: "reduce_prod",
            ..
        })
    ));

    let c32s = Tensor::C32(
        TypedTensor::<Complex32>::from_vec_col_major(
            vec![2],
            vec![Complex32::new(1.0, 1.0), Complex32::new(2.0, -1.0)],
        )
        .unwrap(),
    );
    assert_eq!(
        backend
            .reduce_sum_read(TensorRead::from_tensor(&c32s), &[0])
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap(),
        &[Complex32::new(3.0, 0.0)]
    );

    let c64s = Tensor::C64(
        TypedTensor::<Complex64>::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
        )
        .unwrap(),
    );
    assert_eq!(
        backend
            .reduce_prod_read(TensorRead::from_tensor(&c64s), &[0])
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap(),
        &[Complex64::new(3.0, 1.0)]
    );
}

#[test]
fn test_direct_elementwise_helpers_cover_f32_c32_and_error_paths() {
    let lhs_f32 =
        Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![8.0f32, -2.0]).unwrap());
    let rhs_f32 = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![2.0f32, 5.0]).unwrap());
    let pred_bool =
        Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![false, true]).unwrap());
    let lower_f32 =
        Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![-1.0f32, -1.0]).unwrap());
    let upper_f32 =
        Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 4.0]).unwrap());

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

    let input_c32 = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(3.0, 4.0), Complex32::new(0.0, 0.0)],
        )
        .unwrap(),
    );
    let lhs_c32 = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(3.0, 4.0), Complex32::new(1.0, 0.0)],
        )
        .unwrap(),
    );
    let rhs_c32 = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 2.0)],
        )
        .unwrap(),
    );
    let lower_c32 = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(0.5, 0.0), Complex32::new(0.5, 0.0)],
        )
        .unwrap(),
    );
    let upper_c32 = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(4.0, 0.0), Complex32::new(2.0, 2.0)],
        )
        .unwrap(),
    );

    let abs_c32 = abs(&input_c32).unwrap();
    assert_eq!(abs_c32.dtype(), DType::F32);
    assert_eq!(get_f32(&abs_c32, &[0]), 5.0);
    assert_eq!(get_f32(&abs_c32, &[1]), 0.0);

    let sign_c32 = sign(&input_c32).unwrap();
    assert_eq!(get_c32(&sign_c32, &[1]), Complex32::new(0.0, 0.0));

    assert_ordered_complex_error(maximum(&lhs_c32, &rhs_c32), "maximum");
    assert_ordered_complex_error(minimum(&lhs_c32, &rhs_c32), "minimum");
    let cmp_c32 = compare(&lhs_c32, &rhs_c32, &CompareDir::Eq).unwrap();
    assert!(!get_bool(&cmp_c32, &[0]));
    assert!(!get_bool(&cmp_c32, &[1]));

    let select_c32 = select(&pred_bool, &lhs_c32, &rhs_c32).unwrap();
    assert_eq!(get_c32(&select_c32, &[0]), Complex32::new(1.0, 0.0));
    assert_eq!(get_c32(&select_c32, &[1]), Complex32::new(1.0, 0.0));

    assert_ordered_complex_error(clamp(&lhs_c32, &lower_c32, &upper_c32), "clamp");

    let scalar_f32 = Tensor::F32(TypedTensor::from_vec_col_major(vec![], vec![2.0f32]).unwrap());
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
            &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap())
        ),
        Err(crate::Error::Validation {
            op: "div",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
    assert!(matches!(
        clamp(
            &lhs_f32,
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![1], vec![0.0f32]).unwrap()),
            &upper_f32
        ),
        Err(crate::Error::Validation {
            op: "clamp",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));
}

#[test]
fn equal_bool_add_and_mul_report_unsupported_dtype_not_mismatch() {
    let lhs = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, false]).unwrap());
    let rhs = Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, true]).unwrap());

    assert!(matches!(
        add(&lhs, &rhs),
        Err(crate::Error::Unsupported {
            op: "add",
            message,
        }) if message == "unsupported dtype Bool"
    ));
    assert!(matches!(
        mul(&lhs, &rhs),
        Err(crate::Error::Unsupported {
            op: "mul",
            message,
        }) if message == "unsupported dtype Bool"
    ));
}

#[test]
fn maximum_and_minimum_propagate_nan_independent_of_argument_order() {
    let nan_lhs =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![f64::NAN, 1.0]).unwrap());
    let nan_rhs =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, f64::NAN]).unwrap());

    let max_lr = maximum(&nan_lhs, &nan_rhs).unwrap();
    let max_rl = maximum(&nan_rhs, &nan_lhs).unwrap();
    let min_lr = minimum(&nan_lhs, &nan_rhs).unwrap();
    let min_rl = minimum(&nan_rhs, &nan_lhs).unwrap();

    for tensor in [&max_lr, &max_rl, &min_lr, &min_rl] {
        let data = tensor.as_slice::<f64>().unwrap();
        assert!(data[0].is_nan(), "expected NaN at index 0, got {:?}", data);
        assert!(data[1].is_nan(), "expected NaN at index 1, got {:?}", data);
    }
}

#[test]
fn reduce_max_and_min_propagate_nan_instead_of_leaking_sentinel() {
    let mixed =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![f64::NAN, 1.0, 2.0]).unwrap());
    let all_nan =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![f64::NAN, f64::NAN]).unwrap());

    assert!(reduce_max(&mixed, &[0]).unwrap().as_slice::<f64>().unwrap()[0].is_nan());
    assert!(reduce_min(&mixed, &[0]).unwrap().as_slice::<f64>().unwrap()[0].is_nan());
    assert!(reduce_max(&all_nan, &[0])
        .unwrap()
        .as_slice::<f64>()
        .unwrap()[0]
        .is_nan());
    assert!(reduce_min(&all_nan, &[0])
        .unwrap()
        .as_slice::<f64>()
        .unwrap()[0]
        .is_nan());
}

#[test]
fn reduce_sum_zero_length_axis_is_rejected_like_other_reductions() {
    let empty = Tensor::F64(TypedTensor::from_vec_col_major(vec![0], Vec::<f64>::new()).unwrap());

    assert!(matches!(
        reduce_sum(&empty, &[0]),
        Err(crate::Error::Validation {
            op: "reduce_sum",
            ..
        })
    ));
    assert!(matches!(
        reduce_prod(&empty, &[0]),
        Err(crate::Error::Validation {
            op: "reduce_prod",
            ..
        })
    ));
    assert!(matches!(
        reduce_max(&empty, &[0]),
        Err(crate::Error::Validation {
            op: "reduce_max",
            ..
        })
    ));
    assert!(matches!(
        reduce_min(&empty, &[0]),
        Err(crate::Error::Validation {
            op: "reduce_min",
            ..
        })
    ));
}

#[test]
fn empty_reduction_axes_are_noop_before_dtype_dispatch() {
    let bool_tensor = TypedTensor::from_vec_col_major(vec![2], vec![true, false]).unwrap();
    let bools = Tensor::Bool(bool_tensor.clone());
    let mut backend = CpuBackend::new();

    let sum_owned = reduce_sum(&bools, &[]).unwrap();
    let prod_owned = reduce_prod(&bools, &[]).unwrap();
    let sum_read = backend
        .reduce_sum_read(TensorRead::from_tensor(&bools), &[])
        .unwrap();
    let prod_read = backend
        .reduce_prod_read(
            TensorRead::from_view(TensorView::Bool(bool_tensor.as_view())),
            &[],
        )
        .unwrap();

    for tensor in [sum_owned, prod_owned, sum_read, prod_read] {
        assert_eq!(tensor.shape(), &[2]);
        assert_eq!(tensor.as_slice::<bool>().unwrap(), &[true, false]);
    }
}

#[test]
fn test_direct_elementwise_helpers_cover_f64_c64_dispatch_and_mismatch_paths() {
    let lhs_f64 =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.5f64, -3.0]).unwrap());
    let rhs_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![2.0f64, 4.0]).unwrap());
    let scalar_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![2.0f64]).unwrap());
    let pred_bool =
        Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![false, true]).unwrap());
    let lower_f64 =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![0.0f64, -2.0]).unwrap());
    let upper_f64 =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![2.0f64, 3.0]).unwrap());
    let short_f64 = Tensor::F64(TypedTensor::from_vec_col_major(vec![1], vec![1.0f64]).unwrap());
    let lhs_i32 = Tensor::I32(TypedTensor::from_vec_col_major(vec![2], vec![1i32, 3]).unwrap());
    let rhs_i32 = Tensor::I32(TypedTensor::from_vec_col_major(vec![2], vec![2i32, 3]).unwrap());
    let lhs_i64 = Tensor::I64(TypedTensor::from_vec_col_major(vec![2], vec![5i64, -1]).unwrap());
    let rhs_i64 = Tensor::I64(TypedTensor::from_vec_col_major(vec![2], vec![2i64, -1]).unwrap());
    let lhs_bool =
        Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, false]).unwrap());
    let rhs_bool =
        Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![false, false]).unwrap());

    let add_out = add(&lhs_f64, &rhs_f64).unwrap();
    assert_eq!(get_f64(&add_out, &[0]), 3.5);
    assert_eq!(get_f64(&add_out, &[1]), 1.0);

    let mul_out = mul(&lhs_f64, &rhs_f64).unwrap();
    assert_eq!(get_f64(&mul_out, &[0]), 3.0);
    assert_eq!(get_f64(&mul_out, &[1]), -12.0);

    let div_out = div(
        &rhs_f64,
        &Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![2.0, 2.0]).unwrap()),
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

    let lhs_c64 = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, 0.0)],
        )
        .unwrap(),
    );
    let rhs_c64 = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 2.0)],
        )
        .unwrap(),
    );
    let lower_c64 = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.0)],
        )
        .unwrap(),
    );
    let upper_c64 = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(4.0, 0.0), Complex64::new(2.0, 2.0)],
        )
        .unwrap(),
    );

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
        &Tensor::C64(
            TypedTensor::from_vec_col_major(
                vec![2],
                vec![Complex64::new(1.0, 1.0), Complex64::new(1.0, 0.0)],
            )
            .unwrap(),
        ),
    )
    .unwrap();
    assert_c64_close(get_c64(&div_c64, &[0]), Complex64::new(3.5, 0.5));
    assert_c64_close(get_c64(&div_c64, &[1]), Complex64::new(1.0, 0.0));

    let neg_c64 = neg(&lhs_c64).unwrap();
    assert_c64_close(get_c64(&neg_c64, &[0]), Complex64::new(-3.0, -4.0));
    let conj_c64 = conj(&lhs_c64).unwrap();
    assert_c64_close(get_c64(&conj_c64, &[0]), Complex64::new(3.0, -4.0));

    let cmp_c64 = compare(&lhs_c64, &lhs_c64, &CompareDir::Eq).unwrap();
    assert!(get_bool(&cmp_c64, &[0]));
    assert!(get_bool(&cmp_c64, &[1]));

    assert_ordered_complex_error(compare(&lhs_c64, &rhs_c64, &CompareDir::Lt), "compare");
    assert_ordered_complex_error(compare(&lhs_c64, &rhs_c64, &CompareDir::Le), "compare");
    assert_ordered_complex_error(compare(&lhs_c64, &rhs_c64, &CompareDir::Gt), "compare");
    assert_ordered_complex_error(compare(&lhs_c64, &rhs_c64, &CompareDir::Ge), "compare");

    let select_c64 = select(&pred_bool, &lhs_c64, &rhs_c64).unwrap();
    assert_c64_close(get_c64(&select_c64, &[0]), Complex64::new(1.0, 0.0));
    assert_c64_close(get_c64(&select_c64, &[1]), Complex64::new(1.0, 0.0));

    assert_ordered_complex_error(clamp(&lhs_c64, &lower_c64, &upper_c64), "clamp");

    let lhs_f32 = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]).unwrap());

    assert!(matches!(
        add(&lhs_f32, &rhs_f64),
        Err(crate::Error::Validation {
            op: "add",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
    assert!(matches!(
        mul(&lhs_f32, &rhs_f64),
        Err(crate::Error::Validation {
            op: "mul",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
    assert!(matches!(
        maximum(&lhs_f32, &rhs_f64),
        Err(crate::Error::Validation {
            op: "maximum",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
    assert!(matches!(
        minimum(&lhs_f32, &rhs_f64),
        Err(crate::Error::Validation {
            op: "minimum",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
    assert!(matches!(
        compare(&lhs_f32, &rhs_f64, &CompareDir::Eq),
        Err(crate::Error::Validation {
            op: "compare",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
    assert!(matches!(
        select(&lhs_f32, &lhs_f32, &rhs_f64),
        Err(crate::Error::Validation {
            op: "select",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));
    assert!(matches!(
        clamp(&lhs_f32, &lhs_f32, &rhs_f64),
        Err(crate::Error::Validation {
            op: "clamp",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));

    assert!(matches!(
        add(&lhs_f64, &short_f64),
        Err(crate::Error::Validation {
            op: "add",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));
    assert!(matches!(
        mul(&lhs_f64, &short_f64),
        Err(crate::Error::Validation {
            op: "mul",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));
    assert!(matches!(
        div(&lhs_f64, &short_f64),
        Err(crate::Error::Validation {
            op: "div",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));
    assert!(matches!(
        maximum(&lhs_f64, &short_f64),
        Err(crate::Error::Validation {
            op: "maximum",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));
    assert!(matches!(
        minimum(&lhs_f64, &short_f64),
        Err(crate::Error::Validation {
            op: "minimum",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));
    assert!(matches!(
        compare(&lhs_f64, &short_f64, &CompareDir::Eq),
        Err(crate::Error::Validation {
            op: "compare",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));
    assert!(matches!(
        select(&pred_bool, &short_f64, &rhs_f64),
        Err(crate::Error::Validation {
            op: "select",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));
    assert!(matches!(
        select(&pred_bool, &rhs_f64, &short_f64),
        Err(crate::Error::Validation {
            op: "select",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));
    assert!(matches!(
        clamp(&lhs_f64, &lower_f64, &short_f64),
        Err(crate::Error::Validation {
            op: "clamp",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));
}

#[test]
fn test_reduction_helpers_cover_complex_and_error_paths() {
    let complex = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex32::new(1.0, 1.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(3.0, -1.0),
                Complex32::new(4.0, 2.0),
            ],
        )
        .unwrap(),
    );
    let sum = reduce_sum(&complex, &[0]).unwrap();
    assert_eq!(get_c32(&sum, &[0]), Complex32::new(3.0, 1.0));
    assert_eq!(get_c32(&sum, &[1]), Complex32::new(7.0, 1.0));

    let prod = reduce_prod(&complex, &[]).unwrap();
    assert_eq!(prod.shape(), &[2, 2]);

    assert!(matches!(
        reduce_sum(
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]).unwrap()),
            &[2]
        ),
        Err(crate::Error::Validation {
            op: "reduce_sum",
            source: tenferro_tensor::ValidationError::AxisOutOfBounds { .. },
        })
    ));
    assert!(matches!(
        reduce_prod(
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]).unwrap()),
            &[0, 0]
        ),
        Err(crate::Error::Validation {
            op: "reduce_prod",
            source: tenferro_tensor::ValidationError::DuplicateAxis { .. },
        })
    ));
    assert!(matches!(
        reduce_max(&complex, &[0]),
        Err(crate::Error::Unsupported {
            op: "reduce_max",
            ..
        })
    ));
    assert!(matches!(
        reduce_min(&complex, &[0]),
        Err(crate::Error::Unsupported {
            op: "reduce_min",
            ..
        })
    ));

    let real = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]).unwrap());
    assert!(matches!(
        reduce_max(&real, &[2]),
        Err(crate::Error::Validation {
            op: "reduce_max",
            source: tenferro_tensor::ValidationError::AxisOutOfBounds { .. },
        })
    ));
    assert!(matches!(
        reduce_min(&real, &[0, 0]),
        Err(crate::Error::Validation {
            op: "reduce_min",
            source: tenferro_tensor::ValidationError::DuplicateAxis { .. },
        })
    ));
}

#[test]
fn test_structural_helpers_cover_f32_success_and_error_paths() {
    let matrix = Tensor::F32(
        TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(),
    );
    let transposed = transpose(&matrix, &[1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(get_f32(&transposed, &[1, 0]), 3.0);

    let scalar = Tensor::F32(TypedTensor::from_vec_col_major(vec![], vec![5.0f32]).unwrap());
    let broadcast = broadcast_in_dim(&scalar, &[2], &[]).unwrap();
    assert_eq!(get_f32(&broadcast, &[1]), 5.0);

    let diag = extract_diagonal(&matrix, 0, 1).unwrap();
    assert_eq!(get_f32(&diag, &[0]), 1.0);
    assert_eq!(get_f32(&diag, &[1]), 4.0);

    let embedded = embed_diagonal(
        &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![7.0f32, 8.0]).unwrap()),
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
        Err(crate::Error::Validation {
            op: "transpose",
            source: tenferro_tensor::ValidationError::RankMismatch { .. },
        })
    ));
    assert!(matches!(
        transpose(&matrix, &[0, 0]),
        Err(crate::Error::Validation {
            op: "transpose",
            source: tenferro_tensor::ValidationError::DuplicateAxis { .. },
        })
    ));
    assert!(matches!(
        broadcast_in_dim(
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]).unwrap()),
            &[3, 2],
            &[0]
        ),
        Err(crate::Error::Validation {
            op: "broadcast_in_dim",
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
        })
    ));
    assert!(matches!(
        extract_diagonal(&matrix, 1, 1),
        Err(crate::Error::Validation {
            op: "extract_diagonal",
            source: tenferro_tensor::ValidationError::DuplicateAxis { .. },
        })
    ));
    assert!(matches!(
        embed_diagonal(
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]).unwrap()),
            0,
            2
        ),
        Err(crate::Error::Validation {
            op: "embed_diagonal",
            source: tenferro_tensor::ValidationError::AxisOutOfBounds { .. },
        })
    ));
    let vector = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0f32, 2.0]).unwrap());
    assert!(matches!(
        tril(&vector, 0),
        Err(crate::Error::Validation {
            op: "tril",
            source: tenferro_tensor::ValidationError::RankMismatch { .. },
        })
    ));
    assert!(matches!(
        triu(&vector, 0),
        Err(crate::Error::Validation {
            op: "triu",
            source: tenferro_tensor::ValidationError::RankMismatch { .. },
        })
    ));
}
