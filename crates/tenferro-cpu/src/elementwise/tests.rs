use super::*;

#[test]
fn elementwise_fusion_validation_covers_descriptor_errors_and_empty_outputs() {
    use tenferro_tensor::backend::ElementwiseFusionInst;

    let input = Tensor::F32(TypedTensor::<f32>::from_vec_col_major(vec![1], vec![1.0]).unwrap());

    let wrong_input_count = ElementwiseFusionPlan::new(
        DType::F32,
        2,
        vec![2],
        vec![ElementwiseFusionInst::new(
            ElementwiseFusionOp::Add,
            vec![0, 1],
        )],
    );
    assert!(matches!(
        validate_elementwise_fusion_inputs(&[&input], &wrong_input_count),
        Err(crate::Error::BackendFailure {
            op: ELEMENTWISE_FUSION_OP,
            ..
        })
    ));

    let empty_outputs = ElementwiseFusionPlan::new(DType::F32, 1, Vec::new(), Vec::new());
    assert!(!validate_elementwise_fusion_inputs(&[&input], &empty_outputs).unwrap());

    let dtype_mismatch = ElementwiseFusionPlan::new(DType::F64, 1, vec![0], Vec::new());
    assert!(matches!(
        validate_elementwise_fusion_inputs(&[&input], &dtype_mismatch),
        Err(crate::Error::DTypeMismatch {
            op: ELEMENTWISE_FUSION_OP,
            lhs: DType::F32,
            rhs: DType::F64,
        })
    ));

    let remainder_plan = ElementwiseFusionPlan::new(
        DType::F32,
        2,
        vec![2],
        vec![ElementwiseFusionInst::new(
            ElementwiseFusionOp::Remainder,
            vec![0, 1],
        )],
    );
    assert!(plan_uses_unfused_op(&remainder_plan));
    assert!(!plan_uses_ordered_op(&remainder_plan));

    let maximum_plan = ElementwiseFusionPlan::new(
        DType::F32,
        2,
        vec![2],
        vec![ElementwiseFusionInst::new(
            ElementwiseFusionOp::Maximum,
            vec![0, 1],
        )],
    );
    assert!(plan_uses_ordered_op(&maximum_plan));
    assert!(reject_complex_ordered_dtypes("maximum", &[DType::F32]).is_ok());
    assert!(matches!(
        reject_complex_ordered_dtypes("maximum", &[DType::C64]),
        Err(crate::Error::InvalidConfig { op: "maximum", .. })
    ));
}

#[test]
fn rank_n_outer_product_fast_path_accepts_matrix_operands() {
    let mut buffers = BufferPool::default();
    let lhs_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_data = [7.0_f64, 8.0, 9.0, 10.0];
    let lhs = TypedTensorView::from_slice([2, 3], [1, 2], 0, &lhs_data).unwrap();
    let rhs = TypedTensorView::from_slice([2, 2], [1, 2], 0, &rhs_data).unwrap();

    let out = try_outer_product_with_pool(
        &mut buffers,
        &lhs,
        &[2, 3, 2, 2],
        &[0, 1],
        &rhs,
        &[2, 3, 2, 2],
        &[2, 3],
    )
    .unwrap()
    .expect("rank-N x rank-M pure outer products should use the fast path");

    assert_eq!(out.shape(), &[2, 3, 2, 2]);
    let expected: Vec<f64> = (0..2)
        .flat_map(|d| {
            (0..2).flat_map(move |c| {
                (0..3).flat_map(move |b| {
                    (0..2).map(move |a| lhs_data[a + 2 * b] * rhs_data[c + 2 * d])
                })
            })
        })
        .collect();
    assert_eq!(out.as_slice().unwrap(), expected.as_slice());
}

#[test]
fn batched_outer_product_fast_path_accepts_shared_batch_axis() {
    let mut buffers = BufferPool::default();
    let lhs_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_data = [
        7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    ];
    let lhs = TypedTensorView::from_slice([2, 3], [1, 2], 0, &lhs_data).unwrap();
    let rhs = TypedTensorView::from_slice([4, 3], [1, 4], 0, &rhs_data).unwrap();

    let out = try_outer_product_with_pool(
        &mut buffers,
        &lhs,
        &[2, 4, 3],
        &[0, 2],
        &rhs,
        &[2, 4, 3],
        &[1, 2],
    )
    .unwrap()
    .expect("shared-batch outer products should use the fast path");

    assert_eq!(out.shape(), &[2, 4, 3]);
    let expected: Vec<f64> = (0..3)
        .flat_map(|t| {
            (0..4).flat_map(move |o| (0..2).map(move |j| lhs_data[j + 2 * t] * rhs_data[o + 4 * t]))
        })
        .collect();
    assert_eq!(out.as_slice().unwrap(), expected.as_slice());
}

#[test]
fn outer_product_fast_path_rejects_degenerate_1x1_batched_elementwise() {
    let lhs_data = [1.0_f64; 5];
    let rhs_data = [2.0_f64; 5];
    let lhs = TypedTensorView::from_slice([1, 5], [1, 1], 0, &lhs_data).unwrap();
    let rhs = TypedTensorView::from_slice([1, 5], [1, 1], 0, &rhs_data).unwrap();

    let plan =
        split_outer_product_plan(&lhs, &[1, 1, 5], &[0, 2], &rhs, &[1, 1, 5], &[1, 2]).unwrap();

    assert!(
        plan.is_none(),
        "1x1 per batch should use the ordinary zip-map path"
    );
}

#[test]
fn outer_product_fast_path_rejects_scaling_and_unsupported_axis_layouts() {
    let vector_data = vec![1.0_f64; 5];
    let matrix_data = vec![2.0_f64; 15];
    let vector = TypedTensorView::from_slice([5], [1], 0, &vector_data).unwrap();
    let matrix = TypedTensorView::from_slice([5, 3], [1, 5], 0, &matrix_data).unwrap();

    assert!(
        split_outer_product_plan(&vector, &[5, 3], &[0], &matrix, &[5, 3], &[0, 1])
            .unwrap()
            .is_none(),
        "lhs scaling over a shared axis is not an outer product"
    );
    assert!(
        split_outer_product_plan(&matrix, &[5, 3], &[0, 1], &vector, &[5, 3], &[0])
            .unwrap()
            .is_none(),
        "rhs scaling over a shared axis is not an outer product"
    );

    let lhs_data = vec![1.0_f64; 6];
    let rhs_data = vec![2.0_f64; 20];
    let lhs = TypedTensorView::from_slice([2, 3], [1, 2], 0, &lhs_data).unwrap();
    let rhs = TypedTensorView::from_slice([4, 5], [1, 4], 0, &rhs_data).unwrap();
    assert!(
        split_outer_product_plan(&lhs, &[2, 4, 3, 5], &[0, 2], &rhs, &[2, 4, 3, 5], &[1, 3])
            .unwrap()
            .is_none(),
        "interleaved free axes are not supported by the materialized fast path"
    );

    let lhs_data = [1.0_f64, 2.0];
    let rhs_data = [3.0_f64, 4.0, 5.0];
    let lhs = TypedTensorView::from_slice([2], [1], 0, &lhs_data).unwrap();
    let rhs = TypedTensorView::from_slice([3], [1], 0, &rhs_data).unwrap();
    assert!(
        split_outer_product_plan(&lhs, &[2, 3, 4], &[0], &rhs, &[2, 3, 4], &[1])
            .unwrap()
            .is_none(),
        "every output axis must be covered by lhs, rhs, or a shared batch axis"
    );
}

#[test]
fn outer_product_fast_path_rejects_pure_shared_batch_elementwise() {
    let mut buffers = BufferPool::default();
    let lhs_data = [1.0_f64; 24];
    let rhs_data = [2.0_f64; 24];
    let lhs = TypedTensorView::from_slice([2, 3, 4], [1, 2, 6], 0, &lhs_data).unwrap();
    let rhs = TypedTensorView::from_slice([4, 2, 3], [1, 4, 8], 0, &rhs_data).unwrap();

    let out = try_outer_product_with_pool(
        &mut buffers,
        &lhs,
        &[2, 3, 4],
        &[0, 1, 2],
        &rhs,
        &[2, 3, 4],
        &[2, 0, 1],
    )
    .unwrap();

    assert!(
        out.is_none(),
        "pure shared-batch elementwise should use the ordinary zip-map path"
    );
}

#[test]
fn broadcast_multiply_fallback_handles_permuted_elementwise_without_materialization() {
    let mut buffers = BufferPool::default();
    let lhs_data: Vec<f64> = (0..24).map(|i| (i + 1) as f64).collect();
    let rhs_data: Vec<f64> = (0..24).map(|i| (100 + i) as f64).collect();
    let lhs =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 3, 4], lhs_data.clone()).unwrap());
    let rhs =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![4, 2, 3], rhs_data.clone()).unwrap());

    let out = broadcast_multiply_read_with_pool(
        &mut buffers,
        TensorRead::from_tensor(&lhs),
        &[2, 3, 4],
        &[0, 1, 2],
        TensorRead::from_tensor(&rhs),
        &[2, 3, 4],
        &[2, 0, 1],
    )
    .unwrap()
    .expect("same-rank permuted elementwise multiply should use fallback broadcast mul");

    let expected: Vec<f64> = (0..4)
        .flat_map(|k| {
            let lhs_data = &lhs_data;
            let rhs_data = &rhs_data;
            (0..3).flat_map(move |j| {
                (0..2).map(move |i| {
                    let lhs_offset = i + 2 * j + 6 * k;
                    let rhs_offset = k + 4 * i + 8 * j;
                    lhs_data[lhs_offset] * rhs_data[rhs_offset]
                })
            })
        })
        .collect();

    assert_eq!(out.shape(), &[2, 3, 4]);
    assert_eq!(out.as_slice::<f64>().unwrap(), expected.as_slice());
}

#[test]
fn broadcast_multiply_handles_scalar_full_output_pairs() {
    let mut buffers = BufferPool::default();
    let scalar = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![2.0]).unwrap());
    let vector =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap());

    let lhs_scalar = broadcast_multiply_read_with_pool(
        &mut buffers,
        TensorRead::from_tensor(&scalar),
        &[3],
        &[],
        TensorRead::from_tensor(&vector),
        &[3],
        &[0],
    )
    .unwrap()
    .expect("scalar lhs broadcast multiply should materialize");
    assert_eq!(lhs_scalar.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);

    let rhs_scalar = broadcast_multiply_read_with_pool(
        &mut buffers,
        TensorRead::from_tensor(&vector),
        &[3],
        &[0],
        TensorRead::from_tensor(&scalar),
        &[3],
        &[],
    )
    .unwrap()
    .expect("scalar rhs broadcast multiply should materialize");
    assert_eq!(rhs_scalar.as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0]);

    let other_scalar = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![3.0]).unwrap());
    let both_scalar = broadcast_multiply_read_with_pool(
        &mut buffers,
        TensorRead::from_tensor(&scalar),
        &[3],
        &[],
        TensorRead::from_tensor(&other_scalar),
        &[3],
        &[],
    )
    .unwrap()
    .expect("scalar-scalar broadcast multiply should materialize");
    assert_eq!(both_scalar.as_slice::<f64>().unwrap(), &[6.0, 6.0, 6.0]);

    let complex_scalar =
        Tensor::C64(TypedTensor::from_vec_col_major(vec![], vec![c64(0.5, -1.5)]).unwrap());
    let complex_vector = Tensor::C64(
        TypedTensor::from_vec_col_major(vec![2], vec![c64(1.0, 2.0), c64(-3.0, 0.5)]).unwrap(),
    );
    let complex_value = broadcast_multiply_value_with_pool(
        &mut buffers,
        TensorRead::from_tensor(&complex_scalar),
        &[2],
        &[],
        TensorRead::from_tensor(&complex_vector),
        &[2],
        &[0],
    )
    .unwrap()
    .expect("complex scalar broadcast multiply should materialize");
    let complex_tensor = complex_value.to_tensor().unwrap();
    let complex_data = complex_tensor.as_slice::<Complex<f64>>().unwrap();
    assert_c64_close(complex_data[0], c64(3.5, -0.5));
    assert_c64_close(complex_data[1], c64(-0.75, 4.75));
}

#[test]
fn lazy_outer_product_lhs_prefix_preserves_logical_output_order() {
    let mut buffers = BufferPool::default();
    let lhs_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_data = [10.0_f64, 20.0, 30.0, 40.0];
    let lhs = TypedTensorView::from_slice([2, 3], [3, 1], 0, &lhs_data).unwrap();
    let rhs = TypedTensorView::from_slice([4], [1], 0, &rhs_data).unwrap();

    let out = try_lazy_outer_product_with_pool(
        &mut buffers,
        &lhs,
        &[2, 3, 4],
        &[0, 1],
        &rhs,
        &[2, 3, 4],
        &[2],
    )
    .unwrap()
    .expect("non-canonical lhs physical order should use lazy outer-product output");

    assert_eq!(out.shape, vec![2, 3, 4]);
    assert_ne!(out.strides, col_major_strides(&out.shape).unwrap());
    let value = lazy_outer_product_value(Tensor::F64(out.base), out.shape, out.strides).unwrap();
    let tensor = value.to_tensor().unwrap();
    let expected: Vec<f64> = (0..4)
        .flat_map(|k| {
            (0..3).flat_map(move |j| (0..2).map(move |i| lhs_data[i * 3 + j] * rhs_data[k]))
        })
        .collect();
    assert_eq!(tensor.shape(), &[2, 3, 4]);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), expected.as_slice());
}

#[test]
fn lazy_outer_product_rhs_prefix_preserves_logical_output_order() {
    let mut buffers = BufferPool::default();
    let lhs_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_data = [10.0_f64, 20.0, 30.0, 40.0];
    let lhs = TypedTensorView::from_slice([2, 3], [3, 1], 0, &lhs_data).unwrap();
    let rhs = TypedTensorView::from_slice([4], [1], 0, &rhs_data).unwrap();

    let out = try_lazy_outer_product_with_pool(
        &mut buffers,
        &lhs,
        &[4, 2, 3],
        &[1, 2],
        &rhs,
        &[4, 2, 3],
        &[0],
    )
    .unwrap()
    .expect("rhs-prefix output should still support lazy non-canonical lhs order");

    assert_eq!(out.shape, vec![4, 2, 3]);
    assert_ne!(out.strides, col_major_strides(&out.shape).unwrap());
    let value = lazy_outer_product_value(Tensor::F64(out.base), out.shape, out.strides).unwrap();
    let tensor = value.to_tensor().unwrap();
    let expected: Vec<f64> = (0..3)
        .flat_map(|j| {
            (0..2).flat_map(move |i| (0..4).map(move |k| rhs_data[k] * lhs_data[i * 3 + j]))
        })
        .collect();
    assert_eq!(tensor.shape(), &[4, 2, 3]);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), expected.as_slice());
}

fn c32(real: f32, imag: f32) -> Complex<f32> {
    Complex::new(real, imag)
}

fn c64(real: f64, imag: f64) -> Complex<f64> {
    Complex::new(real, imag)
}

fn assert_c32_close(actual: Complex<f32>, expected: Complex<f32>) {
    assert!((actual.re - expected.re).abs() < 1.0e-5);
    assert!((actual.im - expected.im).abs() < 1.0e-5);
}

fn assert_c64_close(actual: Complex<f64>, expected: Complex<f64>) {
    assert!((actual.re - expected.re).abs() < 1.0e-12);
    assert!((actual.im - expected.im).abs() < 1.0e-12);
}

fn assert_shape_mismatch<T>(result: crate::Result<T>, op: &'static str) {
    assert!(matches!(
        result,
        Err(crate::Error::ShapeMismatch { op: actual, .. }) if actual == op
    ));
}

fn assert_dtype_mismatch<T>(result: crate::Result<T>, op: &'static str) {
    assert!(matches!(
        result,
        Err(crate::Error::DTypeMismatch { op: actual, .. }) if actual == op
    ));
}

fn assert_backend_failure<T>(result: crate::Result<T>, op: &'static str) {
    assert!(matches!(
        result,
        Err(crate::Error::BackendFailure { op: actual, .. }) if actual == op
    ));
}

fn assert_invalid_config_contains<T>(result: crate::Result<T>, op: &'static str, expected: &str) {
    assert!(matches!(
        result,
        Err(crate::Error::InvalidConfig {
            op: actual,
            ref message,
        }) if actual == op && message.contains(expected)
    ));
}

#[test]
fn complex_ordered_ops_are_explicitly_rejected() {
    let lhs = Tensor::C64(TypedTensor::from_vec_col_major(vec![1], vec![c64(1.0, 0.0)]).unwrap());
    let rhs = Tensor::C64(TypedTensor::from_vec_col_major(vec![1], vec![c64(0.0, 1.0)]).unwrap());

    for (op, result) in [
        ("maximum", maximum(&lhs, &rhs)),
        ("minimum", minimum(&lhs, &rhs)),
        ("compare", compare(&lhs, &rhs, &CompareDir::Le)),
    ] {
        assert_invalid_config_contains(result, op, "total order");
    }

    assert_invalid_config_contains(clamp(&lhs, &rhs, &lhs), "clamp", "total order");
}

#[test]
fn typed_view_helpers_cover_scalar_and_validation_paths() {
    let mut buffers = BufferPool::default();
    let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 4.0]).unwrap();
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![10.0, 20.0]).unwrap();
    let scalar = TypedTensor::<f64>::from_vec_col_major(vec![], vec![2.0]).unwrap();
    let short = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![99.0]).unwrap();
    let pred = TypedTensor::<bool>::from_vec_col_major(vec![2], vec![true, false]).unwrap();
    let lower = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 2.0]).unwrap();
    let upper = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 5.0]).unwrap();

    let lhs_view = lhs.as_view();
    let rhs_view = rhs.as_view();
    let scalar_view = scalar.as_view();
    let short_view = short.as_view();
    let pred_view = pred.as_view();
    let lower_view = lower.as_view();
    let upper_view = upper.as_view();

    let same_shape =
        typed_binary_view_with_pool("test_binary", &mut buffers, &lhs_view, &rhs_view, |x, y| {
            x + y
        })
        .unwrap();
    assert_eq!(same_shape.as_slice().unwrap(), &[11.0, 24.0]);

    let scalar_lhs = typed_binary_view_with_pool(
        "test_binary",
        &mut buffers,
        &scalar_view,
        &rhs_view,
        |x, y| x * y,
    )
    .unwrap();
    assert_eq!(scalar_lhs.as_slice().unwrap(), &[20.0, 40.0]);

    let scalar_rhs = typed_binary_view_with_pool(
        "test_binary",
        &mut buffers,
        &lhs_view,
        &scalar_view,
        |x, y| x * y,
    )
    .unwrap();
    assert_eq!(scalar_rhs.as_slice().unwrap(), &[2.0, 8.0]);

    assert_shape_mismatch(
        typed_binary_view_with_pool(
            "test_binary",
            &mut buffers,
            &lhs_view,
            &short_view,
            |x, y| x + y,
        ),
        "test_binary",
    );

    let negated =
        typed_unary_view_with_pool("test_unary", &mut buffers, &lhs_view, |x| -x).unwrap();
    assert_eq!(negated.as_slice().unwrap(), &[-1.0, -4.0]);

    let compared = typed_same_shape_binary_view_with_pool(
        "test_same_shape",
        &mut buffers,
        &lhs_view,
        &rhs_view,
        |x, y| x < y,
    )
    .unwrap();
    assert_eq!(compared.as_slice().unwrap(), &[true, true]);
    assert_shape_mismatch(
        typed_same_shape_binary_view_with_pool(
            "test_same_shape",
            &mut buffers,
            &lhs_view,
            &short_view,
            |x, y| x < y,
        ),
        "test_same_shape",
    );

    let selected =
        typed_select_view_with_pool(&mut buffers, &pred_view, &lhs_view, &rhs_view).unwrap();
    assert_eq!(selected.as_slice().unwrap(), &[1.0, 20.0]);
    assert_shape_mismatch(
        typed_select_view_with_pool(&mut buffers, &pred_view, &short_view, &rhs_view),
        "select",
    );
    assert_shape_mismatch(
        typed_select_view_with_pool(&mut buffers, &pred_view, &lhs_view, &short_view),
        "select",
    );

    let clamped =
        typed_clamp_view_with_pool(&mut buffers, &lhs_view, &lower_view, &upper_view).unwrap();
    assert_eq!(clamped.as_slice().unwrap(), &[1.0, 4.0]);

    let degenerate_lower: TypedTensor<f64> =
        TypedTensor::from_vec_col_major(vec![2], vec![5.0_f64, 5.0]).unwrap();
    let degenerate_upper: TypedTensor<f64> =
        TypedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 3.0]).unwrap();
    let degenerate = typed_clamp_view_with_pool(
        &mut buffers,
        &lhs_view,
        &degenerate_lower.as_view(),
        &degenerate_upper.as_view(),
    )
    .unwrap();
    assert_eq!(degenerate.as_slice().unwrap(), &[3.0, 3.0]);

    assert_shape_mismatch(
        typed_clamp_view_with_pool(&mut buffers, &lhs_view, &short_view, &upper_view),
        "clamp",
    );
    assert_shape_mismatch(
        typed_clamp_view_with_pool(&mut buffers, &lhs_view, &lower_view, &short_view),
        "clamp",
    );
}

#[test]
fn read_as_cpu_view_covers_tensor_and_view_variants() {
    let f32_tensor =
        Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap());
    let f64_tensor =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap());
    let i32_tensor = Tensor::I32(TypedTensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap());
    let i64_tensor = Tensor::I64(TypedTensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap());
    let bool_tensor =
        Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, false]).unwrap());
    let c32_tensor = Tensor::C32(
        TypedTensor::from_vec_col_major(vec![2], vec![c32(1.0, 0.0), c32(0.0, 1.0)]).unwrap(),
    );
    let c64_tensor = Tensor::C64(
        TypedTensor::from_vec_col_major(vec![2], vec![c64(1.0, 0.0), c64(0.0, 1.0)]).unwrap(),
    );

    match read_as_cpu_view(TensorRead::from_tensor(&f32_tensor)) {
        CpuReadView::F32(view) => assert_eq!(view.shape(), &[2]),
        _ => panic!("expected f32 tensor read view"),
    }
    match read_as_cpu_view(TensorRead::from_tensor(&f64_tensor)) {
        CpuReadView::F64(view) => assert_eq!(view.shape(), &[2]),
        _ => panic!("expected f64 tensor read view"),
    }
    match read_as_cpu_view(TensorRead::from_tensor(&i32_tensor)) {
        CpuReadView::I32(view) => assert_eq!(view.shape(), &[2]),
        _ => panic!("expected i32 tensor read view"),
    }
    match read_as_cpu_view(TensorRead::from_tensor(&i64_tensor)) {
        CpuReadView::I64(view) => assert_eq!(view.shape(), &[2]),
        _ => panic!("expected i64 tensor read view"),
    }
    match read_as_cpu_view(TensorRead::from_tensor(&bool_tensor)) {
        CpuReadView::Bool(view) => assert_eq!(view.shape(), &[2]),
        _ => panic!("expected bool tensor read view"),
    }
    match read_as_cpu_view(TensorRead::from_tensor(&c32_tensor)) {
        CpuReadView::C32(view) => assert_eq!(view.shape(), &[2]),
        _ => panic!("expected c32 tensor read view"),
    }
    match read_as_cpu_view(TensorRead::from_tensor(&c64_tensor)) {
        CpuReadView::C64(view) => assert_eq!(view.shape(), &[2]),
        _ => panic!("expected c64 tensor read view"),
    }

    let f32_view_source = TypedTensor::<f32>::from_vec_col_major(vec![1], vec![3.0]).unwrap();
    let f64_view_source = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![3.0]).unwrap();
    let i32_view_source = TypedTensor::<i32>::from_vec_col_major(vec![1], vec![3]).unwrap();
    let i64_view_source = TypedTensor::<i64>::from_vec_col_major(vec![1], vec![3]).unwrap();
    let bool_view_source = TypedTensor::<bool>::from_vec_col_major(vec![1], vec![true]).unwrap();
    let c32_view_source =
        TypedTensor::<Complex<f32>>::from_vec_col_major(vec![1], vec![c32(3.0, 0.0)]).unwrap();
    let c64_view_source =
        TypedTensor::<Complex<f64>>::from_vec_col_major(vec![1], vec![c64(3.0, 0.0)]).unwrap();

    match read_as_cpu_view(TensorRead::from_view(TensorView::F32(
        f32_view_source.as_view(),
    ))) {
        CpuReadView::F32(view) => assert_eq!(view.shape(), &[1]),
        _ => panic!("expected f32 borrowed read view"),
    }
    match read_as_cpu_view(TensorRead::from_view(TensorView::F64(
        f64_view_source.as_view(),
    ))) {
        CpuReadView::F64(view) => assert_eq!(view.shape(), &[1]),
        _ => panic!("expected f64 borrowed read view"),
    }
    match read_as_cpu_view(TensorRead::from_view(TensorView::I32(
        i32_view_source.as_view(),
    ))) {
        CpuReadView::I32(view) => assert_eq!(view.shape(), &[1]),
        _ => panic!("expected i32 borrowed read view"),
    }
    match read_as_cpu_view(TensorRead::from_view(TensorView::I64(
        i64_view_source.as_view(),
    ))) {
        CpuReadView::I64(view) => assert_eq!(view.shape(), &[1]),
        _ => panic!("expected i64 borrowed read view"),
    }
    match read_as_cpu_view(TensorRead::from_view(TensorView::Bool(
        bool_view_source.as_view(),
    ))) {
        CpuReadView::Bool(view) => assert_eq!(view.shape(), &[1]),
        _ => panic!("expected bool borrowed read view"),
    }
    match read_as_cpu_view(TensorRead::from_view(TensorView::C32(
        c32_view_source.as_view(),
    ))) {
        CpuReadView::C32(view) => assert_eq!(view.shape(), &[1]),
        _ => panic!("expected c32 borrowed read view"),
    }
    match read_as_cpu_view(TensorRead::from_view(TensorView::C64(
        c64_view_source.as_view(),
    ))) {
        CpuReadView::C64(view) => assert_eq!(view.shape(), &[1]),
        _ => panic!("expected c64 borrowed read view"),
    }
}

#[test]
fn tensor_read_elementwise_dispatch_covers_view_and_complex_scalar_branches() {
    let mut buffers = BufferPool::default();

    let f32_a = TypedTensor::<f32>::from_vec_col_major(vec![2], vec![1.0, -2.0]).unwrap();
    let f32_b = TypedTensor::<f32>::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap();
    let f32_scalar = TypedTensor::<f32>::from_vec_col_major(vec![], vec![2.0]).unwrap();
    let f64_a = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![-3.0, 0.0, 4.0]).unwrap();
    let f64_b = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 5.0]).unwrap();
    let f64_scalar = TypedTensor::<f64>::from_vec_col_major(vec![], vec![2.0]).unwrap();
    let i32_a = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 4]).unwrap();
    let i32_b = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![2, 3]).unwrap();
    let i64_a = TypedTensor::<i64>::from_vec_col_major(vec![2], vec![5, -1]).unwrap();
    let i64_b = TypedTensor::<i64>::from_vec_col_major(vec![2], vec![2, -1]).unwrap();
    let bool_a = TypedTensor::<bool>::from_vec_col_major(vec![2], vec![true, false]).unwrap();
    let bool_b = TypedTensor::<bool>::from_vec_col_major(vec![2], vec![false, false]).unwrap();
    let pred = TypedTensor::<bool>::from_vec_col_major(vec![2], vec![true, false]).unwrap();
    let c32_a = TypedTensor::<Complex<f32>>::from_vec_col_major(
        vec![2],
        vec![c32(3.0, 4.0), c32(0.0, 2.0)],
    )
    .unwrap();
    let c32_b = TypedTensor::<Complex<f32>>::from_vec_col_major(
        vec![2],
        vec![c32(1.0, 0.0), c32(0.0, 2.0)],
    )
    .unwrap();
    let c64_a = TypedTensor::<Complex<f64>>::from_vec_col_major(
        vec![2],
        vec![c64(3.0, 4.0), c64(0.0, 2.0)],
    )
    .unwrap();
    let c64_b = TypedTensor::<Complex<f64>>::from_vec_col_major(
        vec![2],
        vec![c64(1.0, 0.0), c64(0.0, 2.0)],
    )
    .unwrap();

    let f32_b_tensor = Tensor::F32(f32_b.clone());
    let f64_b_tensor = Tensor::F64(f64_b.clone());
    let c32_b_tensor = Tensor::C32(c32_b.clone());
    let c64_b_tensor = Tensor::C64(c64_b.clone());
    let f32_scalar_tensor = Tensor::F32(f32_scalar.clone());
    let f64_scalar_tensor = Tensor::F64(f64_scalar.clone());

    let add_f32 = add_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::F32(f32_a.as_view())),
        TensorRead::from_tensor(&f32_b_tensor),
    )
    .unwrap();
    assert_eq!(add_f32.as_slice::<f32>().unwrap(), &[4.0, 2.0]);

    let add_c32_scalar = add_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::F32(f32_scalar.as_view())),
        TensorRead::from_tensor(&c32_b_tensor),
    )
    .unwrap();
    assert_c32_close(
        add_c32_scalar.as_slice::<Complex<f32>>().unwrap()[0],
        c32(3.0, 0.0),
    );

    let add_c64_scalar = add_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::C64(c64_b.as_view())),
        TensorRead::from_tensor(&f64_scalar_tensor),
    )
    .unwrap();
    assert_c64_close(
        add_c64_scalar.as_slice::<Complex<f64>>().unwrap()[1],
        c64(2.0, 2.0),
    );

    let mul_f64 = mul_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::F64(f64_a.as_view())),
        TensorRead::from_tensor(&f64_b_tensor),
    )
    .unwrap();
    assert_eq!(mul_f64.as_slice::<f64>().unwrap(), &[-3.0, 0.0, 20.0]);

    let mul_c32_scalar = mul_read_with_pool(
        &mut buffers,
        TensorRead::from_tensor(&c32_b_tensor),
        TensorRead::from_view(TensorView::F32(f32_scalar.as_view())),
    )
    .unwrap();
    assert_c32_close(
        mul_c32_scalar.as_slice::<Complex<f32>>().unwrap()[1],
        c32(0.0, 4.0),
    );

    assert_dtype_mismatch(
        mul_read_with_pool(
            &mut buffers,
            TensorRead::from_view(TensorView::F32(f32_a.as_view())),
            TensorRead::from_view(TensorView::F64(f64_a.as_view())),
        ),
        "mul",
    );

    let div_c32 = div_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::F32(f32_scalar.as_view())),
        TensorRead::from_view(TensorView::C32(c32_b.as_view())),
    )
    .unwrap();
    assert_c32_close(
        div_c32.as_slice::<Complex<f32>>().unwrap()[1],
        c32(0.0, -1.0),
    );

    let div_c64 = div_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::C64(c64_a.as_view())),
        TensorRead::from_view(TensorView::F64(f64_scalar.as_view())),
    )
    .unwrap();
    assert_c64_close(
        div_c64.as_slice::<Complex<f64>>().unwrap()[0],
        c64(1.5, 2.0),
    );

    let div_f32 = div_read_with_pool(
        &mut buffers,
        TensorRead::from_tensor(&f32_b_tensor),
        TensorRead::from_tensor(&f32_scalar_tensor),
    )
    .unwrap();
    assert_eq!(div_f32.as_slice::<f32>().unwrap(), &[1.5, 2.0]);

    let div_f64 = div_read_with_pool(
        &mut buffers,
        TensorRead::from_tensor(&f64_b_tensor),
        TensorRead::from_view(TensorView::F64(f64_scalar.as_view())),
    )
    .unwrap();
    assert_eq!(div_f64.as_slice::<f64>().unwrap(), &[0.5, 1.0, 2.5]);

    let neg = neg_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::C64(c64_a.as_view())),
    )
    .unwrap();
    assert_c64_close(neg.as_slice::<Complex<f64>>().unwrap()[0], c64(-3.0, -4.0));
    let neg_i32 = neg_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::I32(i32_a.as_view())),
    )
    .unwrap();
    assert_eq!(neg_i32.as_slice::<i32>().unwrap(), &[-1, -4]);

    let conj = conj_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::C32(c32_a.as_view())),
    )
    .unwrap();
    assert_c32_close(conj.as_slice::<Complex<f32>>().unwrap()[0], c32(3.0, -4.0));
    assert_backend_failure(
        conj_read_with_pool(
            &mut buffers,
            TensorRead::from_view(TensorView::Bool(bool_a.as_view())),
        ),
        "conj",
    );

    let abs = abs_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::C64(c64_a.as_view())),
    )
    .unwrap();
    assert_eq!(abs.as_slice::<f64>().unwrap()[0], 5.0);

    let sign = sign_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::F64(f64_a.as_view())),
    )
    .unwrap();
    assert_eq!(sign.as_slice::<f64>().unwrap(), &[-1.0, 0.0, 1.0]);

    let sign_complex = sign_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::C32(c32_a.as_view())),
    )
    .unwrap();
    assert_c32_close(
        sign_complex.as_slice::<Complex<f32>>().unwrap()[0],
        c32(0.6, 0.8),
    );

    assert_invalid_config_contains(
        maximum_read_with_pool(
            &mut buffers,
            TensorRead::from_view(TensorView::C32(c32_a.as_view())),
            TensorRead::from_tensor(&c32_b_tensor),
        ),
        "maximum",
        "total order",
    );
    assert_invalid_config_contains(
        minimum_read_with_pool(
            &mut buffers,
            TensorRead::from_view(TensorView::C64(c64_a.as_view())),
            TensorRead::from_tensor(&c64_b_tensor),
        ),
        "minimum",
        "total order",
    );

    let cmp_i32 = compare_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::I32(i32_a.as_view())),
        TensorRead::from_view(TensorView::I32(i32_b.as_view())),
        &CompareDir::Lt,
    )
    .unwrap();
    assert_eq!(cmp_i32.as_slice::<bool>().unwrap(), &[true, false]);

    let cmp_i64 = compare_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::I64(i64_a.as_view())),
        TensorRead::from_view(TensorView::I64(i64_b.as_view())),
        &CompareDir::Ge,
    )
    .unwrap();
    assert_eq!(cmp_i64.as_slice::<bool>().unwrap(), &[true, true]);

    let cmp_bool = compare_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::Bool(bool_a.as_view())),
        TensorRead::from_view(TensorView::Bool(bool_b.as_view())),
        &CompareDir::Eq,
    )
    .unwrap();
    assert_eq!(cmp_bool.as_slice::<bool>().unwrap(), &[false, true]);

    assert_invalid_config_contains(
        compare_read_with_pool(
            &mut buffers,
            TensorRead::from_view(TensorView::C32(c32_a.as_view())),
            TensorRead::from_view(TensorView::C32(c32_b.as_view())),
            &CompareDir::Gt,
        ),
        "compare",
        "total order",
    );

    let select_i64 = select_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::Bool(pred.as_view())),
        TensorRead::from_view(TensorView::I64(i64_a.as_view())),
        TensorRead::from_view(TensorView::I64(i64_b.as_view())),
    )
    .unwrap();
    assert_eq!(select_i64.as_slice::<i64>().unwrap(), &[5, -1]);

    let select_bool = select_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::Bool(pred.as_view())),
        TensorRead::from_view(TensorView::Bool(bool_a.as_view())),
        TensorRead::from_view(TensorView::Bool(bool_b.as_view())),
    )
    .unwrap();
    assert_eq!(select_bool.as_slice::<bool>().unwrap(), &[true, false]);

    let select_c32 = select_read_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::Bool(pred.as_view())),
        TensorRead::from_view(TensorView::C32(c32_a.as_view())),
        TensorRead::from_view(TensorView::C32(c32_b.as_view())),
    )
    .unwrap();
    assert_c32_close(
        select_c32.as_slice::<Complex<f32>>().unwrap()[1],
        c32(0.0, 2.0),
    );

    assert_dtype_mismatch(
        select_read_with_pool(
            &mut buffers,
            TensorRead::from_view(TensorView::Bool(pred.as_view())),
            TensorRead::from_view(TensorView::F32(f32_a.as_view())),
            TensorRead::from_view(TensorView::F64(f64_a.as_view())),
        ),
        "select",
    );
    assert_dtype_mismatch(
        select_read_with_pool(
            &mut buffers,
            TensorRead::from_view(TensorView::F32(f32_a.as_view())),
            TensorRead::from_view(TensorView::F32(f32_a.as_view())),
            TensorRead::from_view(TensorView::F32(f32_b.as_view())),
        ),
        "select",
    );

    assert_invalid_config_contains(
        clamp_read_with_pool(
            &mut buffers,
            TensorRead::from_view(TensorView::C32(c32_a.as_view())),
            TensorRead::from_view(TensorView::C32(c32_b.as_view())),
            TensorRead::from_view(TensorView::C32(c32_a.as_view())),
        ),
        "clamp",
        "total order",
    );
    assert_invalid_config_contains(
        clamp_read_with_pool(
            &mut buffers,
            TensorRead::from_view(TensorView::C64(c64_a.as_view())),
            TensorRead::from_view(TensorView::C64(c64_b.as_view())),
            TensorRead::from_view(TensorView::C64(c64_a.as_view())),
        ),
        "clamp",
        "total order",
    );

    assert_dtype_mismatch(
        clamp_read_with_pool(
            &mut buffers,
            TensorRead::from_view(TensorView::F32(f32_a.as_view())),
            TensorRead::from_view(TensorView::F32(f32_b.as_view())),
            TensorRead::from_view(TensorView::F64(f64_b.as_view())),
        ),
        "clamp",
    );
}

#[test]
fn broadcast_multiply_read_and_value_cover_dtypes_and_error_paths() {
    let mut buffers = BufferPool::default();
    let lhs_shape = [2, 3, 4];
    let lhs_dims = [0, 1];
    let rhs_dims = [2];

    let lhs_f32_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let rhs_f32_data = [10.0_f32, 20.0, 30.0, 40.0];
    let lhs_f32 = TypedTensorView::from_slice([2, 3], [3, 1], 0, &lhs_f32_data).unwrap();
    let rhs_f32 = TypedTensorView::from_slice([4], [1], 0, &rhs_f32_data).unwrap();
    let value_f32 = broadcast_multiply_value_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::F32(lhs_f32.clone())),
        &lhs_shape,
        &lhs_dims,
        TensorRead::from_view(TensorView::F32(rhs_f32)),
        &lhs_shape,
        &rhs_dims,
    )
    .unwrap()
    .expect("non-canonical f32 outer product should stay lazy");
    assert!(matches!(value_f32, TensorValue::View(_)));
    assert_eq!(value_f32.to_tensor().unwrap().shape(), &lhs_shape);

    let lhs_i32_data = [1_i32, 2, 3, 4, 5, 6];
    let rhs_i32_data = [2_i32, 3, 4, 5];
    let lhs_i32 = TypedTensorView::from_slice([2, 3], [3, 1], 0, &lhs_i32_data).unwrap();
    let rhs_i32 = TypedTensorView::from_slice([4], [1], 0, &rhs_i32_data).unwrap();
    let value_i32 = broadcast_multiply_value_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::I32(lhs_i32.clone())),
        &lhs_shape,
        &lhs_dims,
        TensorRead::from_view(TensorView::I32(rhs_i32)),
        &lhs_shape,
        &rhs_dims,
    )
    .unwrap()
    .expect("non-canonical i32 outer product should stay lazy");
    assert!(matches!(value_i32, TensorValue::View(_)));

    let lhs_i64_data = [1_i64, 2, 3, 4, 5, 6];
    let rhs_i64_data = [2_i64, 3, 4, 5];
    let lhs_i64 = TypedTensorView::from_slice([2, 3], [3, 1], 0, &lhs_i64_data).unwrap();
    let rhs_i64 = TypedTensorView::from_slice([4], [1], 0, &rhs_i64_data).unwrap();
    let value_i64 = broadcast_multiply_value_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::I64(lhs_i64.clone())),
        &lhs_shape,
        &lhs_dims,
        TensorRead::from_view(TensorView::I64(rhs_i64)),
        &lhs_shape,
        &rhs_dims,
    )
    .unwrap()
    .expect("non-canonical i64 outer product should stay lazy");
    assert!(matches!(value_i64, TensorValue::View(_)));

    let lhs_c32_data = [
        c32(1.0, 0.0),
        c32(2.0, 0.0),
        c32(3.0, 0.0),
        c32(4.0, 0.0),
        c32(5.0, 0.0),
        c32(6.0, 0.0),
    ];
    let rhs_c32_data = [c32(2.0, 0.0), c32(3.0, 0.0), c32(4.0, 0.0), c32(5.0, 0.0)];
    let lhs_c32 = TypedTensorView::from_slice([2, 3], [3, 1], 0, &lhs_c32_data).unwrap();
    let rhs_c32 = TypedTensorView::from_slice([4], [1], 0, &rhs_c32_data).unwrap();
    let value_c32 = broadcast_multiply_value_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::C32(lhs_c32.clone())),
        &lhs_shape,
        &lhs_dims,
        TensorRead::from_view(TensorView::C32(rhs_c32)),
        &lhs_shape,
        &rhs_dims,
    )
    .unwrap()
    .expect("non-canonical c32 outer product should stay lazy");
    assert!(matches!(value_c32, TensorValue::View(_)));

    let lhs_c64_data = [
        c64(1.0, 0.0),
        c64(2.0, 0.0),
        c64(3.0, 0.0),
        c64(4.0, 0.0),
        c64(5.0, 0.0),
        c64(6.0, 0.0),
    ];
    let rhs_c64_data = [c64(2.0, 0.0), c64(3.0, 0.0), c64(4.0, 0.0), c64(5.0, 0.0)];
    let lhs_c64 = TypedTensorView::from_slice([2, 3], [3, 1], 0, &lhs_c64_data).unwrap();
    let rhs_c64 = TypedTensorView::from_slice([4], [1], 0, &rhs_c64_data).unwrap();
    let value_c64 = broadcast_multiply_value_with_pool(
        &mut buffers,
        TensorRead::from_view(TensorView::C64(lhs_c64.clone())),
        &lhs_shape,
        &lhs_dims,
        TensorRead::from_view(TensorView::C64(rhs_c64)),
        &lhs_shape,
        &rhs_dims,
    )
    .unwrap()
    .expect("non-canonical c64 outer product should stay lazy");
    assert!(matches!(value_c64, TensorValue::View(_)));

    let same_shape_i64_lhs =
        Tensor::I64(TypedTensor::from_vec_col_major(vec![2], vec![2, 3]).unwrap());
    let same_shape_i64_rhs =
        Tensor::I64(TypedTensor::from_vec_col_major(vec![2], vec![4, 5]).unwrap());
    let materialized = broadcast_multiply_value_with_pool(
        &mut buffers,
        TensorRead::from_tensor(&same_shape_i64_lhs),
        &[2],
        &[0],
        TensorRead::from_tensor(&same_shape_i64_rhs),
        &[2],
        &[0],
    )
    .unwrap()
    .expect("same-shape multiply should materialize");
    assert!(matches!(materialized, TensorValue::Tensor(_)));
    assert_eq!(
        materialized.to_tensor().unwrap().as_slice::<i64>().unwrap(),
        &[8, 15]
    );

    let bool_lhs =
        Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, false]).unwrap());
    let bool_rhs =
        Tensor::Bool(TypedTensor::from_vec_col_major(vec![2], vec![true, true]).unwrap());
    assert!(broadcast_multiply_read_with_pool(
        &mut buffers,
        TensorRead::from_tensor(&bool_lhs),
        &[2],
        &[0],
        TensorRead::from_tensor(&bool_rhs),
        &[2],
        &[0],
    )
    .unwrap()
    .is_none());

    assert_shape_mismatch(
        typed_broadcast_mul_view_with_pool(
            &mut buffers,
            &lhs_f32,
            &[2, 3, 4],
            &[0, 1],
            &lhs_f32,
            &[2, 3, 5],
            &[0, 1],
        ),
        "broadcast_multiply",
    );
}
