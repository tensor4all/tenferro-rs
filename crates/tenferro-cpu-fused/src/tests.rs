use super::*;

#[test]
fn elementwise_fusion_validation_covers_descriptor_errors_and_empty_outputs() {
    use tenferro_cpu_basic::reject_complex_ordered_dtypes;
    use tenferro_tensor::backend::ElementwiseFusionInst;

    let input = Tensor::from_typed::<f32>(
        TypedTensor::<f32>::from_vec_col_major(vec![1], vec![1.0]).unwrap(),
    );

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
        Err(crate::Error::Validation {
            op: ELEMENTWISE_FUSION_OP,
            source: tenferro_tensor::ValidationError::InvalidArgument { .. },
            ..
        })
    ));

    let empty_outputs = ElementwiseFusionPlan::new(DType::F32, 1, Vec::new(), Vec::new());
    assert!(!validate_elementwise_fusion_inputs(&[&input], &empty_outputs).unwrap());

    let dtype_mismatch = ElementwiseFusionPlan::new(DType::F64, 1, vec![0], Vec::new());
    assert!(matches!(
        validate_elementwise_fusion_inputs(&[&input], &dtype_mismatch),
        Err(crate::Error::Validation {
            op: ELEMENTWISE_FUSION_OP,
            source: tenferro_tensor::ValidationError::DTypeMismatch {
                expected: _,
                actual: _,
            },
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
        Err(crate::Error::Unsupported { op: "maximum", .. })
    ));
}

#[test]
fn elementwise_fusion_declines_i32_add_multiply_plan() {
    use tenferro_tensor::backend::ElementwiseFusionInst;

    let mut buffers = BufferPool::default();
    let n = ELEMENTWISE_FUSION_MIN_ELEMENTS;
    let lhs_data = (0..n)
        .map(|i| if i == 0 { i32::MAX } else { i as i32 })
        .collect::<Vec<_>>();
    let rhs_data = (0..n)
        .map(|i| (i as i32).wrapping_add(1))
        .collect::<Vec<_>>();
    let lhs = Tensor::from_typed::<i32>(
        TypedTensor::<i32>::from_vec_col_major(vec![n], lhs_data.clone()).unwrap(),
    );
    let rhs = Tensor::from_typed::<i32>(
        TypedTensor::<i32>::from_vec_col_major(vec![n], rhs_data.clone()).unwrap(),
    );
    let plan = ElementwiseFusionPlan::new(
        DType::I32,
        2,
        vec![3],
        vec![
            ElementwiseFusionInst::new(ElementwiseFusionOp::Add, vec![0, 1]),
            ElementwiseFusionInst::new(ElementwiseFusionOp::Multiply, vec![2, 0]),
        ],
    );

    assert!(
        elementwise_fusion_with_pool(&mut buffers, &ExecContext::serial(), &[&lhs, &rhs], &plan)
            .unwrap()
            .is_none(),
        "I32 has no measured vectorized replay and must stay unfused"
    );
}

#[test]
fn elementwise_fusion_executes_reversed_first_input_plan() {
    use tenferro_tensor::backend::ElementwiseFusionInst;

    let mut buffers = BufferPool::default();
    let n = ELEMENTWISE_FUSION_MIN_ELEMENTS;
    let lhs_data = (0..n).map(|i| i as f64 + 1.0).collect::<Vec<_>>();
    let rhs_data = (0..n).map(|i| (i as f64 + 1.0) * 10.0).collect::<Vec<_>>();
    let lhs = Tensor::from_typed::<f64>(
        TypedTensor::<f64>::from_vec_col_major(vec![n], lhs_data).unwrap(),
    );
    let rhs = Tensor::from_typed::<f64>(
        TypedTensor::<f64>::from_vec_col_major(vec![n], rhs_data).unwrap(),
    );
    let plan = ElementwiseFusionPlan::new(
        DType::F64,
        2,
        vec![3],
        vec![
            ElementwiseFusionInst::new(ElementwiseFusionOp::Add, vec![1, 0]),
            ElementwiseFusionInst::new(ElementwiseFusionOp::Multiply, vec![2, 1]),
        ],
    );

    let outputs =
        elementwise_fusion_with_pool(&mut buffers, &ExecContext::serial(), &[&lhs, &rhs], &plan)
            .unwrap()
            .expect("reversed first input order should be handled by strided fused replay");

    assert_eq!(outputs.len(), 1);
    let actual = outputs[0].as_slice::<f64>().unwrap();
    assert_eq!(actual[0], 110.0);
    assert_eq!(actual[1], 440.0);
    assert_eq!(actual[n - 1], 110.0 * (n as f64).powi(2));
}

#[test]
fn elementwise_fusion_executes_f32_plan_and_declines_i64() {
    use tenferro_tensor::backend::ElementwiseFusionInst;

    let mut buffers = BufferPool::default();
    let n = ELEMENTWISE_FUSION_MIN_ELEMENTS;
    let f32_lhs_data = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let f32_rhs_data = (0..n).map(|i| (i as f32) * 2.0).collect::<Vec<_>>();
    let f32_lhs = Tensor::from_typed::<f32>(
        TypedTensor::<f32>::from_vec_col_major(vec![n], f32_lhs_data.clone()).unwrap(),
    );
    let f32_rhs = Tensor::from_typed::<f32>(
        TypedTensor::<f32>::from_vec_col_major(vec![n], f32_rhs_data.clone()).unwrap(),
    );
    let add_plan = ElementwiseFusionPlan::new(
        DType::F32,
        2,
        vec![2],
        vec![ElementwiseFusionInst::new(
            ElementwiseFusionOp::Add,
            vec![0, 1],
        )],
    );

    let f32_outputs = elementwise_fusion_with_pool(
        &mut buffers,
        &ExecContext::serial(),
        &[&f32_lhs, &f32_rhs],
        &add_plan,
    )
    .unwrap()
    .expect("F32 add fusion should be delegated to strided-kernel");
    let f32_actual = f32_outputs[0].as_slice::<f32>().unwrap();
    assert_eq!(f32_actual[3], f32_lhs_data[3] + f32_rhs_data[3]);

    let i64_lhs_data = (0..n).map(|i| i as i64).collect::<Vec<_>>();
    let i64_rhs_data = (0..n)
        .map(|i| (i as i64).wrapping_mul(3))
        .collect::<Vec<_>>();
    let i64_lhs = Tensor::from_typed::<i64>(
        TypedTensor::<i64>::from_vec_col_major(vec![n], i64_lhs_data.clone()).unwrap(),
    );
    let i64_rhs = Tensor::from_typed::<i64>(
        TypedTensor::<i64>::from_vec_col_major(vec![n], i64_rhs_data.clone()).unwrap(),
    );
    let i64_plan = ElementwiseFusionPlan::new(
        DType::I64,
        2,
        vec![2],
        vec![ElementwiseFusionInst::new(
            ElementwiseFusionOp::Add,
            vec![0, 1],
        )],
    );

    assert!(
        elementwise_fusion_with_pool(
            &mut buffers,
            &ExecContext::serial(),
            &[&i64_lhs, &i64_rhs],
            &i64_plan,
        )
        .unwrap()
        .is_none(),
        "I64 has no measured vectorized replay and must stay unfused"
    );
}

#[test]
fn elementwise_fusion_declines_bool_and_complex_dtypes() {
    use tenferro_tensor::backend::ElementwiseFusionInst;

    let mut buffers = BufferPool::default();
    let n = ELEMENTWISE_FUSION_MIN_ELEMENTS;
    let bool_data = (0..n).map(|i| i % 3 == 0).collect::<Vec<_>>();
    let bool_input = Tensor::from_typed::<bool>(
        TypedTensor::<bool>::from_vec_col_major(vec![n], bool_data.clone()).unwrap(),
    );
    let bool_plan = ElementwiseFusionPlan::new(
        DType::Bool,
        1,
        vec![1],
        vec![ElementwiseFusionInst::new(
            ElementwiseFusionOp::Conj,
            vec![0],
        )],
    );

    assert!(
        elementwise_fusion_with_pool(
            &mut buffers,
            &ExecContext::serial(),
            &[&bool_input],
            &bool_plan,
        )
        .unwrap()
        .is_none(),
        "Bool has no measured vectorized replay and must stay unfused"
    );

    let c32_lhs_data = (0..n)
        .map(|i| num_complex::Complex32::new(i as f32, 1.0))
        .collect::<Vec<_>>();
    let c32_rhs_data = (0..n)
        .map(|i| num_complex::Complex32::new(2.0, i as f32))
        .collect::<Vec<_>>();
    let c32_lhs = Tensor::from_typed::<tenferro_tensor::Complex32>(
        TypedTensor::<num_complex::Complex32>::from_vec_col_major(vec![n], c32_lhs_data.clone())
            .unwrap(),
    );
    let c32_rhs = Tensor::from_typed::<tenferro_tensor::Complex32>(
        TypedTensor::<num_complex::Complex32>::from_vec_col_major(vec![n], c32_rhs_data.clone())
            .unwrap(),
    );
    let c32_plan = ElementwiseFusionPlan::new(
        DType::C32,
        2,
        vec![2, 3],
        vec![
            ElementwiseFusionInst::new(ElementwiseFusionOp::Add, vec![0, 1]),
            ElementwiseFusionInst::new(ElementwiseFusionOp::Multiply, vec![2, 1]),
        ],
    );

    assert!(
        elementwise_fusion_with_pool(
            &mut buffers,
            &ExecContext::serial(),
            &[&c32_lhs, &c32_rhs],
            &c32_plan,
        )
        .unwrap()
        .is_none(),
        "C32 has no measured vectorized replay and must stay unfused"
    );

    let c64_lhs_data = (0..n)
        .map(|i| num_complex::Complex64::new(i as f64, -1.0))
        .collect::<Vec<_>>();
    let c64_rhs_data = (0..n)
        .map(|i| num_complex::Complex64::new(1.0, i as f64))
        .collect::<Vec<_>>();
    let c64_lhs = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::<num_complex::Complex64>::from_vec_col_major(vec![n], c64_lhs_data.clone())
            .unwrap(),
    );
    let c64_rhs = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::<num_complex::Complex64>::from_vec_col_major(vec![n], c64_rhs_data.clone())
            .unwrap(),
    );
    let c64_plan = ElementwiseFusionPlan::new(
        DType::C64,
        2,
        vec![2],
        vec![ElementwiseFusionInst::new(
            ElementwiseFusionOp::Add,
            vec![0, 1],
        )],
    );

    assert!(
        elementwise_fusion_with_pool(
            &mut buffers,
            &ExecContext::serial(),
            &[&c64_lhs, &c64_rhs],
            &c64_plan,
        )
        .unwrap()
        .is_none(),
        "C64 has no measured vectorized replay and must stay unfused"
    );
}

#[test]
fn elementwise_fusion_declines_ordered_real_dtypes_for_nan_propagation() {
    use tenferro_tensor::backend::ElementwiseFusionInst;

    let mut buffers = BufferPool::default();
    let n = ELEMENTWISE_FUSION_MIN_ELEMENTS;
    for (dtype, tensor) in [
        (
            DType::F32,
            Tensor::from_typed::<f32>(
                TypedTensor::<f32>::from_vec_col_major(vec![n], vec![1.0_f32; n]).unwrap(),
            ),
        ),
        (
            DType::F64,
            Tensor::from_typed::<f64>(
                TypedTensor::<f64>::from_vec_col_major(vec![n], vec![1.0_f64; n]).unwrap(),
            ),
        ),
    ] {
        for op in [
            ElementwiseFusionOp::Maximum,
            ElementwiseFusionOp::Minimum,
            ElementwiseFusionOp::Clamp,
        ] {
            let plan = ElementwiseFusionPlan::new(
                dtype,
                2,
                vec![2],
                vec![ElementwiseFusionInst::new(op, vec![0, 1])],
            );
            assert!(
                elementwise_fusion_with_pool(
                    &mut buffers,
                    &ExecContext::serial(),
                    &[&tensor, &tensor],
                    &plan,
                )
                .unwrap()
                .is_none(),
                "{dtype:?} {op:?} must stay unfused: upstream `.max`/`.min` drop NaN"
            );
        }
    }
}

#[test]
fn elementwise_fusion_returns_none_for_erased_unsupported_ops() {
    use tenferro_tensor::backend::ElementwiseFusionInst;

    let mut buffers = BufferPool::default();
    let n = ELEMENTWISE_FUSION_MIN_ELEMENTS;
    let lhs = Tensor::from_typed::<i32>(
        TypedTensor::<i32>::from_vec_col_major(vec![n], vec![6; n]).unwrap(),
    );
    let rhs = Tensor::from_typed::<i32>(
        TypedTensor::<i32>::from_vec_col_major(vec![n], vec![2; n]).unwrap(),
    );
    let divide_plan = ElementwiseFusionPlan::new(
        DType::I32,
        2,
        vec![2],
        vec![ElementwiseFusionInst::new(
            ElementwiseFusionOp::Divide,
            vec![0, 1],
        )],
    );
    assert!(
        elementwise_fusion_with_pool(
            &mut buffers,
            &ExecContext::serial(),
            &[&lhs, &rhs],
            &divide_plan
        )
        .unwrap()
        .is_none(),
        "integer divide is not owned by erased strided fusion"
    );

    let complex = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::<num_complex::Complex64>::from_vec_col_major(
            vec![n],
            vec![num_complex::Complex64::new(1.0, 0.0); n],
        )
        .unwrap(),
    );
    let ordered_plan = ElementwiseFusionPlan::new(
        DType::C64,
        2,
        vec![2],
        vec![ElementwiseFusionInst::new(
            ElementwiseFusionOp::Maximum,
            vec![0, 1],
        )],
    );
    assert!(
        elementwise_fusion_with_pool(
            &mut buffers,
            &ExecContext::serial(),
            &[&complex, &complex],
            &ordered_plan
        )
        .unwrap()
        .is_none(),
        "complex ordered ops stay outside erased strided fusion"
    );
}

#[test]
fn elementwise_fusion_returns_none_above_erased_input_limit() {
    use tenferro_tensor::backend::ElementwiseFusionInst;

    let mut buffers = BufferPool::default();
    let n = ELEMENTWISE_FUSION_MIN_ELEMENTS;
    let input_count = ERASED_FUSION_MAX_INPUTS + 1;
    let tensors = (0..input_count)
        .map(|_| {
            Tensor::from_typed::<f64>(
                TypedTensor::<f64>::from_vec_col_major(vec![n], vec![1.0; n]).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let inputs = tensors.iter().collect::<Vec<_>>();
    // Left-fold sum: value `input_count + k` is the sum of inputs `0..=k + 1`.
    let ops = (1..input_count)
        .map(|input| {
            let accumulated = if input == 1 {
                0
            } else {
                input_count + input - 2
            };
            ElementwiseFusionInst::new(ElementwiseFusionOp::Add, vec![accumulated, input])
        })
        .collect::<Vec<_>>();
    let plan = ElementwiseFusionPlan::new(DType::F64, input_count, vec![2 * input_count - 2], ops);

    assert!(
        elementwise_fusion_with_pool(&mut buffers, &ExecContext::serial(), &inputs, &plan)
            .unwrap()
            .is_none(),
        "plans reading more inputs than erased strided fusion accepts are declined"
    );
}
