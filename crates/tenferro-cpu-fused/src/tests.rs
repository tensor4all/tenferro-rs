use super::*;
use strided_basic::Exec;

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
        elementwise_fusion_with_pool(&mut buffers, &[&lhs, &rhs], &plan, Exec::serial())
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

    let outputs = elementwise_fusion_with_pool(&mut buffers, &[&lhs, &rhs], &plan, Exec::serial())
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
        &[&f32_lhs, &f32_rhs],
        &add_plan,
        Exec::serial(),
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
            &[&i64_lhs, &i64_rhs],
            &i64_plan,
            Exec::serial(),
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
        elementwise_fusion_with_pool(&mut buffers, &[&bool_input], &bool_plan, Exec::serial(),)
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
            &[&c32_lhs, &c32_rhs],
            &c32_plan,
            Exec::serial(),
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
            &[&c64_lhs, &c64_rhs],
            &c64_plan,
            Exec::serial(),
        )
        .unwrap()
        .is_none(),
        "C64 has no measured vectorized replay and must stay unfused"
    );
}

/// Bit-for-bit reference for an ordered fused op, from the same `strided_basic`
/// selects the unfused kernels use.
fn ordered_reference<T: num_traits::Float>(
    op: ElementwiseFusionOp,
    input: T,
    lower: T,
    upper: T,
) -> T {
    use strided_basic::{nan_propagating_maximum, nan_propagating_minimum};
    match op {
        ElementwiseFusionOp::Maximum => nan_propagating_maximum(input, lower),
        ElementwiseFusionOp::Minimum => nan_propagating_minimum(input, lower),
        ElementwiseFusionOp::Clamp => {
            nan_propagating_minimum(upper, nan_propagating_maximum(lower, input))
        }
        other => unreachable!("not an ordered op: {other:?}"),
    }
}

/// Above-floor ordered ops fuse and match the unfused route bit for bit,
/// including NaN in every operand position and signed-zero ties.
macro_rules! ordered_fusion_bit_match {
    ($name:ident, $ty:ty, $dtype:expr) => {
        #[test]
        fn $name() {
            use tenferro_tensor::backend::ElementwiseFusionInst;

            let n = ELEMENTWISE_FUSION_MIN_ELEMENTS;
            let mut buffers = BufferPool::default();
            let input_data: Vec<$ty> = (0..n)
                .map(|i| match i % 6 {
                    0 => <$ty>::NAN,
                    1 => 0.0,
                    2 => -0.0,
                    3 => 1.0 + i as $ty,
                    4 => -1.0 - i as $ty,
                    _ => (i as $ty) * 0.5,
                })
                .collect();
            let lower_data: Vec<$ty> = (0..n)
                .map(|i| match i % 6 {
                    0 => 1.0,
                    1 => -0.0,
                    2 => 0.0,
                    3 => <$ty>::NAN,
                    4 => -2.0,
                    _ => 0.25,
                })
                .collect();
            let upper_data: Vec<$ty> = (0..n)
                .map(|i| match i % 6 {
                    0 => 2.0,
                    1 => 0.0,
                    2 => -0.0,
                    3 => 2.0,
                    4 => <$ty>::NAN,
                    _ => 3.0,
                })
                .collect();
            let tensor_of = |data: &[$ty]| {
                Tensor::from_typed::<$ty>(
                    TypedTensor::<$ty>::from_vec_col_major(vec![n], data.to_vec()).unwrap(),
                )
            };
            let input = tensor_of(&input_data);
            let lower = tensor_of(&lower_data);
            let upper = tensor_of(&upper_data);

            for op in [
                ElementwiseFusionOp::Maximum,
                ElementwiseFusionOp::Minimum,
                ElementwiseFusionOp::Clamp,
            ] {
                let (input_count, inputs) = match op {
                    ElementwiseFusionOp::Clamp => (3, vec![&input, &lower, &upper]),
                    _ => (2, vec![&input, &lower]),
                };
                let plan = ElementwiseFusionPlan::new(
                    $dtype,
                    input_count,
                    vec![input_count],
                    vec![ElementwiseFusionInst::new(op, (0..input_count).collect())],
                );
                let outputs =
                    elementwise_fusion_with_pool(&mut buffers, &inputs, &plan, Exec::serial())
                        .unwrap()
                        .unwrap_or_else(|| panic!("{op:?} must fuse above the floor"));
                let actual = outputs[0].as_slice::<$ty>().unwrap();
                for index in 0..n {
                    let expected = ordered_reference(
                        op,
                        input_data[index],
                        lower_data[index],
                        upper_data[index],
                    );
                    assert_eq!(
                        actual[index].to_bits(),
                        expected.to_bits(),
                        "{op:?} diverged at {index}: {:?} != {:?}",
                        actual[index],
                        expected
                    );
                }
            }
        }
    };
}

ordered_fusion_bit_match!(ordered_fusion_matches_unfused_f32, f32, DType::F32);
ordered_fusion_bit_match!(ordered_fusion_matches_unfused_f64, f64, DType::F64);

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
        elementwise_fusion_with_pool(&mut buffers, &[&lhs, &rhs], &divide_plan, Exec::serial(),)
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
            &[&complex, &complex],
            &ordered_plan,
            Exec::serial(),
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
        elementwise_fusion_with_pool(&mut buffers, &inputs, &plan, Exec::serial())
            .unwrap()
            .is_none(),
        "plans reading more inputs than erased strided fusion accepts are declined"
    );
}

#[test]
fn fused_host_rejects_a_dtype_tag_mismatch() {
    let f32_input = Tensor::from_typed::<f32>(
        TypedTensor::<f32>::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap(),
    );
    assert!(matches!(
        fused_host::<f64>(ELEMENTWISE_FUSION_OP, &f32_input),
        Err(crate::Error::UnsupportedDType { .. })
    ));
}

#[test]
fn elementwise_fusion_executes_every_selected_op_against_a_host_reference() {
    use tenferro_tensor::backend::ElementwiseFusionInst;

    let n = ELEMENTWISE_FUSION_MIN_ELEMENTS;
    let mut buffers = BufferPool::default();

    type UnaryReference = fn(f64) -> f64;
    let unary_cases: &[(ElementwiseFusionOp, UnaryReference)] = &[
        (ElementwiseFusionOp::Negate, |x| -x),
        (ElementwiseFusionOp::Conj, |x| x),
        (ElementwiseFusionOp::Abs, |x| x.abs()),
        (ElementwiseFusionOp::Exp, |x| x.exp()),
        (ElementwiseFusionOp::Log, |x| x.ln()),
        (ElementwiseFusionOp::Sin, |x| x.sin()),
        (ElementwiseFusionOp::Cos, |x| x.cos()),
        (ElementwiseFusionOp::Sqrt, |x| x.sqrt()),
        (ElementwiseFusionOp::Rsqrt, |x| 1.0 / x.sqrt()),
        (ElementwiseFusionOp::Expm1, |x| x.exp_m1()),
        (ElementwiseFusionOp::Log1p, |x| x.ln_1p()),
    ];
    let unary_data: Vec<f64> = (0..n).map(|i| 0.5 + (i % 17) as f64 * 0.125).collect();
    for &(op, reference) in unary_cases {
        let input = Tensor::from_typed::<f64>(
            TypedTensor::<f64>::from_vec_col_major(vec![n], unary_data.clone()).unwrap(),
        );
        let plan = ElementwiseFusionPlan::new(
            DType::F64,
            1,
            vec![1],
            vec![ElementwiseFusionInst::new(op, vec![0])],
        );
        let outputs = elementwise_fusion_with_pool(&mut buffers, &[&input], &plan, Exec::serial())
            .unwrap()
            .unwrap_or_else(|| panic!("{op:?} should be selected for F64"));
        let actual = outputs[0].as_slice::<f64>().unwrap();
        for (index, &x) in unary_data.iter().enumerate() {
            let expected = reference(x);
            assert!(
                (actual[index] - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "{op:?} diverged at {index}: {} != {expected}",
                actual[index]
            );
        }
    }

    type BinaryReference = fn(f64, f64) -> f64;
    let binary_cases: &[(ElementwiseFusionOp, BinaryReference)] = &[
        (ElementwiseFusionOp::Divide, |a, b| a / b),
        (ElementwiseFusionOp::Pow, |a, b| a.powf(b)),
    ];
    let lhs_data: Vec<f64> = (0..n).map(|i| 0.5 + (i % 11) as f64 * 0.25).collect();
    let rhs_data: Vec<f64> = (0..n).map(|i| 0.25 + (i % 7) as f64 * 0.125).collect();
    for &(op, reference) in binary_cases {
        let lhs = Tensor::from_typed::<f64>(
            TypedTensor::<f64>::from_vec_col_major(vec![n], lhs_data.clone()).unwrap(),
        );
        let rhs = Tensor::from_typed::<f64>(
            TypedTensor::<f64>::from_vec_col_major(vec![n], rhs_data.clone()).unwrap(),
        );
        let plan = ElementwiseFusionPlan::new(
            DType::F64,
            2,
            vec![2],
            vec![ElementwiseFusionInst::new(op, vec![0, 1])],
        );
        let outputs =
            elementwise_fusion_with_pool(&mut buffers, &[&lhs, &rhs], &plan, Exec::serial())
                .unwrap()
                .unwrap_or_else(|| panic!("{op:?} should be selected for F64"));
        let actual = outputs[0].as_slice::<f64>().unwrap();
        for index in 0..n {
            let expected = reference(lhs_data[index], rhs_data[index]);
            assert!(
                (actual[index] - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "{op:?} diverged at {index}: {} != {expected}",
                actual[index]
            );
        }
    }
}

#[test]
fn elementwise_fusion_rejects_malformed_broadcast_views() {
    use tenferro_tensor::backend::{ElementwiseFusionInputView, ElementwiseFusionInst};
    use tenferro_tensor::ValidationError;

    let n = ELEMENTWISE_FUSION_MIN_ELEMENTS;
    let mut buffers = BufferPool::default();
    let input = Tensor::from_typed::<f64>(
        TypedTensor::<f64>::from_vec_col_major(vec![n], vec![1.0_f64; n]).unwrap(),
    );
    let plan = |view: ElementwiseFusionInputView| {
        ElementwiseFusionPlan::with_input_views(
            DType::F64,
            vec![view],
            vec![1],
            vec![ElementwiseFusionInst::new(
                ElementwiseFusionOp::Negate,
                vec![0],
            )],
        )
    };
    let run = |buffers: &mut BufferPool, input: &Tensor, plan: &ElementwiseFusionPlan| {
        elementwise_fusion_with_pool(buffers, &[input], plan, Exec::serial())
    };

    // `dims` rank disagrees with the input rank.
    let error = run(
        &mut buffers,
        &input,
        &plan(ElementwiseFusionInputView::broadcast_in_dim(
            vec![n],
            vec![0, 0],
        )),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::Error::Validation {
            source: ValidationError::InvalidArgument {
                argument: "configuration",
                ..
            },
            ..
        }
    ));

    // Target axis beyond the broadcast shape.
    let error = run(
        &mut buffers,
        &input,
        &plan(ElementwiseFusionInputView::broadcast_in_dim(
            vec![n],
            vec![1],
        )),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::Error::Validation {
            source: ValidationError::AxisOutOfBounds { axis: 1, rank: 1 },
            ..
        }
    ));

    // Duplicate target axis.
    let rank2 = Tensor::from_typed::<f64>(
        TypedTensor::<f64>::from_vec_col_major(vec![64, 512], vec![1.0_f64; 64 * 512]).unwrap(),
    );
    let error = run(
        &mut buffers,
        &rank2,
        &plan(ElementwiseFusionInputView::broadcast_in_dim(
            vec![64, 512],
            vec![0, 0],
        )),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::Error::Validation {
            source: ValidationError::DuplicateAxis {
                axis: 0,
                role: "broadcast dims"
            },
            ..
        }
    ));

    // Source extent is neither a broadcast nor the target extent.
    let small = Tensor::from_typed::<f64>(
        TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap(),
    );
    let error = run(
        &mut buffers,
        &small,
        &plan(ElementwiseFusionInputView::broadcast_in_dim(
            vec![3],
            vec![0],
        )),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::Error::Validation {
            source: ValidationError::ShapeMismatch(..),
            ..
        }
    ));
}

#[test]
fn elementwise_fusion_declines_empty_outputs_unfused_ops_and_mismatched_dims() {
    use tenferro_tensor::backend::{ElementwiseFusionInputView, ElementwiseFusionInst};

    let n = ELEMENTWISE_FUSION_MIN_ELEMENTS;
    let mut buffers = BufferPool::default();
    let input = Tensor::from_typed::<f64>(
        TypedTensor::<f64>::from_vec_col_major(vec![n], vec![1.0_f64; n]).unwrap(),
    );

    // Empty outputs: validation reports `false` and the region is declined.
    let empty = ElementwiseFusionPlan::new(DType::F64, 1, Vec::new(), Vec::new());
    assert!(
        elementwise_fusion_with_pool(&mut buffers, &[&input], &empty, Exec::serial())
            .unwrap()
            .is_none(),
        "an empty-output plan must be declined"
    );

    // `Remainder` has no `strided_fused` instruction.
    let rhs = Tensor::from_typed::<f64>(
        TypedTensor::<f64>::from_vec_col_major(vec![n], vec![2.0_f64; n]).unwrap(),
    );
    let remainder = ElementwiseFusionPlan::new(
        DType::F64,
        2,
        vec![2],
        vec![ElementwiseFusionInst::new(
            ElementwiseFusionOp::Remainder,
            vec![0, 1],
        )],
    );
    assert!(
        elementwise_fusion_with_pool(&mut buffers, &[&input, &rhs], &remainder, Exec::serial(),)
            .unwrap()
            .is_none(),
        "a plan with an unfused op must be declined"
    );

    // The mapped broadcast view widens the second input, so the input layouts
    // disagree and the plan is declined.
    let widened = ElementwiseFusionPlan::with_input_views(
        DType::F64,
        vec![
            ElementwiseFusionInputView::Identity,
            ElementwiseFusionInputView::broadcast_in_dim(vec![n, 2], vec![0]),
        ],
        vec![3],
        vec![ElementwiseFusionInst::new(
            ElementwiseFusionOp::Add,
            vec![0, 1],
        )],
    );
    assert!(
        elementwise_fusion_with_pool(&mut buffers, &[&input, &input], &widened, Exec::serial(),)
            .unwrap()
            .is_none(),
        "inputs with different mapped shapes must be declined"
    );
}
