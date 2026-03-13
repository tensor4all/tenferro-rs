use super::*;

#[test]
fn structured_reverse_pullback_accepts_dense_cotangent() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = TapeId(501);
    let node_x = NodeId(601);
    let node_alpha = NodeId(602);
    let x = AdTensor::new_reverse(
        StructuredTensor::from_diagonal_vector(
            Tensor::<f64>::from_slice(&[2.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
            2,
        )
        .unwrap(),
        node_x,
        tape,
        None,
    )
    .unwrap();
    let alpha = AdScalar::new_reverse(2.0_f64, node_alpha, tape, None);
    let y = DynAdTensor::from(x.clone())
        .scale(&DynAdScalar::from(alpha.as_value().clone()))
        .unwrap();
    let y = y.as_f64().unwrap();
    let dense_cotangent = AdTensor::new_primal(f64_2x2([1.0, 0.0, 0.0, 1.0]));

    let all_grads = pullback(y, &dense_cotangent).unwrap();
    assert_eq!(
        tensor_to_vec_f64(
            all_grads
                .get(&node_x)
                .expect("missing reverse payload gradient")
        ),
        vec![2.0, 2.0]
    );

    let wrt_tensor = pullback_wrt(y, &dense_cotangent, &[&x]).unwrap();
    let grad_x = wrt_tensor[0]
        .as_ref()
        .expect("missing structured wrt gradient");
    assert_eq!(tensor_to_vec_f64(grad_x), vec![2.0, 2.0]);
    assert_eq!(grad_x.axis_classes(), &[0, 0]);

    let wrt_scalar = pullback_wrt_scalars(y, &dense_cotangent, &[&alpha]).unwrap();
    assert_eq!(wrt_scalar, vec![Some(5.0)]);
}

#[test]
fn reshape_reverse_pullback_accepts_non_contiguous_cotangent_view() {
    let tape = TapeId(502);
    let node_x = NodeId(603);
    let x = AdTensor::new_reverse(f64_2x2([1.0, 2.0, 3.0, 4.0]), node_x, tape, None).unwrap();
    let reshaped = DynAdTensor::from(x.clone()).reshape(&[4]).unwrap();
    let reshaped = reshaped.as_f64().unwrap();

    let base = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[4, 4],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let cotangent_view = base.diagonal(&[(0, 1)]).unwrap();
    let expected = cotangent_view
        .contiguous(MemoryOrder::ColumnMajor)
        .reshape(&[2, 2])
        .unwrap();

    let grads = pullback_wrt(reshaped, &AdTensor::new_primal(cotangent_view), &[&x]).unwrap();
    let grad_x = grads[0].as_ref().expect("missing reshape gradient");
    assert_eq!(tensor_to_vec_f64(grad_x), tensor_to_vec_f64(&expected));
}

#[test]
fn dense_reverse_pullback_accepts_structured_cotangent() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = TapeId(503);
    let node_x = NodeId(604);
    let node_alpha = NodeId(605);
    let x = AdTensor::new_reverse(f64_2x2([2.0, 5.0, 7.0, 3.0]), node_x, tape, None).unwrap();
    let alpha = AdScalar::new_reverse(2.0_f64, node_alpha, tape, None);
    let y = DynAdTensor::from(x.clone())
        .scale(&DynAdScalar::from(alpha.as_value().clone()))
        .unwrap();
    let y = y.as_f64().unwrap();
    assert!(y.is_dense());

    let structured_cotangent = AdTensor::new_primal(
        StructuredTensor::from_diagonal_vector(
            Tensor::<f64>::from_slice(&[1.0, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
            2,
        )
        .unwrap(),
    );

    let all_grads = pullback(y, &structured_cotangent).unwrap();
    assert_eq!(
        tensor_to_vec_f64(
            all_grads
                .get(&node_x)
                .expect("missing dense reverse payload gradient")
        ),
        vec![2.0, 2.0, 2.0, 2.0]
    );

    let wrt_scalar = pullback_wrt_scalars(y, &structured_cotangent, &[&alpha]).unwrap();
    assert_eq!(wrt_scalar, vec![Some(17.0)]);
}

#[test]
fn pullback_helpers_preserve_none_for_untracked_wrt_inputs() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = TapeId(504);
    let node_x = NodeId(606);
    let node_alpha = NodeId(607);
    let x = AdTensor::new_reverse(f64_2x2([1.0, 2.0, 3.0, 4.0]), node_x, tape, None).unwrap();
    let alpha = AdScalar::new_reverse(3.0_f64, node_alpha, tape, None);
    let y = DynAdTensor::from(x.clone())
        .scale(&DynAdScalar::from(alpha.as_value().clone()))
        .unwrap();
    let y = y.as_f64().unwrap();
    let cotangent = AdTensor::new_primal(f64_2x2([1.0, 0.0, 0.0, 1.0]));

    let primal_tensor = AdTensor::new_primal(f64_2x2([9.0, 8.0, 7.0, 6.0]));
    let tensor_grads = pullback_wrt(y, &cotangent, &[&primal_tensor]).unwrap();
    assert_eq!(tensor_grads.len(), 1);
    assert!(tensor_grads[0].is_none());

    let scalar_grads =
        pullback_wrt_scalars(y, &cotangent, &[&AdScalar::new_primal(1.0_f64)]).unwrap();
    assert_eq!(scalar_grads, vec![None]);

    let mixed_grads = pullback_wrt_mixed(y, &cotangent, &[&primal_tensor]).unwrap();
    assert_eq!(mixed_grads.len(), 1);
    assert!(mixed_grads[0].is_none());
}

#[test]
fn pullback_helpers_reject_primal_outputs_and_mixed_tapes() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let primal_output = AdTensor::new_primal(f64_2x2([1.0, 2.0, 3.0, 4.0]));
    let cotangent = AdTensor::new_primal(f64_2x2([1.0, 0.0, 0.0, 1.0]));

    let err = pullback(&primal_output, &cotangent).unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidAdTensor { message }
            if message.contains("ad::pullback requires reverse-mode output tensor")
    ));

    let err = pullback_wrt(&primal_output, &cotangent, &[&primal_output]).unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidAdTensor { message }
            if message.contains("ad::pullback_wrt requires reverse-mode output tensor")
    ));

    let err = pullback_wrt_mixed(&primal_output, &cotangent, &[&primal_output]).unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidAdTensor { message }
            if message.contains("ad::pullback_wrt_mixed requires reverse-mode output tensor")
    ));

    let err = pullback_wrt_scalars(
        &primal_output,
        &cotangent,
        &[&AdScalar::new_primal(1.0_f64)],
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidAdTensor { message }
            if message.contains("ad::pullback_wrt_scalars requires reverse-mode output tensor")
    ));

    let x = AdTensor::new_reverse(
        f64_2x2([4.0, 3.0, 2.0, 1.0]),
        NodeId(608),
        TapeId(505),
        None,
    )
    .unwrap();
    let alpha = AdScalar::new_reverse(2.0_f64, NodeId(611), TapeId(505), None);
    let output = DynAdTensor::from(x)
        .scale(&DynAdScalar::from(alpha.as_value().clone()))
        .unwrap();
    let output = output.as_f64().unwrap();
    let wrt_tensor = AdTensor::new_reverse(
        f64_2x2([1.0, 0.0, 0.0, 1.0]),
        NodeId(609),
        TapeId(506),
        None,
    )
    .unwrap();
    let wrt_scalar = AdScalar::new_reverse(1.0_f64, NodeId(610), TapeId(506), None);

    let err = pullback_wrt(&output, &cotangent, &[&wrt_tensor]).unwrap_err();
    assert!(matches!(
        err,
        Error::MixedReverseTape {
            expected: 505,
            found: 506
        }
    ));

    let err = pullback_wrt_mixed(&output, &cotangent, &[&wrt_tensor]).unwrap_err();
    assert!(matches!(
        err,
        Error::MixedReverseTape {
            expected: 505,
            found: 506
        }
    ));

    let err = pullback_wrt_scalars(&output, &cotangent, &[&wrt_scalar]).unwrap_err();
    assert!(matches!(
        err,
        Error::MixedReverseTape {
            expected: 505,
            found: 506
        }
    ));
}
