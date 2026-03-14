use super::*;

#[test]
fn structured_reverse_pullback_accepts_dense_cotangent() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(
        StructuredTensor::from_diagonal_vector(
            Tensor::<f64>::from_slice(&[2.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
            2,
        )
        .unwrap(),
        &tape,
    );
    let alpha = reverse_leaf_f64(scalar_f64(2.0), &tape);
    let y = DynAdTensor::from(x.clone())
        .scale(&DynAdTensor::from(alpha.clone()))
        .unwrap();
    let y = y.as_f64().unwrap();
    let dense_cotangent = AdTensor::new_primal(f64_2x2([1.0, 0.0, 0.0, 1.0]));

    let all_grads = pullback(y, &dense_cotangent).unwrap();
    let node_x = x.node_id().unwrap();
    let node_alpha = alpha.node_id().unwrap();
    assert_eq!(
        tensor_to_vec_f64(
            all_grads
                .get(&node_x)
                .expect("missing reverse payload gradient")
        ),
        vec![2.0, 2.0]
    );
    assert_eq!(
        tensor_to_vec_f64(
            all_grads
                .get(&node_alpha)
                .expect("missing scalar payload gradient")
        ),
        vec![5.0]
    );

    let wrt = pullback_wrt(y, &dense_cotangent, &[&x, &alpha]).unwrap();
    let grad_x = wrt[0].as_ref().expect("missing structured wrt gradient");
    assert_eq!(tensor_to_vec_f64(grad_x), vec![2.0, 2.0]);
    assert_eq!(grad_x.axis_classes(), &[0, 0]);
    assert_eq!(tensor_to_vec_f64(wrt[1].as_ref().unwrap()), vec![5.0]);
}

#[test]
fn reshape_reverse_pullback_accepts_non_contiguous_cotangent_view() {
    let tape = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(f64_2x2([1.0, 2.0, 3.0, 4.0]), &tape);
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

    let tape = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(f64_2x2([2.0, 5.0, 7.0, 3.0]), &tape);
    let alpha = reverse_leaf_f64(scalar_f64(2.0), &tape);
    let y = DynAdTensor::from(x.clone())
        .scale(&DynAdTensor::from(alpha.clone()))
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
    let node_x = x.node_id().unwrap();
    let node_alpha = alpha.node_id().unwrap();
    assert_eq!(
        tensor_to_vec_f64(
            all_grads
                .get(&node_x)
                .expect("missing dense reverse payload gradient")
        ),
        vec![2.0, 2.0, 2.0, 2.0]
    );
    assert_eq!(
        tensor_to_vec_f64(
            all_grads
                .get(&node_alpha)
                .expect("missing dense reverse scalar gradient")
        ),
        vec![17.0]
    );
}

#[test]
fn pullback_helpers_preserve_none_for_untracked_wrt_inputs() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(f64_2x2([1.0, 2.0, 3.0, 4.0]), &tape);
    let alpha = reverse_leaf_f64(scalar_f64(3.0), &tape);
    let y = DynAdTensor::from(x.clone())
        .scale(&DynAdTensor::from(alpha))
        .unwrap();
    let y = y.as_f64().unwrap();
    let cotangent = AdTensor::new_primal(f64_2x2([1.0, 0.0, 0.0, 1.0]));

    let primal_tensor = AdTensor::new_primal(f64_2x2([9.0, 8.0, 7.0, 6.0]));
    let primal_scalar = AdTensor::new_primal(scalar_f64(1.0));
    let grads = pullback_wrt(y, &cotangent, &[&primal_tensor, &primal_scalar]).unwrap();
    assert_eq!(grads.len(), 2);
    assert!(grads[0].is_none());
    assert!(grads[1].is_none());
}

#[test]
fn pullback_wrt_returns_none_for_disconnected_reverse_tensor() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(f64_2x2([1.0, 2.0, 3.0, 4.0]), &tape);
    let alpha = reverse_leaf_f64(scalar_f64(3.0), &tape);
    let disconnected = reverse_leaf_f64(f64_2x2([9.0, 8.0, 7.0, 6.0]), &tape);
    let y = DynAdTensor::from(x)
        .scale(&DynAdTensor::from(alpha))
        .unwrap();
    let y = y.as_f64().unwrap();
    let cotangent = AdTensor::new_primal(f64_2x2([1.0, 0.0, 0.0, 1.0]));

    let grads = pullback_wrt(y, &cotangent, &[&disconnected]).unwrap();
    assert_eq!(grads.len(), 1);
    assert!(grads[0].is_none());
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

    let tape_output = Tape::<crate::DynTensor>::new();
    let tape_wrt = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(f64_2x2([4.0, 3.0, 2.0, 1.0]), &tape_output);
    let alpha = reverse_leaf_f64(scalar_f64(2.0), &tape_output);
    let output = DynAdTensor::from(x)
        .scale(&DynAdTensor::from(alpha))
        .unwrap();
    let output = output.as_f64().unwrap();
    let wrt_tensor = reverse_leaf_f64(f64_2x2([1.0, 0.0, 0.0, 1.0]), &tape_wrt);

    let err = pullback_wrt(&output, &cotangent, &[&wrt_tensor]).unwrap_err();
    assert!(matches!(err, Error::MixedReverseTape { .. }));
}
