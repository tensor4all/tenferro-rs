use super::*;
use crate::ops::tests::support::{assert_forward_mode, assert_reverse_on_tape};
use crate::AdMode;
use ::chainrules::Tape;

#[test]
fn runtime_helpers_cover_mode_and_shape_paths() {
    let primal =
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let tangent =
        Tensor::<f64>::from_slice(&[0.1, 0.2, 0.3, 0.4], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();

    let ad_primal = AdTensor::new_primal(primal.clone());
    let ad_forward = AdTensor::new_forward(primal.clone(), tangent.clone()).unwrap();
    let tape = Tape::<StructuredTensor<f64>>::new();
    let other_tape = Tape::<StructuredTensor<f64>>::new();
    let ad_reverse = reverse_leaf_f64(primal.clone(), &tape);
    let ad_reverse_other = reverse_leaf_f64(primal.clone(), &other_tape);

    assert!(has_forward(&[&ad_primal, &ad_forward]));
    assert!(has_reverse(&[&ad_primal, &ad_reverse]));
    assert!(has_any_tangent(&[&ad_primal, &ad_forward]));
    let derived_tape = derive_reverse_tape_handle(&[&ad_primal, &ad_reverse])
        .unwrap()
        .expect("expected reverse tape handle");
    assert!(derived_tape.same_tape(&tape));
    assert!(matches!(
        derive_reverse_tape_handle(&[&ad_reverse, &ad_reverse_other]),
        Err(Error::MixedReverseTape {
            expected: e,
            found: f
        }) if e == tape.id() as u64 && f == other_tape.id() as u64
    ));

    let specs = collect_reverse_input_specs(&[&ad_reverse, &ad_primal]);
    assert_eq!(specs.len(), 2);
    assert_eq!(
        specs[0].as_ref().unwrap().node,
        ad_reverse.node_id().unwrap()
    );
    assert!(specs[1].is_none());

    let reshaped = normalize_pullback_shape(
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], MemoryOrder::ColumnMajor).unwrap(),
        &[2, 2],
        "runtime_helper",
    )
    .unwrap();
    assert_eq!(reshaped.dims(), &[2, 2]);

    let tangent_err = normalize_output_tangent_shape(
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap(),
        &[2, 2],
        "runtime_helper",
    )
    .unwrap_err();
    assert!(matches!(
        tangent_err,
        Error::InvalidAdTensor { message } if message.contains("tangent shape mismatch")
    ));

    let wrapped_primal =
        wrap_dense_ad_output("runtime_helper", &[&ad_primal], primal.clone(), None).unwrap();
    assert_eq!(wrapped_primal.mode(), AdMode::Primal);

    let wrapped_forward = wrap_dense_ad_output(
        "runtime_helper",
        &[&ad_forward],
        primal.clone(),
        Some(tangent.clone()),
    )
    .unwrap();
    assert_forward_mode(&wrapped_forward);

    let wrapped_reverse =
        wrap_same_type_dense_ad_output("runtime_helper", &[&ad_reverse], primal.clone(), None)
            .unwrap();
    assert_reverse_on_tape(&wrapped_reverse, &tape);

    let forward_err =
        match wrap_dense_ad_output("runtime_helper", &[&ad_forward], primal.clone(), None) {
            Ok(_) => panic!("forward output without tangent should be rejected"),
            Err(err) => err,
        };
    assert!(matches!(
        forward_err,
        Error::InvalidAdTensor { message } if message.contains("forward-mode inputs must provide tangent output")
    ));
}

#[test]
fn runtime_helpers_cover_scalar_and_tangent_accumulation() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let da = Tensor::<f64>::from_slice(&[0.5, 0.0, -0.5, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let diag_layout = StructuredTensor::from_diagonal_vector(
        Tensor::<f64>::from_slice(&[1.0, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
        2,
    )
    .unwrap();
    let diag_grad =
        Tensor::<f64>::from_slice(&[9.0, 0.0, 0.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let structured_a = StructuredTensor::from_dense(a.clone());
    let structured_b = StructuredTensor::from_dense(b.clone());
    let structured_da = StructuredTensor::from_dense(da.clone());
    let expected = with_cpu_runtime("runtime_helper", |ctx| {
        tenferro_einsum::einsum::<Standard<f64>, CpuBackend>(ctx, "ij,jk->ik", &[&da, &b], None)
            .map_err(Error::from)
    })
    .unwrap();

    let summed = super::scalar::primal::scalar_full_reduction_primal(
        "runtime_helper",
        tenferro_prims::ScalarReductionOp::Sum,
        &a,
    )
    .unwrap();
    assert_eq!(summed.dims(), &[]);
    assert_eq!(
        scalar_from_rank0_tensor(&summed, "runtime_helper").unwrap(),
        10.0
    );
    assert!(matches!(
        scalar_from_rank0_tensor(&a, "runtime_helper"),
        Err(Error::InvalidAdTensor { message }) if message.contains("expects rank-0 cotangent")
    ));

    let broadcast = broadcast_scalar_like(2.5, &a).unwrap();
    assert_eq!(broadcast.dims(), a.dims());
    assert_eq!(as_slice(&broadcast), &[2.5, 2.5, 2.5, 2.5]);

    with_cpu_runtime("runtime_helper", |ctx| {
        let primals = [&a, &b];
        let tangents = [Some(&da), None];
        let tangent = sum_einsum_tangent_terms::<CpuBackend, _, f64>(
            ctx,
            "ij,jk->ik",
            &primals,
            &tangents,
            None,
        )?
        .unwrap();
        assert_eq!(as_slice(&tangent), as_slice(&expected));

        let subs = tenferro_einsum::Subscripts::parse("ij,jk->ik").map_err(Error::from)?;
        let structured_tangent = sum_structured_einsum_tangent_terms::<CpuBackend, _, f64>(
            ctx,
            &subs,
            &[&structured_a, &structured_b],
            &[Some(&structured_da), None],
        )?
        .unwrap();
        assert_eq!(as_slice(structured_tangent.payload()), as_slice(&expected));

        Ok(())
    })
    .unwrap();

    let compressed_dense =
        compress_pullback_like("runtime_helper", expected.clone(), &structured_a).unwrap();
    assert_eq!(as_slice(&compressed_dense), as_slice(&expected));
    let compressed_diag =
        compress_pullback_like("runtime_helper", diag_grad, &diag_layout).unwrap();
    assert_eq!(as_slice(&compressed_diag), &[9.0, 8.0]);
}

#[test]
fn sum_ad_reverse_pullback_broadcasts_scalar_cotangent() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let tape = Tape::<StructuredTensor<f64>>::new();
    let ad_x = reverse_leaf_f64(x.clone(), &tape);
    let out = sum_ad(&ad_x).run().unwrap();
    assert_reverse_on_tape(&out, &tape);

    let cotangent = AdTensor::new_primal(
        Tensor::<f64>::from_slice(&[3.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    let grads = crate::ops::ad::pullback_wrt(&out, &cotangent, &[&ad_x]).unwrap();
    let grad = grads[0].as_ref().unwrap();
    assert_eq!(grad.logical_dims(), &[2, 2]);
    assert_eq!(as_slice(grad.payload()), &[3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn einsum_ad_size_dict_forces_dense_path_and_registers_pullback() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let tape = Tape::<StructuredTensor<f64>>::new();
    let ad_a = reverse_leaf_f64(a.clone(), &tape);
    let ad_b = reverse_leaf_f64(b.clone(), &tape);
    let size_dict = HashMap::new();

    let out = einsum_ad("ij,jk->ik", &[&ad_a, &ad_b])
        .size_dict(&size_dict)
        .run()
        .unwrap();
    assert_reverse_on_tape(&out, &tape);

    let cotangent =
        Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let grads = crate::ops::ad::pullback_wrt(
        &out,
        &AdTensor::new_primal(cotangent.clone()),
        &[&ad_a, &ad_b],
    )
    .unwrap();
    let got_a = grads[0].as_ref().unwrap().payload();
    let got_b = grads[1].as_ref().unwrap().payload();

    let expected = with_cpu_runtime("einsum_dense_path_expected", |ctx| {
        tenferro_einsum::einsum_rrule::<Standard<f64>, CpuBackend>(
            ctx,
            "ij,jk->ik",
            &[&a, &b],
            &cotangent,
        )
        .map_err(Error::from)
    })
    .unwrap();

    assert_eq!(as_slice(got_a), as_slice(&expected[0]));
    assert_eq!(as_slice(got_b), as_slice(&expected[1]));
}
