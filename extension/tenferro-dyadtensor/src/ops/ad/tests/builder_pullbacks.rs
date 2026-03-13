use super::*;

#[test]
fn solve_triangular_builder_reverse_pullback_matches_rrule() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = TapeId(101);
    let node_a = NodeId(11);
    let node_b = NodeId(12);

    let a = f64_2x2([2.0, 0.0, 1.0, 3.0]);
    let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let cotangent =
        Tensor::<f64>::from_slice(&[0.5, -0.25], &[2], MemoryOrder::ColumnMajor).unwrap();

    let ad_a_rev = AdTensor::new_reverse(a.clone(), node_a, tape, None).unwrap();
    let ad_b_rev = AdTensor::new_reverse(b.clone(), node_b, tape, None).unwrap();
    let out = solve_triangular(&ad_a_rev, &ad_b_rev).unwrap();
    assert!(matches!(out.as_value(), AdValue::Reverse { tape: t, .. } if *t == tape));

    let ad_cotangent = AdTensor::new_primal(cotangent);
    let grad_map = pullback(&out, &ad_cotangent).unwrap();
    let grad_a = grad_map.get(&node_a).expect("missing dA");
    let grad_b = grad_map.get(&node_b).expect("missing dB");

    let expected = solve_triangular_rrule(
        &AdTensor::new_primal(a),
        &AdTensor::new_primal(b.clone()),
        &ad_cotangent,
        true,
    )
    .unwrap();

    assert_eq!(grad_a.dims(), &[2, 2]);
    assert!(max_abs_diff(grad_a, &expected.a) < 1e-12);

    let expected_b = expected.b.reshape(b.dims()).unwrap();
    assert_eq!(grad_b.dims(), b.dims());
    assert!(max_abs_diff(grad_b, &expected_b) < 1e-12);
}

#[test]
fn solve_builder_reverse_pullback_matches_rrule() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = TapeId(111);
    let node_a = NodeId(51);
    let node_b = NodeId(52);

    let a = f64_2x2([3.0, 1.0, 1.0, 2.0]);
    let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let cotangent =
        Tensor::<f64>::from_slice(&[0.5, -0.25], &[2], MemoryOrder::ColumnMajor).unwrap();

    let ad_a_rev = AdTensor::new_reverse(a.clone(), node_a, tape, None).unwrap();
    let ad_b_rev = AdTensor::new_reverse(b.clone(), node_b, tape, None).unwrap();
    let out = solve(&ad_a_rev, &ad_b_rev).unwrap();
    assert!(matches!(out.as_value(), AdValue::Reverse { tape: t, .. } if *t == tape));

    let ad_cotangent = AdTensor::new_primal(cotangent.clone());
    let grads = pullback_wrt(&out, &ad_cotangent, &[&ad_a_rev, &ad_b_rev]).unwrap();
    let grad_a = grads[0].as_ref().expect("missing solve dA");
    let grad_b = grads[1].as_ref().expect("missing solve dB");

    let expected = with_cpu_runtime("solve_rrule_expected", |ctx| {
        tenferro_linalg::solve_rrule::<f64, _>(ctx, &a, &b, &cotangent).map_err(Error::from)
    })
    .unwrap();

    let expected_b = if expected.b.dims() == b.dims() {
        expected.b
    } else {
        expected.b.reshape(b.dims()).unwrap()
    };

    assert!(max_abs_diff(grad_a, &expected.a) < 1e-12);
    assert!(max_abs_diff(grad_b, &expected_b) < 1e-12);
}

#[test]
fn norm_builder_reverse_pullback_l1_matches_rrule() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = TapeId(112);
    let node_a = NodeId(61);

    let a = Tensor::<f64>::from_slice(&[1.0, 3.0, -2.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let cotangent: Tensor<f64> = Tensor::from_vec(vec![1.5], &[], &[], 0).unwrap();

    let ad_a_rev = AdTensor::new_reverse(a.clone(), node_a, tape, None).unwrap();
    let out = crate::norm_ad(&ad_a_rev).kind(NormKind::L1).run().unwrap();
    assert!(matches!(out.as_value(), AdValue::Reverse { tape: t, .. } if *t == tape));

    let ad_cotangent = AdTensor::new_primal(cotangent.clone());
    let grads = pullback_wrt(&out, &ad_cotangent, &[&ad_a_rev]).unwrap();
    let grad_a = grads[0].as_ref().expect("missing norm dA");

    let expected = with_cpu_runtime("norm_rrule_expected", |ctx| {
        tenferro_linalg::norm_rrule::<f64, _>(ctx, &a, &cotangent, NormKind::L1)
            .map_err(Error::from)
    })
    .unwrap();

    assert!(max_abs_diff(grad_a, &expected) < 1e-12);
}

#[test]
fn einsum_builder_reverse_pullback_wrt_matches_rrule() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = TapeId(202);
    let node_a = NodeId(31);
    let node_b = NodeId(32);

    let a = f64_2x2([1.0, 3.0, 2.0, 4.0]);
    let b = f64_2x2([2.0, -1.0, 0.5, 1.5]);
    let cotangent = f64_2x2([1.0, 0.0, 0.0, 1.0]);

    let ad_a_rev = AdTensor::new_reverse(a.clone(), node_a, tape, None).unwrap();
    let ad_b_rev = AdTensor::new_reverse(b.clone(), node_b, tape, None).unwrap();
    let out = einsum("ij,jk->ik", &[&ad_a_rev, &ad_b_rev]).unwrap();
    assert!(matches!(out.as_value(), AdValue::Reverse { tape: t, .. } if *t == tape));

    let ad_cotangent = AdTensor::new_primal(cotangent.clone());
    let grads = pullback_wrt(&out, &ad_cotangent, &[&ad_a_rev, &ad_b_rev]).unwrap();
    let grad_a = grads[0].as_ref().expect("missing einsum dA");
    let grad_b = grads[1].as_ref().expect("missing einsum dB");

    let expected = einsum_rrule(
        "ij,jk->ik",
        &[&AdTensor::new_primal(a), &AdTensor::new_primal(b)],
        &AdTensor::new_primal(cotangent),
    )
    .unwrap();

    assert!(max_abs_diff(grad_a, &expected[0]) < 1e-12);
    assert!(max_abs_diff(grad_b, &expected[1]) < 1e-12);
}

#[test]
fn solve_triangular_reverse_pullback_complex_matches_rrule() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = TapeId(303);
    let node_a = NodeId(41);
    let node_b = NodeId(42);

    let a = Tensor::<Complex64>::from_slice(
        &[
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -0.5),
            Complex64::new(3.0, 0.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b = Tensor::<Complex64>::from_slice(
        &[Complex64::new(1.0, 0.5), Complex64::new(2.0, -0.25)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let cotangent = Tensor::<Complex64>::from_slice(
        &[Complex64::new(0.5, 0.0), Complex64::new(-0.25, 0.1)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let ad_a_rev = AdTensor::new_reverse(a.clone(), node_a, tape, None).unwrap();
    let ad_b_rev = AdTensor::new_reverse(b.clone(), node_b, tape, None).unwrap();
    let out = solve_triangular(&ad_a_rev, &ad_b_rev).unwrap();
    let grads = pullback(&out, &AdTensor::new_primal(cotangent.clone())).unwrap();
    let grad_a = grads.get(&node_a).expect("missing complex dA");
    let grad_b = grads.get(&node_b).expect("missing complex dB");

    let expected = solve_triangular_rrule(
        &AdTensor::new_primal(a),
        &AdTensor::new_primal(b.clone()),
        &AdTensor::new_primal(cotangent),
        true,
    )
    .unwrap();

    let expected_b = expected.b.reshape(b.dims()).unwrap();
    assert!(complex_max_abs_diff(grad_a, &expected.a) < 1e-12);
    assert!(complex_max_abs_diff(grad_b, &expected_b) < 1e-12);
}

#[test]
fn svd_builder_reverse_pullback_s_matches_rrule() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = TapeId(401);
    let node_a = NodeId(71);
    let a = f64_2x2([3.0, 1.0, 0.5, 2.0]);

    let ad_a_rev = AdTensor::new_reverse(a.clone(), node_a, tape, None).unwrap();
    let out = svd(&ad_a_rev).unwrap();
    assert!(matches!(out.s.as_value(), AdValue::Reverse { tape: t, .. } if *t == tape));

    let cotangent_s =
        Tensor::<f64>::from_slice(&[1.0, -0.5], &[2], MemoryOrder::ColumnMajor).unwrap();
    let ad_cotangent = AdTensor::new_primal(cotangent_s.clone());
    let grads = pullback_wrt(&out.s, &ad_cotangent, &[&ad_a_rev]).unwrap();
    let grad_a = grads[0].as_ref().expect("missing svd dA");

    let expected = with_cpu_runtime("svd_rrule_expected", |ctx| {
        tenferro_linalg::svd_rrule::<f64, _>(
            ctx,
            &a,
            &tenferro_linalg::SvdCotangent {
                u: None,
                s: Some(cotangent_s.clone()),
                vt: None,
            },
            None,
        )
        .map_err(Error::from)
    })
    .unwrap();

    assert!(max_abs_diff(grad_a, &expected) < 1e-12);
}

#[test]
fn lstsq_builder_reverse_pullback_x_matches_rrule() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = TapeId(402);
    let node_a = NodeId(72);
    let node_b = NodeId(73);
    let a = Tensor::<f64>::from_slice(
        &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();

    let ad_a_rev = AdTensor::new_reverse(a.clone(), node_a, tape, None).unwrap();
    let ad_b_rev = AdTensor::new_reverse(b.clone(), node_b, tape, None).unwrap();
    let out = lstsq(&ad_a_rev, &ad_b_rev).unwrap();
    assert!(matches!(out.x.as_value(), AdValue::Reverse { tape: t, .. } if *t == tape));

    let cotangent_x =
        Tensor::<f64>::from_slice(&[0.3, -0.7], &[2], MemoryOrder::ColumnMajor).unwrap();
    let ad_cotangent = AdTensor::new_primal(cotangent_x.clone());
    let grads = pullback_wrt(&out.x, &ad_cotangent, &[&ad_a_rev, &ad_b_rev]).unwrap();
    let grad_a = grads[0].as_ref().expect("missing lstsq dA");
    let grad_b = grads[1].as_ref().expect("missing lstsq dB");

    let expected = with_cpu_runtime("lstsq_rrule_expected", |ctx| {
        tenferro_linalg::lstsq_rrule::<f64, _>(ctx, &a, &b, &cotangent_x).map_err(Error::from)
    })
    .unwrap();

    assert!(max_abs_diff(grad_a, &expected.a) < 1e-12);
    assert!(max_abs_diff(grad_b, &expected.b) < 1e-12);
}

#[test]
fn eig_builder_reverse_pullback_values_matches_rrule_for_real_wrt() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = TapeId(403);
    let node_a = NodeId(74);
    let a = Tensor::<f64>::from_slice(&[0.0, -1.0, 1.0, 0.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();

    let ad_a_rev = AdTensor::new_reverse(a.clone(), node_a, tape, None).unwrap();
    let out = eig(&ad_a_rev).unwrap();
    assert!(matches!(out.values.as_value(), AdValue::Reverse { tape: t, .. } if *t == tape));

    let cotangent_values = Tensor::<Complex64>::from_slice(
        &[Complex64::new(1.0, 0.0), Complex64::new(-0.25, 0.5)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let ad_cotangent = AdTensor::new_primal(cotangent_values.clone());

    let grads = pullback_wrt_mixed(&out.values, &ad_cotangent, &[&ad_a_rev]).unwrap();
    let grad_a = grads[0].as_ref().expect("missing eig dA");

    let expected = with_cpu_runtime("eig_rrule_expected", |ctx| {
        tenferro_linalg::eig_rrule::<f64, _>(
            ctx,
            &a,
            &tenferro_linalg::EigCotangent {
                values: Some(cotangent_values.clone()),
                vectors: None,
            },
        )
        .map_err(Error::from)
    })
    .unwrap();

    assert!(max_abs_diff(grad_a, &expected) < 1e-12);
}

#[test]
fn multi_output_builders_register_reverse_pullback_smoke() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = TapeId(404);
    let node_a = NodeId(75);
    let a = f64_2x2([4.0, 1.0, 1.0, 3.0]);
    let ad_a_rev = AdTensor::new_reverse(a, node_a, tape, None).unwrap();

    let qr_out = qr(&ad_a_rev).unwrap();
    let qr_cot_q = AdTensor::new_primal(Tensor::<f64>::ones(
        qr_out.q.dims(),
        qr_out.q.primal().logical_memory_space(),
        MemoryOrder::ColumnMajor,
    ));
    assert!(pullback_wrt(&qr_out.q, &qr_cot_q, &[&ad_a_rev]).unwrap()[0].is_some());
    let qr_cot_r = AdTensor::new_primal(Tensor::<f64>::ones(
        qr_out.r.dims(),
        qr_out.r.primal().logical_memory_space(),
        MemoryOrder::ColumnMajor,
    ));
    assert!(pullback_wrt(&qr_out.r, &qr_cot_r, &[&ad_a_rev]).unwrap()[0].is_some());

    let lu_out = lu(&ad_a_rev).unwrap();
    let lu_cot_l = AdTensor::new_primal(Tensor::<f64>::ones(
        lu_out.l.dims(),
        lu_out.l.primal().logical_memory_space(),
        MemoryOrder::ColumnMajor,
    ));
    assert!(pullback_wrt(&lu_out.l, &lu_cot_l, &[&ad_a_rev]).unwrap()[0].is_some());
    let lu_cot_u = AdTensor::new_primal(Tensor::<f64>::ones(
        lu_out.u.dims(),
        lu_out.u.primal().logical_memory_space(),
        MemoryOrder::ColumnMajor,
    ));
    assert!(pullback_wrt(&lu_out.u, &lu_cot_u, &[&ad_a_rev]).unwrap()[0].is_some());

    let eigen_out = eigen(&ad_a_rev).unwrap();
    let eigen_cot_values = AdTensor::new_primal(Tensor::<f64>::ones(
        eigen_out.values.dims(),
        eigen_out.values.primal().logical_memory_space(),
        MemoryOrder::ColumnMajor,
    ));
    assert!(pullback_wrt(&eigen_out.values, &eigen_cot_values, &[&ad_a_rev]).unwrap()[0].is_some());
    let eigen_cot_vectors = AdTensor::new_primal(Tensor::<f64>::ones(
        eigen_out.vectors.dims(),
        eigen_out.vectors.primal().logical_memory_space(),
        MemoryOrder::ColumnMajor,
    ));
    assert!(
        pullback_wrt(&eigen_out.vectors, &eigen_cot_vectors, &[&ad_a_rev]).unwrap()[0].is_some()
    );

    let slogdet_out = slogdet(&ad_a_rev).unwrap();
    let slogdet_cot_sign = AdTensor::new_primal(Tensor::<f64>::ones(
        slogdet_out.sign.dims(),
        slogdet_out.sign.primal().logical_memory_space(),
        MemoryOrder::ColumnMajor,
    ));
    let sign_grad = pullback_wrt(&slogdet_out.sign, &slogdet_cot_sign, &[&ad_a_rev]).unwrap();
    let sign_grad_a = sign_grad[0]
        .as_ref()
        .expect("missing slogdet sign gradient");
    assert!(as_slice(sign_grad_a).iter().all(|x| x.abs() < 1e-12));

    let slogdet_cot_logabs = AdTensor::new_primal(Tensor::<f64>::ones(
        slogdet_out.logabsdet.dims(),
        slogdet_out.logabsdet.primal().logical_memory_space(),
        MemoryOrder::ColumnMajor,
    ));
    assert!(
        pullback_wrt(&slogdet_out.logabsdet, &slogdet_cot_logabs, &[&ad_a_rev]).unwrap()[0]
            .is_some()
    );

    let node_ls_a = NodeId(76);
    let node_ls_b = NodeId(77);
    let a_ls = Tensor::<f64>::from_slice(
        &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_ls = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let ad_ls_a = AdTensor::new_reverse(a_ls, node_ls_a, tape, None).unwrap();
    let ad_ls_b = AdTensor::new_reverse(b_ls, node_ls_b, tape, None).unwrap();
    let lstsq_out = lstsq(&ad_ls_a, &ad_ls_b).unwrap();
    let lstsq_cot_x = AdTensor::new_primal(Tensor::<f64>::ones(
        lstsq_out.x.dims(),
        lstsq_out.x.primal().logical_memory_space(),
        MemoryOrder::ColumnMajor,
    ));
    let grads_x = pullback_wrt(&lstsq_out.x, &lstsq_cot_x, &[&ad_ls_a, &ad_ls_b]).unwrap();
    assert!(grads_x[0].is_some());
    assert!(grads_x[1].is_some());

    let lstsq_cot_residual = AdTensor::new_primal(Tensor::<f64>::ones(
        lstsq_out.residual.dims(),
        lstsq_out.residual.primal().logical_memory_space(),
        MemoryOrder::ColumnMajor,
    ));
    let grads_res = pullback_wrt(
        &lstsq_out.residual,
        &lstsq_cot_residual,
        &[&ad_ls_a, &ad_ls_b],
    )
    .unwrap();
    let grad_res_a = grads_res[0]
        .as_ref()
        .expect("missing lstsq residual gradient for A");
    let grad_res_b = grads_res[1]
        .as_ref()
        .expect("missing lstsq residual gradient for b");
    assert!(as_slice(grad_res_a).iter().all(|x| x.abs() < 1e-12));
    assert!(as_slice(grad_res_b).iter().all(|x| x.abs() < 1e-12));
}
