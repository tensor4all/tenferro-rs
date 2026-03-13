use super::*;

#[test]
fn eager_ad_linalg_and_einsum_cover_all_ops() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = f64_2x2([4.0, 1.0, 1.0, 3.0]);
    let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let tri = f64_2x2([2.0, 0.0, 1.0, 3.0]);
    let general = f64_2x2([0.0, 1.0, -1.0, 0.0]);
    let rect = Tensor::<f64>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let a_ls = Tensor::<f64>::from_slice(
        &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_ls = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();

    let ad_a = AdTensor::new_primal(a.clone());
    let ad_b = AdTensor::new_primal(b);
    let _ad_tri = AdTensor::new_primal(tri);
    let ad_general = AdTensor::new_primal(general);
    let ad_rect = AdTensor::new_primal(rect);
    let ad_ls_a = AdTensor::new_primal(a_ls);
    let ad_ls_b = AdTensor::new_primal(b_ls);

    let out_einsum = einsum("ij,jk->ik", &[&ad_a, &ad_a]).unwrap();
    assert_eq!(out_einsum.dims(), &[2, 2]);
    let out_svd = svd(&ad_a).unwrap();
    assert_eq!(out_svd.s.dims(), &[2]);
    let out_qr = qr(&ad_a).unwrap();
    assert_eq!(out_qr.q.dims(), &[2, 2]);
    let out_lu = lu(&ad_a).unwrap();
    assert_eq!(out_lu.l.dims(), &[2, 2]);
    let out_eigen = eigen(&ad_a).unwrap();
    assert_eq!(out_eigen.values.dims(), &[2]);
    let out_lstsq = lstsq(&ad_ls_a, &ad_ls_b).unwrap();
    assert_eq!(out_lstsq.x.dims(), &[2]);
    assert_eq!(cholesky(&ad_a).unwrap().dims(), &[2, 2]);
    assert_eq!(solve(&ad_a, &ad_b).unwrap().dims(), &[2]);
    assert_eq!(inv(&ad_a).unwrap().dims(), &[2, 2]);
    assert_eq!(det(&ad_a).unwrap().dims(), &[]);
    let out_slogdet = slogdet(&ad_a).unwrap();
    assert_eq!(out_slogdet.sign.dims(), &[]);
    let out_eig = eig(&ad_general).unwrap();
    assert_eq!(out_eig.values.dims(), &[2]);
    assert_eq!(pinv(&ad_rect).unwrap().dims(), &[3, 2]);
    assert_eq!(matrix_exp(&ad_a).unwrap().dims(), &[2, 2]);
    assert_eq!(solve_triangular(&ad_a, &ad_b).unwrap().dims(), &[2]);
    assert_eq!(norm(&ad_a).unwrap().dims(), &[]);
}

#[test]
fn eager_ad_preserves_mode_propagation() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = f64_2x2([4.0, 1.0, 1.0, 3.0]);
    let da = f64_2x2([0.1, 0.0, 0.0, 0.1]);
    let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();

    let ad_a_fwd = AdTensor::new_forward(a.clone(), da).unwrap();
    let ad_b = AdTensor::new_primal(b);
    let out_fwd = solve(&ad_a_fwd, &ad_b).unwrap();
    assert!(matches!(out_fwd.as_value(), AdValue::Forward { .. }));

    let ad_a_rev = AdTensor::new_reverse(a.clone(), NodeId(1), TapeId(11), None).unwrap();
    let ad_b_rev = AdTensor::new_reverse(a, NodeId(2), TapeId(11), None).unwrap();
    let out_rev = einsum("ij,jk->ik", &[&ad_a_rev, &ad_b_rev]).unwrap();
    assert!(matches!(out_rev.as_value(), AdValue::Reverse { tape, .. } if *tape == TapeId(11)));
}

#[test]
fn eager_local_einsum_rules_cover_rrule_frule_hvp() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = f64_2x2([1.0, 3.0, 2.0, 4.0]);
    let b = f64_2x2([2.0, -1.0, 0.5, 1.5]);
    let da = f64_2x2([0.1, 0.0, -0.2, 0.3]);
    let grad_c = f64_2x2([1.0, 0.0, 0.0, 1.0]);
    let dgrad_c = f64_2x2([0.5, 0.0, 0.0, 0.5]);

    let ad_a = AdTensor::new_primal(a);
    let ad_b = AdTensor::new_primal(b);
    let ad_da = AdTensor::new_primal(da);
    let ad_grad_c = AdTensor::new_primal(grad_c);
    let ad_dgrad_c = AdTensor::new_primal(dgrad_c);

    let grads = einsum_rrule("ij,jk->ik", &[&ad_a, &ad_b], &ad_grad_c).unwrap();
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].dims(), &[2, 2]);
    assert_eq!(grads[1].dims(), &[2, 2]);

    let jvp = einsum_frule("ij,jk->ik", &[&ad_a, &ad_b], &[Some(&ad_da), None]).unwrap();
    assert_eq!(jvp.dims(), &[2, 2]);

    let hvp = einsum_hvp(
        "ij,jk->ik",
        &[&ad_a, &ad_b],
        &[Some(&ad_da), None],
        &ad_grad_c,
        &ad_dgrad_c,
    )
    .unwrap();
    assert_eq!(hvp.len(), 2);
    assert_eq!(hvp[0].0.dims(), &[2, 2]);
    assert_eq!(hvp[0].1.dims(), &[2, 2]);
}
