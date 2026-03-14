use super::*;
use crate::ops::tests::support::{assert_forward_mode, assert_reverse_on_tape};
use crate::AdMode;
use ::chainrules::Tape;

#[test]
fn public_ad_builders_cover_helper_paths_and_builder_options() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = Tensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let da = Tensor::<f64>::from_slice(&[0.1, 0.0, 0.0, 0.1], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let b = Tensor::<f64>::from_slice(&[2.0, 0.0, 0.0, 2.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let db = Tensor::<f64>::from_slice(&[0.0, 0.2, 0.3, 0.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();

    let ad_a_fwd = AdTensor::new_forward(a.clone(), da.clone()).unwrap();
    let out_unary_fwd = cholesky_ad(&ad_a_fwd).run().unwrap();
    assert_forward_mode(&out_unary_fwd);

    let tape_rev = Tape::<StructuredTensor<f64>>::new();
    let ad_a_rev = reverse_leaf_f64(a.clone(), &tape_rev);
    let out_unary_rev = cholesky_ad(&ad_a_rev).run().unwrap();
    let unary_cotangent = AdTensor::new_primal(
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    );
    let unary_grads =
        crate::ops::ad::pullback_wrt(&out_unary_rev, &unary_cotangent, &[&ad_a_rev]).unwrap();
    assert!(unary_grads[0].is_some());
    assert_eq!(unary_grads[0].as_ref().unwrap().logical_dims(), &[2, 2]);

    let ad_b_fwd = AdTensor::new_forward(b.clone(), db.clone()).unwrap();
    let out_binary_fwd = solve_ad(&ad_a_fwd, &ad_b_fwd).run().unwrap();
    assert_forward_mode(&out_binary_fwd);

    let ad_b_rev = reverse_leaf_f64(b.clone(), &tape_rev);
    let out_binary_rev = solve_ad(&ad_a_rev, &ad_b_rev).run().unwrap();
    let binary_cotangent = AdTensor::new_primal(
        Tensor::<f64>::from_slice(&[4.0, 3.0, 2.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    );
    let binary_grads =
        crate::ops::ad::pullback_wrt(&out_binary_rev, &binary_cotangent, &[&ad_a_rev, &ad_b_rev])
            .unwrap();
    assert!(binary_grads[0].is_some());
    assert!(binary_grads[1].is_some());

    let opts = SvdOptions {
        max_rank: Some(1),
        cutoff: Some(1e-9),
    };
    let out_svd = svd_ad(&AdTensor::new_primal(a.clone()))
        .options(&opts)
        .run()
        .unwrap();
    assert_primal_mode(&out_svd.u);
    assert_primal_mode(&out_svd.s);
    assert_primal_mode(&out_svd.vt);

    let ad_multi_fwd = AdTensor::new_forward(a.clone(), da).unwrap();
    let out_qr_fwd = qr_ad(&ad_multi_fwd).run().unwrap();
    assert_forward_mode(&out_qr_fwd.q);
    assert_forward_mode(&out_qr_fwd.r);

    let out_lu_fwd = lu_ad(&ad_multi_fwd).pivot(LuPivot::NoPivot).run().unwrap();
    assert_forward_mode(&out_lu_fwd.l);
    assert_forward_mode(&out_lu_fwd.u);

    let out_eigen_fwd = eigen_ad(&ad_multi_fwd).run().unwrap();
    assert_forward_mode(&out_eigen_fwd.values);
    assert_forward_mode(&out_eigen_fwd.vectors);

    let out_slogdet_fwd = slogdet_ad(&ad_multi_fwd).run().unwrap();
    assert_forward_mode(&out_slogdet_fwd.sign);
    assert_forward_mode(&out_slogdet_fwd.logabsdet);

    let tape_multi = Tape::<StructuredTensor<f64>>::new();
    let ad_multi_rev = reverse_leaf_f64(a, &tape_multi);
    let out_svd_rev = svd_ad(&ad_multi_rev).run().unwrap();
    let cot_matrix = AdTensor::new_primal(
        Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    );
    assert!(
        crate::ops::ad::pullback_wrt(&out_svd_rev.u, &cot_matrix, &[&ad_multi_rev]).unwrap()[0]
            .is_some()
    );
    assert!(
        crate::ops::ad::pullback_wrt(&out_svd_rev.vt, &cot_matrix, &[&ad_multi_rev]).unwrap()[0]
            .is_some()
    );

    let out_qr_rev = qr_ad(&ad_multi_rev).run().unwrap();
    assert!(
        crate::ops::ad::pullback_wrt(&out_qr_rev.q, &cot_matrix, &[&ad_multi_rev]).unwrap()[0]
            .is_some()
    );
    assert!(
        crate::ops::ad::pullback_wrt(&out_qr_rev.r, &cot_matrix, &[&ad_multi_rev]).unwrap()[0]
            .is_some()
    );

    let out_lu_rev = lu_ad(&ad_multi_rev).pivot(LuPivot::Partial).run().unwrap();
    assert!(
        crate::ops::ad::pullback_wrt(&out_lu_rev.l, &cot_matrix, &[&ad_multi_rev]).unwrap()[0]
            .is_some()
    );
    assert!(
        crate::ops::ad::pullback_wrt(&out_lu_rev.u, &cot_matrix, &[&ad_multi_rev]).unwrap()[0]
            .is_some()
    );

    let out_eigen_rev = eigen_ad(&ad_multi_rev).run().unwrap();
    let cot_values = AdTensor::new_primal(
        Tensor::<f64>::from_slice(&[1.0, -1.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    );
    assert!(
        crate::ops::ad::pullback_wrt(&out_eigen_rev.values, &cot_values, &[&ad_multi_rev],)
            .unwrap()[0]
            .is_some()
    );
    assert!(
        crate::ops::ad::pullback_wrt(&out_eigen_rev.vectors, &cot_matrix, &[&ad_multi_rev],)
            .unwrap()[0]
            .is_some()
    );

    let out_slogdet_rev = slogdet_ad(&ad_multi_rev).run().unwrap();
    let cot_scalar = AdTensor::new_primal(
        Tensor::<f64>::from_slice(&[1.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    assert!(
        crate::ops::ad::pullback_wrt(&out_slogdet_rev.sign, &cot_scalar, &[&ad_multi_rev],)
            .unwrap()[0]
            .is_some()
    );
    assert!(crate::ops::ad::pullback_wrt(
        &out_slogdet_rev.logabsdet,
        &cot_scalar,
        &[&ad_multi_rev],
    )
    .unwrap()[0]
        .is_some());
}

#[test]
fn primal_linalg_builders_cover_all_ops() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = Tensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let tri = Tensor::<f64>::from_slice(&[2.0, 0.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let a_general =
        Tensor::<f64>::from_slice(&[0.0, 1.0, -1.0, 0.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let a_rect = Tensor::<f64>::from_slice(
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

    let out_svd = svd(&a).run().unwrap();
    assert_eq!(out_svd.s.dims(), &[2]);
    let out_qr = qr(&a).run().unwrap();
    assert_eq!(out_qr.q.dims(), &[2, 2]);
    let out_lu = lu(&a).pivot(LuPivot::Partial).run().unwrap();
    assert_eq!(out_lu.l.dims(), &[2, 2]);
    let out_eigen = eigen(&a).run().unwrap();
    assert_eq!(out_eigen.values.dims(), &[2]);
    let out_lstsq = lstsq(&a_ls, &b_ls).run().unwrap();
    assert_eq!(out_lstsq.x.dims(), &[2]);
    let out_cholesky = cholesky(&a).run().unwrap();
    assert_eq!(out_cholesky.dims(), &[2, 2]);
    let out_cholesky_ex = cholesky_ex(&a).run().unwrap();
    assert_eq!(out_cholesky_ex.l.dims(), &[2, 2]);
    assert_eq!(out_cholesky_ex.info, vec![0]);
    let out_solve = solve(&a, &b).run().unwrap();
    assert_eq!(out_solve.dims(), &[2]);
    let out_solve_ex = solve_ex(&a, &b).run().unwrap();
    assert_eq!(out_solve_ex.solution.dims(), &[2]);
    assert_eq!(out_solve_ex.info, vec![0]);
    let out_inv = inv(&a).run().unwrap();
    assert_eq!(out_inv.dims(), &[2, 2]);
    let out_inv_ex = inv_ex(&a).run().unwrap();
    assert_eq!(out_inv_ex.inverse.dims(), &[2, 2]);
    assert_eq!(out_inv_ex.info, vec![0]);
    let out_lu_factor = lu_factor(&a).run().unwrap();
    assert_eq!(out_lu_factor.factors.dims(), &[2, 2]);
    assert_eq!(out_lu_factor.pivots.len(), 2);
    let out_lu_factor_ex = lu_factor_ex(&a).run().unwrap();
    assert_eq!(out_lu_factor_ex.factors.dims(), &[2, 2]);
    assert_eq!(out_lu_factor_ex.pivots.len(), 2);
    assert_eq!(out_lu_factor_ex.info, vec![0]);
    let out_lu_solve = lu_solve(&out_lu_factor.factors, &b)
        .pivots(&out_lu_factor.pivots)
        .run()
        .unwrap();
    assert_eq!(out_lu_solve.dims(), &[2]);
    let out_det = det(&a).run().unwrap();
    assert_eq!(out_det.dims(), &[]);
    let out_slogdet = slogdet(&a).run().unwrap();
    assert_eq!(out_slogdet.sign.dims(), &[]);
    let out_eig = eig(&a_general).run().unwrap();
    assert_eq!(out_eig.values.dims(), &[2]);
    let out_pinv = pinv(&a_rect).rcond(1e-12).run().unwrap();
    assert_eq!(out_pinv.dims(), &[3, 2]);
    let out_exp = matrix_exp(&a).run().unwrap();
    assert_eq!(out_exp.dims(), &[2, 2]);
    let out_power = matrix_power(&a).exponent(3).run().unwrap();
    assert_eq!(out_power.dims(), &[2, 2]);
    let out_tri = solve_triangular(&tri, &b).upper(true).run().unwrap();
    assert_eq!(out_tri.dims(), &[2]);
    let out_norm = norm(&a).kind(NormKind::Fro).run().unwrap();
    assert_eq!(out_norm.dims(), &[]);
    let out_cond = cond(&a).kind(NormKind::Spectral).run().unwrap();
    assert_eq!(out_cond.dims(), &[]);
    let cross_a =
        Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let cross_b =
        Tensor::<f64>::from_slice(&[0.0, 1.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let out_cross = cross(&cross_a, &cross_b).run().unwrap();
    assert_eq!(out_cross.dims(), &[3]);
    let reflectors = Tensor::<f64>::from_slice(
        &[
            1.0, 0.0, 0.0, 0.0, //
            2.0, 1.0, 0.0, 0.0, //
            3.0, 4.0, 1.0, 0.0,
        ],
        &[4, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let tau = Tensor::<f64>::from_slice(&[0.0, 0.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let out_householder = householder_product(&reflectors, &tau).run().unwrap();
    assert_eq!(out_householder.dims(), &[4, 3]);
    let out_vander = vander(&cross_a).columns(4).increasing(true).run().unwrap();
    assert_eq!(out_vander.dims(), &[3, 4]);
    let eye4 = Tensor::<f64>::from_slice(
        &[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
        &[4, 4],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let tensorized = eye4.reshape(&[2, 2, 2, 2]).unwrap();
    let out_tensorinv = tensorinv(&tensorized).ind(2).run().unwrap();
    assert_eq!(out_tensorinv.dims(), &[2, 2, 2, 2]);
    let rhs_tensor =
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let out_tensorsolve = tensorsolve(&tensorized, &rhs_tensor)
        .dims(&[3, 2])
        .run()
        .unwrap();
    assert_eq!(out_tensorsolve.dims(), &[2, 2]);
}

#[test]
fn ad_linalg_builders_cover_all_ops_in_primal_mode() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = Tensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let tri = Tensor::<f64>::from_slice(&[2.0, 0.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let a_general =
        Tensor::<f64>::from_slice(&[0.0, 1.0, -1.0, 0.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let a_rect = Tensor::<f64>::from_slice(
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

    let ad_a = AdTensor::new_primal(a);
    let ad_b = AdTensor::new_primal(b);
    let ad_tri = AdTensor::new_primal(tri);
    let ad_general = AdTensor::new_primal(a_general);
    let ad_rect = AdTensor::new_primal(a_rect);
    let ad_ls_a = AdTensor::new_primal(a_ls);
    let ad_ls_b = AdTensor::new_primal(b_ls);

    let out_svd = svd_ad(&ad_a).run().unwrap();
    assert_primal_mode(&out_svd.u);
    assert_primal_mode(&out_svd.s);
    assert_primal_mode(&out_svd.vt);

    let out_qr = qr_ad(&ad_a).run().unwrap();
    assert_primal_mode(&out_qr.q);
    assert_primal_mode(&out_qr.r);

    let out_lu = lu_ad(&ad_a).run().unwrap();
    assert_primal_mode(&out_lu.l);
    assert_primal_mode(&out_lu.u);

    let out_eigen = eigen_ad(&ad_a).run().unwrap();
    assert_primal_mode(&out_eigen.values);
    assert_primal_mode(&out_eigen.vectors);

    let out_lstsq = lstsq_ad(&ad_ls_a, &ad_ls_b).run().unwrap();
    assert_primal_mode(&out_lstsq.x);
    assert_primal_mode(&out_lstsq.residual);

    assert_primal_mode(&cholesky_ad(&ad_a).run().unwrap());
    assert_primal_mode(&solve_ad(&ad_a, &ad_b).run().unwrap());
    assert_primal_mode(&inv_ad(&ad_a).run().unwrap());
    assert_primal_mode(&det_ad(&ad_a).run().unwrap());

    let out_slogdet = slogdet_ad(&ad_a).run().unwrap();
    assert_primal_mode(&out_slogdet.sign);
    assert_primal_mode(&out_slogdet.logabsdet);

    let out_eig = eig_ad(&ad_general).run().unwrap();
    assert_eq!(out_eig.values.mode(), AdMode::Primal);
    assert_eq!(out_eig.vectors.mode(), AdMode::Primal);

    assert_primal_mode(&pinv_ad(&ad_rect).run().unwrap());
    assert_primal_mode(&matrix_exp_ad(&ad_a).run().unwrap());
    assert_primal_mode(&solve_triangular_ad(&ad_tri, &ad_b).run().unwrap());
    assert_primal_mode(&norm_ad(&ad_a).kind(NormKind::Fro).run().unwrap());
}

#[test]
fn ad_mode_propagation_forward_and_reverse() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = Tensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let da = Tensor::<f64>::from_slice(&[0.1, 0.0, 0.0, 0.1], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();

    let ad_a_fwd = AdTensor::new_forward(a.clone(), da).unwrap();
    let ad_b = AdTensor::new_primal(b);
    let out_fwd = solve_ad(&ad_a_fwd, &ad_b).run().unwrap();
    assert_forward_mode(&out_fwd);

    let out_tri_fwd = solve_triangular_ad(&ad_a_fwd, &ad_b).run().unwrap();
    assert_forward_mode(&out_tri_fwd);

    let tape = Tape::<StructuredTensor<f64>>::new();
    let ad_a_rev = reverse_leaf_f64(a.clone(), &tape);
    let ad_b_rev = reverse_leaf_f64(a, &tape);
    let out_rev = einsum_ad("ij,jk->ik", &[&ad_a_rev, &ad_b_rev])
        .run()
        .unwrap();
    assert_reverse_on_tape(&out_rev, &tape);

    let out_tri_rev = solve_triangular_ad(&ad_a_rev, &ad_b_rev).run().unwrap();
    assert_reverse_on_tape(&out_tri_rev, &tape);
}
