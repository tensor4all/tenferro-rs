use super::*;
use tenferro_prims::{CudaContext, RocmContext};
use tenferro_tensor::MemoryOrder;

fn as_slice<T: Scalar>(t: &Tensor<T>) -> &[T] {
    t.buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
}

#[test]
fn run_requires_runtime() {
    let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let err = qr(&t).run().err();
    assert!(matches!(err, Some(Error::RuntimeNotConfigured)));
}

#[test]
fn run_with_cuda_runtime_returns_unsupported_runtime_error() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cuda(CudaContext::new()));
    let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let err = qr(&t).run().err();
    assert!(matches!(
        err,
        Some(Error::UnsupportedRuntimeOp {
            op: "qr",
            runtime: "cuda"
        })
    ));
}

#[test]
fn run_with_rocm_runtime_returns_unsupported_runtime_error() {
    let _guard = crate::set_default_runtime(RuntimeContext::Rocm(RocmContext::new()));
    let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let err = qr(&t).run().err();
    assert!(matches!(
        err,
        Some(Error::UnsupportedRuntimeOp {
            op: "qr",
            runtime: "rocm"
        })
    ));
}

#[test]
fn primal_einsum_builder_runs() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let out = einsum("ij,jk->ik", &[&a, &b]).run().unwrap();
    assert_eq!(out.dims(), &[2, 2]);
    assert_eq!(as_slice(&out).len(), 4);
}

#[test]
fn primal_qr_builder_runs() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let t = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let out = qr(&t).run().unwrap();
    assert_eq!(out.q.dims(), &[2, 2]);
    assert_eq!(out.r.dims(), &[2, 2]);
}

#[test]
fn solve_triangular_ad_supports_forward_mode() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = Tensor::<f64>::from_slice(&[2.0, 0.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let b = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let da = Tensor::<f64>::from_slice(&[0.1, 0.0, -0.2, 0.1], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    let db = Tensor::<f64>::from_slice(&[0.2, -0.1], &[2], MemoryOrder::ColumnMajor).unwrap();

    let ad_a = AdTensor::new_forward(a, da).unwrap();
    let ad_b = AdTensor::new_forward(b, db).unwrap();
    let out = solve_triangular_ad(&ad_a, &ad_b).run().unwrap();
    assert!(matches!(out.as_value(), AdValue::Forward { .. }));
    assert_eq!(out.dims(), &[2]);
}

fn assert_primal_mode(t: &AdTensor<f64>) {
    assert!(matches!(t.as_value(), AdValue::Primal(_)));
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
    let out_solve = solve(&a, &b).run().unwrap();
    assert_eq!(out_solve.dims(), &[2]);
    let out_inv = inv(&a).run().unwrap();
    assert_eq!(out_inv.dims(), &[2, 2]);
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
    let out_tri = solve_triangular(&tri, &b).upper(true).run().unwrap();
    assert_eq!(out_tri.dims(), &[2]);
    let out_norm = norm(&a).kind(NormKind::Fro).run().unwrap();
    assert_eq!(out_norm.dims(), &[]);
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
    assert!(matches!(out_eig.values.as_value(), AdValue::Primal(_)));
    assert!(matches!(out_eig.vectors.as_value(), AdValue::Primal(_)));

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
    assert!(matches!(out_fwd.as_value(), AdValue::Forward { .. }));

    let out_tri_fwd = solve_triangular_ad(&ad_a_fwd, &ad_b).run().unwrap();
    assert!(matches!(out_tri_fwd.as_value(), AdValue::Forward { .. }));

    let ad_a_rev = AdTensor::new_reverse(a.clone(), NodeId(1), TapeId(11), None).unwrap();
    let ad_b_rev = AdTensor::new_reverse(a, NodeId(2), TapeId(11), None).unwrap();
    let out_rev = einsum_ad("ij,jk->ik", &[&ad_a_rev, &ad_b_rev])
        .run()
        .unwrap();
    assert!(matches!(out_rev.as_value(), AdValue::Reverse { tape, .. } if *tape == TapeId(11)));

    let out_tri_rev = solve_triangular_ad(&ad_a_rev, &ad_b_rev).run().unwrap();
    assert!(matches!(out_tri_rev.as_value(), AdValue::Reverse { tape, .. } if *tape == TapeId(11)));
}
