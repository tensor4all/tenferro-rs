use std::collections::HashMap;

use super::*;
use crate::RuntimeContext;
use tenferro_algebra::Standard;
use tenferro_linalg::SvdOptions;
use tenferro_prims::CpuBackend;
use tenferro_prims::{CpuContext, CudaContext, RocmContext};
use tenferro_tensor::MemoryOrder;

mod organization;
mod runtime_dispatch;
mod support;

pub(crate) use support::with_cpu_runtime;

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
fn runtime_helpers_cover_mode_and_shape_paths() {
    let primal =
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let tangent =
        Tensor::<f64>::from_slice(&[0.1, 0.2, 0.3, 0.4], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();

    let ad_primal = AdTensor::new_primal(primal.clone());
    let ad_forward = AdTensor::new_forward(primal.clone(), tangent.clone()).unwrap();
    let ad_reverse = AdTensor::new_reverse(primal.clone(), NodeId(41), TapeId(91), None).unwrap();
    let ad_reverse_other =
        AdTensor::new_reverse(primal.clone(), NodeId(42), TapeId(92), None).unwrap();

    assert!(has_forward(&[&ad_primal, &ad_forward]));
    assert!(has_reverse(&[&ad_primal, &ad_reverse]));
    assert!(has_any_tangent(&[&ad_primal, &ad_forward]));
    assert_eq!(
        derive_reverse_tape(&[&ad_primal, &ad_reverse]).unwrap(),
        Some(TapeId(91))
    );
    assert!(matches!(
        derive_reverse_tape(&[&ad_reverse, &ad_reverse_other]),
        Err(Error::MixedReverseTape {
            expected: 91,
            found: 92
        })
    ));

    let node_a = derive_reverse_node("helper", &[&ad_primal, &ad_reverse], &[2, 2], 7, TapeId(91));
    let node_b = derive_reverse_node("helper", &[&ad_primal, &ad_reverse], &[2, 2], 8, TapeId(91));
    assert_ne!(node_a, node_b);

    let specs = collect_reverse_input_specs(&[&ad_reverse, &ad_primal]);
    assert_eq!(specs.len(), 2);
    assert_eq!(specs[0].as_ref().unwrap().node, NodeId(41));
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
        wrap_dense_ad_output("runtime_helper", &[&ad_primal], primal.clone(), None, 0).unwrap();
    assert!(matches!(wrapped_primal.as_value(), AdValue::Primal(_)));

    let wrapped_forward = wrap_dense_ad_output(
        "runtime_helper",
        &[&ad_forward],
        primal.clone(),
        Some(tangent.clone()),
        0,
    )
    .unwrap();
    assert!(matches!(
        wrapped_forward.as_value(),
        AdValue::Forward { .. }
    ));

    let wrapped_reverse =
        wrap_dense_ad_output("runtime_helper", &[&ad_reverse], primal.clone(), None, 0).unwrap();
    assert!(matches!(
        wrapped_reverse.as_value(),
        AdValue::Reverse { node, tape, .. } if *node == derive_reverse_node("runtime_helper", &[&ad_reverse], &[2, 2], 0, TapeId(91))
            && *tape == TapeId(91)
    ));

    let forward_err =
        wrap_dense_ad_output("runtime_helper", &[&ad_forward], primal.clone(), None, 0)
            .unwrap_err();
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

    let summed = super::scalar_runtime::scalar_full_reduction_primal(
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
    let ad_x = AdTensor::new_reverse(x.clone(), NodeId(51), TapeId(151), None).unwrap();
    let out = sum_ad(&ad_x).run().unwrap();
    assert!(matches!(out.as_value(), AdValue::Reverse { .. }));

    let cotangent = AdTensor::new_primal(
        Tensor::<f64>::from_slice(&[3.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    let grads = crate::api::ad::pullback_wrt(&out, &cotangent, &[&ad_x]).unwrap();
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
    let ad_a = AdTensor::new_reverse(a.clone(), NodeId(61), TapeId(161), None).unwrap();
    let ad_b = AdTensor::new_reverse(b.clone(), NodeId(62), TapeId(161), None).unwrap();
    let size_dict = HashMap::new();

    let out = einsum_ad("ij,jk->ik", &[&ad_a, &ad_b])
        .size_dict(&size_dict)
        .run()
        .unwrap();
    assert!(matches!(out.as_value(), AdValue::Reverse { .. }));

    let cotangent =
        Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
    let grads = crate::api::ad::pullback_wrt(
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
    assert!(matches!(out_unary_fwd.as_value(), AdValue::Forward { .. }));

    let ad_a_rev = AdTensor::new_reverse(a.clone(), NodeId(71), TapeId(171), None).unwrap();
    let out_unary_rev = cholesky_ad(&ad_a_rev).run().unwrap();
    let unary_cotangent = AdTensor::new_primal(
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    );
    let unary_grads =
        crate::api::ad::pullback_wrt(&out_unary_rev, &unary_cotangent, &[&ad_a_rev]).unwrap();
    assert!(unary_grads[0].is_some());
    assert_eq!(unary_grads[0].as_ref().unwrap().logical_dims(), &[2, 2]);

    let ad_b_fwd = AdTensor::new_forward(b.clone(), db.clone()).unwrap();
    let out_binary_fwd = solve_ad(&ad_a_fwd, &ad_b_fwd).run().unwrap();
    assert!(matches!(out_binary_fwd.as_value(), AdValue::Forward { .. }));

    let ad_b_rev = AdTensor::new_reverse(b.clone(), NodeId(72), TapeId(171), None).unwrap();
    let out_binary_rev = solve_ad(&ad_a_rev, &ad_b_rev).run().unwrap();
    let binary_cotangent = AdTensor::new_primal(
        Tensor::<f64>::from_slice(&[4.0, 3.0, 2.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    );
    let binary_grads =
        crate::api::ad::pullback_wrt(&out_binary_rev, &binary_cotangent, &[&ad_a_rev, &ad_b_rev])
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
    assert!(matches!(out_qr_fwd.q.as_value(), AdValue::Forward { .. }));
    assert!(matches!(out_qr_fwd.r.as_value(), AdValue::Forward { .. }));

    let out_lu_fwd = lu_ad(&ad_multi_fwd).pivot(LuPivot::NoPivot).run().unwrap();
    assert!(matches!(out_lu_fwd.l.as_value(), AdValue::Forward { .. }));
    assert!(matches!(out_lu_fwd.u.as_value(), AdValue::Forward { .. }));

    let out_eigen_fwd = eigen_ad(&ad_multi_fwd).run().unwrap();
    assert!(matches!(
        out_eigen_fwd.values.as_value(),
        AdValue::Forward { .. }
    ));
    assert!(matches!(
        out_eigen_fwd.vectors.as_value(),
        AdValue::Forward { .. }
    ));

    let out_slogdet_fwd = slogdet_ad(&ad_multi_fwd).run().unwrap();
    assert!(matches!(
        out_slogdet_fwd.sign.as_value(),
        AdValue::Forward { .. }
    ));
    assert!(matches!(
        out_slogdet_fwd.logabsdet.as_value(),
        AdValue::Forward { .. }
    ));

    let ad_multi_rev = AdTensor::new_reverse(a, NodeId(73), TapeId(173), None).unwrap();
    let out_svd_rev = svd_ad(&ad_multi_rev).run().unwrap();
    let cot_matrix = AdTensor::new_primal(
        Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    );
    assert!(
        crate::api::ad::pullback_wrt(&out_svd_rev.u, &cot_matrix, &[&ad_multi_rev]).unwrap()[0]
            .is_some()
    );
    assert!(
        crate::api::ad::pullback_wrt(&out_svd_rev.vt, &cot_matrix, &[&ad_multi_rev]).unwrap()[0]
            .is_some()
    );

    let out_qr_rev = qr_ad(&ad_multi_rev).run().unwrap();
    assert!(
        crate::api::ad::pullback_wrt(&out_qr_rev.q, &cot_matrix, &[&ad_multi_rev]).unwrap()[0]
            .is_some()
    );
    assert!(
        crate::api::ad::pullback_wrt(&out_qr_rev.r, &cot_matrix, &[&ad_multi_rev]).unwrap()[0]
            .is_some()
    );

    let out_lu_rev = lu_ad(&ad_multi_rev).pivot(LuPivot::Partial).run().unwrap();
    assert!(
        crate::api::ad::pullback_wrt(&out_lu_rev.l, &cot_matrix, &[&ad_multi_rev]).unwrap()[0]
            .is_some()
    );
    assert!(
        crate::api::ad::pullback_wrt(&out_lu_rev.u, &cot_matrix, &[&ad_multi_rev]).unwrap()[0]
            .is_some()
    );

    let out_eigen_rev = eigen_ad(&ad_multi_rev).run().unwrap();
    let cot_values = AdTensor::new_primal(
        Tensor::<f64>::from_slice(&[1.0, -1.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    );
    assert!(
        crate::api::ad::pullback_wrt(&out_eigen_rev.values, &cot_values, &[&ad_multi_rev],)
            .unwrap()[0]
            .is_some()
    );
    assert!(
        crate::api::ad::pullback_wrt(&out_eigen_rev.vectors, &cot_matrix, &[&ad_multi_rev],)
            .unwrap()[0]
            .is_some()
    );

    let out_slogdet_rev = slogdet_ad(&ad_multi_rev).run().unwrap();
    let cot_scalar = AdTensor::new_primal(
        Tensor::<f64>::from_slice(&[1.0], &[], MemoryOrder::ColumnMajor).unwrap(),
    );
    assert!(
        crate::api::ad::pullback_wrt(&out_slogdet_rev.sign, &cot_scalar, &[&ad_multi_rev],)
            .unwrap()[0]
            .is_some()
    );
    assert!(crate::api::ad::pullback_wrt(
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
