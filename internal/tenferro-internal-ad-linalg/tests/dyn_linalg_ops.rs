use tenferro_internal_ad_core::{new_reverse_leaf, LinearizableOp, LinearizedOp};
use tenferro_internal_frontend_core::{DynTensor, StructuredTensor};
use tenferro_internal_runtime::{set_default_runtime, RuntimeContext};
use tenferro_linalg::NormKind;
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

use tenferro_internal_ad_linalg::{
    det_dyn_value, norm_dyn_value, qr_dyn_value, solve_dyn_values, svd_dyn_value, DetOp, QrOp,
    SolveOp, SvdOp,
};

fn dyn_f64(values: &[f64], dims: &[usize]) -> DynTensor {
    let dense = Tensor::<f64>::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap();
    DynTensor::F64(StructuredTensor::from(dense))
}

fn dyn_scalar(value: f64) -> DynTensor {
    dyn_f64(&[value], &[])
}

fn f64_values(tensor: &DynTensor) -> Vec<f64> {
    match tensor {
        DynTensor::F64(value) => value
            .to_dense()
            .unwrap()
            .buffer()
            .as_slice()
            .unwrap()
            .to_vec(),
        other => panic!("expected f64 dyn tensor, got {other:?}"),
    }
}

fn max_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn assert_vec_close(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert!(
        max_abs_diff(lhs, rhs) <= tol,
        "left: {lhs:?}\nright: {rhs:?}\nmax_abs_diff: {}",
        max_abs_diff(lhs, rhs)
    );
}

#[test]
fn solve_det_and_norm_dyn_values_use_linearized_runtime() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = new_reverse_leaf(dyn_f64(&[2.0, 0.0, 0.0, 3.0], &[2, 2]));
    let b = new_reverse_leaf(dyn_f64(&[4.0, 9.0], &[2]));

    let solved = solve_dyn_values(&a, &b).unwrap();
    assert_eq!(f64_values(solved.primal()), vec![2.0, 3.0]);

    let det = det_dyn_value(&a).unwrap();
    let det_grad = det.grad_wrt_with_seed(dyn_scalar(1.0), &[&a]).unwrap();
    assert_eq!(
        f64_values(&det_grad[0].clone().unwrap()),
        vec![3.0, 0.0, 0.0, 2.0]
    );

    let norm = norm_dyn_value(&b, NormKind::Fro).unwrap();
    let norm_grad = norm.grad_wrt_with_seed(dyn_scalar(1.0), &[&b]).unwrap();
    let grad = f64_values(&norm_grad[0].clone().unwrap());
    assert!((grad[0] - 4.0 / 97.0_f64.sqrt()).abs() < 1e-12);
    assert!((grad[1] - 9.0 / 97.0_f64.sqrt()).abs() < 1e-12);

    let solve_op = SolveOp;
    let outputs = solve_op.primal(&[a.primal(), b.primal()]).unwrap();
    let linearized = solve_op
        .linearize(&[a.primal(), b.primal()], &outputs)
        .unwrap();
    let jvp = linearized
        .jvp(&[None, Some(dyn_f64(&[1.0, 1.0], &[2]))])
        .unwrap();
    assert_eq!(f64_values(&jvp[0].clone().unwrap()), vec![0.5, 1.0 / 3.0]);

    let det_op = DetOp;
    let det_outputs = det_op.primal(&[a.primal()]).unwrap();
    let det_linearized = det_op.linearize(&[a.primal()], &det_outputs).unwrap();
    let det_jvp = det_linearized
        .jvp(&[Some(dyn_f64(&[1.0, 0.0, 0.0, 0.0], &[2, 2]))])
        .unwrap();
    assert_eq!(f64_values(&det_jvp[0].clone().unwrap()), vec![3.0]);
}

#[test]
fn dyn_linalg_ops_qr_linearized_jvp_preserves_packaging_and_optional_tangent_behavior() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = new_reverse_leaf(dyn_f64(&[3.0, 0.0, 0.0, 4.0], &[2, 2]));

    let qr = qr_dyn_value(&a).unwrap();
    assert_eq!(qr.q.primal().dims(), &[2, 2]);
    assert_eq!(qr.r.primal().dims(), &[2, 2]);
    let q_values = f64_values(qr.q.primal());
    let r_values = f64_values(qr.r.primal());
    assert_vec_close(&q_values, &[1.0, 0.0, 0.0, 1.0], 1.0e-12);
    assert_vec_close(&r_values, &[3.0, 0.0, 0.0, 4.0], 1.0e-12);
    assert!(max_abs_diff(&q_values, &r_values) > 1.0e-3);

    let qr_op = QrOp;
    let qr_outputs = qr_op.primal(&[a.primal()]).unwrap();
    let qr_linearized = qr_op.linearize(&[a.primal()], &qr_outputs).unwrap();

    let qr_none = qr_linearized.jvp(&[None]).unwrap();
    assert!(qr_none.iter().all(Option::is_none));

    let qr_jvp = qr_linearized
        .jvp(&[Some(dyn_f64(&[1.0, 0.0, 0.0, 0.0], &[2, 2]))])
        .unwrap();
    assert_eq!(qr_jvp.len(), 2);
    assert_eq!(qr_jvp[0].as_ref().unwrap().dims(), &[2, 2]);
    assert_eq!(qr_jvp[1].as_ref().unwrap().dims(), &[2, 2]);

    let epsilon = 1.0e-6;
    let tangent = [1.0, 0.0, 0.0, 0.0];
    let perturbed_plus = new_reverse_leaf(dyn_f64(
        &[
            3.0 + epsilon * tangent[0],
            0.0 + epsilon * tangent[1],
            0.0 + epsilon * tangent[2],
            4.0 + epsilon * tangent[3],
        ],
        &[2, 2],
    ));
    let perturbed_minus = new_reverse_leaf(dyn_f64(
        &[
            3.0 - epsilon * tangent[0],
            0.0 - epsilon * tangent[1],
            0.0 - epsilon * tangent[2],
            4.0 - epsilon * tangent[3],
        ],
        &[2, 2],
    ));
    let qr_plus = qr_dyn_value(&perturbed_plus).unwrap();
    let qr_minus = qr_dyn_value(&perturbed_minus).unwrap();
    let expected_q = f64_values(qr_plus.q.primal())
        .iter()
        .zip(f64_values(qr_minus.q.primal()))
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    let expected_r = f64_values(qr_plus.r.primal())
        .iter()
        .zip(f64_values(qr_minus.r.primal()))
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    assert_vec_close(
        &f64_values(qr_jvp[0].as_ref().unwrap()),
        &expected_q,
        5.0e-4,
    );
    assert_vec_close(
        &f64_values(qr_jvp[1].as_ref().unwrap()),
        &expected_r,
        5.0e-4,
    );
}

#[test]
fn dyn_linalg_ops_svd_linearized_jvp_preserves_packaging_and_optional_tangent_behavior() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = new_reverse_leaf(dyn_f64(&[3.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[3, 2]));

    let svd = svd_dyn_value(&a, None).unwrap();
    assert_eq!(svd.u.primal().dims(), &[3, 2]);
    assert_eq!(svd.s.primal().dims(), &[2]);
    assert_eq!(svd.vt.primal().dims(), &[2, 2]);
    assert_vec_close(&f64_values(svd.s.primal()), &[3.0, 1.0], 1.0e-12);
    assert_vec_close(
        &f64_values(svd.u.primal()),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        1.0e-12,
    );
    assert_vec_close(&f64_values(svd.vt.primal()), &[1.0, 0.0, 0.0, 1.0], 1.0e-12);

    let svd_op = SvdOp::default();
    let svd_outputs = svd_op.primal(&[a.primal()]).unwrap();
    let svd_linearized = svd_op.linearize(&[a.primal()], &svd_outputs).unwrap();
    let svd_none = svd_linearized.jvp(&[None]).unwrap();
    assert!(svd_none.iter().all(Option::is_none));

    let svd_jvp = svd_linearized
        .jvp(&[Some(dyn_f64(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[3, 2]))])
        .unwrap();
    assert_eq!(svd_jvp.len(), 3);
    assert_eq!(svd_jvp[0].as_ref().unwrap().dims(), &[3, 2]);
    assert_eq!(svd_jvp[1].as_ref().unwrap().dims(), &[2]);
    assert_eq!(svd_jvp[2].as_ref().unwrap().dims(), &[2, 2]);

    let epsilon = 1.0e-6;
    let tangent = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let perturbed_plus = new_reverse_leaf(dyn_f64(
        &[
            3.0 + epsilon * tangent[0],
            0.0 + epsilon * tangent[1],
            0.0 + epsilon * tangent[2],
            0.0 + epsilon * tangent[3],
            0.0 + epsilon * tangent[4],
            0.0 + epsilon * tangent[5],
        ],
        &[3, 2],
    ));
    let perturbed_minus = new_reverse_leaf(dyn_f64(
        &[
            3.0 - epsilon * tangent[0],
            0.0 - epsilon * tangent[1],
            0.0 - epsilon * tangent[2],
            0.0 - epsilon * tangent[3],
            0.0 - epsilon * tangent[4],
            0.0 - epsilon * tangent[5],
        ],
        &[3, 2],
    ));
    let svd_plus = svd_dyn_value(&perturbed_plus, None).unwrap();
    let svd_minus = svd_dyn_value(&perturbed_minus, None).unwrap();
    let expected_u = f64_values(svd_plus.u.primal())
        .iter()
        .zip(f64_values(svd_minus.u.primal()))
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    let expected_s = f64_values(svd_plus.s.primal())
        .iter()
        .zip(f64_values(svd_minus.s.primal()))
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    let expected_vt = f64_values(svd_plus.vt.primal())
        .iter()
        .zip(f64_values(svd_minus.vt.primal()))
        .map(|(plus, minus)| (plus - minus) / (2.0 * epsilon))
        .collect::<Vec<_>>();
    assert_vec_close(
        &f64_values(svd_jvp[0].as_ref().unwrap()),
        &expected_u,
        5.0e-4,
    );
    assert_vec_close(
        &f64_values(svd_jvp[1].as_ref().unwrap()),
        &expected_s,
        5.0e-4,
    );
    assert_vec_close(
        &f64_values(svd_jvp[2].as_ref().unwrap()),
        &expected_vt,
        5.0e-4,
    );
}
