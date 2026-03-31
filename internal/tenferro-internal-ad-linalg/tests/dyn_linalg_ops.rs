use tenferro_internal_ad_core::{new_reverse_leaf, LinearizableOp, LinearizedOp};
use tenferro_internal_frontend_core::{DynTensor, StructuredTensor};
use tenferro_internal_runtime::{set_default_runtime, RuntimeContext};
use tenferro_linalg::NormKind;
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

use tenferro_internal_ad_linalg::{
    det_dyn_value, norm_dyn_value, qr_dyn_value, solve_dyn_values, svd_dyn_value, DetOp, NormOp,
    QrOp, SolveOp, SvdOp,
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

fn assert_all_none(values: &[Option<DynTensor>]) {
    assert!(values.iter().all(Option::is_none));
}

fn matmul_col_major(
    lhs: &[f64],
    lhs_rows: usize,
    lhs_cols: usize,
    rhs: &[f64],
    rhs_rows: usize,
    rhs_cols: usize,
) -> Vec<f64> {
    assert_eq!(lhs.len(), lhs_rows * lhs_cols);
    assert_eq!(rhs.len(), rhs_rows * rhs_cols);
    assert_eq!(lhs_cols, rhs_rows);

    let mut out = vec![0.0; lhs_rows * rhs_cols];
    for col in 0..rhs_cols {
        for row in 0..lhs_rows {
            let mut acc = 0.0;
            for k in 0..lhs_cols {
                let lhs_idx = row + k * lhs_rows;
                let rhs_idx = k + col * rhs_rows;
                acc += lhs[lhs_idx] * rhs[rhs_idx];
            }
            out[row + col * lhs_rows] = acc;
        }
    }
    out
}

fn scale_cols_col_major(values: &[f64], rows: usize, cols: usize, scales: &[f64]) -> Vec<f64> {
    assert_eq!(values.len(), rows * cols);
    assert_eq!(cols, scales.len());
    let mut out = values.to_vec();
    for col in 0..cols {
        for row in 0..rows {
            out[row + col * rows] *= scales[col];
        }
    }
    out
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
    let solve_none = linearized.jvp(&[None, None]).unwrap();
    assert_all_none(&solve_none);
    let jvp = linearized
        .jvp(&[None, Some(dyn_f64(&[1.0, 1.0], &[2]))])
        .unwrap();
    assert_eq!(f64_values(&jvp[0].clone().unwrap()), vec![0.5, 1.0 / 3.0]);

    let det_op = DetOp;
    let det_outputs = det_op.primal(&[a.primal()]).unwrap();
    let det_linearized = det_op.linearize(&[a.primal()], &det_outputs).unwrap();
    let det_none = det_linearized.jvp(&[None]).unwrap();
    assert_all_none(&det_none);
    let det_jvp = det_linearized
        .jvp(&[Some(dyn_f64(&[1.0, 0.0, 0.0, 0.0], &[2, 2]))])
        .unwrap();
    assert_eq!(f64_values(&det_jvp[0].clone().unwrap()), vec![3.0]);

    let norm_op = NormOp::new(NormKind::Fro);
    let norm_outputs = norm_op.primal(&[b.primal()]).unwrap();
    let norm_linearized = norm_op.linearize(&[b.primal()], &norm_outputs).unwrap();
    let norm_none = norm_linearized.jvp(&[None]).unwrap();
    assert_all_none(&norm_none);
}

#[test]
fn dyn_linalg_ops_qr_linearized_jvp_preserves_packaging_and_optional_tangent_behavior() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = new_reverse_leaf(dyn_f64(&[2.0, 1.0, 0.5, 3.0, 4.0, 5.0], &[3, 2]));

    let qr = qr_dyn_value(&a).unwrap();
    assert_eq!(qr.q.primal().dims(), &[3, 2]);
    assert_eq!(qr.r.primal().dims(), &[2, 2]);

    let qr_op = QrOp;
    let qr_outputs = qr_op.primal(&[a.primal()]).unwrap();
    let qr_linearized = qr_op.linearize(&[a.primal()], &qr_outputs).unwrap();

    let qr_none = qr_linearized.jvp(&[None]).unwrap();
    assert!(qr_none.iter().all(Option::is_none));

    let qr_jvp = qr_linearized
        .jvp(&[Some(dyn_f64(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[3, 2]))])
        .unwrap();
    assert_eq!(qr_jvp.len(), 2);
    assert_eq!(qr_jvp[0].as_ref().unwrap().dims(), &[3, 2]);
    assert_eq!(qr_jvp[1].as_ref().unwrap().dims(), &[2, 2]);

    let q = f64_values(&qr_outputs[0]);
    let r = f64_values(&qr_outputs[1]);
    let dq = f64_values(qr_jvp[0].as_ref().unwrap());
    let dr = f64_values(qr_jvp[1].as_ref().unwrap());
    let reconstructed = {
        let dq_r = matmul_col_major(&dq, 3, 2, &r, 2, 2);
        let q_dr = matmul_col_major(&q, 3, 2, &dr, 2, 2);
        dq_r.iter()
            .zip(q_dr.iter())
            .map(|(lhs, rhs)| lhs + rhs)
            .collect::<Vec<_>>()
    };
    assert_vec_close(&reconstructed, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5.0e-4);
}

#[test]
fn dyn_linalg_ops_svd_linearized_jvp_preserves_packaging_and_optional_tangent_behavior() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = new_reverse_leaf(dyn_f64(&[2.0, 1.0, 0.5, 3.0, 4.0, 5.0], &[3, 2]));

    let svd = svd_dyn_value(&a, None).unwrap();
    assert_eq!(svd.u.primal().dims(), &[3, 2]);
    assert_eq!(svd.s.primal().dims(), &[2]);
    assert_eq!(svd.vt.primal().dims(), &[2, 2]);

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

    let u = f64_values(&svd_outputs[0]);
    let s = f64_values(&svd_outputs[1]);
    let vt = f64_values(&svd_outputs[2]);
    let du = f64_values(svd_jvp[0].as_ref().unwrap());
    let ds = f64_values(svd_jvp[1].as_ref().unwrap());
    let dvt = f64_values(svd_jvp[2].as_ref().unwrap());
    let reconstructed = {
        let du_s = scale_cols_col_major(&du, 3, 2, &s);
        let u_ds = scale_cols_col_major(&u, 3, 2, &ds);
        let u_s = scale_cols_col_major(&u, 3, 2, &s);
        let term1 = matmul_col_major(&du_s, 3, 2, &vt, 2, 2);
        let term2 = matmul_col_major(&u_ds, 3, 2, &vt, 2, 2);
        let term3 = matmul_col_major(&u_s, 3, 2, &dvt, 2, 2);
        term1
            .iter()
            .zip(term2.iter())
            .zip(term3.iter())
            .map(|((lhs, mid), rhs)| lhs + mid + rhs)
            .collect::<Vec<_>>()
    };
    assert_vec_close(&reconstructed, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5.0e-4);
}
