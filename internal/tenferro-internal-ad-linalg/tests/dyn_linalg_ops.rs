use num_complex::Complex64;
use tenferro_internal_ad_core::{new_reverse_leaf, LinearizableOp, LinearizedOp};
use tenferro_internal_frontend_core::{DynTensor, StructuredTensor};
use tenferro_internal_runtime::{set_default_runtime, RuntimeContext};
use tenferro_linalg::{LuPivot, NormKind};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

use tenferro_internal_ad_linalg::{
    cholesky_dyn_value, det_dyn_value, eig_dyn_value, eigen_dyn_value, inv_dyn_value,
    lstsq_dyn_values, lu_dyn_value, matrix_exp_dyn_value, norm_dyn_value, pinv_dyn_value,
    qr_dyn_value, slogdet_dyn_value, solve_dyn_values, svd_dyn_value, CholeskyOp, DetOp, EigOp,
    EigenOp, InvOp, LstsqOp, LuOp, MatrixExpOp, NormOp, PInvOp, QrOp, SlogdetOp, SolveOp, SvdOp,
};

fn dyn_f64(values: &[f64], dims: &[usize]) -> DynTensor {
    let dense = Tensor::<f64>::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap();
    DynTensor::F64(StructuredTensor::from(dense))
}

fn dyn_scalar(value: f64) -> DynTensor {
    dyn_f64(&[value], &[])
}

fn dyn_c64(values: &[Complex64], dims: &[usize]) -> DynTensor {
    let dense = Tensor::<Complex64>::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap();
    DynTensor::C64(StructuredTensor::from(dense))
}

fn f64_values(tensor: &DynTensor) -> Vec<f64> {
    match tensor {
        DynTensor::F64(value) => value.to_dense().unwrap().to_vec(),
        other => panic!("expected f64 dyn tensor, got {other:?}"),
    }
}

fn c64_values(tensor: &DynTensor) -> Vec<Complex64> {
    match tensor {
        DynTensor::C64(value) => value.to_dense().unwrap().to_vec(),
        other => panic!("expected c64 dyn tensor, got {other:?}"),
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

fn assert_complex_vec_close(lhs: &[Complex64], rhs: &[Complex64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        let diff = (*lhs - *rhs).norm();
        assert!(
            diff <= tol,
            "left: {lhs:?}\nright: {rhs:?}\nabs_diff: {diff}"
        );
    }
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

#[test]
fn dyn_linalg_ops_lstsq_and_lu_linearized_jvp_preserve_packaging_and_auxiliary_outputs() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = new_reverse_leaf(dyn_f64(&[2.0, 0.0, 0.0, 3.0], &[2, 2]));
    let b = new_reverse_leaf(dyn_f64(&[4.0, 9.0], &[2]));

    let lstsq = lstsq_dyn_values(&a, &b).unwrap();
    assert_eq!(lstsq.solution.primal().dims(), &[2]);
    assert_eq!(lstsq.residuals.primal().dims(), &[2]);
    assert_eq!(lstsq.rank, vec![2]);
    assert_eq!(f64_values(&lstsq.singular_values), vec![3.0, 2.0]);

    let lstsq_op = LstsqOp;
    let lstsq_outputs = lstsq_op.primal(&[a.primal(), b.primal()]).unwrap();
    let lstsq_linearized = lstsq_op
        .linearize(&[a.primal(), b.primal()], &lstsq_outputs)
        .unwrap();
    assert_all_none(&lstsq_linearized.jvp(&[None, None]).unwrap());

    let lstsq_jvp = lstsq_linearized
        .jvp(&[None, Some(dyn_f64(&[1.0, 1.0], &[2]))])
        .unwrap();
    assert_eq!(lstsq_jvp.len(), 2);
    assert_eq!(lstsq_jvp[0].as_ref().unwrap().dims(), &[2]);
    assert_eq!(lstsq_jvp[1].as_ref().unwrap().dims(), &[2]);
    assert_eq!(f64_values(lstsq_jvp[1].as_ref().unwrap()), vec![0.0, 0.0]);

    let lu = lu_dyn_value(&a, LuPivot::Partial).unwrap();
    assert_eq!(lu.p.primal().dims(), &[2, 2]);
    assert_eq!(lu.l.primal().dims(), &[2, 2]);
    assert_eq!(lu.u.primal().dims(), &[2, 2]);

    let lu_op = LuOp::new(LuPivot::Partial);
    let lu_outputs = lu_op.primal(&[a.primal()]).unwrap();
    let lu_linearized = lu_op.linearize(&[a.primal()], &lu_outputs).unwrap();
    assert_all_none(&lu_linearized.jvp(&[None]).unwrap());

    let lu_jvp = lu_linearized
        .jvp(&[Some(dyn_f64(&[1.0, 0.0, 0.0, 0.0], &[2, 2]))])
        .unwrap();
    assert_eq!(lu_jvp.len(), 3);
    assert!(lu_jvp[0].is_none());
    assert_eq!(lu_jvp[1].as_ref().unwrap().dims(), &[2, 2]);
    assert_eq!(lu_jvp[2].as_ref().unwrap().dims(), &[2, 2]);
}

#[test]
fn dyn_linalg_ops_eig_and_eigen_linearized_jvp_preserve_complex_and_real_packaging() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = new_reverse_leaf(dyn_f64(&[2.0, 0.0, 0.0, 3.0], &[2, 2]));

    let eig = eig_dyn_value(&a).unwrap();
    assert_eq!(eig.values.primal().dims(), &[2]);
    assert_eq!(eig.vectors.primal().dims(), &[2, 2]);

    let eig_op = EigOp;
    let eig_outputs = eig_op.primal(&[a.primal()]).unwrap();
    let eig_linearized = eig_op.linearize(&[a.primal()], &eig_outputs).unwrap();
    assert_all_none(&eig_linearized.jvp(&[None]).unwrap());

    let eig_jvp = eig_linearized
        .jvp(&[Some(dyn_f64(&[1.0, 0.0, 0.0, 0.0], &[2, 2]))])
        .unwrap();
    assert_eq!(eig_jvp.len(), 2);
    assert_eq!(eig_jvp[0].as_ref().unwrap().dims(), &[2]);
    assert_eq!(eig_jvp[1].as_ref().unwrap().dims(), &[2, 2]);
    assert_eq!(
        c64_values(eig_jvp[0].as_ref().unwrap()),
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    );

    let eigen = eigen_dyn_value(&a).unwrap();
    assert_eq!(eigen.values.primal().dims(), &[2]);
    assert_eq!(eigen.vectors.primal().dims(), &[2, 2]);

    let eigen_op = EigenOp;
    let eigen_outputs = eigen_op.primal(&[a.primal()]).unwrap();
    let eigen_linearized = eigen_op.linearize(&[a.primal()], &eigen_outputs).unwrap();
    assert_all_none(&eigen_linearized.jvp(&[None]).unwrap());

    let eigen_jvp = eigen_linearized
        .jvp(&[Some(dyn_f64(&[1.0, 0.0, 0.0, 0.0], &[2, 2]))])
        .unwrap();
    assert_eq!(eigen_jvp.len(), 2);
    assert_eq!(eigen_jvp[0].as_ref().unwrap().dims(), &[2]);
    assert_eq!(eigen_jvp[1].as_ref().unwrap().dims(), &[2, 2]);
    assert_eq!(f64_values(eigen_jvp[0].as_ref().unwrap()), vec![1.0, 0.0]);
}

#[test]
fn dyn_linalg_ops_complex_eigen_linearized_jvp_and_vjp_support() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = new_reverse_leaf(dyn_c64(
        &[
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(3.0, 0.0),
        ],
        &[2, 2],
    ));

    let eigen = eigen_dyn_value(&a).unwrap();
    assert_eq!(eigen.values.primal().dims(), &[2]);
    assert_eq!(eigen.vectors.primal().dims(), &[2, 2]);

    let eigen_op = EigenOp;
    let eigen_outputs = eigen_op.primal(&[a.primal()]).unwrap();
    let eigen_linearized = eigen_op.linearize(&[a.primal()], &eigen_outputs).unwrap();
    assert_all_none(&eigen_linearized.jvp(&[None]).unwrap());

    let eigen_jvp = eigen_linearized
        .jvp(&[Some(dyn_c64(
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            &[2, 2],
        ))])
        .unwrap();
    assert_eq!(eigen_jvp.len(), 2);
    assert_eq!(eigen_jvp[0].as_ref().unwrap().dims(), &[2]);
    assert_eq!(eigen_jvp[1].as_ref().unwrap().dims(), &[2, 2]);
    assert_eq!(f64_values(eigen_jvp[0].as_ref().unwrap()), vec![1.0, 0.0]);

    let cotangent_values = dyn_f64(&[1.0, 1.0], &[2]);
    let grad = eigen
        .values
        .grad_wrt_with_seed(cotangent_values, &[&a])
        .unwrap();
    assert_eq!(
        c64_values(grad[0].as_ref().unwrap()),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]
    );
}

#[test]
fn dyn_linalg_ops_batch_a_complex_jvp_and_vjp_support() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let z1 = Complex64::new(1.0, 1.0);
    let z2 = Complex64::new(2.0, -1.0);
    let a = new_reverse_leaf(dyn_c64(
        &[z1, Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), z2],
        &[2, 2],
    ));
    let da = dyn_c64(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[2, 2],
    );
    let e11 = dyn_c64(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[2, 2],
    );

    let inv = inv_dyn_value(&a).unwrap();
    let inv_op = InvOp;
    let inv_outputs = inv_op.primal(&[a.primal()]).unwrap();
    assert_eq!(inv.primal().dims(), &[2, 2]);
    let inv_linearized = inv_op.linearize(&[a.primal()], &inv_outputs).unwrap();
    let inv_jvp = inv_linearized.jvp(&[Some(da.clone())]).unwrap();
    assert_complex_vec_close(
        &c64_values(inv_jvp[0].as_ref().unwrap()),
        &[
            Complex64::new(0.0, 0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        1.0e-12,
    );
    let inv_vjp = inv_linearized.vjp(&[Some(e11.clone())], &[true]).unwrap();
    assert_complex_vec_close(
        &c64_values(inv_vjp[0].as_ref().unwrap()),
        &[
            Complex64::new(0.0, -0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        1.0e-12,
    );

    let hermitian = new_reverse_leaf(dyn_c64(
        &[
            Complex64::new(4.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(9.0, 0.0),
        ],
        &[2, 2],
    ));
    let dhermitian = dyn_c64(
        &[
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[2, 2],
    );
    let cholesky = cholesky_dyn_value(&hermitian).unwrap();
    assert_eq!(cholesky.primal().dims(), &[2, 2]);
    let cholesky_op = CholeskyOp;
    let cholesky_outputs = cholesky_op.primal(&[hermitian.primal()]).unwrap();
    let cholesky_linearized = cholesky_op
        .linearize(&[hermitian.primal()], &cholesky_outputs)
        .unwrap();
    let cholesky_jvp = cholesky_linearized
        .jvp(&[Some(dhermitian.clone())])
        .unwrap();
    assert_complex_vec_close(
        &c64_values(cholesky_jvp[0].as_ref().unwrap()),
        &[
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        1.0e-12,
    );
    let cholesky_vjp = cholesky_linearized
        .vjp(&[Some(e11.clone())], &[true])
        .unwrap();
    assert_complex_vec_close(
        &c64_values(cholesky_vjp[0].as_ref().unwrap()),
        &[
            Complex64::new(0.25, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        1.0e-12,
    );

    let pinv = pinv_dyn_value(&a, None).unwrap();
    assert_eq!(pinv.primal().dims(), &[2, 2]);
    let pinv_op = PInvOp::new(None);
    let pinv_outputs = pinv_op.primal(&[a.primal()]).unwrap();
    let pinv_linearized = pinv_op.linearize(&[a.primal()], &pinv_outputs).unwrap();
    let pinv_jvp = pinv_linearized.jvp(&[Some(da.clone())]).unwrap();
    assert_complex_vec_close(
        &c64_values(pinv_jvp[0].as_ref().unwrap()),
        &[
            Complex64::new(0.0, 0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        1.0e-12,
    );
    let pinv_vjp = pinv_linearized.vjp(&[Some(e11.clone())], &[true]).unwrap();
    assert_complex_vec_close(
        &c64_values(pinv_vjp[0].as_ref().unwrap()),
        &[
            Complex64::new(0.0, -0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        1.0e-12,
    );

    let matrix_exp = matrix_exp_dyn_value(&a).unwrap();
    assert_eq!(matrix_exp.primal().dims(), &[2, 2]);
    let matrix_exp_op = MatrixExpOp;
    let matrix_exp_outputs = matrix_exp_op.primal(&[a.primal()]).unwrap();
    let matrix_exp_linearized = matrix_exp_op
        .linearize(&[a.primal()], &matrix_exp_outputs)
        .unwrap();
    let matrix_exp_jvp = matrix_exp_linearized.jvp(&[Some(da)]).unwrap();
    assert_complex_vec_close(
        &c64_values(matrix_exp_jvp[0].as_ref().unwrap()),
        &[
            z1.exp(),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        1.0e-12,
    );
    let matrix_exp_vjp = matrix_exp_linearized.vjp(&[Some(e11)], &[true]).unwrap();
    assert_complex_vec_close(
        &c64_values(matrix_exp_vjp[0].as_ref().unwrap()),
        &[
            z1.conj().exp(),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        1.0e-12,
    );
}

#[test]
fn dyn_linalg_ops_complex_det_and_slogdet_jvp_and_vjp_support() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let a = new_reverse_leaf(dyn_c64(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, -1.0),
        ],
        &[2, 2],
    ));
    let da = dyn_c64(
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        &[2, 2],
    );

    let det = det_dyn_value(&a).unwrap();
    assert_complex_vec_close(
        &c64_values(det.primal()),
        &[Complex64::new(3.0, 1.0)],
        1.0e-12,
    );
    let det_grad = det
        .grad_wrt_with_seed(dyn_c64(&[Complex64::new(1.0, 0.0)], &[]), &[&a])
        .unwrap();
    assert_complex_vec_close(
        &c64_values(det_grad[0].as_ref().unwrap()),
        &[
            Complex64::new(2.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, -1.0),
        ],
        1.0e-12,
    );

    let det_op = DetOp;
    let det_outputs = det_op.primal(&[a.primal()]).unwrap();
    let det_linearized = det_op.linearize(&[a.primal()], &det_outputs).unwrap();
    assert_all_none(&det_linearized.jvp(&[None]).unwrap());
    let det_jvp = det_linearized.jvp(&[Some(da.clone())]).unwrap();
    assert_complex_vec_close(
        &c64_values(det_jvp[0].as_ref().unwrap()),
        &[Complex64::new(2.0, -1.0)],
        1.0e-12,
    );

    let slogdet = slogdet_dyn_value(&a).unwrap();
    assert_complex_vec_close(
        &c64_values(slogdet.sign.primal()),
        &[Complex64::new(3.0 / 10.0_f64.sqrt(), 1.0 / 10.0_f64.sqrt())],
        1.0e-12,
    );
    assert_vec_close(
        &f64_values(slogdet.logabsdet.primal()),
        &[0.5 * 10.0_f64.ln()],
        1.0e-12,
    );
    let slogdet_grads = slogdet
        .logabsdet
        .grad_wrt_with_seed(dyn_scalar(1.0), &[&a])
        .unwrap();
    assert_complex_vec_close(
        &c64_values(slogdet_grads[0].as_ref().unwrap()),
        &[
            Complex64::new(0.5, 0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.4, -0.2),
        ],
        1.0e-12,
    );

    let slogdet_op = SlogdetOp;
    let slogdet_outputs = slogdet_op.primal(&[a.primal()]).unwrap();
    let slogdet_linearized = slogdet_op
        .linearize(&[a.primal()], &slogdet_outputs)
        .unwrap();
    let slogdet_none = slogdet_linearized.jvp(&[None]).unwrap();
    assert!(slogdet_none.iter().all(Option::is_none));
    let slogdet_jvp = slogdet_linearized.jvp(&[Some(da)]).unwrap();
    assert_complex_vec_close(
        &c64_values(slogdet_jvp[0].as_ref().unwrap()),
        &[Complex64::new(
            0.5 / 10.0_f64.sqrt(),
            -1.5 / 10.0_f64.sqrt(),
        )],
        1.0e-12,
    );
    assert_vec_close(
        &f64_values(slogdet_jvp[1].as_ref().unwrap()),
        &[0.5],
        1.0e-12,
    );
}
