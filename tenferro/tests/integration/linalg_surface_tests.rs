use num_complex::Complex64;
use tenferro::{set_default_runtime, LuPivot, NormKind, RuntimeContext, SvdOptions, Tensor};
use tenferro_prims::CpuContext;

fn approx_eq(lhs: &[f64], rhs: &[f64]) {
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        assert!((lhs - rhs).abs() < 1.0e-12, "lhs={lhs}, rhs={rhs}");
    }
}

fn with_cpu_runtime() -> tenferro::DefaultRuntimeGuard {
    set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)))
}

#[test]
fn tensor_einsum_is_public_and_backpropagates() {
    let _runtime = with_cpu_runtime();
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .requires_grad_(true);
    let y = Tensor::from_slice(&[3.0_f64, 4.0], &[2])
        .unwrap()
        .requires_grad_(true);

    let out = Tensor::einsum("i,i->", &[&x, &y]).unwrap();
    assert_eq!(out.dims(), &[] as &[usize]);
    assert_eq!(out.try_to_vec::<f64>().unwrap(), vec![11.0]);

    out.backward().unwrap();

    approx_eq(
        &x.grad().unwrap().unwrap().try_to_vec::<f64>().unwrap(),
        &[3.0, 4.0],
    );
    approx_eq(
        &y.grad().unwrap().unwrap().try_to_vec::<f64>().unwrap(),
        &[1.0, 2.0],
    );
}

#[test]
fn tensor_solve_det_and_norm_are_public() {
    let _runtime = with_cpu_runtime();
    let a = Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2]).unwrap();
    let b = Tensor::from_slice(&[4.0_f64, 9.0], &[2]).unwrap();

    let solved = a.solve(&b).unwrap();
    approx_eq(&solved.try_to_vec::<f64>().unwrap(), &[2.0, 3.0]);

    let det = a.det().unwrap();
    assert_eq!(det.dims(), &[] as &[usize]);
    approx_eq(&det.try_to_vec::<f64>().unwrap(), &[6.0]);

    let norm = b.norm(NormKind::Fro).unwrap();
    assert_eq!(norm.dims(), &[] as &[usize]);
    approx_eq(&norm.try_to_vec::<f64>().unwrap(), &[97.0_f64.sqrt()]);
}

#[test]
fn tensor_qr_and_svd_return_tensor_wrappers() {
    let _runtime = with_cpu_runtime();
    let a = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0, 0.0, 1.0], &[3, 2])
        .unwrap()
        .requires_grad_(true);

    let qr = a.qr().unwrap();
    assert_eq!(qr.q.dims(), &[3, 2]);
    assert_eq!(qr.r.dims(), &[2, 2]);
    assert!(qr.q.requires_grad());
    assert!(qr.r.requires_grad());
    assert!(qr.q.shares_reverse_graph(&qr.r));

    let svd = a
        .svd(Some(SvdOptions {
            max_rank: None,
            cutoff: None,
        }))
        .unwrap();
    assert_eq!(svd.u.dims(), &[3, 2]);
    assert_eq!(svd.s.dims(), &[2]);
    assert_eq!(svd.vt.dims(), &[2, 2]);
    assert!(svd.u.requires_grad());
    assert!(svd.s.requires_grad());
    assert!(svd.vt.requires_grad());
    assert!(svd.u.shares_reverse_graph(&svd.s));
    assert!(svd.s.shares_reverse_graph(&svd.vt));
}

#[test]
fn tensor_additional_linalg_primitives_are_public() {
    let _runtime = with_cpu_runtime();
    let a = Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2]).unwrap();
    let rhs = Tensor::from_slice(&[4.0_f64, 9.0], &[2]).unwrap();
    let spd = Tensor::from_slice(&[4.0_f64, 0.0, 0.0, 9.0], &[2, 2]).unwrap();
    let matrix_exp_input = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 2.0], &[2, 2]).unwrap();

    let triangular = a.solve_triangular(&rhs, true).unwrap();
    approx_eq(&triangular.try_to_vec::<f64>().unwrap(), &[2.0, 3.0]);

    let inv = a.inv().unwrap();
    approx_eq(
        &inv.try_to_vec::<f64>().unwrap(),
        &[0.5, 0.0, 0.0, 1.0 / 3.0],
    );

    let slogdet = a.slogdet().unwrap();
    approx_eq(&slogdet.sign.try_to_vec::<f64>().unwrap(), &[1.0]);
    approx_eq(
        &slogdet.logabsdet.try_to_vec::<f64>().unwrap(),
        &[(6.0_f64).ln()],
    );

    let cholesky = spd.cholesky().unwrap();
    approx_eq(
        &cholesky.try_to_vec::<f64>().unwrap(),
        &[2.0, 0.0, 0.0, 3.0],
    );

    let pinv = a.pinv(None).unwrap();
    approx_eq(
        &pinv.try_to_vec::<f64>().unwrap(),
        &[0.5, 0.0, 0.0, 1.0 / 3.0],
    );

    let matrix_exp = matrix_exp_input.matrix_exp().unwrap();
    approx_eq(
        &matrix_exp.try_to_vec::<f64>().unwrap(),
        &[1.0_f64.exp(), 0.0, 0.0, 2.0_f64.exp()],
    );
}

#[test]
fn tensor_lstsq_lu_eig_and_eigen_are_public() {
    let _runtime = with_cpu_runtime();
    let a = Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2]).unwrap();
    let rhs = Tensor::from_slice(&[4.0_f64, 9.0], &[2]).unwrap();

    let lstsq = a.lstsq(&rhs).unwrap();
    approx_eq(&lstsq.x.try_to_vec::<f64>().unwrap(), &[2.0, 3.0]);
    approx_eq(&lstsq.residual.try_to_vec::<f64>().unwrap(), &[0.0, 0.0]);

    let lu = a.lu(LuPivot::Partial).unwrap();
    approx_eq(&lu.p.try_to_vec::<f64>().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
    approx_eq(&lu.l.try_to_vec::<f64>().unwrap(), &[1.0, 0.0, 0.0, 1.0]);
    approx_eq(&lu.u.try_to_vec::<f64>().unwrap(), &[2.0, 0.0, 0.0, 3.0]);

    let eig = a.eig().unwrap();
    assert_eq!(
        eig.values.try_to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)]
    );
    assert_eq!(
        eig.vectors.try_to_vec::<Complex64>().unwrap(),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]
    );

    let eigen = a.eigen().unwrap();
    approx_eq(&eigen.values.try_to_vec::<f64>().unwrap(), &[2.0, 3.0]);
    approx_eq(
        &eigen.vectors.try_to_vec::<f64>().unwrap(),
        &[1.0, 0.0, 0.0, 1.0],
    );
}
