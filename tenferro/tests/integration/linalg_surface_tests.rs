use tenferro::{set_default_runtime, NormKind, RuntimeContext, SvdOptions, Tensor};
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
