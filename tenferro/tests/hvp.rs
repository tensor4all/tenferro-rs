//! Higher-order AD tests: Hessian-vector products.
//!
//! Primary approach: Forward-over-Reverse (FoR) = jvp(grad(f), x, v),
//! matching JAX's standard HVP pattern.

use tenferro::einsum::einsum;
use tenferro::engine::Engine;
use tenferro::traced::TracedTensor;
use tenferro::{qr, svd, CpuBackend, Tensor, TypedTensor};

const TOL: f64 = 1e-5;
const FD_H: f64 = 1e-5;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn f64_scalar(val: f64) -> Tensor {
    f64_tensor(vec![], vec![val])
}

fn get_f64_data(t: &Tensor) -> &[f64] {
    match t {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected F64"),
    }
}

fn eval_tensor(traced: TracedTensor) -> Tensor {
    let mut engine = Engine::new(CpuBackend::new());
    let mut t = traced;
    t.eval(&mut engine).unwrap().clone()
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        let scale = e.abs().max(1.0);
        assert!(
            diff / scale < tol,
            "index {i}: actual={a}, expected={e}, rel_diff={}",
            diff / scale
        );
    }
}

// ═════════════════════════════════════════════════════════════
// Part 1: Basic scalar functions
// ═════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────
// f(x) = x^3  →  f''(x) = 6x
// FoR: jvp(grad(f, x), x, v=1)
// ─────────────────────────────────────────────────────────────
#[test]
fn hvp_for_scalar_cubic() {
    let x_val = 2.5_f64;

    let x = TracedTensor::from_tensor(f64_scalar(x_val));
    let y = &(&x * &x) * &x;

    let g = y.grad(&x).unwrap(); // reverse: f'(x) = 3x^2
    let v = TracedTensor::from_tensor(f64_scalar(1.0));
    let hv = g.jvp(&x, &v); // forward-of-reverse: f''(x)

    let hv_tensor = eval_tensor(hv);
    let hv_val = get_f64_data(&hv_tensor)[0];
    let expected = 6.0 * x_val;
    assert!(
        (hv_val - expected).abs() < TOL,
        "f''({x_val}): actual={hv_val}, expected={expected}"
    );
}

// ─────────────────────────────────────────────────────────────
// f(x) = exp(sin(x))
//   f'(x) = cos(x) exp(sin(x))
//   f''(x) = (-sin(x) + cos²(x)) exp(sin(x))
// ─────────────────────────────────────────────────────────────
#[test]
fn hvp_for_scalar_exp_sin() {
    let x_val = 1.3_f64;

    let x = TracedTensor::from_tensor(f64_scalar(x_val));
    let y = x.sin().exp();

    let g = y.grad(&x).unwrap();
    let v = TracedTensor::from_tensor(f64_scalar(1.0));
    let hv = g.jvp(&x, &v);

    let hv_tensor = eval_tensor(hv);
    let hv_val = get_f64_data(&hv_tensor)[0];
    let expected = (-x_val.sin() + x_val.cos().powi(2)) * x_val.sin().exp();
    assert!(
        (hv_val - expected).abs() < TOL,
        "f''({x_val}): actual={hv_val}, expected={expected}"
    );
}

// ─────────────────────────────────────────────────────────────
// f(x) = sum(exp(x))  (vector → scalar)
//   H = diag(exp(x)),  Hv = exp(x) ⊙ v
// FoR: jvp(grad(f, x), x, v)
// ─────────────────────────────────────────────────────────────
#[test]
fn hvp_for_vector_exp_sum() {
    let x_data = vec![0.5, 1.0, -0.3];
    let v_data = vec![1.0, 2.0, 3.0];
    let n = x_data.len();

    let x = TracedTensor::from_tensor(f64_tensor(vec![n], x_data.clone()));
    let y = x.exp().reduce_sum(&[0]);

    let g = y.grad(&x).unwrap(); // shape [n]
    let v = TracedTensor::from_tensor(f64_tensor(vec![n], v_data.clone()));
    let hv = g.jvp(&x, &v); // FoR: shape [n]

    let hv_tensor = eval_tensor(hv);
    let hv_val = get_f64_data(&hv_tensor);
    let expected: Vec<f64> = x_data
        .iter()
        .zip(&v_data)
        .map(|(xi, vi)| xi.exp() * vi)
        .collect();
    assert_close(hv_val, &expected, TOL);
}

// ─────────────────────────────────────────────────────────────
// f(x) = x^T A x  (quadratic form, vector → scalar)
//   H = A + A^T,  Hv = (A + A^T) v
// ─────────────────────────────────────────────────────────────
#[test]
fn hvp_for_quadratic_form() {
    #[rustfmt::skip]
    let a_data = vec![
        2.0, 0.5,
        1.5, 3.0,
    ];
    let x_data = vec![1.0, -1.0];
    let v_data = vec![0.5, 2.0];
    let n = 2;

    let a = TracedTensor::from_tensor(f64_tensor(vec![n, n], a_data));
    let x = TracedTensor::from_tensor(f64_tensor(vec![n], x_data));
    let v = TracedTensor::from_tensor(f64_tensor(vec![n], v_data));

    let mut engine = Engine::new(CpuBackend::new());
    let y = einsum(&mut engine, &[&x, &a, &x], "i,ij,j->").unwrap();

    let g = y.grad(&x).unwrap();
    let hv = g.jvp(&x, &v);

    let hv_tensor = eval_tensor(hv);
    let hv_val = get_f64_data(&hv_tensor);
    // H = A + A^T = [[4, 2], [2, 6]]
    // Hv = [4*0.5 + 2*2.0, 2*0.5 + 6*2.0] = [6.0, 13.0]
    assert_close(hv_val, &[6.0, 13.0], TOL);
}

// ─────────────────────────────────────────────────────────────
// Same quadratic form but via RoR: grad(dot(grad, v), x)
// Verifies both approaches agree.
// ─────────────────────────────────────────────────────────────
#[test]
fn hvp_ror_quadratic_form() {
    #[rustfmt::skip]
    let a_data = vec![
        2.0, 0.5,
        1.5, 3.0,
    ];
    let x_data = vec![1.0, -1.0];
    let v_data = vec![0.5, 2.0];
    let n = 2;

    let a = TracedTensor::from_tensor(f64_tensor(vec![n, n], a_data));
    let x = TracedTensor::from_tensor(f64_tensor(vec![n], x_data));
    let v = TracedTensor::from_tensor(f64_tensor(vec![n], v_data));

    let mut engine = Engine::new(CpuBackend::new());
    let y = einsum(&mut engine, &[&x, &a, &x], "i,ij,j->").unwrap();

    let g = y.grad(&x).unwrap();
    let gv = einsum(&mut engine, &[&g, &v], "i,i->").unwrap();
    let hvp = gv.grad(&x).unwrap();

    let hvp_tensor = eval_tensor(hvp);
    let hvp_val = get_f64_data(&hvp_tensor);
    assert_close(hvp_val, &[6.0, 13.0], TOL);
}

// ═════════════════════════════════════════════════════════════
// Part 2: Linalg (SVD, QR)
// ═════════════════════════════════════════════════════════════

/// Finite-difference gradient of a matrix → scalar function.
fn fd_matrix_grad(
    f: &dyn Fn(&[f64], usize, usize) -> f64,
    a: &[f64],
    m: usize,
    n: usize,
) -> Vec<f64> {
    let mut grad = vec![0.0; m * n];
    for idx in 0..m * n {
        let mut ap = a.to_vec();
        let mut am = a.to_vec();
        ap[idx] += FD_H;
        am[idx] -= FD_H;
        grad[idx] = (f(&ap, m, n) - f(&am, m, n)) / (2.0 * FD_H);
    }
    grad
}

/// Finite-difference HVP: (grad(A+hV) - grad(A-hV)) / 2h
fn fd_matrix_hvp(
    f: &dyn Fn(&[f64], usize, usize) -> f64,
    a: &[f64],
    v: &[f64],
    m: usize,
    n: usize,
) -> Vec<f64> {
    let mut ap = a.to_vec();
    let mut am = a.to_vec();
    for i in 0..m * n {
        ap[i] += FD_H * v[i];
        am[i] -= FD_H * v[i];
    }
    let gp = fd_matrix_grad(f, &ap, m, n);
    let gm = fd_matrix_grad(f, &am, m, n);
    gp.iter()
        .zip(&gm)
        .map(|(a, b)| (a - b) / (2.0 * FD_H))
        .collect()
}

// ─────────────────────────────────────────────────────────────
// f(A) = sum(sigma(A)^2), computed via SVD
// FoR HVP verified against finite differences
// ─────────────────────────────────────────────────────────────
#[test]
fn hvp_for_svd_sum_sigma_sq() {
    let m = 3;
    let n = 2;
    #[rustfmt::skip]
    let a_data = vec![
        3.0, 0.1, 0.2,
        0.3, 2.0, 0.4,
    ]; // col-major [3,2]
    let v_data: Vec<f64> = (0..m * n).map(|i| 0.1 * (i as f64 + 1.0)).collect();

    let a_tt = TracedTensor::from_tensor(f64_tensor(vec![m, n], a_data.clone()));
    let v_tt = TracedTensor::from_tensor(f64_tensor(vec![m, n], v_data.clone()));

    let (_u, s, _vt) = svd(&a_tt);
    let y = (&s * &s).reduce_sum(&[0]);

    let g = y.grad(&a_tt).unwrap(); // shape [m, n]
    let hv = g.jvp(&a_tt, &v_tt); // FoR HVP, shape [m, n]

    let hv_tensor = eval_tensor(hv);
    let hv_val = get_f64_data(&hv_tensor);

    let f_ref = |data: &[f64], rows: usize, cols: usize| -> f64 {
        let a_ref = TracedTensor::from_tensor(f64_tensor(vec![rows, cols], data.to_vec()));
        let (_u, s, _vt) = svd(&a_ref);
        let y_ref = (&s * &s).reduce_sum(&[0]);
        let t = eval_tensor(y_ref);
        get_f64_data(&t)[0]
    };
    let fd = fd_matrix_hvp(&f_ref, &a_data, &v_data, m, n);

    assert_close(hv_val, &fd, 1e-3);
}

// ─────────────────────────────────────────────────────────────
// f(A) = sum(diag(R)^2) where A = QR
// FoR HVP verified against FD of the AD gradient
// ─────────────────────────────────────────────────────────────
#[test]
fn hvp_for_qr_diag_r_sq() {
    let n = 3;
    #[rustfmt::skip]
    let a_data = vec![
        4.0, 0.1, 0.2,
        0.1, 3.0, 0.3,
        0.2, 0.3, 5.0,
    ]; // col-major [3,3]
    let v_data: Vec<f64> = (0..n * n).map(|i| 0.1 * (i as f64 + 1.0)).collect();

    let a_tt = TracedTensor::from_tensor(f64_tensor(vec![n, n], a_data.clone()));
    let v_tt = TracedTensor::from_tensor(f64_tensor(vec![n, n], v_data.clone()));

    let (_q, r) = qr(&a_tt);
    let diag_r = r.extract_diag(0, 1);
    let y = (&diag_r * &diag_r).reduce_sum(&[0]);

    let g = y.grad(&a_tt).unwrap();
    let hv = g.jvp(&a_tt, &v_tt);

    let hv_tensor = eval_tensor(hv);
    let hv_val = get_f64_data(&hv_tensor);

    // FD of AD gradient: (grad(A+hV) - grad(A-hV)) / 2h
    // Uses AD for the inner gradient to avoid nested FD noise.
    let h = 1e-5;
    let ap: Vec<f64> = a_data.iter().zip(&v_data).map(|(a, v)| a + h * v).collect();
    let am: Vec<f64> = a_data.iter().zip(&v_data).map(|(a, v)| a - h * v).collect();

    let grad_at = |data: Vec<f64>| -> Vec<f64> {
        let a_ref = TracedTensor::from_tensor(f64_tensor(vec![n, n], data));
        let (_q, r) = qr(&a_ref);
        let diag_r = r.extract_diag(0, 1);
        let y_ref = (&diag_r * &diag_r).reduce_sum(&[0]);
        let g_ref = y_ref.grad(&a_ref).unwrap();
        let t = eval_tensor(g_ref);
        get_f64_data(&t).to_vec()
    };
    let gp = grad_at(ap);
    let gm = grad_at(am);
    let fd: Vec<f64> = gp
        .iter()
        .zip(&gm)
        .map(|(a, b)| (a - b) / (2.0 * h))
        .collect();

    assert_close(hv_val, &fd, 1e-3);
}
