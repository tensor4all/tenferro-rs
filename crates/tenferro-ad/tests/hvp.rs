//! Higher-order AD tests: Hessian-vector products.
//!
//! Primary approach: Forward-over-Reverse (FoR) = jvp(grad(f), x, v),
//! matching JAX's standard HVP pattern.

use tenferro_ad::TracedTensorAdExt;
mod support;
use support::RunTraced;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::traced::TracedTensor;
use tenferro_runtime::{DotGeneralConfig, GraphExecutor, Tensor, TypedTensor};

const TOL: f64 = 1e-5;
fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
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
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let t = traced;
    t.run_with(&mut engine).unwrap().clone()
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

fn matvec(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
    lhs.dot_general(
        rhs,
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    )
}

fn vector_dot(lhs: &TracedTensor, rhs: &TracedTensor) -> TracedTensor {
    lhs.dot_general(
        rhs,
        DotGeneralConfig {
            lhs_contracting_dims: vec![0],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    )
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

    let x = TracedTensor::from_tensor_concrete_shape(f64_scalar(x_val));
    let y = &(&x * &x) * &x;

    let g = y.grad(&x).unwrap(); // reverse: f'(x) = 3x^2
    let v = TracedTensor::from_tensor_concrete_shape(f64_scalar(1.0));
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

    let x = TracedTensor::from_tensor_concrete_shape(f64_scalar(x_val));
    let y = x.sin().exp();

    let g = y.grad(&x).unwrap();
    let v = TracedTensor::from_tensor_concrete_shape(f64_scalar(1.0));
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

    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![n], x_data.clone()));
    let y = x.exp().reduce_sum(&[0]);

    let g = y.grad(&x).unwrap(); // shape [n]
    let v = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![n], v_data.clone()));
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

    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![n, n], a_data));
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![n], x_data));
    let v = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![n], v_data));

    let ax = matvec(&a, &x);
    let y = vector_dot(&x, &ax);

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

    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![n, n], a_data));
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![n], x_data));
    let v = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![n], v_data));

    let ax = matvec(&a, &x);
    let y = vector_dot(&x, &ax);

    let g = y.grad(&x).unwrap();
    let gv = vector_dot(&g, &v);
    let hvp = gv.grad(&x).unwrap();

    let hvp_tensor = eval_tensor(hvp);
    let hvp_val = get_f64_data(&hvp_tensor);
    assert_close(hvp_val, &[6.0, 13.0], TOL);
}
