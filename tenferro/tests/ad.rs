use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro::engine::Engine;
use tenferro::traced::TracedTensor;
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{DotGeneralConfig, Tensor, TypedTensor};

const TOL: f64 = 1e-6;

fn finite_diff_scalar(
    f: impl Fn(&[f64]) -> f64,
    x: &[f64],
    idx: usize,
    h: f64,
) -> f64 {
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[idx] += h;
    xm[idx] -= h;
    (f(&xp) - f(&xm)) / (2.0 * h)
}

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn get_f64_data(tensor: &Tensor) -> &[f64] {
    match tensor {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected f64 tensor"),
    }
}

fn eval_tensor(mut traced: TracedTensor) -> Tensor {
    let mut engine = Engine::new(CpuBackend::new());
    traced.eval(&mut engine).unwrap().clone()
}

fn eval_scalar(traced: TracedTensor) -> f64 {
    get_f64_data(&eval_tensor(traced))[0]
}

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

fn assert_close_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn grad_x_squared() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = x.traced_mul(&x);
    let loss = y.traced_reduce_sum(&[0]);
    let grad = loss.grad(&x);

    let result = eval_tensor(grad);
    assert_close_slice(get_f64_data(&result), &[2.0, 4.0, 6.0]);
}

#[test]
fn grad_matmul_sum() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], a_data.clone()));
    let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], b_data.clone()));
    let matmul = a.traced_dot_general(&b, matmul_config());
    let loss = matmul.traced_reduce_sum(&[0, 1]);
    let grad = loss.grad(&a);

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);

    let f = |xs: &[f64]| {
        let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], xs.to_vec()));
        let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], b_data.clone()));
        let matmul = a.traced_dot_general(&b, matmul_config());
        let loss = matmul.traced_reduce_sum(&[0, 1]);
        eval_scalar(loss)
    };

    for index in 0..a_data.len() {
        let expected = finite_diff_scalar(&f, &a_data, index, 1e-6);
        assert!(
            (grad_data[index] - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {}",
            grad_data[index]
        );
    }
}

#[test]
fn jvp_elementwise_mul() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = TracedTensor::from_tensor(f64_tensor(vec![3], vec![4.0, 5.0, 6.0]));
    let dx = TracedTensor::from_tensor(f64_tensor(vec![3], vec![1.0, 0.0, 0.0]));

    let prod = x.traced_mul(&y);
    let jvp = prod.jvp(&x, &dx);

    let result = eval_tensor(jvp);
    assert_close_slice(get_f64_data(&result), &[4.0, 0.0, 0.0]);
}

#[test]
fn vjp_matmul() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let cotangent = TracedTensor::from_tensor(f64_tensor(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]));

    let y = a.traced_dot_general(&b, matmul_config());
    let vjp = y.vjp(&a, &cotangent);

    let result = eval_tensor(vjp);
    assert_close_slice(get_f64_data(&result), &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
}

#[test]
fn grad_nonscalar_errors() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let y = a.traced_dot_general(&b, matmul_config());

    let result = catch_unwind(AssertUnwindSafe(|| y.grad(&a)));
    assert!(result.is_err());
}

#[test]
fn grad_broadcast_reduce() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = x.traced_broadcast_in_dim(&[3, 3], &[0]);
    let loss = y.traced_reduce_sum(&[0, 1]);
    let grad = loss.grad(&x);

    let result = eval_tensor(grad);
    assert_close_slice(get_f64_data(&result), &[3.0, 3.0, 3.0]);
}

#[test]
fn grad_reshape() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
    let y = x.traced_reshape(&[2, 2]);
    let loss = y.traced_reduce_sum(&[0, 1]);
    let grad = loss.grad(&x);

    let result = eval_tensor(grad);
    assert_eq!(result.shape(), &[4]);
    assert_close_slice(get_f64_data(&result), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn grad_transpose() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let y = a.traced_transpose(&[1, 0]);
    let loss = y.traced_reduce_sum(&[0, 1]);
    let grad = loss.grad(&a);

    let result = eval_tensor(grad);
    assert_eq!(result.shape(), &[2, 3]);
    assert_close_slice(get_f64_data(&result), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}
