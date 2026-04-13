use tenferro::{DotGeneralConfig, EagerTensor, Tensor};

const FD_H: f64 = 1.0e-6;
const TOL: f64 = 1.0e-5;
const FD_TOL: f64 = 1.0e-4;

fn assert_close_slice(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tol,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn finite_diff_scalar(f: impl Fn(&[f64]) -> f64, x: &[f64], index: usize) -> f64 {
    let mut plus = x.to_vec();
    let mut minus = x.to_vec();
    plus[index] += FD_H;
    minus[index] -= FD_H;
    (f(&plus) - f(&minus)) / (2.0 * FD_H)
}

fn finite_diff_lhs(
    f: impl Fn(&[f64], &[f64]) -> f64,
    lhs: &[f64],
    rhs: &[f64],
    index: usize,
) -> f64 {
    let mut plus = lhs.to_vec();
    let mut minus = lhs.to_vec();
    plus[index] += FD_H;
    minus[index] -= FD_H;
    (f(&plus, rhs) - f(&minus, rhs)) / (2.0 * FD_H)
}

fn finite_diff_rhs(
    f: impl Fn(&[f64], &[f64]) -> f64,
    lhs: &[f64],
    rhs: &[f64],
    index: usize,
) -> f64 {
    let mut plus = rhs.to_vec();
    let mut minus = rhs.to_vec();
    plus[index] += FD_H;
    minus[index] -= FD_H;
    (f(lhs, &plus) - f(lhs, &minus)) / (2.0 * FD_H)
}

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    }
}

fn eager_matmul_sum(lhs: &[f64], rhs: &[f64]) -> f64 {
    let a = EagerTensor::from_tensor(Tensor::new(vec![2, 3], lhs.to_vec()));
    let b = EagerTensor::from_tensor(Tensor::new(vec![3, 2], rhs.to_vec()));
    let loss = a
        .dot_general(&b, matmul_config())
        .unwrap()
        .reduce_sum(&[0, 1])
        .unwrap();
    f64_data(loss.data())[0]
}

#[test]
fn eager_x_squared_gradient_matches_finite_difference() {
    let x_data = vec![1.0, 2.0, 3.0];
    let x = EagerTensor::requires_grad(Tensor::new(vec![3], x_data.clone()));
    let loss = (&x * &x).reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();
    let grad = x.grad().unwrap();

    let grad_data = f64_data(grad.as_ref());
    let expected: Vec<f64> = (0..x_data.len())
        .map(|index| {
            finite_diff_scalar(
                |values| values.iter().map(|v| v * v).sum::<f64>(),
                &x_data,
                index,
            )
        })
        .collect();
    assert_close_slice(grad_data, &expected, FD_TOL);
}

#[test]
fn eager_matmul_gradients_match_finite_difference() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let a = EagerTensor::requires_grad(Tensor::new(vec![2, 3], a_data.clone()));
    let b = EagerTensor::requires_grad(Tensor::new(vec![3, 2], b_data.clone()));
    let loss = a
        .dot_general(&b, matmul_config())
        .unwrap()
        .reduce_sum(&[0, 1])
        .unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();
    let grad_a_data = f64_data(grad_a.as_ref());
    let grad_b_data = f64_data(grad_b.as_ref());

    let expected_a: Vec<f64> = (0..a_data.len())
        .map(|index| finite_diff_lhs(eager_matmul_sum, &a_data, &b_data, index))
        .collect();
    let expected_b: Vec<f64> = (0..b_data.len())
        .map(|index| finite_diff_rhs(eager_matmul_sum, &a_data, &b_data, index))
        .collect();

    assert_close_slice(grad_a_data, &expected_a, FD_TOL);
    assert_close_slice(grad_b_data, &expected_b, FD_TOL);
}

#[test]
fn eager_exp_gradient_matches_primal() {
    let x = EagerTensor::requires_grad(Tensor::new(vec![3], vec![0.0, 1.0, 2.0]));
    let loss = x.exp().unwrap().reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad = x.grad().unwrap();
    let expected = vec![1.0, 1.0_f64.exp(), 2.0_f64.exp()];
    assert_close_slice(f64_data(grad.as_ref()), &expected, TOL);
}

#[test]
fn eager_fan_out_accumulates_gradient() {
    let x = EagerTensor::requires_grad(Tensor::new(vec![3], vec![1.0, 2.0, 3.0]));
    let loss = (&x + &x).reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad = x.grad().unwrap();
    assert_close_slice(f64_data(grad.as_ref()), &[2.0, 2.0, 2.0], TOL);
}

#[test]
fn eager_detach_cuts_one_gradient_path() {
    let x = EagerTensor::requires_grad(Tensor::new(vec![3], vec![1.0, 2.0, 3.0]));
    let detached = x.detach();
    let loss = (&detached * &x).reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad = x.grad().unwrap();
    assert_close_slice(f64_data(grad.as_ref()), &[1.0, 2.0, 3.0], TOL);
    assert!(detached.grad().is_none());
}

#[test]
fn eager_untracked_tensor_behaves_like_plain_tensor() {
    let x = EagerTensor::from_tensor(Tensor::new(vec![3], vec![1.0, 2.0, 3.0]));
    let y = EagerTensor::from_tensor(Tensor::new(vec![3], vec![4.0, 5.0, 6.0]));
    let z = &x * &y;

    assert_close_slice(f64_data(z.data()), &[4.0, 10.0, 18.0], TOL);
    assert!(x.grad().is_none());
    assert!(y.grad().is_none());
    assert!(z.grad().is_none());
}
