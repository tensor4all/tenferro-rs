use num_complex::Complex64;
use tenferro::{EagerTensor, Tensor};

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn c64_data(tensor: &Tensor) -> &[Complex64] {
    tensor.as_slice::<Complex64>().unwrap()
}

#[test]
fn svd_returns_correct_shapes() {
    let a = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]));
    let (u, s, vt) = a.svd().unwrap();

    assert_eq!(u.data().shape(), &[2, 2]);
    assert_eq!(s.data().shape(), &[2]);
    assert_eq!(vt.data().shape(), &[2, 2]);
}

#[test]
fn qr_returns_correct_shapes() {
    let a = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]));
    let (q, r) = a.qr().unwrap();

    assert_eq!(q.data().shape(), &[2, 2]);
    assert_eq!(r.data().shape(), &[2, 2]);
}

#[test]
fn qr_second_output_backward_records_selected_output_slot() {
    let a = EagerTensor::requires_grad(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]));
    let (_q, r) = a.qr().unwrap();
    let loss = r.reduce_sum(&[0, 1]).unwrap();

    let _cotangents = loss.backward().unwrap();

    let grad = a.grad().expect("gradient for qr input");
    assert_eq!(grad.shape(), &[2, 2]);
}

#[test]
fn cholesky_of_identity() {
    let a = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]));
    let l = a.cholesky().unwrap();

    assert_eq!(l.data().shape(), &[2, 2]);
    assert_eq!(f64_data(l.data()), &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn svd_gradient_smoke() {
    let a = EagerTensor::requires_grad(Tensor::from_vec(vec![2, 2], vec![3.0_f64, 0.0, 0.0, 1.0]));
    let (_, s, _) = a.svd().unwrap();
    let loss = s.reduce_sum(&[0]).unwrap();

    let _cotangents = loss.backward().unwrap();

    let grad = a.grad();
    assert!(grad.is_some());
    assert_eq!(grad.unwrap().shape(), &[2, 2]);
}

#[test]
fn lu_returns_expected_factors_for_swap_matrix() {
    let a = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![0.0_f64, 1.0, 1.0, 0.0]));
    let (p, l, u, parity) = a.lu().unwrap();

    assert_eq!(p.data().shape(), &[2, 2]);
    assert_eq!(l.data().shape(), &[2, 2]);
    assert_eq!(u.data().shape(), &[2, 2]);
    assert_eq!(parity.data().shape(), &[] as &[usize]);

    assert_eq!(f64_data(p.data()), &[0.0, 1.0, 1.0, 0.0]);
    assert_eq!(f64_data(l.data()), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(f64_data(u.data()), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(f64_data(parity.data()), &[-1.0]);
}

#[test]
fn full_piv_lu_returns_expected_shapes() {
    let a = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]));
    let (p, l, u, q, parity) = a.full_piv_lu().unwrap();

    assert_eq!(p.data().shape(), &[2, 2]);
    assert_eq!(l.data().shape(), &[2, 2]);
    assert_eq!(u.data().shape(), &[2, 2]);
    assert_eq!(q.data().shape(), &[2, 2]);
    assert_eq!(parity.data().shape(), &[] as &[usize]);
}

#[test]
fn full_piv_lu_solve_returns_expected_solution() {
    let a = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]));
    let b = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 1], vec![-1.0_f64, 5.0]));
    let x = a.full_piv_lu_solve(&b).unwrap();

    assert_eq!(x.data().shape(), &[2, 1]);
    assert_eq!(f64_data(x.data()), &[4.0, -1.0]);
}

#[test]
fn eigh_returns_expected_values_for_diagonal_matrix() {
    let a = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]));
    let (values, vectors) = a.eigh().unwrap();

    assert_eq!(values.data().shape(), &[2]);
    assert_eq!(vectors.data().shape(), &[2, 2]);

    let mut sorted = f64_data(values.data()).to_vec();
    sorted.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    assert_eq!(sorted, vec![1.0, 3.0]);
}

#[test]
fn eig_returns_expected_complex_values_for_diagonal_matrix() {
    let a = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]));
    let (values, vectors) = a.eig().unwrap();

    assert_eq!(values.data().shape(), &[2]);
    assert_eq!(vectors.data().shape(), &[2, 2]);

    let mut sorted = c64_data(values.data()).to_vec();
    sorted.sort_by(|lhs, rhs| {
        lhs.re
            .partial_cmp(&rhs.re)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    assert_eq!(
        sorted,
        vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)]
    );
}

#[test]
fn triangular_solve_returns_expected_solution() {
    let a = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]));
    let b = EagerTensor::from_tensor(Tensor::from_vec(vec![2, 1], vec![4.0_f64, 8.0]));
    let x = a.triangular_solve(&b, true, true, false, false).unwrap();

    assert_eq!(x.data().shape(), &[2, 1]);
    assert_eq!(f64_data(x.data()), &[2.0, 2.0]);
}
