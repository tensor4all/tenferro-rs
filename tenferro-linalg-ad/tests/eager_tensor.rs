use num_complex::Complex64;
use std::sync::{Arc, OnceLock};
use tenferro_ad::{CpuBackend, EagerRuntime, EagerTensor, Tensor};

fn test_ctx() -> Arc<EagerRuntime> {
    static CTX: OnceLock<Arc<EagerRuntime>> = OnceLock::new();
    CTX.get_or_init(|| EagerRuntime::with_cpu_backend(CpuBackend::new()))
        .clone()
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn c64_data(tensor: &Tensor) -> &[Complex64] {
    tensor.as_slice::<Complex64>().unwrap()
}

#[test]
fn svd_returns_correct_shapes() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]),
        test_ctx(),
    );
    let (u, s, vt) = tenferro_linalg_ad::eager_tensor::svd(&a).unwrap();

    assert_eq!(u.data().shape(), &[2, 2]);
    assert_eq!(s.data().shape(), &[2]);
    assert_eq!(vt.data().shape(), &[2, 2]);
}

#[test]
fn qr_returns_correct_shapes() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]),
        test_ctx(),
    );
    let (q, r) = tenferro_linalg_ad::eager_tensor::qr(&a).unwrap();

    assert_eq!(q.data().shape(), &[2, 2]);
    assert_eq!(r.data().shape(), &[2, 2]);
}

#[test]
fn cholesky_of_identity() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]),
        test_ctx(),
    );
    let l = tenferro_linalg_ad::eager_tensor::cholesky(&a).unwrap();

    assert_eq!(l.data().shape(), &[2, 2]);
    assert_eq!(f64_data(l.data()), &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn lu_returns_expected_factors_for_swap_matrix() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 1.0, 1.0, 0.0]),
        test_ctx(),
    );
    let (p, l, u, parity) = tenferro_linalg_ad::eager_tensor::lu(&a).unwrap();

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
fn full_piv_lu_solve_returns_expected_solution() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]),
        test_ctx(),
    );
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![-1.0_f64, 5.0]),
        test_ctx(),
    );
    let x = tenferro_linalg_ad::eager_tensor::full_piv_lu_solve(&a, &b).unwrap();

    assert_eq!(x.data().shape(), &[2, 1]);
    assert_eq!(f64_data(x.data()), &[4.0, -1.0]);
}

#[test]
fn solve_returns_expected_solution() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]),
        test_ctx(),
    );
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![4.0_f64, 8.0]),
        test_ctx(),
    );
    let x = tenferro_linalg_ad::eager_tensor::solve(&a, &b).unwrap();

    assert_eq!(x.data().shape(), &[2, 1]);
    assert_eq!(f64_data(x.data()), &[2.0, 2.0]);
}

#[test]
fn eig_returns_expected_complex_values_for_diagonal_matrix() {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]),
        test_ctx(),
    );
    let (values, vectors) = tenferro_linalg_ad::eager_tensor::eig(&a).unwrap();

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
