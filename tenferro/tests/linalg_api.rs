use tenferro::engine::Engine;
use tenferro::traced::{eval_all, TracedTensor};
use tenferro::{cholesky, eigh, qr, solve, svd, triangular_solve};
use tenferro_tensor::{cpu::CpuBackend, Tensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn get_f64_data(t: &Tensor) -> &[f64] {
    match t {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected F64"),
    }
}

#[test]
fn svd_free_function_returns_three_outputs() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0]));
    let (mut u, mut s, mut vt) = svd(&a);
    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut u, &mut s, &mut vt]).unwrap();

    assert_eq!(results[0].shape(), &[2, 2]);
    assert_eq!(results[1].shape(), &[2]);
    assert_eq!(results[2].shape(), &[2, 2]);

    let mut singular_values = get_f64_data(&results[1]).to_vec();
    singular_values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    assert_eq!(singular_values, vec![1.0, 2.0]);
}

#[test]
fn qr_free_function_returns_q_and_r() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]));
    let (mut q, mut r) = qr(&a);
    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut q, &mut r]).unwrap();

    assert_eq!(results[0].shape(), &[2, 2]);
    assert_eq!(results[1].shape(), &[2, 2]);
    assert_eq!(get_f64_data(&results[0]), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(get_f64_data(&results[1]), &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn eigh_free_function_returns_values_and_vectors() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0]));
    let (mut values, mut vectors) = eigh(&a);
    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut values, &mut vectors]).unwrap();

    assert_eq!(results[0].shape(), &[2]);
    assert_eq!(results[1].shape(), &[2, 2]);

    let mut eigenvalues = get_f64_data(&results[0]).to_vec();
    eigenvalues.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    assert_eq!(eigenvalues, vec![1.0, 3.0]);
}

#[test]
fn linalg_single_output_free_functions_eval() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 2], vec![4.0, 0.0, 0.0, 9.0]));
    let b = TracedTensor::from_tensor(f64_tensor(vec![2, 1], vec![8.0, 27.0]));

    let mut chol = cholesky(&a);
    let mut solved = solve(&a, &b);
    let mut triangular = triangular_solve(&a, &b, true, true, false, false);

    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut chol, &mut solved, &mut triangular]).unwrap();

    assert_eq!(get_f64_data(&results[0]), &[2.0, 0.0, 0.0, 3.0]);
    assert_eq!(get_f64_data(&results[1]), &[2.0, 3.0]);
    assert_eq!(get_f64_data(&results[2]), &[2.0, 3.0]);
}
