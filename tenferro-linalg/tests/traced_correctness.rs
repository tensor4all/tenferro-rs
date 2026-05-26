use num_complex::Complex64;
use tenferro_runtime::{
    CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor, TypedTensor,
};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

fn get_f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn get_c64_data(tensor: &Tensor) -> &[Complex64] {
    tensor.as_slice::<Complex64>().unwrap()
}

fn run_many(outputs: &[&TracedTensor]) -> Vec<Tensor> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(outputs).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_linalg::register_runtime)
        .unwrap();
    executor.run_many(&program).unwrap()
}

#[test]
fn traced_tensor_namespace_exposes_svd() {
    let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
    let (_u, s, _vt) = tenferro_linalg::traced_tensor::svd(&a);

    assert_eq!(s.rank, 1);
}

#[test]
fn svd_traced_tensor_returns_three_outputs() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0]));
    let (u, s, vt) = tenferro_linalg::svd(&a);
    let results = run_many(&[&u, &s, &vt]);

    assert_eq!(results[0].shape(), &[2, 2]);
    assert_eq!(results[1].shape(), &[2]);
    assert_eq!(results[2].shape(), &[2, 2]);

    let mut singular_values = get_f64_data(&results[1]).to_vec();
    singular_values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    assert_eq!(singular_values, vec![1.0, 2.0]);
}

#[test]
fn qr_traced_tensor_returns_q_and_r() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]));
    let (q, r) = tenferro_linalg::qr(&a);
    let results = run_many(&[&q, &r]);

    assert_eq!(results[0].shape(), &[2, 2]);
    assert_eq!(results[1].shape(), &[2, 2]);
    assert_eq!(get_f64_data(&results[0]), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(get_f64_data(&results[1]), &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn eigh_traced_tensor_returns_values_and_vectors() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0]));
    let (values, vectors) = tenferro_linalg::eigh(&a);
    let results = run_many(&[&values, &vectors]);

    assert_eq!(results[0].shape(), &[2]);
    assert_eq!(results[1].shape(), &[2, 2]);

    let mut eigenvalues = get_f64_data(&results[0]).to_vec();
    eigenvalues.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    assert_eq!(eigenvalues, vec![1.0, 3.0]);
}

#[test]
fn linalg_single_output_traced_tensor_functions_eval() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![4.0, 0.0, 0.0, 9.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![8.0, 27.0]));

    let chol = tenferro_linalg::cholesky(&a);
    let solved = tenferro_linalg::solve(&a, &b);
    let triangular = tenferro_linalg::triangular_solve(&a, &b, true, true, false, false);
    let results = run_many(&[&chol, &solved, &triangular]);

    assert_eq!(get_f64_data(&results[0]), &[2.0, 0.0, 0.0, 3.0]);
    assert_eq!(get_f64_data(&results[1]), &[2.0, 3.0]);
    assert_eq!(get_f64_data(&results[2]), &[2.0, 3.0]);
}

#[test]
fn lu_traced_tensor_returns_four_outputs() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0]));
    let (p, l, u, parity) = tenferro_linalg::lu(&a);
    let results = run_many(&[&p, &l, &u, &parity]);

    assert_eq!(results[0].shape(), &[2, 2]);
    assert_eq!(results[1].shape(), &[2, 2]);
    assert_eq!(results[2].shape(), &[2, 2]);
    assert_eq!(results[3].shape(), &[] as &[usize]);
    assert_eq!(get_f64_data(&results[3]), &[-1.0]);
}

#[test]
fn full_piv_lu_solve_traced_tensor_eval() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![-1.0, 5.0]));
    let x = tenferro_linalg::full_piv_lu_solve(&a, &b);
    let results = run_many(&[&x]);

    assert_eq!(results[0].shape(), &[2, 1]);
    assert_eq!(get_f64_data(&results[0]), &[4.0, -1.0]);
}

#[test]
fn eig_traced_tensor_returns_complex_outputs() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0]));
    let (values, vectors) = tenferro_linalg::eig(&a);
    let results = run_many(&[&values, &vectors]);

    assert_eq!(results[0].shape(), &[2]);
    assert_eq!(results[1].shape(), &[2, 2]);

    let mut eigenvalues = get_c64_data(&results[0]).to_vec();
    eigenvalues.sort_by(|lhs, rhs| {
        lhs.re
            .partial_cmp(&rhs.re)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    assert_eq!(eigenvalues[0], Complex64::new(1.0, 0.0));
    assert_eq!(eigenvalues[1], Complex64::new(3.0, 0.0));
}

#[test]
fn determinant_inverse_and_eigenvalue_helpers_eval() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0]));

    let (sign, logabsdet) = tenferro_linalg::slogdet(&a);
    let determinant = tenferro_linalg::det(&a);
    let inverse = tenferro_linalg::inv(&a);
    let eigvals = tenferro_linalg::eigvals(&a);
    let eigvalsh = tenferro_linalg::eigvalsh(&a);
    let results = run_many(&[
        &sign,
        &logabsdet,
        &determinant,
        &inverse,
        &eigvals,
        &eigvalsh,
    ]);

    assert_eq!(get_f64_data(&results[0]), &[1.0]);
    assert_f64_eq(get_f64_data(&results[1])[0], (8.0f64).ln());
    assert_f64_eq(get_f64_data(&results[2])[0], 8.0);
    assert_tensor_f64_eq(get_f64_data(&results[3]), &[0.5, 0.0, 0.0, 0.25]);

    let mut general = get_c64_data(&results[4]).to_vec();
    general.sort_by(|lhs, rhs| {
        lhs.re
            .partial_cmp(&rhs.re)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    assert_eq!(general[0], Complex64::new(2.0, 0.0));
    assert_eq!(general[1], Complex64::new(4.0, 0.0));

    let mut hermitian = get_f64_data(&results[5]).to_vec();
    hermitian.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    assert_eq!(hermitian, vec![2.0, 4.0]);
}

#[test]
fn pseudoinverse_and_norm_eval() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0]));

    let pseudo_inverse = tenferro_linalg::pinv(&a);
    let frob = tenferro_linalg::norm(&a, None, Some(&[0, 1]), false);
    let results = run_many(&[&pseudo_inverse, &frob]);

    assert_tensor_f64_eq(get_f64_data(&results[0]), &[0.5, 0.0, 0.0, 0.25]);
    assert_f64_eq(get_f64_data(&results[1])[0], (20.0f64).sqrt());
}

#[test]
fn norm_supports_vector_zero_and_matrix_induced_orders() {
    let vector =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![1.0, 0.0, 2.0, -3.0]));
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]));

    let zero_norm = tenferro_linalg::norm(&vector, Some(0.0), Some(&[0]), false);
    let matrix_one = tenferro_linalg::norm(&matrix, Some(1.0), Some(&[0, 1]), false);
    let matrix_neg_one = tenferro_linalg::norm(&matrix, Some(-1.0), Some(&[0, 1]), false);
    let matrix_inf = tenferro_linalg::norm(&matrix, Some(f64::INFINITY), Some(&[0, 1]), false);
    let matrix_neg_inf =
        tenferro_linalg::norm(&matrix, Some(f64::NEG_INFINITY), Some(&[0, 1]), false);
    let results = run_many(&[
        &zero_norm,
        &matrix_one,
        &matrix_neg_one,
        &matrix_inf,
        &matrix_neg_inf,
    ]);

    assert_f64_eq(get_f64_data(&results[0])[0], 3.0);
    assert_f64_eq(get_f64_data(&results[1])[0], 6.0);
    assert_f64_eq(get_f64_data(&results[2])[0], 4.0);
    assert_f64_eq(get_f64_data(&results[3])[0], 7.0);
    assert_f64_eq(get_f64_data(&results[4])[0], 3.0);
}

fn assert_f64_eq(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1.0e-10,
        "expected {expected}, got {actual}"
    );
}

fn assert_tensor_f64_eq(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "tensor length mismatch");
    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_f64_eq(actual, expected);
    }
}
