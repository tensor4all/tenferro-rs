use num_complex::Complex64;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{GraphCompiler, Tensor, TracedTensor, TypedTensor};

use super::support;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data).unwrap())
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
    support::run_all(&program, &[]).unwrap()
}

#[test]
fn traced_tensor_linalg_ext_exposes_svd() {
    let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap();
    let (_u, s, _vt) = a.svd().unwrap();

    assert_eq!(s.rank, 1);
}

#[test]
fn svd_traced_tensor_returns_three_outputs() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0]))
            .unwrap();
    let (u, s, vt) = a.svd().unwrap();
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
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]))
            .unwrap();
    let (q, r) = a.qr().unwrap();
    let results = run_many(&[&q, &r]);

    assert_eq!(results[0].shape(), &[2, 2]);
    assert_eq!(results[1].shape(), &[2, 2]);
    assert_eq!(get_f64_data(&results[0]), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(get_f64_data(&results[1]), &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn eigh_traced_tensor_returns_values_and_vectors() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0]))
            .unwrap();
    let (values, vectors) = a.eigh().unwrap();
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
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![4.0, 0.0, 0.0, 9.0]))
            .unwrap();
    let b =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![8.0, 27.0])).unwrap();

    let chol = a.cholesky().unwrap();
    let solved = a.solve(&b).unwrap();
    let triangular = a.triangular_solve(&b, true, true, false, false).unwrap();
    let results = run_many(&[&chol, &solved, &triangular]);

    assert_eq!(get_f64_data(&results[0]), &[2.0, 0.0, 0.0, 3.0]);
    assert_eq!(get_f64_data(&results[1]), &[2.0, 3.0]);
    assert_eq!(get_f64_data(&results[2]), &[2.0, 3.0]);
}

#[test]
fn traced_solve_accepts_a_tiny_nonzero_complex_pivot() {
    // What: traced solve's LU-prepared lowering preserves a representable nonzero complex pivot.
    let scale = 2.0_f64.powi(-600);
    let a = TracedTensor::from_tensor_concrete_shape(Tensor::C64(
        TypedTensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(scale, 0.0)]).unwrap(),
    ))
    .unwrap();
    let b = TracedTensor::from_tensor_concrete_shape(Tensor::C64(
        TypedTensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(3.0 * scale, 0.0)])
            .unwrap(),
    ))
    .unwrap();

    let solved = a.solve(&b).unwrap();
    let results = run_many(&[&solved]);

    assert_eq!(get_c64_data(&results[0]), &[Complex64::new(3.0, 0.0)]);
}

#[test]
fn lu_traced_tensor_returns_four_outputs() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0]))
            .unwrap();
    let (p, l, u, parity) = a.lu().unwrap();
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
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]))
            .unwrap();
    let b =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![-1.0, 5.0])).unwrap();
    let x = a.full_piv_lu_solve(&b).unwrap();
    let results = run_many(&[&x]);

    assert_eq!(results[0].shape(), &[2, 1]);
    assert_eq!(get_f64_data(&results[0]), &[4.0, -1.0]);
}

#[test]
fn eig_traced_tensor_returns_complex_outputs() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0]))
            .unwrap();
    let (values, vectors) = a.eig().unwrap();
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
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0]))
            .unwrap();

    let (sign, logabsdet) = a.slogdet().unwrap();
    let determinant = a.det().unwrap();
    let inverse = a.inv().unwrap();
    let eigvals = a.eigvals().unwrap();
    let eigvalsh = a.eigvalsh().unwrap();
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
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0]))
            .unwrap();

    let pseudo_inverse = a.pinv().unwrap();
    let frob = a.norm(None, Some(&[0, 1]), false).unwrap();
    let results = run_many(&[&pseudo_inverse, &frob]);

    assert_tensor_f64_eq(get_f64_data(&results[0]), &[0.5, 0.0, 0.0, 0.25]);
    assert_f64_eq(get_f64_data(&results[1])[0], (20.0f64).sqrt());
}

#[test]
fn norm_supports_vector_zero_and_matrix_induced_orders() {
    let vector =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![1.0, 0.0, 2.0, -3.0]))
            .unwrap();
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]))
            .unwrap();

    let zero_norm = vector.norm(Some(0.0), Some(&[0]), false).unwrap();
    let matrix_one = matrix.norm(Some(1.0), Some(&[0, 1]), false).unwrap();
    let matrix_neg_one = matrix.norm(Some(-1.0), Some(&[0, 1]), false).unwrap();
    let matrix_inf = matrix
        .norm(Some(f64::INFINITY), Some(&[0, 1]), false)
        .unwrap();
    let matrix_neg_inf = matrix
        .norm(Some(f64::NEG_INFINITY), Some(&[0, 1]), false)
        .unwrap();
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
