use num_complex::Complex64;
use tenferro::engine::Engine;
use tenferro::traced::{eval_all, TracedTensor};
use tenferro::{
    cholesky, det, eig, eigh, full_piv_lu, full_piv_lu_solve, inv, lu, norm, pinv, pinv_with_rtol,
    qr, slogdet, solve, svd, triangular_solve,
};
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

fn get_c64_data(t: &Tensor) -> &[Complex64] {
    match t {
        Tensor::C64(inner) => inner.host_data(),
        _ => panic!("expected C64"),
    }
}

#[test]
fn svd_free_function_returns_three_outputs() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0]));
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
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]));
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
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0]));
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
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![4.0, 0.0, 0.0, 9.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![8.0, 27.0]));

    let mut chol = cholesky(&a);
    let mut solved = solve(&a, &b);
    let mut triangular = triangular_solve(&a, &b, true, true, false, false);

    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut chol, &mut solved, &mut triangular]).unwrap();

    assert_eq!(get_f64_data(&results[0]), &[2.0, 0.0, 0.0, 3.0]);
    assert_eq!(get_f64_data(&results[1]), &[2.0, 3.0]);
    assert_eq!(get_f64_data(&results[2]), &[2.0, 3.0]);
}

#[test]
fn lu_free_function_returns_four_outputs() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![0.0, 1.0, 1.0, 0.0]));
    let (mut p, mut l, mut u, mut parity) = lu(&a);
    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut p, &mut l, &mut u, &mut parity]).unwrap();

    assert_eq!(results[0].shape(), &[2, 2]);
    assert_eq!(results[1].shape(), &[2, 2]);
    assert_eq!(results[2].shape(), &[2, 2]);
    assert_eq!(results[3].shape(), &[] as &[usize]);
    assert_eq!(get_f64_data(&results[3]), &[-1.0]);
}

#[test]
fn full_piv_lu_free_function_returns_five_outputs() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]));
    let (mut p, mut l, mut u, mut q, mut parity) = full_piv_lu(&a);
    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(
        &mut engine,
        &mut [&mut p, &mut l, &mut u, &mut q, &mut parity],
    )
    .unwrap();

    assert_eq!(results[0].shape(), &[2, 2]);
    assert_eq!(results[1].shape(), &[2, 2]);
    assert_eq!(results[2].shape(), &[2, 2]);
    assert_eq!(results[3].shape(), &[2, 2]);
    assert_eq!(results[4].shape(), &[] as &[usize]);
}

#[test]
fn full_piv_lu_solve_free_function_eval() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![-1.0, 5.0]));
    let mut x = full_piv_lu_solve(&a, &b);

    let mut engine = Engine::new(CpuBackend::new());
    let out = x.eval(&mut engine).unwrap();

    assert_eq!(out.shape(), &[2, 1]);
    assert_eq!(get_f64_data(out), &[4.0, -1.0]);
}

#[test]
fn eig_free_function_returns_complex_outputs() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 3.0]));
    let (mut values, mut vectors) = eig(&a);
    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut values, &mut vectors]).unwrap();

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
fn determinant_inverse_pseudoinverse_and_norm_eval() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0]));

    let (mut sign, mut logabsdet) = slogdet(&a);
    let mut determinant = det(&a);
    let mut inverse = inv(&a);
    let mut pseudo_inverse = pinv(&a);
    let mut frob = norm(&a, None, Some(&[0, 1]), false);

    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(
        &mut engine,
        &mut [
            &mut sign,
            &mut logabsdet,
            &mut determinant,
            &mut inverse,
            &mut pseudo_inverse,
            &mut frob,
        ],
    )
    .unwrap();

    assert_eq!(get_f64_data(&results[0]), &[1.0]);
    assert_f64_eq(get_f64_data(&results[1])[0], (8.0f64).ln());
    assert_f64_eq(get_f64_data(&results[2])[0], 8.0);
    assert_tensor_f64_eq(get_f64_data(&results[3]), &[0.5, 0.0, 0.0, 0.25]);
    assert_tensor_f64_eq(get_f64_data(&results[4]), &[0.5, 0.0, 0.0, 0.25]);
    assert_f64_eq(get_f64_data(&results[5])[0], (20.0f64).sqrt());
}

#[test]
fn pinv_with_large_rtol_discards_all_singular_values() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0]));
    let mut pseudo_inverse = pinv_with_rtol(&a, 1.0);

    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut pseudo_inverse]).unwrap();

    assert_tensor_f64_eq(get_f64_data(&results[0]), &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn norm_supports_vector_zero_and_matrix_induced_orders() {
    let vector =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![1.0, 0.0, 2.0, -3.0]));
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0]));

    let mut zero_norm = norm(&vector, Some(0.0), Some(&[0]), false);
    let mut matrix_one = norm(&matrix, Some(1.0), Some(&[0, 1]), false);
    let mut matrix_neg_one = norm(&matrix, Some(-1.0), Some(&[0, 1]), false);
    let mut matrix_inf = norm(&matrix, Some(f64::INFINITY), Some(&[0, 1]), false);
    let mut matrix_neg_inf = norm(&matrix, Some(f64::NEG_INFINITY), Some(&[0, 1]), false);

    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(
        &mut engine,
        &mut [
            &mut zero_norm,
            &mut matrix_one,
            &mut matrix_neg_one,
            &mut matrix_inf,
            &mut matrix_neg_inf,
        ],
    )
    .unwrap();

    assert_f64_eq(get_f64_data(&results[0])[0], 3.0);
    assert_f64_eq(get_f64_data(&results[1])[0], 6.0);
    assert_f64_eq(get_f64_data(&results[2])[0], 4.0);
    assert_f64_eq(get_f64_data(&results[3])[0], 7.0);
    assert_f64_eq(get_f64_data(&results[4])[0], 3.0);
}

#[test]
fn matrix_neg_inf_norm_grad_flows_to_unique_min_row() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 3],
        vec![1.0, 4.0, 2.0, -2.0, 1.0, 3.0, 1.0, -1.0, -4.0],
    ));
    let mut y = norm(&a, Some(f64::NEG_INFINITY), Some(&[0, 1]), false);
    let mut grad = y.grad(&a).unwrap();

    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut y, &mut grad]).unwrap();

    assert_f64_eq(get_f64_data(&results[0])[0], 4.0);
    assert_tensor_f64_eq(
        get_f64_data(&results[1]),
        &[1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    );
}

#[test]
fn norm_neg_inf_grad_matches_oracle_identity_case_without_keepdim() {
    let a = norm_neg_inf_oracle_input();
    let mut y = norm(&a, Some(f64::NEG_INFINITY), None, false);
    let axes: Vec<usize> = (0..y.rank).collect();
    let cotangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![], vec![1.0]));
    let scalar = (&y * &cotangent).reduce_sum(&axes);
    let mut grad = scalar.grad(&a).unwrap();

    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut y, &mut grad]).unwrap();

    assert_f64_eq(get_f64_data(&results[0])[0], 16.527183497626066);
    assert_tensor_f64_eq(
        get_f64_data(&results[1]),
        &[
            -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
}

#[test]
fn norm_neg_inf_grad_matches_oracle_identity_case_after_jvp_eval() {
    let a = norm_neg_inf_oracle_input();
    let direction = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![5, 5],
        vec![
            0.17548577202687607,
            -0.08038046416504352,
            -0.09593884869015058,
            0.09944469922788578,
            -0.00430783457831506,
            -0.17894488376803996,
            0.27160227582436686,
            0.15082825691690627,
            -0.1849797984995558,
            -0.27123041745011045,
            0.3051659515141916,
            -0.35910913963528646,
            0.29007282185641586,
            0.0532052721557607,
            0.002230017953854305,
            -0.1526866859550694,
            0.024401036746981766,
            -0.02631195518935404,
            -0.20424801159758899,
            0.10520118272166035,
            0.41011263558409017,
            -0.07375746729542286,
            0.06648016535187662,
            -0.23967231054325988,
            0.29217435382015194,
        ],
    ));
    let cotangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![], vec![1.0]));

    let y = norm(&a, Some(f64::NEG_INFINITY), None, false);
    let mut jvp = y.jvp(&a, &direction);
    let axes: Vec<usize> = (0..y.rank).collect();
    let scalar = (&y * &cotangent).reduce_sum(&axes);
    let mut grad = scalar.grad(&a).unwrap();

    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut jvp, &mut grad]).unwrap();

    assert_f64_eq(get_f64_data(&results[0])[0], -0.5591327894020484);
    assert_tensor_f64_eq(
        get_f64_data(&results[1]),
        &[
            -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
}

#[test]
fn norm_neg_inf_grad_matches_oracle_identity_case_when_only_grad_is_evaluated() {
    let a = norm_neg_inf_oracle_input();
    let cotangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![], vec![1.0]));
    let y = norm(&a, Some(f64::NEG_INFINITY), None, false);
    let axes: Vec<usize> = (0..y.rank).collect();
    let scalar = (&y * &cotangent).reduce_sum(&axes);
    let mut grad = scalar.grad(&a).unwrap();

    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut grad]).unwrap();

    assert_tensor_f64_eq(
        get_f64_data(&results[0]),
        &[
            -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
            0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
}

#[test]
fn norm_neg_inf_grad_matches_oracle_identity_case_with_keepdim() {
    let a = norm_neg_inf_oracle_input();
    let mut y = norm(&a, Some(f64::NEG_INFINITY), None, true);
    let axes: Vec<usize> = (0..y.rank).collect();
    let cotangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![1, 1], vec![-1.0]));
    let scalar = (&y * &cotangent).reduce_sum(&axes);
    let mut grad = scalar.grad(&a).unwrap();

    let mut engine = Engine::new(CpuBackend::new());
    let results = eval_all(&mut engine, &mut [&mut y, &mut grad]).unwrap();

    assert_tensor_f64_eq(get_f64_data(&results[0]), &[16.527183497626066]);
    assert_tensor_f64_eq(
        get_f64_data(&results[1]),
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
}

fn norm_neg_inf_oracle_input() -> TracedTensor {
    TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![5, 5],
        vec![
            -4.826984902407649,
            -4.146041530864057,
            5.576059452216908,
            -3.063683029231029,
            -3.8432258180800494,
            -7.582430495695129,
            -5.4215280659972755,
            -8.315684908389088,
            -3.342174545322517,
            -3.0355148286483775,
            -0.6046891126565539,
            -4.784169829877467,
            4.177597026003685,
            -5.439777184204883,
            -2.146076312776824,
            -1.5411692216498662,
            -4.150805063878843,
            2.047386382824099,
            4.480518929058965,
            -2.6718482427688133,
            -1.9719097652168658,
            7.380839390984031,
            -4.076012721760325,
            2.685009210157367,
            -7.7232222137058715,
        ],
    ))
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

#[test]
fn traced_solve_rejects_singular_matrix() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 2.0, 2.0, 4.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![1.0, 2.0]));
    let mut x = solve(&a, &b);
    let mut engine = Engine::new(CpuBackend::new());
    let err = x.eval(&mut engine).unwrap_err();
    assert!(
        err.to_string().contains("singular"),
        "expected singular matrix error, got: {err}"
    );
}

#[test]
fn traced_solve_rejects_singular_matrix_with_vector_rhs() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 2.0, 2.0, 4.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![1.0, 2.0]));
    let mut x = solve(&a, &b);
    let mut engine = Engine::new(CpuBackend::new());
    let err = x.eval(&mut engine).unwrap_err();
    assert!(
        err.to_string().contains("singular"),
        "expected singular matrix error, got: {err}"
    );
}

#[test]
fn traced_inv_rejects_singular_matrix() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 2.0, 2.0, 4.0]));
    let mut result = inv(&a);
    let mut engine = Engine::new(CpuBackend::new());
    let err = result.eval(&mut engine).unwrap_err();
    assert!(
        err.to_string().contains("singular"),
        "expected singular matrix error, got: {err}"
    );
}

#[test]
fn traced_solve_accepts_nonsingular_matrix() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![8.0, 27.0]));
    let mut x = solve(&a, &b);
    let mut engine = Engine::new(CpuBackend::new());
    let result = x.eval(&mut engine).unwrap();
    assert_tensor_f64_eq(get_f64_data(&result), &[4.0, 6.75]);
}
