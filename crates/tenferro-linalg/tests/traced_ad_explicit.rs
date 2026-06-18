#![cfg(feature = "autodiff")]

use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{DType, Error, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
use tenferro_tensor::TypedTensor;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn c64_tensor(shape: Vec<usize>, data: Vec<num_complex::Complex64>) -> Tensor {
    Tensor::C64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn get_f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn get_c64_data(tensor: &Tensor) -> &[num_complex::Complex64] {
    tensor.as_slice::<num_complex::Complex64>().unwrap()
}

fn eval(output: &TracedTensor) -> Tensor {
    eval_many(&[output]).remove(0)
}

fn eval_many(outputs: &[&TracedTensor]) -> Vec<Tensor> {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(outputs).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_linalg::register_runtime)
        .unwrap();
    executor.run_many(&program).unwrap()
}

fn reduce_all(tensor: &TracedTensor) -> TracedTensor {
    let axes: Vec<usize> = (0..tensor.rank).collect();
    tensor.reduce_sum(&axes).unwrap()
}

fn weighted_square_sum(
    tensor: &TracedTensor,
    shape: Vec<usize>,
    weights: Vec<f64>,
) -> TracedTensor {
    let weights = TracedTensor::from_tensor_concrete_shape(f64_tensor(shape, weights));
    let squared = (tensor * tensor).unwrap();
    reduce_all(&(&squared * &weights).unwrap())
}

fn assert_finite_tensor(tensor: &Tensor) {
    if let Ok(values) = tensor.as_slice::<f64>() {
        assert!(values.iter().all(|value| value.is_finite()), "{values:?}");
        return;
    }
    if let Ok(values) = tensor.as_slice::<num_complex::Complex64>() {
        assert!(
            values
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite()),
            "{values:?}"
        );
        return;
    }
    panic!("expected floating tensor, got {:?}", tensor.dtype());
}

fn assert_close_slice(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tol,
            "idx {idx}: expected {expected}, got {actual}, diff {diff}"
        );
    }
}

fn assert_close_scalar(name: &str, actual: f64, expected: f64, tol: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "{name}: expected {expected}, got {actual}, diff {diff}"
    );
}

fn assert_close_complex_scalar(
    name: &str,
    actual: num_complex::Complex64,
    expected: num_complex::Complex64,
    tol: f64,
) {
    let diff = (actual - expected).norm();
    assert!(
        diff <= tol,
        "{name}: expected {expected}, got {actual}, diff {diff}"
    );
}

fn ad_context() -> AdContext {
    AdContext::builder()
        .with_extension_rules(tenferro_linalg::ad_rules().unwrap())
        .build()
        .unwrap()
}

fn assert_invalid_ad_graph_build(
    result: std::thread::Result<tenferro_runtime::Result<TracedTensor>>,
    transform: &'static str,
    linalg_op: &str,
    detail: &str,
) {
    let err = result
        .unwrap_or_else(|_| panic!("{linalg_op} {transform} should return Err, not panic"))
        .unwrap_err();
    match err {
        Error::InvalidGraphBuild { op, message } => {
            assert_eq!(op, transform);
            assert!(
                message.contains(linalg_op),
                "expected {linalg_op} in message, got: {message}"
            );
            assert!(
                message.contains(detail),
                "expected {detail:?} in message, got: {message}"
            );
        }
        other => panic!("expected InvalidGraphBuild, got {other:?}"),
    }
}

#[test]
fn svd_singular_value_sum_jvp_uses_extension_ad_rule() {
    let ad = ad_context();
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        vec![9.0, 0.0, 0.0, 0.0, 4.0, 0.0],
    ));
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    ));

    let (_u, s, _vt) = tenferro_linalg::traced_tensor::svd(&a).unwrap();
    let y = s.reduce_sum(&[0]).unwrap();
    let dy = ad.jvp(&y, &a, &da).unwrap();
    let out = eval(&dy);

    assert_eq!(out.shape(), &[] as &[usize]);
    assert_eq!(get_f64_data(&out), &[3.0]);
}

#[test]
fn full_piv_lu_solve_grad_uses_extension_ad_rule() {
    let ad = ad_context();
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![-1.0, 5.0]));

    let x = tenferro_linalg::traced_tensor::full_piv_lu_solve(&a, &b).unwrap();
    let loss = x.reduce_sum(&[0, 1]).unwrap();
    let grad_b = ad.grad(&loss, &b).unwrap();
    let out = eval(&grad_b);

    assert_eq!(out.shape(), &[2, 1]);
}

#[test]
fn full_piv_lu_solve_grad_rejects_rank1_operands_without_panic() {
    let ad = ad_context();
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![2.0, 3.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![4.0, 9.0]));
    let x = tenferro_linalg::traced_tensor::full_piv_lu_solve(&a, &b).unwrap();
    let loss = reduce_all(&x);

    let result = catch_unwind(AssertUnwindSafe(|| ad.grad(&loss, &a)));

    assert_invalid_ad_graph_build(
        result,
        "vjp",
        "tenferro-linalg.full_piv_lu_solve",
        "rank >= 2",
    );
}

#[test]
fn triangular_solve_grad_rejects_rank1_operands_without_panic() {
    let ad = ad_context();
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![2.0, 3.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![4.0, 9.0]));
    let x =
        tenferro_linalg::traced_tensor::triangular_solve(&a, &b, true, true, false, false).unwrap();
    let loss = reduce_all(&x);

    let result = catch_unwind(AssertUnwindSafe(|| ad.grad(&loss, &a)));

    assert_invalid_ad_graph_build(
        result,
        "vjp",
        "tenferro-linalg.triangular_solve",
        "rank >= 2",
    );
}

#[test]
fn full_piv_lu_solve_jvp_rejects_non_square_lhs_without_panic() {
    let ad = ad_context();
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![4.0, 9.0]));
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
    ));
    let x = tenferro_linalg::traced_tensor::full_piv_lu_solve(&a, &b).unwrap();
    let loss = reduce_all(&x);

    let result = catch_unwind(AssertUnwindSafe(|| ad.jvp(&loss, &a, &da)));

    assert_invalid_ad_graph_build(result, "jvp", "tenferro-linalg.full_piv_lu_solve", "square");
}

#[test]
fn triangular_solve_jvp_rejects_non_square_lhs_without_panic() {
    let ad = ad_context();
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![4.0, 9.0]));
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
    ));
    let x =
        tenferro_linalg::traced_tensor::triangular_solve(&a, &b, true, true, false, false).unwrap();
    let loss = reduce_all(&x);

    let result = catch_unwind(AssertUnwindSafe(|| ad.jvp(&loss, &a, &da)));

    assert_invalid_ad_graph_build(result, "jvp", "tenferro-linalg.triangular_solve", "square");
}

#[test]
fn svd_values_grad_matches_finite_diff() {
    let ad = ad_context();
    let data = vec![3.0, 0.1, 0.2, 0.3, 2.0, 0.4];
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], data.clone()));

    let (_u, s, _vt) = tenferro_linalg::traced_tensor::svd(&a).unwrap();
    let loss = s.reduce_sum(&[0]).unwrap();
    let grad = ad.grad(&loss, &a).unwrap();
    let out = eval(&grad);

    let actual = get_f64_data(&out);
    assert_eq!(actual.len(), data.len());
    for (idx, actual_value) in actual.iter().enumerate().take(data.len()) {
        let expected = finite_diff_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], xs.to_vec()));
                let (_u, s, _vt) = tenferro_linalg::traced_tensor::svd(&input).unwrap();
                get_f64_data(&eval(&s.reduce_sum(&[0]).unwrap()))[0]
            },
            &data,
            idx,
            1.0e-6,
        );
        let diff = (*actual_value - expected).abs();
        assert!(
            diff <= 1.0e-4,
            "idx {idx}: expected {expected}, got {}, diff {diff}",
            actual_value
        );
    }
}

#[test]
fn spectral_norm_jvp_matches_finite_diff_through_values_only_svd() {
    let ad = ad_context();
    let data = vec![3.0, 0.1, 0.2, 0.3, 2.0, 0.4];
    let tangent_data = vec![0.04, -0.03, 0.05, 0.02, -0.06, 0.01];
    let matrix = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], data.clone()));
    let tangent =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], tangent_data.clone()));

    let norm =
        tenferro_linalg::traced_tensor::norm(&matrix, Some(2.0), Some(&[0, 1]), false).unwrap();
    let actual = eval(&ad.jvp(&norm, &matrix, &tangent).unwrap());

    assert_close_scalar(
        "spectral norm directional JVP",
        get_f64_data(&actual)[0],
        finite_diff_directional_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], xs.to_vec()));
                let norm =
                    tenferro_linalg::traced_tensor::norm(&input, Some(2.0), Some(&[0, 1]), false)
                        .unwrap();
                get_f64_data(&eval(&norm))[0]
            },
            &data,
            &tangent_data,
            1.0e-6,
        ),
        1.0e-4,
    );
}

#[test]
fn remaining_linalg_ops_jvp_match_finite_diff_except_full_piv_lu() {
    let ad = ad_context();

    let spd_data = vec![3.0, 0.2, 0.2, 4.0];
    let dspd_data = vec![0.3, 0.1, 0.1, -0.2];
    let spd = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], spd_data.clone()));
    let dspd = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], dspd_data.clone()));

    let general_data = vec![2.0, 0.5, 0.25, 3.0];
    let dgeneral_data = vec![0.2, 0.1, -0.1, 0.4];
    let general =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], general_data.clone()));
    let dgeneral =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], dgeneral_data.clone()));

    let rhs_data = vec![1.0, 2.0];
    let drhs_data = vec![0.4, -0.3];
    let rhs = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], rhs_data.clone()));
    let drhs = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], drhs_data.clone()));

    let chol_loss = reduce_all(&tenferro_linalg::traced_tensor::cholesky(&spd).unwrap());
    let chol_tangent = ad.jvp(&chol_loss, &spd, &dspd).unwrap();

    let (q, r) = tenferro_linalg::traced_tensor::qr(&general).unwrap();
    let qr_loss = (&reduce_all(&q) + &reduce_all(&r)).unwrap();
    let qr_tangent = ad.jvp(&qr_loss, &general, &dgeneral).unwrap();

    let (eigh_values, _eigh_vectors) = tenferro_linalg::traced_tensor::eigh(&spd).unwrap();
    let eigh_loss = reduce_all(&eigh_values);
    let eigh_tangent = ad.jvp(&eigh_loss, &spd, &dspd).unwrap();

    let (eig_values, _eig_vectors) = tenferro_linalg::traced_tensor::eig(&general).unwrap();
    let eig_loss = reduce_all(&eig_values);
    let eig_tangent = ad.jvp(&eig_loss, &general, &dgeneral).unwrap();

    let solve_loss = reduce_all(&tenferro_linalg::traced_tensor::solve(&general, &rhs).unwrap());
    let solve_tangent = ad.jvp(&solve_loss, &rhs, &drhs).unwrap();
    let triangular_loss = reduce_all(
        &tenferro_linalg::traced_tensor::triangular_solve(&general, &rhs, true, true, false, false)
            .unwrap(),
    );
    let triangular_tangent = ad.jvp(&triangular_loss, &rhs, &drhs).unwrap();

    let (_p, lu_l, lu_u, _parity) = tenferro_linalg::traced_tensor::lu(&general).unwrap();
    let lu_loss = (&reduce_all(&lu_l) + &reduce_all(&lu_u)).unwrap();
    let lu_tangent = ad.jvp(&lu_loss, &general, &dgeneral).unwrap();

    let (_p, full_lu_l, full_lu_u, _q, _full_parity) =
        tenferro_linalg::traced_tensor::full_piv_lu(&general).unwrap();
    let full_lu_loss = (&reduce_all(&full_lu_l) + &reduce_all(&full_lu_u)).unwrap();
    let full_lu_tangent = ad.jvp(&full_lu_loss, &general, &dgeneral).unwrap();

    let outputs = [
        &chol_tangent,
        &qr_tangent,
        &eigh_tangent,
        &eig_tangent,
        &solve_tangent,
        &triangular_tangent,
        &lu_tangent,
        &full_lu_tangent,
    ];
    let results = eval_many(&outputs);

    assert_close_scalar(
        "cholesky directional JVP",
        get_f64_data(&results[0])[0],
        finite_diff_directional_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()));
                let loss = reduce_all(&tenferro_linalg::traced_tensor::cholesky(&input).unwrap());
                get_f64_data(&eval(&loss))[0]
            },
            &spd_data,
            &dspd_data,
            1.0e-6,
        ),
        1.0e-4,
    );
    assert_close_scalar(
        "qr directional JVP",
        get_f64_data(&results[1])[0],
        finite_diff_directional_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()));
                let (q, r) = tenferro_linalg::traced_tensor::qr(&input).unwrap();
                let loss = (&reduce_all(&q) + &reduce_all(&r)).unwrap();
                get_f64_data(&eval(&loss))[0]
            },
            &general_data,
            &dgeneral_data,
            1.0e-6,
        ),
        1.0e-4,
    );
    assert_close_scalar(
        "eigh directional JVP",
        get_f64_data(&results[2])[0],
        finite_diff_directional_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()));
                let (values, _vectors) = tenferro_linalg::traced_tensor::eigh(&input).unwrap();
                get_f64_data(&eval(&reduce_all(&values)))[0]
            },
            &spd_data,
            &dspd_data,
            1.0e-6,
        ),
        1.0e-4,
    );
    assert_close_complex_scalar(
        "eig directional JVP",
        get_c64_data(&results[3])[0],
        finite_diff_directional_complex(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()));
                let (values, _vectors) = tenferro_linalg::traced_tensor::eig(&input).unwrap();
                get_c64_data(&eval(&reduce_all(&values)))[0]
            },
            &general_data,
            &dgeneral_data,
            1.0e-6,
        ),
        1.0e-4,
    );
    assert_close_scalar(
        "solve rhs directional JVP",
        get_f64_data(&results[4])[0],
        finite_diff_directional_scalar(
            |xs| {
                let rhs =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], xs.to_vec()));
                let loss =
                    reduce_all(&tenferro_linalg::traced_tensor::solve(&general, &rhs).unwrap());
                get_f64_data(&eval(&loss))[0]
            },
            &rhs_data,
            &drhs_data,
            1.0e-6,
        ),
        1.0e-4,
    );
    assert_close_scalar(
        "triangular_solve rhs directional JVP",
        get_f64_data(&results[5])[0],
        finite_diff_directional_scalar(
            |xs| {
                let rhs =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], xs.to_vec()));
                let loss = reduce_all(
                    &tenferro_linalg::traced_tensor::triangular_solve(
                        &general, &rhs, true, true, false, false,
                    )
                    .unwrap(),
                );
                get_f64_data(&eval(&loss))[0]
            },
            &rhs_data,
            &drhs_data,
            1.0e-6,
        ),
        1.0e-4,
    );
    assert_close_scalar(
        "lu directional JVP",
        get_f64_data(&results[6])[0],
        finite_diff_directional_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()));
                let (_p, l, u, _parity) = tenferro_linalg::traced_tensor::lu(&input).unwrap();
                let loss = (&reduce_all(&l) + &reduce_all(&u)).unwrap();
                get_f64_data(&eval(&loss))[0]
            },
            &general_data,
            &dgeneral_data,
            1.0e-6,
        ),
        1.0e-4,
    );

    assert_eq!(results[7].shape(), &[] as &[usize]);
    assert_finite_tensor(&results[7]);
}

#[test]
fn eigvalsh_jvp_matches_finite_diff_through_values_only_eigh() {
    let ad = ad_context();
    let data = vec![2.0, 0.2, 0.2, 4.0];
    let tangent_data = vec![0.1, 0.03, 0.03, -0.2];
    let matrix = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], data.clone()));
    let tangent =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], tangent_data.clone()));

    let values = tenferro_linalg::traced_tensor::eigvalsh(&matrix).unwrap();
    let loss = reduce_all(&values);
    let actual = eval(&ad.jvp(&loss, &matrix, &tangent).unwrap());

    assert_close_scalar(
        "eigvalsh directional JVP",
        get_f64_data(&actual)[0],
        finite_diff_directional_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()));
                let values = tenferro_linalg::traced_tensor::eigvalsh(&input).unwrap();
                get_f64_data(&eval(&reduce_all(&values)))[0]
            },
            &data,
            &tangent_data,
            1.0e-6,
        ),
        1.0e-4,
    );
}

#[test]
fn eigvals_jvp_matches_finite_diff() {
    let ad = ad_context();
    let data = vec![2.0, 0.5, 0.25, 3.0];
    let tangent_data = vec![0.2, 0.1, -0.1, 0.4];
    let matrix = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], data.clone()));
    let tangent =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], tangent_data.clone()));

    let values = tenferro_linalg::traced_tensor::eigvals(&matrix).unwrap();
    let loss = reduce_all(&values);
    let actual = eval(&ad.jvp(&loss, &matrix, &tangent).unwrap());

    assert_close_complex_scalar(
        "eigvals directional JVP",
        get_c64_data(&actual)[0],
        finite_diff_directional_complex(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()));
                let values = tenferro_linalg::traced_tensor::eigvals(&input).unwrap();
                get_c64_data(&eval(&reduce_all(&values)))[0]
            },
            &data,
            &tangent_data,
            1.0e-6,
        ),
        1.0e-4,
    );
}

#[test]
fn solve_matrix_operand_jvp_matches_finite_diff() {
    let ad = ad_context();

    let matrix_data = vec![2.0, 0.5, 0.25, 3.0];
    let matrix_tangent_data = vec![0.1, -0.05, 0.2, -0.1];
    let rhs_data = vec![1.0, 2.0];
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], matrix_data.clone()));
    let matrix_tangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2],
        matrix_tangent_data.clone(),
    ));
    let rhs = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], rhs_data.clone()));

    let loss = reduce_all(&tenferro_linalg::traced_tensor::solve(&matrix, &rhs).unwrap());
    let tangent = ad.jvp(&loss, &matrix, &matrix_tangent).unwrap();
    let result = eval(&tangent);

    assert_close_scalar(
        "solve matrix directional JVP",
        get_f64_data(&result)[0],
        finite_diff_directional_scalar(
            |xs| {
                let matrix =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()));
                let rhs = TracedTensor::from_tensor_concrete_shape(f64_tensor(
                    vec![2, 1],
                    rhs_data.clone(),
                ));
                let loss =
                    reduce_all(&tenferro_linalg::traced_tensor::solve(&matrix, &rhs).unwrap());
                get_f64_data(&eval(&loss))[0]
            },
            &matrix_data,
            &matrix_tangent_data,
            1.0e-6,
        ),
        1.0e-4,
    );
}

#[test]
fn triangular_solve_matrix_operand_jvp_matches_finite_diff() {
    let ad = ad_context();

    let matrix_data = vec![2.0, 0.25, 0.0, 3.0];
    let matrix_tangent_data = vec![0.1, -0.05, 0.0, -0.2];
    let rhs_data = vec![1.0, 2.0];
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], matrix_data.clone()));
    let matrix_tangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2],
        matrix_tangent_data.clone(),
    ));
    let rhs = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], rhs_data.clone()));

    let loss = reduce_all(
        &tenferro_linalg::traced_tensor::triangular_solve(&matrix, &rhs, true, true, false, false)
            .unwrap(),
    );
    let tangent = ad.jvp(&loss, &matrix, &matrix_tangent).unwrap();
    let result = eval(&tangent);

    assert_close_scalar(
        "triangular_solve matrix directional JVP",
        get_f64_data(&result)[0],
        finite_diff_directional_scalar(
            |xs| {
                let matrix =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()));
                let rhs = TracedTensor::from_tensor_concrete_shape(f64_tensor(
                    vec![2, 1],
                    rhs_data.clone(),
                ));
                let loss = reduce_all(
                    &tenferro_linalg::traced_tensor::triangular_solve(
                        &matrix, &rhs, true, true, false, false,
                    )
                    .unwrap(),
                );
                get_f64_data(&eval(&loss))[0]
            },
            &matrix_data,
            &matrix_tangent_data,
            1.0e-6,
        ),
        1.0e-4,
    );
}

#[test]
fn svd_vector_observable_jvp_matches_finite_diff() {
    let ad = ad_context();

    let matrix_data = vec![3.0, 0.1, 0.2, 0.3, 2.0, 0.4];
    let matrix_tangent_data = vec![0.05, -0.03, 0.02, 0.04, -0.06, 0.01];
    let u_weights = vec![0.7, 1.1, -0.3, 0.2, -0.5, 0.9];
    let vt_weights = vec![1.2, -0.4, 0.6, 0.8];
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], matrix_data.clone()));
    let matrix_tangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        matrix_tangent_data.clone(),
    ));

    let (u, _s, vt) = tenferro_linalg::traced_tensor::svd(&matrix).unwrap();
    let u_loss = weighted_square_sum(&u, vec![3, 2], u_weights.clone());
    let vt_loss = weighted_square_sum(&vt, vec![2, 2], vt_weights.clone());
    let loss = (&u_loss + &vt_loss).unwrap();
    let tangent = ad.jvp(&loss, &matrix, &matrix_tangent).unwrap();
    let result = eval(&tangent);

    assert_close_scalar(
        "svd vector observable directional JVP",
        get_f64_data(&result)[0],
        finite_diff_directional_scalar(
            |xs| {
                let matrix =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], xs.to_vec()));
                let (u, _s, vt) = tenferro_linalg::traced_tensor::svd(&matrix).unwrap();
                let u_loss = weighted_square_sum(&u, vec![3, 2], u_weights.clone());
                let vt_loss = weighted_square_sum(&vt, vec![2, 2], vt_weights.clone());
                let loss = (&u_loss + &vt_loss).unwrap();
                get_f64_data(&eval(&loss))[0]
            },
            &matrix_data,
            &matrix_tangent_data,
            1.0e-6,
        ),
        1.0e-4,
    );
}

#[test]
fn eigh_vector_observable_jvp_matches_finite_diff() {
    let ad = ad_context();

    let matrix_data = vec![2.0, 0.2, 0.2, 4.0];
    let matrix_tangent_data = vec![0.1, 0.03, 0.03, -0.2];
    let vector_weights = vec![0.6, -0.7, 1.3, 0.4];
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], matrix_data.clone()));
    let matrix_tangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2],
        matrix_tangent_data.clone(),
    ));

    let (_values, vectors) = tenferro_linalg::traced_tensor::eigh(&matrix).unwrap();
    let loss = weighted_square_sum(&vectors, vec![2, 2], vector_weights.clone());
    let tangent = ad.jvp(&loss, &matrix, &matrix_tangent).unwrap();
    let result = eval(&tangent);

    assert_close_scalar(
        "eigh vector observable directional JVP",
        get_f64_data(&result)[0],
        finite_diff_directional_scalar(
            |xs| {
                let matrix =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()));
                let (_values, vectors) = tenferro_linalg::traced_tensor::eigh(&matrix).unwrap();
                let loss = weighted_square_sum(&vectors, vec![2, 2], vector_weights.clone());
                get_f64_data(&eval(&loss))[0]
            },
            &matrix_data,
            &matrix_tangent_data,
            1.0e-6,
        ),
        1.0e-4,
    );
}

#[test]
fn solve_and_triangular_solve_grad_use_extension_transpose_rules() {
    let ad = ad_context();

    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![4.0, 8.0]));

    let solve_loss = reduce_all(&tenferro_linalg::traced_tensor::solve(&a, &b).unwrap());
    let solve_grad_b = ad.grad(&solve_loss, &b).unwrap();

    let triangular_loss = reduce_all(
        &tenferro_linalg::traced_tensor::triangular_solve(&a, &b, true, true, false, false)
            .unwrap(),
    );
    let triangular_grad_b = ad.grad(&triangular_loss, &b).unwrap();

    let results = eval_many(&[&solve_grad_b, &triangular_grad_b]);
    for result in &results {
        assert_eq!(result.shape(), &[2, 1]);
        assert_eq!(get_f64_data(result), &[0.5, 0.25]);
    }
}

#[test]
fn complex_eigh_values_grad_executes_with_complex_input_dtype() {
    let ad = ad_context();
    let h = TracedTensor::from_tensor_concrete_shape(c64_tensor(
        vec![2, 2],
        vec![
            num_complex::Complex64::new(3.0, 0.0),
            num_complex::Complex64::new(0.2, -0.4),
            num_complex::Complex64::new(0.2, 0.4),
            num_complex::Complex64::new(2.0, 0.0),
        ],
    ));
    let (values, _vectors) = tenferro_linalg::traced_tensor::eigh(&h).unwrap();
    let loss = values.reduce_sum(&[0]).unwrap();

    let grad = ad.grad(&loss, &h).unwrap();
    let out = eval(&grad);

    assert_eq!(out.dtype(), DType::C64);
    assert_eq!(out.shape(), &[2, 2]);
    assert_finite_tensor(&out);
}

#[test]
fn batched_solve_sum_grad_wrt_matrix_uses_native_batch_layout() {
    let ad = ad_context();

    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2, 2],
        vec![2.0, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 5.0],
    ));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 1, 2],
        vec![4.0, 8.0, 6.0, 10.0],
    ));

    let x = tenferro_linalg::traced_tensor::solve(&a, &b).unwrap();
    let loss = x.reduce_sum(&[0, 1, 2]).unwrap();
    let grad_a = ad.grad(&loss, &a).unwrap();
    let out = eval(&grad_a);

    assert_eq!(out.shape(), &[2, 2, 2]);
    assert_close_slice(
        get_f64_data(&out),
        &[-1.0, -0.5, -1.0, -0.5, -2.0 / 3.0, -0.4, -2.0 / 3.0, -0.4],
        1.0e-12,
    );
}

fn finite_diff_scalar(f: impl Fn(&[f64]) -> f64, base: &[f64], index: usize, step: f64) -> f64 {
    let mut plus = base.to_vec();
    let mut minus = base.to_vec();
    plus[index] += step;
    minus[index] -= step;
    (f(&plus) - f(&minus)) / (2.0 * step)
}

fn finite_diff_directional_scalar(
    f: impl Fn(&[f64]) -> f64,
    base: &[f64],
    tangent: &[f64],
    step: f64,
) -> f64 {
    let mut plus = base.to_vec();
    let mut minus = base.to_vec();
    for ((plus, minus), &direction) in plus.iter_mut().zip(minus.iter_mut()).zip(tangent) {
        *plus += step * direction;
        *minus -= step * direction;
    }
    (f(&plus) - f(&minus)) / (2.0 * step)
}

fn finite_diff_directional_complex(
    f: impl Fn(&[f64]) -> num_complex::Complex64,
    base: &[f64],
    tangent: &[f64],
    step: f64,
) -> num_complex::Complex64 {
    let mut plus = base.to_vec();
    let mut minus = base.to_vec();
    for ((plus, minus), &direction) in plus.iter_mut().zip(minus.iter_mut()).zip(tangent) {
        *plus += step * direction;
        *minus -= step * direction;
    }
    (f(&plus) - f(&minus)) / (2.0 * step)
}
