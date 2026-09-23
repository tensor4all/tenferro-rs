#![cfg(feature = "autodiff")]

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro_ad::AdContext;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::{DType, Error, GraphCompiler, Tensor, TracedTensor};
use tenferro_tensor::{Error as TensorError, TypedTensor};

use super::support;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn c64_tensor(shape: Vec<usize>, data: Vec<num_complex::Complex64>) -> Tensor {
    Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(shape, data).unwrap(),
    )
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
    support::run_all(&program, &[]).unwrap()
}

fn reduce_all(tensor: &TracedTensor) -> TracedTensor {
    let axes: Vec<usize> = (0..tensor.rank).collect();
    tensor.reduce_sum(Some(&axes)).unwrap()
}

fn graph_op_debugs(tensor: &TracedTensor) -> Vec<String> {
    let mut ops = Vec::new();
    collect_graph_op_debugs(tensor.graph(), &mut ops);
    ops
}

fn collect_graph_op_debugs(graph: &computegraph::graph::Graph<StdTensorOp>, ops: &mut Vec<String>) {
    for parent in graph.parents() {
        collect_graph_op_debugs(parent, ops);
    }
    ops.extend(
        graph
            .operations()
            .iter()
            .map(|node| format!("{:?}", node.operation)),
    );
}

fn weighted_square_sum(
    tensor: &TracedTensor,
    shape: Vec<usize>,
    weights: Vec<f64>,
) -> TracedTensor {
    let weights = TracedTensor::from_tensor_concrete_shape(f64_tensor(shape, weights)).unwrap();
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
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
        .unwrap()
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
        Error::Validation {
            op,
            phase: tenferro_runtime::ErrorPhase::GraphBuild,
            source: tenferro_tensor::ValidationError::InvalidArgument { message, .. },
        } => {
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
        other => panic!("expected graph-build validation, got {other:?}"),
    }
}

fn assert_traced_op_rank_mismatch<T>(
    result: std::thread::Result<tenferro_runtime::Result<T>>,
    linalg_op: &'static str,
    actual: usize,
) {
    let err = match result.unwrap_or_else(|_| panic!("{linalg_op} should return Err, not panic")) {
        Ok(_) => panic!("{linalg_op} should reject rank < 2 inputs"),
        Err(err) => err,
    };
    match err {
        Error::TensorRuntime(TensorError::Validation {
            op,
            source:
                tenferro_tensor::ValidationError::RankMismatch {
                    expected: 2,
                    actual: got,
                },
        }) => {
            assert_eq!(op, linalg_op);
            assert_eq!(got, actual);
        }
        other => panic!("expected RankMismatch for {linalg_op}, got {other:?}"),
    }
}

#[test]
fn svd_singular_value_sum_jvp_uses_extension_ad_rule() {
    let ad = ad_context();
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        vec![9.0, 0.0, 0.0, 0.0, 4.0, 0.0],
    ))
    .unwrap();
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    ))
    .unwrap();

    let (_u, s, _vt) = a.svd().unwrap();
    let y = s.reduce_sum(Some(&[0])).unwrap();
    let dy = ad.jvp(&y, &a, &da).unwrap();
    let out = eval(&dy);

    assert_eq!(out.shape(), &[] as &[usize]);
    assert_eq!(get_f64_data(&out), &[3.0]);
}

#[test]
fn full_piv_lu_solve_grad_uses_extension_ad_rule() {
    let ad = ad_context();
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]))
            .unwrap();
    let b =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![-1.0, 5.0])).unwrap();

    let x = a.full_piv_lu_solve(&b).unwrap();
    let loss = x.reduce_sum(Some(&[0, 1])).unwrap();
    let grad_b = ad.grad(&loss, &b).unwrap();
    let out = eval(&grad_b);

    assert_eq!(out.shape(), &[2, 1]);
}

#[test]
fn full_piv_lu_solve_grad_rejects_rank1_operands_without_panic() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![2.0, 3.0])).unwrap();
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![4.0, 9.0])).unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| a.full_piv_lu_solve(&b)));

    assert_traced_op_rank_mismatch(result, "tenferro-linalg.full_piv_lu_solve", 1);
}

#[test]
fn triangular_solve_grad_rejects_rank1_operands_without_panic() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![2.0, 3.0])).unwrap();
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![4.0, 9.0])).unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| {
        a.triangular_solve(&b, true, true, false, false)
    }));

    assert_traced_op_rank_mismatch(result, "tenferro-linalg.triangular_solve", 1);
}

#[test]
fn full_piv_lu_solve_jvp_rejects_non_square_lhs_without_panic() {
    let ad = ad_context();
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ))
    .unwrap();
    let b =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![4.0, 9.0])).unwrap();
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
    ))
    .unwrap();
    let x = a.full_piv_lu_solve(&b).unwrap();
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
    ))
    .unwrap();
    let b =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![4.0, 9.0])).unwrap();
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![0.1, 0.0, 0.0, 0.0, 0.1, 0.0],
    ))
    .unwrap();
    let x = a.triangular_solve(&b, true, true, false, false).unwrap();
    let loss = reduce_all(&x);

    let result = catch_unwind(AssertUnwindSafe(|| ad.jvp(&loss, &a, &da)));

    assert_invalid_ad_graph_build(result, "jvp", "tenferro-linalg.triangular_solve", "square");
}

#[test]
fn square_lu_jvp_does_not_materialize_solve_input_augmentations() {
    let ad = ad_context();
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.5, 0.25, 3.0]))
            .unwrap();
    let tangent =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![0.2, 0.1, -0.1, 0.4]))
            .unwrap();
    let (_p, l, u, _parity) = matrix.lu().unwrap();
    let loss = (&reduce_all(&l) + &reduce_all(&u)).unwrap();
    let jvp = ad.jvp(&loss, &matrix, &tangent).unwrap();

    let op_debugs = graph_op_debugs(&jvp);
    assert!(
        !op_debugs.iter().any(|op| op.contains("EmbedDiag")),
        "square LU JVP should not build a dense identity for triangular solve inputs: {op_debugs:#?}"
    );
    assert!(
        !op_debugs.iter().any(|op| op.contains("BroadcastInDim")),
        "square LU JVP should not broadcast identity diagonals: {op_debugs:#?}"
    );
    assert_eq!(
        op_debugs
            .iter()
            .filter(|op| op.contains("Tril { k: -1 }"))
            .count(),
        1,
        "square LU JVP should keep only the output-forming strict-lower mask: {op_debugs:#?}"
    );
    assert_eq!(
        op_debugs
            .iter()
            .filter(|op| op.contains("Triu { k: 0 }"))
            .count(),
        1,
        "square LU JVP should keep only the output-forming upper mask: {op_debugs:#?}"
    );
}

#[test]
fn svd_values_grad_matches_finite_diff() {
    let ad = ad_context();
    let data = vec![3.0, 0.1, 0.2, 0.3, 2.0, 0.4];
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], data.clone())).unwrap();

    let (_u, s, _vt) = a.svd().unwrap();
    let loss = s.reduce_sum(Some(&[0])).unwrap();
    let grad = ad.grad(&loss, &a).unwrap();
    let out = eval(&grad);

    let actual = get_f64_data(&out);
    assert_eq!(actual.len(), data.len());
    for (idx, actual_value) in actual.iter().enumerate().take(data.len()) {
        let expected = finite_diff_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], xs.to_vec()))
                        .unwrap();
                let (_u, s, _vt) = input.svd().unwrap();
                get_f64_data(&eval(&s.reduce_sum(Some(&[0])).unwrap()))[0]
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
fn complex_svd_values_grad_executes_with_complex_input_dtype() {
    let ad = ad_context();
    let a = TracedTensor::from_tensor_concrete_shape(c64_tensor(
        vec![2, 2],
        vec![
            num_complex::Complex64::new(3.0, 0.0),
            num_complex::Complex64::new(0.2, -0.4),
            num_complex::Complex64::new(0.2, 0.4),
            num_complex::Complex64::new(2.0, 0.0),
        ],
    ))
    .unwrap();
    let (_u, values, _vt) = a.svd().unwrap();
    let loss = values.reduce_sum(Some(&[0])).unwrap();

    let grad = ad.grad(&loss, &a).unwrap();
    let out = eval(&grad);

    assert_eq!(out.dtype(), DType::C64);
    assert_eq!(out.shape(), &[2, 2]);
    assert_finite_tensor(&out);
}

#[test]
fn frobenius_norm_jvp_matches_finite_diff_through_fused_sum_squares() {
    let ad = ad_context();
    let data = vec![3.0, 0.1, 0.2, 0.3, 2.0, 0.4];
    let tangent_data = vec![0.04, -0.03, 0.05, 0.02, -0.06, 0.01];
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], data.clone())).unwrap();
    let tangent =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], tangent_data.clone()))
            .unwrap();

    let norm = matrix.norm(None, Some(&[0, 1]), false).unwrap();
    assert!(
        graph_op_debugs(&norm)
            .iter()
            .any(|op| op.contains("ReduceSumSquares")),
        "Frobenius norm should retain the fused semantic primitive"
    );
    let jvp = ad.jvp(&norm, &matrix, &tangent).unwrap();

    assert_close_scalar(
        "Frobenius norm directional JVP",
        get_f64_data(&eval(&jvp))[0],
        finite_diff_directional_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], xs.to_vec()))
                        .unwrap();
                get_f64_data(&eval(&input.norm(None, Some(&[0, 1]), false).unwrap()))[0]
            },
            &data,
            &tangent_data,
            1.0e-6,
        ),
        1.0e-6,
    );
}

#[test]
fn frobenius_norm_grad_matches_finite_diff_through_fused_sum_squares() {
    let ad = ad_context();
    let data = vec![3.0, 0.1, 0.2, 0.3, 2.0, 0.4];
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], data.clone())).unwrap();
    let norm = matrix.norm(None, Some(&[0, 1]), false).unwrap();
    let actual = eval(&ad.grad(&norm, &matrix).unwrap());

    for (index, &value) in get_f64_data(&actual).iter().enumerate() {
        let expected = finite_diff_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], xs.to_vec()))
                        .unwrap();
                get_f64_data(&eval(&input.norm(None, Some(&[0, 1]), false).unwrap()))[0]
            },
            &data,
            index,
            1.0e-6,
        );
        assert_close_scalar("Frobenius norm gradient", value, expected, 1.0e-6);
    }
}

#[test]
fn spectral_norm_jvp_matches_finite_diff_through_values_only_svd() {
    let ad = ad_context();
    let data = vec![3.0, 0.1, 0.2, 0.3, 2.0, 0.4];
    let tangent_data = vec![0.04, -0.03, 0.05, 0.02, -0.06, 0.01];
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], data.clone())).unwrap();
    let tangent =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], tangent_data.clone()))
            .unwrap();

    let norm = matrix.norm(Some(2.0), Some(&[0, 1]), false).unwrap();
    let primal_op_debugs = graph_op_debugs(&norm);
    assert!(
        primal_op_debugs.iter().any(|op| op.contains("Svd {")),
        "spectral norm primal graph should carry full SVD residuals for AD: {primal_op_debugs:#?}"
    );
    assert!(
        !primal_op_debugs.iter().any(|op| op.contains("SvdVals")),
        "spectral norm primal graph should not use values-only SVD before compile-time pruning: {primal_op_debugs:#?}"
    );
    let jvp = ad.jvp(&norm, &matrix, &tangent).unwrap();
    let actual = eval(&jvp);

    assert_close_scalar(
        "spectral norm directional JVP",
        get_f64_data(&actual)[0],
        finite_diff_directional_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], xs.to_vec()))
                        .unwrap();
                let norm = input.norm(Some(2.0), Some(&[0, 1]), false).unwrap();
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
    let spd =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], spd_data.clone())).unwrap();
    let dspd = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], dspd_data.clone()))
        .unwrap();

    let general_data = vec![2.0, 0.5, 0.25, 3.0];
    let dgeneral_data = vec![0.2, 0.1, -0.1, 0.4];
    let general =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], general_data.clone()))
            .unwrap();
    let dgeneral =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], dgeneral_data.clone()))
            .unwrap();

    let rhs_data = vec![1.0, 2.0];
    let drhs_data = vec![0.4, -0.3];
    let rhs =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], rhs_data.clone())).unwrap();
    let drhs = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], drhs_data.clone()))
        .unwrap();

    let chol_loss = reduce_all(&spd.cholesky().unwrap());
    let chol_tangent = ad.jvp(&chol_loss, &spd, &dspd).unwrap();

    let (q, r) = general.qr().unwrap();
    let qr_loss = (&reduce_all(&q) + &reduce_all(&r)).unwrap();
    let qr_tangent = ad.jvp(&qr_loss, &general, &dgeneral).unwrap();

    let (eigh_values, _eigh_vectors) = spd.eigh().unwrap();
    let eigh_loss = reduce_all(&eigh_values);
    let eigh_tangent = ad.jvp(&eigh_loss, &spd, &dspd).unwrap();

    let (eig_values, _eig_vectors) = general.eig().unwrap();
    let eig_loss = reduce_all(&eig_values);
    let eig_tangent = ad.jvp(&eig_loss, &general, &dgeneral).unwrap();

    let solve_loss = reduce_all(&general.solve(&rhs).unwrap());
    let solve_tangent = ad.jvp(&solve_loss, &rhs, &drhs).unwrap();
    let triangular_loss = reduce_all(
        &general
            .triangular_solve(&rhs, true, true, false, false)
            .unwrap(),
    );
    let triangular_tangent = ad.jvp(&triangular_loss, &rhs, &drhs).unwrap();

    let (_p, lu_l, lu_u, _parity) = general.lu().unwrap();
    let lu_loss = (&reduce_all(&lu_l) + &reduce_all(&lu_u)).unwrap();
    let lu_tangent = ad.jvp(&lu_loss, &general, &dgeneral).unwrap();

    let (_p, full_lu_l, full_lu_u, _q, _full_parity) = general.full_piv_lu().unwrap();
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
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()))
                        .unwrap();
                let loss = reduce_all(&input.cholesky().unwrap());
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
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()))
                        .unwrap();
                let (q, r) = input.qr().unwrap();
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
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()))
                        .unwrap();
                let (values, _vectors) = input.eigh().unwrap();
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
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()))
                        .unwrap();
                let (values, _vectors) = input.eig().unwrap();
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
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], xs.to_vec()))
                        .unwrap();
                let loss = reduce_all(&general.solve(&rhs).unwrap());
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
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], xs.to_vec()))
                        .unwrap();
                let loss = reduce_all(
                    &general
                        .triangular_solve(&rhs, true, true, false, false)
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
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()))
                        .unwrap();
                let (_p, l, u, _parity) = input.lu().unwrap();
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
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], data.clone())).unwrap();
    let tangent =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], tangent_data.clone()))
            .unwrap();

    let values = matrix.eigvalsh().unwrap();
    let loss = reduce_all(&values);
    let actual = eval(&ad.jvp(&loss, &matrix, &tangent).unwrap());

    assert_close_scalar(
        "eigvalsh directional JVP",
        get_f64_data(&actual)[0],
        finite_diff_directional_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()))
                        .unwrap();
                let values = input.eigvalsh().unwrap();
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
fn complex_hermitian_eigvalsh_jvp_matches_the_trace_of_the_tangent() {
    // Regression: the eigenvalue linear rule projects the tangent with
    // `diag(V^H dA V)`, whose Hadamard form needs `conj(V^H)` and not `V^H`.
    // A real input cannot see the difference, so this case has to be complex.
    // The reference is exact for a Hermitian matrix: d(sum of eigenvalues) is
    // the trace of the self-adjoint tangent.
    let ad = ad_context();
    let base = vec![
        num_complex::Complex64::new(3.0, 0.0),
        num_complex::Complex64::new(0.2, 0.4),
        num_complex::Complex64::new(0.2, -0.4),
        num_complex::Complex64::new(-1.0, 0.0),
    ];
    let tangent = vec![
        num_complex::Complex64::new(0.7, 0.0),
        num_complex::Complex64::new(-0.3, -0.5),
        num_complex::Complex64::new(-0.3, 0.5),
        num_complex::Complex64::new(-0.3, 0.0),
    ];
    let matrix =
        TracedTensor::from_tensor_concrete_shape(c64_tensor(vec![2, 2], base.clone())).unwrap();
    let direction =
        TracedTensor::from_tensor_concrete_shape(c64_tensor(vec![2, 2], tangent.clone())).unwrap();

    let values = matrix.eigvalsh().unwrap();
    let loss = reduce_all(&values);
    let actual = eval(&ad.jvp(&loss, &matrix, &direction).unwrap());

    assert_close_scalar(
        "complex Hermitian eigvalsh directional JVP",
        get_f64_data(&actual)[0],
        tangent[0].re + tangent[3].re,
        1.0e-10,
    );

    // The same JVP must stay real for a purely imaginary Hermitian tangent:
    // `tr(dA)` is real, so the imaginary part of the tangent may not leak in.
    let imaginary = vec![
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(0.9, 0.0),
        num_complex::Complex64::new(0.9, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
    ];
    let direction =
        TracedTensor::from_tensor_concrete_shape(c64_tensor(vec![2, 2], imaginary)).unwrap();
    let actual = eval(&ad.jvp(&loss, &matrix, &direction).unwrap());
    assert_close_scalar(
        "complex Hermitian eigvalsh JVP with a zero-trace tangent",
        get_f64_data(&actual)[0],
        0.0,
        1.0e-10,
    );
}

#[test]
fn eigvals_jvp_matches_finite_diff() {
    let ad = ad_context();
    let data = vec![2.0, 0.5, 0.25, 3.0];
    let tangent_data = vec![0.2, 0.1, -0.1, 0.4];
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], data.clone())).unwrap();
    let tangent =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], tangent_data.clone()))
            .unwrap();

    let values = matrix.eigvals().unwrap();
    let loss = reduce_all(&values);
    let actual = eval(&ad.jvp(&loss, &matrix, &tangent).unwrap());

    assert_close_complex_scalar(
        "eigvals directional JVP",
        get_c64_data(&actual)[0],
        finite_diff_directional_complex(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()))
                        .unwrap();
                let values = input.eigvals().unwrap();
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
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], matrix_data.clone()))
            .unwrap();
    let matrix_tangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2],
        matrix_tangent_data.clone(),
    ))
    .unwrap();
    let rhs =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], rhs_data.clone())).unwrap();

    let loss = reduce_all(&matrix.solve(&rhs).unwrap());
    let tangent = ad.jvp(&loss, &matrix, &matrix_tangent).unwrap();
    let result = eval(&tangent);

    assert_close_scalar(
        "solve matrix directional JVP",
        get_f64_data(&result)[0],
        finite_diff_directional_scalar(
            |xs| {
                let matrix =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()))
                        .unwrap();
                let rhs = TracedTensor::from_tensor_concrete_shape(f64_tensor(
                    vec![2, 1],
                    rhs_data.clone(),
                ))
                .unwrap();
                let loss = reduce_all(&matrix.solve(&rhs).unwrap());
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
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], matrix_data.clone()))
            .unwrap();
    let matrix_tangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2],
        matrix_tangent_data.clone(),
    ))
    .unwrap();
    let rhs =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], rhs_data.clone())).unwrap();

    let loss = reduce_all(
        &matrix
            .triangular_solve(&rhs, true, true, false, false)
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
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()))
                        .unwrap();
                let rhs = TracedTensor::from_tensor_concrete_shape(f64_tensor(
                    vec![2, 1],
                    rhs_data.clone(),
                ))
                .unwrap();
                let loss = reduce_all(
                    &matrix
                        .triangular_solve(&rhs, true, true, false, false)
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
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], matrix_data.clone()))
            .unwrap();
    let matrix_tangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        matrix_tangent_data.clone(),
    ))
    .unwrap();

    let (u, _s, vt) = matrix.svd().unwrap();
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
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], xs.to_vec()))
                        .unwrap();
                let (u, _s, vt) = matrix.svd().unwrap();
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
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], matrix_data.clone()))
            .unwrap();
    let matrix_tangent = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2],
        matrix_tangent_data.clone(),
    ))
    .unwrap();

    let (_values, vectors) = matrix.eigh().unwrap();
    let loss = weighted_square_sum(&vectors, vec![2, 2], vector_weights.clone());
    let tangent = ad.jvp(&loss, &matrix, &matrix_tangent).unwrap();
    let result = eval(&tangent);

    assert_close_scalar(
        "eigh vector observable directional JVP",
        get_f64_data(&result)[0],
        finite_diff_directional_scalar(
            |xs| {
                let matrix =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()))
                        .unwrap();
                let (_values, vectors) = matrix.eigh().unwrap();
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
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0]))
            .unwrap();
    let b =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![4.0, 8.0])).unwrap();

    let solve_loss = reduce_all(&a.solve(&b).unwrap());
    let solve_grad_b = ad.grad(&solve_loss, &b).unwrap();

    let triangular_loss = reduce_all(&a.triangular_solve(&b, true, true, false, false).unwrap());
    let triangular_grad_b = ad.grad(&triangular_loss, &b).unwrap();

    let results = eval_many(&[&solve_grad_b, &triangular_grad_b]);
    for result in &results {
        assert_eq!(result.shape(), &[2, 1]);
        assert_eq!(get_f64_data(result), &[0.5, 0.25]);
    }
}

fn lu_sum_loss(input: &TracedTensor) -> TracedTensor {
    let (_p, l, u, _parity) = input.lu().unwrap();
    (&reduce_all(&l) + &reduce_all(&u)).unwrap()
}

fn qr_sum_loss(input: &TracedTensor) -> TracedTensor {
    let (q, r) = input.qr().unwrap();
    (&reduce_all(&q) + &reduce_all(&r)).unwrap()
}

fn eigh_sum_loss(input: &TracedTensor) -> TracedTensor {
    let (values, vectors) = input.eigh().unwrap();
    (&reduce_all(&values) + &reduce_all(&vectors)).unwrap()
}

fn eigvalsh_sum_loss(input: &TracedTensor) -> TracedTensor {
    reduce_all(&input.eigvalsh().unwrap())
}

#[derive(Debug)]
struct GraphOpSummary {
    counts: BTreeMap<String, usize>,
    multi_output_ops: usize,
}

#[derive(Debug)]
struct CompiledProgramSummary {
    input_count: usize,
    instruction_count: usize,
    counts: BTreeMap<String, usize>,
}

fn compiled_program_summary(output: &TracedTensor) -> CompiledProgramSummary {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(output).unwrap();
    let mut counts = BTreeMap::new();
    for operation in program.program().operations() {
        let label = match operation.op() {
            tenferro_runtime::program::SemanticOpRef::Core(op) => format!("{op:?}")
                .split([' ', '{', '('])
                .next()
                .unwrap_or("Core")
                .to_string(),
            tenferro_runtime::program::SemanticOpRef::Extension(_) => "Extension".to_string(),
            _ => "Unknown".to_string(),
        };
        *counts.entry(label).or_insert(0) += 1;
    }
    let instruction_count = program.program().operations().len();
    CompiledProgramSummary {
        input_count: program.input_count(),
        instruction_count,
        counts,
    }
}

fn graph_op_summary(output: &TracedTensor) -> GraphOpSummary {
    let mut counts = BTreeMap::new();
    let mut multi_output_ops = 0;
    for node in output.graph().operations() {
        *counts.entry(op_label(&node.operation)).or_insert(0) += 1;
        if node.outputs.len() > 1 {
            multi_output_ops += 1;
        }
    }
    GraphOpSummary {
        counts,
        multi_output_ops,
    }
}

fn expected_counts(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|(name, count)| ((*name).to_string(), *count))
        .collect()
}

fn op_label(op: &StdTensorOp) -> String {
    match op {
        StdTensorOp::Extension(ext) => {
            let debug = format!("{ext:?}");
            for linalg_op in [
                "LuFactorSolve",
                "Lu",
                "Qr",
                "TriangularSolve",
                "LuSolvePrepared",
                "FullPivLuSolve",
                "Eigh",
                "EighVals",
                "Svd",
                "SvdVals",
                "Eig",
                "EigVals",
                "Cholesky",
                "FullPivLu",
                "LuFactor",
            ] {
                if debug.contains(&format!("op: {linalg_op}")) {
                    return format!("Extension({linalg_op})");
                }
            }
            format!("Extension({debug})")
        }
        other => {
            let debug = format!("{other:?}");
            debug
                .split([' ', '{', '('])
                .next()
                .unwrap_or(debug.as_str())
                .to_string()
        }
    }
}

fn assert_gradient_matches_finite_difference(
    name: &str,
    shape: Vec<usize>,
    data: Vec<f64>,
    loss: fn(&TracedTensor) -> TracedTensor,
) {
    let ad = ad_context();
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(shape.clone(), data.clone())).unwrap();
    let grad = ad.grad(&loss(&matrix), &matrix).unwrap();
    let result = eval(&grad);
    let actual = get_f64_data(&result);

    assert_eq!(result.shape(), shape.as_slice(), "{name} gradient shape");
    assert_eq!(actual.len(), data.len(), "{name} gradient length");
    for (idx, actual_value) in actual.iter().enumerate() {
        let expected = finite_diff_scalar(
            |xs| {
                let input = TracedTensor::from_tensor_concrete_shape(f64_tensor(
                    shape.clone(),
                    xs.to_vec(),
                ))
                .unwrap();
                get_f64_data(&eval(&loss(&input)))[0]
            },
            &data,
            idx,
            1.0e-6,
        );
        let diff = (*actual_value - expected).abs();
        assert!(
            diff <= 1.0e-4,
            "{name} idx {idx}: expected {expected}, got {}, diff {diff}",
            actual_value
        );
    }
}

#[test]
fn lu_sum_grad_compiled_program_does_not_retain_tangent_sweep() {
    let ad = ad_context();
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.5, 0.25, 3.0]))
            .unwrap();
    let grad = ad.grad(&lu_sum_loss(&matrix), &matrix).unwrap();
    let summary = compiled_program_summary(&grad);

    assert_eq!(
        summary.input_count, 2,
        "compiled LU VJP should bind only the primal input and cotangent: {summary:?}"
    );
    assert_eq!(
        summary.counts.get("DotGeneral").copied().unwrap_or(0),
        3,
        "compiled LU VJP should not retain the zero tangent matmul chain: {summary:?}"
    );
    assert_eq!(
        summary.counts.get("Extension").copied().unwrap_or(0),
        3,
        "compiled LU VJP should contain primal LU plus two triangular-solve pullback ops: {summary:?}"
    );
    assert!(
        summary.instruction_count <= 21,
        "compiled LU VJP should stay near the direct transpose baseline: {summary:?}"
    );
}

#[test]
fn eigvalsh_sum_grad_compiled_program_does_not_retain_tangent_sweep() {
    let ad = ad_context();
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.2, 0.2, 4.0]))
            .unwrap();
    let grad = ad.grad(&eigvalsh_sum_loss(&matrix), &matrix).unwrap();
    let summary = compiled_program_summary(&grad);

    assert_eq!(
        summary.input_count, 2,
        "compiled eigvalsh VJP should bind only the primal input and cotangent: {summary:?}"
    );
    assert_eq!(
        summary.counts.get("DotGeneral").copied().unwrap_or(0),
        1,
        "compiled eigvalsh VJP should pull the eigenvalue coefficient back with one \
matmul and a Hadamard scale, not two matmuls: {summary:?}"
    );
    assert_eq!(
        summary.counts.get("Mul").copied().unwrap_or(0),
        1,
        "compiled eigvalsh VJP should scale by the eigenvector Hadamard factor once: {summary:?}"
    );
    assert!(
        summary.counts.get("Extension").copied().unwrap_or(0) <= 2,
        "compiled eigvalsh VJP should not retain extra extension ops from the zero tangent chain: {summary:?}"
    );
    assert!(
        summary.instruction_count <= 13,
        "compiled eigvalsh VJP should stay compact after linearize+transpose: {summary:?}"
    );
}

#[test]
fn lu_sum_grad_optimized_graph_is_structurally_compact() {
    let ad = ad_context();
    for (name, shape, data, _expected) in [
        (
            "square",
            vec![2, 2],
            vec![2.0, 0.5, 0.25, 3.0],
            expected_counts(&[
                ("Add", 1),
                ("BroadcastInDim", 2),
                ("DotGeneral", 3),
                ("Extension(TriangularSolve)", 2),
                ("Reshape", 2),
                ("Tril", 1),
                ("Triu", 1),
            ]),
        ),
        (
            "wide",
            vec![2, 3],
            vec![2.0, 0.5, 0.25, 3.0, 1.0, -0.4],
            expected_counts(&[
                ("Add", 1),
                ("BroadcastInDim", 2),
                ("DotGeneral", 4),
                ("Extension(TriangularSolve)", 2),
                ("Reshape", 2),
                ("Tril", 1),
                ("Triu", 1),
            ]),
        ),
        (
            "tall",
            vec![3, 2],
            vec![2.0, 0.5, 0.1, 0.25, 3.0, -0.4],
            expected_counts(&[
                ("Add", 1),
                ("BroadcastInDim", 2),
                ("DotGeneral", 4),
                ("Extension(TriangularSolve)", 2),
                ("Reshape", 2),
                ("Tril", 1),
                ("Triu", 1),
            ]),
        ),
    ] {
        let matrix = TracedTensor::from_tensor_concrete_shape(f64_tensor(shape, data)).unwrap();
        let grad = ad.grad(&lu_sum_loss(&matrix), &matrix).unwrap();
        let summary = graph_op_summary(&grad);

        assert!(
            summary.counts.get("Extension(Lu)").copied().unwrap_or(0) <= 1,
            "{name}: semantic VJP should keep at most one primal LU residual carrier: {summary:?}"
        );
        assert!(
            summary.multi_output_ops <= 1,
            "{name}: semantic VJP should bound uncompiled multi-output residual carriers: {summary:?}"
        );
        assert_eq!(
            summary
                .counts
                .get("Extension(TriangularSolve)")
                .copied()
                .unwrap_or(0),
            2,
            "{name}: LU pullback should still use the two triangular-solve structure"
        );
    }
}

#[test]
fn eigvalsh_sum_grad_optimized_graph_is_structurally_compact() {
    let ad = ad_context();
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.2, 0.2, 4.0]))
            .unwrap();
    let grad = ad.grad(&eigvalsh_sum_loss(&matrix), &matrix).unwrap();
    let summary = graph_op_summary(&grad);

    assert_eq!(
        summary
            .counts
            .get("Extension(EighVals)")
            .copied()
            .unwrap_or(0),
        0,
        "optimized generic VJP should not keep a values-only Eigh transpose carrier"
    );
    assert!(
        summary.counts.get("Extension(Eigh)").copied().unwrap_or(0) <= 2,
        "semantic VJP should bound full Eigh residual carriers before compile pruning: {summary:?}"
    );
    assert!(
        summary.multi_output_ops <= 2,
        "semantic VJP should bound uncompiled multi-output Eigh residual carriers: {summary:?}"
    );
    let expected = expected_counts(&[
        ("Add", 2),
        ("BroadcastInDim", 2),
        ("DotGeneral", 1),
        ("EmbedDiag", 1),
        ("ExtractDiag", 1),
        ("Mul", 1),
        ("Transpose", 3),
        ("Tril", 1),
    ]);
    assert_eq!(
        summary.counts.get("DotGeneral").copied().unwrap_or(0),
        expected.get("DotGeneral").copied().unwrap(),
        "eigvalsh VJP should pull the eigenvalue coefficient back with one matmul \
and a Hadamard scale instead of two matmuls"
    );
    assert_eq!(
        summary.counts.get("Mul").copied().unwrap_or(0),
        expected.get("Mul").copied().unwrap(),
        "eigvalsh VJP should apply the eigenvector Hadamard factor exactly once"
    );
}

#[test]
fn full_eigh_sum_grad_optimized_graph_is_structurally_compact() {
    let ad = ad_context();
    let matrix =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.2, 0.2, 4.0]))
            .unwrap();
    let grad = ad.grad(&eigh_sum_loss(&matrix), &matrix).unwrap();
    let summary = graph_op_summary(&grad);

    assert!(
        summary.counts.get("Extension(Eigh)").copied().unwrap_or(0) <= 1,
        "semantic VJP should keep at most one full Eigh residual carrier before compile pruning: {summary:?}"
    );
    assert!(
        summary.multi_output_ops <= 1,
        "semantic VJP should bound uncompiled multi-output Eigh residual carriers: {summary:?}"
    );
    assert_eq!(
        summary.counts.get("DotGeneral").copied().unwrap_or(0),
        5,
        "full Eigh semantic VJP currently keeps source and pullback matmuls before compile pruning"
    );
}

#[test]
fn qr_sum_grad_optimized_graph_is_structurally_compact() {
    let ad = ad_context();
    for (name, shape, data, expected) in [
        (
            "square",
            vec![2, 2],
            vec![2.0, 0.5, 0.25, 3.0],
            expected_counts(&[
                ("Add", 4),
                ("BroadcastInDim", 2),
                ("DotGeneral", 3),
                ("EmbedDiag", 1),
                ("Extension(TriangularSolve)", 1),
                ("ExtractDiag", 1),
                ("Mul", 1),
                ("Neg", 1),
                ("Reshape", 2),
                ("Transpose", 3),
                ("Triu", 1),
            ]),
        ),
        (
            "wide",
            vec![2, 3],
            vec![2.0, 0.5, 0.25, 3.0, 1.0, -0.4],
            expected_counts(&[
                ("Add", 7),
                ("BroadcastInDim", 2),
                ("DotGeneral", 9),
                ("EmbedDiag", 1),
                ("Extension(TriangularSolve)", 1),
                ("ExtractDiag", 1),
                ("Mul", 1),
                ("Neg", 3),
                ("Reshape", 2),
                ("Transpose", 3),
                ("Triu", 1),
            ]),
        ),
        (
            "tall",
            vec![3, 2],
            vec![2.0, 0.5, 0.1, 0.25, 3.0, -0.4],
            expected_counts(&[
                ("Add", 4),
                ("BroadcastInDim", 2),
                ("DotGeneral", 3),
                ("EmbedDiag", 1),
                ("Extension(TriangularSolve)", 1),
                ("ExtractDiag", 1),
                ("Mul", 1),
                ("Neg", 1),
                ("Reshape", 2),
                ("Transpose", 3),
                ("Triu", 1),
            ]),
        ),
    ] {
        let matrix = TracedTensor::from_tensor_concrete_shape(f64_tensor(shape, data)).unwrap();
        let grad = ad.grad(&qr_sum_loss(&matrix), &matrix).unwrap();
        let summary = graph_op_summary(&grad);

        assert!(
            summary.counts.get("Extension(Qr)").copied().unwrap_or(0) <= 1,
            "{name}: semantic VJP should keep at most one QR residual carrier before compile pruning: {summary:?}"
        );
        assert!(
            summary.multi_output_ops <= 1,
            "{name}: semantic VJP should bound uncompiled multi-output QR residual carriers: {summary:?}"
        );
        assert_eq!(
            summary
                .counts
                .get("Extension(TriangularSolve)")
                .copied()
                .unwrap_or(0),
            expected
                .get("Extension(TriangularSolve)")
                .copied()
                .unwrap_or(0),
            "{name}: QR pullback should keep the triangular solve count"
        );
    }
}

#[test]
fn lu_qr_sum_grads_match_finite_diff() {
    assert_gradient_matches_finite_difference(
        "lu square sum gradient",
        vec![2, 2],
        vec![2.0, 0.5, 0.25, 3.0],
        lu_sum_loss,
    );
    assert_gradient_matches_finite_difference(
        "lu wide sum gradient",
        vec![2, 3],
        vec![2.0, 0.5, 0.25, 3.0, 1.0, -0.4],
        lu_sum_loss,
    );
    assert_gradient_matches_finite_difference(
        "lu tall sum gradient",
        vec![3, 2],
        vec![2.0, 0.5, 0.1, 0.25, 3.0, -0.4],
        lu_sum_loss,
    );
    assert_gradient_matches_finite_difference(
        "qr full-rank sum gradient",
        vec![3, 2],
        vec![2.0, 0.5, 0.1, 0.25, 3.0, -0.4],
        qr_sum_loss,
    );
    assert_gradient_matches_finite_difference(
        "qr wide sum gradient",
        vec![2, 3],
        vec![2.0, 0.5, 0.25, 3.0, 1.0, -0.4],
        qr_sum_loss,
    );
}

#[test]
fn full_eigh_sum_grad_matches_finite_diff() {
    assert_gradient_matches_finite_difference(
        "full eigh sum gradient",
        vec![2, 2],
        vec![2.0, 0.2, 0.2, 4.0],
        eigh_sum_loss,
    );
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
    ))
    .unwrap();
    let (values, _vectors) = h.eigh().unwrap();
    let loss = values.reduce_sum(Some(&[0])).unwrap();

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
    ))
    .unwrap();
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 1, 2],
        vec![4.0, 8.0, 6.0, 10.0],
    ))
    .unwrap();

    let x = a.solve(&b).unwrap();
    let loss = x.reduce_sum(Some(&[0, 1, 2])).unwrap();
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

/// Batched 3x3 systems whose dominant cyclic subdiagonal forces row pivoting.
fn pivoting_solve_fixture() -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![
        0.3, -0.2, 0.1, 0.2, 0.4, -0.3, -0.1, 0.25, 0.15, //
        -0.35, 0.05, 0.2, 0.1, -0.15, 0.3, 0.2, 0.1, -0.25,
    ];
    for k in 0..2 {
        for col in 0..3 {
            a[k * 9 + (col + 1) % 3 + col * 3] += 4.0;
        }
    }
    let b = vec![
        1.0, -2.0, 0.5, 0.25, 1.5, -1.0, -0.5, 2.0, 1.0, 0.75, -1.25, 0.5,
    ];
    (a, b)
}

fn solve_linalg_op_debugs(output: &TracedTensor) -> Vec<String> {
    graph_op_debugs(output)
        .into_iter()
        .filter(|debug| debug.contains("op: LuFactor") || debug.contains("op: Solve"))
        .collect()
}

#[test]
fn traced_primal_solve_compiles_to_one_fused_factor_solve() {
    let (a_data, b_data) = pivoting_solve_fixture();
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 3, 2], a_data.clone()))
        .unwrap();
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2, 2], b_data.clone()))
        .unwrap();
    let x = a.solve(&b).unwrap();

    let ops = solve_linalg_op_debugs(&x);
    assert_eq!(ops.len(), 1, "one fused op per solve: {ops:#?}");
    assert!(ops[0].contains("op: LuFactorSolve"), "{ops:#?}");

    // The compiler keeps the fused op (no LuFactor then LuSolvePrepared split)
    // even though a primal only program never reads the factors.
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&x).unwrap();
    let extension_ops: Vec<String> = program
        .program()
        .operations()
        .filter_map(|operation| match operation.op() {
            tenferro_runtime::program::SemanticOpRef::Extension(ext) => Some(format!("{ext:?}")),
            _ => None,
        })
        .collect();
    assert_eq!(extension_ops.len(), 1, "{extension_ops:#?}");
    assert!(
        extension_ops[0].contains("op: LuFactorSolve"),
        "primal solve should compile to the fused factor solve: {extension_ops:#?}"
    );

    let solved = eval(&x);
    let expected = support::with_cpu_linalg(&mut tenferro_cpu::CpuBackend::new(), |session| {
        use tenferro_linalg::LinalgBackend;
        session.solve(
            &f64_tensor(vec![3, 3, 2], a_data.clone()),
            &f64_tensor(vec![3, 2, 2], b_data.clone()),
        )
    })
    .unwrap();
    assert_close_slice(get_f64_data(&solved), get_f64_data(&expected), 1.0e-13);
}

#[test]
fn traced_solve_vjp_reuses_fused_factors_and_matches_finite_diff() {
    let ad = ad_context();
    let (a_data, b_data) = pivoting_solve_fixture();
    let weights: Vec<f64> = (0..12).map(|i| 0.5 + 0.125 * i as f64).collect();
    let loss_of = |a: &[f64], b: &[f64]| {
        let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 3, 2], a.to_vec()))
            .unwrap();
        let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2, 2], b.to_vec()))
            .unwrap();
        let x = a.solve(&b).unwrap();
        (
            a,
            b,
            weighted_square_sum(&x, vec![3, 2, 2], weights.clone()),
        )
    };

    let (a, b, loss) = loss_of(&a_data, &b_data);
    let grad_a = ad.grad(&loss, &a).unwrap();
    let grad_b = ad.grad(&loss, &b).unwrap();
    let grads = eval_many(&[&grad_a, &grad_b]);

    // The adjoint solve reads the fused op's saved factors: the gradient
    // graph holds exactly one factorization, the fused forward op.
    let ops = solve_linalg_op_debugs(&grad_a);
    let factorizations = ops.iter().filter(|op| op.contains("op: LuFactor")).count();
    let fused = ops
        .iter()
        .filter(|op| op.contains("op: LuFactorSolve"))
        .count();
    assert_eq!((factorizations, fused), (1, 1), "{ops:#?}");

    let a_direction: Vec<f64> = (0..18).map(|i| ((i * 7) % 5) as f64 * 0.1 - 0.2).collect();
    let b_direction: Vec<f64> = (0..12).map(|i| ((i * 3) % 4) as f64 * 0.15 - 0.2).collect();
    let dot = |g: &[f64], d: &[f64]| g.iter().zip(d).map(|(g, d)| g * d).sum::<f64>();
    let fd_a = finite_diff_directional_scalar(
        |xs| get_f64_data(&eval(&loss_of(xs, &b_data).2))[0],
        &a_data,
        &a_direction,
        1.0e-6,
    );
    let fd_b = finite_diff_directional_scalar(
        |xs| get_f64_data(&eval(&loss_of(&a_data, xs).2))[0],
        &b_data,
        &b_direction,
        1.0e-6,
    );
    assert_close_scalar(
        "solve VJP wrt A",
        dot(get_f64_data(&grads[0]), &a_direction),
        fd_a,
        1.0e-6,
    );
    assert_close_scalar(
        "solve VJP wrt B",
        dot(get_f64_data(&grads[1]), &b_direction),
        fd_b,
        1.0e-6,
    );

    // JVP through the same lowering agrees with the VJP pairing.
    let a_tangent =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 3, 2], a_direction.clone()))
            .unwrap();
    let jvp = eval(&ad.jvp(&loss, &a, &a_tangent).unwrap());
    assert_close_scalar("solve JVP wrt A", get_f64_data(&jvp)[0], fd_a, 1.0e-6);
    let b_tangent =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2, 2], b_direction.clone()))
            .unwrap();
    let jvp = eval(&ad.jvp(&loss, &b, &b_tangent).unwrap());
    assert_close_scalar("solve JVP wrt B", get_f64_data(&jvp)[0], fd_b, 1.0e-6);
}

#[test]
fn traced_complex_solve_vjp_matches_finite_diff() {
    use num_complex::Complex64;
    let ad = ad_context();
    let (a_re, b_re) = pivoting_solve_fixture();
    let a_data: Vec<Complex64> = a_re
        .iter()
        .enumerate()
        .map(|(i, &re)| Complex64::new(re, 0.1 * ((i % 3) as f64 - 1.0)))
        .collect();
    let b_data: Vec<Complex64> = b_re
        .iter()
        .enumerate()
        .map(|(i, &re)| Complex64::new(re, 0.2 * ((i % 4) as f64 - 1.5)))
        .collect();
    // Real loss sum |x|^2 of the complex solution.
    let loss_of = |a: &[Complex64], b: &[Complex64]| {
        let a = TracedTensor::from_tensor_concrete_shape(c64_tensor(vec![3, 3, 2], a.to_vec()))
            .unwrap();
        let b = TracedTensor::from_tensor_concrete_shape(c64_tensor(vec![3, 2, 2], b.to_vec()))
            .unwrap();
        let x = a.solve(&b).unwrap();
        let loss = reduce_all(&(&x.conj().unwrap() * &x).unwrap());
        (a, b, loss)
    };
    let (a, b, loss) = loss_of(&a_data, &b_data);
    let grads = eval_many(&[&ad.grad(&loss, &a).unwrap(), &ad.grad(&loss, &b).unwrap()]);

    // Split each complex entry into (re, im) coordinates for finite differences.
    let pack = |v: &[Complex64]| v.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<f64>>();
    let unpack = |v: &[f64]| {
        v.chunks_exact(2)
            .map(|p| Complex64::new(p[0], p[1]))
            .collect::<Vec<_>>()
    };
    let loss_value = |a: &[Complex64], b: &[Complex64]| get_c64_data(&eval(&loss_of(a, b).2))[0].re;
    for (label, grad, base, is_a) in [
        ("A", &grads[0], &a_data, true),
        ("B", &grads[1], &b_data, false),
    ] {
        let direction: Vec<f64> = (0..2 * base.len())
            .map(|i| ((i * 5) % 7) as f64 * 0.05 - 0.15)
            .collect();
        let fd = finite_diff_directional_scalar(
            |xs| {
                let z = unpack(xs);
                if is_a {
                    loss_value(&z, &b_data)
                } else {
                    loss_value(&a_data, &z)
                }
            },
            &pack(base),
            &direction,
            1.0e-6,
        );
        // tenferro follows the PyTorch convention: the gradient of a real loss
        // is dL/dRe + i dL/dIm, so the directional derivative is Re <g, d>.
        let analytic: f64 = get_c64_data(grad)
            .iter()
            .zip(unpack(&direction))
            .map(|(g, d)| g.re * d.re + g.im * d.im)
            .sum();
        assert_close_scalar(
            &format!("complex solve VJP wrt {label}"),
            analytic,
            fd,
            1.0e-6,
        );
    }
}
