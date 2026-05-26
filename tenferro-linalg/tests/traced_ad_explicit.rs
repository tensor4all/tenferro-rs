#![cfg(feature = "autodiff")]

use tenferro_ad::AdContext;
use tenferro_runtime::{CpuBackend, GraphCompiler, GraphExecutor, Tensor, TracedTensor};
use tenferro_tensor::TypedTensor;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
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
    tensor.reduce_sum(&axes)
}

fn assert_finite_tensor(tensor: &Tensor) {
    if let Some(values) = tensor.as_slice::<f64>() {
        assert!(values.iter().all(|value| value.is_finite()), "{values:?}");
        return;
    }
    if let Some(values) = tensor.as_slice::<num_complex::Complex64>() {
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

fn ad_context() -> AdContext {
    AdContext::builder()
        .with_core_rules()
        .with_extension_rules(tenferro_linalg::ad_rules().unwrap())
        .build()
        .unwrap()
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

    let (_u, s, _vt) = tenferro_linalg::svd(&a);
    let y = s.reduce_sum(&[0]);
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

    let x = tenferro_linalg::full_piv_lu_solve(&a, &b);
    let loss = x.reduce_sum(&[0, 1]);
    let grad_b = ad.grad(&loss, &b).unwrap();
    let out = eval(&grad_b);

    assert_eq!(out.shape(), &[2, 1]);
}

#[test]
fn svd_values_grad_matches_finite_diff() {
    let ad = ad_context();
    let data = vec![3.0, 0.1, 0.2, 0.3, 2.0, 0.4];
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], data.clone()));

    let (_u, s, _vt) = tenferro_linalg::svd(&a);
    let loss = s.reduce_sum(&[0]);
    let grad = ad.grad(&loss, &a).unwrap();
    let out = eval(&grad);

    let actual = get_f64_data(&out);
    assert_eq!(actual.len(), data.len());
    for idx in 0..data.len() {
        let expected = finite_diff_scalar(
            |xs| {
                let input =
                    TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], xs.to_vec()));
                let (_u, s, _vt) = tenferro_linalg::svd(&input);
                get_f64_data(&eval(&s.reduce_sum(&[0])))[0]
            },
            &data,
            idx,
            1.0e-6,
        );
        let diff = (actual[idx] - expected).abs();
        assert!(
            diff <= 1.0e-4,
            "idx {idx}: expected {expected}, got {}, diff {diff}",
            actual[idx]
        );
    }
}

#[test]
fn remaining_linalg_ops_jvp_use_extension_ad_rule() {
    let ad = ad_context();

    let spd_data = vec![3.0, 0.2, 0.2, 4.0];
    let spd = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], spd_data.clone()));
    let dspd =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![0.3, 0.1, 0.1, -0.2]));

    let general_data = vec![2.0, 0.5, 0.25, 3.0];
    let general =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], general_data.clone()));
    let dgeneral =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![0.2, 0.1, -0.1, 0.4]));

    let rhs = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![1.0, 2.0]));
    let drhs = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![0.4, -0.3]));

    let chol_loss = reduce_all(&tenferro_linalg::cholesky(&spd));
    let chol_tangent = ad.jvp(&chol_loss, &spd, &dspd).unwrap();

    let (q, r) = tenferro_linalg::qr(&general);
    let qr_loss = &reduce_all(&q) + &reduce_all(&r);
    let qr_tangent = ad.jvp(&qr_loss, &general, &dgeneral).unwrap();

    let (eigh_values, _eigh_vectors) = tenferro_linalg::eigh(&spd);
    let eigh_loss = reduce_all(&eigh_values);
    let eigh_tangent = ad.jvp(&eigh_loss, &spd, &dspd).unwrap();

    let (eig_values, _eig_vectors) = tenferro_linalg::eig(&general);
    let eig_loss = reduce_all(&eig_values);
    let eig_tangent = ad.jvp(&eig_loss, &general, &dgeneral).unwrap();

    let solve_loss = reduce_all(&tenferro_linalg::solve(&general, &rhs));
    let solve_tangent = ad.jvp(&solve_loss, &rhs, &drhs).unwrap();
    let triangular_loss = reduce_all(&tenferro_linalg::triangular_solve(
        &general, &rhs, true, true, false, false,
    ));
    let triangular_tangent = ad.jvp(&triangular_loss, &rhs, &drhs).unwrap();

    let (_p, lu_l, lu_u, _parity) = tenferro_linalg::lu(&general);
    let lu_loss = &reduce_all(&lu_l) + &reduce_all(&lu_u);
    let lu_tangent = ad.jvp(&lu_loss, &general, &dgeneral).unwrap();

    let (_p, full_lu_l, full_lu_u, _q, _full_parity) = tenferro_linalg::full_piv_lu(&general);
    let full_lu_loss = &reduce_all(&full_lu_l) + &reduce_all(&full_lu_u);
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

    for result in &results {
        assert_eq!(result.shape(), &[] as &[usize]);
        assert_finite_tensor(result);
    }

    assert_eq!(get_c64_data(&results[3]).len(), 1);
}

#[test]
fn solve_and_triangular_solve_grad_use_extension_transpose_rules() {
    let ad = ad_context();

    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![2.0, 0.0, 0.0, 4.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 1], vec![4.0, 8.0]));

    let solve_loss = reduce_all(&tenferro_linalg::solve(&a, &b));
    let solve_grad_b = ad.grad(&solve_loss, &b).unwrap();

    let triangular_loss = reduce_all(&tenferro_linalg::triangular_solve(
        &a, &b, true, true, false, false,
    ));
    let triangular_grad_b = ad.grad(&triangular_loss, &b).unwrap();

    let results = eval_many(&[&solve_grad_b, &triangular_grad_b]);
    for result in &results {
        assert_eq!(result.shape(), &[2, 1]);
        assert_eq!(get_f64_data(result), &[0.5, 0.25]);
    }
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

    let x = tenferro_linalg::solve(&a, &b);
    let loss = x.reduce_sum(&[0, 1, 2]);
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
