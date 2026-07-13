#![cfg(feature = "autodiff")]

use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_ext_tropical::traced::tropical_dot_general_fused;
use tenferro_ext_tropical::tropical_ad_rules;
use tenferro_ext_tropical::{einsum::tropical_einsum_with_argmax, TropicalKind};
use tenferro_runtime::{Error, GraphCompiler, GraphExecutor, Tensor, TracedTensor};

fn run_traced(output: &TracedTensor) -> Tensor {
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(output).expect("compile traced graph");
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_ext_tropical::register_runtime)
        .expect("register tropical runtime");
    executor.run(&program).expect("execute traced graph")
}

fn tropical_ad() -> AdContext {
    AdContext::builder()
        .with_extension_rules(tropical_ad_rules().expect("tropical AD rules"))
        .build()
        .expect("AD context")
}

fn assert_f64_data(tensor: &Tensor, expected: &[f64]) {
    assert_eq!(tensor.as_slice::<f64>().expect("f64 tensor"), expected);
}

fn assert_f64_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (left - right).abs() <= tolerance,
            "entry {index}: actual {left} differs from expected {right}"
        );
    }
}

fn max_plus_eager_ij_jk_to_ik(a_data: Vec<f64>, b_data: Vec<f64>) -> Vec<f64> {
    let a = Tensor::from_vec_col_major(vec![2, 2], a_data).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 2], b_data).unwrap();
    let out = tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&a, &b], "ij,jk->ik")
        .expect("eager tropical einsum");
    out.output.as_slice::<f64>().expect("f64 output").to_vec()
}

fn add_scaled(data: &[f64], direction: &[f64], scale: f64) -> Vec<f64> {
    data.iter()
        .zip(direction)
        .map(|(&value, &delta)| value + scale * delta)
        .collect()
}

fn fd_jvp_lhs(a_data: &[f64], b_data: &[f64], direction: &[f64]) -> Vec<f64> {
    let step = 1.0e-6;
    let plus = max_plus_eager_ij_jk_to_ik(add_scaled(a_data, direction, step), b_data.to_vec());
    let minus = max_plus_eager_ij_jk_to_ik(add_scaled(a_data, direction, -step), b_data.to_vec());
    plus.iter()
        .zip(&minus)
        .map(|(&left, &right)| (left - right) / (2.0 * step))
        .collect()
}

fn fd_jvp_rhs(a_data: &[f64], b_data: &[f64], direction: &[f64]) -> Vec<f64> {
    let step = 1.0e-6;
    let plus = max_plus_eager_ij_jk_to_ik(a_data.to_vec(), add_scaled(b_data, direction, step));
    let minus = max_plus_eager_ij_jk_to_ik(a_data.to_vec(), add_scaled(b_data, direction, -step));
    plus.iter()
        .zip(&minus)
        .map(|(&left, &right)| (left - right) / (2.0 * step))
        .collect()
}

fn weighted_max_plus_scalar(a_data: Vec<f64>, b_data: Vec<f64>, weights: &[f64]) -> f64 {
    max_plus_eager_ij_jk_to_ik(a_data, b_data)
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| value * weight)
        .sum()
}

fn fd_gradient_lhs(a_data: &[f64], b_data: &[f64], weights: &[f64]) -> Vec<f64> {
    let step = 1.0e-6;
    (0..a_data.len())
        .map(|index| {
            let mut plus_a = a_data.to_vec();
            let mut minus_a = a_data.to_vec();
            plus_a[index] += step;
            minus_a[index] -= step;
            let plus = weighted_max_plus_scalar(plus_a, b_data.to_vec(), weights);
            let minus = weighted_max_plus_scalar(minus_a, b_data.to_vec(), weights);
            (plus - minus) / (2.0 * step)
        })
        .collect()
}

fn fd_gradient_rhs(a_data: &[f64], b_data: &[f64], weights: &[f64]) -> Vec<f64> {
    let step = 1.0e-6;
    (0..b_data.len())
        .map(|index| {
            let mut plus_b = b_data.to_vec();
            let mut minus_b = b_data.to_vec();
            plus_b[index] += step;
            minus_b[index] -= step;
            let plus = weighted_max_plus_scalar(a_data.to_vec(), plus_b, weights);
            let minus = weighted_max_plus_scalar(a_data.to_vec(), minus_b, weights);
            (plus - minus) / (2.0 * step)
        })
        .collect()
}

fn assert_sum_gradients(
    a_data: Vec<f64>,
    b_data: Vec<f64>,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    expected_a: &[f64],
    expected_b: &[f64],
) {
    let a = TracedTensor::from_vec_col_major(a_shape, a_data).unwrap();
    let b = TracedTensor::from_vec_col_major(b_shape, b_data).unwrap();
    let out = tropical_dot_general_fused(&a, &b).unwrap();
    let loss = out.reduce_sum(&[0, 1]).unwrap();
    let ad = tropical_ad();

    let grad_a = ad.grad(&loss, &a).expect("grad wrt a");
    let grad_b = ad.grad(&loss, &b).expect("grad wrt b");

    assert_f64_data(&run_traced(&grad_a), expected_a);
    assert_f64_data(&run_traced(&grad_b), expected_b);
}

#[test]
fn finite_difference_jvp_matches_unique_winner_max_plus_lhs_and_rhs() {
    let a_data = vec![1.0_f64, 3.0, 4.0, 0.0];
    let b_data = vec![2.0_f64, 0.0, -1.0, 5.0];
    let da_data = vec![0.25_f64, -0.5, 0.75, 1.25];
    let db_data = vec![-0.4_f64, 0.3, 1.1, -0.2];

    let a = TracedTensor::from_vec_col_major(vec![2, 2], a_data.clone()).unwrap();
    let b = TracedTensor::from_vec_col_major(vec![2, 2], b_data.clone()).unwrap();
    let da = TracedTensor::from_vec_col_major(vec![2, 2], da_data.clone()).unwrap();
    let db = TracedTensor::from_vec_col_major(vec![2, 2], db_data.clone()).unwrap();
    let out = tropical_dot_general_fused(&a, &b).unwrap();
    let ad = tropical_ad();

    let jvp_a = ad.jvp(&out, &a, &da).expect("jvp wrt a");
    let jvp_b = ad.jvp(&out, &b, &db).expect("jvp wrt b");
    let actual_a = run_traced(&jvp_a);
    let actual_b = run_traced(&jvp_b);

    assert_f64_close(
        actual_a.as_slice::<f64>().expect("lhs jvp"),
        &fd_jvp_lhs(&a_data, &b_data, &da_data),
        1.0e-8,
    );
    assert_f64_close(
        actual_b.as_slice::<f64>().expect("rhs jvp"),
        &fd_jvp_rhs(&a_data, &b_data, &db_data),
        1.0e-8,
    );
}

#[test]
fn finite_difference_gradient_matches_unique_winner_weighted_scalarization() {
    let a_data = vec![1.0_f64, 3.0, 4.0, 0.0];
    let b_data = vec![2.0_f64, 0.0, -1.0, 5.0];
    let weights_data = vec![0.5_f64, -1.25, 2.0, 0.75];

    let a = TracedTensor::from_vec_col_major(vec![2, 2], a_data.clone()).unwrap();
    let b = TracedTensor::from_vec_col_major(vec![2, 2], b_data.clone()).unwrap();
    let weights = TracedTensor::from_vec_col_major(vec![2, 2], weights_data.clone()).unwrap();
    let out = tropical_dot_general_fused(&a, &b).unwrap();
    let weighted = (&out * &weights).unwrap();
    let loss = weighted.reduce_sum(&[0, 1]).unwrap();
    let ad = tropical_ad();

    let grad_a = ad.grad(&loss, &a).expect("grad wrt a");
    let grad_b = ad.grad(&loss, &b).expect("grad wrt b");
    let actual_a = run_traced(&grad_a);
    let actual_b = run_traced(&grad_b);

    assert_f64_close(
        actual_a.as_slice::<f64>().expect("lhs grad"),
        &fd_gradient_lhs(&a_data, &b_data, &weights_data),
        1.0e-8,
    );
    assert_f64_close(
        actual_b.as_slice::<f64>().expect("rhs grad"),
        &fd_gradient_rhs(&a_data, &b_data, &weights_data),
        1.0e-8,
    );
}

#[test]
fn tangent_shape_constraint_rejects_independent_tropical_tangent_mismatch() {
    let symbolic_matrix = |shape: Vec<usize>| {
        let len = shape.iter().product();
        TracedTensor::from_tensor_symbolic_shape(
            Tensor::from_vec_col_major(shape, vec![1.0_f64; len]).unwrap(),
        )
        .unwrap()
    };
    let lhs = symbolic_matrix(vec![2, 3]);
    let rhs = symbolic_matrix(vec![3, 4]);
    let tangent = symbolic_matrix(vec![2, 3]);
    assert_ne!(
        lhs.axis_sym_dim(0).unwrap(),
        tangent.axis_sym_dim(0).unwrap(),
        "primal and tangent must have independent symbolic origins"
    );
    let output = tropical_dot_general_fused(&lhs, &rhs).unwrap();
    let jvp = tropical_ad().jvp(&output, &lhs, &tangent).unwrap();
    let mut compiler = GraphCompiler::new();
    compiler
        .compile(&jvp)
        .expect("matching independent tangent shape should compile");

    let mismatched_tangent = symbolic_matrix(vec![5, 3]);
    let mismatched_jvp = tropical_ad()
        .jvp(&output, &lhs, &mismatched_tangent)
        .unwrap();
    let error = GraphCompiler::new()
        .compile(&mismatched_jvp)
        .expect_err("mismatched tropical tangent axis must fail during compilation");
    assert!(matches!(
        error,
        Error::ShapeConstraintViolation {
            family: "tenferro-ext-tropical.einsum_jvp.v1",
            ..
        }
    ));
}

#[test]
fn traced_fused_forward_matches_eager_max_plus_einsum() {
    let a_data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let b_data = vec![10.0_f64, 20.0, 30.0, 40.0];
    let a = TracedTensor::from_vec_col_major(vec![2, 2], a_data.clone()).unwrap();
    let b = TracedTensor::from_vec_col_major(vec![2, 2], b_data.clone()).unwrap();
    let fused = tropical_dot_general_fused(&a, &b).unwrap();

    let eager_a = Tensor::from_vec_col_major(vec![2, 2], a_data).unwrap();
    let eager_b = Tensor::from_vec_col_major(vec![2, 2], b_data).unwrap();
    let eager =
        tropical_einsum_with_argmax(TropicalKind::MaxPlus, &[&eager_a, &eager_b], "ij,jk->ik")
            .expect("eager tropical einsum");

    assert_f64_data(
        &run_traced(&fused),
        eager.output.as_slice::<f64>().expect("eager f64 output"),
    );
}

#[test]
fn unique_winner_max_plus_gradient_routes_to_winning_entries() {
    assert_sum_gradients(
        vec![1.0_f64, 2.0, 3.0, 4.0],
        vec![10.0_f64, 20.0, 30.0, 40.0],
        vec![2, 2],
        vec![2, 2],
        &[0.0, 0.0, 2.0, 2.0],
        &[0.0, 2.0, 0.0, 2.0],
    );
}

#[test]
fn mixed_winner_max_plus_gradient_routes_each_output_to_its_winner() {
    assert_sum_gradients(
        vec![10.0_f64, 0.0, 0.0, 10.0],
        vec![0.0_f64, 0.0, 0.0, 0.0],
        vec![2, 2],
        vec![2, 2],
        &[2.0, 0.0, 0.0, 2.0],
        &[1.0, 1.0, 1.0, 1.0],
    );
}

#[test]
fn first_winner_ties_route_only_to_k_zero() {
    assert_sum_gradients(
        vec![1.0_f64, 1.0],
        vec![2.0_f64, 2.0],
        vec![1, 2],
        vec![2, 1],
        &[1.0, 0.0],
        &[1.0, 0.0],
    );
}
