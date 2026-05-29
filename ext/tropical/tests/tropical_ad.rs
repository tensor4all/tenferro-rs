#![cfg(feature = "autodiff")]

use tenferro_ad::AdContext;
use tenferro_cpu::CpuBackend;
use tenferro_ext_tropical::einsum::{tropical_einsum_with_argmax, TropicalEinsumKind};
use tenferro_ext_tropical::traced::tropical_dot_general_fused;
use tenferro_ext_tropical::tropical_ad_rules;
use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TracedTensor};

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
        .with_core_rules()
        .with_extension_rules(tropical_ad_rules().expect("tropical AD rules"))
        .build()
        .expect("AD context")
}

fn assert_f64_data(tensor: &Tensor, expected: &[f64]) {
    assert_eq!(tensor.as_slice::<f64>().expect("f64 tensor"), expected);
}

fn assert_sum_gradients(
    a_data: Vec<f64>,
    b_data: Vec<f64>,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    expected_a: &[f64],
    expected_b: &[f64],
) {
    let a = TracedTensor::from_vec_col_major(a_shape, a_data);
    let b = TracedTensor::from_vec_col_major(b_shape, b_data);
    let out = tropical_dot_general_fused(&a, &b);
    let loss = out.reduce_sum(&[0, 1]);
    let ad = tropical_ad();

    let grad_a = ad.grad(&loss, &a).expect("grad wrt a");
    let grad_b = ad.grad(&loss, &b).expect("grad wrt b");

    assert_f64_data(&run_traced(&grad_a), expected_a);
    assert_f64_data(&run_traced(&grad_b), expected_b);
}

#[test]
fn traced_fused_forward_matches_eager_max_plus_einsum() {
    let a_data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let b_data = vec![10.0_f64, 20.0, 30.0, 40.0];
    let a = TracedTensor::from_vec_col_major(vec![2, 2], a_data.clone());
    let b = TracedTensor::from_vec_col_major(vec![2, 2], b_data.clone());
    let fused = tropical_dot_general_fused(&a, &b);

    let eager_a = Tensor::from_vec_col_major(vec![2, 2], a_data);
    let eager_b = Tensor::from_vec_col_major(vec![2, 2], b_data);
    let eager = tropical_einsum_with_argmax(
        TropicalEinsumKind::MaxPlus,
        &[&eager_a, &eager_b],
        "ij,jk->ik",
    )
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
