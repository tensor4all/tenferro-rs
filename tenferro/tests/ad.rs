use std::collections::HashMap;
use std::sync::Arc;

use computegraph::compile::compile;
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use num_complex::Complex64;
use tenferro::compiler::{compile_to_exec, lower_to_stablehlo};
use tenferro::einsum::einsum;
use tenferro::engine::Engine;
use tenferro::exec::eval_exec_ir;
use tenferro::traced::TracedTensor;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorBackend, TypedTensor};
use tidu::{differentiate, transpose};

const TOL: f64 = 1e-6;

fn finite_diff_scalar(f: impl Fn(&[f64]) -> f64, x: &[f64], idx: usize, h: f64) -> f64 {
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[idx] += h;
    xm[idx] -= h;
    (f(&xp) - f(&xm)) / (2.0 * h)
}

fn finite_diff_scalar_lhs(
    f: impl Fn(&[f64], &[f64]) -> f64,
    lhs: &[f64],
    rhs: &[f64],
    idx: usize,
    h: f64,
) -> f64 {
    let mut lp = lhs.to_vec();
    let mut lm = lhs.to_vec();
    lp[idx] += h;
    lm[idx] -= h;
    (f(&lp, rhs) - f(&lm, rhs)) / (2.0 * h)
}

fn finite_diff_scalar_rhs(
    f: impl Fn(&[f64], &[f64]) -> f64,
    lhs: &[f64],
    rhs: &[f64],
    idx: usize,
    h: f64,
) -> f64 {
    let mut rp = rhs.to_vec();
    let mut rm = rhs.to_vec();
    rp[idx] += h;
    rm[idx] -= h;
    (f(lhs, &rp) - f(lhs, &rm)) / (2.0 * h)
}

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn c64_tensor(shape: Vec<usize>, data: Vec<Complex64>) -> Tensor {
    Tensor::C64(TypedTensor::from_vec(shape, data))
}

fn get_f64_data(tensor: &Tensor) -> &[f64] {
    match tensor {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected f64 tensor"),
    }
}

fn get_c64_data(tensor: &Tensor) -> &[Complex64] {
    match tensor {
        Tensor::C64(inner) => inner.host_data(),
        _ => panic!("expected c64 tensor"),
    }
}

fn eval_tensor(mut traced: TracedTensor) -> Tensor {
    let mut engine = Engine::new(CpuBackend::new());
    traced.eval(&mut engine).unwrap().clone()
}

fn eval_scalar(traced: TracedTensor) -> f64 {
    get_f64_data(&eval_tensor(traced))[0]
}

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    }
}

fn tensor_input_key(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

fn eval_fragment_outputs(
    roots: Vec<Arc<Fragment<StdTensorOp>>>,
    outputs: &[GlobalValKey<StdTensorOp>],
    inputs_map: &HashMap<TensorInputKey, Tensor>,
) -> Vec<Tensor> {
    let view = resolve(roots);
    let graph = materialize_merge(&view, outputs);
    let compiled = compile(&graph);
    let stablehlo = lower_to_stablehlo(&compiled);
    let exec = compile_to_exec(&stablehlo);
    let inputs = graph
        .inputs
        .iter()
        .map(|key| match key {
            GlobalValKey::Input(k) => inputs_map.get(k).expect("missing tensor input").clone(),
            _ => panic!("expected input key"),
        })
        .collect();
    let mut backend = CpuBackend::new();
    eval_exec_ir(&mut backend, &exec, inputs)
}

fn scalar_f64_tensor(value: f64) -> Tensor {
    f64_tensor(vec![], vec![value])
}

fn build_svd_values_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let input_key = tensor_input_key(10_000);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let svd = builder.add_op(StdTensorOp::Svd, vec![ValRef::Local(a)], OpMode::Primal);
    let loss = builder.add_op(
        StdTensorOp::ReduceSum {
            axes: vec![0],
            input_shape: vec![2],
        },
        vec![ValRef::Local(svd[1])],
        OpMode::Primal,
    );
    let loss_key = builder.global_key(loss[0]).clone();
    builder.set_outputs(vec![loss[0]]);
    (Arc::new(builder.build()), input_key, loss_key)
}

fn build_svd_uv_product_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let input_key = tensor_input_key(15_000);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let svd = builder.add_op(StdTensorOp::Svd, vec![ValRef::Local(a)], OpMode::Primal);
    let uv_product = builder.add_op(
        StdTensorOp::DotGeneral(matmul_config()),
        vec![ValRef::Local(svd[0]), ValRef::Local(svd[2])],
        OpMode::Primal,
    );
    let loss = builder.add_op(
        StdTensorOp::ReduceSum {
            axes: vec![0, 1],
            input_shape: vec![2, 2],
        },
        vec![ValRef::Local(uv_product[0])],
        OpMode::Primal,
    );
    let loss_key = builder.global_key(loss[0]).clone();
    builder.set_outputs(vec![loss[0]]);
    (Arc::new(builder.build()), input_key, loss_key)
}

fn build_eigh_values_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let input_key = tensor_input_key(20_000);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let eigh = builder.add_op(StdTensorOp::Eigh, vec![ValRef::Local(a)], OpMode::Primal);
    let loss = builder.add_op(
        StdTensorOp::ReduceSum {
            axes: vec![0],
            input_shape: vec![2],
        },
        vec![ValRef::Local(eigh[0])],
        OpMode::Primal,
    );
    let loss_key = builder.global_key(loss[0]).clone();
    builder.set_outputs(vec![loss[0]]);
    (Arc::new(builder.build()), input_key, loss_key)
}

fn build_eigh_projector_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    TensorInputKey,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let input_key = tensor_input_key(25_000);
    let weights_key = tensor_input_key(25_001);
    let probe_key = tensor_input_key(25_002);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let weights = builder.add_input(weights_key.clone());
    let probe = builder.add_input(probe_key.clone());
    let eigh = builder.add_op(StdTensorOp::Eigh, vec![ValRef::Local(a)], OpMode::Primal);
    let diag = builder.add_op(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1,
        },
        vec![ValRef::Local(weights)],
        OpMode::Primal,
    );
    let weighted_vectors = builder.add_op(
        StdTensorOp::DotGeneral(matmul_config()),
        vec![ValRef::Local(eigh[1]), ValRef::Local(diag[0])],
        OpMode::Primal,
    );
    let vt = builder.add_op(
        StdTensorOp::Transpose { perm: vec![1, 0] },
        vec![ValRef::Local(eigh[1])],
        OpMode::Primal,
    );
    let projector = builder.add_op(
        StdTensorOp::DotGeneral(matmul_config()),
        vec![ValRef::Local(weighted_vectors[0]), ValRef::Local(vt[0])],
        OpMode::Primal,
    );
    let weighted = builder.add_op(
        StdTensorOp::Mul,
        vec![ValRef::Local(projector[0]), ValRef::Local(probe)],
        OpMode::Primal,
    );
    let loss = builder.add_op(
        StdTensorOp::ReduceSum {
            axes: vec![0, 1],
            input_shape: vec![2, 2],
        },
        vec![ValRef::Local(weighted[0])],
        OpMode::Primal,
    );
    let loss_key = builder.global_key(loss[0]).clone();
    builder.set_outputs(vec![loss[0]]);
    (
        Arc::new(builder.build()),
        input_key,
        weights_key,
        probe_key,
        loss_key,
    )
}

fn build_solve_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let a_key = tensor_input_key(30_000);
    let b_key = tensor_input_key(30_001);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(a_key.clone());
    let b = builder.add_input(b_key.clone());
    let solve = builder.add_op(
        StdTensorOp::Solve,
        vec![ValRef::Local(a), ValRef::Local(b)],
        OpMode::Primal,
    );
    let loss = builder.add_op(
        StdTensorOp::ReduceSum {
            axes: vec![0, 1],
            input_shape: vec![2, 1],
        },
        vec![ValRef::Local(solve[0])],
        OpMode::Primal,
    );
    let loss_key = builder.global_key(loss[0]).clone();
    builder.set_outputs(vec![loss[0]]);
    (Arc::new(builder.build()), a_key, b_key, loss_key)
}

fn build_triangular_solve_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let a_key = tensor_input_key(35_000);
    let b_key = tensor_input_key(35_001);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(a_key.clone());
    let b = builder.add_input(b_key.clone());
    let solve = builder.add_op(
        StdTensorOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        },
        vec![ValRef::Local(a), ValRef::Local(b)],
        OpMode::Primal,
    );
    let loss = builder.add_op(
        StdTensorOp::ReduceSum {
            axes: vec![0, 1],
            input_shape: vec![2, 1],
        },
        vec![ValRef::Local(solve[0])],
        OpMode::Primal,
    );
    let loss_key = builder.global_key(loss[0]).clone();
    builder.set_outputs(vec![loss[0]]);
    (Arc::new(builder.build()), a_key, b_key, loss_key)
}

fn grad_from_fragment_with_inputs(
    fragment: Arc<Fragment<StdTensorOp>>,
    loss_key: GlobalValKey<StdTensorOp>,
    input_key: TensorInputKey,
    mut inputs_map: HashMap<TensorInputKey, Tensor>,
) -> Tensor {
    let view = resolve(vec![fragment.clone()]);
    let linear = differentiate(
        &view,
        std::slice::from_ref(&loss_key),
        std::slice::from_ref(&input_key),
        0,
    );
    let transposed = transpose(&linear);
    let linear_fragment = Arc::new(linear.fragment);
    let grad_key = transposed.tangent_outputs[0]
        .map(|id| transposed.fragment.vals()[id].key.clone())
        .expect("expected active gradient output");
    let cotangent_input_key = match &transposed.fragment.vals()[transposed.tangent_inputs[0].1].key
    {
        GlobalValKey::Input(key) => key.clone(),
        _ => panic!("expected cotangent input"),
    };
    let transposed_fragment = Arc::new(transposed.fragment);

    inputs_map.insert(cotangent_input_key, scalar_f64_tensor(1.0));

    eval_fragment_outputs(
        vec![fragment, linear_fragment, transposed_fragment.clone()],
        &[grad_key],
        &inputs_map,
    )
    .into_iter()
    .next()
    .expect("gradient output")
}

fn grad_from_fragment(
    fragment: Arc<Fragment<StdTensorOp>>,
    loss_key: GlobalValKey<StdTensorOp>,
    input_key: TensorInputKey,
    input: Tensor,
) -> Tensor {
    let mut inputs_map = HashMap::new();
    inputs_map.insert(input_key.clone(), input);
    grad_from_fragment_with_inputs(fragment, loss_key, input_key, inputs_map)
}

fn sum_svd_values(data: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = f64_tensor(vec![2, 2], data.to_vec());
    let outputs = TensorBackend::svd(&mut backend, &input);
    get_f64_data(&outputs[1]).iter().sum()
}

fn sum_svd_uv_product(data: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = f64_tensor(vec![2, 2], data.to_vec());
    let outputs = TensorBackend::svd(&mut backend, &input);
    let product =
        TensorBackend::dot_general(&mut backend, &outputs[0], &outputs[2], &matmul_config());
    get_f64_data(&product).iter().sum()
}

fn sum_eigh_values(data: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = f64_tensor(vec![2, 2], data.to_vec());
    let outputs = TensorBackend::eigh(&mut backend, &input);
    get_f64_data(&outputs[0]).iter().sum()
}

fn sum_eigh_projector(data: &[f64], weights: &[f64], probe: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = f64_tensor(vec![2, 2], data.to_vec());
    let outputs = TensorBackend::eigh(&mut backend, &input);
    let diag =
        TensorBackend::embed_diagonal(&mut backend, &f64_tensor(vec![2], weights.to_vec()), 0, 1);
    let weighted_vectors =
        TensorBackend::dot_general(&mut backend, &outputs[1], &diag, &matmul_config());
    let vt = TensorBackend::transpose(&mut backend, &outputs[1], &[1, 0]);
    let projector =
        TensorBackend::dot_general(&mut backend, &weighted_vectors, &vt, &matmul_config());
    let weighted = TensorBackend::mul(
        &mut backend,
        &projector,
        &f64_tensor(vec![2, 2], probe.to_vec()),
    );
    get_f64_data(&weighted).iter().sum()
}

fn sum_solve(a: &[f64], b: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let a_tensor = f64_tensor(vec![2, 2], a.to_vec());
    let b_tensor = f64_tensor(vec![2, 1], b.to_vec());
    let out = TensorBackend::solve(&mut backend, &a_tensor, &b_tensor);
    get_f64_data(&out).iter().sum()
}

fn sum_triangular_solve(a: &[f64], b: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let a_tensor = f64_tensor(vec![2, 2], a.to_vec());
    let b_tensor = f64_tensor(vec![2, 1], b.to_vec());
    let out = TensorBackend::triangular_solve(
        &mut backend,
        &a_tensor,
        &b_tensor,
        true,
        true,
        false,
        false,
    );
    get_f64_data(&out).iter().sum()
}

fn assert_close_slice(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

fn assert_close_slice_c64(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).norm() <= TOL,
            "index {index}: expected {expected:?}, got {actual:?}"
        );
    }
}

fn assert_grad_matches_finite_diff(actual: &[f64], base: &[f64], f: impl Fn(&[f64]) -> f64) {
    assert_eq!(actual.len(), base.len());
    for index in 0..base.len() {
        let expected = finite_diff_scalar(&f, base, index, 1e-6);
        assert!(
            (actual[index] - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {}",
            actual[index]
        );
    }
}

fn assert_grad_matches_finite_diff_lhs(
    actual: &[f64],
    lhs: &[f64],
    rhs: &[f64],
    f: &impl Fn(&[f64], &[f64]) -> f64,
) {
    assert_eq!(actual.len(), lhs.len());
    for index in 0..lhs.len() {
        let expected = finite_diff_scalar_lhs(&f, lhs, rhs, index, 1e-6);
        assert!(
            (actual[index] - expected).abs() <= TOL,
            "lhs index {index}: expected {expected}, got {}",
            actual[index]
        );
    }
}

fn assert_grad_matches_finite_diff_rhs(
    actual: &[f64],
    lhs: &[f64],
    rhs: &[f64],
    f: &impl Fn(&[f64], &[f64]) -> f64,
) {
    assert_eq!(actual.len(), rhs.len());
    for index in 0..rhs.len() {
        let expected = finite_diff_scalar_rhs(&f, lhs, rhs, index, 1e-6);
        assert!(
            (actual[index] - expected).abs() <= TOL,
            "rhs index {index}: expected {expected}, got {}",
            actual[index]
        );
    }
}

#[test]
fn grad_x_squared() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = x.traced_mul(&x);
    let loss = y.traced_reduce_sum(&[0]);
    assert!(loss.shape.is_empty());
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_close_slice(get_f64_data(&result), &[2.0, 4.0, 6.0]);
}

#[test]
fn grad_extract_diag_sum() {
    let a = TracedTensor::from_tensor(f64_tensor(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));
    let diag = a.traced_extract_diag(0, 1);
    let loss = diag.traced_reduce_sum(&[0]);
    assert!(loss.shape.is_empty());
    let grad = loss.grad(&a).unwrap();

    let result = eval_tensor(grad);
    assert_eq!(result.shape(), &[3, 3]);
    assert_close_slice(
        get_f64_data(&result),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
}

#[test]
fn grad_embed_diag_sum() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], vec![2.0, -1.0, 4.0]));
    let diag = x.traced_embed_diag(0, 1);
    let loss = diag.traced_reduce_sum(&[0, 1]);
    assert!(loss.shape.is_empty());
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_eq!(result.shape(), &[3]);
    assert_close_slice(get_f64_data(&result), &[1.0, 1.0, 1.0]);
}

#[test]
fn jvp_extract_diag() {
    let a = TracedTensor::from_tensor(f64_tensor(
        vec![3, 3],
        vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
    ));
    let da = TracedTensor::from_tensor(f64_tensor(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));

    let diag = a.traced_extract_diag(0, 1);
    let jvp = diag.jvp(&a, &da);

    let result = eval_tensor(jvp);
    assert_eq!(result.shape(), &[3]);
    assert_close_slice(get_f64_data(&result), &[1.0, 5.0, 9.0]);
}

#[test]
fn jvp_embed_diag() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], vec![3.0, 4.0, 5.0]));
    let dx = TracedTensor::from_tensor(f64_tensor(vec![3], vec![0.5, -1.0, 2.0]));

    let diag = x.traced_embed_diag(0, 1);
    let jvp = diag.jvp(&x, &dx);

    let result = eval_tensor(jvp);
    assert_eq!(result.shape(), &[3, 3]);
    assert_close_slice(
        get_f64_data(&result),
        &[0.5, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0],
    );
}

#[test]
fn grad_matmul_sum() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], a_data.clone()));
    let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], b_data.clone()));
    let matmul = a.traced_dot_general(&b, matmul_config());
    let loss = matmul.traced_reduce_sum(&[0, 1]);
    assert!(loss.shape.is_empty());
    let grad = loss.grad(&a).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);

    let f = |xs: &[f64]| {
        let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], xs.to_vec()));
        let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], b_data.clone()));
        let matmul = a.traced_dot_general(&b, matmul_config());
        let loss = matmul.traced_reduce_sum(&[0, 1]);
        eval_scalar(loss)
    };

    for index in 0..a_data.len() {
        let expected = finite_diff_scalar(&f, &a_data, index, 1e-6);
        assert!(
            (grad_data[index] - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {}",
            grad_data[index]
        );
    }
}

#[test]
fn grad_matmul_sum_wrt_rhs() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], a_data.clone()));
    let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], b_data.clone()));
    let matmul = a.traced_dot_general(&b, matmul_config());
    let loss = matmul.traced_reduce_sum(&[0, 1]);
    assert!(loss.shape.is_empty());
    let grad = loss.grad(&b).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);

    let f = |xs: &[f64]| {
        let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], a_data.clone()));
        let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], xs.to_vec()));
        let matmul = a.traced_dot_general(&b, matmul_config());
        let loss = matmul.traced_reduce_sum(&[0, 1]);
        eval_scalar(loss)
    };

    for index in 0..b_data.len() {
        let expected = finite_diff_scalar(&f, &b_data, index, 1e-6);
        assert!(
            (grad_data[index] - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {}",
            grad_data[index]
        );
    }
}

#[test]
fn grad_matmul_sum_shared_input() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0];

    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 2], a_data.clone()));
    let matmul = a.traced_dot_general(&a, matmul_config());
    let loss = matmul.traced_reduce_sum(&[0, 1]);
    let grad = loss.grad(&a).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);

    let f = |xs: &[f64]| {
        let a = TracedTensor::from_tensor(f64_tensor(vec![2, 2], xs.to_vec()));
        let matmul = a.traced_dot_general(&a, matmul_config());
        let loss = matmul.traced_reduce_sum(&[0, 1]);
        eval_scalar(loss)
    };

    for index in 0..a_data.len() {
        let expected = finite_diff_scalar(&f, &a_data, index, 1e-6);
        assert!(
            (grad_data[index] - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {}",
            grad_data[index]
        );
    }
}

#[test]
fn grad_batched_matmul_sum() {
    let a_shape = vec![2, 3, 4];
    let b_shape = vec![2, 4, 5];
    let a_data: Vec<f64> = (0..24).map(|idx| 0.25 + idx as f64 * 0.1).collect();
    let b_data: Vec<f64> = (0..40).map(|idx| 0.5 + idx as f64 * 0.05).collect();

    let a = TracedTensor::from_tensor(f64_tensor(a_shape.clone(), a_data.clone()));
    let b = TracedTensor::from_tensor(f64_tensor(b_shape.clone(), b_data.clone()));
    let mut engine = Engine::new(CpuBackend::new());
    let product = einsum(&mut engine, &[&a, &b], "bij,bjk->bik").unwrap();
    let loss = product.traced_reduce_sum(&[0, 1, 2]);
    let grad = loss.grad(&a).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);

    let f = |xs: &[f64]| {
        let a = TracedTensor::from_tensor(f64_tensor(a_shape.clone(), xs.to_vec()));
        let b = TracedTensor::from_tensor(f64_tensor(b_shape.clone(), b_data.clone()));
        let mut engine = Engine::new(CpuBackend::new());
        let product = einsum(&mut engine, &[&a, &b], "bij,bjk->bik").unwrap();
        let loss = product.traced_reduce_sum(&[0, 1, 2]);
        eval_scalar(loss)
    };

    for index in 0..a_data.len() {
        let expected = finite_diff_scalar(&f, &a_data, index, 1e-6);
        assert!(
            (grad_data[index] - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {}",
            grad_data[index]
        );
    }
}

#[test]
fn grad_mul_sum_wrt_both_inputs() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = TracedTensor::from_tensor(f64_tensor(vec![3], vec![4.0, 5.0, 6.0]));
    let loss = x.traced_mul(&y).traced_reduce_sum(&[0]);

    let grad_x = loss.grad(&x).unwrap();
    let grad_y = loss.grad(&y).unwrap();

    let grad_x_tensor = eval_tensor(grad_x);
    let grad_y_tensor = eval_tensor(grad_y);
    assert_close_slice(get_f64_data(&grad_x_tensor), &[4.0, 5.0, 6.0]);
    assert_close_slice(get_f64_data(&grad_y_tensor), &[1.0, 2.0, 3.0]);
}

#[test]
fn jvp_elementwise_mul() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = TracedTensor::from_tensor(f64_tensor(vec![3], vec![4.0, 5.0, 6.0]));
    let dx = TracedTensor::from_tensor(f64_tensor(vec![3], vec![1.0, 0.0, 0.0]));

    let prod = x.traced_mul(&y);
    let jvp = prod.jvp(&x, &dx);

    let result = eval_tensor(jvp);
    assert_close_slice(get_f64_data(&result), &[4.0, 0.0, 0.0]);
}

#[test]
fn jvp_elementwise_mul_y_tangent() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = TracedTensor::from_tensor(f64_tensor(vec![3], vec![4.0, 5.0, 6.0]));
    let dy = TracedTensor::from_tensor(f64_tensor(vec![3], vec![0.0, -1.0, 2.0]));

    let prod = x.traced_mul(&y);
    let jvp = prod.jvp(&y, &dy);

    let result = eval_tensor(jvp);
    assert_close_slice(get_f64_data(&result), &[0.0, -2.0, 6.0]);
}

#[test]
fn jvp_elementwise_add_y_tangent() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = TracedTensor::from_tensor(f64_tensor(vec![3], vec![4.0, 5.0, 6.0]));
    let dy = TracedTensor::from_tensor(f64_tensor(vec![3], vec![0.5, -1.0, 2.0]));

    let sum = x.traced_add(&y);
    let jvp = sum.jvp(&y, &dy);

    let result = eval_tensor(jvp);
    assert_close_slice(get_f64_data(&result), &[0.5, -1.0, 2.0]);
}

#[test]
fn grad_neg_sum() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], vec![1.0, -2.0, 3.0]));
    let loss = x.traced_neg().traced_reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_close_slice(get_f64_data(&result), &[-1.0, -1.0, -1.0]);
}

#[test]
fn grad_conj_sum_complex() {
    let x = TracedTensor::from_tensor(c64_tensor(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
    ));
    let loss = x.traced_conj().traced_reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_close_slice_c64(
        get_c64_data(&result),
        &[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
    );
}

#[test]
fn vjp_matmul() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let cotangent = TracedTensor::from_tensor(f64_tensor(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]));

    let y = a.traced_dot_general(&b, matmul_config());
    let vjp = y.vjp(&a, &cotangent);

    let result = eval_tensor(vjp);
    assert_close_slice(get_f64_data(&result), &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
}

#[test]
fn grad_nonscalar_errors() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let b = TracedTensor::from_tensor(f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let y = a.traced_dot_general(&b, matmul_config());

    assert!(y.grad(&a).is_err());
}

#[test]
fn grad_full_vector_reduction() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
    let loss = x.traced_reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_eq!(result.shape(), &[4]);
    assert_close_slice(get_f64_data(&result), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn grad_broadcast_reduce() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = x.traced_broadcast_in_dim(&[3, 3], &[0]);
    let loss = y.traced_reduce_sum(&[0, 1]);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_close_slice(get_f64_data(&result), &[3.0, 3.0, 3.0]);
}

#[test]
fn grad_reshape() {
    let x = TracedTensor::from_tensor(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
    let y = x.traced_reshape(&[2, 2]);
    let loss = y.traced_reduce_sum(&[0, 1]);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_eq!(result.shape(), &[4]);
    assert_close_slice(get_f64_data(&result), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn grad_transpose() {
    let a = TracedTensor::from_tensor(f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let y = a.traced_transpose(&[1, 0]);
    let loss = y.traced_reduce_sum(&[0, 1]);
    let grad = loss.grad(&a).unwrap();

    let result = eval_tensor(grad);
    assert_eq!(result.shape(), &[2, 3]);
    assert_close_slice(get_f64_data(&result), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn grad_exp() {
    let x_data = vec![0.2, -0.7, 1.3];
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let loss = x.traced_exp().traced_reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| x.exp()).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.traced_exp().traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_log() {
    let x_data = vec![0.8, 1.5, 2.4];
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let loss = x.traced_log().traced_reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| 1.0 / x).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.traced_log().traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_sin_cos() {
    let x_data = vec![0.2, -0.7, 1.3];

    let x_sin = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let sin_loss = x_sin.traced_sin().traced_reduce_sum(&[0]);
    let sin_grad = sin_loss.grad(&x_sin).unwrap();
    let sin_grad_tensor = eval_tensor(sin_grad);
    let sin_grad_data = get_f64_data(&sin_grad_tensor);
    let expected_sin: Vec<f64> = x_data.iter().map(|x| x.cos()).collect();
    assert_close_slice(sin_grad_data, &expected_sin);

    let f_sin = |xs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.traced_sin().traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(sin_grad_data, &x_data, f_sin);

    let x_cos = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let cos_loss = x_cos.traced_cos().traced_reduce_sum(&[0]);
    let cos_grad = cos_loss.grad(&x_cos).unwrap();
    let cos_grad_tensor = eval_tensor(cos_grad);
    let cos_grad_data = get_f64_data(&cos_grad_tensor);
    let expected_cos: Vec<f64> = x_data.iter().map(|x| -x.sin()).collect();
    assert_close_slice(cos_grad_data, &expected_cos);

    let f_cos = |xs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.traced_cos().traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(cos_grad_data, &x_data, f_cos);
}

#[test]
fn grad_div() {
    let x_data = vec![1.2, -2.4, 3.6];
    let y_data = vec![0.5, -1.5, 2.0];

    let x = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let y = TracedTensor::from_tensor(f64_tensor(vec![3], y_data.clone()));
    let loss = x.traced_div(&y).traced_reduce_sum(&[0]);

    let grad_x = loss.grad(&x).unwrap();
    let grad_y = loss.grad(&y).unwrap();
    let grad_x_tensor = eval_tensor(grad_x);
    let grad_y_tensor = eval_tensor(grad_y);
    let grad_x_data = get_f64_data(&grad_x_tensor);
    let grad_y_data = get_f64_data(&grad_y_tensor);

    let expected_x: Vec<f64> = y_data.iter().map(|y| 1.0 / y).collect();
    let expected_y: Vec<f64> = x_data
        .iter()
        .zip(y_data.iter())
        .map(|(x, y)| -x / (y * y))
        .collect();
    assert_close_slice(grad_x_data, &expected_x);
    assert_close_slice(grad_y_data, &expected_y);

    let f = |lhs: &[f64], rhs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], lhs.to_vec()));
        let y = TracedTensor::from_tensor(f64_tensor(vec![3], rhs.to_vec()));
        eval_scalar(x.traced_div(&y).traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff_lhs(grad_x_data, &x_data, &y_data, &f);
    assert_grad_matches_finite_diff_rhs(grad_y_data, &x_data, &y_data, &f);
}

#[test]
fn grad_sqrt() {
    let x_data = vec![0.8, 1.5, 3.2];
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let loss = x.traced_sqrt().traced_reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| 0.5 / x.sqrt()).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.traced_sqrt().traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_tanh() {
    let x_data = vec![0.2, -0.7, 1.3];
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let loss = x.traced_tanh().traced_reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| 1.0 - x.tanh().powi(2)).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.traced_tanh().traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_pow() {
    let x_data = vec![0.7, 1.3, 2.1];
    let y_data = vec![2.0, 2.0, 2.0];
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let y = TracedTensor::from_tensor(f64_tensor(vec![3], y_data.clone()));
    let loss = x.traced_pow(&y).traced_reduce_sum(&[0]);

    let grad_x = loss.grad(&x).unwrap();
    let grad_x_tensor = eval_tensor(grad_x);
    let grad_x_data = get_f64_data(&grad_x_tensor);
    let expected_x: Vec<f64> = x_data.iter().map(|x| 2.0 * x).collect();
    assert_close_slice(grad_x_data, &expected_x);

    let f = |lhs: &[f64], rhs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], lhs.to_vec()));
        let y = TracedTensor::from_tensor(f64_tensor(vec![3], rhs.to_vec()));
        eval_scalar(x.traced_pow(&y).traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff_lhs(grad_x_data, &x_data, &y_data, &f);
}

#[test]
fn grad_pow_wrt_exponent() {
    let x_data = vec![1.2, 1.8, 2.5];
    let y_data = vec![0.5, 1.5, 2.0];
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let y = TracedTensor::from_tensor(f64_tensor(vec![3], y_data.clone()));
    let loss = x.traced_pow(&y).traced_reduce_sum(&[0]);

    let grad_y = loss.grad(&y).unwrap();
    let grad_y_tensor = eval_tensor(grad_y);
    let grad_y_data = get_f64_data(&grad_y_tensor);
    let expected_y: Vec<f64> = x_data
        .iter()
        .zip(y_data.iter())
        .map(|(x, y)| x.ln() * x.powf(*y))
        .collect();
    assert_close_slice(grad_y_data, &expected_y);

    let f = |lhs: &[f64], rhs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], lhs.to_vec()));
        let y = TracedTensor::from_tensor(f64_tensor(vec![3], rhs.to_vec()));
        eval_scalar(x.traced_pow(&y).traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff_rhs(grad_y_data, &x_data, &y_data, &f);
}

#[test]
fn grad_abs() {
    let x_data = vec![-1.7, 0.8, 2.3];
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let loss = x.traced_abs().traced_reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected = [-1.0, 1.0, 1.0];
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.traced_abs().traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_sign() {
    let x_data = vec![-1.7, 0.8, 2.3];
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let loss = x.traced_sign().traced_reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    assert_close_slice(grad_data, &[0.0, 0.0, 0.0]);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.traced_sign().traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_rsqrt() {
    let x_data = vec![0.8, 1.5, 3.2];
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let loss = x.traced_rsqrt().traced_reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| -0.5 / (x * x.sqrt())).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.traced_rsqrt().traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_expm1() {
    let x_data = vec![0.2, -0.7, 1.3];
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let loss = x.traced_expm1().traced_reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| x.exp()).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.traced_expm1().traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_log1p() {
    let x_data = vec![0.2, 0.7, 1.3];
    let x = TracedTensor::from_tensor(f64_tensor(vec![3], x_data.clone()));
    let loss = x.traced_log1p().traced_reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| 1.0 / (x + 1.0)).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.traced_log1p().traced_reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_svd_values_matches_finite_diff() {
    let a_data = vec![3.0, -1.0, 0.5, 2.0];
    let (fragment, input_key, loss_key) = build_svd_values_sum_fragment();
    let grad = grad_from_fragment(
        fragment,
        loss_key,
        input_key,
        f64_tensor(vec![2, 2], a_data.clone()),
    );
    let grad_data = get_f64_data(&grad);

    for index in 0..a_data.len() {
        let expected = finite_diff_scalar(sum_svd_values, &a_data, index, 1e-6);
        assert!(
            (grad_data[index] - expected).abs() <= 1e-5,
            "index {index}: expected {expected}, got {}",
            grad_data[index]
        );
    }
}

#[test]
fn grad_svd_uv_product_matches_finite_diff() {
    let a_data = vec![4.0, 0.75, 1.25, 2.5];
    let (fragment, input_key, loss_key) = build_svd_uv_product_sum_fragment();
    let grad = grad_from_fragment(
        fragment,
        loss_key,
        input_key,
        f64_tensor(vec![2, 2], a_data.clone()),
    );
    let grad_data = get_f64_data(&grad);

    for index in 0..a_data.len() {
        let expected = finite_diff_scalar(sum_svd_uv_product, &a_data, index, 1e-6);
        assert!(
            (grad_data[index] - expected).abs() <= 1e-4,
            "index {index}: expected {expected}, got {}",
            grad_data[index]
        );
    }
}

#[test]
fn grad_eigh_values_matches_finite_diff() {
    let a_data = vec![4.0, 1.0, 1.0, 2.0];
    let (fragment, input_key, loss_key) = build_eigh_values_sum_fragment();
    let grad = grad_from_fragment(
        fragment,
        loss_key,
        input_key,
        f64_tensor(vec![2, 2], a_data.clone()),
    );
    let grad_data = get_f64_data(&grad);

    for index in 0..a_data.len() {
        let expected = finite_diff_scalar(sum_eigh_values, &a_data, index, 1e-6);
        assert!(
            (grad_data[index] - expected).abs() <= 1e-6,
            "index {index}: expected {expected}, got {}",
            grad_data[index]
        );
    }
}

#[test]
fn grad_eigh_projector_matches_finite_diff() {
    let a_data = vec![5.0, 1.5, 1.5, 2.0];
    let weights_data = vec![0.8, -0.3];
    let probe_data = vec![1.2, -0.4, 0.6, 0.9];
    let (fragment, input_key, weights_key, probe_key, loss_key) =
        build_eigh_projector_sum_fragment();
    let mut inputs_map = HashMap::new();
    inputs_map.insert(input_key.clone(), f64_tensor(vec![2, 2], a_data.clone()));
    inputs_map.insert(weights_key, f64_tensor(vec![2], weights_data.clone()));
    inputs_map.insert(probe_key, f64_tensor(vec![2, 2], probe_data.clone()));
    let grad = grad_from_fragment_with_inputs(fragment, loss_key, input_key, inputs_map);
    let grad_data = get_f64_data(&grad);

    // `eigh` reads the lower triangle, so compare only the represented degrees of freedom.
    for index in [0usize, 1, 3] {
        let expected = finite_diff_scalar(
            |xs| sum_eigh_projector(xs, &weights_data, &probe_data),
            &a_data,
            index,
            1e-6,
        );
        assert!(
            (grad_data[index] - expected).abs() <= 1e-4,
            "index {index}: expected {expected}, got {}",
            grad_data[index]
        );
    }
}

#[test]
fn grad_solve_matches_finite_diff() {
    let a_data = vec![4.0, 1.0, 0.5, 3.0];
    let b_data = vec![1.5, -2.0];
    let (fragment, a_key, b_key, loss_key) = build_solve_sum_fragment();

    let mut a_inputs = HashMap::new();
    a_inputs.insert(a_key.clone(), f64_tensor(vec![2, 2], a_data.clone()));
    a_inputs.insert(b_key.clone(), f64_tensor(vec![2, 1], b_data.clone()));
    let grad_a =
        grad_from_fragment_with_inputs(fragment.clone(), loss_key.clone(), a_key, a_inputs);
    let grad_a_data = get_f64_data(&grad_a);

    let mut b_inputs = HashMap::new();
    b_inputs.insert(
        tensor_input_key(30_000),
        f64_tensor(vec![2, 2], a_data.clone()),
    );
    b_inputs.insert(b_key.clone(), f64_tensor(vec![2, 1], b_data.clone()));
    let grad_b = grad_from_fragment_with_inputs(fragment, loss_key, b_key, b_inputs);
    let grad_b_data = get_f64_data(&grad_b);

    assert_grad_matches_finite_diff_lhs(grad_a_data, &a_data, &b_data, &sum_solve);
    assert_grad_matches_finite_diff_rhs(grad_b_data, &a_data, &b_data, &sum_solve);
}

#[test]
fn grad_triangular_solve_rhs_matches_finite_diff() {
    let a_data = vec![2.0, -1.0, 0.0, 3.0];
    let b_data = vec![1.0, 4.0];
    let (fragment, a_key, b_key, loss_key) = build_triangular_solve_sum_fragment();

    let mut inputs_map = HashMap::new();
    inputs_map.insert(a_key, f64_tensor(vec![2, 2], a_data.clone()));
    inputs_map.insert(b_key.clone(), f64_tensor(vec![2, 1], b_data.clone()));
    let grad_b = grad_from_fragment_with_inputs(fragment, loss_key, b_key, inputs_map);
    let grad_b_data = get_f64_data(&grad_b);

    assert_grad_matches_finite_diff_rhs(grad_b_data, &a_data, &b_data, &sum_triangular_solve);
}
