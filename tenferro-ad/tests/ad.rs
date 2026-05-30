use tenferro_ad::TracedTensorAdExt;
#[path = "ad/reductions_and_indexing.rs"]
mod reductions_and_indexing;
mod support;
use std::collections::HashMap;
use std::sync::Arc;
use support::{run_many_traced_with, RunTraced};

use chainrules_core::ADKey;
use computegraph::compile::compile;
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use computegraph::LocalValId;
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_ops::ad::context::{
    lookup_global_metadata, register_scoped_global_metadata_batch, GlobalMetadataScope, TensorMeta,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeGuardContext, SymDim};
use tenferro_runtime::compiler::compile_std_to_exec;
use tenferro_runtime::exec::eval_exec_ir;
use tenferro_runtime::shape_infer::{infer_output_dtype, infer_output_extents};
use tenferro_runtime::traced_tensor::matmul;
use tenferro_runtime::{GraphExecutor, TracedTensor};
use tenferro_tensor::{
    DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor,
    TensorReduction, TypedTensor,
};
use tidu::{differentiate, transpose, LinearFragment};

const TOL: f64 = 1e-6;

fn finite_diff_scalar(f: &impl Fn(&[f64]) -> f64, x: &[f64], idx: usize, h: f64) -> f64 {
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[idx] += h;
    xm[idx] -= h;
    (f(&xp) - f(&xm)) / (2.0 * h)
}

fn finite_diff_scalar_lhs(
    f: &impl Fn(&[f64], &[f64]) -> f64,
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
    f: &impl Fn(&[f64], &[f64]) -> f64,
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

fn finite_diff_tensor_directional(
    f: impl Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    tangent: &[f64],
    h: f64,
) -> Vec<f64> {
    assert_eq!(x.len(), tangent.len());

    let xp: Vec<f64> = x
        .iter()
        .zip(tangent.iter())
        .map(|(&value, &delta)| value + h * delta)
        .collect();
    let xm: Vec<f64> = x
        .iter()
        .zip(tangent.iter())
        .map(|(&value, &delta)| value - h * delta)
        .collect();

    let yp = f(&xp);
    let ym = f(&xm);
    assert_eq!(yp.len(), ym.len());

    yp.iter()
        .zip(ym.iter())
        .map(|(&plus, &minus)| (plus - minus) / (2.0 * h))
        .collect()
}

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

fn i64_tensor(shape: Vec<usize>, data: Vec<i64>) -> Tensor {
    Tensor::I64(TypedTensor::from_vec_col_major(shape, data))
}

fn c64_tensor(shape: Vec<usize>, data: Vec<Complex64>) -> Tensor {
    Tensor::C64(TypedTensor::from_vec_col_major(shape, data))
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

fn tensor_meta_from_tensor(tensor: &Tensor) -> TensorMeta {
    TensorMeta::exact(
        tensor.dtype(),
        tensor.shape().iter().copied().map(SymDim::from).collect(),
    )
}

fn register_fragment_metadata_for_test(
    fragment: &Fragment<StdTensorOp>,
    seeded: impl IntoIterator<Item = (GlobalValKey<StdTensorOp>, TensorMeta)>,
) -> GlobalMetadataScope {
    let seeded: Vec<_> = seeded.into_iter().collect();
    let mut known: HashMap<_, _> = seeded.iter().cloned().collect();

    let mut registrations = seeded;
    for op_node in fragment.ops() {
        let input_metas: Vec<_> = op_node
            .inputs
            .iter()
            .map(|input| {
                let key = match input {
                    ValRef::Local(local_id) => &fragment.vals()[*local_id].key,
                    ValRef::External(key) => key,
                };
                match known.get(key).cloned() {
                    Some(meta) => meta,
                    None => {
                        let meta = lookup_global_metadata(key).unwrap_or_else(|| {
                            panic!("test metadata registration missing input {:?}", key)
                        });
                        known.insert(key.clone(), meta.clone());
                        meta
                    }
                }
            })
            .collect();
        let input_shape_exprs: Vec<Vec<DimExpr>> = input_metas
            .iter()
            .enumerate()
            .map(|(input_idx, meta)| DimExpr::input_shape(input_idx, meta.shape.len()))
            .collect();
        let input_shape_refs: Vec<&[DimExpr]> =
            input_shape_exprs.iter().map(Vec::as_slice).collect();
        let input_dtypes: Vec<DType> = input_metas.iter().map(|meta| meta.dtype).collect();
        let output_dtype = infer_output_dtype(&op_node.op, &input_dtypes);
        let resolved_inputs: Vec<&[SymDim]> = input_metas
            .iter()
            .map(|meta| meta.shape.as_slice())
            .collect();

        for (&output_id, extents) in op_node
            .outputs
            .iter()
            .zip(infer_output_extents(&op_node.op, &input_shape_refs))
        {
            let resolved_extents = extents
                .into_iter()
                .map(|extent| extent.map(|dim| SymDim::from_dim_expr(&dim, &resolved_inputs)))
                .collect();
            let meta = TensorMeta::with_extents(output_dtype, resolved_extents);
            let key = fragment.vals()[output_id].key.clone();
            known.insert(key.clone(), meta.clone());
            registrations.push((key, meta));
        }
    }

    register_scoped_global_metadata_batch(registrations)
}

fn eval_tensor(traced: TracedTensor) -> Tensor {
    let mut engine = GraphExecutor::new(CpuBackend::new());
    traced.run_with(&mut engine).unwrap().clone()
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
    }
}

fn batched_matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![1],
        lhs_batch_dims: vec![0],
        rhs_batch_dims: vec![0],
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
    let mut input_dtypes = Vec::with_capacity(graph.inputs.len());
    let mut input_shapes = Vec::with_capacity(graph.inputs.len());
    let inputs = graph
        .inputs
        .iter()
        .map(|key| match key {
            GlobalValKey::Input(k) => {
                let tensor = inputs_map.get(k).expect("missing tensor input");
                input_dtypes.push(tensor.dtype());
                input_shapes.push(DimExpr::from_concrete(tensor.shape()));
                tensor.clone()
            }
            _ => panic!("expected input key"),
        })
        .collect();
    let exec = compile_std_to_exec(&compiled, &input_dtypes, &input_shapes);
    let mut backend = CpuBackend::new();
    eval_exec_ir(&mut backend, &exec, inputs).unwrap()
}

fn scalar_f64_tensor(value: f64) -> Tensor {
    f64_tensor(vec![], vec![value])
}

fn grad_from_fragment_with_inputs_and_cotangent(
    fragment: Arc<Fragment<StdTensorOp>>,
    loss_key: GlobalValKey<StdTensorOp>,
    input_key: TensorInputKey,
    mut inputs_map: HashMap<TensorInputKey, Tensor>,
    cotangent: Tensor,
) -> Tensor {
    let _primal_metadata_scope = register_fragment_metadata_for_test(
        fragment.as_ref(),
        inputs_map.iter().map(|(key, tensor)| {
            (
                GlobalValKey::Input(key.clone()),
                tensor_meta_from_tensor(tensor),
            )
        }),
    );
    let view = resolve(vec![fragment.clone()]);
    let mut ad_ctx = ShapeGuardContext::with_global_metadata();
    let linear = differentiate(
        &view,
        std::slice::from_ref(&loss_key),
        std::slice::from_ref(&input_key),
        0,
        &mut ad_ctx,
        &HashMap::new(),
    );
    let _linear_metadata_scope = register_fragment_metadata_for_test(
        &linear.fragment,
        vec![(
            GlobalValKey::Input(input_key.tangent_of(0)),
            tensor_meta_from_tensor(inputs_map.get(&input_key).expect("missing input tensor")),
        )],
    );
    ad_ctx.refresh_global_metadata();
    let linear_tangent_input_ids: Vec<LocalValId> = linear
        .tangent_inputs
        .iter()
        .map(|(_, local_id)| *local_id)
        .collect();
    let transposed = transpose(&linear, &mut ad_ctx);
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

    inputs_map.insert(cotangent_input_key, cotangent);

    // Linear-mode tangent ops in the transposed fragment may reference values
    // whose dependency chain passes through the linear fragment's tangent
    // inputs (e.g. shape-source references). Provide zero placeholders for
    // those inputs so materialize_merge can satisfy the dependency.
    let input_tensor = inputs_map
        .get(&input_key)
        .cloned()
        .expect("missing primal input tensor for zero-tangent fill");
    let zero_tangent = zeros_by_dtype(input_tensor.dtype(), input_tensor.shape().to_vec());
    for local_id in linear_tangent_input_ids {
        if let GlobalValKey::Input(key) = &linear_fragment.vals()[local_id].key {
            inputs_map
                .entry(key.clone())
                .or_insert_with(|| zero_tangent.clone());
        }
    }

    eval_fragment_outputs(
        vec![fragment, linear_fragment, transposed_fragment.clone()],
        &[grad_key],
        &inputs_map,
    )
    .into_iter()
    .next()
    .expect("gradient output")
}

fn zeros_by_dtype(dtype: DType, shape: Vec<usize>) -> Tensor {
    match dtype {
        DType::F32 => Tensor::F32(TypedTensor::zeros(shape)),
        DType::F64 => Tensor::F64(TypedTensor::zeros(shape)),
        DType::I32 => Tensor::I32(TypedTensor::zeros(shape)),
        DType::I64 => Tensor::I64(TypedTensor::zeros(shape)),
        DType::Bool => {
            let n_elements = shape.iter().product();
            Tensor::Bool(TypedTensor::from_vec_col_major(
                shape,
                vec![false; n_elements],
            ))
        }
        DType::C32 => Tensor::C32(TypedTensor::zeros(shape)),
        DType::C64 => Tensor::C64(TypedTensor::zeros(shape)),
    }
}

fn grad_from_fragment_with_inputs(
    fragment: Arc<Fragment<StdTensorOp>>,
    loss_key: GlobalValKey<StdTensorOp>,
    input_key: TensorInputKey,
    inputs_map: HashMap<TensorInputKey, Tensor>,
) -> Tensor {
    grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        loss_key,
        input_key,
        inputs_map,
        scalar_f64_tensor(1.0),
    )
}

fn jvp_from_fragment_with_inputs(
    fragment: Arc<Fragment<StdTensorOp>>,
    output_key: GlobalValKey<StdTensorOp>,
    input_key: TensorInputKey,
    mut inputs_map: HashMap<TensorInputKey, Tensor>,
    tangent: Tensor,
) -> Tensor {
    let _primal_metadata_scope = register_fragment_metadata_for_test(
        fragment.as_ref(),
        inputs_map.iter().map(|(key, tensor)| {
            (
                GlobalValKey::Input(key.clone()),
                tensor_meta_from_tensor(tensor),
            )
        }),
    );
    let view = resolve(vec![fragment.clone()]);
    let mut ad_ctx = ShapeGuardContext::with_global_metadata();
    let linear = differentiate(
        &view,
        std::slice::from_ref(&output_key),
        std::slice::from_ref(&input_key),
        0,
        &mut ad_ctx,
        &HashMap::new(),
    );
    let _linear_metadata_scope = register_fragment_metadata_for_test(
        &linear.fragment,
        vec![(
            GlobalValKey::Input(input_key.tangent_of(0)),
            tensor_meta_from_tensor(&tangent),
        )],
    );
    let tangent_key = linear.tangent_outputs[0]
        .map(|id| linear.fragment.vals()[id].key.clone())
        .expect("expected active tangent output");
    let tangent_input_key = match &linear.fragment.vals()[linear.tangent_inputs[0].1].key {
        GlobalValKey::Input(key) => key.clone(),
        _ => panic!("expected tangent input"),
    };
    let linear_fragment = Arc::new(linear.fragment);

    inputs_map.insert(tangent_input_key, tangent);

    eval_fragment_outputs(vec![fragment, linear_fragment], &[tangent_key], &inputs_map)
        .into_iter()
        .next()
        .expect("tangent output")
}

fn build_unary_fragment(
    op: StdTensorOp,
    input_key: TensorInputKey,
) -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let input = builder.add_input(input_key.clone());
    let output = builder.add_op(op, vec![ValRef::Local(input)], OpMode::Primal)[0];
    let output_key = builder.global_key(output).clone();
    builder.set_outputs(vec![output]);
    (Arc::new(builder.build()), input_key, output_key)
}

fn eval_f64_reduction_op(op: &StdTensorOp, input_shape: &[usize], data: &[f64]) -> Vec<f64> {
    let mut backend = CpuBackend::new();
    let output = match op {
        StdTensorOp::ReduceProd { axes } => {
            let input = f64_tensor(input_shape.to_vec(), data.to_vec());
            backend.reduce_prod(&input, axes).unwrap()
        }
        StdTensorOp::ReduceMax { axes } => {
            let input = f64_tensor(input_shape.to_vec(), data.to_vec());
            backend.reduce_max(&input, axes).unwrap()
        }
        StdTensorOp::ReduceMin { axes } => {
            let input = f64_tensor(input_shape.to_vec(), data.to_vec());
            backend.reduce_min(&input, axes).unwrap()
        }
        _ => panic!("expected reduction op"),
    };
    get_f64_data(&output).to_vec()
}

fn transpose_primal_unary_op_with_inputs(
    op: StdTensorOp,
    input_key: TensorInputKey,
    input: Tensor,
    cotangent: Tensor,
) -> Tensor {
    let tangent_input_key = input_key.tangent_of(90_000);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let tangent_input = builder.add_input(tangent_input_key.clone());
    let output = builder.add_op(op, vec![ValRef::Local(tangent_input)], OpMode::Primal)[0];
    builder.set_outputs(vec![output]);

    let linear = LinearFragment {
        fragment: builder.build(),
        tangent_inputs: vec![(input_key, tangent_input)],
        tangent_outputs: vec![Some(output)],
    };
    let _linear_metadata_scope = register_fragment_metadata_for_test(
        &linear.fragment,
        vec![(
            GlobalValKey::Input(tangent_input_key.clone()),
            tensor_meta_from_tensor(&input),
        )],
    );
    let mut ad_ctx = ShapeGuardContext::with_global_metadata();
    let transposed = transpose(&linear, &mut ad_ctx);
    let linear_fragment = Arc::new(linear.fragment);
    let cotangent_input_key = match &transposed.fragment.vals()[transposed.tangent_inputs[0].1].key
    {
        GlobalValKey::Input(key) => key.clone(),
        _ => panic!("expected cotangent seed input"),
    };
    let output_key = transposed.tangent_outputs[0]
        .map(|id| transposed.fragment.vals()[id].key.clone())
        .expect("expected active transpose output");
    let transposed_fragment = Arc::new(transposed.fragment);

    let mut inputs_map = HashMap::new();
    inputs_map.insert(tangent_input_key, input);
    inputs_map.insert(cotangent_input_key, cotangent);

    eval_fragment_outputs(
        vec![linear_fragment, transposed_fragment],
        &[output_key],
        &inputs_map,
    )
    .into_iter()
    .next()
    .expect("transpose output")
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

fn assert_jvp_matches_finite_diff(
    actual: &[f64],
    base: &[f64],
    tangent: &[f64],
    f: impl Fn(&[f64]) -> Vec<f64>,
) {
    let expected = finite_diff_tensor_directional(f, base, tangent, 1e-6);
    assert_close_slice(actual, &expected);
}

fn assert_grad_matches_finite_diff(actual: &[f64], base: &[f64], f: impl Fn(&[f64]) -> f64) {
    assert_eq!(actual.len(), base.len());
    for (index, &actual_value) in actual.iter().enumerate().take(base.len()) {
        let expected = finite_diff_scalar(&f, base, index, 1e-6);
        assert!(
            (actual_value - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {}",
            actual_value
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
    for (index, &actual_value) in actual.iter().enumerate().take(lhs.len()) {
        let expected = finite_diff_scalar_lhs(&f, lhs, rhs, index, 1e-6);
        assert!(
            (actual_value - expected).abs() <= TOL,
            "lhs index {index}: expected {expected}, got {}",
            actual_value
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
    for (index, &actual_value) in actual.iter().enumerate().take(rhs.len()) {
        let expected = finite_diff_scalar_rhs(&f, lhs, rhs, index, 1e-6);
        assert!(
            (actual_value - expected).abs() <= TOL,
            "rhs index {index}: expected {expected}, got {}",
            actual_value
        );
    }
}

#[test]
fn grad_x_squared() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = &x * &x;
    let loss = y.reduce_sum(&[0]);
    assert_eq!(loss.rank, 0);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_close_slice(get_f64_data(&result), &[2.0, 4.0, 6.0]);
}

#[test]
fn grad_extract_diag_sum() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));
    let diag = a.extract_diag(0, 1);
    let loss = diag.reduce_sum(&[0]);
    assert_eq!(loss.rank, 0);
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
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![2.0, -1.0, 4.0]));
    let diag = x.embed_diag(0, 1);
    let loss = diag.reduce_sum(&[0, 1]);
    assert_eq!(loss.rank, 0);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_eq!(result.shape(), &[3]);
    assert_close_slice(get_f64_data(&result), &[1.0, 1.0, 1.0]);
}

#[test]
fn jvp_extract_diag() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 3],
        vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
    ));
    let da = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ));

    let diag = a.extract_diag(0, 1);
    let jvp = diag.jvp(&a, &da);

    let result = eval_tensor(jvp);
    assert_eq!(result.shape(), &[3]);
    assert_close_slice(get_f64_data(&result), &[1.0, 5.0, 9.0]);
}

#[test]
fn jvp_embed_diag() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![3.0, 4.0, 5.0]));
    let dx = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![0.5, -1.0, 2.0]));

    let diag = x.embed_diag(0, 1);
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

    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 3], a_data.clone()));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], b_data.clone()));
    let loss = matmul(&a, &b).sum(&[0, 1]);
    assert_eq!(loss.rank, 0);
    let grad = loss.grad(&a).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);

    let f = |xs: &[f64]| {
        let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 3], xs.to_vec()));
        let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], b_data.clone()));
        eval_scalar(matmul(&a, &b).sum(&[0, 1]))
    };

    for (index, &grad_value) in grad_data.iter().enumerate().take(a_data.len()) {
        let expected = finite_diff_scalar(&f, &a_data, index, 1e-6);
        assert!(
            (grad_value - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {}",
            grad_value
        );
    }
}

#[test]
fn grad_matmul_sum_wrt_rhs() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 3], a_data.clone()));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], b_data.clone()));
    let matmul = a.dot_general(&b, matmul_config());
    let loss = matmul.reduce_sum(&[0, 1]);
    assert_eq!(loss.rank, 0);
    let grad = loss.grad(&b).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);

    let f = |xs: &[f64]| {
        let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 3], a_data.clone()));
        let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3, 2], xs.to_vec()));
        let matmul = a.dot_general(&b, matmul_config());
        let loss = matmul.reduce_sum(&[0, 1]);
        eval_scalar(loss)
    };

    for (index, &grad_value) in grad_data.iter().enumerate().take(b_data.len()) {
        let expected = finite_diff_scalar(&f, &b_data, index, 1e-6);
        assert!(
            (grad_value - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {}",
            grad_value
        );
    }
}

#[test]
fn grad_matmul_sum_shared_input() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0];

    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], a_data.clone()));
    let matmul = a.dot_general(&a, matmul_config());
    let loss = matmul.reduce_sum(&[0, 1]);
    let grad = loss.grad(&a).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);

    let f = |xs: &[f64]| {
        let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], xs.to_vec()));
        let matmul = a.dot_general(&a, matmul_config());
        let loss = matmul.reduce_sum(&[0, 1]);
        eval_scalar(loss)
    };

    for (index, &grad_value) in grad_data.iter().enumerate().take(a_data.len()) {
        let expected = finite_diff_scalar(&f, &a_data, index, 1e-6);
        assert!(
            (grad_value - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {}",
            grad_value
        );
    }
}

#[test]
fn grad_batched_matmul_sum() {
    let a_shape = vec![2, 3, 4];
    let b_shape = vec![2, 4, 5];
    let a_data: Vec<f64> = (0..24).map(|idx| 0.25 + idx as f64 * 0.1).collect();
    let b_data: Vec<f64> = (0..40).map(|idx| 0.5 + idx as f64 * 0.05).collect();

    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(a_shape.clone(), a_data.clone()));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(b_shape.clone(), b_data.clone()));
    let product = a.dot_general(&b, batched_matmul_config());
    let loss = product.reduce_sum(&[0, 1, 2]);
    let grad = loss.grad(&a).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);

    let f = |xs: &[f64]| {
        let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(a_shape.clone(), xs.to_vec()));
        let b =
            TracedTensor::from_tensor_concrete_shape(f64_tensor(b_shape.clone(), b_data.clone()));
        let product = a.dot_general(&b, batched_matmul_config());
        let loss = product.reduce_sum(&[0, 1, 2]);
        eval_scalar(loss)
    };

    for (index, &grad_value) in grad_data.iter().enumerate().take(a_data.len()) {
        let expected = finite_diff_scalar(&f, &a_data, index, 1e-6);
        assert!(
            (grad_value - expected).abs() <= TOL,
            "index {index}: expected {expected}, got {}",
            grad_value
        );
    }
}

#[test]
fn grad_mul_sum_wrt_both_inputs() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![4.0, 5.0, 6.0]));
    let loss = (&x * &y).reduce_sum(&[0]);

    let grad_x = loss.grad(&x).unwrap();
    let grad_y = loss.grad(&y).unwrap();

    let grad_x_tensor = eval_tensor(grad_x);
    let grad_y_tensor = eval_tensor(grad_y);
    assert_close_slice(get_f64_data(&grad_x_tensor), &[4.0, 5.0, 6.0]);
    assert_close_slice(get_f64_data(&grad_y_tensor), &[1.0, 2.0, 3.0]);
}

#[test]
fn jvp_elementwise_mul() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![4.0, 5.0, 6.0]));
    let dx = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 0.0, 0.0]));

    let prod = &x * &y;
    let jvp = prod.jvp(&x, &dx);

    let result = eval_tensor(jvp);
    assert_close_slice(get_f64_data(&result), &[4.0, 0.0, 0.0]);
}

#[test]
fn jvp_elementwise_mul_y_tangent() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![4.0, 5.0, 6.0]));
    let dy = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![0.0, -1.0, 2.0]));

    let prod = &x * &y;
    let jvp = prod.jvp(&y, &dy);

    let result = eval_tensor(jvp);
    assert_close_slice(get_f64_data(&result), &[0.0, -2.0, 6.0]);
}

#[test]
fn jvp_elementwise_add_y_tangent() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![4.0, 5.0, 6.0]));
    let dy = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![0.5, -1.0, 2.0]));

    let sum = &x + &y;
    let jvp = sum.jvp(&y, &dy);

    let result = eval_tensor(jvp);
    assert_close_slice(get_f64_data(&result), &[0.5, -1.0, 2.0]);
}

#[test]
fn grad_neg_sum() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, -2.0, 3.0]));
    let loss = (-&x).reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_close_slice(get_f64_data(&result), &[-1.0, -1.0, -1.0]);
}

#[test]
fn grad_conj_sum_complex() {
    let x = TracedTensor::from_tensor_concrete_shape(c64_tensor(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
    ));
    let loss = x.conj().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_close_slice_c64(
        get_c64_data(&result),
        &[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
    );
}

#[test]
fn scale_real_eval_and_grad_sum() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));

    let y = x.scale_real(2.0);
    let y_eval = eval_tensor(y);
    assert_close_slice(get_f64_data(&y_eval), &[2.0, 4.0, 6.0]);

    let loss = x.scale_real(2.0).reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();
    let grad_eval = eval_tensor(grad);
    assert_close_slice(get_f64_data(&grad_eval), &[2.0, 2.0, 2.0]);
}

#[test]
fn scale_complex_eval_and_grad_complex_sum() {
    let factor = Complex64::new(0.5, -1.5);
    let x = TracedTensor::from_tensor_concrete_shape(c64_tensor(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
    ));

    let y = x.scale_complex(factor);
    let y_eval = eval_tensor(y);
    assert_close_slice_c64(
        get_c64_data(&y_eval),
        &[Complex64::new(3.5, -0.5), Complex64::new(-0.75, 4.75)],
    );

    let loss = x.scale_complex(factor).reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();
    let grad_eval = eval_tensor(grad);
    assert_close_slice_c64(get_c64_data(&grad_eval), &[factor.conj(), factor.conj()]);
}

#[test]
fn convert_eval_jvp_and_vjp_follow_real_complex_adjoint_rules() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![1.25, -2.5]));
    let dx = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2], vec![0.5, -1.0]));
    let cotangent = TracedTensor::from_tensor_concrete_shape(c64_tensor(
        vec![2],
        vec![Complex64::new(3.0, -7.0), Complex64::new(-2.5, 4.0)],
    ));

    let roundtrip = x.convert(DType::C64).convert(DType::F64);
    let jvp = x.convert(DType::C64).jvp(&x, &dx);
    let vjp = x.convert(DType::C64).vjp(&x, &cotangent);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let results = run_many_traced_with(&mut engine, &[&roundtrip, &jvp, &vjp]).unwrap();

    assert_close_slice(get_f64_data(&results[0]), &[1.25, -2.5]);
    assert_close_slice_c64(
        get_c64_data(&results[1]),
        &[Complex64::new(0.5, 0.0), Complex64::new(-1.0, 0.0)],
    );
    assert_close_slice(get_f64_data(&results[2]), &[3.0, -2.5]);
}

#[test]
fn vjp_matmul() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let cotangent =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]));

    let y = a.dot_general(&b, matmul_config());
    let vjp = y.vjp(&a, &cotangent);

    let result = eval_tensor(vjp);
    assert_close_slice(get_f64_data(&result), &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
}

#[test]
fn grad_nonscalar_errors() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let y = a.dot_general(&b, matmul_config());

    assert!(y.grad(&a).is_err());
}

#[test]
fn grad_full_vector_reduction() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
    let loss = x.reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_eq!(result.shape(), &[4]);
    assert_close_slice(get_f64_data(&result), &[1.0, 1.0, 1.0, 1.0]);
}
