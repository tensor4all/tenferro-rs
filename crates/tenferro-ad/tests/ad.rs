use tenferro_ad::TracedTensorAdExt;
#[path = "ad/reductions_and_indexing.rs"]
mod reductions_and_indexing;
mod support;
use std::collections::HashMap;
use std::sync::Arc;
use support::{run_many_traced_with, RunTraced};

use computegraph::compile::compile;
use computegraph::graph::{Graph, GraphBuilder};
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{OperationRole, ValueKey, ValueRef};
use computegraph::LocalValueId;
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_ops::ad::context::{
    lookup_global_metadata, register_scoped_global_metadata_batch, GlobalMetadataScope, TensorMeta,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeGuardContext, SymDim};
use tenferro_runtime::extension::compile_std_to_exec;
use tenferro_runtime::extension::{infer_output_dtype, infer_output_extents};
use tenferro_runtime::traced_tensor::matmul;
use tenferro_runtime::{GraphExecutor, TracedTensor};
use tenferro_tensor::{
    DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor,
    TensorReduction, TypedTensor,
};
use tidu::{linear_transpose, linearize, ADKey};

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

fn register_graph_metadata_for_test(
    graph: &Graph<StdTensorOp>,
    seeded: impl IntoIterator<Item = (ValueKey<StdTensorOp>, TensorMeta)>,
) -> GlobalMetadataScope {
    let seeded: Vec<_> = seeded.into_iter().collect();
    let mut known: HashMap<_, _> = seeded.iter().cloned().collect();

    let mut registrations = seeded;
    for op_node in graph.operations() {
        let input_metas: Vec<_> = op_node
            .inputs
            .iter()
            .map(|input| {
                let key = match input {
                    ValueRef::Local(local_id) => &graph.values()[*local_id].key,
                    ValueRef::External(key) => key,
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
        let output_dtype = infer_output_dtype(&op_node.operation, &input_dtypes);
        let resolved_inputs: Vec<&[SymDim]> = input_metas
            .iter()
            .map(|meta| meta.shape.as_slice())
            .collect();

        for (&output_id, extents) in op_node
            .outputs
            .iter()
            .zip(infer_output_extents(&op_node.operation, &input_shape_refs))
        {
            let resolved_extents = extents
                .into_iter()
                .map(|extent| extent.map(|dim| SymDim::from_dim_expr(&dim, &resolved_inputs)))
                .collect();
            let meta = TensorMeta::with_extents(output_dtype, resolved_extents);
            let key = graph.values()[output_id].key.clone();
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

fn eval_graph_outputs(
    roots: Vec<Arc<Graph<StdTensorOp>>>,
    outputs: &[ValueKey<StdTensorOp>],
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
            ValueKey::Input(k) => {
                let tensor = inputs_map.get(k).expect("missing tensor input");
                input_dtypes.push(tensor.dtype());
                input_shapes.push(DimExpr::from_concrete(tensor.shape()));
                tensor.clone()
            }
            _ => panic!("expected input key"),
        })
        .collect();
    let exec = compile_std_to_exec(&compiled, &input_dtypes, &input_shapes);
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.eval_exec_ir(&exec, inputs).unwrap()
}

fn scalar_f64_tensor(value: f64) -> Tensor {
    f64_tensor(vec![], vec![value])
}

fn grad_from_graph_with_inputs_and_cotangent(
    graph: Arc<Graph<StdTensorOp>>,
    loss_key: ValueKey<StdTensorOp>,
    input_key: TensorInputKey,
    mut inputs_map: HashMap<TensorInputKey, Tensor>,
    cotangent: Tensor,
) -> Tensor {
    let _primal_metadata_scope = register_graph_metadata_for_test(
        graph.as_ref(),
        inputs_map.iter().map(|(key, tensor)| {
            (
                ValueKey::Input(key.clone()),
                tensor_meta_from_tensor(tensor),
            )
        }),
    );
    let view = resolve(vec![graph.clone()]);
    let mut ad_ctx = ShapeGuardContext::with_global_metadata();
    let linear = linearize(
        &view,
        std::slice::from_ref(&loss_key),
        std::slice::from_ref(&input_key),
        0,
        &mut ad_ctx,
        &HashMap::new(),
    );
    let _linear_metadata_scope = register_graph_metadata_for_test(
        linear.as_graph(),
        vec![(
            ValueKey::Input(input_key.tangent_of(0)),
            tensor_meta_from_tensor(inputs_map.get(&input_key).expect("missing input tensor")),
        )],
    );
    ad_ctx.refresh_global_metadata();
    let linear_tangent_input_ids: Vec<LocalValueId> = linear
        .tangent_inputs()
        .iter()
        .map(|(_, local_id)| *local_id)
        .collect();
    let transposed = linear_transpose(&linear, &mut ad_ctx);
    let linear_graph = Arc::new(linear.into_graph());
    let grad_key = transposed.tangent_outputs()[0]
        .map(|id| transposed.as_graph().values()[id].key.clone())
        .expect("expected active gradient output");
    let cotangent_input_key =
        match &transposed.as_graph().values()[transposed.tangent_inputs()[0].1].key {
            ValueKey::Input(key) => key.clone(),
            _ => panic!("expected cotangent input"),
        };
    let transposed_graph = Arc::new(transposed.into_graph());

    inputs_map.insert(cotangent_input_key, cotangent);

    // Linear-mode tangent ops in the transposed graph may reference values
    // whose dependency chain passes through the linear graph's tangent
    // inputs (e.g. shape-source references). Provide zero placeholders for
    // those inputs so materialize_merge can satisfy the dependency.
    let input_tensor = inputs_map
        .get(&input_key)
        .cloned()
        .expect("missing primal input tensor for zero-tangent fill");
    let zero_tangent = zeros_by_dtype(input_tensor.dtype(), input_tensor.shape().to_vec());
    for local_id in linear_tangent_input_ids {
        if let ValueKey::Input(key) = &linear_graph.values()[local_id].key {
            inputs_map
                .entry(key.clone())
                .or_insert_with(|| zero_tangent.clone());
        }
    }

    eval_graph_outputs(
        vec![graph, linear_graph, transposed_graph.clone()],
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

fn grad_from_graph_with_inputs(
    graph: Arc<Graph<StdTensorOp>>,
    loss_key: ValueKey<StdTensorOp>,
    input_key: TensorInputKey,
    inputs_map: HashMap<TensorInputKey, Tensor>,
) -> Tensor {
    grad_from_graph_with_inputs_and_cotangent(
        graph,
        loss_key,
        input_key,
        inputs_map,
        scalar_f64_tensor(1.0),
    )
}

fn jvp_from_graph_with_inputs(
    graph: Arc<Graph<StdTensorOp>>,
    output_key: ValueKey<StdTensorOp>,
    input_key: TensorInputKey,
    mut inputs_map: HashMap<TensorInputKey, Tensor>,
    tangent: Tensor,
) -> Tensor {
    let _primal_metadata_scope = register_graph_metadata_for_test(
        graph.as_ref(),
        inputs_map.iter().map(|(key, tensor)| {
            (
                ValueKey::Input(key.clone()),
                tensor_meta_from_tensor(tensor),
            )
        }),
    );
    let view = resolve(vec![graph.clone()]);
    let mut ad_ctx = ShapeGuardContext::with_global_metadata();
    let linear = linearize(
        &view,
        std::slice::from_ref(&output_key),
        std::slice::from_ref(&input_key),
        0,
        &mut ad_ctx,
        &HashMap::new(),
    );
    let _linear_metadata_scope = register_graph_metadata_for_test(
        linear.as_graph(),
        vec![(
            ValueKey::Input(input_key.tangent_of(0)),
            tensor_meta_from_tensor(&tangent),
        )],
    );
    let tangent_key = linear.tangent_outputs()[0]
        .map(|id| linear.as_graph().values()[id].key.clone())
        .expect("expected active tangent output");
    let tangent_input_key = match &linear.as_graph().values()[linear.tangent_inputs()[0].1].key {
        ValueKey::Input(key) => key.clone(),
        _ => panic!("expected tangent input"),
    };
    let linear_graph = Arc::new(linear.into_graph());

    inputs_map.insert(tangent_input_key, tangent);

    eval_graph_outputs(vec![graph, linear_graph], &[tangent_key], &inputs_map)
        .into_iter()
        .next()
        .expect("tangent output")
}

fn build_unary_graph(
    op: StdTensorOp,
    input_key: TensorInputKey,
) -> (
    Arc<Graph<StdTensorOp>>,
    TensorInputKey,
    ValueKey<StdTensorOp>,
) {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let input = builder.add_input(input_key.clone());
    let output = builder.add_operation(op, vec![ValueRef::Local(input)], OperationRole::Primary)[0];
    let output_key = builder.global_key(output).clone();
    builder.set_outputs(vec![output]);
    (Arc::new(builder.build()), input_key, output_key)
}

fn build_binary_graph(
    op: StdTensorOp,
    lhs_key: TensorInputKey,
    rhs_key: TensorInputKey,
) -> (
    Arc<Graph<StdTensorOp>>,
    TensorInputKey,
    TensorInputKey,
    ValueKey<StdTensorOp>,
) {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let lhs = builder.add_input(lhs_key.clone());
    let rhs = builder.add_input(rhs_key.clone());
    let output = builder.add_operation(
        op,
        vec![ValueRef::Local(lhs), ValueRef::Local(rhs)],
        OperationRole::Primary,
    )[0];
    let output_key = builder.global_key(output).clone();
    builder.set_outputs(vec![output]);
    (Arc::new(builder.build()), lhs_key, rhs_key, output_key)
}

fn mixed_binary_vjp_lhs(
    op: StdTensorOp,
    key_base: u64,
    lhs: Tensor,
    rhs: Tensor,
    cotangent: Tensor,
) -> Tensor {
    let (graph, lhs_key, rhs_key, output_key) = build_binary_graph(
        op,
        tensor_input_key(key_base),
        tensor_input_key(key_base + 1),
    );
    let mut inputs_map = HashMap::new();
    inputs_map.insert(lhs_key.clone(), lhs);
    inputs_map.insert(rhs_key, rhs);
    grad_from_graph_with_inputs_and_cotangent(graph, output_key, lhs_key, inputs_map, cotangent)
}

fn mixed_binary_vjp_rhs(
    op: StdTensorOp,
    key_base: u64,
    lhs: Tensor,
    rhs: Tensor,
    cotangent: Tensor,
) -> Tensor {
    let (graph, lhs_key, rhs_key, output_key) = build_binary_graph(
        op,
        tensor_input_key(key_base),
        tensor_input_key(key_base + 1),
    );
    let mut inputs_map = HashMap::new();
    inputs_map.insert(lhs_key, lhs);
    inputs_map.insert(rhs_key.clone(), rhs);
    grad_from_graph_with_inputs_and_cotangent(graph, output_key, rhs_key, inputs_map, cotangent)
}

fn mixed_binary_jvp_lhs(
    op: StdTensorOp,
    key_base: u64,
    lhs: Tensor,
    rhs: Tensor,
    tangent: Tensor,
) -> Tensor {
    let (graph, lhs_key, rhs_key, output_key) = build_binary_graph(
        op,
        tensor_input_key(key_base),
        tensor_input_key(key_base + 1),
    );
    let mut inputs_map = HashMap::new();
    inputs_map.insert(lhs_key.clone(), lhs);
    inputs_map.insert(rhs_key, rhs);
    jvp_from_graph_with_inputs(graph, output_key, lhs_key, inputs_map, tangent)
}

fn mixed_binary_jvp_rhs(
    op: StdTensorOp,
    key_base: u64,
    lhs: Tensor,
    rhs: Tensor,
    tangent: Tensor,
) -> Tensor {
    let (graph, lhs_key, rhs_key, output_key) = build_binary_graph(
        op,
        tensor_input_key(key_base),
        tensor_input_key(key_base + 1),
    );
    let mut inputs_map = HashMap::new();
    inputs_map.insert(lhs_key, lhs);
    inputs_map.insert(rhs_key.clone(), rhs);
    jvp_from_graph_with_inputs(graph, output_key, rhs_key, inputs_map, tangent)
}

fn col_major_2d(row: usize, col: usize, rows: usize) -> usize {
    row + col * rows
}

fn expected_matmul_real_complex_tangent(
    lhs_tangent: &[f64],
    rhs: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for inner in 0..k {
                acc += Complex64::new(lhs_tangent[col_major_2d(row, inner, m)], 0.0)
                    * rhs[col_major_2d(inner, col, k)];
            }
            out[col_major_2d(row, col, m)] = acc;
        }
    }
    out
}

fn expected_matmul_complex_real_tangent(
    lhs: &[Complex64],
    rhs_tangent: &[f64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for inner in 0..k {
                acc += lhs[col_major_2d(row, inner, m)]
                    * Complex64::new(rhs_tangent[col_major_2d(inner, col, k)], 0.0);
            }
            out[col_major_2d(row, col, m)] = acc;
        }
    }
    out
}

fn expected_dot_vjp_lhs_real(
    rhs: &[Complex64],
    cotangent: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; m * k];
    for row in 0..m {
        for inner in 0..k {
            let mut acc = Complex64::new(0.0, 0.0);
            for col in 0..n {
                acc +=
                    cotangent[col_major_2d(row, col, m)] * rhs[col_major_2d(inner, col, k)].conj();
            }
            out[col_major_2d(row, inner, m)] = acc.re;
        }
    }
    out
}

fn expected_dot_vjp_rhs_real(
    lhs: &[Complex64],
    cotangent: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; k * n];
    for inner in 0..k {
        for col in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for row in 0..m {
                acc +=
                    lhs[col_major_2d(row, inner, m)].conj() * cotangent[col_major_2d(row, col, m)];
            }
            out[col_major_2d(inner, col, k)] = acc.re;
        }
    }
    out
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
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let input_id = builder.add_input(input_key.clone());
    let output =
        builder.add_operation(op, vec![ValueRef::Local(input_id)], OperationRole::Primary)[0];
    builder.set_outputs(vec![output]);

    let graph = Arc::new(builder.build());
    let output_key = graph.values()[output].key.clone();
    let _primal_metadata_scope = register_graph_metadata_for_test(
        &graph,
        vec![(
            ValueKey::Input(input_key.clone()),
            tensor_meta_from_tensor(&input),
        )],
    );
    let view = resolve(vec![graph.clone()]);
    let mut ad_ctx = ShapeGuardContext::with_global_metadata();
    let linear = linearize(
        &view,
        std::slice::from_ref(&output_key),
        std::slice::from_ref(&input_key),
        90_000,
        &mut ad_ctx,
        &HashMap::new(),
    );
    let tangent_input_key = input_key.tangent_of(90_000);
    let _linear_metadata_scope = register_graph_metadata_for_test(
        linear.as_graph(),
        vec![(
            ValueKey::Input(tangent_input_key.clone()),
            tensor_meta_from_tensor(&input),
        )],
    );
    ad_ctx.refresh_global_metadata();
    let transposed = linear_transpose(&linear, &mut ad_ctx);
    let linear_graph = Arc::new(linear.into_graph());
    let cotangent_input_key =
        match &transposed.as_graph().values()[transposed.tangent_inputs()[0].1].key {
            ValueKey::Input(key) => key.clone(),
            _ => panic!("expected cotangent seed input"),
        };
    let output_key = transposed.tangent_outputs()[0]
        .map(|id| transposed.as_graph().values()[id].key.clone())
        .expect("expected active transpose output");
    let transposed_graph = Arc::new(transposed.into_graph());

    let mut inputs_map = HashMap::new();
    inputs_map.insert(input_key, input.clone());
    inputs_map.insert(tangent_input_key, input);
    inputs_map.insert(cotangent_input_key, cotangent);

    eval_graph_outputs(
        vec![graph, linear_graph, transposed_graph],
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

fn complex_unary_vjp_key(op: &StdTensorOp) -> TensorInputKey {
    let tag = match op {
        StdTensorOp::Exp => 1,
        StdTensorOp::Log => 2,
        StdTensorOp::Sin => 3,
        StdTensorOp::Cos => 4,
        StdTensorOp::Tanh => 5,
        StdTensorOp::Sqrt => 6,
        StdTensorOp::Rsqrt => 7,
        StdTensorOp::Expm1 => 8,
        StdTensorOp::Log1p => 9,
        _ => panic!("unexpected complex unary VJP op: {op:?}"),
    };
    tensor_input_key(91_000 + tag)
}

fn assert_complex_unary_vjp(op: StdTensorOp, derivative: impl Fn(Complex64) -> Complex64) {
    let input_key = complex_unary_vjp_key(&op);
    let input_data = vec![Complex64::new(0.5, 0.75), Complex64::new(-0.25, 0.4)];
    let cotangent_data = vec![Complex64::new(0.25, -1.2), Complex64::new(-0.5, 0.75)];
    let grad = transpose_primal_unary_op_with_inputs(
        op,
        input_key,
        c64_tensor(vec![2], input_data.clone()),
        c64_tensor(vec![2], cotangent_data.clone()),
    );

    let expected: Vec<_> = input_data
        .iter()
        .zip(cotangent_data.iter())
        .map(|(&x, &ct)| ct * derivative(x).conj())
        .collect();
    assert_close_slice_c64(get_c64_data(&grad), &expected);
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
fn complex_unary_vjps_conjugate_holomorphic_derivatives() {
    assert_complex_unary_vjp(StdTensorOp::Exp, |x| x.exp());
    assert_complex_unary_vjp(StdTensorOp::Log, |x| Complex64::new(1.0, 0.0) / x);
    assert_complex_unary_vjp(StdTensorOp::Sin, |x| x.cos());
    assert_complex_unary_vjp(StdTensorOp::Cos, |x| -x.sin());
    assert_complex_unary_vjp(StdTensorOp::Tanh, |x| {
        Complex64::new(1.0, 0.0) - x.tanh() * x.tanh()
    });
    assert_complex_unary_vjp(StdTensorOp::Sqrt, |x| {
        Complex64::new(1.0, 0.0) / (Complex64::new(2.0, 0.0) * x.sqrt())
    });
    assert_complex_unary_vjp(StdTensorOp::Rsqrt, |x| {
        -Complex64::new(1.0, 0.0) / (Complex64::new(2.0, 0.0) * x * x.sqrt())
    });
    assert_complex_unary_vjp(StdTensorOp::Expm1, |x| x.exp());
    assert_complex_unary_vjp(StdTensorOp::Log1p, |x| {
        Complex64::new(1.0, 0.0) / (Complex64::new(1.0, 0.0) + x)
    });
}

#[test]
fn complex_div_and_pow_vjps_conjugate_holomorphic_coefficients() {
    let x_data = vec![Complex64::new(0.8, 0.35), Complex64::new(1.1, -0.2)];
    let y_data = vec![Complex64::new(1.5, -0.25), Complex64::new(0.7, 0.45)];
    let cotangent_data = vec![Complex64::new(-0.3, 0.9), Complex64::new(1.25, -0.5)];

    let x = TracedTensor::from_tensor_concrete_shape(c64_tensor(vec![2], x_data.clone()));
    let y = TracedTensor::from_tensor_concrete_shape(c64_tensor(vec![2], y_data.clone()));
    let cotangent =
        TracedTensor::from_tensor_concrete_shape(c64_tensor(vec![2], cotangent_data.clone()));
    let quotient = x.div(&y);

    let div_vjp_x = eval_tensor(quotient.vjp(&x, &cotangent));
    let div_vjp_y = eval_tensor(quotient.vjp(&y, &cotangent));

    let expected_div_x: Vec<_> = y_data
        .iter()
        .zip(cotangent_data.iter())
        .map(|(&y, &ct)| ct * (Complex64::new(1.0, 0.0) / y).conj())
        .collect();
    let expected_div_y: Vec<_> = x_data
        .iter()
        .zip(y_data.iter())
        .zip(cotangent_data.iter())
        .map(|((&x, &y), &ct)| ct * (-(x / (y * y))).conj())
        .collect();
    assert_close_slice_c64(get_c64_data(&div_vjp_x), &expected_div_x);
    assert_close_slice_c64(get_c64_data(&div_vjp_y), &expected_div_y);

    let pow = x.pow(&y);
    let pow_vjp_x = eval_tensor(pow.vjp(&x, &cotangent));
    let pow_vjp_y = eval_tensor(pow.vjp(&y, &cotangent));

    let expected_pow_x: Vec<_> = x_data
        .iter()
        .zip(y_data.iter())
        .zip(cotangent_data.iter())
        .map(|((&x, &y), &ct)| {
            let pow_xy = x.powc(y);
            ct * (y * pow_xy / x).conj()
        })
        .collect();
    let expected_pow_y: Vec<_> = x_data
        .iter()
        .zip(y_data.iter())
        .zip(cotangent_data.iter())
        .map(|((&x, &y), &ct)| {
            let pow_xy = x.powc(y);
            ct * (x.ln() * pow_xy).conj()
        })
        .collect();
    assert_close_slice_c64(get_c64_data(&pow_vjp_x), &expected_pow_x);
    assert_close_slice_c64(get_c64_data(&pow_vjp_y), &expected_pow_y);
}

#[test]
fn mixed_real_complex_add_vjp_projects_lhs_to_real_tangent_space() {
    let lhs_data = vec![1.0, -2.0];
    let rhs_data = vec![Complex64::new(0.5, 1.25), Complex64::new(-3.0, 0.75)];
    let cotangent_data = vec![Complex64::new(4.0, -9.0), Complex64::new(-1.5, 2.25)];

    let grad = mixed_binary_vjp_lhs(
        StdTensorOp::Add,
        92_000,
        f64_tensor(vec![2], lhs_data),
        c64_tensor(vec![2], rhs_data),
        c64_tensor(vec![2], cotangent_data),
    );

    assert_close_slice(get_f64_data(&grad), &[4.0, -1.5]);
}

#[test]
fn mixed_complex_real_add_vjp_projects_rhs_to_real_tangent_space() {
    let lhs_data = vec![Complex64::new(0.5, 1.25), Complex64::new(-3.0, 0.75)];
    let rhs_data = vec![1.0, -2.0];
    let cotangent_data = vec![Complex64::new(4.0, -9.0), Complex64::new(-1.5, 2.25)];

    let grad = mixed_binary_vjp_rhs(
        StdTensorOp::Add,
        92_040,
        c64_tensor(vec![2], lhs_data),
        f64_tensor(vec![2], rhs_data),
        c64_tensor(vec![2], cotangent_data),
    );

    assert_close_slice(get_f64_data(&grad), &[4.0, -1.5]);
}

#[test]
fn mixed_real_complex_add_jvp_promotes_lhs_tangent() {
    let lhs_data = vec![1.0, -2.0];
    let rhs_data = vec![Complex64::new(0.5, 1.25), Complex64::new(-3.0, 0.75)];
    let tangent_data = vec![4.0, -1.5];

    let tangent = mixed_binary_jvp_lhs(
        StdTensorOp::Add,
        92_050,
        f64_tensor(vec![2], lhs_data),
        c64_tensor(vec![2], rhs_data),
        f64_tensor(vec![2], tangent_data),
    );

    assert_close_slice_c64(
        get_c64_data(&tangent),
        &[Complex64::new(4.0, 0.0), Complex64::new(-1.5, 0.0)],
    );
}

#[test]
fn mixed_real_complex_mul_vjp_projects_lhs_to_real_tangent_space() {
    let lhs_data = vec![1.5, -0.75];
    let rhs_data = vec![Complex64::new(2.0, 3.0), Complex64::new(-1.0, 0.5)];
    let cotangent_data = vec![Complex64::new(5.0, 7.0), Complex64::new(-2.0, 4.0)];

    let grad = mixed_binary_vjp_lhs(
        StdTensorOp::Mul,
        92_010,
        f64_tensor(vec![2], lhs_data),
        c64_tensor(vec![2], rhs_data.clone()),
        c64_tensor(vec![2], cotangent_data.clone()),
    );

    let expected: Vec<_> = rhs_data
        .iter()
        .zip(cotangent_data.iter())
        .map(|(&rhs, &ct)| (ct * rhs.conj()).re)
        .collect();
    assert_close_slice(get_f64_data(&grad), &expected);
}

#[test]
fn mixed_complex_real_mul_vjp_projects_rhs_to_real_tangent_space() {
    let lhs_data = vec![Complex64::new(2.0, 3.0), Complex64::new(-1.0, 0.5)];
    let rhs_data = vec![1.5, -0.75];
    let cotangent_data = vec![Complex64::new(5.0, 7.0), Complex64::new(-2.0, 4.0)];

    let grad = mixed_binary_vjp_rhs(
        StdTensorOp::Mul,
        92_060,
        c64_tensor(vec![2], lhs_data.clone()),
        f64_tensor(vec![2], rhs_data),
        c64_tensor(vec![2], cotangent_data.clone()),
    );

    let expected: Vec<_> = lhs_data
        .iter()
        .zip(cotangent_data.iter())
        .map(|(&lhs, &ct)| (ct * lhs.conj()).re)
        .collect();
    assert_close_slice(get_f64_data(&grad), &expected);
}

#[test]
fn mixed_real_complex_mul_jvp_promotes_lhs_tangent() {
    let lhs_data = vec![1.5, -0.75];
    let rhs_data = vec![Complex64::new(2.0, 3.0), Complex64::new(-1.0, 0.5)];
    let tangent_data = vec![5.0, -2.0];

    let tangent = mixed_binary_jvp_lhs(
        StdTensorOp::Mul,
        92_070,
        f64_tensor(vec![2], lhs_data),
        c64_tensor(vec![2], rhs_data.clone()),
        f64_tensor(vec![2], tangent_data.clone()),
    );

    let expected: Vec<_> = tangent_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(&dx, &rhs)| Complex64::new(dx, 0.0) * rhs)
        .collect();
    assert_close_slice_c64(get_c64_data(&tangent), &expected);
}

#[test]
fn mixed_real_complex_div_vjp_projects_lhs_to_real_tangent_space() {
    let lhs_data = vec![3.0, -1.25];
    let rhs_data = vec![Complex64::new(1.5, -0.25), Complex64::new(0.75, 0.5)];
    let cotangent_data = vec![Complex64::new(-0.3, 0.9), Complex64::new(1.25, -0.5)];

    let grad = mixed_binary_vjp_lhs(
        StdTensorOp::Div,
        92_020,
        f64_tensor(vec![2], lhs_data),
        c64_tensor(vec![2], rhs_data.clone()),
        c64_tensor(vec![2], cotangent_data.clone()),
    );

    let expected: Vec<_> = rhs_data
        .iter()
        .zip(cotangent_data.iter())
        .map(|(&rhs, &ct)| (ct * (Complex64::new(1.0, 0.0) / rhs).conj()).re)
        .collect();
    assert_close_slice(get_f64_data(&grad), &expected);
}

#[test]
fn mixed_complex_real_div_vjp_projects_rhs_to_real_tangent_space() {
    let lhs_data = vec![Complex64::new(3.0, -1.25), Complex64::new(0.75, 0.5)];
    let rhs_data = vec![1.5, 0.75];
    let cotangent_data = vec![Complex64::new(-0.3, 0.9), Complex64::new(1.25, -0.5)];

    let grad = mixed_binary_vjp_rhs(
        StdTensorOp::Div,
        92_080,
        c64_tensor(vec![2], lhs_data.clone()),
        f64_tensor(vec![2], rhs_data.clone()),
        c64_tensor(vec![2], cotangent_data.clone()),
    );

    let expected: Vec<_> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .zip(cotangent_data.iter())
        .map(|((&lhs, &rhs), &ct)| {
            let coeff = -lhs / Complex64::new(rhs * rhs, 0.0);
            (ct * coeff.conj()).re
        })
        .collect();
    assert_close_slice(get_f64_data(&grad), &expected);
}

#[test]
fn mixed_real_complex_div_jvp_promotes_lhs_tangent() {
    let lhs_data = vec![3.0, -1.25];
    let rhs_data = vec![Complex64::new(1.5, -0.25), Complex64::new(0.75, 0.5)];
    let tangent_data = vec![-0.3, 1.25];

    let tangent = mixed_binary_jvp_lhs(
        StdTensorOp::Div,
        92_090,
        f64_tensor(vec![2], lhs_data),
        c64_tensor(vec![2], rhs_data.clone()),
        f64_tensor(vec![2], tangent_data.clone()),
    );

    let expected: Vec<_> = tangent_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(&dx, &rhs)| Complex64::new(dx, 0.0) / rhs)
        .collect();
    assert_close_slice_c64(get_c64_data(&tangent), &expected);
}

#[test]
fn mixed_real_complex_pow_vjp_projects_lhs_to_real_tangent_space() {
    let lhs_data = vec![1.2, 2.5];
    let rhs_data = vec![Complex64::new(0.75, 0.5), Complex64::new(1.5, -0.25)];
    let cotangent_data = vec![Complex64::new(-0.5, 1.25), Complex64::new(0.75, -0.3)];

    let grad = mixed_binary_vjp_lhs(
        StdTensorOp::Pow,
        92_030,
        f64_tensor(vec![2], lhs_data.clone()),
        c64_tensor(vec![2], rhs_data.clone()),
        c64_tensor(vec![2], cotangent_data.clone()),
    );

    let expected: Vec<_> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .zip(cotangent_data.iter())
        .map(|((&lhs, &rhs), &ct)| {
            let lhs = Complex64::new(lhs, 0.0);
            let coeff = rhs * lhs.powc(rhs - Complex64::new(1.0, 0.0));
            (ct * coeff.conj()).re
        })
        .collect();
    assert_close_slice(get_f64_data(&grad), &expected);
}

#[test]
fn mixed_complex_real_pow_vjp_projects_rhs_to_real_tangent_space() {
    let lhs_data = vec![Complex64::new(1.2, 0.4), Complex64::new(2.5, -0.2)];
    let rhs_data = vec![0.75, 1.5];
    let cotangent_data = vec![Complex64::new(-0.5, 1.25), Complex64::new(0.75, -0.3)];

    let grad = mixed_binary_vjp_rhs(
        StdTensorOp::Pow,
        92_100,
        c64_tensor(vec![2], lhs_data.clone()),
        f64_tensor(vec![2], rhs_data.clone()),
        c64_tensor(vec![2], cotangent_data.clone()),
    );

    let expected: Vec<_> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .zip(cotangent_data.iter())
        .map(|((&lhs, &rhs), &ct)| {
            let coeff = lhs.ln() * lhs.powc(Complex64::new(rhs, 0.0));
            (ct * coeff.conj()).re
        })
        .collect();
    assert_close_slice(get_f64_data(&grad), &expected);
}

#[test]
fn mixed_real_complex_pow_jvp_promotes_lhs_tangent() {
    let lhs_data = vec![1.2, 2.5];
    let rhs_data = vec![Complex64::new(0.75, 0.5), Complex64::new(1.5, -0.25)];
    let tangent_data = vec![-0.5, 0.75];

    let tangent = mixed_binary_jvp_lhs(
        StdTensorOp::Pow,
        92_110,
        f64_tensor(vec![2], lhs_data.clone()),
        c64_tensor(vec![2], rhs_data.clone()),
        f64_tensor(vec![2], tangent_data.clone()),
    );

    let expected: Vec<_> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .zip(tangent_data.iter())
        .map(|((&lhs, &rhs), &dx)| {
            let lhs = Complex64::new(lhs, 0.0);
            let coeff = rhs * lhs.powc(rhs - Complex64::new(1.0, 0.0));
            coeff * Complex64::new(dx, 0.0)
        })
        .collect();
    assert_close_slice_c64(get_c64_data(&tangent), &expected);
}

#[test]
fn mixed_real_complex_dot_general_vjp_projects_lhs_to_real_tangent_space() {
    let lhs_data = vec![1.0, -2.0, 0.5, 1.25, -0.75, 2.5];
    let rhs_data = vec![
        Complex64::new(0.5, 1.0),
        Complex64::new(-1.0, 0.25),
        Complex64::new(2.0, -0.5),
        Complex64::new(1.5, -1.25),
        Complex64::new(-0.25, 0.75),
        Complex64::new(0.8, 0.4),
    ];
    let cotangent_data = vec![
        Complex64::new(0.25, -0.5),
        Complex64::new(1.5, 0.75),
        Complex64::new(-0.75, 1.25),
        Complex64::new(0.6, -1.0),
    ];

    let grad = mixed_binary_vjp_lhs(
        StdTensorOp::DotGeneral {
            config: matmul_config(),
        },
        92_120,
        f64_tensor(vec![2, 3], lhs_data),
        c64_tensor(vec![3, 2], rhs_data.clone()),
        c64_tensor(vec![2, 2], cotangent_data.clone()),
    );

    let expected = expected_dot_vjp_lhs_real(&rhs_data, &cotangent_data, 2, 3, 2);
    assert_close_slice(get_f64_data(&grad), &expected);
}

#[test]
fn mixed_complex_real_dot_general_vjp_projects_rhs_to_real_tangent_space() {
    let lhs_data = vec![
        Complex64::new(0.5, 1.0),
        Complex64::new(-1.0, 0.25),
        Complex64::new(2.0, -0.5),
        Complex64::new(1.5, -1.25),
        Complex64::new(-0.25, 0.75),
        Complex64::new(0.8, 0.4),
    ];
    let rhs_data = vec![1.0, -2.0, 0.5, 1.25, -0.75, 2.5];
    let cotangent_data = vec![
        Complex64::new(0.25, -0.5),
        Complex64::new(1.5, 0.75),
        Complex64::new(-0.75, 1.25),
        Complex64::new(0.6, -1.0),
    ];

    let grad = mixed_binary_vjp_rhs(
        StdTensorOp::DotGeneral {
            config: matmul_config(),
        },
        92_130,
        c64_tensor(vec![2, 3], lhs_data.clone()),
        f64_tensor(vec![3, 2], rhs_data),
        c64_tensor(vec![2, 2], cotangent_data.clone()),
    );

    let expected = expected_dot_vjp_rhs_real(&lhs_data, &cotangent_data, 2, 3, 2);
    assert_close_slice(get_f64_data(&grad), &expected);
}

#[test]
fn mixed_real_complex_dot_general_jvp_promotes_lhs_tangent() {
    let lhs_data = vec![1.0, -2.0, 0.5, 1.25, -0.75, 2.5];
    let rhs_data = vec![
        Complex64::new(0.5, 1.0),
        Complex64::new(-1.0, 0.25),
        Complex64::new(2.0, -0.5),
        Complex64::new(1.5, -1.25),
        Complex64::new(-0.25, 0.75),
        Complex64::new(0.8, 0.4),
    ];
    let tangent_data = vec![0.25, -0.5, 1.5, 0.75, -0.75, 1.25];

    let tangent = mixed_binary_jvp_lhs(
        StdTensorOp::DotGeneral {
            config: matmul_config(),
        },
        92_140,
        f64_tensor(vec![2, 3], lhs_data),
        c64_tensor(vec![3, 2], rhs_data.clone()),
        f64_tensor(vec![2, 3], tangent_data.clone()),
    );

    let expected = expected_matmul_real_complex_tangent(&tangent_data, &rhs_data, 2, 3, 2);
    assert_close_slice_c64(get_c64_data(&tangent), &expected);
}

#[test]
fn mixed_complex_real_dot_general_jvp_promotes_rhs_tangent() {
    let lhs_data = vec![
        Complex64::new(0.5, 1.0),
        Complex64::new(-1.0, 0.25),
        Complex64::new(2.0, -0.5),
        Complex64::new(1.5, -1.25),
        Complex64::new(-0.25, 0.75),
        Complex64::new(0.8, 0.4),
    ];
    let rhs_data = vec![1.0, -2.0, 0.5, 1.25, -0.75, 2.5];
    let tangent_data = vec![0.25, -0.5, 1.5, 0.75, -0.75, 1.25];

    let tangent = mixed_binary_jvp_rhs(
        StdTensorOp::DotGeneral {
            config: matmul_config(),
        },
        92_150,
        c64_tensor(vec![2, 3], lhs_data.clone()),
        f64_tensor(vec![3, 2], rhs_data),
        f64_tensor(vec![3, 2], tangent_data.clone()),
    );

    let expected = expected_matmul_complex_real_tangent(&lhs_data, &tangent_data, 2, 3, 2);
    assert_close_slice_c64(get_c64_data(&tangent), &expected);
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
