mod support;
use std::collections::HashMap;
use std::sync::Arc;
use support::{einsum, run_many_traced_with, RunTraced};

use chainrules_core::ADKey;
use computegraph::compile::compile;
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, OpMode, ValRef};
use computegraph::LocalValId;
use num_complex::Complex64;
use tenferro::compiler::compile_std_to_exec;
use tenferro::exec::eval_exec_ir;
use tenferro::shape_infer::{infer_output_dtype, infer_output_extents};
use tenferro::traced_tensor::matmul;
use tenferro::{GraphExecutor, TracedTensor};
use tenferro_ops::ad::context::{
    lookup_global_metadata, register_scoped_global_metadata_batch, GlobalMetadataScope, TensorMeta,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeGuardContext, SymDim};
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{
    DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig, Tensor,
    TensorBackend, TypedTensor,
};
use tidu::{differentiate, transpose, LinearFragment};

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

fn finite_diff_complex(
    f: impl Fn(&[Complex64]) -> f64,
    x: &[Complex64],
    idx: usize,
    h: f64,
) -> Complex64 {
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[idx] += Complex64::new(h, 0.0);
    xm[idx] -= Complex64::new(h, 0.0);
    let df_dre = (f(&xp) - f(&xm)) / (2.0 * h);

    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[idx] += Complex64::new(0.0, h);
    xm[idx] -= Complex64::new(0.0, h);
    let df_dim = (f(&xp) - f(&xm)) / (2.0 * h);

    Complex64::new(df_dre, df_dim)
}

fn finite_diff_complex_lhs(
    f: impl Fn(&[Complex64], &[Complex64]) -> f64,
    lhs: &[Complex64],
    rhs: &[Complex64],
    idx: usize,
    h: f64,
) -> Complex64 {
    let mut lp = lhs.to_vec();
    let mut lm = lhs.to_vec();
    lp[idx] += Complex64::new(h, 0.0);
    lm[idx] -= Complex64::new(h, 0.0);
    let df_dre = (f(&lp, rhs) - f(&lm, rhs)) / (2.0 * h);

    let mut lp = lhs.to_vec();
    let mut lm = lhs.to_vec();
    lp[idx] += Complex64::new(0.0, h);
    lm[idx] -= Complex64::new(0.0, h);
    let df_dim = (f(&lp, rhs) - f(&lm, rhs)) / (2.0 * h);

    Complex64::new(df_dre, df_dim)
}

fn finite_diff_complex_rhs(
    f: impl Fn(&[Complex64], &[Complex64]) -> f64,
    lhs: &[Complex64],
    rhs: &[Complex64],
    idx: usize,
    h: f64,
) -> Complex64 {
    let mut rp = rhs.to_vec();
    let mut rm = rhs.to_vec();
    rp[idx] += Complex64::new(h, 0.0);
    rm[idx] -= Complex64::new(h, 0.0);
    let df_dre = (f(lhs, &rp) - f(lhs, &rm)) / (2.0 * h);

    let mut rp = rhs.to_vec();
    let mut rm = rhs.to_vec();
    rp[idx] += Complex64::new(0.0, h);
    rm[idx] -= Complex64::new(0.0, h);
    let df_dim = (f(lhs, &rp) - f(lhs, &rm)) / (2.0 * h);

    Complex64::new(df_dre, df_dim)
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

fn svd_op(_shape: Vec<usize>, eps: f64) -> StdTensorOp {
    StdTensorOp::Svd { eps }
}

fn qr_op(_shape: Vec<usize>) -> StdTensorOp {
    StdTensorOp::Qr
}

fn eigh_op(_shape: Vec<usize>, eps: f64) -> StdTensorOp {
    StdTensorOp::Eigh { eps }
}

fn cholesky_op(_shape: Vec<usize>) -> StdTensorOp {
    StdTensorOp::Cholesky
}

fn triangular_solve_op(_lhs_shape: Vec<usize>, _rhs_shape: Vec<usize>) -> StdTensorOp {
    StdTensorOp::TriangularSolve {
        left_side: true,
        lower: true,
        transpose_a: false,
        unit_diagonal: false,
    }
}

fn full_piv_lu_solve_op(_lhs_shape: Vec<usize>, _rhs_shape: Vec<usize>) -> StdTensorOp {
    StdTensorOp::FullPivLuSolve { transpose_a: false }
}

fn solve_dot_general_config(rank: usize) -> DotGeneralConfig {
    let batch_dims: Vec<usize> = (2..rank).collect();
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: batch_dims.clone(),
        rhs_batch_dims: batch_dims,
    }
}

fn add_solve_composition(
    builder: &mut FragmentBuilder<StdTensorOp>,
    a: LocalValId,
    b: LocalValId,
    lhs_shape: Vec<usize>,
    _rhs_shape: Vec<usize>,
) -> LocalValId {
    let lu = builder.add_op(StdTensorOp::Lu, vec![ValRef::Local(a)], OpMode::Primal);
    let rank = lhs_shape.len();
    let pb = builder.add_op(
        {
            let config = solve_dot_general_config(rank);
            StdTensorOp::DotGeneral { config }
        },
        vec![ValRef::Local(lu[0]), ValRef::Local(b)],
        OpMode::Primal,
    )[0];
    let z = builder.add_op(
        StdTensorOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: true,
        },
        vec![ValRef::Local(lu[1]), ValRef::Local(pb)],
        OpMode::Primal,
    )[0];
    builder.add_op(
        StdTensorOp::TriangularSolve {
            left_side: true,
            lower: false,
            transpose_a: false,
            unit_diagonal: false,
        },
        vec![ValRef::Local(lu[2]), ValRef::Local(z)],
        OpMode::Primal,
    )[0]
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

fn scalar_c64_tensor(value: Complex64) -> Tensor {
    c64_tensor(vec![], vec![value])
}

fn add_real_reduce_sum_loss(
    builder: &mut FragmentBuilder<StdTensorOp>,
    input: computegraph::LocalValId,
    input_shape: Vec<usize>,
) -> computegraph::LocalValId {
    let conjugated = builder.add_op(
        StdTensorOp::Conj,
        vec![ValRef::Local(input)],
        OpMode::Primal,
    );
    let summed = builder.add_op(
        StdTensorOp::Add,
        vec![ValRef::Local(input), ValRef::Local(conjugated[0])],
        OpMode::Primal,
    );
    let half = builder.add_op(StdTensorOp::constant_f64(0.5), vec![], OpMode::Primal);
    let scaled = builder.add_op(
        StdTensorOp::Mul,
        vec![ValRef::Local(half[0]), ValRef::Local(summed[0])],
        OpMode::Primal,
    );
    builder.add_op(
        StdTensorOp::ReduceSum {
            axes: (0..input_shape.len()).collect(),
        },
        vec![ValRef::Local(scaled[0])],
        OpMode::Primal,
    )[0]
}

fn build_svd_values_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let input_key = tensor_input_key(10_000);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let svd = builder.add_op(
        svd_op(vec![2, 2], 1.0e-12),
        vec![ValRef::Local(a)],
        OpMode::Primal,
    );
    let loss = builder.add_op(
        StdTensorOp::ReduceSum { axes: vec![0] },
        vec![ValRef::Local(svd[1])],
        OpMode::Primal,
    );
    let loss_key = builder.global_key(loss[0]).clone();
    builder.set_outputs(vec![loss[0]]);
    (Arc::new(builder.build()), input_key, loss_key)
}

fn build_svd_values_real_sum_fragment(
    input_key: TensorInputKey,
    input_shape: Vec<usize>,
) -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let singular_shape = vec![input_shape[0].min(input_shape[1])];
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let svd = builder.add_op(
        svd_op(input_shape.clone(), 1.0e-12),
        vec![ValRef::Local(a)],
        OpMode::Primal,
    );
    let loss = add_real_reduce_sum_loss(&mut builder, svd[1], singular_shape);
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), input_key, loss_key)
}

fn build_svd_values_real_sum_complex_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    build_svd_values_real_sum_fragment(tensor_input_key(12_000), vec![2, 2])
}

fn build_svd_values_real_sum_complex_3x3_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    build_svd_values_real_sum_fragment(tensor_input_key(12_100), vec![3, 3])
}

fn build_svd_uv_product_fragment(
    input_key: TensorInputKey,
    product_shape: Vec<usize>,
    real_loss: bool,
) -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let svd = builder.add_op(
        svd_op(product_shape.clone(), 1.0e-12),
        vec![ValRef::Local(a)],
        OpMode::Primal,
    );
    let uv_product = builder.add_op(
        {
            let config = matmul_config();
            StdTensorOp::DotGeneral { config }
        },
        vec![ValRef::Local(svd[0]), ValRef::Local(svd[2])],
        OpMode::Primal,
    );
    let loss = if real_loss {
        add_real_reduce_sum_loss(&mut builder, uv_product[0], product_shape)
    } else {
        builder.add_op(
            StdTensorOp::ReduceSum { axes: vec![0, 1] },
            vec![ValRef::Local(uv_product[0])],
            OpMode::Primal,
        )[0]
    };
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), input_key, loss_key)
}

fn build_svd_uv_product_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    build_svd_uv_product_fragment(tensor_input_key(15_000), vec![2, 2], false)
}

fn build_svd_uv_product_real_sum_complex_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    build_svd_uv_product_fragment(tensor_input_key(15_100), vec![3, 3], true)
}

fn build_svd_reconstruction_sum_fragment(
    input_key: TensorInputKey,
    shape: Vec<usize>,
    eps: f64,
) -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let m = shape[0];
    let n = shape[1];
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let svd = builder.add_op(
        svd_op(vec![m, n], eps),
        vec![ValRef::Local(a)],
        OpMode::Primal,
    );
    let diag_s = builder.add_op(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1,
        },
        vec![ValRef::Local(svd[1])],
        OpMode::Primal,
    );
    let us = builder.add_op(
        {
            let config = matmul_config();
            StdTensorOp::DotGeneral { config }
        },
        vec![ValRef::Local(svd[0]), ValRef::Local(diag_s[0])],
        OpMode::Primal,
    );
    let reconstructed = builder.add_op(
        {
            let config = matmul_config();
            StdTensorOp::DotGeneral { config }
        },
        vec![ValRef::Local(us[0]), ValRef::Local(svd[2])],
        OpMode::Primal,
    );
    let loss = builder.add_op(
        StdTensorOp::ReduceSum { axes: vec![0, 1] },
        vec![ValRef::Local(reconstructed[0])],
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
    let eigh = builder.add_op(
        eigh_op(vec![2, 2], 1.0e-12),
        vec![ValRef::Local(a)],
        OpMode::Primal,
    );
    let loss = builder.add_op(
        StdTensorOp::ReduceSum { axes: vec![0] },
        vec![ValRef::Local(eigh[0])],
        OpMode::Primal,
    );
    let loss_key = builder.global_key(loss[0]).clone();
    builder.set_outputs(vec![loss[0]]);
    (Arc::new(builder.build()), input_key, loss_key)
}

fn build_eigh_values_real_sum_complex_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let input_key = tensor_input_key(22_000);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let eigh = builder.add_op(
        eigh_op(vec![2, 2], 1.0e-12),
        vec![ValRef::Local(a)],
        OpMode::Primal,
    );
    let loss = add_real_reduce_sum_loss(&mut builder, eigh[0], vec![2]);
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), input_key, loss_key)
}

fn build_eigh_projector_fragment(
    input_key: TensorInputKey,
    weights_key: TensorInputKey,
    probe_key: TensorInputKey,
    use_adjoint: bool,
    real_loss: bool,
) -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    TensorInputKey,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let weights = builder.add_input(weights_key.clone());
    let probe = builder.add_input(probe_key.clone());
    let eigh = builder.add_op(
        eigh_op(vec![2, 2], 1.0e-12),
        vec![ValRef::Local(a)],
        OpMode::Primal,
    );
    let diag = builder.add_op(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1,
        },
        vec![ValRef::Local(weights)],
        OpMode::Primal,
    );
    let weighted_vectors = builder.add_op(
        {
            let config = matmul_config();
            StdTensorOp::DotGeneral { config }
        },
        vec![ValRef::Local(eigh[1]), ValRef::Local(diag[0])],
        OpMode::Primal,
    );
    let vt = if use_adjoint {
        let conjugated = builder.add_op(
            StdTensorOp::Conj,
            vec![ValRef::Local(eigh[1])],
            OpMode::Primal,
        );
        builder.add_op(
            StdTensorOp::Transpose { perm: vec![1, 0] },
            vec![ValRef::Local(conjugated[0])],
            OpMode::Primal,
        )
    } else {
        builder.add_op(
            StdTensorOp::Transpose { perm: vec![1, 0] },
            vec![ValRef::Local(eigh[1])],
            OpMode::Primal,
        )
    };
    let projector = builder.add_op(
        {
            let config = matmul_config();
            StdTensorOp::DotGeneral { config }
        },
        vec![ValRef::Local(weighted_vectors[0]), ValRef::Local(vt[0])],
        OpMode::Primal,
    );
    let weighted = builder.add_op(
        StdTensorOp::Mul,
        vec![ValRef::Local(projector[0]), ValRef::Local(probe)],
        OpMode::Primal,
    );
    let loss = if real_loss {
        add_real_reduce_sum_loss(&mut builder, weighted[0], vec![2, 2])
    } else {
        builder.add_op(
            StdTensorOp::ReduceSum { axes: vec![0, 1] },
            vec![ValRef::Local(weighted[0])],
            OpMode::Primal,
        )[0]
    };
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (
        Arc::new(builder.build()),
        input_key,
        weights_key,
        probe_key,
        loss_key,
    )
}

fn build_eigh_projector_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    TensorInputKey,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    build_eigh_projector_fragment(
        tensor_input_key(25_000),
        tensor_input_key(25_001),
        tensor_input_key(25_002),
        false,
        false,
    )
}

fn build_eigh_projector_real_sum_complex_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    TensorInputKey,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    build_eigh_projector_fragment(
        tensor_input_key(26_000),
        tensor_input_key(26_001),
        tensor_input_key(26_002),
        true,
        true,
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
    let solve = add_solve_composition(&mut builder, a, b, vec![2, 2], vec![2, 1]);
    let loss = builder.add_op(
        StdTensorOp::ReduceSum { axes: vec![0, 1] },
        vec![ValRef::Local(solve)],
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
        triangular_solve_op(vec![2, 2], vec![2, 1]),
        vec![ValRef::Local(a), ValRef::Local(b)],
        OpMode::Primal,
    );
    let loss = builder.add_op(
        StdTensorOp::ReduceSum { axes: vec![0, 1] },
        vec![ValRef::Local(solve[0])],
        OpMode::Primal,
    );
    let loss_key = builder.global_key(loss[0]).clone();
    builder.set_outputs(vec![loss[0]]);
    (Arc::new(builder.build()), a_key, b_key, loss_key)
}

fn build_full_piv_lu_solve_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let a_key = tensor_input_key(38_000);
    let b_key = tensor_input_key(38_001);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(a_key.clone());
    let b = builder.add_input(b_key.clone());
    let solve = builder.add_op(
        full_piv_lu_solve_op(vec![2, 2], vec![2, 1]),
        vec![ValRef::Local(a), ValRef::Local(b)],
        OpMode::Primal,
    );
    let loss = builder.add_op(
        StdTensorOp::ReduceSum { axes: vec![0, 1] },
        vec![ValRef::Local(solve[0])],
        OpMode::Primal,
    );
    let loss_key = builder.global_key(loss[0]).clone();
    builder.set_outputs(vec![loss[0]]);
    (Arc::new(builder.build()), a_key, b_key, loss_key)
}

fn build_full_piv_lu_lu_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let input_key = tensor_input_key(38_100);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let lu = builder.add_op(
        StdTensorOp::FullPivLu,
        vec![ValRef::Local(a)],
        OpMode::Primal,
    );
    let l_sum = builder.add_op(
        StdTensorOp::ReduceSum { axes: vec![0, 1] },
        vec![ValRef::Local(lu[1])],
        OpMode::Primal,
    )[0];
    let u_sum = builder.add_op(
        StdTensorOp::ReduceSum { axes: vec![0, 1] },
        vec![ValRef::Local(lu[2])],
        OpMode::Primal,
    )[0];
    let loss = builder.add_op(
        StdTensorOp::Add,
        vec![ValRef::Local(l_sum), ValRef::Local(u_sum)],
        OpMode::Primal,
    )[0];
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), input_key, loss_key)
}

fn build_solve_real_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let a_key = tensor_input_key(32_000);
    let b_key = tensor_input_key(32_001);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(a_key.clone());
    let b = builder.add_input(b_key.clone());
    let solve = add_solve_composition(&mut builder, a, b, vec![2, 2], vec![2, 1]);
    let loss = add_real_reduce_sum_loss(&mut builder, solve, vec![2, 1]);
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), a_key, b_key, loss_key)
}

fn build_triangular_solve_real_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let a_key = tensor_input_key(37_000);
    let b_key = tensor_input_key(37_001);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(a_key.clone());
    let b = builder.add_input(b_key.clone());
    let solve = builder.add_op(
        triangular_solve_op(vec![2, 2], vec![2, 1]),
        vec![ValRef::Local(a), ValRef::Local(b)],
        OpMode::Primal,
    );
    let loss = add_real_reduce_sum_loss(&mut builder, solve[0], vec![2, 1]);
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), a_key, b_key, loss_key)
}

fn build_cholesky_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let input_key = tensor_input_key(40_000);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let cholesky = builder.add_op(
        cholesky_op(vec![3, 3]),
        vec![ValRef::Local(a)],
        OpMode::Primal,
    );
    let loss = builder.add_op(
        StdTensorOp::ReduceSum { axes: vec![0, 1] },
        vec![ValRef::Local(cholesky[0])],
        OpMode::Primal,
    );
    let loss_key = builder.global_key(loss[0]).clone();
    builder.set_outputs(vec![loss[0]]);
    (Arc::new(builder.build()), input_key, loss_key)
}

fn build_cholesky_real_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let input_key = tensor_input_key(42_000);
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let cholesky = builder.add_op(
        cholesky_op(vec![2, 2]),
        vec![ValRef::Local(a)],
        OpMode::Primal,
    );
    let loss = add_real_reduce_sum_loss(&mut builder, cholesky[0], vec![2, 2]);
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), input_key, loss_key)
}

fn build_qr_r_fragment(
    input_key: TensorInputKey,
    real_loss: bool,
) -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let a = builder.add_input(input_key.clone());
    let qr = builder.add_op(qr_op(vec![2, 2]), vec![ValRef::Local(a)], OpMode::Primal);
    let loss = if real_loss {
        add_real_reduce_sum_loss(&mut builder, qr[1], vec![2, 2])
    } else {
        builder.add_op(
            StdTensorOp::ReduceSum { axes: vec![0, 1] },
            vec![ValRef::Local(qr[1])],
            OpMode::Primal,
        )[0]
    };
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), input_key, loss_key)
}

fn build_qr_r_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    build_qr_r_fragment(tensor_input_key(45_000), false)
}

fn build_qr_r_real_sum_fragment() -> (
    Arc<Fragment<StdTensorOp>>,
    TensorInputKey,
    GlobalValKey<StdTensorOp>,
) {
    build_qr_r_fragment(tensor_input_key(47_000), true)
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
        DType::I64 => Tensor::I64(TypedTensor::zeros(shape)),
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

fn sum_svd_values(data: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = f64_tensor(vec![2, 2], data.to_vec());
    let outputs = TensorBackend::svd(&mut backend, &input).unwrap();
    get_f64_data(&outputs[1]).iter().sum()
}

fn sum_svd_uv_product(data: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = f64_tensor(vec![2, 2], data.to_vec());
    let outputs = TensorBackend::svd(&mut backend, &input).unwrap();
    let product =
        TensorBackend::dot_general(&mut backend, &outputs[0], &outputs[2], &matmul_config())
            .unwrap();
    get_f64_data(&product).iter().sum()
}

fn sum_svd_reconstruction(data: &[f64], shape: [usize; 2]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = f64_tensor(shape.to_vec(), data.to_vec());
    let outputs = TensorBackend::svd(&mut backend, &input).unwrap();
    let diag_s = TensorBackend::embed_diagonal(&mut backend, &outputs[1], 0, 1).unwrap();
    let us =
        TensorBackend::dot_general(&mut backend, &outputs[0], &diag_s, &matmul_config()).unwrap();
    let reconstructed =
        TensorBackend::dot_general(&mut backend, &us, &outputs[2], &matmul_config()).unwrap();
    get_f64_data(&reconstructed).iter().sum()
}

fn sum_eigh_values(data: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = f64_tensor(vec![2, 2], data.to_vec());
    let outputs = TensorBackend::eigh(&mut backend, &input).unwrap();
    get_f64_data(&outputs[0]).iter().sum()
}

fn sum_eigh_projector(data: &[f64], weights: &[f64], probe: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = f64_tensor(vec![2, 2], data.to_vec());
    let outputs = TensorBackend::eigh(&mut backend, &input).unwrap();
    let diag =
        TensorBackend::embed_diagonal(&mut backend, &f64_tensor(vec![2], weights.to_vec()), 0, 1)
            .unwrap();
    let weighted_vectors =
        TensorBackend::dot_general(&mut backend, &outputs[1], &diag, &matmul_config()).unwrap();
    let vt = TensorBackend::transpose(&mut backend, &outputs[1], &[1, 0]).unwrap();
    let projector =
        TensorBackend::dot_general(&mut backend, &weighted_vectors, &vt, &matmul_config()).unwrap();
    let weighted = TensorBackend::mul(
        &mut backend,
        &projector,
        &f64_tensor(vec![2, 2], probe.to_vec()),
    )
    .unwrap();
    get_f64_data(&weighted).iter().sum()
}

fn sum_solve(a: &[f64], b: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let a_tensor = f64_tensor(vec![2, 2], a.to_vec());
    let b_tensor = f64_tensor(vec![2, 1], b.to_vec());
    let out = TensorBackend::solve(&mut backend, &a_tensor, &b_tensor).unwrap();
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
    )
    .unwrap();
    get_f64_data(&out).iter().sum()
}

fn sum_full_piv_lu_solve(a: &[f64], b: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let a_tensor = f64_tensor(vec![2, 2], a.to_vec());
    let b_tensor = f64_tensor(vec![2, 1], b.to_vec());
    let out = TensorBackend::full_piv_lu_solve(&mut backend, &a_tensor, &b_tensor, false).unwrap();
    get_f64_data(&out).iter().sum()
}

fn sum_full_piv_lu_lu(data: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = f64_tensor(vec![2, 2], data.to_vec());
    let outputs = TensorBackend::full_piv_lu(&mut backend, &input).unwrap();
    get_f64_data(&outputs[1]).iter().sum::<f64>() + get_f64_data(&outputs[2]).iter().sum::<f64>()
}

fn sum_cholesky(data: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = f64_tensor(vec![3, 3], data.to_vec());
    let output = TensorBackend::cholesky(&mut backend, &input).unwrap();
    get_f64_data(&output).iter().sum()
}

fn sum_qr_r(data: &[f64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = f64_tensor(vec![3, 2], data.to_vec());
    let outputs = TensorBackend::qr(&mut backend, &input).unwrap();
    get_f64_data(&outputs[1]).iter().sum()
}

fn sum_qr_r_real_parts_complex(data: &[Complex64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = c64_tensor(vec![3, 2], data.to_vec());
    let outputs = TensorBackend::qr(&mut backend, &input).unwrap();
    sum_real_parts(get_c64_data(&outputs[1]))
}

fn sum_real_parts(values: &[Complex64]) -> f64 {
    values.iter().map(|value| value.re).sum()
}

fn sum_svd_values_complex(data: &[Complex64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = c64_tensor(vec![2, 2], data.to_vec());
    let outputs = TensorBackend::svd(&mut backend, &input).unwrap();
    sum_real_parts(get_c64_data(&outputs[1]))
}

fn sum_svd_values_complex_3x3(data: &[Complex64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = c64_tensor(vec![3, 3], data.to_vec());
    let outputs = TensorBackend::svd(&mut backend, &input).unwrap();
    sum_real_parts(get_c64_data(&outputs[1]))
}

fn sum_svd_uv_product_real_parts_complex(data: &[Complex64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = c64_tensor(vec![3, 3], data.to_vec());
    let outputs = TensorBackend::svd(&mut backend, &input).unwrap();
    let product =
        TensorBackend::dot_general(&mut backend, &outputs[0], &outputs[2], &matmul_config())
            .unwrap();
    sum_real_parts(get_c64_data(&product))
}

fn sum_eigh_values_complex(data: &[Complex64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = c64_tensor(vec![2, 2], data.to_vec());
    let outputs = TensorBackend::eigh(&mut backend, &input).unwrap();
    sum_real_parts(get_c64_data(&outputs[0]))
}

fn sum_eigh_projector_real_parts_complex(
    data: &[Complex64],
    weights: &[Complex64],
    probe: &[Complex64],
) -> f64 {
    let mut backend = CpuBackend::new();
    let input = c64_tensor(vec![2, 2], data.to_vec());
    let outputs = TensorBackend::eigh(&mut backend, &input).unwrap();
    let diag =
        TensorBackend::embed_diagonal(&mut backend, &c64_tensor(vec![2], weights.to_vec()), 0, 1)
            .unwrap();
    let weighted_vectors =
        TensorBackend::dot_general(&mut backend, &outputs[1], &diag, &matmul_config()).unwrap();
    let vectors_conj = TensorBackend::conj(&mut backend, &outputs[1]).unwrap();
    let vh = TensorBackend::transpose(&mut backend, &vectors_conj, &[1, 0]).unwrap();
    let projector =
        TensorBackend::dot_general(&mut backend, &weighted_vectors, &vh, &matmul_config()).unwrap();
    let weighted = TensorBackend::mul(
        &mut backend,
        &projector,
        &c64_tensor(vec![2, 2], probe.to_vec()),
    )
    .unwrap();
    sum_real_parts(get_c64_data(&weighted))
}

fn sum_cholesky_real_parts_complex(data: &[Complex64]) -> f64 {
    let mut backend = CpuBackend::new();
    let input = c64_tensor(vec![2, 2], data.to_vec());
    let output = TensorBackend::cholesky(&mut backend, &input).unwrap();
    sum_real_parts(get_c64_data(&output))
}

fn sum_solve_real_parts_complex(a: &[Complex64], b: &[Complex64]) -> f64 {
    let mut backend = CpuBackend::new();
    let a_tensor = c64_tensor(vec![2, 2], a.to_vec());
    let b_tensor = c64_tensor(vec![2, 1], b.to_vec());
    let output = TensorBackend::solve(&mut backend, &a_tensor, &b_tensor).unwrap();
    sum_real_parts(get_c64_data(&output))
}

fn sum_triangular_solve_real_parts_complex(a: &[Complex64], b: &[Complex64]) -> f64 {
    let mut backend = CpuBackend::new();
    let a_tensor = c64_tensor(vec![2, 2], a.to_vec());
    let b_tensor = c64_tensor(vec![2, 1], b.to_vec());
    let output = TensorBackend::triangular_solve(
        &mut backend,
        &a_tensor,
        &b_tensor,
        true,
        true,
        false,
        false,
    )
    .unwrap();
    sum_real_parts(get_c64_data(&output))
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

fn assert_grad_matches_complex_finite_diff(
    actual: &[Complex64],
    base: &[Complex64],
    indices: &[usize],
    tol: f64,
    f: impl Fn(&[Complex64]) -> f64,
) {
    for &index in indices {
        let expected = finite_diff_complex(&f, base, index, 1e-6);
        assert!(
            (actual[index] - expected).norm() <= tol,
            "index {index}: expected {expected:?}, got {:?}",
            actual[index]
        );
    }
}

fn assert_grad_matches_complex_finite_diff_lhs(
    actual: &[Complex64],
    lhs: &[Complex64],
    rhs: &[Complex64],
    tol: f64,
    f: &impl Fn(&[Complex64], &[Complex64]) -> f64,
) {
    assert_eq!(actual.len(), lhs.len());
    for index in 0..lhs.len() {
        let expected = finite_diff_complex_lhs(f, lhs, rhs, index, 1e-6);
        assert!(
            (actual[index] - expected).norm() <= tol,
            "lhs index {index}: expected {expected:?}, got {:?}",
            actual[index]
        );
    }
}

fn assert_grad_matches_complex_finite_diff_rhs(
    actual: &[Complex64],
    lhs: &[Complex64],
    rhs: &[Complex64],
    tol: f64,
    f: &impl Fn(&[Complex64], &[Complex64]) -> f64,
) {
    assert_eq!(actual.len(), rhs.len());
    for index in 0..rhs.len() {
        let expected = finite_diff_complex_rhs(f, lhs, rhs, index, 1e-6);
        assert!(
            (actual[index] - expected).norm() <= tol,
            "rhs index {index}: expected {expected:?}, got {:?}",
            actual[index]
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

    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(a_shape.clone(), a_data.clone()));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(b_shape.clone(), b_data.clone()));
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let product = einsum(&mut engine, &[&a, &b], "bij,bjk->bik").unwrap();
    let loss = product.reduce_sum(&[0, 1, 2]);
    let grad = loss.grad(&a).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);

    let f = |xs: &[f64]| {
        let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(a_shape.clone(), xs.to_vec()));
        let b =
            TracedTensor::from_tensor_concrete_shape(f64_tensor(b_shape.clone(), b_data.clone()));
        let mut engine = GraphExecutor::new(CpuBackend::new());
        let product = einsum(&mut engine, &[&a, &b], "bij,bjk->bik").unwrap();
        let loss = product.reduce_sum(&[0, 1, 2]);
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
fn grad_real_sum_of_eigvals_matches_trace_gradient() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![3, 3],
        vec![1.0, 0.0, 0.0, 0.2, 2.0, 0.0, -0.1, 0.3, 4.0],
    ));

    let loss = tenferro::traced_tensor::eigvals(&a)
        .convert(DType::F64)
        .reduce_sum(&[0]);
    let grad = loss.grad(&a).unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let results = run_many_traced_with(&mut engine, &[&loss, &grad]).unwrap();

    assert_close_slice(get_f64_data(&results[0]), &[7.0]);
    assert_close_slice(
        get_f64_data(&results[1]),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
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

#[test]
fn test_reduce_prod_jvp() {
    let op = StdTensorOp::ReduceProd { axes: vec![0] };
    let input_shape = vec![4usize];
    let (fragment, input_key, output_key) =
        build_unary_fragment(op.clone(), tensor_input_key(60_000));
    let x_data = vec![1.5, -2.0, 0.75, 4.0];
    let dx_data = vec![0.25, -0.5, 1.0, 0.75];
    let mut inputs_map = HashMap::new();
    inputs_map.insert(
        input_key.clone(),
        f64_tensor(input_shape.clone(), x_data.clone()),
    );

    let tangent = jvp_from_fragment_with_inputs(
        fragment,
        output_key,
        input_key,
        inputs_map,
        f64_tensor(input_shape.clone(), dx_data.clone()),
    );

    assert_jvp_matches_finite_diff(get_f64_data(&tangent), &x_data, &dx_data, |xs| {
        eval_f64_reduction_op(&op, &input_shape, xs)
    });
}

#[test]
fn test_reduce_prod_vjp() {
    let op = StdTensorOp::ReduceProd { axes: vec![0] };
    let input_shape = vec![2usize, 3];
    let input_key = tensor_input_key(60_001);
    let x_data = vec![1.5, -2.0, 0.75, 4.0, -0.5, 2.0];
    let cotangent = vec![0.5, -1.0, 2.0];
    let grad = transpose_primal_unary_op_with_inputs(
        op.clone(),
        input_key,
        f64_tensor(input_shape.clone(), x_data.clone()),
        f64_tensor(vec![3], cotangent.clone()),
    );

    assert_grad_matches_finite_diff(get_f64_data(&grad), &x_data, |xs| {
        eval_f64_reduction_op(&op, &input_shape, xs)
            .iter()
            .zip(cotangent.iter())
            .map(|(value, weight)| value * weight)
            .sum()
    });
}

#[test]
fn test_reduce_max_jvp() {
    let op = StdTensorOp::ReduceMax { axes: vec![0] };
    let input_shape = vec![2usize, 3];
    let (fragment, input_key, output_key) =
        build_unary_fragment(op.clone(), tensor_input_key(60_002));
    let x_data = vec![2.0, 2.0, 4.0, 1.0, -3.0, -3.0];
    let dx_data = vec![1.0, -0.5, 0.75, -1.25, 2.0, -1.0];
    let mut inputs_map = HashMap::new();
    inputs_map.insert(
        input_key.clone(),
        f64_tensor(input_shape.clone(), x_data.clone()),
    );

    let tangent = jvp_from_fragment_with_inputs(
        fragment,
        output_key,
        input_key,
        inputs_map,
        f64_tensor(input_shape.clone(), dx_data.clone()),
    );

    assert_jvp_matches_finite_diff(get_f64_data(&tangent), &x_data, &dx_data, |xs| {
        eval_f64_reduction_op(&op, &input_shape, xs)
    });
}

#[test]
fn test_reduce_max_vjp() {
    let op = StdTensorOp::ReduceMax { axes: vec![0] };
    let input_shape = vec![4usize];
    let input_key = tensor_input_key(60_003);
    let x_data = vec![1.0, 3.0, 3.0, -2.0];
    let cotangent = 2.5;
    let grad = transpose_primal_unary_op_with_inputs(
        op.clone(),
        input_key,
        f64_tensor(input_shape.clone(), x_data.clone()),
        scalar_f64_tensor(cotangent),
    );

    assert_grad_matches_finite_diff(get_f64_data(&grad), &x_data, |xs| {
        cotangent * eval_f64_reduction_op(&op, &input_shape, xs)[0]
    });
}

#[test]
fn test_reduce_min_jvp() {
    let op = StdTensorOp::ReduceMin { axes: vec![0] };
    let input_shape = vec![4usize];
    let (fragment, input_key, output_key) =
        build_unary_fragment(op.clone(), tensor_input_key(60_004));
    let x_data = vec![1.0, -2.0, 4.0, -2.0];
    let dx_data = vec![0.5, 1.0, -1.0, -0.5];
    let mut inputs_map = HashMap::new();
    inputs_map.insert(
        input_key.clone(),
        f64_tensor(input_shape.clone(), x_data.clone()),
    );

    let tangent = jvp_from_fragment_with_inputs(
        fragment,
        output_key,
        input_key,
        inputs_map,
        f64_tensor(input_shape.clone(), dx_data.clone()),
    );

    assert_jvp_matches_finite_diff(get_f64_data(&tangent), &x_data, &dx_data, |xs| {
        eval_f64_reduction_op(&op, &input_shape, xs)
    });
}

#[test]
fn test_reduce_min_vjp() {
    let op = StdTensorOp::ReduceMin { axes: vec![0] };
    let input_shape = vec![2usize, 3];
    let input_key = tensor_input_key(60_005);
    let x_data = vec![-4.0, -4.0, 0.5, 2.0, 1.0, 1.0];
    let cotangent = vec![1.5, -0.25, 0.75];
    let grad = transpose_primal_unary_op_with_inputs(
        op.clone(),
        input_key,
        f64_tensor(input_shape.clone(), x_data.clone()),
        f64_tensor(vec![3], cotangent.clone()),
    );

    assert_grad_matches_finite_diff(get_f64_data(&grad), &x_data, |xs| {
        eval_f64_reduction_op(&op, &input_shape, xs)
            .iter()
            .zip(cotangent.iter())
            .map(|(value, weight)| value * weight)
            .sum()
    });
}

#[test]
fn grad_broadcast_reduce() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let y = x.broadcast_in_dim(&[3, 3], &[0]);
    let loss = y.reduce_sum(&[0, 1]);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_close_slice(get_f64_data(&result), &[3.0, 3.0, 3.0]);
}

#[test]
fn grad_broadcast_add_singleton_lhs() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![1], vec![1.0]));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let loss = (&a + &b).sum(&[0]);
    let grad = loss.grad(&a).unwrap();

    let result = eval_tensor(grad);
    assert_close_slice(get_f64_data(&result), &[3.0]);
}

#[test]
fn grad_reshape() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
    let y = x.reshape(&[2, 2]);
    let loss = y.reduce_sum(&[0, 1]);
    let grad = loss.grad(&x).unwrap();

    let result = eval_tensor(grad);
    assert_eq!(result.shape(), &[4]);
    assert_close_slice(get_f64_data(&result), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn grad_transpose() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let y = a.transpose(&[1, 0]);
    let loss = y.reduce_sum(&[0, 1]);
    let grad = loss.grad(&a).unwrap();

    let result = eval_tensor(grad);
    assert_eq!(result.shape(), &[2, 3]);
    assert_close_slice(get_f64_data(&result), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn grad_exp() {
    let x_data = vec![0.2, -0.7, 1.3];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.exp().sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| x.exp()).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.exp().sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_log() {
    let x_data = vec![0.8, 1.5, 2.4];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.log().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| 1.0 / x).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.log().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_sin_cos() {
    let x_data = vec![0.2, -0.7, 1.3];

    let x_sin = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let sin_loss = x_sin.sin().reduce_sum(&[0]);
    let sin_grad = sin_loss.grad(&x_sin).unwrap();
    let sin_grad_tensor = eval_tensor(sin_grad);
    let sin_grad_data = get_f64_data(&sin_grad_tensor);
    let expected_sin: Vec<f64> = x_data.iter().map(|x| x.cos()).collect();
    assert_close_slice(sin_grad_data, &expected_sin);

    let f_sin = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.sin().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(sin_grad_data, &x_data, f_sin);

    let x_cos = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let cos_loss = x_cos.cos().reduce_sum(&[0]);
    let cos_grad = cos_loss.grad(&x_cos).unwrap();
    let cos_grad_tensor = eval_tensor(cos_grad);
    let cos_grad_data = get_f64_data(&cos_grad_tensor);
    let expected_cos: Vec<f64> = x_data.iter().map(|x| -x.sin()).collect();
    assert_close_slice(cos_grad_data, &expected_cos);

    let f_cos = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.cos().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(cos_grad_data, &x_data, f_cos);
}

#[test]
fn grad_div() {
    let x_data = vec![1.2, -2.4, 3.6];
    let y_data = vec![0.5, -1.5, 2.0];

    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], y_data.clone()));
    let loss = (&x / &y).reduce_sum(&[0]);

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
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], lhs.to_vec()));
        let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], rhs.to_vec()));
        eval_scalar((&x / &y).reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff_lhs(grad_x_data, &x_data, &y_data, &f);
    assert_grad_matches_finite_diff_rhs(grad_y_data, &x_data, &y_data, &f);
}

#[test]
fn grad_sqrt() {
    let x_data = vec![0.8, 1.5, 3.2];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.sqrt().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| 0.5 / x.sqrt()).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.sqrt().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_tanh() {
    let x_data = vec![0.2, -0.7, 1.3];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.tanh().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| 1.0 - x.tanh().powi(2)).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.tanh().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_pow() {
    let x_data = vec![0.7, 1.3, 2.1];
    let y_data = vec![2.0, 2.0, 2.0];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], y_data.clone()));
    let loss = x.pow(&y).reduce_sum(&[0]);

    let grad_x = loss.grad(&x).unwrap();
    let grad_x_tensor = eval_tensor(grad_x);
    let grad_x_data = get_f64_data(&grad_x_tensor);
    let expected_x: Vec<f64> = x_data.iter().map(|x| 2.0 * x).collect();
    assert_close_slice(grad_x_data, &expected_x);

    let f = |lhs: &[f64], rhs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], lhs.to_vec()));
        let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], rhs.to_vec()));
        eval_scalar(x.pow(&y).reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff_lhs(grad_x_data, &x_data, &y_data, &f);
}

#[test]
fn grad_pow_wrt_exponent() {
    let x_data = vec![1.2, 1.8, 2.5];
    let y_data = vec![0.5, 1.5, 2.0];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], y_data.clone()));
    let loss = x.pow(&y).reduce_sum(&[0]);

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
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], lhs.to_vec()));
        let y = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], rhs.to_vec()));
        eval_scalar(x.pow(&y).reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff_rhs(grad_y_data, &x_data, &y_data, &f);
}

#[test]
fn grad_abs() {
    let x_data = vec![-1.7, 0.8, 2.3];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.abs().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected = [-1.0, 1.0, 1.0];
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.abs().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_sign() {
    let x_data = vec![-1.7, 0.8, 2.3];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.sign().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    assert_close_slice(grad_data, &[0.0, 0.0, 0.0]);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.sign().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_rsqrt() {
    let x_data = vec![0.8, 1.5, 3.2];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.rsqrt().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| -0.5 / (x * x.sqrt())).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.rsqrt().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_expm1() {
    let x_data = vec![0.2, -0.7, 1.3];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.expm1().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| x.exp()).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.expm1().reduce_sum(&[0]))
    };
    assert_grad_matches_finite_diff(grad_data, &x_data, f);
}

#[test]
fn grad_log1p() {
    let x_data = vec![0.2, 0.7, 1.3];
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], x_data.clone()));
    let loss = x.log1p().reduce_sum(&[0]);
    let grad = loss.grad(&x).unwrap();

    let grad_tensor = eval_tensor(grad);
    let grad_data = get_f64_data(&grad_tensor);
    let expected: Vec<f64> = x_data.iter().map(|x| 1.0 / (x + 1.0)).collect();
    assert_close_slice(grad_data, &expected);

    let f = |xs: &[f64]| {
        let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], xs.to_vec()));
        eval_scalar(x.log1p().reduce_sum(&[0]))
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
fn grad_svd_rectangular_tall_matches_finite_diff() {
    let shape = [4, 2];
    let a_data = vec![3.0, -0.5, 1.2, 0.8, 0.7, 2.1, -1.4, 0.3];
    let (fragment, input_key, loss_key) =
        build_svd_reconstruction_sum_fragment(tensor_input_key(16_000), shape.to_vec(), 1.0e-12);
    let grad = grad_from_fragment(
        fragment,
        loss_key,
        input_key,
        f64_tensor(shape.to_vec(), a_data.clone()),
    );
    let grad_data = get_f64_data(&grad);

    for index in 0..a_data.len() {
        let expected =
            finite_diff_scalar(|xs| sum_svd_reconstruction(xs, shape), &a_data, index, 1e-6);
        assert!(
            (grad_data[index] - expected).abs() <= 1e-4,
            "index {index}: expected {expected}, got {}",
            grad_data[index]
        );
    }
}

#[test]
fn grad_svd_rectangular_wide_matches_finite_diff() {
    let shape = [2, 4];
    let a_data = vec![2.4, -0.6, 1.1, 0.5, -1.3, 0.9, 0.7, 1.8];
    let (fragment, input_key, loss_key) =
        build_svd_reconstruction_sum_fragment(tensor_input_key(16_100), shape.to_vec(), 1.0e-12);
    let grad = grad_from_fragment(
        fragment,
        loss_key,
        input_key,
        f64_tensor(shape.to_vec(), a_data.clone()),
    );
    let grad_data = get_f64_data(&grad);

    for index in 0..a_data.len() {
        let expected =
            finite_diff_scalar(|xs| sum_svd_reconstruction(xs, shape), &a_data, index, 1e-6);
        assert!(
            (grad_data[index] - expected).abs() <= 1e-4,
            "index {index}: expected {expected}, got {}",
            grad_data[index]
        );
    }
}

#[test]
fn grad_svd_custom_eps() {
    let shape = [2, 2];
    let a_data = vec![2.2, -0.4, 1.1, 0.8];
    let (fragment, input_key, loss_key) =
        build_svd_reconstruction_sum_fragment(tensor_input_key(16_200), shape.to_vec(), 1.0e-8);
    let grad = grad_from_fragment(
        fragment,
        loss_key,
        input_key,
        f64_tensor(shape.to_vec(), a_data.clone()),
    );
    let grad_data = get_f64_data(&grad);

    for index in 0..a_data.len() {
        let expected =
            finite_diff_scalar(|xs| sum_svd_reconstruction(xs, shape), &a_data, index, 1e-6);
        assert!(
            (grad_data[index] - expected).abs() <= 1e-5,
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
fn grad_full_piv_lu_solve_matches_finite_diff() {
    let a_data = vec![0.2, 2.0, 1.0, 3.0];
    let b_data = vec![-1.0, 5.0];
    let (fragment, a_key, b_key, loss_key) = build_full_piv_lu_solve_sum_fragment();

    let mut a_inputs = HashMap::new();
    a_inputs.insert(a_key.clone(), f64_tensor(vec![2, 2], a_data.clone()));
    a_inputs.insert(b_key.clone(), f64_tensor(vec![2, 1], b_data.clone()));
    let grad_a =
        grad_from_fragment_with_inputs(fragment.clone(), loss_key.clone(), a_key, a_inputs);
    let grad_a_data = get_f64_data(&grad_a);

    let mut b_inputs = HashMap::new();
    b_inputs.insert(
        tensor_input_key(38_000),
        f64_tensor(vec![2, 2], a_data.clone()),
    );
    b_inputs.insert(b_key.clone(), f64_tensor(vec![2, 1], b_data.clone()));
    let grad_b = grad_from_fragment_with_inputs(fragment, loss_key, b_key, b_inputs);
    let grad_b_data = get_f64_data(&grad_b);

    assert_grad_matches_finite_diff_lhs(grad_a_data, &a_data, &b_data, &sum_full_piv_lu_solve);
    assert_grad_matches_finite_diff_rhs(grad_b_data, &a_data, &b_data, &sum_full_piv_lu_solve);
}

#[test]
fn jvp_full_piv_lu_lu_matches_finite_diff() {
    let a_data = vec![0.2, 2.0, 1.0, 3.0];
    let tangent_data = vec![0.5, -1.0, 1.5, -0.25];
    let (fragment, input_key, loss_key) = build_full_piv_lu_lu_sum_fragment();

    let mut inputs = HashMap::new();
    inputs.insert(input_key.clone(), f64_tensor(vec![2, 2], a_data.clone()));
    let tangent = jvp_from_fragment_with_inputs(
        fragment,
        loss_key,
        input_key,
        inputs,
        f64_tensor(vec![2, 2], tangent_data.clone()),
    );

    assert_jvp_matches_finite_diff(get_f64_data(&tangent), &a_data, &tangent_data, |xs| {
        vec![sum_full_piv_lu_lu(xs)]
    });
}

#[test]
fn grad_full_piv_lu_lu_matches_finite_diff() {
    let a_data = vec![0.2, 2.0, 1.0, 3.0];
    let (fragment, input_key, loss_key) = build_full_piv_lu_lu_sum_fragment();

    let grad = grad_from_fragment(
        fragment,
        loss_key,
        input_key,
        f64_tensor(vec![2, 2], a_data.clone()),
    );

    assert_grad_matches_finite_diff(get_f64_data(&grad), &a_data, sum_full_piv_lu_lu);
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

#[test]
fn grad_cholesky_matches_finite_diff_on_lower_triangle() {
    let a_data = vec![5.0, 1.0, 0.5, 1.0, 4.0, 0.75, 0.5, 0.75, 3.5];
    let (fragment, input_key, loss_key) = build_cholesky_sum_fragment();
    let grad = grad_from_fragment(
        fragment,
        loss_key,
        input_key,
        f64_tensor(vec![3, 3], a_data.clone()),
    );
    let grad_data = get_f64_data(&grad);

    for index in [0usize, 1, 2, 4, 5, 8] {
        let expected = finite_diff_scalar(sum_cholesky, &a_data, index, 1e-6);
        assert!(
            (grad_data[index] - expected).abs() <= 1e-4,
            "index {index}: expected {expected}, got {}",
            grad_data[index]
        );
    }
}

#[test]
fn grad_qr_r_matches_finite_diff() {
    let a_data = vec![1.5, 0.2, 0.4, 0.3, 1.7, -0.6];
    let (fragment, input_key, loss_key) = build_qr_r_sum_fragment();
    let grad = grad_from_fragment(
        fragment,
        loss_key,
        input_key,
        f64_tensor(vec![3, 2], a_data.clone()),
    );
    let grad_data = get_f64_data(&grad);

    for index in 0..a_data.len() {
        let expected = finite_diff_scalar(sum_qr_r, &a_data, index, 1e-6);
        assert!(
            (grad_data[index] - expected).abs() <= 1e-4,
            "index {index}: expected {expected}, got {}",
            grad_data[index]
        );
    }
}

#[test]
fn grad_eigh_values_complex_matches_finite_diff() {
    let a_data = vec![
        Complex64::new(3.0, 0.0),
        Complex64::new(1.0, -2.0),
        Complex64::new(1.0, 2.0),
        Complex64::new(4.0, 0.0),
    ];
    let (fragment, input_key, loss_key) = build_eigh_values_real_sum_complex_fragment();
    let grad = grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        loss_key,
        input_key.clone(),
        HashMap::from([(input_key, c64_tensor(vec![2, 2], a_data.clone()))]),
        scalar_c64_tensor(Complex64::new(1.0, 0.0)),
    );
    let grad_data = get_c64_data(&grad);

    assert_grad_matches_complex_finite_diff(
        grad_data,
        &a_data,
        &[0, 1, 3],
        1e-5,
        sum_eigh_values_complex,
    );
}

#[test]
fn grad_eigh_projector_complex_matches_finite_diff() {
    let a_data = vec![
        Complex64::new(4.0, 0.0),
        Complex64::new(1.0, -2.0),
        Complex64::new(1.0, 2.0),
        Complex64::new(2.5, 0.0),
    ];
    let weights_data = vec![Complex64::new(0.8, 0.3), Complex64::new(-0.4, 0.5)];
    let probe_data = vec![
        Complex64::new(1.2, -0.2),
        Complex64::new(-0.4, 0.7),
        Complex64::new(0.6, -0.3),
        Complex64::new(0.9, 0.1),
    ];
    let (fragment, input_key, weights_key, probe_key, loss_key) =
        build_eigh_projector_real_sum_complex_fragment();
    let grad = grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        loss_key,
        input_key.clone(),
        HashMap::from([
            (input_key, c64_tensor(vec![2, 2], a_data.clone())),
            (weights_key, c64_tensor(vec![2], weights_data.clone())),
            (probe_key, c64_tensor(vec![2, 2], probe_data.clone())),
        ]),
        scalar_c64_tensor(Complex64::new(1.0, 0.0)),
    );
    let grad_data = get_c64_data(&grad);

    assert_grad_matches_complex_finite_diff(grad_data, &a_data, &[0, 1, 3], 4e-4, |xs| {
        sum_eigh_projector_real_parts_complex(xs, &weights_data, &probe_data)
    });
}

#[test]
fn grad_svd_values_complex_with_gauge_matches_finite_diff() {
    let a_data = vec![
        Complex64::new(3.0, 0.4),
        Complex64::new(0.7, -0.5),
        Complex64::new(-0.4, 0.6),
        Complex64::new(-0.6, 0.8),
        Complex64::new(2.1, 0.2),
        Complex64::new(0.3, -1.0),
        Complex64::new(0.2, -0.3),
        Complex64::new(-0.9, 0.4),
        Complex64::new(1.4, 0.7),
    ];
    let (fragment, input_key, loss_key) = build_svd_values_real_sum_complex_3x3_fragment();
    let grad = grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        loss_key,
        input_key.clone(),
        HashMap::from([(input_key, c64_tensor(vec![3, 3], a_data.clone()))]),
        scalar_c64_tensor(Complex64::new(1.0, 0.0)),
    );
    let grad_data = get_c64_data(&grad);

    assert_grad_matches_complex_finite_diff(
        grad_data,
        &a_data,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8],
        3e-4,
        sum_svd_values_complex_3x3,
    );
}

#[test]
fn grad_svd_uv_product_complex_matches_finite_diff() {
    let a_data = vec![
        Complex64::new(2.7, -0.2),
        Complex64::new(-0.5, 0.9),
        Complex64::new(0.3, -0.4),
        Complex64::new(0.4, 0.7),
        Complex64::new(1.9, 0.1),
        Complex64::new(-0.8, 0.6),
        Complex64::new(-0.2, 0.5),
        Complex64::new(0.6, -1.2),
        Complex64::new(1.3, 0.8),
    ];
    let (fragment, input_key, loss_key) = build_svd_uv_product_real_sum_complex_fragment();
    let grad = grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        loss_key,
        input_key.clone(),
        HashMap::from([(input_key, c64_tensor(vec![3, 3], a_data.clone()))]),
        scalar_c64_tensor(Complex64::new(1.0, 0.0)),
    );
    let grad_data = get_c64_data(&grad);

    assert_grad_matches_complex_finite_diff(
        grad_data,
        &a_data,
        &[0, 1, 2, 3, 4, 5, 6, 7, 8],
        8e-4,
        sum_svd_uv_product_real_parts_complex,
    );
}

#[test]
fn grad_svd_values_complex_matches_finite_diff() {
    let a_data = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-0.7, 0.3),
        Complex64::new(0.2, -1.1),
        Complex64::new(2.3, -0.4),
    ];
    let (fragment, input_key, loss_key) = build_svd_values_real_sum_complex_fragment();
    let grad = grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        loss_key,
        input_key.clone(),
        HashMap::from([(input_key, c64_tensor(vec![2, 2], a_data.clone()))]),
        scalar_c64_tensor(Complex64::new(1.0, 0.0)),
    );
    let grad_data = get_c64_data(&grad);

    assert_grad_matches_complex_finite_diff(
        grad_data,
        &a_data,
        &[0, 1, 2, 3],
        2e-4,
        sum_svd_values_complex,
    );
}

#[test]
fn grad_cholesky_complex_matches_finite_diff() {
    let a_data = vec![
        Complex64::new(4.0, 0.0),
        Complex64::new(1.0, -1.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(3.0, 0.0),
    ];
    let (fragment, input_key, loss_key) = build_cholesky_real_sum_fragment();
    let grad = grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        loss_key,
        input_key.clone(),
        HashMap::from([(input_key, c64_tensor(vec![2, 2], a_data.clone()))]),
        scalar_c64_tensor(Complex64::new(1.0, 0.0)),
    );
    let grad_data = get_c64_data(&grad);

    assert_grad_matches_complex_finite_diff(
        grad_data,
        &a_data,
        &[0, 1, 3],
        2e-4,
        sum_cholesky_real_parts_complex,
    );
}

#[test]
fn grad_qr_r_complex_matches_finite_diff() {
    let a_data = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0001),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, -0.0001),
        Complex64::new(1.7, 0.0),
        Complex64::new(0.4, 0.00008),
    ];
    let (fragment, input_key, loss_key) = build_qr_r_real_sum_fragment();
    let grad = grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        loss_key,
        input_key.clone(),
        HashMap::from([(input_key, c64_tensor(vec![3, 2], a_data.clone()))]),
        scalar_c64_tensor(Complex64::new(1.0, 0.0)),
    );
    let grad_data = get_c64_data(&grad);

    assert_grad_matches_complex_finite_diff(
        grad_data,
        &a_data,
        &[0, 1, 2, 3, 4, 5],
        4e-4,
        sum_qr_r_real_parts_complex,
    );
}

#[test]
fn grad_solve_complex_matches_finite_diff() {
    let a_data = vec![
        Complex64::new(2.0, 0.5),
        Complex64::new(1.1, -0.2),
        Complex64::new(-0.3, 0.7),
        Complex64::new(3.0, -0.4),
    ];
    let b_data = vec![Complex64::new(1.0, -0.25), Complex64::new(-0.5, 1.5)];
    let (fragment, a_key, b_key, loss_key) = build_solve_real_sum_fragment();

    let grad_a = grad_from_fragment_with_inputs_and_cotangent(
        fragment.clone(),
        loss_key.clone(),
        a_key.clone(),
        HashMap::from([
            (a_key.clone(), c64_tensor(vec![2, 2], a_data.clone())),
            (b_key.clone(), c64_tensor(vec![2, 1], b_data.clone())),
        ]),
        scalar_c64_tensor(Complex64::new(1.0, 0.0)),
    );
    let grad_a_data = get_c64_data(&grad_a);

    let grad_b = grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        loss_key,
        b_key.clone(),
        HashMap::from([
            (a_key, c64_tensor(vec![2, 2], a_data.clone())),
            (b_key, c64_tensor(vec![2, 1], b_data.clone())),
        ]),
        scalar_c64_tensor(Complex64::new(1.0, 0.0)),
    );
    let grad_b_data = get_c64_data(&grad_b);

    assert_grad_matches_complex_finite_diff_lhs(
        grad_a_data,
        &a_data,
        &b_data,
        2e-4,
        &sum_solve_real_parts_complex,
    );
    assert_grad_matches_complex_finite_diff_rhs(
        grad_b_data,
        &a_data,
        &b_data,
        2e-4,
        &sum_solve_real_parts_complex,
    );
}

#[test]
fn grad_triangular_solve_complex_matches_finite_diff() {
    let a_data = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(0.75, -0.5),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.5, 0.25),
    ];
    let b_data = vec![Complex64::new(0.5, 1.0), Complex64::new(-1.2, 0.3)];
    let (fragment, a_key, b_key, loss_key) = build_triangular_solve_real_sum_fragment();
    let grad_b = grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        loss_key,
        b_key.clone(),
        HashMap::from([
            (a_key, c64_tensor(vec![2, 2], a_data.clone())),
            (b_key, c64_tensor(vec![2, 1], b_data.clone())),
        ]),
        scalar_c64_tensor(Complex64::new(1.0, 0.0)),
    );
    let grad_b_data = get_c64_data(&grad_b);

    assert_grad_matches_complex_finite_diff_rhs(
        grad_b_data,
        &a_data,
        &b_data,
        1e-5,
        &sum_triangular_solve_real_parts_complex,
    );
}

#[test]
fn grad_svd_values_repeated_singular_values_is_finite() {
    let a_data = vec![1.0, 0.0, 0.0, 1.0];
    let (fragment, input_key, loss_key) = build_svd_values_sum_fragment();
    let grad = grad_from_fragment(
        fragment,
        loss_key,
        input_key,
        f64_tensor(vec![2, 2], a_data),
    );

    for &value in get_f64_data(&grad) {
        assert!(value.is_finite(), "svd gradient not finite: {value}");
    }
}

#[test]
fn grad_eigh_values_degenerate_spectrum_is_finite() {
    let a_data = vec![2.0, 0.0, 0.0, 2.0];
    let (fragment, input_key, loss_key) = build_eigh_values_sum_fragment();
    let grad = grad_from_fragment(
        fragment,
        loss_key,
        input_key,
        f64_tensor(vec![2, 2], a_data),
    );

    for &value in get_f64_data(&grad) {
        assert!(value.is_finite(), "eigh gradient not finite: {value}");
    }
}

/// Build `y = ReduceSum(Gather(operand, indices, config))` and return the
/// fragment with keys for later gradient extraction.
fn build_gather_reduce_sum_fragment(
    operand_key: TensorInputKey,
    indices_key: TensorInputKey,
    config: GatherConfig,
    reduce_axes: Vec<usize>,
) -> (Arc<Fragment<StdTensorOp>>, GlobalValKey<StdTensorOp>) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let operand = builder.add_input(operand_key);
    let indices = builder.add_input(indices_key);
    let gathered = builder.add_op(
        StdTensorOp::Gather(config),
        vec![ValRef::Local(operand), ValRef::Local(indices)],
        OpMode::Primal,
    )[0];
    let loss = builder.add_op(
        StdTensorOp::ReduceSum { axes: reduce_axes },
        vec![ValRef::Local(gathered)],
        OpMode::Primal,
    )[0];
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), loss_key)
}

#[test]
fn grad_gather_reduce_sum_accumulates_indices_correctly() {
    // operand is rank-1 with five distinct values; gather reads indices
    // [2, 4, 0, 2, 2] so the gradient of the summed output w.r.t. the
    // operand should count the number of times each index is read: index
    // 0 once, index 2 three times, index 4 once, and other slots zero.
    let operand_key = tensor_input_key(60_000);
    let indices_key = tensor_input_key(60_001);
    let config = GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };
    let (fragment, loss_key) =
        build_gather_reduce_sum_fragment(operand_key.clone(), indices_key.clone(), config, vec![0]);

    let operand_data = vec![10.0_f64, 20.0, 30.0, 40.0, 50.0];
    let indices_data = vec![2_i64, 4, 0, 2, 2];
    let mut inputs_map = HashMap::new();
    inputs_map.insert(
        operand_key.clone(),
        f64_tensor(vec![5], operand_data.clone()),
    );
    inputs_map.insert(indices_key, i64_tensor(vec![5, 1], indices_data));

    let grad = grad_from_fragment_with_inputs(fragment, loss_key, operand_key, inputs_map);
    assert_eq!(grad.shape(), &[5]);
    assert_close_slice(get_f64_data(&grad), &[1.0, 0.0, 3.0, 0.0, 1.0]);
}

#[test]
fn grad_traced_index_select_repeated_positions_accumulates() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0]));
    let weights =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![10.0, 20.0, 30.0]));

    let selected = x.index_select(0, &[1, 1, 2]).unwrap();
    let loss = (&selected * &weights).reduce_sum(&[0]);
    let grad = eval_tensor(loss.grad(&x).unwrap());

    assert_eq!(grad.shape(), &[3]);
    assert_close_slice(get_f64_data(&grad), &[0.0, 30.0, 30.0]);
}

#[test]
fn jvp_traced_index_select_gathers_tangent() {
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]));
    let tangent =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![0.5, 1.5, 2.5, 3.5]));

    let y = x.index_select(0, &[3, 1, 3]).unwrap();
    let tangent_y = eval_tensor(y.jvp(&x, &tangent));

    assert_eq!(tangent_y.shape(), &[3]);
    assert_close_slice(get_f64_data(&tangent_y), &[3.5, 1.5, 3.5]);
}

/// Build `y = ReduceSum(Scatter(operand, indices, updates, config))`,
/// where the Scatter output has the same shape as `operand`.
fn build_scatter_reduce_sum_fragment(
    operand_key: TensorInputKey,
    indices_key: TensorInputKey,
    updates_key: TensorInputKey,
    config: ScatterConfig,
    reduce_axes: Vec<usize>,
) -> (Arc<Fragment<StdTensorOp>>, GlobalValKey<StdTensorOp>) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let operand = builder.add_input(operand_key);
    let indices = builder.add_input(indices_key);
    let updates = builder.add_input(updates_key);
    let scattered = builder.add_op(
        StdTensorOp::Scatter(config),
        vec![
            ValRef::Local(operand),
            ValRef::Local(indices),
            ValRef::Local(updates),
        ],
        OpMode::Primal,
    )[0];
    let loss = builder.add_op(
        StdTensorOp::ReduceSum { axes: reduce_axes },
        vec![ValRef::Local(scattered)],
        OpMode::Primal,
    )[0];
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), loss_key)
}

fn build_weighted_unary_sum_fragment(
    input_key: TensorInputKey,
    weights_key: TensorInputKey,
    op: StdTensorOp,
    reduce_axes: Vec<usize>,
) -> (Arc<Fragment<StdTensorOp>>, GlobalValKey<StdTensorOp>) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let input = builder.add_input(input_key);
    let weights = builder.add_input(weights_key);
    let output = builder.add_op(op, vec![ValRef::Local(input)], OpMode::Primal)[0];
    let weighted = builder.add_op(
        StdTensorOp::Mul,
        vec![ValRef::Local(output), ValRef::Local(weights)],
        OpMode::Primal,
    )[0];
    let loss = builder.add_op(
        StdTensorOp::ReduceSum { axes: reduce_axes },
        vec![ValRef::Local(weighted)],
        OpMode::Primal,
    )[0];
    let loss_key = builder.global_key(loss).clone();
    builder.set_outputs(vec![loss]);
    (Arc::new(builder.build()), loss_key)
}

fn build_dynamic_slice_fragment(
    input_key: TensorInputKey,
    starts_key: TensorInputKey,
    slice_sizes: Vec<usize>,
) -> (Arc<Fragment<StdTensorOp>>, GlobalValKey<StdTensorOp>) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let input = builder.add_input(input_key);
    let starts = builder.add_input(starts_key);
    let output = builder.add_op(
        StdTensorOp::DynamicSlice { slice_sizes },
        vec![ValRef::Local(input), ValRef::Local(starts)],
        OpMode::Primal,
    )[0];
    let output_key = builder.global_key(output).clone();
    builder.set_outputs(vec![output]);
    (Arc::new(builder.build()), output_key)
}

fn build_dynamic_update_slice_fragment(
    operand_key: TensorInputKey,
    update_key: TensorInputKey,
    starts_key: TensorInputKey,
) -> (Arc<Fragment<StdTensorOp>>, GlobalValKey<StdTensorOp>) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let operand = builder.add_input(operand_key);
    let update = builder.add_input(update_key);
    let starts = builder.add_input(starts_key);
    let output = builder.add_op(
        StdTensorOp::DynamicUpdateSlice,
        vec![
            ValRef::Local(operand),
            ValRef::Local(update),
            ValRef::Local(starts),
        ],
        OpMode::Primal,
    )[0];
    let output_key = builder.global_key(output).clone();
    builder.set_outputs(vec![output]);
    (Arc::new(builder.build()), output_key)
}

#[test]
fn grad_scatter_reduce_sum_wrt_updates_is_ones() {
    // `y = reduce_sum(scatter(operand, indices, updates, config))`. The
    // scatter backward feeds `cot_out` (a ones tensor of operand shape)
    // into the inverse Gather at each `scatter_indices` entry. With all
    // indices in range, the updates gradient is ones of the updates shape.
    let operand_key = tensor_input_key(61_000);
    let indices_key = tensor_input_key(61_001);
    let updates_key = tensor_input_key(61_002);
    let config = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let (fragment, loss_key) = build_scatter_reduce_sum_fragment(
        operand_key.clone(),
        indices_key.clone(),
        updates_key.clone(),
        config,
        vec![0],
    );

    let operand_data = vec![0.0_f64, 0.0, 0.0, 0.0];
    let indices_data = vec![1_i64, 3, 0];
    let updates_data = vec![5.0_f64, 7.0, 9.0];
    let mut inputs_map = HashMap::new();
    inputs_map.insert(operand_key, f64_tensor(vec![4], operand_data));
    inputs_map.insert(indices_key, i64_tensor(vec![3, 1], indices_data));
    inputs_map.insert(
        updates_key.clone(),
        f64_tensor(vec![3], updates_data.clone()),
    );

    let grad = grad_from_fragment_with_inputs(fragment, loss_key, updates_key, inputs_map);
    assert_eq!(grad.shape(), &[3]);
    // reduce_sum over the scatter output contributes a 1 to each updated
    // slot; the inverse Gather reads one value for each `indices` entry.
    assert_close_slice(get_f64_data(&grad), &[1.0, 1.0, 1.0]);
}

#[test]
fn jvp_dynamic_slice_matches_finite_diff() {
    let input_key = tensor_input_key(61_500);
    let starts_key = tensor_input_key(61_501);
    let (fragment, output_key) =
        build_dynamic_slice_fragment(input_key.clone(), starts_key.clone(), vec![3]);

    let input_data = vec![0.5_f64, -1.0, 2.5, 4.0, -3.0];
    let starts_data = vec![1_i64];
    let tangent_data = vec![1.25_f64, -0.75, 3.0, 2.5, -1.0];
    let inputs_map = HashMap::from([
        (input_key.clone(), f64_tensor(vec![5], input_data.clone())),
        (starts_key, i64_tensor(vec![1], starts_data)),
    ]);

    let tangent = jvp_from_fragment_with_inputs(
        fragment,
        output_key,
        input_key,
        inputs_map,
        f64_tensor(vec![5], tangent_data.clone()),
    );

    assert_jvp_matches_finite_diff(get_f64_data(&tangent), &input_data, &tangent_data, |xs| {
        xs[1..4].to_vec()
    });
}

#[test]
fn grad_dynamic_slice_clamped_start_matches_finite_diff() {
    let input_key = tensor_input_key(61_510);
    let starts_key = tensor_input_key(61_511);
    let (fragment, output_key) =
        build_dynamic_slice_fragment(input_key.clone(), starts_key.clone(), vec![3]);

    let input_data = vec![0.5_f64, -1.0, 2.5, 4.0, -3.0];
    let starts_data = vec![4_i64];
    let cotangent_data = vec![0.5_f64, -1.0, 2.0];
    let inputs_map = HashMap::from([
        (input_key.clone(), f64_tensor(vec![5], input_data.clone())),
        (starts_key, i64_tensor(vec![1], starts_data)),
    ]);

    let grad = grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        output_key,
        input_key,
        inputs_map,
        f64_tensor(vec![3], cotangent_data.clone()),
    );

    let loss = |xs: &[f64]| {
        xs[2..5]
            .iter()
            .zip(cotangent_data.iter())
            .map(|(&value, &weight)| value * weight)
            .sum()
    };
    let expected: Vec<f64> = (0..input_data.len())
        .map(|idx| finite_diff_scalar(loss, &input_data, idx, 1.0e-6))
        .collect();
    assert_close_slice(get_f64_data(&grad), &expected);
}

#[test]
fn jvp_dynamic_update_slice_update_matches_finite_diff() {
    let operand_key = tensor_input_key(61_520);
    let update_key = tensor_input_key(61_521);
    let starts_key = tensor_input_key(61_522);
    let (fragment, output_key) = build_dynamic_update_slice_fragment(
        operand_key.clone(),
        update_key.clone(),
        starts_key.clone(),
    );

    let operand_data = vec![10.0_f64, 11.0, 12.0, 13.0, 14.0];
    let update_data = vec![1.0_f64, 2.0, 3.0];
    let starts_data = vec![4_i64];
    let tangent_data = vec![0.5_f64, -1.0, 2.0];
    let inputs_map = HashMap::from([
        (operand_key, f64_tensor(vec![5], operand_data.clone())),
        (update_key.clone(), f64_tensor(vec![3], update_data.clone())),
        (starts_key, i64_tensor(vec![1], starts_data)),
    ]);

    let tangent = jvp_from_fragment_with_inputs(
        fragment,
        output_key,
        update_key,
        inputs_map,
        f64_tensor(vec![3], tangent_data.clone()),
    );

    assert_jvp_matches_finite_diff(get_f64_data(&tangent), &update_data, &tangent_data, |upd| {
        let mut out = operand_data.clone();
        out[2..5].copy_from_slice(upd);
        out
    });
}

#[test]
fn grad_dynamic_update_slice_matches_finite_diff() {
    let operand_key = tensor_input_key(61_530);
    let update_key = tensor_input_key(61_531);
    let starts_key = tensor_input_key(61_532);
    let (fragment, output_key) = build_dynamic_update_slice_fragment(
        operand_key.clone(),
        update_key.clone(),
        starts_key.clone(),
    );

    let operand_data = vec![10.0_f64, 11.0, 12.0, 13.0, 14.0];
    let update_data = vec![1.0_f64, 2.0, 3.0];
    let starts_data = vec![4_i64];
    let cotangent_data = vec![0.5_f64, -0.25, 1.0, 2.0, -1.5];
    let inputs_map = HashMap::from([
        (
            operand_key.clone(),
            f64_tensor(vec![5], operand_data.clone()),
        ),
        (update_key.clone(), f64_tensor(vec![3], update_data.clone())),
        (starts_key, i64_tensor(vec![1], starts_data)),
    ]);

    let grad_operand = grad_from_fragment_with_inputs_and_cotangent(
        fragment.clone(),
        output_key.clone(),
        operand_key,
        inputs_map.clone(),
        f64_tensor(vec![5], cotangent_data.clone()),
    );
    let grad_update = grad_from_fragment_with_inputs_and_cotangent(
        fragment,
        output_key,
        update_key,
        inputs_map,
        f64_tensor(vec![5], cotangent_data.clone()),
    );

    let loss_with = |operand: &[f64], update: &[f64]| {
        let mut out = operand.to_vec();
        out[2..5].copy_from_slice(update);
        out.iter()
            .zip(cotangent_data.iter())
            .map(|(&value, &weight)| value * weight)
            .sum()
    };
    let expected_operand: Vec<f64> = (0..operand_data.len())
        .map(|idx| finite_diff_scalar_lhs(loss_with, &operand_data, &update_data, idx, 1.0e-6))
        .collect();
    let expected_update: Vec<f64> = (0..update_data.len())
        .map(|idx| finite_diff_scalar_rhs(loss_with, &operand_data, &update_data, idx, 1.0e-6))
        .collect();

    assert_close_slice(get_f64_data(&grad_operand), &expected_operand);
    assert_close_slice(get_f64_data(&grad_update), &expected_update);
}

#[test]
fn grad_slice_weighted_sum_matches_finite_diff() {
    let input_key = tensor_input_key(62_000);
    let weights_key = tensor_input_key(62_001);
    let config = SliceConfig {
        starts: vec![1],
        limits: vec![5],
        strides: vec![2],
    };
    let (fragment, loss_key) = build_weighted_unary_sum_fragment(
        input_key.clone(),
        weights_key.clone(),
        StdTensorOp::Slice(config),
        vec![0],
    );

    let input_data = vec![0.5_f64, -1.0, 2.5, 4.0, -3.0];
    let weights_data = vec![1.25_f64, -0.75];
    let inputs_map = HashMap::from([
        (input_key.clone(), f64_tensor(vec![5], input_data.clone())),
        (weights_key, f64_tensor(vec![2], weights_data.clone())),
    ]);

    let grad = grad_from_fragment_with_inputs(fragment, loss_key, input_key, inputs_map);
    assert_grad_matches_finite_diff(get_f64_data(&grad), &input_data, |xs| {
        xs[1] * weights_data[0] + xs[3] * weights_data[1]
    });
}

#[test]
fn grad_pad_weighted_sum_matches_finite_diff() {
    let input_key = tensor_input_key(63_000);
    let weights_key = tensor_input_key(63_001);
    let config = PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![2],
        interior_padding: vec![1],
    };
    let (fragment, loss_key) = build_weighted_unary_sum_fragment(
        input_key.clone(),
        weights_key.clone(),
        StdTensorOp::Pad(config),
        vec![0],
    );

    let input_data = vec![2.0_f64, -1.5, 0.25];
    let weights_data = vec![0.5_f64, 1.25, -0.5, 2.0, 0.75, -1.0, 3.0, -2.5];
    let inputs_map = HashMap::from([
        (input_key.clone(), f64_tensor(vec![3], input_data.clone())),
        (weights_key, f64_tensor(vec![8], weights_data.clone())),
    ]);

    let grad = grad_from_fragment_with_inputs(fragment, loss_key, input_key, inputs_map);
    assert_grad_matches_finite_diff(get_f64_data(&grad), &input_data, |xs| {
        xs[0] * weights_data[1] + xs[1] * weights_data[3] + xs[2] * weights_data[5]
    });
}

#[test]
fn grad_reverse_weighted_sum_matches_finite_diff() {
    let input_key = tensor_input_key(64_000);
    let weights_key = tensor_input_key(64_001);
    let (fragment, loss_key) = build_weighted_unary_sum_fragment(
        input_key.clone(),
        weights_key.clone(),
        StdTensorOp::Reverse { axes: vec![0] },
        vec![0],
    );

    let input_data = vec![1.0_f64, -2.0, 3.5, 0.25];
    let weights_data = vec![0.5_f64, -1.0, 2.0, 1.5];
    let inputs_map = HashMap::from([
        (input_key.clone(), f64_tensor(vec![4], input_data.clone())),
        (weights_key, f64_tensor(vec![4], weights_data.clone())),
    ]);

    let grad = grad_from_fragment_with_inputs(fragment, loss_key, input_key, inputs_map);
    assert_grad_matches_finite_diff(get_f64_data(&grad), &input_data, |xs| {
        xs[0] * weights_data[3]
            + xs[1] * weights_data[2]
            + xs[2] * weights_data[1]
            + xs[3] * weights_data[0]
    });
}

#[test]
fn dropped_traced_graph_releases_registered_metadata() {
    let leaf_key;
    let derived_key;
    let y;

    {
        let x = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]);
        leaf_key = GlobalValKey::Input(x.input_key().expect("leaf input key"));

        y = &x + &x;
        derived_key = y.fragment.vals()[y.val].key.clone();

        assert!(lookup_global_metadata(&leaf_key).is_some());
        assert!(lookup_global_metadata(&derived_key).is_some());
    }

    assert!(lookup_global_metadata(&leaf_key).is_some());
    assert!(lookup_global_metadata(&derived_key).is_some());

    drop(y);

    assert!(lookup_global_metadata(&leaf_key).is_none());
    assert!(lookup_global_metadata(&derived_key).is_none());
}
