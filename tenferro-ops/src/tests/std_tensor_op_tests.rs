use crate::ad::context::ShapeGuardContext;
use crate::dim_expr::DimExpr;
use crate::ext_op::{register_extension_rule, ExtensionAdRule, ExtensionOp};
use crate::std_tensor_op::StdTensorOp;
use crate::{SymDim, TensorMeta};
use chainrules_core::{ADRuleKind, ADRuleResult, PrimitiveOp};
use computegraph::fragment::{Fragment, FragmentBuilder};
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::{GraphOp, OpEmitter};
use num_complex::{Complex32, Complex64};
use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig,
};

use crate::input_key::TensorInputKey;

macro_rules! shape {
    ($($dim:expr),* $(,)?) => {
        DimExpr::from_concrete(&[$($dim),*])
    };
}

fn sym_shape(dims: &[usize]) -> Vec<SymDim> {
    dims.iter().copied().map(SymDim::from).collect()
}

fn tensor_input_key(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

fn add_input_keys(
    builder: &mut FragmentBuilder<StdTensorOp>,
    start_id: u64,
    count: usize,
) -> Vec<GlobalValKey<StdTensorOp>> {
    (0..count)
        .map(|offset| {
            let local = builder.add_input(tensor_input_key(start_id + offset as u64));
            builder.global_key(local).clone()
        })
        .collect()
}

fn external_inputs(start_id: u64, count: usize) -> Vec<ValRef<StdTensorOp>> {
    (0..count)
        .map(|offset| {
            ValRef::External(GlobalValKey::Input(tensor_input_key(
                start_id + offset as u64,
            )))
        })
        .collect()
}

fn linear_mode(active_mask: &[bool]) -> OpMode {
    OpMode::Linear {
        active_mask: active_mask.to_vec(),
    }
}

fn seed_dot_general_input_metadata(
    ctx: &mut ShapeGuardContext,
    keys: &[GlobalValKey<StdTensorOp>],
) {
    let shapes = [
        vec![SymDim::from(2usize), SymDim::from(3usize)],
        vec![SymDim::from(3usize), SymDim::from(4usize)],
    ];
    for (key, shape) in keys.iter().zip(shapes) {
        ctx.insert_metadata(key.clone(), TensorMeta::exact(DType::F64, shape));
    }
}

fn seed_dot_general_ref_metadata(ctx: &mut ShapeGuardContext, inputs: &[ValRef<StdTensorOp>]) {
    let keys: Vec<_> = inputs
        .iter()
        .map(|input| match input {
            ValRef::External(key) => key.clone(),
            ValRef::Local(local_id) => {
                panic!("expected external input in test helper, got local {local_id}")
            }
        })
        .collect();
    seed_dot_general_input_metadata(ctx, &keys);
}

fn seed_uniform_ref_metadata(
    ctx: &mut ShapeGuardContext,
    inputs: &[ValRef<StdTensorOp>],
    shape: Vec<SymDim>,
) {
    seed_uniform_ref_metadata_with_dtype(ctx, inputs, DType::F64, shape);
}

fn seed_uniform_ref_metadata_with_dtype(
    ctx: &mut ShapeGuardContext,
    inputs: &[ValRef<StdTensorOp>],
    dtype: DType,
    shape: Vec<SymDim>,
) {
    for input in inputs {
        let key = match input {
            ValRef::External(key) => key.clone(),
            ValRef::Local(local_id) => {
                panic!("expected external input in test helper, got local {local_id}")
            }
        };
        ctx.insert_metadata(key, TensorMeta::exact(dtype, shape.clone()));
    }
}

fn run_linearize_case(
    op: StdTensorOp,
    n_primal_in: usize,
    n_primal_out: usize,
    tangent_mask: &[bool],
) -> (Vec<Option<LocalValId>>, Fragment<StdTensorOp>) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let primal_in = add_input_keys(&mut builder, 100, n_primal_in);
    let primal_out = add_input_keys(&mut builder, 200, n_primal_out);
    if matches!(&op, StdTensorOp::DotGeneral { .. }) {
        seed_dot_general_input_metadata(&mut ad_ctx, &primal_in);
    }
    let tangent_in: Vec<Option<LocalValId>> = tangent_mask
        .iter()
        .enumerate()
        .map(|(offset, &active)| {
            active.then(|| builder.add_input(tensor_input_key(300 + offset as u64)))
        })
        .collect();
    let result = op.linearize(
        &mut builder,
        &primal_in,
        &primal_out,
        &tangent_in,
        &mut ad_ctx,
    );
    (result, builder.build())
}

fn run_transpose_case(
    op: StdTensorOp,
    n_inputs: usize,
    active_mask: &[bool],
    cotangent_present: bool,
) -> (
    Vec<Option<LocalValId>>,
    Option<LocalValId>,
    Fragment<StdTensorOp>,
) {
    run_transpose_case_with_input_shape(op, n_inputs, active_mask, cotangent_present, None)
}

fn run_transpose_case_with_input_shape(
    op: StdTensorOp,
    n_inputs: usize,
    active_mask: &[bool],
    cotangent_present: bool,
    input_shape: Option<Vec<SymDim>>,
) -> (
    Vec<Option<LocalValId>>,
    Option<LocalValId>,
    Fragment<StdTensorOp>,
) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = cotangent_present.then(|| builder.add_input(tensor_input_key(400)));
    let inputs = external_inputs(500, n_inputs);
    if matches!(&op, StdTensorOp::DotGeneral { .. }) {
        seed_dot_general_ref_metadata(&mut ad_ctx, &inputs);
    } else if let Some(shape) = input_shape {
        seed_uniform_ref_metadata(&mut ad_ctx, &inputs, shape);
    }
    let result = op.transpose_rule(
        &mut builder,
        &[cotangent],
        &inputs,
        &linear_mode(active_mask),
        &mut ad_ctx,
    );
    (result, cotangent, builder.build())
}

/// Variant of `run_transpose_case` that seeds input shape metadata for
/// op kinds whose transpose rules query `ctx.shape_of`.
fn run_transpose_case_with_input_shapes(
    op: StdTensorOp,
    n_inputs: usize,
    active_mask: &[bool],
    cotangent_present: bool,
    input_shapes: &[&[usize]],
) -> (
    Vec<Option<LocalValId>>,
    Option<LocalValId>,
    Fragment<StdTensorOp>,
) {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = cotangent_present.then(|| builder.add_input(tensor_input_key(400)));
    let inputs = external_inputs(500, n_inputs);
    assert_eq!(
        input_shapes.len(),
        n_inputs,
        "seed shapes must match input count",
    );
    for (input, shape) in inputs.iter().zip(input_shapes.iter()) {
        let ValRef::External(key) = input else {
            panic!("expected external input reference");
        };
        ad_ctx.insert_metadata(
            key.clone(),
            TensorMeta::exact(
                DType::F64,
                shape.iter().copied().map(SymDim::from).collect(),
            ),
        );
    }
    let result = op.transpose_rule(
        &mut builder,
        &[cotangent],
        &inputs,
        &linear_mode(active_mask),
        &mut ad_ctx,
    );
    (result, cotangent, builder.build())
}

#[test]
fn test_std_tensor_op_input_output_counts() {
    assert_eq!(StdTensorOp::Add.n_inputs(), 2);
    assert_eq!(StdTensorOp::Mul.n_inputs(), 2);
    assert_eq!(StdTensorOp::Neg.n_inputs(), 1);
    assert_eq!(StdTensorOp::Conj.n_inputs(), 1);
    assert_eq!(
        StdTensorOp::DotGeneral {
            config: DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        }
        .n_inputs(),
        2
    );
    assert_eq!(StdTensorOp::ReduceSum { axes: vec![0] }.n_inputs(), 1);
    assert_eq!(StdTensorOp::constant_f64(1.0).n_inputs(), 0);
    assert_eq!(
        StdTensorOp::ExtractDiag {
            axis_a: 0,
            axis_b: 1
        }
        .n_inputs(),
        1
    );
    assert_eq!(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1
        }
        .n_inputs(),
        1
    );
    assert_eq!(
        StdTensorOp::Gather(GatherConfig {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        })
        .n_inputs(),
        2
    );
    assert_eq!(
        StdTensorOp::Scatter(ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        })
        .n_inputs(),
        3
    );
    assert_eq!(
        StdTensorOp::DynamicSlice {
            slice_sizes: vec![1]
        }
        .n_inputs(),
        2
    );
    assert_eq!(StdTensorOp::DynamicUpdateSlice.n_inputs(), 3);
    assert_eq!(
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![1],
            interior_padding: vec![0],
        })
        .n_inputs(),
        1
    );

    assert_eq!(StdTensorOp::Add.n_outputs(), 1);
    assert_eq!(StdTensorOp::Neg.n_outputs(), 1);
    assert_eq!(StdTensorOp::Compare(CompareDir::Eq).n_inputs(), 2);
    assert_eq!(StdTensorOp::Select.n_inputs(), 3);
    assert_eq!(StdTensorOp::Clamp.n_inputs(), 3);
    assert_eq!(StdTensorOp::Div.n_inputs(), 2);
    assert_eq!(StdTensorOp::Pow.n_inputs(), 2);
    assert_eq!(StdTensorOp::Abs.n_inputs(), 1);
    assert_eq!(StdTensorOp::Exp.n_inputs(), 1);
    assert_eq!(StdTensorOp::Log1p.n_inputs(), 1);
    assert_eq!(StdTensorOp::constant_f64(1.0).n_outputs(), 1);
    assert_eq!(
        StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1
        }
        .n_outputs(),
        1
    );
    assert_eq!(
        StdTensorOp::Gather(GatherConfig {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        })
        .n_outputs(),
        1
    );
    assert_eq!(
        StdTensorOp::Scatter(ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        })
        .n_outputs(),
        1
    );
    assert_eq!(
        StdTensorOp::DynamicSlice {
            slice_sizes: vec![1]
        }
        .n_outputs(),
        1
    );
    assert_eq!(StdTensorOp::DynamicUpdateSlice.n_outputs(), 1);
    assert_eq!(
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![1],
            interior_padding: vec![0],
        })
        .n_outputs(),
        1
    );
    assert_eq!(StdTensorOp::Tril { k: -1 }.n_inputs(), 1);
    assert_eq!(StdTensorOp::Tril { k: -1 }.n_outputs(), 1);
    assert_eq!(StdTensorOp::Triu { k: 1 }.n_inputs(), 1);
    assert_eq!(StdTensorOp::Triu { k: 1 }.n_outputs(), 1);
}

#[test]
fn test_std_tensor_op_remaining_input_output_counts() {
    let slice = tenferro_tensor::SliceConfig {
        starts: vec![0, 1],
        limits: vec![2, 3],
        strides: vec![1, 1],
    };

    assert_eq!(StdTensorOp::Maximum.n_inputs(), 2);
    assert_eq!(StdTensorOp::Maximum.n_outputs(), 1);
    assert_eq!(StdTensorOp::Minimum.n_inputs(), 2);
    assert_eq!(StdTensorOp::Minimum.n_outputs(), 1);
    assert_eq!(
        StdTensorOp::Convert {
            from: DType::F64,
            to: DType::C64,
        }
        .n_inputs(),
        1
    );
    assert_eq!(
        StdTensorOp::Convert {
            from: DType::F64,
            to: DType::C64,
        }
        .n_outputs(),
        1
    );
    assert_eq!(StdTensorOp::Slice(slice.clone()).n_inputs(), 1);
    assert_eq!(StdTensorOp::Slice(slice).n_outputs(), 1);
    assert_eq!(StdTensorOp::Reverse { axes: vec![0, 2] }.n_inputs(), 1);
    assert_eq!(StdTensorOp::Reverse { axes: vec![0, 2] }.n_outputs(), 1);
    assert_eq!(StdTensorOp::ShapeOf { axis: 1 }.n_inputs(), 1);
    assert_eq!(StdTensorOp::ShapeOf { axis: 1 }.n_outputs(), 1);
    assert_eq!(StdTensorOp::DynamicTruncate { axis: 0 }.n_inputs(), 2);
    assert_eq!(StdTensorOp::DynamicTruncate { axis: 0 }.n_outputs(), 1);
    assert_eq!(StdTensorOp::PadToMatch { axis: 0 }.n_inputs(), 2);
    assert_eq!(StdTensorOp::PadToMatch { axis: 0 }.n_outputs(), 1);
}

#[test]
fn test_std_tensor_op_concatenate_counts_use_recorded_arity() {
    let op = StdTensorOp::Concatenate {
        axis: 0,
        n_inputs: 3,
    };

    assert_eq!(op.n_inputs(), 3);
    assert_eq!(op.n_outputs(), 1);
}

#[test]
fn test_std_tensor_op_reduction_counts_cover_remaining_variants() {
    assert_eq!(StdTensorOp::ReduceProd { axes: vec![1] }.n_inputs(), 1);
    assert_eq!(StdTensorOp::ReduceProd { axes: vec![1] }.n_outputs(), 1);
    assert_eq!(StdTensorOp::ReduceMax { axes: vec![0] }.n_inputs(), 1);
    assert_eq!(StdTensorOp::ReduceMax { axes: vec![0] }.n_outputs(), 1);
    assert_eq!(StdTensorOp::ReduceMin { axes: vec![0, 1] }.n_inputs(), 1);
    assert_eq!(StdTensorOp::ReduceMin { axes: vec![0, 1] }.n_outputs(), 1);
}

#[test]
fn test_std_tensor_op_linearize_add_delegates_to_ad_module() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let dx = builder.add_input(TensorInputKey::User { id: 1 });
    let dy = builder.add_input(TensorInputKey::User { id: 2 });

    let result =
        StdTensorOp::add().linearize(&mut builder, &[], &[], &[Some(dx), Some(dy)], &mut ad_ctx);

    assert_eq!(result.len(), 1);
    assert!(result[0].is_some());
    let fragment = builder.build();
    assert_eq!(fragment.ops().len(), 1);
    assert_eq!(fragment.ops()[0].op, StdTensorOp::Add);
    assert_eq!(
        fragment.ops()[0].mode,
        OpMode::Linear {
            active_mask: vec![true, true],
        }
    );
}

#[test]
fn test_std_tensor_op_transpose_rule_add_fans_out_cotangent() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let ct = builder.add_input(TensorInputKey::User { id: 3 });
    let inputs = vec![
        ValRef::External(GlobalValKey::Input(TensorInputKey::User { id: 10 })),
        ValRef::External(GlobalValKey::Input(TensorInputKey::User { id: 11 })),
    ];

    let result = StdTensorOp::add().transpose_rule(
        &mut builder,
        &[Some(ct)],
        &inputs,
        &OpMode::Primal,
        &mut ad_ctx,
    );

    assert_eq!(result, vec![Some(ct), Some(ct)]);
    let fragment = builder.build();
    assert!(fragment.ops().is_empty());
}

#[test]
fn test_std_tensor_op_mul_transpose_skips_real_conjugates_and_keeps_complex_conjugates() {
    let (real_result, _, real_fragment) = run_transpose_case_with_input_shape(
        StdTensorOp::Mul,
        2,
        &[true, true],
        true,
        Some(sym_shape(&[2])),
    );

    assert!(real_result.iter().all(Option::is_some));
    assert!(
        real_fragment
            .ops()
            .iter()
            .all(|op| op.op != StdTensorOp::Conj),
        "real mul transpose should not emit no-op Conj nodes"
    );

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input_key(401));
    let inputs = external_inputs(510, 2);
    seed_uniform_ref_metadata_with_dtype(&mut ad_ctx, &inputs, DType::C64, sym_shape(&[2]));
    let complex_result = StdTensorOp::Mul.transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &inputs,
        &linear_mode(&[true, true]),
        &mut ad_ctx,
    );
    let complex_fragment = builder.build();

    assert!(complex_result.iter().all(Option::is_some));
    assert!(
        complex_fragment
            .ops()
            .iter()
            .any(|op| op.op == StdTensorOp::Conj),
        "complex mul transpose must still conjugate inactive primal factors"
    );
}

#[test]
fn test_std_tensor_op_conj_ad_skips_real_identity_and_keeps_complex_conjugation() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let primal_in = add_input_keys(&mut builder, 530, 1);
    ad_ctx.insert_metadata(
        primal_in[0].clone(),
        TensorMeta::exact(DType::F64, sym_shape(&[2])),
    );
    let tangent = builder.add_input(tensor_input_key(540));
    let real_linearized =
        StdTensorOp::Conj.linearize(&mut builder, &primal_in, &[], &[Some(tangent)], &mut ad_ctx);
    let real_linear_fragment = builder.build();
    assert_eq!(real_linearized, vec![Some(tangent)]);
    assert!(real_linear_fragment.ops().is_empty());

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let input = ValRef::External(GlobalValKey::Input(tensor_input_key(541)));
    seed_uniform_ref_metadata_with_dtype(
        &mut ad_ctx,
        std::slice::from_ref(&input),
        DType::F64,
        sym_shape(&[2]),
    );
    let cotangent = builder.add_input(tensor_input_key(542));
    let real_transposed = StdTensorOp::Conj.transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &[input],
        &linear_mode(&[true]),
        &mut ad_ctx,
    );
    let real_transpose_fragment = builder.build();
    assert_eq!(real_transposed, vec![Some(cotangent)]);
    assert!(real_transpose_fragment.ops().is_empty());

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let primal_in = add_input_keys(&mut builder, 550, 1);
    ad_ctx.insert_metadata(
        primal_in[0].clone(),
        TensorMeta::exact(DType::C64, sym_shape(&[2])),
    );
    let tangent = builder.add_input(tensor_input_key(560));
    let complex_linearized =
        StdTensorOp::Conj.linearize(&mut builder, &primal_in, &[], &[Some(tangent)], &mut ad_ctx);
    let complex_linear_fragment = builder.build();
    assert!(complex_linearized[0].is_some());
    assert_eq!(complex_linear_fragment.ops().len(), 1);
    assert_eq!(complex_linear_fragment.ops()[0].op, StdTensorOp::Conj);

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let input = ValRef::External(GlobalValKey::Input(tensor_input_key(561)));
    seed_uniform_ref_metadata_with_dtype(
        &mut ad_ctx,
        std::slice::from_ref(&input),
        DType::C64,
        sym_shape(&[2]),
    );
    let cotangent = builder.add_input(tensor_input_key(562));
    let complex_transposed = StdTensorOp::Conj.transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &[input],
        &linear_mode(&[true]),
        &mut ad_ctx,
    );
    let complex_transpose_fragment = builder.build();
    assert!(complex_transposed[0].is_some());
    assert_eq!(complex_transpose_fragment.ops().len(), 1);
    assert_eq!(complex_transpose_fragment.ops()[0].op, StdTensorOp::Conj);
}

#[test]
fn test_std_tensor_op_hash_covers_remaining_variants() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let variants = [
        StdTensorOp::Compare(CompareDir::Ge),
        StdTensorOp::Tril { k: -2 },
        StdTensorOp::Triu { k: 3 },
        StdTensorOp::Gather(GatherConfig {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: 1,
            slice_sizes: vec![1],
        }),
        StdTensorOp::Scatter(ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        }),
        StdTensorOp::Slice(tenferro_tensor::SliceConfig {
            starts: vec![0],
            limits: vec![1],
            strides: vec![1],
        }),
        StdTensorOp::DynamicSlice {
            slice_sizes: vec![1],
        },
        StdTensorOp::DynamicUpdateSlice,
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![0],
            edge_padding_high: vec![0],
            interior_padding: vec![0],
        }),
        StdTensorOp::Convert {
            from: DType::F64,
            to: DType::C64,
        },
        StdTensorOp::Concatenate {
            axis: 1,
            n_inputs: 3,
        },
        StdTensorOp::Reverse { axes: vec![0] },
        StdTensorOp::ShapeOf { axis: 0 },
        StdTensorOp::DynamicTruncate { axis: 0 },
        StdTensorOp::PadToMatch { axis: 0 },
        StdTensorOp::ReduceProd { axes: vec![0] },
        StdTensorOp::ReduceMax { axes: vec![0] },
        StdTensorOp::ReduceMin { axes: vec![0] },
    ];

    for op in variants {
        let mut hasher = DefaultHasher::new();
        op.hash(&mut hasher);
        assert_ne!(hasher.finish(), 0, "unexpected zero hash for {op:?}");
    }

    let mut lhs = DefaultHasher::new();
    StdTensorOp::constant_f64(1.25).hash(&mut lhs);
    let mut rhs = DefaultHasher::new();
    StdTensorOp::constant_f64(1.25).hash(&mut rhs);
    assert_eq!(lhs.finish(), rhs.finish());
}

#[test]
fn test_std_tensor_op_pad_to_match_transpose_uses_static_slice_for_concrete_shape() {
    let (result, _, fragment) = run_transpose_case_with_input_shapes(
        StdTensorOp::PadToMatch { axis: 0 },
        2,
        &[true, false],
        true,
        &[&[2], &[3]],
    );

    assert!(result[0].is_some());
    assert_eq!(result[1], None);
    assert_eq!(fragment.ops().len(), 1);
    assert_eq!(
        fragment.ops()[0].op,
        StdTensorOp::Slice(tenferro_tensor::SliceConfig {
            starts: vec![0],
            limits: vec![2],
            strides: vec![1],
        })
    );
}

#[test]
fn test_std_tensor_op_dynamic_truncate_linearize_uses_static_slice_for_narrowed_metadata() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let primal_in = add_input_keys(&mut builder, 960, 2);
    let primal_out = add_input_keys(&mut builder, 970, 1);
    let tangent = builder.add_input(tensor_input_key(980));

    ad_ctx.insert_metadata(
        primal_in[0].clone(),
        TensorMeta::exact(DType::F64, vec![SymDim::from(3usize)]),
    );
    ad_ctx.insert_metadata(
        primal_out[0].clone(),
        TensorMeta::exact(DType::F64, vec![SymDim::from(2usize)]),
    );

    let result = StdTensorOp::DynamicTruncate { axis: 0 }.linearize(
        &mut builder,
        &primal_in,
        &primal_out,
        &[Some(tangent), None],
        &mut ad_ctx,
    );
    let fragment = builder.build();

    assert!(result[0].is_some());
    assert_eq!(fragment.ops().len(), 1);
    assert_eq!(
        fragment.ops()[0].op,
        StdTensorOp::Slice(tenferro_tensor::SliceConfig {
            starts: vec![0],
            limits: vec![2],
            strides: vec![1],
        })
    );
}

#[test]
fn test_std_tensor_op_constant_constructors_encode_expected_bytes() {
    assert_eq!(
        StdTensorOp::constant_f64(1.25),
        StdTensorOp::Constant {
            dtype: DType::F64,
            bytes: 1.25_f64.to_le_bytes().to_vec(),
        }
    );
    assert_eq!(
        StdTensorOp::constant_f32(1.25),
        StdTensorOp::Constant {
            dtype: DType::F32,
            bytes: 1.25_f32.to_le_bytes().to_vec(),
        }
    );

    let c64 = Complex64::new(1.0, -2.0);
    let mut c64_bytes = Vec::new();
    c64_bytes.extend_from_slice(&c64.re.to_le_bytes());
    c64_bytes.extend_from_slice(&c64.im.to_le_bytes());
    assert_eq!(
        StdTensorOp::constant_c64(c64),
        StdTensorOp::Constant {
            dtype: DType::C64,
            bytes: c64_bytes,
        }
    );

    let c32 = Complex32::new(1.0, -2.0);
    let mut c32_bytes = Vec::new();
    c32_bytes.extend_from_slice(&c32.re.to_le_bytes());
    c32_bytes.extend_from_slice(&c32.im.to_le_bytes());
    assert_eq!(
        StdTensorOp::constant_c32(c32),
        StdTensorOp::Constant {
            dtype: DType::C32,
            bytes: c32_bytes,
        }
    );
}

#[test]
fn test_std_tensor_op_linearize_none_tangent_paths_return_none() {
    let unary_primal_in = [
        StdTensorOp::Sin,
        StdTensorOp::Cos,
        StdTensorOp::Log1p,
        StdTensorOp::Abs,
    ];
    for op in unary_primal_in {
        let (result, fragment) = run_linearize_case(op.clone(), 1, 0, &[false]);
        assert_eq!(result, vec![None], "unexpected result for {op:?}");
        assert!(fragment.ops().is_empty(), "expected no ops for {op:?}");
    }

    let unary_primal_out = [StdTensorOp::Tanh, StdTensorOp::Sqrt, StdTensorOp::Expm1];
    for op in unary_primal_out {
        let (result, fragment) = run_linearize_case(op.clone(), 0, 1, &[false]);
        assert_eq!(result, vec![None], "unexpected result for {op:?}");
        assert!(fragment.ops().is_empty(), "expected no ops for {op:?}");
    }

    let (result, fragment) = run_linearize_case(StdTensorOp::Rsqrt, 1, 1, &[false]);
    assert_eq!(result, vec![None]);
    assert!(fragment.ops().is_empty());

    let (result, fragment) = run_linearize_case(StdTensorOp::Pow, 2, 1, &[false, false]);
    assert_eq!(result, vec![None]);
    assert!(fragment.ops().is_empty());

    let (constant_result, constant_fragment) =
        run_linearize_case(StdTensorOp::constant_f64(1.0), 0, 0, &[]);
    assert_eq!(constant_result, vec![None]);
    assert!(constant_fragment.ops().is_empty());
}

#[test]
fn test_std_tensor_op_analytic_linearize_emits_ops_for_remaining_variants() {
    let (sin_result, sin_fragment) = run_linearize_case(StdTensorOp::Sin, 1, 0, &[true]);
    assert!(sin_result[0].is_some());
    assert_eq!(sin_fragment.ops().len(), 2);
    assert_eq!(sin_fragment.ops()[0].op, StdTensorOp::Cos);
    assert_eq!(sin_fragment.ops()[1].op, StdTensorOp::Mul);

    let (cos_result, cos_fragment) = run_linearize_case(StdTensorOp::Cos, 1, 0, &[true]);
    assert!(cos_result[0].is_some());
    assert_eq!(cos_fragment.ops().len(), 3);
    assert_eq!(cos_fragment.ops()[0].op, StdTensorOp::Sin);
    assert_eq!(cos_fragment.ops()[1].op, StdTensorOp::Neg);
    assert_eq!(cos_fragment.ops()[2].op, StdTensorOp::Mul);

    let (tanh_result, tanh_fragment) = run_linearize_case(StdTensorOp::Tanh, 0, 1, &[true]);
    assert!(tanh_result[0].is_some());
    assert_eq!(tanh_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (sqrt_result, sqrt_fragment) = run_linearize_case(StdTensorOp::Sqrt, 0, 1, &[true]);
    assert!(sqrt_result[0].is_some());
    assert_eq!(sqrt_fragment.ops().last().unwrap().op, StdTensorOp::Div);

    let (rsqrt_result, rsqrt_fragment) = run_linearize_case(StdTensorOp::Rsqrt, 1, 1, &[true]);
    assert!(rsqrt_result[0].is_some());
    assert_eq!(rsqrt_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (pow_result, pow_fragment) = run_linearize_case(StdTensorOp::Pow, 2, 1, &[true, true]);
    assert!(pow_result[0].is_some());
    assert_eq!(pow_fragment.ops().last().unwrap().op, StdTensorOp::Add);

    let (expm1_result, expm1_fragment) = run_linearize_case(StdTensorOp::Expm1, 0, 1, &[true]);
    assert!(expm1_result[0].is_some());
    assert_eq!(expm1_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (log1p_result, log1p_fragment) = run_linearize_case(StdTensorOp::Log1p, 1, 0, &[true]);
    assert!(log1p_result[0].is_some());
    assert_eq!(log1p_fragment.ops().last().unwrap().op, StdTensorOp::Div);
}

#[test]
fn test_std_tensor_op_elementwise_special_cases_are_covered() {
    let (div_sum_result, div_sum_fragment) =
        run_linearize_case(StdTensorOp::Div, 2, 1, &[true, true]);
    assert!(div_sum_result[0].is_some());
    assert_eq!(div_sum_fragment.ops().last().unwrap().op, StdTensorOp::Add);

    let (div_result, div_fragment) = run_linearize_case(StdTensorOp::Div, 2, 1, &[false, true]);
    assert!(div_result[0].is_some());
    assert_eq!(div_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (abs_result, abs_fragment) = run_linearize_case(StdTensorOp::Abs, 1, 0, &[true]);
    assert!(abs_result[0].is_some());
    assert_eq!(abs_fragment.ops()[0].op, StdTensorOp::Sign);
    assert_eq!(abs_fragment.ops()[1].op, StdTensorOp::Mul);

    let (sign_result, sign_fragment) = run_linearize_case(StdTensorOp::Sign, 0, 0, &[true]);
    assert!(sign_result[0].is_some());
    assert_eq!(sign_fragment.ops().len(), 2);
    assert_eq!(sign_fragment.ops()[0].op, StdTensorOp::Neg);
    assert_eq!(sign_fragment.ops()[1].op, StdTensorOp::Add);

    let (transpose_div_result, _, transpose_div_fragment) =
        run_transpose_case(StdTensorOp::Div, 2, &[false, true], true);
    assert_eq!(transpose_div_result[0], None);
    assert!(transpose_div_result[1].is_some());
    assert_eq!(
        transpose_div_fragment.ops().last().unwrap().op,
        StdTensorOp::Mul
    );

    let (transpose_sign_result, _, transpose_sign_fragment) =
        run_transpose_case(StdTensorOp::Sign, 1, &[true], true);
    assert!(transpose_sign_result[0].is_some());
    assert_eq!(transpose_sign_fragment.ops().len(), 2);
    assert_eq!(transpose_sign_fragment.ops()[0].op, StdTensorOp::Neg);
    assert_eq!(transpose_sign_fragment.ops()[1].op, StdTensorOp::Add);

    let (transpose_div_none_result, _, transpose_div_none_fragment) =
        run_transpose_case(StdTensorOp::Div, 2, &[true, true], false);
    assert_eq!(transpose_div_none_result, vec![None, None]);
    assert!(transpose_div_none_fragment.ops().is_empty());

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input_key(920));
    let div_primal_mode = StdTensorOp::Div.transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &external_inputs(921, 2),
        &OpMode::Primal,
        &mut ad_ctx,
    );
    assert_eq!(div_primal_mode, vec![None, None]);
    assert!(builder.build().ops().is_empty());

    let (transpose_abs_result, _, transpose_abs_fragment) =
        run_transpose_case(StdTensorOp::Abs, 1, &[true], true);
    assert!(transpose_abs_result[0].is_some());
    assert_eq!(transpose_abs_fragment.ops()[0].op, StdTensorOp::Sign);
    assert_eq!(transpose_abs_fragment.ops()[1].op, StdTensorOp::Mul);
}

#[test]
fn test_std_tensor_op_constant_transpose_rule_has_no_inputs_or_ops() {
    let (result, _, fragment) = run_transpose_case(StdTensorOp::constant_f64(1.0), 0, &[], true);
    assert!(result.is_empty());
    assert!(fragment.ops().is_empty());
}

#[test]
fn test_std_tensor_op_analytic_transpose_rule_emits_ops_for_remaining_variants() {
    let (exp_result, _, exp_fragment) = run_transpose_case(StdTensorOp::Exp, 1, &[true], true);
    assert!(exp_result[0].is_some());
    assert_eq!(exp_fragment.ops()[0].op, StdTensorOp::Exp);
    assert_eq!(exp_fragment.ops()[1].op, StdTensorOp::Mul);

    let (log_result, _, log_fragment) = run_transpose_case(StdTensorOp::Log, 1, &[true], true);
    assert!(log_result[0].is_some());
    assert_eq!(log_fragment.ops()[0].op, StdTensorOp::Div);

    let (sin_result, _, sin_fragment) = run_transpose_case(StdTensorOp::Sin, 1, &[true], true);
    assert!(sin_result[0].is_some());
    assert_eq!(sin_fragment.ops().len(), 2);
    assert_eq!(sin_fragment.ops()[0].op, StdTensorOp::Cos);
    assert_eq!(sin_fragment.ops()[1].op, StdTensorOp::Mul);

    let (cos_result, _, cos_fragment) = run_transpose_case(StdTensorOp::Cos, 1, &[true], true);
    assert!(cos_result[0].is_some());
    assert_eq!(cos_fragment.ops().len(), 3);
    assert_eq!(cos_fragment.ops()[0].op, StdTensorOp::Sin);
    assert_eq!(cos_fragment.ops()[1].op, StdTensorOp::Neg);
    assert_eq!(cos_fragment.ops()[2].op, StdTensorOp::Mul);

    let (tanh_result, _, tanh_fragment) = run_transpose_case(StdTensorOp::Tanh, 1, &[true], true);
    assert!(tanh_result[0].is_some());
    assert_eq!(tanh_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (sqrt_result, _, sqrt_fragment) = run_transpose_case(StdTensorOp::Sqrt, 1, &[true], true);
    assert!(sqrt_result[0].is_some());
    assert_eq!(sqrt_fragment.ops().last().unwrap().op, StdTensorOp::Div);

    let (rsqrt_result, _, rsqrt_fragment) =
        run_transpose_case(StdTensorOp::Rsqrt, 1, &[true], true);
    assert!(rsqrt_result[0].is_some());
    assert_eq!(rsqrt_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (pow_result, _, pow_fragment) =
        run_transpose_case(StdTensorOp::Pow, 2, &[true, true], true);
    assert!(pow_result[0].is_some());
    assert!(pow_result[1].is_some());
    assert_eq!(pow_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (expm1_result, _, expm1_fragment) =
        run_transpose_case(StdTensorOp::Expm1, 1, &[true], true);
    assert!(expm1_result[0].is_some());
    assert_eq!(expm1_fragment.ops().last().unwrap().op, StdTensorOp::Mul);

    let (log1p_result, _, log1p_fragment) =
        run_transpose_case(StdTensorOp::Log1p, 1, &[true], true);
    assert!(log1p_result[0].is_some());
    assert_eq!(log1p_fragment.ops().last().unwrap().op, StdTensorOp::Div);
}

#[test]
fn test_std_tensor_op_transpose_none_or_inactive_paths_return_none() {
    let unary_ops = [
        StdTensorOp::Sin,
        StdTensorOp::Cos,
        StdTensorOp::Tanh,
        StdTensorOp::Sqrt,
        StdTensorOp::Rsqrt,
        StdTensorOp::Expm1,
        StdTensorOp::Log1p,
        StdTensorOp::Abs,
        StdTensorOp::Sign,
    ];

    for op in unary_ops {
        let (none_result, _, none_fragment) = run_transpose_case(op.clone(), 1, &[true], false);
        assert_eq!(none_result, vec![None], "unexpected None path for {op:?}");
        assert!(none_fragment.ops().is_empty(), "expected no ops for {op:?}");

        let mut builder = FragmentBuilder::<StdTensorOp>::new();
        let mut ad_ctx = ShapeGuardContext::default();
        let cotangent = builder.add_input(tensor_input_key(900));
        let result = op.transpose_rule(
            &mut builder,
            &[Some(cotangent)],
            &external_inputs(901, 1),
            &OpMode::Primal,
            &mut ad_ctx,
        );
        assert_eq!(result, vec![None], "unexpected inactive path for {op:?}");
        assert!(
            builder.build().ops().is_empty(),
            "expected no ops for {op:?}"
        );
    }
}

#[test]
fn test_std_tensor_op_structural_special_cases_cover_identity_and_empty_axes() {
    let (transpose_none_result, transpose_none_fragment) =
        run_linearize_case(StdTensorOp::Transpose { perm: vec![1, 0] }, 0, 0, &[false]);
    assert_eq!(transpose_none_result, vec![None]);
    assert!(transpose_none_fragment.ops().is_empty());

    let (identity_transpose_result, identity_transpose_fragment) =
        run_linearize_case(StdTensorOp::Transpose { perm: vec![0, 1] }, 0, 0, &[true]);
    assert!(identity_transpose_result[0].is_some());
    assert!(identity_transpose_fragment.ops().is_empty());

    let reshape = StdTensorOp::Reshape {
        to_shape: shape![2, 2],
    };
    let (reshape_linear_result, reshape_linear_fragment) =
        run_linearize_case(reshape.clone(), 0, 0, &[true]);
    assert!(reshape_linear_result[0].is_some());
    assert_eq!(reshape_linear_fragment.ops().len(), 1);
    assert_eq!(reshape_linear_fragment.ops()[0].op, reshape);

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let primal_in = add_input_keys(&mut builder, 904, 1);
    ad_ctx.insert_metadata(
        primal_in[0].clone(),
        TensorMeta::exact(DType::F64, sym_shape(&[2, 2])),
    );
    let tangent = builder.add_input(tensor_input_key(905));
    let identity_reshape = StdTensorOp::Reshape {
        to_shape: shape![2, 2],
    };
    let identity_reshape_result =
        identity_reshape.linearize(&mut builder, &primal_in, &[], &[Some(tangent)], &mut ad_ctx);
    let identity_reshape_fragment = builder.build();
    assert_eq!(identity_reshape_result, vec![Some(tangent)]);
    assert!(identity_reshape_fragment.ops().is_empty());

    let (transpose_result, _, transpose_fragment) = run_transpose_case(
        StdTensorOp::Transpose { perm: vec![0, 1] },
        1,
        &[true],
        true,
    );
    assert!(transpose_result[0].is_some());
    assert!(transpose_fragment.ops().is_empty());

    let (broadcast_linear_result, broadcast_linear_fragment) = run_linearize_case(
        StdTensorOp::BroadcastInDim {
            shape: shape![2, 2, 3],
            dims: vec![0, 2],
        },
        0,
        0,
        &[true],
    );
    assert!(broadcast_linear_result[0].is_some());
    assert_eq!(
        broadcast_linear_fragment.ops()[0].op,
        StdTensorOp::BroadcastInDim {
            shape: shape![2, 2, 3],
            dims: vec![0, 2],
        }
    );

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let primal_in = add_input_keys(&mut builder, 906, 1);
    ad_ctx.insert_metadata(
        primal_in[0].clone(),
        TensorMeta::exact(DType::F64, sym_shape(&[2, 3])),
    );
    let tangent = builder.add_input(tensor_input_key(907));
    let identity_broadcast = StdTensorOp::BroadcastInDim {
        shape: shape![2, 3],
        dims: vec![0, 1],
    };
    let identity_broadcast_result =
        identity_broadcast.linearize(&mut builder, &primal_in, &[], &[Some(tangent)], &mut ad_ctx);
    let identity_broadcast_fragment = builder.build();
    assert_eq!(identity_broadcast_result, vec![Some(tangent)]);
    assert!(identity_broadcast_fragment.ops().is_empty());

    let (broadcast_none_result, broadcast_none_fragment) = run_linearize_case(
        StdTensorOp::BroadcastInDim {
            shape: shape![2, 2, 3],
            dims: vec![0, 2],
        },
        0,
        0,
        &[false],
    );
    assert_eq!(broadcast_none_result, vec![None]);
    assert!(broadcast_none_fragment.ops().is_empty());

    let (reshape_transpose_result, _, reshape_transpose_fragment) =
        run_transpose_case_with_input_shape(
            reshape,
            1,
            &[true],
            true,
            Some(vec![SymDim::from(4usize)]),
        );
    assert!(reshape_transpose_result[0].is_some());
    assert_eq!(reshape_transpose_fragment.ops().len(), 1);
    assert_eq!(
        reshape_transpose_fragment.ops()[0].op,
        StdTensorOp::Reshape {
            to_shape: DimExpr::input_shape(1, 1),
        }
    );

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input_key(910));
    let result = StdTensorOp::BroadcastInDim {
        shape: shape![2, 3],
        dims: vec![0, 1],
    }
    .transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &external_inputs(911, 1),
        &linear_mode(&[true]),
        &mut ad_ctx,
    );
    assert_eq!(result, vec![Some(cotangent)]);
    assert!(builder.build().ops().is_empty());

    let (tril_result, tril_fragment) =
        run_linearize_case(StdTensorOp::Tril { k: -1 }, 0, 0, &[true]);
    assert!(tril_result[0].is_some());
    assert_eq!(tril_fragment.ops()[0].op, StdTensorOp::Tril { k: -1 });

    let (triu_result, triu_fragment) =
        run_linearize_case(StdTensorOp::Triu { k: 2 }, 0, 0, &[true]);
    assert!(triu_result[0].is_some());
    assert_eq!(triu_fragment.ops()[0].op, StdTensorOp::Triu { k: 2 });

    let slice = StdTensorOp::Slice(tenferro_tensor::SliceConfig {
        starts: vec![1],
        limits: vec![5],
        strides: vec![2],
    });
    let (slice_result, slice_fragment) = run_linearize_case(slice.clone(), 0, 0, &[true]);
    assert!(slice_result[0].is_some());
    assert_eq!(slice_fragment.ops()[0].op, slice.clone());

    let (transpose_slice_result, _, transpose_slice_fragment) =
        run_transpose_case_with_input_shapes(slice, 1, &[true], true, &[&[5]]);
    assert!(transpose_slice_result[0].is_some());
    assert_eq!(transpose_slice_fragment.ops().len(), 1);
    assert_eq!(
        transpose_slice_fragment.ops()[0].op,
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![1],
            interior_padding: vec![1],
        })
    );

    let pad = StdTensorOp::Pad(PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![2],
        interior_padding: vec![1],
    });
    let (transpose_pad_result, _, transpose_pad_fragment) =
        run_transpose_case_with_input_shapes(pad, 1, &[true], true, &[&[3]]);
    assert!(transpose_pad_result[0].is_some());
    assert_eq!(transpose_pad_fragment.ops().len(), 1);
    assert_eq!(
        transpose_pad_fragment.ops()[0].op,
        StdTensorOp::Slice(tenferro_tensor::SliceConfig {
            starts: vec![1],
            limits: vec![6],
            strides: vec![2],
        })
    );

    let cropped_pad = StdTensorOp::Pad(PadConfig {
        edge_padding_low: vec![-1],
        edge_padding_high: vec![0],
        interior_padding: vec![0],
    });
    let (cropped_pad_result, _, cropped_pad_fragment) =
        run_transpose_case_with_input_shapes(cropped_pad, 1, &[true], true, &[&[4]]);
    assert!(cropped_pad_result[0].is_some());
    assert_eq!(cropped_pad_fragment.ops().len(), 2);
    assert_eq!(
        cropped_pad_fragment.ops()[0].op,
        StdTensorOp::Slice(tenferro_tensor::SliceConfig {
            starts: vec![0],
            limits: vec![3],
            strides: vec![1],
        })
    );
    assert_eq!(
        cropped_pad_fragment.ops()[1].op,
        StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![0],
            interior_padding: vec![0],
        })
    );

    let reverse = StdTensorOp::Reverse { axes: vec![0, 2] };
    let (reverse_result, reverse_fragment) = run_linearize_case(reverse.clone(), 0, 0, &[true]);
    assert!(reverse_result[0].is_some());
    assert_eq!(reverse_fragment.ops()[0].op, reverse.clone());

    let (transpose_reverse_result, _, transpose_reverse_fragment) =
        run_transpose_case(reverse.clone(), 1, &[true], true);
    assert!(transpose_reverse_result[0].is_some());
    assert_eq!(transpose_reverse_fragment.ops()[0].op, reverse);

    let (transpose_tril_result, _, transpose_tril_fragment) =
        run_transpose_case(StdTensorOp::Tril { k: 0 }, 1, &[true], true);
    assert!(transpose_tril_result[0].is_some());
    assert_eq!(
        transpose_tril_fragment.ops()[0].op,
        StdTensorOp::Tril { k: 0 }
    );

    let (transpose_triu_result, _, transpose_triu_fragment) =
        run_transpose_case(StdTensorOp::Triu { k: 1 }, 1, &[true], true);
    assert!(transpose_triu_result[0].is_some());
    assert_eq!(
        transpose_triu_fragment.ops()[0].op,
        StdTensorOp::Triu { k: 1 }
    );

    let (transpose_transpose_none_result, _, transpose_transpose_none_fragment) =
        run_transpose_case(
            StdTensorOp::Transpose {
                perm: vec![2, 0, 1],
            },
            1,
            &[true],
            false,
        );
    assert_eq!(transpose_transpose_none_result, vec![None]);
    assert!(transpose_transpose_none_fragment.ops().is_empty());

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let none_broadcast = StdTensorOp::BroadcastInDim {
        shape: shape![2, 2, 3],
        dims: vec![0, 2],
    }
    .transpose_rule(
        &mut builder,
        &[None],
        &external_inputs(930, 1),
        &linear_mode(&[true]),
        &mut ad_ctx,
    );
    assert_eq!(none_broadcast, vec![None]);
    assert!(builder.build().ops().is_empty());
}

#[test]
fn test_std_tensor_op_convert_linearize_and_transpose_swap_dtypes() {
    let convert = StdTensorOp::Convert {
        from: DType::F64,
        to: DType::C64,
    };

    let (linear_result, linear_fragment) = run_linearize_case(convert.clone(), 0, 0, &[true]);
    assert!(linear_result[0].is_some());
    assert_eq!(linear_fragment.ops().len(), 1);
    assert_eq!(linear_fragment.ops()[0].op, convert);

    let (linear_none_result, linear_none_fragment) =
        run_linearize_case(convert.clone(), 0, 0, &[false]);
    assert_eq!(linear_none_result, vec![None]);
    assert!(linear_none_fragment.ops().is_empty());

    let (transpose_result, _, transpose_fragment) =
        run_transpose_case(convert.clone(), 1, &[true], true);
    assert!(transpose_result[0].is_some());
    assert_eq!(transpose_fragment.ops().len(), 1);
    assert_eq!(
        transpose_fragment.ops()[0].op,
        StdTensorOp::Convert {
            from: DType::C64,
            to: DType::F64,
        }
    );

    let (transpose_inactive_result, _, transpose_inactive_fragment) =
        run_transpose_case(convert, 1, &[false], true);
    assert_eq!(transpose_inactive_result, vec![None]);
    assert!(transpose_inactive_fragment.ops().is_empty());
}

#[test]
fn test_std_tensor_op_contraction_special_cases_cover_none_and_scalar_paths() {
    let matmul = StdTensorOp::DotGeneral {
        config: DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    };
    let (linearize_none_result, linearize_none_fragment) =
        run_linearize_case(matmul.clone(), 2, 0, &[false, false]);
    assert_eq!(linearize_none_result, vec![None]);
    assert!(linearize_none_fragment.ops().is_empty());

    let (transpose_none_result, _, transpose_none_fragment) =
        run_transpose_case(matmul.clone(), 2, &[true, true], false);
    assert_eq!(transpose_none_result, vec![None, None]);
    assert!(transpose_none_fragment.ops().is_empty());

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input_key(980));
    seed_dot_general_ref_metadata(&mut ad_ctx, &external_inputs(981, 2));
    let primal_mode_result = matmul.transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &external_inputs(981, 2),
        &OpMode::Primal,
        &mut ad_ctx,
    );
    assert_eq!(primal_mode_result, vec![None, None]);
    assert!(builder.build().ops().is_empty());

    let reduce = StdTensorOp::ReduceSum { axes: vec![1] };
    let (reduce_linearize_none_result, reduce_linearize_none_fragment) =
        run_linearize_case(reduce.clone(), 0, 0, &[false]);
    assert_eq!(reduce_linearize_none_result, vec![None]);
    assert!(reduce_linearize_none_fragment.ops().is_empty());

    let (reduce_transpose_result, _, reduce_transpose_fragment) =
        run_transpose_case_with_input_shapes(reduce.clone(), 1, &[true], true, &[&[2, 3]]);
    assert!(reduce_transpose_result[0].is_some());
    assert_eq!(reduce_transpose_fragment.ops().len(), 1);
    // The transpose rule reads input shape via `ctx.shape_of` and emits
    // `BroadcastInDim` with `InputDim` references into the primal input
    // (which sits at op input index 1 after the cotangent at index 0).
    assert_eq!(
        reduce_transpose_fragment.ops()[0].op,
        StdTensorOp::BroadcastInDim {
            shape: vec![
                DimExpr::InputDim {
                    input_idx: 1,
                    axis: 0,
                },
                DimExpr::InputDim {
                    input_idx: 1,
                    axis: 1,
                },
            ],
            dims: vec![0],
        }
    );

    let (reduce_transpose_none_result, _, reduce_transpose_none_fragment) =
        run_transpose_case(reduce, 1, &[true], false);
    assert_eq!(reduce_transpose_none_result, vec![None]);
    assert!(reduce_transpose_none_fragment.ops().is_empty());

    let scalar_contract = StdTensorOp::DotGeneral {
        config: DotGeneralConfig {
            lhs_contracting_dims: vec![1, 0],
            rhs_contracting_dims: vec![0, 1],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
    };
    let (scalar_transpose_result, _, scalar_transpose_fragment) =
        run_transpose_case(scalar_contract.clone(), 2, &[true, false], true);
    assert!(scalar_transpose_result[0].is_some());
    assert_eq!(scalar_transpose_result[1], None);
    assert_eq!(
        scalar_transpose_fragment.ops()[0].op,
        StdTensorOp::Reshape { to_shape: shape![] }
    );
    assert!(scalar_transpose_fragment
        .ops()
        .iter()
        .all(|node| node.op != StdTensorOp::Conj));
    assert!(matches!(
        scalar_transpose_fragment.ops()[1].op,
        StdTensorOp::DotGeneral { .. }
    ));
    assert_eq!(
        scalar_transpose_fragment.ops().last().unwrap().op,
        StdTensorOp::Transpose { perm: vec![1, 0] }
    );

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let cotangent = builder.add_input(tensor_input_key(990));
    let inputs = external_inputs(991, 2);
    for (input, shape) in inputs.iter().zip([&[2, 3][..], &[3, 2][..]]) {
        let ValRef::External(key) = input else {
            unreachable!("external_inputs returns external refs")
        };
        ad_ctx.insert_metadata(
            key.clone(),
            TensorMeta::exact(
                DType::C64,
                shape.iter().copied().map(SymDim::from).collect(),
            ),
        );
    }
    let complex_transpose_result = scalar_contract.transpose_rule(
        &mut builder,
        &[Some(cotangent)],
        &inputs,
        &linear_mode(&[true, false]),
        &mut ad_ctx,
    );
    let complex_transpose_fragment = builder.build();
    assert!(complex_transpose_result[0].is_some());
    assert_eq!(complex_transpose_result[1], None);
    assert!(complex_transpose_fragment
        .ops()
        .iter()
        .any(|node| node.op == StdTensorOp::Conj));
}

#[derive(Clone, Debug)]
struct RuleOnlyExt {
    family: &'static str,
}

impl ExtensionOp for RuleOnlyExt {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<RuleOnlyExt>()
            .is_some_and(|rhs| rhs.family == self.family)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn n_outputs(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        vec![(input_dtypes[0], input_shapes[0].to_vec())]
    }

    fn eager_execute(
        &self,
        inputs: &[&tenferro_tensor::Tensor],
    ) -> tenferro_tensor::Result<Vec<tenferro_tensor::Tensor>> {
        Ok(vec![inputs[0].clone()])
    }
}

#[derive(Debug)]
struct RuleOnlyIdentityAd {
    family: &'static str,
}

impl ExtensionAdRule for RuleOnlyIdentityAd {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn linearize(
        &self,
        _op: &dyn ExtensionOp,
        _builder: &mut FragmentBuilder<StdTensorOp>,
        _primal_in: &[GlobalValKey<StdTensorOp>],
        _primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        Ok(vec![tangent_in[0]])
    }

    fn transpose_rule(
        &self,
        _op: &dyn ExtensionOp,
        _emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<StdTensorOp>],
        _mode: &OpMode,
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        Ok(vec![cotangent_out[0]])
    }
}

#[test]
fn extension_try_linearize_uses_registered_rule() {
    let family = "stdtensor.rule_only_identity.v1";
    let _ = register_extension_rule(Arc::new(RuleOnlyIdentityAd { family }));
    let op = StdTensorOp::Extension(Arc::new(RuleOnlyExt { family }));
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let dx = builder.add_input(tensor_input_key(900));
    let result = op
        .try_linearize(&mut builder, &[], &[], &[Some(dx)], &mut ad_ctx)
        .expect("registered extension rule should linearize");

    assert_eq!(result, vec![Some(dx)]);
}

#[test]
fn extension_try_transpose_uses_registered_rule() {
    let family = "stdtensor.rule_only_transpose.v1";
    let _ = register_extension_rule(Arc::new(RuleOnlyIdentityAd { family }));
    let op = StdTensorOp::Extension(Arc::new(RuleOnlyExt { family }));
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let ct = builder.add_input(tensor_input_key(901));
    let result = op
        .try_transpose_rule(
            &mut builder,
            &[Some(ct)],
            &external_inputs(910, 1),
            &linear_mode(&[true]),
            &mut ad_ctx,
        )
        .expect("registered extension rule should transpose");

    assert_eq!(result, vec![Some(ct)]);
}

#[test]
fn extension_try_linearize_reports_missing_rule() {
    let family = "stdtensor.missing_rule.v1";
    let op = StdTensorOp::Extension(Arc::new(RuleOnlyExt { family }));
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ad_ctx = ShapeGuardContext::default();
    let dx = builder.add_input(tensor_input_key(920));
    let err = op
        .try_linearize(&mut builder, &[], &[], &[Some(dx)], &mut ad_ctx)
        .expect_err("missing extension rule should be an AD error");

    assert_eq!(err.rule(), ADRuleKind::Linearize);
    assert!(err.to_string().contains(family));
}
