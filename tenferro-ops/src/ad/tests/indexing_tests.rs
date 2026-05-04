//! Unit tests for the `Gather` / `Scatter` AD rules in `ad::indexing`.

use chainrules_core::PrimitiveOp;
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use tenferro_tensor::{DType, GatherConfig, ScatterConfig};

use crate::ad::context::ShapeGuardContext;
use crate::dim_expr::DimExpr;
use crate::input_key::TensorInputKey;
use crate::std_tensor_op::StdTensorOp;
use crate::{SymDim, TensorMeta};

fn tensor_input(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

fn input_key(id: u64) -> GlobalValKey<StdTensorOp> {
    GlobalValKey::Input(tensor_input(id))
}

fn meta(shape: &[usize]) -> TensorMeta {
    TensorMeta::exact(DType::F64, shape.iter().copied().map(Into::into).collect())
}

fn seed_metadata(ctx: &mut ShapeGuardContext, entries: &[(GlobalValKey<StdTensorOp>, &[usize])]) {
    for (key, shape) in entries {
        ctx.insert_metadata(key.clone(), meta(shape));
    }
}

/// A 1-D gather that reads three scalar entries from a length-5 operand.
fn rank1_gather_config() -> GatherConfig {
    GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    }
}

/// The additive scatter config matching `rank1_gather_config` under the
/// Gather↔Scatter inversion.
fn rank1_scatter_config() -> ScatterConfig {
    ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    }
}

#[test]
fn linearize_gather_reuses_primal_indices_and_emits_gather() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(1);
    let indices_key = input_key(2);
    let operand_tangent = builder.add_input(tensor_input(3));

    let config = rank1_gather_config();
    let op = StdTensorOp::Gather(config.clone());
    let primal_in = vec![operand_key.clone(), indices_key.clone()];
    let tangent_in = [Some(operand_tangent), None];

    let result = op.linearize(&mut builder, &primal_in, &[], &tangent_in, &mut ctx);

    assert_eq!(result.len(), 1);
    let tangent_out = result[0].expect("output tangent must be active");
    let fragment = builder.build();

    assert_eq!(fragment.ops().len(), 1);
    let gather_op = &fragment.ops()[0];
    assert_eq!(gather_op.op, StdTensorOp::Gather(config));
    assert_eq!(
        gather_op.mode,
        OpMode::Linear {
            active_mask: vec![true, false],
        }
    );
    // The second operand of the linearised Gather must be the *primal* indices.
    assert_eq!(gather_op.inputs[1], ValRef::External(indices_key),);
    // The returned local id must be one of the emitted gather's outputs.
    assert!(gather_op.outputs.contains(&tangent_out));
}

#[test]
fn linearize_gather_inactive_tangent_returns_none() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(10);
    let indices_key = input_key(11);

    let op = StdTensorOp::Gather(rank1_gather_config());
    let primal_in = vec![operand_key, indices_key];
    let tangent_in: [Option<LocalValId>; 2] = [None, None];

    let result = op.linearize(&mut builder, &primal_in, &[], &tangent_in, &mut ctx);
    assert_eq!(result, vec![None]);
    assert!(builder.build().ops().is_empty());
}

#[test]
fn linearize_dynamic_gather_reuses_primal_indices_and_shape_sources() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(101);
    let indices_key = input_key(102);
    let shape_source_key = input_key(103);
    let operand_tangent = builder.add_input(tensor_input(104));

    let slice_sizes = vec![
        DimExpr::Const(1),
        DimExpr::InputDim {
            input_idx: 2,
            axis: 1,
        },
    ];
    let op = StdTensorOp::GatherDynamicSliceSizes {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: slice_sizes.clone(),
    };
    let primal_in = vec![operand_key, indices_key.clone(), shape_source_key.clone()];
    let tangent_in = [Some(operand_tangent), None, None];

    let result = op.linearize(&mut builder, &primal_in, &[], &tangent_in, &mut ctx);

    assert_eq!(result.len(), 1);
    let tangent_out = result[0].expect("output tangent must be active");
    let fragment = builder.build();

    assert_eq!(fragment.ops().len(), 1);
    let gather = &fragment.ops()[0];
    assert_eq!(gather.op, op);
    assert_eq!(gather.inputs[0], ValRef::Local(operand_tangent));
    assert_eq!(gather.inputs[1], ValRef::External(indices_key));
    assert_eq!(gather.inputs[2], ValRef::External(shape_source_key));
    assert_eq!(
        gather.mode,
        OpMode::Linear {
            active_mask: vec![true, false, false],
        }
    );
    assert!(gather.outputs.contains(&tangent_out));
}

#[test]
fn linearize_dynamic_slice_reuses_primal_starts() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(105);
    let starts_key = input_key(106);
    let operand_tangent = builder.add_input(tensor_input(107));

    let slice_sizes = vec![3];
    let op = StdTensorOp::DynamicSlice {
        slice_sizes: slice_sizes.clone(),
    };
    let primal_in = vec![operand_key, starts_key.clone()];
    let tangent_in = [Some(operand_tangent), None];

    let result = op.linearize(&mut builder, &primal_in, &[], &tangent_in, &mut ctx);

    assert_eq!(result.len(), 1);
    let tangent_out = result[0].expect("output tangent must be active");
    let fragment = builder.build();

    assert_eq!(fragment.ops().len(), 1);
    let dynamic_slice = &fragment.ops()[0];
    assert_eq!(dynamic_slice.op, StdTensorOp::DynamicSlice { slice_sizes });
    assert_eq!(dynamic_slice.inputs[0], ValRef::Local(operand_tangent));
    assert_eq!(dynamic_slice.inputs[1], ValRef::External(starts_key));
    assert_eq!(
        dynamic_slice.mode,
        OpMode::Linear {
            active_mask: vec![true, false],
        }
    );
    assert!(dynamic_slice.outputs.contains(&tangent_out));
}

#[test]
fn linearize_dynamic_slice_inactive_tangent_returns_none() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(108);
    let starts_key = input_key(109);

    let op = StdTensorOp::DynamicSlice {
        slice_sizes: vec![3],
    };
    let primal_in = vec![operand_key, starts_key];
    let tangent_in: [Option<LocalValId>; 2] = [None, None];

    let result = op.linearize(&mut builder, &primal_in, &[], &tangent_in, &mut ctx);
    assert_eq!(result, vec![None]);
    assert!(builder.build().ops().is_empty());
}

#[test]
fn linearize_dynamic_update_slice_reuses_primal_starts() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(110);
    let update_key = input_key(111);
    let starts_key = input_key(112);
    let operand_tangent = builder.add_input(tensor_input(113));
    let update_tangent = builder.add_input(tensor_input(114));

    let op = StdTensorOp::DynamicUpdateSlice;
    let primal_in = vec![operand_key, update_key, starts_key.clone()];
    let tangent_in = [Some(operand_tangent), Some(update_tangent), None];

    let result = op.linearize(&mut builder, &primal_in, &[], &tangent_in, &mut ctx);

    assert_eq!(result.len(), 1);
    let tangent_out = result[0].expect("output tangent must be active");
    let fragment = builder.build();

    assert_eq!(fragment.ops().len(), 1);
    let update_slice = &fragment.ops()[0];
    assert_eq!(update_slice.op, StdTensorOp::DynamicUpdateSlice);
    assert_eq!(update_slice.inputs[0], ValRef::Local(operand_tangent));
    assert_eq!(update_slice.inputs[1], ValRef::Local(update_tangent));
    assert_eq!(update_slice.inputs[2], ValRef::External(starts_key));
    assert_eq!(
        update_slice.mode,
        OpMode::Linear {
            active_mask: vec![true, true, false],
        }
    );
    assert!(update_slice.outputs.contains(&tangent_out));
}

#[test]
fn transpose_dynamic_slice_emits_dynamic_update_slice() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(115));
    let operand_key = input_key(116);
    let starts_key = input_key(117);
    seed_metadata(
        &mut ctx,
        &[(operand_key.clone(), &[5]), (starts_key.clone(), &[1])],
    );

    let inputs = vec![
        ValRef::External(operand_key.clone()),
        ValRef::External(starts_key.clone()),
    ];
    let result = (StdTensorOp::DynamicSlice {
        slice_sizes: vec![3],
    })
    .transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OpMode::Linear {
            active_mask: vec![true, false],
        },
        &mut ctx,
    );

    assert!(result[0].is_some(), "operand cotangent must be active");
    assert_eq!(result[1], None, "starts cotangent must stay None");

    let fragment = builder.build();
    let update_slice = fragment
        .ops()
        .iter()
        .find(|op_node| matches!(op_node.op, StdTensorOp::DynamicUpdateSlice))
        .expect("expected a DynamicUpdateSlice op");
    assert!(matches!(update_slice.inputs[0], ValRef::Local(_)));
    assert_ne!(update_slice.inputs[0], ValRef::External(operand_key));
    assert_eq!(update_slice.inputs[1], ValRef::Local(cot));
    assert_eq!(update_slice.inputs[2], ValRef::External(starts_key));
    assert_eq!(
        update_slice.mode,
        OpMode::Linear {
            active_mask: vec![false, true, false],
        }
    );
}

#[test]
fn transpose_dynamic_update_slice_returns_operand_and_update_cotangents() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(118));
    let operand_key = input_key(119);
    let update_key = input_key(120);
    let starts_key = input_key(121);
    seed_metadata(
        &mut ctx,
        &[
            (operand_key.clone(), &[5]),
            (update_key.clone(), &[3]),
            (starts_key.clone(), &[1]),
        ],
    );

    let inputs = vec![
        ValRef::External(operand_key),
        ValRef::External(update_key),
        ValRef::External(starts_key.clone()),
    ];
    let result = StdTensorOp::DynamicUpdateSlice.transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OpMode::Linear {
            active_mask: vec![true, true, false],
        },
        &mut ctx,
    );

    assert!(result[0].is_some(), "operand cotangent must be active");
    assert!(result[1].is_some(), "update cotangent must be active");
    assert_eq!(result[2], None, "starts cotangent must stay None");

    let fragment = builder.build();
    assert!(
        fragment
            .ops()
            .iter()
            .any(|op_node| matches!(op_node.op, StdTensorOp::DynamicUpdateSlice)),
        "operand cotangent should be masked by DynamicUpdateSlice"
    );
    let update_ct = fragment
        .ops()
        .iter()
        .find(|op_node| matches!(op_node.op, StdTensorOp::DynamicSlice { .. }))
        .expect("expected a DynamicSlice op for update cotangent");
    assert_eq!(
        update_ct.op,
        StdTensorOp::DynamicSlice {
            slice_sizes: vec![3],
        }
    );
    assert_eq!(update_ct.inputs[0], ValRef::Local(cot));
    assert_eq!(update_ct.inputs[1], ValRef::External(starts_key));
}

#[test]
fn transpose_gather_emits_scatter_with_inverted_config() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(20));
    let operand_key = input_key(21);
    let indices_key = input_key(22);
    seed_metadata(
        &mut ctx,
        &[(operand_key.clone(), &[5]), (indices_key.clone(), &[3, 1])],
    );

    let config = rank1_gather_config();
    let op = StdTensorOp::Gather(config.clone());
    let inputs = vec![
        ValRef::External(operand_key.clone()),
        ValRef::External(indices_key.clone()),
    ];

    let result = op.transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OpMode::Linear {
            active_mask: vec![true, false],
        },
        &mut ctx,
    );
    assert!(result[0].is_some(), "operand cotangent must be active");
    assert_eq!(result[1], None, "indices cotangent must stay None");

    let fragment = builder.build();
    // Under StableHLO add-scatter semantics the inverse scatter must use
    // a zero operand built from Constant + BroadcastInDim, so the fragment
    // contains [Constant, BroadcastInDim, Scatter].
    assert_eq!(fragment.ops().len(), 3);
    let scatter = fragment
        .ops()
        .iter()
        .find(|op_node| matches!(op_node.op, StdTensorOp::Scatter(_)))
        .expect("expected a Scatter op in the fragment");
    let StdTensorOp::Scatter(scatter_cfg) = &scatter.op else {
        panic!("expected Scatter op, got {:?}", scatter.op);
    };
    assert_eq!(scatter_cfg.update_window_dims, config.offset_dims);
    assert_eq!(
        scatter_cfg.inserted_window_dims,
        config.collapsed_slice_dims
    );
    assert_eq!(
        scatter_cfg.scatter_dims_to_operand_dims,
        config.start_index_map
    );
    assert_eq!(scatter_cfg.index_vector_dim, config.index_vector_dim);
    // The scatter's first input (operand) must be the broadcast zero
    // local, not the primal forward-gather operand.
    assert!(matches!(scatter.inputs[0], ValRef::Local(_)));
    assert_ne!(scatter.inputs[0], ValRef::External(operand_key));
    // Indices are reused from the primal; updates are the cotangent.
    assert_eq!(scatter.inputs[1], ValRef::External(indices_key));
    assert_eq!(scatter.inputs[2], ValRef::Local(cot));
    assert_eq!(
        scatter.mode,
        OpMode::Linear {
            active_mask: vec![false, false, true],
        }
    );
}

#[test]
fn transpose_gather_inactive_path_returns_all_none() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(30);
    let indices_key = input_key(31);
    let inputs = vec![ValRef::External(operand_key), ValRef::External(indices_key)];

    let op = StdTensorOp::Gather(rank1_gather_config());

    let result = op.transpose_rule(
        &mut builder,
        &[None],
        &inputs,
        &OpMode::Linear {
            active_mask: vec![true, false],
        },
        &mut ctx,
    );
    assert_eq!(result, vec![None, None]);
    assert!(builder.build().ops().is_empty());
}

#[test]
fn transpose_dynamic_gather_emits_scatter_and_ignores_shape_sources() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(120));
    let operand_key = input_key(121);
    let indices_key = input_key(122);
    let shape_source_key = input_key(123);
    seed_metadata(
        &mut ctx,
        &[
            (operand_key.clone(), &[4, 2]),
            (indices_key.clone(), &[1, 1]),
            (shape_source_key.clone(), &[1, 2]),
        ],
    );

    let op = StdTensorOp::GatherDynamicSliceSizes {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![
            DimExpr::Const(1),
            DimExpr::InputDim {
                input_idx: 2,
                axis: 1,
            },
        ],
    };
    let inputs = vec![
        ValRef::External(operand_key.clone()),
        ValRef::External(indices_key.clone()),
        ValRef::External(shape_source_key),
    ];

    let result = op.transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OpMode::Linear {
            active_mask: vec![true, false, false],
        },
        &mut ctx,
    );

    assert!(result[0].is_some(), "operand cotangent must be active");
    assert_eq!(result[1], None, "indices cotangent must stay None");
    assert_eq!(result[2], None, "shape source cotangent must stay None");

    let fragment = builder.build();
    let scatter = fragment
        .ops()
        .iter()
        .find(|op_node| matches!(op_node.op, StdTensorOp::Scatter(_)))
        .expect("expected a Scatter op in the fragment");
    let StdTensorOp::Scatter(scatter_cfg) = &scatter.op else {
        panic!("expected Scatter op, got {:?}", scatter.op);
    };
    assert_eq!(scatter_cfg.update_window_dims, vec![1]);
    assert_eq!(scatter_cfg.inserted_window_dims, vec![0]);
    assert_eq!(scatter_cfg.scatter_dims_to_operand_dims, vec![0]);
    assert_eq!(scatter_cfg.index_vector_dim, 1);
    assert!(matches!(scatter.inputs[0], ValRef::Local(_)));
    assert_ne!(scatter.inputs[0], ValRef::External(operand_key));
    assert_eq!(scatter.inputs[1], ValRef::External(indices_key));
    assert_eq!(scatter.inputs[2], ValRef::Local(cot));
}

#[test]
fn linearize_scatter_inactive_tangents_returns_none() {
    // Both operand_dot = None and updates_dot = None => [None] with no ops.
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(50);
    let indices_key = input_key(51);
    let updates_key = input_key(52);
    let op = StdTensorOp::Scatter(rank1_scatter_config());
    let primal_in = vec![operand_key, indices_key, updates_key];
    let tangent_in: [Option<LocalValId>; 3] = [None, None, None];
    let result = op.linearize(&mut builder, &primal_in, &[], &tangent_in, &mut ctx);
    assert_eq!(result, vec![None]);
    assert!(builder.build().ops().is_empty());
}

#[test]
fn linearize_scatter_operand_only_is_identity_passthrough() {
    // Only operand_dot is active: the output tangent is the operand
    // tangent itself, with no ops emitted.
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(40);
    let indices_key = input_key(41);
    let updates_key = input_key(42);
    let operand_tangent = builder.add_input(tensor_input(43));
    seed_metadata(
        &mut ctx,
        &[
            (operand_key.clone(), &[5]),
            (indices_key.clone(), &[3, 1]),
            (updates_key.clone(), &[3]),
        ],
    );

    let config = rank1_scatter_config();
    let op = StdTensorOp::Scatter(config);
    let primal_in = vec![operand_key, indices_key, updates_key];
    let tangent_in = [Some(operand_tangent), None, None];

    let result = op.linearize(&mut builder, &primal_in, &[], &tangent_in, &mut ctx);
    assert_eq!(
        result,
        vec![Some(operand_tangent)],
        "operand-only scatter tangent must be the operand tangent itself"
    );
    let fragment = builder.build();
    assert!(
        fragment.ops().is_empty(),
        "identity passthrough must not emit any ops"
    );
}

#[test]
fn linearize_scatter_updates_only_uses_zero_operand() {
    // Only updates_dot is active: emit Scatter(zeros_like(operand), ..., d_updates)
    // via Constant + BroadcastInDim + Scatter.
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(40);
    let indices_key = input_key(41);
    let updates_key = input_key(42);
    let updates_tangent = builder.add_input(tensor_input(43));
    seed_metadata(
        &mut ctx,
        &[
            (operand_key.clone(), &[5]),
            (indices_key.clone(), &[3, 1]),
            (updates_key.clone(), &[3]),
        ],
    );

    let config = rank1_scatter_config();
    let op = StdTensorOp::Scatter(config.clone());
    let primal_in = vec![
        operand_key.clone(),
        indices_key.clone(),
        updates_key.clone(),
    ];
    let tangent_in = [None, None, Some(updates_tangent)];

    let result = op.linearize(&mut builder, &primal_in, &[], &tangent_in, &mut ctx);
    assert_eq!(result.len(), 1);
    let _ = result[0].expect("output tangent must be active");

    let fragment = builder.build();
    // Fragment: [Constant, BroadcastInDim, Scatter]
    assert_eq!(
        fragment.ops().len(),
        3,
        "expected Constant + BroadcastInDim + Scatter in the fragment"
    );
    let scatter = fragment
        .ops()
        .iter()
        .find(|op_node| matches!(op_node.op, StdTensorOp::Scatter(_)))
        .expect("expected a Scatter op");
    assert_eq!(scatter.op, StdTensorOp::Scatter(config));
    assert_eq!(
        scatter.mode,
        OpMode::Linear {
            active_mask: vec![false, false, true],
        }
    );
    // The scatter's operand must be a freshly built zero local, not the
    // primal operand.
    assert!(matches!(scatter.inputs[0], ValRef::Local(_)));
    assert_ne!(scatter.inputs[0], ValRef::External(operand_key));
    assert_eq!(scatter.inputs[1], ValRef::External(indices_key));
    assert_eq!(scatter.inputs[2], ValRef::Local(updates_tangent));

    // And the fragment should contain a matching Constant + BroadcastInDim
    // pair feeding the scatter's operand.
    let constant_count = fragment
        .ops()
        .iter()
        .filter(|op_node| matches!(op_node.op, StdTensorOp::Constant { .. }))
        .count();
    let broadcast_count = fragment
        .ops()
        .iter()
        .filter(|op_node| matches!(op_node.op, StdTensorOp::BroadcastInDim { .. }))
        .count();
    assert_eq!(constant_count, 1, "exactly one Constant(zero) expected");
    assert_eq!(broadcast_count, 1, "exactly one BroadcastInDim expected");
}

#[test]
fn linearize_scatter_both_tangents_emit_single_scatter() {
    // Both operand_dot and updates_dot active => single Scatter with
    // active_mask [true, false, true], no zero-operand plumbing.
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(40);
    let indices_key = input_key(41);
    let updates_key = input_key(42);
    let operand_tangent = builder.add_input(tensor_input(43));
    let updates_tangent = builder.add_input(tensor_input(44));
    seed_metadata(
        &mut ctx,
        &[
            (operand_key.clone(), &[5]),
            (indices_key.clone(), &[3, 1]),
            (updates_key.clone(), &[3]),
        ],
    );

    let config = rank1_scatter_config();
    let op = StdTensorOp::Scatter(config.clone());
    let primal_in = vec![operand_key, indices_key.clone(), updates_key];
    let tangent_in = [Some(operand_tangent), None, Some(updates_tangent)];

    let result = op.linearize(&mut builder, &primal_in, &[], &tangent_in, &mut ctx);
    assert_eq!(result.len(), 1);
    let _ = result[0].expect("output tangent must be active");

    let fragment = builder.build();
    assert_eq!(
        fragment.ops().len(),
        1,
        "both-tangents case must emit exactly one Scatter"
    );
    let scatter = &fragment.ops()[0];
    assert_eq!(scatter.op, StdTensorOp::Scatter(config));
    assert_eq!(
        scatter.mode,
        OpMode::Linear {
            active_mask: vec![true, false, true],
        }
    );
    assert_eq!(scatter.inputs[0], ValRef::Local(operand_tangent));
    assert_eq!(scatter.inputs[1], ValRef::External(indices_key));
    assert_eq!(scatter.inputs[2], ValRef::Local(updates_tangent));
}

#[test]
fn transpose_scatter_emits_identity_for_operand_and_gather_for_updates() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(60));
    let operand_key = input_key(61);
    let indices_key = input_key(62);
    let updates_key = input_key(63);
    seed_metadata(
        &mut ctx,
        &[
            (operand_key.clone(), &[5]),
            (indices_key.clone(), &[3, 1]),
            (updates_key.clone(), &[3]),
        ],
    );

    let config = rank1_scatter_config();
    let op = StdTensorOp::Scatter(config.clone());
    let inputs = vec![
        ValRef::External(operand_key),
        ValRef::External(indices_key.clone()),
        ValRef::External(updates_key),
    ];

    let result = op.transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OpMode::Linear {
            active_mask: vec![true, false, true],
        },
        &mut ctx,
    );
    assert_eq!(
        result[0],
        Some(cot),
        "operand cotangent must be an identity passthrough of cot_out"
    );
    assert_eq!(result[1], None, "indices cotangent must stay None");
    assert!(result[2].is_some(), "updates cotangent must be active");

    let fragment = builder.build();
    assert_eq!(
        fragment.ops().len(),
        1,
        "identity passthrough must not emit extra ops; only the Gather"
    );
    let gather = &fragment.ops()[0];
    let StdTensorOp::Gather(gather_cfg) = &gather.op else {
        panic!("expected Gather op, got {:?}", gather.op);
    };
    assert_eq!(gather_cfg.offset_dims, config.update_window_dims);
    assert_eq!(gather_cfg.collapsed_slice_dims, config.inserted_window_dims);
    assert_eq!(
        gather_cfg.start_index_map,
        config.scatter_dims_to_operand_dims
    );
    assert_eq!(gather_cfg.index_vector_dim, config.index_vector_dim);
    assert_eq!(
        gather_cfg.slice_sizes,
        vec![1],
        "all operand dims are inserted_window_dims => slice_sizes = [1]"
    );
    assert_eq!(gather.inputs[0], ValRef::Local(cot));
    assert_eq!(gather.inputs[1], ValRef::External(indices_key));
}

#[test]
fn transpose_scatter_updates_only_emits_only_gather() {
    // active_mask[0] = false, active_mask[2] = true → only the updates
    // cotangent comes back, via a single Gather.
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(60));
    let operand_key = input_key(61);
    let indices_key = input_key(62);
    let updates_key = input_key(63);
    seed_metadata(
        &mut ctx,
        &[
            (operand_key.clone(), &[5]),
            (indices_key.clone(), &[3, 1]),
            (updates_key.clone(), &[3]),
        ],
    );

    let config = rank1_scatter_config();
    let op = StdTensorOp::Scatter(config);
    let inputs = vec![
        ValRef::External(operand_key),
        ValRef::External(indices_key.clone()),
        ValRef::External(updates_key),
    ];

    let result = op.transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OpMode::Linear {
            active_mask: vec![false, false, true],
        },
        &mut ctx,
    );
    assert_eq!(
        result[0], None,
        "operand cotangent is None when operand is inactive"
    );
    assert_eq!(result[1], None, "indices cotangent must stay None");
    assert!(result[2].is_some(), "updates cotangent must be active");

    let fragment = builder.build();
    assert_eq!(fragment.ops().len(), 1);
    let gather = &fragment.ops()[0];
    let StdTensorOp::Gather(_) = &gather.op else {
        panic!("expected Gather op, got {:?}", gather.op);
    };
    assert_eq!(gather.inputs[0], ValRef::Local(cot));
    assert_eq!(gather.inputs[1], ValRef::External(indices_key));
}

#[test]
fn transpose_scatter_operand_only_is_identity_and_no_ops() {
    // active_mask[0] = true, active_mask[2] = false → operand cotangent is
    // cot_out (identity passthrough); no ops emitted.
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(60));
    let operand_key = input_key(61);
    let indices_key = input_key(62);
    let updates_key = input_key(63);
    seed_metadata(
        &mut ctx,
        &[
            (operand_key.clone(), &[5]),
            (indices_key.clone(), &[3, 1]),
            (updates_key.clone(), &[3]),
        ],
    );

    let config = rank1_scatter_config();
    let op = StdTensorOp::Scatter(config);
    let inputs = vec![
        ValRef::External(operand_key),
        ValRef::External(indices_key),
        ValRef::External(updates_key),
    ];

    let result = op.transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OpMode::Linear {
            active_mask: vec![true, false, false],
        },
        &mut ctx,
    );
    assert_eq!(
        result,
        vec![Some(cot), None, None],
        "operand-only transpose must be identity passthrough and no-op"
    );
    assert!(builder.build().ops().is_empty());
}

#[test]
fn transpose_scatter_inactive_updates_returns_all_none() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(70));
    let operand_key = input_key(71);
    let indices_key = input_key(72);
    let updates_key = input_key(73);
    seed_metadata(
        &mut ctx,
        &[
            (operand_key.clone(), &[5]),
            (indices_key.clone(), &[3, 1]),
            (updates_key.clone(), &[3]),
        ],
    );
    let inputs = vec![
        ValRef::External(operand_key),
        ValRef::External(indices_key),
        ValRef::External(updates_key),
    ];

    let op = StdTensorOp::Scatter(rank1_scatter_config());
    let result = op.transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OpMode::Linear {
            active_mask: vec![false, false, false],
        },
        &mut ctx,
    );
    assert_eq!(result, vec![None, None, None]);
    assert!(builder.build().ops().is_empty());
}

#[test]
fn transpose_scatter_window_dims_derive_slice_sizes_from_updates_shape() {
    // Scatter 2-slabs into a rank-2 operand, along axis 0. The inverse
    // Gather must read a window of 2 along operand axis 1.
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(80));
    let operand_key = input_key(81);
    let indices_key = input_key(82);
    let updates_key = input_key(83);
    // operand: [4, 2], indices: [1, 1] (scalar index), updates: [1, 2]
    seed_metadata(
        &mut ctx,
        &[
            (operand_key.clone(), &[4, 2]),
            (indices_key.clone(), &[1, 1]),
            (updates_key.clone(), &[1, 2]),
        ],
    );

    let config = ScatterConfig {
        update_window_dims: vec![1],   // updates axis 1 is the window (size 2)
        inserted_window_dims: vec![0], // operand axis 0 is scalar-indexed
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let op = StdTensorOp::Scatter(config);
    let inputs = vec![
        ValRef::External(operand_key),
        ValRef::External(indices_key),
        ValRef::External(updates_key),
    ];

    let result = op.transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OpMode::Linear {
            active_mask: vec![false, false, true],
        },
        &mut ctx,
    );
    assert!(result[2].is_some());

    let fragment = builder.build();
    let StdTensorOp::Gather(cfg) = &fragment.ops()[0].op else {
        panic!("expected Gather op");
    };
    // operand rank = 2, inserted_window_dims = [0] -> window dim is operand
    // axis 1, which inherits updates.shape[1] = 2.
    assert_eq!(
        cfg.slice_sizes,
        vec![1, 2],
        "slice_sizes must be 1 for inserted dim and updates.shape[1] for the window dim"
    );
    assert_eq!(cfg.offset_dims, vec![1]);
}

#[test]
fn transpose_scatter_symbolic_window_dim_emits_dynamic_gather() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(900));
    let operand_key = input_key(901);
    let indices_key = input_key(902);
    let updates_key = input_key(903);

    ctx.insert_metadata(
        operand_key.clone(),
        TensorMeta::exact(DType::F64, vec![SymDim::from(4usize), SymDim::from(2usize)]),
    );
    ctx.insert_metadata(
        indices_key.clone(),
        TensorMeta::exact(DType::I64, vec![SymDim::from(1usize), SymDim::from(1usize)]),
    );
    ctx.insert_metadata(
        updates_key.clone(),
        TensorMeta::exact(
            DType::F64,
            vec![SymDim::from(1usize), SymDim::tensor_axis(903, 1)],
        ),
    );

    let config = ScatterConfig {
        update_window_dims: vec![1],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };

    let op = StdTensorOp::Scatter(config);
    let inputs = vec![
        ValRef::External(operand_key),
        ValRef::External(indices_key),
        ValRef::External(updates_key.clone()),
    ];
    let result = op.transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OpMode::Linear {
            active_mask: vec![false, false, true],
        },
        &mut ctx,
    );

    assert!(result[2].is_some());
    let fragment = builder.build();
    let gather = fragment.ops().last().expect("expected gather op");
    match &gather.op {
        StdTensorOp::GatherDynamicSliceSizes { slice_sizes, .. } => {
            assert_eq!(
                slice_sizes[1],
                DimExpr::InputDim {
                    input_idx: 2,
                    axis: 1,
                }
            );
            assert_eq!(gather.inputs[2], ValRef::External(updates_key));
        }
        other => panic!("expected GatherDynamicSliceSizes, got {other:?}"),
    }
}
