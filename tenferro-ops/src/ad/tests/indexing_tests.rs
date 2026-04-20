//! Unit tests for the `Gather` / `Scatter` AD rules in `ad::indexing`.

use chainrules_core::PrimitiveOp;
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use tenferro_tensor::{DType, GatherConfig, ScatterConfig};

use crate::ad::context::ShapeGuardContext;
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
    TensorMeta {
        dtype: DType::F64,
        shape: shape.iter().copied().map(SymDim::from).collect(),
    }
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
    assert_eq!(fragment.ops().len(), 1);
    let scatter = &fragment.ops()[0];
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
    // The scatter's third input (updates) must be the cotangent; indices are
    // reused from the primal.
    assert_eq!(scatter.inputs[1], ValRef::External(indices_key));
    assert_eq!(scatter.inputs[2], ValRef::Local(cot));
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
fn linearize_scatter_threads_updates_tangent_through_same_scatter() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(40);
    let indices_key = input_key(41);
    let updates_key = input_key(42);
    let updates_tangent = builder.add_input(tensor_input(43));

    let config = rank1_scatter_config();
    let op = StdTensorOp::Scatter(config.clone());
    let primal_in = vec![operand_key.clone(), indices_key.clone(), updates_key];
    let tangent_in = [None, None, Some(updates_tangent)];

    let result = op.linearize(&mut builder, &primal_in, &[], &tangent_in, &mut ctx);

    assert_eq!(result.len(), 1);
    let _tangent_out = result[0].expect("output tangent must be active");
    let fragment = builder.build();
    assert_eq!(fragment.ops().len(), 1);
    let emitted = &fragment.ops()[0];
    assert_eq!(emitted.op, StdTensorOp::Scatter(config));
    assert_eq!(
        emitted.mode,
        OpMode::Linear {
            active_mask: vec![false, false, true],
        }
    );
    assert_eq!(emitted.inputs[0], ValRef::External(operand_key));
    assert_eq!(emitted.inputs[1], ValRef::External(indices_key));
    assert_eq!(emitted.inputs[2], ValRef::Local(updates_tangent));
}

#[test]
fn linearize_scatter_inactive_updates_tangent_returns_none() {
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
fn transpose_scatter_emits_gather_for_updates_cotangent() {
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
            active_mask: vec![false, false, true],
        },
        &mut ctx,
    );
    assert_eq!(
        result[0], None,
        "operand cotangent is None (zeros-based scatter)"
    );
    assert_eq!(result[1], None, "indices cotangent must stay None");
    assert!(result[2].is_some(), "updates cotangent must be active");

    let fragment = builder.build();
    assert_eq!(fragment.ops().len(), 1);
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
