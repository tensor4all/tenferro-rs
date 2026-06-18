//! Unit tests for the `Gather` / `Scatter` AD rules in `ad::indexing`.

use computegraph::graph::GraphBuilder;
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use tenferro_tensor::{DType, GatherConfig, ScatterConfig};

use crate::ad::context::ShapeGuardContext;
use crate::dim_expr::DimExpr;
use crate::input_key::TensorInputKey;
use crate::std_tensor_op::StdTensorOp;
use crate::{SymDim, TensorMeta};

mod scatter;

fn tensor_input(id: u64) -> TensorInputKey {
    TensorInputKey::User { id }
}

fn input_key(id: u64) -> ValueKey<StdTensorOp> {
    ValueKey::Input(tensor_input(id))
}

fn meta(shape: &[usize]) -> TensorMeta {
    TensorMeta::exact(DType::F64, shape.iter().copied().map(Into::into).collect())
}

fn seed_metadata(ctx: &mut ShapeGuardContext, entries: &[(ValueKey<StdTensorOp>, &[usize])]) {
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
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(1);
    let indices_key = input_key(2);
    let operand_tangent = builder.add_input(tensor_input(3));

    let config = rank1_gather_config();
    let op = StdTensorOp::Gather(config.clone());
    let primal_in = vec![operand_key.clone(), indices_key.clone()];
    let tangent_in = [Some(operand_tangent), None];

    let result = op
        .jvp_rule(&mut builder, &primal_in, &[], &tangent_in, &mut ctx)
        .unwrap();

    assert_eq!(result.len(), 1);
    let tangent_out = result[0].expect("output tangent must be active");
    let graph = builder.build();

    assert_eq!(graph.operations().len(), 1);
    let gather_op = &graph.operations()[0];
    assert_eq!(gather_op.operation, StdTensorOp::Gather(config));
    assert_eq!(
        gather_op.role,
        OperationRole::Linearized {
            active_mask: vec![true, false],
        }
    );
    // The second operand of the linearised Gather must be the *primal* indices.
    assert_eq!(gather_op.inputs[1], ValueRef::External(indices_key),);
    // The returned local id must be one of the emitted gather's outputs.
    assert!(gather_op.outputs.contains(&tangent_out));
}

#[test]
fn linearize_gather_inactive_tangent_returns_none() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(10);
    let indices_key = input_key(11);

    let op = StdTensorOp::Gather(rank1_gather_config());
    let primal_in = vec![operand_key, indices_key];
    let tangent_in: [Option<LocalValueId>; 2] = [None, None];

    let result = op
        .jvp_rule(&mut builder, &primal_in, &[], &tangent_in, &mut ctx)
        .unwrap();
    assert_eq!(result, vec![None]);
    assert!(builder.build().operations().is_empty());
}

#[test]
fn linearize_dynamic_gather_reuses_primal_indices_and_shape_sources() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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

    let result = op
        .jvp_rule(&mut builder, &primal_in, &[], &tangent_in, &mut ctx)
        .unwrap();

    assert_eq!(result.len(), 1);
    let tangent_out = result[0].expect("output tangent must be active");
    let graph = builder.build();

    assert_eq!(graph.operations().len(), 1);
    let gather = &graph.operations()[0];
    assert_eq!(gather.operation, op);
    assert_eq!(gather.inputs[0], ValueRef::Local(operand_tangent));
    assert_eq!(gather.inputs[1], ValueRef::External(indices_key));
    assert_eq!(gather.inputs[2], ValueRef::External(shape_source_key));
    assert_eq!(
        gather.role,
        OperationRole::Linearized {
            active_mask: vec![true, false, false],
        }
    );
    assert!(gather.outputs.contains(&tangent_out));
}

#[test]
fn linearize_dynamic_slice_reuses_primal_starts() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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

    let result = op
        .jvp_rule(&mut builder, &primal_in, &[], &tangent_in, &mut ctx)
        .unwrap();

    assert_eq!(result.len(), 1);
    let tangent_out = result[0].expect("output tangent must be active");
    let graph = builder.build();

    assert_eq!(graph.operations().len(), 1);
    let dynamic_slice = &graph.operations()[0];
    assert_eq!(
        dynamic_slice.operation,
        StdTensorOp::DynamicSlice { slice_sizes }
    );
    assert_eq!(dynamic_slice.inputs[0], ValueRef::Local(operand_tangent));
    assert_eq!(dynamic_slice.inputs[1], ValueRef::External(starts_key));
    assert_eq!(
        dynamic_slice.role,
        OperationRole::Linearized {
            active_mask: vec![true, false],
        }
    );
    assert!(dynamic_slice.outputs.contains(&tangent_out));
}

#[test]
fn linearize_dynamic_slice_inactive_tangent_returns_none() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(108);
    let starts_key = input_key(109);

    let op = StdTensorOp::DynamicSlice {
        slice_sizes: vec![3],
    };
    let primal_in = vec![operand_key, starts_key];
    let tangent_in: [Option<LocalValueId>; 2] = [None, None];

    let result = op
        .jvp_rule(&mut builder, &primal_in, &[], &tangent_in, &mut ctx)
        .unwrap();
    assert_eq!(result, vec![None]);
    assert!(builder.build().operations().is_empty());
}

#[test]
fn linearize_dynamic_update_slice_reuses_primal_starts() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(110);
    let update_key = input_key(111);
    let starts_key = input_key(112);
    let operand_tangent = builder.add_input(tensor_input(113));
    let update_tangent = builder.add_input(tensor_input(114));

    let op = StdTensorOp::DynamicUpdateSlice;
    let primal_in = vec![operand_key, update_key, starts_key.clone()];
    let tangent_in = [Some(operand_tangent), Some(update_tangent), None];

    let result = op
        .jvp_rule(&mut builder, &primal_in, &[], &tangent_in, &mut ctx)
        .unwrap();

    assert_eq!(result.len(), 1);
    let tangent_out = result[0].expect("output tangent must be active");
    let graph = builder.build();

    assert_eq!(graph.operations().len(), 1);
    let update_slice = &graph.operations()[0];
    assert_eq!(update_slice.operation, StdTensorOp::DynamicUpdateSlice);
    assert_eq!(update_slice.inputs[0], ValueRef::Local(operand_tangent));
    assert_eq!(update_slice.inputs[1], ValueRef::Local(update_tangent));
    assert_eq!(update_slice.inputs[2], ValueRef::External(starts_key));
    assert_eq!(
        update_slice.role,
        OperationRole::Linearized {
            active_mask: vec![true, true, false],
        }
    );
    assert!(update_slice.outputs.contains(&tangent_out));
}

#[test]
fn transpose_dynamic_slice_emits_dynamic_update_slice() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(115));
    let operand_key = input_key(116);
    let starts_key = input_key(117);
    seed_metadata(
        &mut ctx,
        &[(operand_key.clone(), &[5]), (starts_key.clone(), &[1])],
    );

    let inputs = vec![
        ValueRef::External(operand_key.clone()),
        ValueRef::External(starts_key.clone()),
    ];
    let result = (StdTensorOp::DynamicSlice {
        slice_sizes: vec![3],
    })
    .transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OperationRole::Linearized {
            active_mask: vec![true, false],
        },
        &mut ctx,
    )
    .unwrap();

    assert!(result[0].is_some(), "operand cotangent must be active");
    assert_eq!(result[1], None, "starts cotangent must stay None");

    let graph = builder.build();
    let update_slice = graph
        .operations()
        .iter()
        .find(|op_node| matches!(op_node.operation, StdTensorOp::DynamicUpdateSlice))
        .expect("expected a DynamicUpdateSlice op");
    assert!(matches!(update_slice.inputs[0], ValueRef::Local(_)));
    assert_ne!(update_slice.inputs[0], ValueRef::External(operand_key));
    assert_eq!(update_slice.inputs[1], ValueRef::Local(cot));
    assert_eq!(update_slice.inputs[2], ValueRef::External(starts_key));
    assert_eq!(
        update_slice.role,
        OperationRole::Linearized {
            active_mask: vec![false, true, false],
        }
    );
}

#[test]
fn transpose_dynamic_update_slice_returns_operand_and_update_cotangents() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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
        ValueRef::External(operand_key),
        ValueRef::External(update_key),
        ValueRef::External(starts_key.clone()),
    ];
    let result = StdTensorOp::DynamicUpdateSlice
        .transpose_rule(
            &mut builder,
            &[Some(cot)],
            &inputs,
            &OperationRole::Linearized {
                active_mask: vec![true, true, false],
            },
            &mut ctx,
        )
        .unwrap();

    assert!(result[0].is_some(), "operand cotangent must be active");
    assert!(result[1].is_some(), "update cotangent must be active");
    assert_eq!(result[2], None, "starts cotangent must stay None");

    let graph = builder.build();
    assert!(
        graph
            .operations()
            .iter()
            .any(|op_node| matches!(op_node.operation, StdTensorOp::DynamicUpdateSlice)),
        "operand cotangent should be masked by DynamicUpdateSlice"
    );
    let update_ct = graph
        .operations()
        .iter()
        .find(|op_node| matches!(op_node.operation, StdTensorOp::DynamicSlice { .. }))
        .expect("expected a DynamicSlice op for update cotangent");
    assert_eq!(
        update_ct.operation,
        StdTensorOp::DynamicSlice {
            slice_sizes: vec![3],
        }
    );
    assert_eq!(update_ct.inputs[0], ValueRef::Local(cot));
    assert_eq!(update_ct.inputs[1], ValueRef::External(starts_key));
}

#[test]
fn transpose_gather_emits_scatter_with_inverted_config() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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
        ValueRef::External(operand_key.clone()),
        ValueRef::External(indices_key.clone()),
    ];

    let result = op
        .transpose_rule(
            &mut builder,
            &[Some(cot)],
            &inputs,
            &OperationRole::Linearized {
                active_mask: vec![true, false],
            },
            &mut ctx,
        )
        .unwrap();
    assert!(result[0].is_some(), "operand cotangent must be active");
    assert_eq!(result[1], None, "indices cotangent must stay None");

    let graph = builder.build();
    // Under StableHLO add-scatter semantics the inverse scatter must use
    // a zero operand built from Constant + BroadcastInDim, so the graph
    // contains [Constant, BroadcastInDim, Scatter].
    assert_eq!(graph.operations().len(), 3);
    let scatter = graph
        .operations()
        .iter()
        .find(|op_node| matches!(op_node.operation, StdTensorOp::Scatter(_)))
        .expect("expected a Scatter op in the graph");
    let StdTensorOp::Scatter(scatter_cfg) = &scatter.operation else {
        panic!("expected Scatter op, got {:?}", scatter.operation);
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
    assert!(matches!(scatter.inputs[0], ValueRef::Local(_)));
    assert_ne!(scatter.inputs[0], ValueRef::External(operand_key));
    // Indices are reused from the primal; updates are the cotangent.
    assert_eq!(scatter.inputs[1], ValueRef::External(indices_key));
    assert_eq!(scatter.inputs[2], ValueRef::Local(cot));
    assert_eq!(
        scatter.role,
        OperationRole::Linearized {
            active_mask: vec![false, false, true],
        }
    );
}

#[test]
fn transpose_gather_inactive_path_returns_all_none() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(30);
    let indices_key = input_key(31);
    let inputs = vec![
        ValueRef::External(operand_key),
        ValueRef::External(indices_key),
    ];

    let op = StdTensorOp::Gather(rank1_gather_config());

    let result = op
        .transpose_rule(
            &mut builder,
            &[None],
            &inputs,
            &OperationRole::Linearized {
                active_mask: vec![true, false],
            },
            &mut ctx,
        )
        .unwrap();
    assert_eq!(result, vec![None, None]);
    assert!(builder.build().operations().is_empty());
}

#[test]
fn transpose_dynamic_gather_emits_scatter_and_ignores_shape_sources() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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
        ValueRef::External(operand_key.clone()),
        ValueRef::External(indices_key.clone()),
        ValueRef::External(shape_source_key),
    ];

    let result = op
        .transpose_rule(
            &mut builder,
            &[Some(cot)],
            &inputs,
            &OperationRole::Linearized {
                active_mask: vec![true, false, false],
            },
            &mut ctx,
        )
        .unwrap();

    assert!(result[0].is_some(), "operand cotangent must be active");
    assert_eq!(result[1], None, "indices cotangent must stay None");
    assert_eq!(result[2], None, "shape source cotangent must stay None");

    let graph = builder.build();
    let scatter = graph
        .operations()
        .iter()
        .find(|op_node| matches!(op_node.operation, StdTensorOp::Scatter(_)))
        .expect("expected a Scatter op in the graph");
    let StdTensorOp::Scatter(scatter_cfg) = &scatter.operation else {
        panic!("expected Scatter op, got {:?}", scatter.operation);
    };
    assert_eq!(scatter_cfg.update_window_dims, vec![1]);
    assert_eq!(scatter_cfg.inserted_window_dims, vec![0]);
    assert_eq!(scatter_cfg.scatter_dims_to_operand_dims, vec![0]);
    assert_eq!(scatter_cfg.index_vector_dim, 1);
    assert!(matches!(scatter.inputs[0], ValueRef::Local(_)));
    assert_ne!(scatter.inputs[0], ValueRef::External(operand_key));
    assert_eq!(scatter.inputs[1], ValueRef::External(indices_key));
    assert_eq!(scatter.inputs[2], ValueRef::Local(cot));
}

#[test]
fn transpose_scatter_returns_none_for_mismatched_update_window_dims() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(130));
    let operand_key = input_key(131);
    let indices_key = input_key(132);
    let updates_key = input_key(133);
    seed_metadata(
        &mut ctx,
        &[
            (operand_key.clone(), &[4, 2]),
            (indices_key.clone(), &[1, 1]),
            (updates_key.clone(), &[1, 2]),
        ],
    );
    let op = StdTensorOp::Scatter(ScatterConfig {
        update_window_dims: vec![0],
        inserted_window_dims: vec![],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    });
    let inputs = vec![
        ValueRef::External(operand_key),
        ValueRef::External(indices_key),
        ValueRef::External(updates_key),
    ];

    let result = op
        .transpose_rule(
            &mut builder,
            &[Some(cot)],
            &inputs,
            &OperationRole::Linearized {
                active_mask: vec![false, false, true],
            },
            &mut ctx,
        )
        .unwrap();

    assert_eq!(result, vec![None, None, None]);
    assert!(builder.build().operations().is_empty());
}
