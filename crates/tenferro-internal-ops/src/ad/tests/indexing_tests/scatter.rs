use super::*;

#[test]
fn linearize_scatter_inactive_tangents_returns_none() {
    // Both operand_dot = None and updates_dot = None => [None] with no ops.
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let operand_key = input_key(50);
    let indices_key = input_key(51);
    let updates_key = input_key(52);
    let op = StdTensorOp::Scatter(rank1_scatter_config());
    let primal_in = vec![operand_key, indices_key, updates_key];
    let tangent_in: [Option<LocalValueId>; 3] = [None, None, None];
    let result = op
        .jvp_rule(&mut builder, &primal_in, &[], &tangent_in, &mut ctx)
        .unwrap();
    assert_eq!(result, vec![None]);
    assert!(builder.build().operations().is_empty());
}

#[test]
fn linearize_scatter_operand_only_is_identity_passthrough() {
    // Only operand_dot is active: the output tangent is the operand
    // tangent itself, with no ops emitted.
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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

    let result = op
        .jvp_rule(&mut builder, &primal_in, &[], &tangent_in, &mut ctx)
        .unwrap();
    assert_eq!(
        result,
        vec![Some(operand_tangent)],
        "operand-only scatter tangent must be the operand tangent itself"
    );
    let graph = builder.build();
    assert!(
        graph.operations().is_empty(),
        "identity passthrough must not emit any ops"
    );
}

#[test]
fn linearize_scatter_updates_only_uses_zero_operand() {
    // Only updates_dot is active: emit Scatter(zeros_like(operand), ..., d_updates)
    // via Constant + BroadcastInDim + Scatter.
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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

    let result = op
        .jvp_rule(&mut builder, &primal_in, &[], &tangent_in, &mut ctx)
        .unwrap();
    assert_eq!(result.len(), 1);
    let _ = result[0].expect("output tangent must be active");

    let graph = builder.build();
    // Graph: [Constant, BroadcastInDim, Scatter]
    assert_eq!(
        graph.operations().len(),
        3,
        "expected Constant + BroadcastInDim + Scatter in the graph"
    );
    let scatter = graph
        .operations()
        .iter()
        .find(|op_node| matches!(op_node.operation, StdTensorOp::Scatter(_)))
        .expect("expected a Scatter op");
    assert_eq!(scatter.operation, StdTensorOp::Scatter(config));
    assert_eq!(
        scatter.role,
        OperationRole::Linearized {
            active_mask: vec![false, false, true],
        }
    );
    // The scatter's operand must be a freshly built zero local, not the
    // primal operand.
    assert!(matches!(scatter.inputs[0], ValueRef::Local(_)));
    assert_ne!(scatter.inputs[0], ValueRef::External(operand_key));
    assert_eq!(scatter.inputs[1], ValueRef::External(indices_key));
    assert_eq!(scatter.inputs[2], ValueRef::Local(updates_tangent));

    // And the graph should contain a matching Constant + BroadcastInDim
    // pair feeding the scatter's operand.
    let constant_count = graph
        .operations()
        .iter()
        .filter(|op_node| matches!(op_node.operation, StdTensorOp::Constant { .. }))
        .count();
    let broadcast_count = graph
        .operations()
        .iter()
        .filter(|op_node| matches!(op_node.operation, StdTensorOp::BroadcastInDim { .. }))
        .count();
    assert_eq!(constant_count, 1, "exactly one Constant(zero) expected");
    assert_eq!(broadcast_count, 1, "exactly one BroadcastInDim expected");
}

#[test]
fn linearize_scatter_both_tangents_emit_single_scatter() {
    // Both operand_dot and updates_dot active => single Scatter with
    // active_mask [true, false, true], no zero-operand plumbing.
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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

    let result = op
        .jvp_rule(&mut builder, &primal_in, &[], &tangent_in, &mut ctx)
        .unwrap();
    assert_eq!(result.len(), 1);
    let _ = result[0].expect("output tangent must be active");

    let graph = builder.build();
    assert_eq!(
        graph.operations().len(),
        1,
        "both-tangents case must emit exactly one Scatter"
    );
    let scatter = &graph.operations()[0];
    assert_eq!(scatter.operation, StdTensorOp::Scatter(config));
    assert_eq!(
        scatter.role,
        OperationRole::Linearized {
            active_mask: vec![true, false, true],
        }
    );
    assert_eq!(scatter.inputs[0], ValueRef::Local(operand_tangent));
    assert_eq!(scatter.inputs[1], ValueRef::External(indices_key));
    assert_eq!(scatter.inputs[2], ValueRef::Local(updates_tangent));
}

#[test]
fn transpose_scatter_emits_identity_for_operand_and_gather_for_updates() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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
        ValueRef::External(operand_key),
        ValueRef::External(indices_key.clone()),
        ValueRef::External(updates_key),
    ];

    let result = op
        .transpose_rule(
            &mut builder,
            &[Some(cot)],
            &inputs,
            &OperationRole::Linearized {
                active_mask: vec![true, false, true],
            },
            &mut ctx,
        )
        .unwrap();
    assert_eq!(
        result[0],
        Some(cot),
        "operand cotangent must be an identity passthrough of cot_out"
    );
    assert_eq!(result[1], None, "indices cotangent must stay None");
    assert!(result[2].is_some(), "updates cotangent must be active");

    let graph = builder.build();
    assert_eq!(
        graph.operations().len(),
        1,
        "identity passthrough must not emit extra ops; only the Gather"
    );
    let gather = &graph.operations()[0];
    let StdTensorOp::Gather(gather_cfg) = &gather.operation else {
        panic!("expected Gather op, got {:?}", gather.operation);
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
    assert_eq!(gather.inputs[0], ValueRef::Local(cot));
    assert_eq!(gather.inputs[1], ValueRef::External(indices_key));
}

#[test]
fn transpose_scatter_updates_only_emits_only_gather() {
    // active_mask[0] = false, active_mask[2] = true → only the updates
    // cotangent comes back, via a single Gather.
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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
        ValueRef::External(operand_key),
        ValueRef::External(indices_key.clone()),
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
    assert_eq!(
        result[0], None,
        "operand cotangent is None when operand is inactive"
    );
    assert_eq!(result[1], None, "indices cotangent must stay None");
    assert!(result[2].is_some(), "updates cotangent must be active");

    let graph = builder.build();
    assert_eq!(graph.operations().len(), 1);
    let gather = &graph.operations()[0];
    let StdTensorOp::Gather(_) = &gather.operation else {
        panic!("expected Gather op, got {:?}", gather.operation);
    };
    assert_eq!(gather.inputs[0], ValueRef::Local(cot));
    assert_eq!(gather.inputs[1], ValueRef::External(indices_key));
}

#[test]
fn transpose_scatter_operand_only_is_identity_and_no_ops() {
    // active_mask[0] = true, active_mask[2] = false → operand cotangent is
    // cot_out (identity passthrough); no ops emitted.
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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
                active_mask: vec![true, false, false],
            },
            &mut ctx,
        )
        .unwrap();
    assert_eq!(
        result,
        vec![Some(cot), None, None],
        "operand-only transpose must be identity passthrough and no-op"
    );
    assert!(builder.build().operations().is_empty());
}

#[test]
fn transpose_scatter_inactive_updates_returns_all_none() {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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
        ValueRef::External(operand_key),
        ValueRef::External(indices_key),
        ValueRef::External(updates_key),
    ];

    let op = StdTensorOp::Scatter(rank1_scatter_config());
    let result = op
        .transpose_rule(
            &mut builder,
            &[Some(cot)],
            &inputs,
            &OperationRole::Linearized {
                active_mask: vec![false, false, false],
            },
            &mut ctx,
        )
        .unwrap();
    assert_eq!(result, vec![None, None, None]);
    assert!(builder.build().operations().is_empty());
}

#[test]
fn transpose_scatter_window_dims_derive_slice_sizes_from_updates_shape() {
    // Scatter 2-slabs into a rank-2 operand, along axis 0. The inverse
    // Gather must read a window of 2 along operand axis 1.
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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
    assert!(result[2].is_some());

    let graph = builder.build();
    let StdTensorOp::Gather(cfg) = &graph.operations()[0].operation else {
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
    let mut builder = GraphBuilder::<StdTensorOp>::new();
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
        ValueRef::External(operand_key),
        ValueRef::External(indices_key),
        ValueRef::External(updates_key.clone()),
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

    assert!(result[2].is_some());
    let graph = builder.build();
    let gather = graph.operations().last().expect("expected gather op");
    match &gather.operation {
        StdTensorOp::GatherDynamicSliceSizes { slice_sizes, .. } => {
            assert_eq!(
                slice_sizes[1],
                DimExpr::InputDim {
                    input_idx: 2,
                    axis: 1,
                }
            );
            assert_eq!(gather.inputs[2], ValueRef::External(updates_key));
        }
        other => panic!("expected GatherDynamicSliceSizes, got {other:?}"),
    }
}
