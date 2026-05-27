use super::*;

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
