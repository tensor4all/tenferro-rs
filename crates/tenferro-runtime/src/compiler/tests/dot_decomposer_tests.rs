use super::*;

fn dot_general_exec_instr(
    lhs_contracting_dims: Vec<usize>,
    rhs_contracting_dims: Vec<usize>,
    lhs_batch_dims: Vec<usize>,
    rhs_batch_dims: Vec<usize>,
    input_slots: Vec<usize>,
    output_slot: usize,
    output_shape: Vec<DimExpr>,
) -> ExecInstruction {
    ExecInstruction {
        op: ExecOp::DotGeneral(DotGeneralConfig {
            lhs_contracting_dims,
            rhs_contracting_dims,
            lhs_batch_dims,
            rhs_batch_dims,
        }),
        semantic_operation_index: None,
        input_slots,
        output_slots: vec![output_slot],
        dtype: DType::F64,
        output_extents: vec![exact_extents(&output_shape)].into(),
        output_shapes: vec![output_shape].into(),
        last_use: Vec::new(),
    }
}

#[test]
fn test_dot_decomposer_already_canonical_is_noop() {
    // LHS [M, K] rank 2, RHS [K, N] rank 2, cont=[1], rhs_cont=[0]. Canonical.
    let instr = dot_general_exec_instr(
        vec![1],
        vec![0],
        vec![],
        vec![],
        vec![0, 1],
        2,
        dim_shape(&[4, 6]),
    );
    let mut program = make_exec_program(vec![instr], vec![0, 1], vec![2], 3);

    dot_decomposer(&mut program, &[dim_shape(&[4, 5]), dim_shape(&[5, 6])]).unwrap();

    assert_eq!(program.instructions.len(), 1);
    assert_eq!(program.n_slots, 3);
}

#[test]
fn test_dot_decomposer_missing_slot_metadata_returns_error() {
    let instr = dot_general_exec_instr(
        vec![1],
        vec![0],
        vec![],
        vec![],
        vec![0, 2],
        3,
        dim_shape(&[4, 6]),
    );
    let mut program = make_exec_program(vec![instr], vec![0, 1], vec![3], 4);

    let err = dot_decomposer(&mut program, &[dim_shape(&[4, 5]), dim_shape(&[5, 6])]).unwrap_err();

    assert!(matches!(
        err,
        Error::Internal(ref message)
            if message.contains("missing shape for slot 2")
    ));
}

#[test]
fn test_dot_decomposer_mismatched_batch_dim_count_returns_error() {
    let instr = dot_general_exec_instr(
        vec![1],
        vec![0],
        vec![0],
        vec![],
        vec![0, 1],
        2,
        dim_shape(&[4, 6]),
    );
    let mut program = make_exec_program(vec![instr], vec![0, 1], vec![2], 3);

    let err = dot_decomposer(&mut program, &[dim_shape(&[4, 5]), dim_shape(&[5, 6])]).unwrap_err();

    assert!(matches!(
        err,
        Error::Internal(ref message)
            if message.contains("lhs batch dim count 1")
    ));
}

#[test]
fn test_dot_decomposer_multi_contracting_dim() {
    // LHS [a, b, M, K1, K2] rank 5, RHS [a, b, K1, K2, N] rank 5,
    // lhs_batch=[0, 1], rhs_batch=[0, 1], lhs_cont=[3, 4], rhs_cont=[2, 3].
    // Output: [M, N, a, b] (free_L=[M], free_R=[N], batch=[a, b]).
    let instr = dot_general_exec_instr(
        vec![3, 4],
        vec![2, 3],
        vec![0, 1],
        vec![0, 1],
        vec![0, 1],
        2,
        dim_shape(&[7, 11, 2, 3]),
    );
    let mut program = make_exec_program(vec![instr], vec![0, 1], vec![2], 3);

    dot_decomposer(
        &mut program,
        &[dim_shape(&[2, 3, 7, 4, 5]), dim_shape(&[2, 3, 4, 5, 11])],
    )
    .unwrap();

    // Expected instruction chain:
    //   Transpose(slot 0) -> N1        // [M, K1, K2, a, b]
    //   Reshape(N1)       -> N2        // [M, K1*K2, a, b]
    //   Transpose(slot 1) -> N3        // [K1, K2, N, a, b]
    //   Reshape(N3)       -> N4        // [K1*K2, N, a, b]
    //   DotGeneral(N2, N4) -> slot 2   // canonical, [M, N, a, b]
    // No output Reshape because fi_L = fi_R = 1.
    assert_eq!(program.instructions.len(), 5);

    assert!(matches!(
        program.instructions[0].op,
        ExecOp::Transpose { .. }
    ));
    assert!(matches!(program.instructions[1].op, ExecOp::Reshape { .. }));
    assert!(matches!(
        program.instructions[2].op,
        ExecOp::Transpose { .. }
    ));
    assert!(matches!(program.instructions[3].op, ExecOp::Reshape { .. }));
    let dot = &program.instructions[4];
    assert_eq!(dot.output_slots, vec![2]);
    match &dot.op {
        ExecOp::DotGeneral(config) => {
            assert_eq!(config.lhs_contracting_dims, vec![1]);
            assert_eq!(config.rhs_contracting_dims, vec![0]);
            assert_eq!(config.lhs_batch_dims, vec![2, 3]);
            assert_eq!(config.rhs_batch_dims, vec![2, 3]);
        }
        _ => panic!("expected canonical DotGeneral"),
    }
}

#[test]
fn test_dot_decomposer_multi_free_dim_emits_output_reshape() {
    // LHS [M1, M2, K] rank 3, RHS [K, N] rank 2, no batch,
    // lhs_cont=[2], rhs_cont=[0]. free_L=[0, 1] (multi-free).
    // Output: [M1, M2, N].
    let instr = dot_general_exec_instr(
        vec![2],
        vec![0],
        vec![],
        vec![],
        vec![0, 1],
        2,
        dim_shape(&[2, 3, 5]),
    );
    let mut program = make_exec_program(vec![instr], vec![0, 1], vec![2], 3);

    dot_decomposer(&mut program, &[dim_shape(&[2, 3, 4]), dim_shape(&[4, 5])]).unwrap();

    // Expected: merge Reshape for LHS (no Transpose needed), canonical
    // DotGeneral, output Reshape.
    assert_eq!(program.instructions.len(), 3);

    let reshape_lhs = &program.instructions[0];
    assert!(matches!(reshape_lhs.op, ExecOp::Reshape { .. }));
    assert_eq!(reshape_lhs.input_slots, vec![0]);

    let dot = &program.instructions[1];
    match &dot.op {
        ExecOp::DotGeneral(config) => {
            assert_eq!(config.lhs_contracting_dims, vec![1]);
            assert_eq!(config.rhs_contracting_dims, vec![0]);
            assert!(config.lhs_batch_dims.is_empty());
            assert!(config.rhs_batch_dims.is_empty());
        }
        _ => panic!("expected canonical DotGeneral"),
    }

    let out_reshape = &program.instructions[2];
    match &out_reshape.op {
        ExecOp::Reshape { .. } => {}
        _ => panic!("expected output Reshape"),
    }
    // Output Reshape must carry the original LHS/RHS as shape providers so
    // the dynamic axes can be recovered at runtime.
    assert_eq!(out_reshape.input_slots.len(), 3);
    assert_eq!(out_reshape.input_slots[1], 0);
    assert_eq!(out_reshape.input_slots[2], 1);
    assert_eq!(out_reshape.output_slots, vec![2]);
}

#[test]
fn test_dot_decomposer_preserves_upper_bound_extents_in_merge_reshape() {
    let truncate_shape = dim_shape(&[5, 3, 4]);
    let truncated_lhs = ExecInstruction {
        op: ExecOp::DynamicTruncate { axis: 0 },
        semantic_operation_index: None,
        input_slots: vec![0, 1],
        output_slots: vec![2],
        dtype: DType::F64,
        output_shapes: vec![truncate_shape.clone()].into(),
        output_extents: vec![vec![
            ShapeExtent::upper_bound(DimExpr::Const(5)),
            ShapeExtent::exact(DimExpr::Const(3)),
            ShapeExtent::exact(DimExpr::Const(4)),
        ]]
        .into(),
        last_use: Vec::new(),
    };
    let dot = dot_general_exec_instr(
        vec![2],
        vec![0],
        vec![],
        vec![],
        vec![2, 3],
        4,
        dim_shape(&[5, 3, 2]),
    );
    let mut program = make_exec_program(vec![truncated_lhs, dot], vec![0, 1, 3], vec![4], 5);

    dot_decomposer(
        &mut program,
        &[dim_shape(&[5, 3, 4]), Vec::new(), dim_shape(&[4, 2])],
    )
    .unwrap();

    let lhs_merge = program
        .instructions
        .iter()
        .find(|instr| matches!(instr.op, ExecOp::Reshape { .. }) && instr.input_slots == vec![2])
        .expect("expected LHS merge reshape from truncated input");

    assert_eq!(
        lhs_merge.output_extents[0][0],
        ShapeExtent::upper_bound(DimExpr::mul(DimExpr::Const(5), DimExpr::Const(3))),
        "merged free dimension must remain an upper bound"
    );
    assert_eq!(
        lhs_merge.output_extents[0][1],
        ShapeExtent::exact(DimExpr::Const(4))
    );
}

#[test]
fn test_dot_decomposer_noncanonical_dot_then_permute_downstream() {
    // A non-canonical DotGeneral whose output is consumed by a downstream
    // Transpose must still produce the original output shape after the
    // decomp, so the downstream Transpose doesn't fall out of bounds.
    //
    // LHS [a, M, K] rank 3, RHS [a, K, N] rank 3, lhs_batch=[0], rhs_batch=[0],
    // lhs_cont=[2], rhs_cont=[1]. Canonical-by-shape but batch is leading;
    // decomp must canonicalize to batch-trailing.
    let dot = dot_general_exec_instr(
        vec![2],
        vec![1],
        vec![0],
        vec![0],
        vec![0, 1],
        2,
        dim_shape(&[3, 5, 7]), // [M, N, a]
    );
    let downstream = make_exec_instr(
        ExecOp::Transpose {
            perm: vec![2, 0, 1],
        },
        vec![2],
        vec![3],
    );
    let mut program = make_exec_program(vec![dot, downstream], vec![0, 1], vec![3], 4);

    dot_decomposer(
        &mut program,
        &[dim_shape(&[7, 3, 4]), dim_shape(&[7, 4, 5])],
    )
    .unwrap();

    // Downstream Transpose must still refer to the same slot (the original
    // DotGeneral output slot) and must still see the original [M, N, a]
    // shape.
    let downstream = program
        .instructions
        .iter()
        .find(|i| matches!(&i.op, ExecOp::Transpose { perm } if perm == &vec![2, 0, 1]))
        .expect("downstream Transpose must be preserved");
    assert_eq!(downstream.input_slots, vec![2]);
}

#[test]
fn test_dot_decomposer_noncanonical_dot_then_dot_downstream() {
    // First DotGeneral non-canonical, feeds its output as LHS to a second
    // DotGeneral. The decomp must not break the second DotGeneral's
    // dimension contract on the shared slot.
    //
    // Op 0: LHS [a, M1, M2, K] rank 4, RHS [a, K, N] rank 3,
    //       lhs_batch=[0], rhs_batch=[0], lhs_cont=[3], rhs_cont=[1].
    //       Output: [M1, M2, N, a] rank 4.
    //
    // Op 1: DotGeneral(Output_0 [M1, M2, N, a] rank 4, RHS2 [a, N, P] rank 3,
    //       lhs_batch=[3], rhs_batch=[0], lhs_cont=[2], rhs_cont=[1]).
    //       Output: [M1, M2, P, a].
    let first = dot_general_exec_instr(
        vec![3],
        vec![1],
        vec![0],
        vec![0],
        vec![0, 1],
        2,
        dim_shape(&[3, 4, 5, 6]),
    );
    let second = dot_general_exec_instr(
        vec![2],
        vec![1],
        vec![3],
        vec![0],
        vec![2, 3],
        4,
        dim_shape(&[3, 4, 7, 6]),
    );
    let mut program = make_exec_program(vec![first, second], vec![0, 1, 3], vec![4], 5);

    dot_decomposer(
        &mut program,
        &[
            dim_shape(&[6, 3, 4, 8]), // LHS 0: [a, M1, M2, K]
            dim_shape(&[6, 8, 5]),    // RHS 0: [a, K, N]
            dim_shape(&[6, 5, 7]),    // RHS 1: [a, N, P]
        ],
    )
    .unwrap();

    // Two canonical DotGenerals must be present after decomp, and the
    // first dot's output (slot 2) must still be consumed somewhere in the
    // decomposed program (possibly via an intermediate Reshape rather than
    // directly by the second DotGeneral).
    let dot_count = program
        .instructions
        .iter()
        .filter(|i| matches!(i.op, ExecOp::DotGeneral(_)))
        .count();
    assert_eq!(dot_count, 2);
    for i in program
        .instructions
        .iter()
        .filter(|i| matches!(i.op, ExecOp::DotGeneral(_)))
    {
        match &i.op {
            ExecOp::DotGeneral(config) => {
                assert_eq!(config.lhs_contracting_dims.len(), 1);
                assert_eq!(config.rhs_contracting_dims.len(), 1);
            }
            _ => unreachable!(),
        }
    }
    let slot2_has_consumer = program
        .instructions
        .iter()
        .any(|i| i.input_slots.contains(&2));
    assert!(
        slot2_has_consumer,
        "slot 2 (first-dot output) must still be consumed"
    );
    // Final output slot still resolves in the program.
    let produces_slot_4 = program
        .instructions
        .iter()
        .any(|i| i.output_slots.contains(&4));
    assert!(produces_slot_4, "final output slot 4 must be produced");
}
