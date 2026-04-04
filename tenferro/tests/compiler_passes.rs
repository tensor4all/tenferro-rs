use tenferro::compiler::{
    compile_to_exec, dot_decomposer, dot_dimension_sorter, reduction_simplification,
    transpose_folding,
};
use tenferro::exec::ExecOp;
use tenferro::stablehlo::{StableHloInstruction, StableHloOp, StableHloProgram};
use tenferro_ops::config::DotGeneralConfig;
use tenferro_tensor::DType;

// ============================================================================
// Helper: build a StableHloProgram from a list of instructions
// ============================================================================

fn make_program(
    instructions: Vec<StableHloInstruction>,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    n_slots: usize,
) -> StableHloProgram {
    StableHloProgram {
        instructions,
        input_slots,
        output_slots,
        n_slots,
    }
}

fn make_instr(
    op: StableHloOp,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
) -> StableHloInstruction {
    StableHloInstruction {
        op,
        input_slots,
        output_slots,
        dtype: DType::F64,
    }
}

// ============================================================================
// Pass 1: DotDimensionSorter
// ============================================================================

#[test]
fn test_dot_dimension_sorter_sorts_contracting() {
    // DotGeneral with lhs_contracting={3,2}, rhs_contracting={2,1}
    // These are consecutive-if-sorted but not sorted for lhs.
    // After sorting: lhs_contracting={2,3}, rhs_contracting={1,2}
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![3, 2],
        rhs_contracting_dims: vec![2, 1],
        lhs_batch_dims: vec![0],
        rhs_batch_dims: vec![0],
    };
    let instr = make_instr(StableHloOp::DotGeneral(config), vec![0, 1], vec![2]);
    let mut program = make_program(vec![instr], vec![0, 1], vec![2], 3);

    dot_dimension_sorter(&mut program);

    match &program.instructions[0].op {
        StableHloOp::DotGeneral(c) => {
            assert_eq!(c.lhs_contracting_dims, vec![2, 3]);
            assert_eq!(c.rhs_contracting_dims, vec![1, 2]);
        }
        _ => panic!("expected DotGeneral"),
    }
}

#[test]
fn test_dot_dimension_sorter_already_sorted() {
    // Already sorted: lhs_contracting={1}, rhs_contracting={0}
    // Should be unchanged.
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let instr = make_instr(StableHloOp::DotGeneral(config.clone()), vec![0, 1], vec![2]);
    let mut program = make_program(vec![instr], vec![0, 1], vec![2], 3);

    dot_dimension_sorter(&mut program);

    match &program.instructions[0].op {
        StableHloOp::DotGeneral(c) => {
            assert_eq!(c.lhs_contracting_dims, vec![1]);
            assert_eq!(c.rhs_contracting_dims, vec![0]);
        }
        _ => panic!("expected DotGeneral"),
    }
}

#[test]
fn test_dot_dimension_sorter_rhs_consecutive_unsorted() {
    // lhs_contracting={2,0} (not consecutive: max-min=2, len=2, gap)
    // rhs_contracting={3,2} (consecutive-if-sorted but not sorted)
    // After sorting by rhs: perm=[1,0], both get permuted.
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2, 0],
        rhs_contracting_dims: vec![3, 2],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let instr = make_instr(StableHloOp::DotGeneral(config), vec![0, 1], vec![2]);
    let mut program = make_program(vec![instr], vec![0, 1], vec![2], 3);

    dot_dimension_sorter(&mut program);

    match &program.instructions[0].op {
        StableHloOp::DotGeneral(c) => {
            // perm = argsort([3,2]) = [1,0]
            // apply_perm([2,0], [1,0]) = [0, 2]
            // apply_perm([3,2], [1,0]) = [2, 3]
            assert_eq!(c.lhs_contracting_dims, vec![0, 2]);
            assert_eq!(c.rhs_contracting_dims, vec![2, 3]);
        }
        _ => panic!("expected DotGeneral"),
    }
}

// ============================================================================
// Pass 2: ReductionSimplification
// ============================================================================

#[test]
fn test_reduction_simplification_noop() {
    // ReduceSum at index 0, DotGeneral at index 1.
    // Already in correct order — pass is a no-op validation.
    let reduce = make_instr(StableHloOp::ReduceSum { axes: vec![2] }, vec![0], vec![1]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_instr(StableHloOp::DotGeneral(config), vec![1, 2], vec![3]);
    let mut program = make_program(vec![reduce, dot], vec![0, 2], vec![3], 4);

    // Should not panic.
    reduction_simplification(&mut program);

    // Verify instructions are unchanged.
    assert_eq!(program.instructions.len(), 2);
    assert!(matches!(
        program.instructions[0].op,
        StableHloOp::ReduceSum { .. }
    ));
    assert!(matches!(
        program.instructions[1].op,
        StableHloOp::DotGeneral(_)
    ));
}

// ============================================================================
// Pass 4: TransposeFolding
// ============================================================================

#[test]
fn test_transpose_folding_absorbs_transpose_on_lhs() {
    // Program:
    //   slot 0: input A (shape [2,3])
    //   slot 1: input B (shape [2,3])
    //   slot 2 = Transpose(slot 0, perm=[1,0])  => A^T shape [3,2]
    //   slot 3 = DotGeneral(slot 2, slot 1, lhs_contract={1}, rhs_contract={0})
    //
    // After folding: DotGeneral reads slot 0 directly.
    //   lhs_contracting: perm[1] = 0  =>  {0}
    let transpose = make_instr(
        StableHloOp::Transpose { perm: vec![1, 0] },
        vec![0],
        vec![2],
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_instr(StableHloOp::DotGeneral(config), vec![2, 1], vec![3]);
    let mut program = make_program(vec![transpose, dot], vec![0, 1], vec![3], 4);

    transpose_folding(&mut program);

    // The DotGeneral should now read slot 0 (bypassing transpose).
    let dot_instr = &program.instructions[1];
    assert_eq!(dot_instr.input_slots[0], 0); // bypassed transpose
    assert_eq!(dot_instr.input_slots[1], 1);
    match &dot_instr.op {
        StableHloOp::DotGeneral(c) => {
            // Original lhs_contracting was {1}, perm={1,0}
            // New: perm[1] = 0, so lhs_contracting = {0}
            assert_eq!(c.lhs_contracting_dims, vec![0]);
            assert_eq!(c.rhs_contracting_dims, vec![0]);
        }
        _ => panic!("expected DotGeneral"),
    }
}

#[test]
fn test_transpose_folding_absorbs_transpose_on_rhs() {
    // slot 0: input A
    // slot 1: input B
    // slot 2 = Transpose(slot 1, perm=[1,0])  => B^T
    // slot 3 = DotGeneral(slot 0, slot 2, lhs_contract={1}, rhs_contract={0})
    //
    // After folding: rhs reads slot 1 directly.
    //   rhs_contracting: perm[0] = 1  =>  {1}
    let transpose = make_instr(
        StableHloOp::Transpose { perm: vec![1, 0] },
        vec![1],
        vec![2],
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_instr(StableHloOp::DotGeneral(config), vec![0, 2], vec![3]);
    let mut program = make_program(vec![transpose, dot], vec![0, 1], vec![3], 4);

    transpose_folding(&mut program);

    let dot_instr = &program.instructions[1];
    assert_eq!(dot_instr.input_slots[0], 0);
    assert_eq!(dot_instr.input_slots[1], 1); // bypassed transpose
    match &dot_instr.op {
        StableHloOp::DotGeneral(c) => {
            assert_eq!(c.lhs_contracting_dims, vec![1]);
            // perm[0] = 1, so rhs_contracting = {1}
            assert_eq!(c.rhs_contracting_dims, vec![1]);
        }
        _ => panic!("expected DotGeneral"),
    }
}

#[test]
fn test_transpose_folding_unfoldable_batch_changed() {
    // Transpose that changes batch dims => not foldable.
    // slot 0: input A (batch dim 0, but perm shuffles it)
    // slot 2 = Transpose(slot 0, perm=[1,0,2])  => changes dim 0
    // slot 3 = DotGeneral with batch_dims=[0]
    let transpose = make_instr(
        StableHloOp::Transpose {
            perm: vec![1, 0, 2],
        },
        vec![0],
        vec![2],
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![0],
        rhs_batch_dims: vec![0],
    };
    let dot = make_instr(StableHloOp::DotGeneral(config), vec![2, 1], vec![3]);
    let mut program = make_program(vec![transpose, dot], vec![0, 1], vec![3], 4);

    transpose_folding(&mut program);

    // Not foldable: DotGeneral still reads slot 2 (transpose output).
    let dot_instr = &program.instructions[1];
    assert_eq!(dot_instr.input_slots[0], 2); // unchanged
}

#[test]
fn test_transpose_folding_fixed_point() {
    // Two transposes feeding into the same DotGeneral.
    // Both should be folded in at most 2 iterations.
    let transpose_a = make_instr(
        StableHloOp::Transpose { perm: vec![1, 0] },
        vec![0],
        vec![2],
    );
    let transpose_b = make_instr(
        StableHloOp::Transpose { perm: vec![1, 0] },
        vec![1],
        vec![3],
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_instr(StableHloOp::DotGeneral(config), vec![2, 3], vec![4]);
    let mut program = make_program(vec![transpose_a, transpose_b, dot], vec![0, 1], vec![4], 5);

    transpose_folding(&mut program);

    let dot_instr = &program.instructions[2];
    // Both transposes folded: reads slots 0, 1 directly.
    assert_eq!(dot_instr.input_slots[0], 0);
    assert_eq!(dot_instr.input_slots[1], 1);
    match &dot_instr.op {
        StableHloOp::DotGeneral(c) => {
            // LHS: perm[1]=0 => contracting {0}
            assert_eq!(c.lhs_contracting_dims, vec![0]);
            // RHS: perm[0]=1 => contracting {1}
            assert_eq!(c.rhs_contracting_dims, vec![1]);
        }
        _ => panic!("expected DotGeneral"),
    }
}

// ============================================================================
// Pass 3: DotDecomposer
// ============================================================================

#[test]
fn test_dot_decomposer_canonical_noop() {
    // Already canonical: batch=[], 1 contracting dim, rank <= 2.
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let instr = make_instr(StableHloOp::DotGeneral(config.clone()), vec![0, 1], vec![2]);
    let mut program = make_program(vec![instr], vec![0, 1], vec![2], 3);

    dot_decomposer(&mut program);

    // Should be unchanged.
    assert_eq!(program.instructions.len(), 1);
    match &program.instructions[0].op {
        StableHloOp::DotGeneral(c) => {
            assert_eq!(c.lhs_contracting_dims, vec![1]);
            assert_eq!(c.rhs_contracting_dims, vec![0]);
        }
        _ => panic!("expected DotGeneral"),
    }
}

#[test]
fn test_dot_decomposer_with_leading_batch() {
    // Canonical with batch: batch=[0], 1 contracting dim.
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![1],
        lhs_batch_dims: vec![0],
        rhs_batch_dims: vec![0],
    };
    let instr = make_instr(StableHloOp::DotGeneral(config.clone()), vec![0, 1], vec![2]);
    let mut program = make_program(vec![instr], vec![0, 1], vec![2], 3);

    dot_decomposer(&mut program);

    // Already canonical — should be unchanged.
    assert_eq!(program.instructions.len(), 1);
}

#[test]
fn test_dot_decomposer_non_leading_batch() {
    // Non-canonical: batch_dims=[1] (not leading), 1 contracting dim.
    // For a rank-3 LHS with batch=1, contracting=2, non_contracting=0:
    //   target_order(LHS) = [1, 0, 2] (batch + non_contracting + contracting)
    // For a rank-3 RHS with batch=1, contracting=0, non_contracting=2:
    //   target_order(RHS) = [1, 0, 2] (batch + contracting + non_contracting)
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![1],
        rhs_batch_dims: vec![1],
    };
    let instr = make_instr(StableHloOp::DotGeneral(config), vec![0, 1], vec![2]);
    let mut program = make_program(vec![instr], vec![0, 1], vec![2], 3);

    dot_decomposer(&mut program);

    // Should have inserted transposes + new DotGeneral.
    // Check that the final DotGeneral now has leading batch dims.
    let dot_instr = program
        .instructions
        .iter()
        .find(|i| matches!(i.op, StableHloOp::DotGeneral(_)));
    assert!(dot_instr.is_some(), "DotGeneral should still exist");

    match &dot_instr.unwrap().op {
        StableHloOp::DotGeneral(c) => {
            assert_eq!(c.lhs_batch_dims, vec![0]);
            assert_eq!(c.rhs_batch_dims, vec![0]);
        }
        _ => panic!("expected DotGeneral"),
    }
}

// ============================================================================
// Full pipeline: compile_to_exec
// ============================================================================

#[test]
fn test_full_pipeline_matmul() {
    // Simple matmul "ij,jk->ik" as StableHLO:
    //   slot 0: A (input, shape [I, J])
    //   slot 1: B (input, shape [J, K])
    //   slot 2: DotGeneral(A, B, lhs_contract={1}, rhs_contract={0})
    //
    // All passes should be no-ops for this canonical case.
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_instr(StableHloOp::DotGeneral(config), vec![0, 1], vec![2]);
    let program = make_program(vec![dot], vec![0, 1], vec![2], 3);

    let exec = compile_to_exec(&program);

    // Should produce exactly 1 ExecOp::BatchedGemm.
    assert_eq!(exec.instructions.len(), 1);
    match &exec.instructions[0].op {
        ExecOp::BatchedGemm(c) => {
            assert_eq!(c.lhs_contracting_dims, vec![1]);
            assert_eq!(c.rhs_contracting_dims, vec![0]);
        }
        _ => panic!("expected BatchedGemm, got {:?}", exec.instructions[0].op),
    }
    assert_eq!(exec.input_slots, vec![0, 1]);
    assert_eq!(exec.output_slots, vec![2]);
}

#[test]
fn test_full_pipeline_transpose_matmul() {
    // "ji,jk->ik": Transpose(A) then DotGeneral
    //   slot 0: A (input, shape [J, I])
    //   slot 1: B (input, shape [J, K])
    //   slot 2: Transpose(A, perm=[1,0])  => shape [I, J]
    //   slot 3: DotGeneral(slot 2, slot 1, lhs_contract={1}, rhs_contract={0})
    //
    // TransposeFolding should absorb the Transpose into DotGeneral.
    let transpose = make_instr(
        StableHloOp::Transpose { perm: vec![1, 0] },
        vec![0],
        vec![2],
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_instr(StableHloOp::DotGeneral(config), vec![2, 1], vec![3]);
    let program = make_program(vec![transpose, dot], vec![0, 1], vec![3], 4);

    let exec = compile_to_exec(&program);

    // The BatchedGemm should now read slot 0 directly (transpose folded).
    let gemm_instr = exec
        .instructions
        .iter()
        .find(|i| matches!(i.op, ExecOp::BatchedGemm(_)));
    assert!(gemm_instr.is_some(), "should have BatchedGemm");

    let gemm = gemm_instr.unwrap();
    assert_eq!(gemm.input_slots[0], 0); // reads A directly
    assert_eq!(gemm.input_slots[1], 1);
    match &gemm.op {
        ExecOp::BatchedGemm(c) => {
            // perm[1] = 0, so lhs_contracting becomes {0}
            assert_eq!(c.lhs_contracting_dims, vec![0]);
            assert_eq!(c.rhs_contracting_dims, vec![0]);
        }
        _ => panic!("expected BatchedGemm"),
    }
}

#[test]
fn test_full_pipeline_reduce_then_matmul() {
    // "ijk,jl->il": ReduceSum(A, axes=[2]) then DotGeneral
    //   slot 0: A (input, shape [I, J, K])
    //   slot 1: B (input, shape [J, L])
    //   slot 2: ReduceSum(slot 0, axes=[2])  => shape [I, J]
    //   slot 3: DotGeneral(slot 2, slot 1, lhs_contract={1}, rhs_contract={0})
    let reduce = make_instr(StableHloOp::ReduceSum { axes: vec![2] }, vec![0], vec![2]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_instr(StableHloOp::DotGeneral(config), vec![2, 1], vec![3]);
    let program = make_program(vec![reduce, dot], vec![0, 1], vec![3], 4);

    let exec = compile_to_exec(&program);

    // Should have ReduceSum followed by BatchedGemm.
    assert_eq!(exec.instructions.len(), 2);
    assert!(matches!(exec.instructions[0].op, ExecOp::ReduceSum { .. }));
    assert!(matches!(exec.instructions[1].op, ExecOp::BatchedGemm(_)));
}
