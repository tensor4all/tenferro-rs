use computegraph::compile::{CompiledProgram, Instruction};
use tenferro::compiler::{
    compile_std_to_exec, conj_sinking, dot_conj_folding, dot_decomposer, dot_dimension_sorter,
    eliminate_dead_code, transpose_folding,
};
use tenferro::exec::{ExecInstruction, ExecOp, ExecProgram};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeExtent;
use tenferro_tensor::{DType, DotGeneralConfig};

fn dim_shape(shape: &[usize]) -> Vec<DimExpr> {
    DimExpr::from_concrete(shape)
}

fn exact_extents(shape: &[DimExpr]) -> Vec<ShapeExtent<DimExpr>> {
    shape.iter().cloned().map(ShapeExtent::exact).collect()
}

fn make_exec_program(
    instructions: Vec<ExecInstruction>,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    n_slots: usize,
) -> ExecProgram {
    ExecProgram {
        instructions,
        input_slots,
        output_slots,
        n_slots,
    }
}

fn make_exec_instr(
    op: ExecOp,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
) -> ExecInstruction {
    ExecInstruction {
        op,
        input_slots,
        output_slots: output_slots.clone(),
        dtype: DType::F64,
        output_shapes: vec![Vec::new(); output_slots.len()],
        output_extents: vec![Vec::new(); output_slots.len()],
        last_use: Vec::new(),
    }
}

fn make_exec_instr_with_meta(
    op: ExecOp,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    dtype: DType,
    output_shapes: Vec<Vec<DimExpr>>,
) -> ExecInstruction {
    ExecInstruction {
        op,
        input_slots,
        output_slots,
        dtype,
        output_extents: output_shapes
            .iter()
            .map(|shape| exact_extents(shape))
            .collect(),
        output_shapes,
        last_use: Vec::new(),
    }
}

fn make_std_instr(
    op: StdTensorOp,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
) -> Instruction<StdTensorOp> {
    Instruction {
        op,
        inputs,
        outputs,
    }
}

#[test]
fn test_dot_dimension_sorter_sorts_contracting() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![3, 2],
        rhs_contracting_dims: vec![2, 1],
        lhs_batch_dims: vec![0],
        rhs_batch_dims: vec![0],
    };
    let instr = make_exec_instr(ExecOp::DotGeneral(config), vec![0, 1], vec![2]);
    let mut program = make_exec_program(vec![instr], vec![0, 1], vec![2], 3);

    dot_dimension_sorter(&mut program);

    match &program.instructions[0].op {
        ExecOp::DotGeneral(config) => {
            assert_eq!(config.lhs_contracting_dims, vec![2, 3]);
            assert_eq!(config.rhs_contracting_dims, vec![1, 2]);
        }
        _ => panic!("expected DotGeneral"),
    }
}

#[test]
fn test_dot_dimension_sorter_already_sorted() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let instr = make_exec_instr(ExecOp::DotGeneral(config.clone()), vec![0, 1], vec![2]);
    let mut program = make_exec_program(vec![instr], vec![0, 1], vec![2], 3);

    dot_dimension_sorter(&mut program);

    match &program.instructions[0].op {
        ExecOp::DotGeneral(actual) => assert_eq!(actual, &config),
        _ => panic!("expected DotGeneral"),
    }
}

#[test]
fn test_dot_dimension_sorter_rhs_consecutive_unsorted() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2, 0],
        rhs_contracting_dims: vec![3, 2],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let instr = make_exec_instr(ExecOp::DotGeneral(config), vec![0, 1], vec![2]);
    let mut program = make_exec_program(vec![instr], vec![0, 1], vec![2], 3);

    dot_dimension_sorter(&mut program);

    match &program.instructions[0].op {
        ExecOp::DotGeneral(actual) => {
            assert_eq!(actual.lhs_contracting_dims, vec![0, 2]);
            assert_eq!(actual.rhs_contracting_dims, vec![2, 3]);
        }
        _ => panic!("expected DotGeneral"),
    }
}

#[test]
fn test_transpose_folding_absorbs_transpose_on_lhs() {
    let transpose = make_exec_instr(ExecOp::Transpose { perm: vec![1, 0] }, vec![0], vec![2]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_exec_instr(ExecOp::DotGeneral(config), vec![2, 1], vec![3]);
    let mut program = make_exec_program(vec![transpose, dot], vec![0, 1], vec![3], 4);

    transpose_folding(&mut program);

    let dot_instr = &program.instructions[1];
    assert_eq!(dot_instr.input_slots, vec![0, 1]);
    match &dot_instr.op {
        ExecOp::DotGeneral(config) => {
            assert_eq!(config.lhs_contracting_dims, vec![0]);
            assert_eq!(config.rhs_contracting_dims, vec![0]);
        }
        _ => panic!("expected DotGeneral"),
    }
}

#[test]
fn test_transpose_folding_absorbs_transpose_on_rhs() {
    let transpose = make_exec_instr(ExecOp::Transpose { perm: vec![1, 0] }, vec![1], vec![2]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_exec_instr(ExecOp::DotGeneral(config), vec![0, 2], vec![3]);
    let mut program = make_exec_program(vec![transpose, dot], vec![0, 1], vec![3], 4);

    transpose_folding(&mut program);

    let dot_instr = &program.instructions[1];
    assert_eq!(dot_instr.input_slots, vec![0, 1]);
    match &dot_instr.op {
        ExecOp::DotGeneral(config) => {
            assert_eq!(config.lhs_contracting_dims, vec![1]);
            assert_eq!(config.rhs_contracting_dims, vec![1]);
        }
        _ => panic!("expected DotGeneral"),
    }
}

#[test]
fn test_transpose_folding_rejects_free_dim_reorder() {
    let transpose = make_exec_instr(
        ExecOp::Transpose {
            perm: vec![1, 0, 2, 3],
        },
        vec![0],
        vec![2],
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![3],
        rhs_batch_dims: vec![1],
    };
    let dot = make_exec_instr(ExecOp::DotGeneral(config.clone()), vec![2, 1], vec![3]);
    let mut program = make_exec_program(vec![transpose, dot], vec![0, 1], vec![3], 4);

    transpose_folding(&mut program);

    let dot_instr = &program.instructions[1];
    assert_eq!(dot_instr.input_slots, vec![2, 1]);
    match &dot_instr.op {
        ExecOp::DotGeneral(actual) => assert_eq!(actual, &config),
        _ => panic!("expected DotGeneral"),
    }
}

#[test]
fn test_transpose_folding_fixed_point() {
    let transpose_a = make_exec_instr(ExecOp::Transpose { perm: vec![1, 0] }, vec![0], vec![2]);
    let transpose_b = make_exec_instr(ExecOp::Transpose { perm: vec![1, 0] }, vec![1], vec![3]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_exec_instr(ExecOp::DotGeneral(config), vec![2, 3], vec![4]);
    let mut program =
        make_exec_program(vec![transpose_a, transpose_b, dot], vec![0, 1], vec![4], 5);

    transpose_folding(&mut program);

    let dot_instr = &program.instructions[2];
    assert_eq!(dot_instr.input_slots, vec![0, 1]);
    match &dot_instr.op {
        ExecOp::DotGeneral(config) => {
            assert_eq!(config.lhs_contracting_dims, vec![0]);
            assert_eq!(config.rhs_contracting_dims, vec![1]);
        }
        _ => panic!("expected DotGeneral"),
    }
}

#[test]
fn test_conj_sinking_pushes_through_transpose() {
    let transpose = make_exec_instr_with_meta(
        ExecOp::Transpose { perm: vec![1, 0] },
        vec![0],
        vec![1],
        DType::C64,
        vec![dim_shape(&[3, 2])],
    );
    let conj = make_exec_instr_with_meta(
        ExecOp::Conj,
        vec![1],
        vec![2],
        DType::C64,
        vec![dim_shape(&[3, 2])],
    );
    let mut program = make_exec_program(vec![transpose, conj], vec![0], vec![2], 3);

    conj_sinking(&mut program, &[DType::C64], &[dim_shape(&[2, 3])]);
    eliminate_dead_code(&mut program);

    assert_eq!(program.instructions.len(), 2);
    assert!(matches!(program.instructions[0].op, ExecOp::Conj));
    assert_eq!(program.instructions[0].input_slots, vec![0]);
    match &program.instructions[1].op {
        ExecOp::Transpose { perm } => {
            assert_eq!(perm, &vec![1, 0]);
            assert_eq!(
                program.instructions[1].input_slots,
                program.instructions[0].output_slots
            );
            assert_eq!(program.instructions[1].output_slots, vec![2]);
        }
        other => panic!("expected Transpose after sinking, got {other:?}"),
    }
}

#[test]
fn test_conj_sinking_cancels_double_conj() {
    let conj_a = make_exec_instr_with_meta(
        ExecOp::Conj,
        vec![0],
        vec![1],
        DType::C64,
        vec![dim_shape(&[2])],
    );
    let conj_b = make_exec_instr_with_meta(
        ExecOp::Conj,
        vec![1],
        vec![2],
        DType::C64,
        vec![dim_shape(&[2])],
    );
    let mut program = make_exec_program(vec![conj_a, conj_b], vec![0], vec![2], 3);

    conj_sinking(&mut program, &[DType::C64], &[dim_shape(&[2])]);
    eliminate_dead_code(&mut program);

    assert_eq!(program.output_slots, vec![0]);
    assert!(program.instructions.is_empty());
}

#[test]
fn test_conj_sinking_does_not_push_convert_onto_i64() {
    let convert_i64_to_f64 = make_exec_instr_with_meta(
        ExecOp::Convert { to: DType::F64 },
        vec![0],
        vec![1],
        DType::F64,
        vec![dim_shape(&[2])],
    );
    let conj = make_exec_instr_with_meta(
        ExecOp::Conj,
        vec![1],
        vec![2],
        DType::F64,
        vec![dim_shape(&[2])],
    );
    let mut program = make_exec_program(vec![convert_i64_to_f64, conj], vec![0], vec![2], 3);

    conj_sinking(&mut program, &[DType::I64], &[dim_shape(&[2])]);

    assert_eq!(program.instructions.len(), 2);
    assert!(matches!(
        program.instructions[0].op,
        ExecOp::Convert { to: DType::F64 }
    ));
    assert!(matches!(program.instructions[1].op, ExecOp::Conj));
    assert_eq!(program.instructions[1].input_slots, vec![1]);
}

#[test]
fn test_conj_sinking_does_not_push_convert_to_i64() {
    let convert_f64_to_i64 = make_exec_instr_with_meta(
        ExecOp::Convert { to: DType::I64 },
        vec![0],
        vec![1],
        DType::I64,
        vec![dim_shape(&[2])],
    );
    let conj = make_exec_instr_with_meta(
        ExecOp::Conj,
        vec![1],
        vec![2],
        DType::I64,
        vec![dim_shape(&[2])],
    );
    let mut program = make_exec_program(vec![convert_f64_to_i64, conj], vec![0], vec![2], 3);

    conj_sinking(&mut program, &[DType::F64], &[dim_shape(&[2])]);

    assert_eq!(program.instructions.len(), 2);
    assert!(matches!(
        program.instructions[0].op,
        ExecOp::Convert { to: DType::I64 }
    ));
    assert!(matches!(program.instructions[1].op, ExecOp::Conj));
    assert_eq!(program.instructions[1].input_slots, vec![1]);
}

#[test]
fn test_conj_sinking_preserves_transpose_folding_path_to_dot_conj() {
    let transpose = make_exec_instr_with_meta(
        ExecOp::Transpose { perm: vec![1, 0] },
        vec![0],
        vec![2],
        DType::C64,
        vec![dim_shape(&[3, 2])],
    );
    let conj = make_exec_instr_with_meta(
        ExecOp::Conj,
        vec![2],
        vec![3],
        DType::C64,
        vec![dim_shape(&[3, 2])],
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_exec_instr_with_meta(
        ExecOp::DotGeneral(config),
        vec![3, 1],
        vec![4],
        DType::C64,
        vec![dim_shape(&[2, 4])],
    );
    let mut program = make_exec_program(vec![transpose, conj, dot], vec![0, 1], vec![4], 5);

    conj_sinking(
        &mut program,
        &[DType::C64, DType::C64],
        &[dim_shape(&[2, 3]), dim_shape(&[3, 4])],
    );
    transpose_folding(&mut program);
    dot_conj_folding(&mut program);
    eliminate_dead_code(&mut program);

    assert_eq!(program.instructions.len(), 1);
    let dot = &program.instructions[0];
    assert_eq!(dot.input_slots, vec![0, 1]);
    match &dot.op {
        ExecOp::DotGeneralWithConj {
            config,
            lhs_conj,
            rhs_conj,
        } => {
            assert_eq!(config.lhs_contracting_dims, vec![1]);
            assert_eq!(config.rhs_contracting_dims, vec![0]);
            assert!(*lhs_conj);
            assert!(!*rhs_conj);
        }
        other => panic!("expected DotGeneralWithConj, got {other:?}"),
    }
}

#[test]
fn test_dot_conj_folding_absorbs_both_operands() {
    let lhs_conj = make_exec_instr_with_meta(
        ExecOp::Conj,
        vec![0],
        vec![2],
        DType::C64,
        vec![dim_shape(&[2, 3])],
    );
    let rhs_conj = make_exec_instr_with_meta(
        ExecOp::Conj,
        vec![1],
        vec![3],
        DType::C64,
        vec![dim_shape(&[3, 4])],
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_exec_instr_with_meta(
        ExecOp::DotGeneral(config),
        vec![2, 3],
        vec![4],
        DType::C64,
        vec![dim_shape(&[2, 4])],
    );
    let mut program = make_exec_program(vec![lhs_conj, rhs_conj, dot], vec![0, 1], vec![4], 5);

    dot_conj_folding(&mut program);
    eliminate_dead_code(&mut program);

    assert_eq!(program.instructions.len(), 1);
    assert_eq!(program.instructions[0].input_slots, vec![0, 1]);
    match &program.instructions[0].op {
        ExecOp::DotGeneralWithConj {
            lhs_conj, rhs_conj, ..
        } => {
            assert!(*lhs_conj);
            assert!(*rhs_conj);
        }
        other => panic!("expected DotGeneralWithConj, got {other:?}"),
    }
}

#[test]
fn test_full_pipeline_matmul() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let program = CompiledProgram {
        instructions: vec![make_std_instr(
            StdTensorOp::DotGeneral {
                config: config.clone(),
            },
            vec![0, 1],
            vec![2],
        )],
        input_slots: vec![0, 1],
        output_slots: vec![2],
        n_slots: 3,
    };

    let exec = compile_std_to_exec(
        &program,
        &[DType::F64, DType::F64],
        &[dim_shape(&[2, 3]), dim_shape(&[3, 4])],
    );

    assert_eq!(exec.instructions.len(), 1);
    match &exec.instructions[0].op {
        ExecOp::DotGeneral(config) => {
            assert_eq!(config.lhs_contracting_dims, vec![1]);
            assert_eq!(config.rhs_contracting_dims, vec![0]);
        }
        other => panic!("expected DotGeneral, got {other:?}"),
    }
    assert_eq!(exec.instructions[0].output_shapes, vec![dim_shape(&[2, 4])]);
}

#[test]
fn test_full_pipeline_transpose_matmul() {
    let transpose = make_std_instr(
        StdTensorOp::Transpose { perm: vec![1, 0] },
        vec![0],
        vec![2],
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_std_instr(
        StdTensorOp::DotGeneral {
            config: config.clone(),
        },
        vec![2, 1],
        vec![3],
    );
    let program = CompiledProgram {
        instructions: vec![transpose, dot],
        input_slots: vec![0, 1],
        output_slots: vec![3],
        n_slots: 4,
    };

    let exec = compile_std_to_exec(
        &program,
        &[DType::F64, DType::F64],
        &[dim_shape(&[3, 2]), dim_shape(&[3, 4])],
    );

    // `transpose_folding` absorbs the Transpose into DotGeneral dim-numbers.
    // The compiled pipeline leaves the non-canonical DotGeneral for the backend
    // strided GEMM path instead of materializing a canonical layout.
    let dot_instr = exec
        .instructions
        .iter()
        .find(|instr| matches!(instr.op, ExecOp::DotGeneral(_)))
        .expect("expected DotGeneral after direct lowering");
    match &dot_instr.op {
        ExecOp::DotGeneral(config) => {
            assert_eq!(config.lhs_contracting_dims, vec![0]);
            assert_eq!(config.rhs_contracting_dims, vec![0]);
            assert!(config.lhs_batch_dims.is_empty());
            assert!(config.rhs_batch_dims.is_empty());
        }
        _ => panic!("expected DotGeneral"),
    }

    assert_eq!(dot_instr.input_slots[0], 0);
    assert_eq!(dot_instr.input_slots[1], 1);
    let transpose_count = exec
        .instructions
        .iter()
        .filter(|i| matches!(i.op, ExecOp::Transpose { .. }))
        .count();
    assert_eq!(transpose_count, 0, "dead-code elimination failed");
    assert_eq!(exec.instructions[0].output_shapes, vec![dim_shape(&[2, 4])]);
}

#[test]
fn test_full_pipeline_dot_absorbs_conj_without_layout_materialization() {
    let conj = make_std_instr(StdTensorOp::Conj, vec![1], vec![2]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_std_instr(StdTensorOp::DotGeneral { config }, vec![0, 2], vec![3]);
    let program = CompiledProgram {
        instructions: vec![conj, dot],
        input_slots: vec![0, 1],
        output_slots: vec![3],
        n_slots: 4,
    };

    let exec = compile_std_to_exec(
        &program,
        &[DType::C64, DType::C64],
        &[dim_shape(&[2, 3]), dim_shape(&[3, 4, 5])],
    );

    assert!(
        exec.instructions
            .iter()
            .all(|instr| !matches!(instr.op, ExecOp::Conj)),
        "Conj should be folded into the decomposed DotGeneral"
    );
    let dot_instr = exec
        .instructions
        .iter()
        .find(|instr| matches!(instr.op, ExecOp::DotGeneralWithConj { .. }))
        .expect("expected DotGeneralWithConj after decomposing conj input");
    match &dot_instr.op {
        ExecOp::DotGeneralWithConj {
            config,
            lhs_conj,
            rhs_conj,
        } => {
            assert_eq!(config.lhs_contracting_dims, vec![1]);
            assert_eq!(config.rhs_contracting_dims, vec![0]);
            assert!(!*lhs_conj);
            assert!(*rhs_conj);
        }
        _ => unreachable!(),
    }
    assert_eq!(
        exec.instructions
            .last()
            .expect("compiled program should have an output instruction")
            .output_shapes,
        vec![dim_shape(&[2, 4, 5])]
    );
}

// ----------------------------------------------------------------------------
// DotDecomposer tests
// ----------------------------------------------------------------------------

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
        input_slots,
        output_slots: vec![output_slot],
        dtype: DType::F64,
        output_extents: vec![exact_extents(&output_shape)],
        output_shapes: vec![output_shape],
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

    dot_decomposer(&mut program, &[dim_shape(&[4, 5]), dim_shape(&[5, 6])]);

    assert_eq!(program.instructions.len(), 1);
    assert_eq!(program.n_slots, 3);
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
    );

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

    dot_decomposer(&mut program, &[dim_shape(&[2, 3, 4]), dim_shape(&[4, 5])]);

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
        input_slots: vec![0, 1],
        output_slots: vec![2],
        dtype: DType::F64,
        output_shapes: vec![truncate_shape.clone()],
        output_extents: vec![vec![
            ShapeExtent::upper_bound(DimExpr::Const(5)),
            ShapeExtent::exact(DimExpr::Const(3)),
            ShapeExtent::exact(DimExpr::Const(4)),
        ]],
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
    );

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
    );

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
    );

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

#[test]
fn test_eliminate_dead_code_removes_unused_transpose() {
    // A Transpose whose output is never consumed must be removed.
    let transpose = make_exec_instr(ExecOp::Transpose { perm: vec![1, 0] }, vec![0], vec![2]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_exec_instr(ExecOp::DotGeneral(config), vec![0, 1], vec![3]);
    let mut program = make_exec_program(vec![transpose, dot], vec![0, 1], vec![3], 4);

    eliminate_dead_code(&mut program);

    assert_eq!(program.instructions.len(), 1);
    assert!(matches!(program.instructions[0].op, ExecOp::DotGeneral(_)));
}

#[test]
fn test_full_pipeline_multi_free_dim_decomp_runs_correctly() {
    // End-to-end: build a traced graph with multiple free dims, compile,
    // and run it through the CPU backend. Compare the output to the
    // equivalent matmul result.
    use tenferro::{CpuBackend, Tensor, TypedTensor};

    // LHS [M1, M2, K] * RHS [K, N] => [M1, M2, N]. The compile pipeline
    // should emit a canonical DotGeneral + output Reshape. Run end-to-end
    // to verify numerical correctness.
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let program = CompiledProgram {
        instructions: vec![make_std_instr(
            StdTensorOp::DotGeneral { config },
            vec![0, 1],
            vec![2],
        )],
        input_slots: vec![0, 1],
        output_slots: vec![2],
        n_slots: 3,
    };

    let exec = compile_std_to_exec(
        &program,
        &[DType::F64, DType::F64],
        &[dim_shape(&[2, 3, 4]), dim_shape(&[4, 5])],
    );

    // Build concrete inputs: LHS = sequential 0..24, reshaped as [2, 3, 4];
    //                       RHS = sequential 0..20, reshaped as [4, 5].
    let lhs_data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let rhs_data: Vec<f64> = (0..20).map(|x| x as f64).collect();
    let lhs = Tensor::F64(TypedTensor::<f64>::from_vec(
        vec![2, 3, 4],
        lhs_data.clone(),
    ));
    let rhs = Tensor::F64(TypedTensor::<f64>::from_vec(vec![4, 5], rhs_data.clone()));

    let mut backend = CpuBackend::default();
    let mut outputs = tenferro::exec::eval_exec_ir(&mut backend, &exec, vec![lhs, rhs])
        .expect("executing decomposed program must not fail");
    let out = outputs.remove(0);
    let typed = match &out {
        Tensor::F64(inner) => inner,
        other => panic!("expected F64 tensor, got {other:?}"),
    };
    assert_eq!(typed.shape, vec![2, 3, 5]);

    // Reference: column-major (tenferro storage convention) matmul.
    // For LHS shape [2, 3, 4], flat index = m1 + 2*m2 + 6*k.
    // For RHS shape [4, 5], flat index = k + 4*n.
    // For output shape [2, 3, 5], flat index = m1 + 2*m2 + 6*n.
    let mut expected = vec![0.0f64; 2 * 3 * 5];
    for m1 in 0..2 {
        for m2 in 0..3 {
            for n in 0..5 {
                let mut acc = 0.0;
                for k in 0..4 {
                    acc += lhs_data[m1 + 2 * m2 + 6 * k] * rhs_data[k + 4 * n];
                }
                expected[m1 + 2 * m2 + 6 * n] = acc;
            }
        }
    }
    for (i, (got, want)) in typed.host_data().iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-9,
            "index {i}: got {got}, expected {want}"
        );
    }
}
