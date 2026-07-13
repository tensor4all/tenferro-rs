use super::{
    algebraic_layout_simplifier, compile_std_to_exec, compile_std_to_exec_with_options,
    conj_sinking, dot_conj_folding, dot_decomposer, dot_dimension_sorter, eliminate_dead_code,
    populate_last_use, producer_index_by_slot, record_producer, resolve_slot_redirect,
    slot_use_counts, transpose_folding, CompilerOptions, OptimizerConfig,
};
use crate::exec::{ExecInstruction, ExecOp, ExecProgram};
use crate::{Error, GraphExecutor};
use computegraph::compile::{CompiledProgram, Instruction};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeExtent;
use tenferro_tensor::{DType, DotGeneralConfig};

#[path = "tests/dot_decomposer_tests.rs"]
mod dot_decomposer_tests;
#[path = "tests/shape_constraints.rs"]
mod shape_constraints;

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
        shape_guards: Vec::new(),
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
        output_shapes: vec![Vec::new(); output_slots.len()].into(),
        output_extents: vec![Vec::new(); output_slots.len()].into(),
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
            .collect::<Vec<_>>()
            .into(),
        output_shapes: output_shapes.into(),
        last_use: Vec::new(),
    }
}

fn make_std_instr(
    op: StdTensorOp,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
) -> Instruction<StdTensorOp> {
    Instruction {
        operation: op,
        inputs,
        outputs,
    }
}

#[test]
fn compile_default_options_match_default_entrypoint() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
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

    let dtypes = [DType::F64, DType::F64];
    let shapes = [dim_shape(&[2, 3]), dim_shape(&[3, 4])];
    let default = compile_std_to_exec(&program, &dtypes, &shapes).unwrap();
    let explicit =
        compile_std_to_exec_with_options(&program, &dtypes, &shapes, CompilerOptions::default())
            .unwrap();

    assert_eq!(default.instructions.len(), explicit.instructions.len());
    assert_eq!(default.n_slots, explicit.n_slots);
}

#[test]
fn compile_reports_missing_slot_metadata_as_error() {
    let program = CompiledProgram {
        instructions: vec![make_std_instr(StdTensorOp::Neg, vec![1], vec![2])],
        input_slots: vec![0],
        output_slots: vec![2],
        n_slots: 3,
    };

    let err = compile_std_to_exec(&program, &[DType::F64], &[dim_shape(&[2])]).unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidCompiledGraph { ref message }
            if message.contains("missing dtype for slot 1")
    ));
}

#[test]
fn compile_reports_incompatible_broadcast_as_error() {
    let program = CompiledProgram {
        instructions: vec![make_std_instr(StdTensorOp::Add, vec![0, 1], vec![2])],
        input_slots: vec![0, 1],
        output_slots: vec![2],
        n_slots: 3,
    };

    let err = compile_std_to_exec(
        &program,
        &[DType::F64, DType::F64],
        &[dim_shape(&[2]), dim_shape(&[3])],
    )
    .unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidCompiledGraph { ref message }
            if message.contains("incompatible Add/Mul broadcast dimensions: 2 and 3")
    ));
}

#[test]
fn resolve_slot_redirect_rejects_cycles() {
    let err = resolve_slot_redirect(0, &[1, 0]).unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidCompiledGraph { ref message }
            if message.contains("redirect cycle") && message.contains("slot 0")
    ));
}

#[test]
fn record_producer_rejects_out_of_range_output_slot() {
    let instr = make_exec_instr(ExecOp::Negate, vec![0], vec![3]);
    let mut producers = vec![None];

    let err = record_producer(&mut producers, &instr).unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidCompiledGraph { ref message }
            if message.contains("producer output slot 3")
    ));
}

#[test]
fn producer_index_by_slot_rejects_out_of_range_output_slot() {
    let instr = make_exec_instr(ExecOp::Negate, vec![0], vec![3]);
    let program = make_exec_program(vec![instr], vec![0], vec![], 1);

    let err = producer_index_by_slot(&program).unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidCompiledGraph { ref message }
            if message.contains("producer output slot 3")
    ));
}

#[test]
fn producer_index_by_slot_rejects_duplicate_output_slot() {
    let first = make_exec_instr(ExecOp::Negate, vec![0], vec![1]);
    let second = make_exec_instr(ExecOp::Negate, vec![0], vec![1]);
    let program = make_exec_program(vec![first, second], vec![0], vec![1], 2);

    let err = producer_index_by_slot(&program).unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidCompiledGraph { ref message }
            if message == "producer output slot 1 has duplicate producers at instructions 0 and 1"
    ));
}

#[test]
fn slot_use_counts_rejects_out_of_range_slots() {
    let instr = make_exec_instr(ExecOp::Negate, vec![2], vec![0]);
    let program = make_exec_program(vec![instr], vec![0], vec![], 1);

    let err = slot_use_counts(&program).unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidCompiledGraph { ref message }
            if message.contains("use-count input slot 2")
    ));
}

#[test]
fn populate_last_use_rejects_out_of_range_output_slot() {
    let mut program = make_exec_program(Vec::new(), vec![0], vec![2], 1);

    let err = populate_last_use(&mut program).unwrap_err();

    assert!(matches!(
        err,
        Error::InvalidCompiledGraph { ref message }
            if message.contains("last-use output slot 2")
    ));
}

#[test]
fn algebraic_layout_simplifier_removes_identity_transpose() {
    let transpose = make_exec_instr_with_meta(
        ExecOp::Transpose { perm: vec![0, 1] },
        vec![0],
        vec![1],
        DType::F64,
        vec![dim_shape(&[2, 3])],
    );
    let neg = make_exec_instr_with_meta(
        ExecOp::Negate,
        vec![1],
        vec![2],
        DType::F64,
        vec![dim_shape(&[2, 3])],
    );
    let mut program = make_exec_program(vec![transpose, neg], vec![0], vec![2], 3);

    algebraic_layout_simplifier(&mut program, &[dim_shape(&[2, 3])]).unwrap();
    eliminate_dead_code(&mut program);

    assert_eq!(program.instructions.len(), 1);
    assert!(matches!(program.instructions[0].op, ExecOp::Negate));
    assert_eq!(program.instructions[0].input_slots, vec![0]);
}

#[test]
fn algebraic_layout_simplifier_composes_adjacent_inverse_transposes() {
    let transpose_a = make_exec_instr_with_meta(
        ExecOp::Transpose {
            perm: vec![1, 2, 0],
        },
        vec![0],
        vec![1],
        DType::F64,
        vec![dim_shape(&[3, 4, 2])],
    );
    let transpose_b = make_exec_instr_with_meta(
        ExecOp::Transpose {
            perm: vec![2, 0, 1],
        },
        vec![1],
        vec![2],
        DType::F64,
        vec![dim_shape(&[2, 3, 4])],
    );
    let mut program = make_exec_program(vec![transpose_a, transpose_b], vec![0], vec![2], 3);

    algebraic_layout_simplifier(&mut program, &[dim_shape(&[2, 3, 4])]).unwrap();
    eliminate_dead_code(&mut program);

    assert_eq!(program.output_slots, vec![0]);
    assert!(program.instructions.is_empty());
}

#[test]
fn algebraic_layout_simplifier_removes_identity_reshape() {
    let reshape = make_exec_instr_with_meta(
        ExecOp::Reshape {
            shape: DimExpr::input_shape(0, 2),
        },
        vec![0],
        vec![1],
        DType::F64,
        vec![dim_shape(&[2, 3])],
    );
    let neg = make_exec_instr_with_meta(
        ExecOp::Negate,
        vec![1],
        vec![2],
        DType::F64,
        vec![dim_shape(&[2, 3])],
    );
    let mut program = make_exec_program(vec![reshape, neg], vec![0], vec![2], 3);

    algebraic_layout_simplifier(&mut program, &[dim_shape(&[2, 3])]).unwrap();
    eliminate_dead_code(&mut program);

    assert_eq!(program.instructions.len(), 1);
    assert!(matches!(program.instructions[0].op, ExecOp::Negate));
    assert_eq!(program.instructions[0].input_slots, vec![0]);
}

#[test]
fn algebraic_layout_simplifier_keeps_rank_reducing_reshape() {
    let reshape = make_exec_instr_with_meta(
        ExecOp::Reshape {
            shape: DimExpr::input_shape(0, 2),
        },
        vec![0],
        vec![1],
        DType::F64,
        vec![dim_shape(&[2, 3])],
    );
    let neg = make_exec_instr_with_meta(
        ExecOp::Negate,
        vec![1],
        vec![2],
        DType::F64,
        vec![dim_shape(&[2, 3])],
    );
    let mut program = make_exec_program(vec![reshape, neg], vec![0], vec![2], 3);

    algebraic_layout_simplifier(&mut program, &[dim_shape(&[2, 3, 1])]).unwrap();
    eliminate_dead_code(&mut program);

    assert_eq!(program.instructions.len(), 2);
    assert!(matches!(program.instructions[0].op, ExecOp::Reshape { .. }));
    assert_eq!(program.instructions[1].input_slots, vec![1]);
}

#[test]
fn algebraic_layout_simplifier_has_no_repeated_whole_program_fixpoint_loop() {
    let source = include_str!("mod.rs");
    let (_, after_start) = source
        .split_once("pub(crate) fn algebraic_layout_simplifier(")
        .expect("algebraic_layout_simplifier should exist");
    let (simplifier_body, _) = after_start
        .split_once("fn algebraic_layout_simplifier_one_pass(")
        .expect("one-pass helper should follow the public simplifier wrapper");

    assert!(
        !simplifier_body.contains("loop {"),
        "algebraic_layout_simplifier must not recompute producer/use/shape tables in a whole-program fixpoint loop"
    );
}

#[test]
fn optimizer_config_has_no_duplicate_layout_chain_transpose_folding_flag() {
    assert!(
        !include_str!("options.rs").contains("layout_chain_transpose_folding"),
        "layout_chain_transpose_folding was a duplicate transpose_folding pass and should not remain configurable"
    );
    assert!(
        !include_str!("optimizer/mod.rs").contains("layout_chain_transpose_folding"),
        "optimizer pipeline should not run a duplicate layout-chain transpose pass"
    );
    assert!(
        !include_str!("mod.rs").contains("fn layout_chain_transpose_folding"),
        "compiler should not keep a duplicate layout_chain_transpose_folding implementation"
    );
}

#[test]
fn transpose_folding_ignores_non_identity_reshape_chain() {
    let transpose = make_exec_instr_with_meta(
        ExecOp::Transpose { perm: vec![1, 0] },
        vec![0],
        vec![2],
        DType::F64,
        vec![dim_shape(&[3, 2])],
    );
    let reshape = make_exec_instr_with_meta(
        ExecOp::Reshape {
            shape: dim_shape(&[6]),
        },
        vec![2],
        vec![3],
        DType::F64,
        vec![dim_shape(&[6])],
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let dot = make_exec_instr_with_meta(
        ExecOp::DotGeneral(config.clone()),
        vec![3, 1],
        vec![4],
        DType::F64,
        vec![dim_shape(&[4])],
    );
    let mut program = make_exec_program(vec![transpose, reshape, dot], vec![0, 1], vec![4], 5);

    transpose_folding(&mut program);

    assert_eq!(program.instructions[1].input_slots, vec![2]);
    assert_eq!(program.instructions[2].input_slots, vec![3, 1]);
    match &program.instructions[2].op {
        ExecOp::DotGeneral(actual) => assert_eq!(actual, &config),
        other => panic!("expected DotGeneral, got {other:?}"),
    }
}

#[test]
fn dot_decomposer_is_disabled_by_default_and_opt_in() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
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
    let dtypes = [DType::F64, DType::F64];
    let shapes = [dim_shape(&[2, 3, 4]), dim_shape(&[4, 5])];

    let default_exec = compile_std_to_exec(&program, &dtypes, &shapes).unwrap();
    assert_eq!(
        default_exec
            .instructions
            .iter()
            .filter(|instr| matches!(instr.op, ExecOp::DotGeneral(_)))
            .count(),
        1
    );
    assert!(default_exec
        .instructions
        .iter()
        .all(|instr| !matches!(instr.op, ExecOp::Reshape { .. })));

    let decomposed = compile_std_to_exec_with_options(
        &program,
        &dtypes,
        &shapes,
        CompilerOptions {
            optimizer: OptimizerConfig {
                dot_decomposer: true,
                ..OptimizerConfig::default()
            },
        },
    )
    .unwrap();
    assert!(decomposed
        .instructions
        .iter()
        .any(|instr| matches!(instr.op, ExecOp::Reshape { .. })));
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

    conj_sinking(&mut program, &[DType::C64], &[dim_shape(&[2, 3])]).unwrap();
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

    conj_sinking(&mut program, &[DType::C64], &[dim_shape(&[2])]).unwrap();
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

    conj_sinking(&mut program, &[DType::I64], &[dim_shape(&[2])]).unwrap();

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

    conj_sinking(&mut program, &[DType::F64], &[dim_shape(&[2])]).unwrap();

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
    )
    .unwrap();
    transpose_folding(&mut program);
    dot_conj_folding(&mut program).unwrap();
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

    dot_conj_folding(&mut program).unwrap();
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
fn dot_conj_folding_preserves_rank_reducing_reshape_operand() {
    let lhs_conj = make_exec_instr_with_meta(
        ExecOp::Conj,
        vec![0],
        vec![2],
        DType::C64,
        vec![dim_shape(&[2, 3, 1])],
    );
    let reshape = make_exec_instr_with_meta(
        ExecOp::Reshape {
            shape: dim_shape(&[2, 3]),
        },
        vec![2],
        vec![3],
        DType::C64,
        vec![dim_shape(&[2, 3])],
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
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
    let mut program = make_exec_program(vec![lhs_conj, reshape, dot], vec![0, 1], vec![4], 5);

    dot_conj_folding(&mut program).unwrap();
    eliminate_dead_code(&mut program);

    assert_eq!(program.instructions.len(), 2);
    assert!(matches!(program.instructions[0].op, ExecOp::Reshape { .. }));
    assert_eq!(program.instructions[0].input_slots, vec![0]);
    assert_eq!(program.instructions[1].input_slots, vec![3, 1]);
    assert!(matches!(
        program.instructions[1].op,
        ExecOp::DotGeneralWithConj {
            lhs_conj: true,
            rhs_conj: false,
            ..
        }
    ));
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
    )
    .unwrap();

    assert_eq!(exec.instructions.len(), 1);
    match &exec.instructions[0].op {
        ExecOp::DotGeneral(config) => {
            assert_eq!(config.lhs_contracting_dims, vec![1]);
            assert_eq!(config.rhs_contracting_dims, vec![0]);
        }
        other => panic!("expected DotGeneral, got {other:?}"),
    }
    assert_eq!(
        exec.instructions[0].output_shapes.as_slice(),
        &[dim_shape(&[2, 4])]
    );
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
    )
    .unwrap();

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
    assert_eq!(
        exec.instructions[0].output_shapes.as_slice(),
        &[dim_shape(&[2, 4])]
    );
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
    )
    .unwrap();

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
            .output_shapes
            .as_slice(),
        &[dim_shape(&[2, 4, 5])]
    );
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
    use crate::{Tensor, TypedTensor};
    use tenferro_cpu::CpuBackend;

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
    )
    .unwrap();

    // Build concrete inputs: LHS = sequential 0..24, reshaped as [2, 3, 4];
    //                       RHS = sequential 0..20, reshaped as [4, 5].
    let lhs_data: Vec<f64> = (0..24).map(|x| x as f64).collect();
    let rhs_data: Vec<f64> = (0..20).map(|x| x as f64).collect();
    let lhs = Tensor::F64(
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3, 4], lhs_data.clone()).unwrap(),
    );
    let rhs =
        Tensor::F64(TypedTensor::<f64>::from_vec_col_major(vec![4, 5], rhs_data.clone()).unwrap());

    let mut executor = GraphExecutor::new(CpuBackend::default());
    let mut outputs = executor
        .eval_exec_ir(&exec, vec![lhs, rhs])
        .expect("executing decomposed program must not fail");
    let out = outputs.remove(0);
    let typed = match &out {
        Tensor::F64(inner) => inner,
        other => panic!("expected F64 tensor, got {other:?}"),
    };
    assert_eq!(typed.shape(), &[2, 3, 5]);

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
    for (i, (got, want)) in typed
        .host_data()
        .unwrap()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert!(
            (got - want).abs() < 1e-9,
            "index {i}: got {got}, expected {want}"
        );
    }
}
