use computegraph::compile::{CompiledProgram, Instruction};
use tenferro::compiler::{compile_std_to_exec, dot_dimension_sorter, transpose_folding};
use tenferro::exec::{ExecInstruction, ExecOp, ExecProgram};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, DotGeneralConfig};

fn dim_shape(shape: &[usize]) -> Vec<DimExpr> {
    DimExpr::from_concrete(shape)
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
                lhs_rank: 2,
                rhs_rank: 2,
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
            lhs_rank: 2,
            rhs_rank: 2,
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

    let dot_instr = exec
        .instructions
        .iter()
        .find(|instr| matches!(instr.op, ExecOp::DotGeneral(_)))
        .expect("expected DotGeneral after direct lowering");
    assert_eq!(dot_instr.input_slots, vec![0, 1]);
    match &dot_instr.op {
        ExecOp::DotGeneral(config) => {
            assert_eq!(config.lhs_contracting_dims, vec![0]);
            assert_eq!(config.rhs_contracting_dims, vec![0]);
        }
        _ => panic!("expected DotGeneral"),
    }
}
