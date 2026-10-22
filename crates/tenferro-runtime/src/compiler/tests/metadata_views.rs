//! The metadata view fold (#2226): which instructions it removes and which it
//! must leave as commands.

use super::{make_exec_instr, make_exec_program};
use crate::compiler::fold_metadata_views;
use crate::exec::{ExecInstruction, ExecOp, ExecProgram, InputView};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_tensor::{DType, SliceConfig};

fn transpose(input: usize, output: usize) -> ExecInstruction {
    make_exec_instr(
        ExecOp::Transpose { perm: vec![1, 0] },
        vec![input],
        vec![output],
    )
}

fn fold_all(program: &mut ExecProgram) {
    fold_metadata_views(program, |_, _| true);
}

fn ops(program: &ExecProgram) -> Vec<String> {
    program
        .instructions
        .iter()
        .map(|instruction| format!("{:?}", instruction.op))
        .collect()
}

#[test]
fn transpose_with_one_reading_consumer_becomes_an_input_view() {
    let mut program = make_exec_program(
        vec![
            make_exec_instr(ExecOp::Negate, vec![0], vec![1]),
            transpose(1, 2),
            make_exec_instr(ExecOp::Add, vec![3, 2], vec![4]),
        ],
        vec![0, 3],
        vec![4],
        5,
    );

    fold_all(&mut program);

    assert_eq!(ops(&program), vec!["Negate", "Add"]);
    let add = &program.instructions[1];
    assert_eq!(add.input_slots, vec![3, 1]);
    assert_eq!(
        add.input_views,
        vec![(1, InputView::Transpose { perm: vec![1, 0] })]
    );
}

#[test]
fn a_chain_of_views_folds_in_application_order() {
    let mut program = make_exec_program(
        vec![
            make_exec_instr(ExecOp::Negate, vec![0], vec![1]),
            transpose(1, 2),
            make_exec_instr(
                ExecOp::Slice(SliceConfig {
                    starts: vec![0, 0],
                    limits: vec![1, 2],
                    strides: vec![1, 1],
                }),
                vec![2],
                vec![3],
            ),
            make_exec_instr(ExecOp::Exp, vec![3], vec![4]),
        ],
        vec![0],
        vec![4],
        5,
    );

    fold_all(&mut program);

    assert_eq!(ops(&program), vec!["Negate", "Exp"]);
    let exp = &program.instructions[1];
    assert_eq!(exp.input_slots, vec![1]);
    assert_eq!(exp.input_views.len(), 2);
    assert!(matches!(
        exp.input_views[0],
        (0, InputView::Transpose { .. })
    ));
    assert!(matches!(exp.input_views[1], (0, InputView::Slice(_))));
}

#[test]
fn a_program_output_keeps_its_command() {
    let mut program = make_exec_program(
        vec![
            make_exec_instr(ExecOp::Negate, vec![0], vec![1]),
            transpose(1, 2),
            make_exec_instr(ExecOp::Exp, vec![2], vec![3]),
        ],
        vec![0],
        vec![2, 3],
        4,
    );

    fold_all(&mut program);

    assert_eq!(
        ops(&program),
        vec!["Negate", "Transpose { perm: [1, 0] }", "Exp"]
    );
}

#[test]
fn a_value_with_two_consumers_keeps_its_command() {
    let mut program = make_exec_program(
        vec![
            make_exec_instr(ExecOp::Negate, vec![0], vec![1]),
            transpose(1, 2),
            make_exec_instr(ExecOp::Exp, vec![2], vec![3]),
            make_exec_instr(ExecOp::Add, vec![2, 3], vec![4]),
        ],
        vec![0],
        vec![4],
        5,
    );

    fold_all(&mut program);

    assert_eq!(program.instructions.len(), 4);
    assert!(program
        .instructions
        .iter()
        .all(|instruction| instruction.input_views.is_empty()));
}

#[test]
fn an_owned_input_consumer_keeps_the_command() {
    let mut program = make_exec_program(
        vec![
            make_exec_instr(ExecOp::Negate, vec![0], vec![1]),
            transpose(1, 2),
            make_exec_instr(ExecOp::Convert { to: DType::F32 }, vec![2], vec![3]),
        ],
        vec![0],
        vec![3],
        4,
    );

    fold_all(&mut program);

    assert_eq!(program.instructions.len(), 3);
}

#[test]
fn a_pair_on_different_placements_keeps_the_command() {
    let mut program = make_exec_program(
        vec![
            make_exec_instr(ExecOp::Negate, vec![0], vec![1]),
            transpose(1, 2),
            make_exec_instr(ExecOp::Exp, vec![2], vec![3]),
        ],
        vec![0],
        vec![3],
        4,
    );

    fold_metadata_views(&mut program, |_, _| false);

    assert_eq!(program.instructions.len(), 3);
}

#[test]
fn an_external_dtype_value_keeps_the_command() {
    let mut program = make_exec_program(
        vec![
            make_exec_instr(ExecOp::Negate, vec![0], vec![1]),
            transpose(1, 2),
            make_exec_instr(ExecOp::Exp, vec![2], vec![3]),
        ],
        vec![0],
        vec![3],
        4,
    );
    program.instructions[1].dtype = DType::External(std::any::TypeId::of::<()>());

    fold_all(&mut program);

    assert_eq!(program.instructions.len(), 3);
}

#[test]
fn reshape_folds_only_from_a_compact_engine_result() {
    // Reshape of an engine result (Negate output): a view.
    let mut program = make_exec_program(
        vec![
            make_exec_instr(ExecOp::Negate, vec![0], vec![1]),
            make_exec_instr(
                ExecOp::Reshape {
                    shape: vec![DimExpr::Const(4)],
                },
                vec![1],
                vec![2],
            ),
            make_exec_instr(ExecOp::Exp, vec![2], vec![3]),
        ],
        vec![0],
        vec![3],
        4,
    );
    fold_all(&mut program);
    assert_eq!(ops(&program), vec!["Negate", "Exp"]);
    assert!(matches!(
        program.instructions[1].input_views[0],
        (0, InputView::Reshape { .. })
    ));

    // Reshape of a kept transpose (two consumers, so the transpose stays and
    // may yield a strided borrowed view of a program input): stays a command.
    let mut program = make_exec_program(
        vec![
            transpose(0, 1),
            make_exec_instr(
                ExecOp::Reshape {
                    shape: vec![DimExpr::Const(4)],
                },
                vec![1],
                vec![2],
            ),
            make_exec_instr(ExecOp::Add, vec![1, 1], vec![3]),
            make_exec_instr(ExecOp::Exp, vec![2], vec![4]),
        ],
        vec![0],
        vec![3, 4],
        5,
    );
    fold_all(&mut program);
    assert_eq!(
        ops(&program),
        vec![
            "Transpose { perm: [1, 0] }",
            "Reshape { shape: [Const(4)] }",
            "Add",
            "Exp"
        ]
    );

    // Reshape of a program input: stays a command (the input may be a view).
    let mut program = make_exec_program(
        vec![
            make_exec_instr(
                ExecOp::Reshape {
                    shape: vec![DimExpr::Const(4)],
                },
                vec![0],
                vec![1],
            ),
            make_exec_instr(ExecOp::Exp, vec![1], vec![2]),
        ],
        vec![0],
        vec![2],
        3,
    );
    fold_all(&mut program);
    assert_eq!(program.instructions.len(), 2);
}

#[test]
fn a_view_of_a_program_input_folds_into_its_consumer() {
    let mut program = make_exec_program(
        vec![
            make_exec_instr(
                ExecOp::BroadcastInDim {
                    shape: vec![
                        DimExpr::InputDim {
                            input_idx: 0,
                            axis: 0,
                        },
                        DimExpr::Const(3),
                    ],
                    dims: vec![0],
                },
                vec![0],
                vec![1],
            ),
            make_exec_instr(ExecOp::Multiply, vec![1, 2], vec![3]),
        ],
        vec![0, 2],
        vec![3],
        4,
    );

    fold_all(&mut program);

    assert_eq!(ops(&program), vec!["Multiply"]);
    let multiply = &program.instructions[0];
    assert_eq!(multiply.input_slots, vec![0, 2]);
    assert!(matches!(
        &multiply.input_views[..],
        [(0, InputView::BroadcastInDim { shape, dims })]
            if shape.len() == 2 && dims == &vec![0]
    ));
}
