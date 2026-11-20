use super::{
    algebraic_layout_simplifier, compile_std_to_exec, dim_shape, eliminate_dead_code,
    make_exec_instr_with_meta, make_exec_program, make_std_instr, CompiledProgram, DType, DimExpr,
    ExecOp, ExecProgram, StdTensorOp,
};

fn identity_chain(length: usize) -> ExecProgram {
    let instructions = (0..length)
        .map(|index| {
            let op = if index % 2 == 0 {
                ExecOp::Transpose { perm: vec![0, 1] }
            } else {
                ExecOp::Reshape {
                    shape: DimExpr::input_shape(0, 2),
                }
            };
            make_exec_instr_with_meta(
                op,
                vec![index],
                vec![index + 1],
                DType::F64,
                vec![dim_shape(&[2, 3])],
            )
        })
        .collect();
    make_exec_program(instructions, vec![0], vec![length], length + 1)
}

#[test]
fn resolves_identity_chains_for_fanout_and_program_outputs() {
    let mut program = identity_chain(128);
    program.instructions.push(make_exec_instr_with_meta(
        ExecOp::Add,
        vec![1, 64],
        vec![129],
        DType::F64,
        vec![dim_shape(&[2, 3])],
    ));
    program.instructions.push(make_exec_instr_with_meta(
        ExecOp::Negate,
        vec![128],
        vec![130],
        DType::F64,
        vec![dim_shape(&[2, 3])],
    ));
    program.output_slots = vec![0, 1, 64, 128, 129, 130, 128];
    program.n_slots = 131;

    algebraic_layout_simplifier(&mut program, &[dim_shape(&[2, 3])]).unwrap();
    eliminate_dead_code(&mut program);

    assert_eq!(program.output_slots, vec![0, 0, 0, 0, 129, 130, 0]);
    assert_eq!(program.instructions.len(), 2);
    assert!(matches!(program.instructions[0].op, ExecOp::Add));
    assert_eq!(program.instructions[0].input_slots, vec![0, 0]);
    assert!(matches!(program.instructions[1].op, ExecOp::Negate));
    assert_eq!(program.instructions[1].input_slots, vec![0]);
}

#[test]
fn composes_transposes_across_identity_aliases() {
    let instructions = [
        (
            ExecOp::Transpose {
                perm: vec![1, 2, 0],
            },
            [3, 4, 2],
        ),
        (
            ExecOp::Reshape {
                shape: DimExpr::input_shape(0, 3),
            },
            [3, 4, 2],
        ),
        (
            ExecOp::Transpose {
                perm: vec![1, 2, 0],
            },
            [4, 2, 3],
        ),
        (
            ExecOp::Transpose {
                perm: vec![0, 1, 2],
            },
            [4, 2, 3],
        ),
        (
            ExecOp::Transpose {
                perm: vec![1, 2, 0],
            },
            [2, 3, 4],
        ),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (op, shape))| {
        make_exec_instr_with_meta(
            op,
            vec![index],
            vec![index + 1],
            DType::F64,
            vec![dim_shape(&shape)],
        )
    })
    .collect();
    let mut program = make_exec_program(instructions, vec![0], vec![5], 6);

    algebraic_layout_simplifier(&mut program, &[dim_shape(&[2, 3, 4])]).unwrap();

    assert!(
        matches!(&program.instructions[2].op, ExecOp::Transpose { perm } if perm == &[2, 0, 1])
    );
    assert_eq!(program.instructions[2].input_slots, vec![0]);
    assert_eq!(program.output_slots, vec![0]);
    eliminate_dead_code(&mut program);
    assert!(program.instructions.is_empty());
}

#[test]
fn compile_eliminates_identity_layout_chain() {
    let length = 128;
    let mut instructions = (0..length)
        .map(|index| {
            let op = if index % 2 == 0 {
                StdTensorOp::Transpose { perm: vec![0, 1] }
            } else {
                StdTensorOp::Reshape {
                    to_shape: DimExpr::input_shape(0, 2),
                }
            };
            make_std_instr(op, vec![index], vec![index + 1])
        })
        .collect::<Vec<_>>();
    instructions.push(make_std_instr(
        StdTensorOp::Add,
        vec![1, length],
        vec![length + 1],
    ));
    let program = CompiledProgram {
        instructions,
        input_slots: vec![0],
        output_slots: vec![length, length + 1],
        n_slots: length + 2,
    };

    let compiled = compile_std_to_exec(&program, &[DType::F64], &[dim_shape(&[2, 3])]).unwrap();

    assert_eq!(compiled.output_slots, vec![0, length + 1]);
    assert_eq!(compiled.instructions.len(), 1);
    assert!(matches!(compiled.instructions[0].op, ExecOp::Add));
    assert_eq!(compiled.instructions[0].input_slots, vec![0, 0]);
}

#[test]
fn layout_rewrites_do_not_rescan_program_uses() {
    let source = include_str!("../mod.rs");
    let (_, pass) = source
        .split_once("fn algebraic_layout_simplifier_one_pass(")
        .unwrap();
    let (pass, _) = pass.split_once("fn collect_slot_shapes(").unwrap();
    assert!(
        !pass.contains("replace_slot_uses(") && !pass.contains("resolve_slot_redirect("),
        "each layout rewrite must use a canonical alias, without a whole-program scan or per-use slot-table allocation"
    );
}

#[test]
#[ignore = "manual release-mode compiler scaling measurement"]
fn benchmark_algebraic_layout_simplifier_scaling() {
    use std::hint::black_box;
    use std::time::Instant;

    let shapes = [dim_shape(&[2, 3])];
    println!("instructions,repetition,nanoseconds");
    for length in [64, 256, 1024, 4096] {
        let template = identity_chain(length);
        for repetition in 0..21 {
            let mut program = template.clone();
            let start = Instant::now();
            algebraic_layout_simplifier(black_box(&mut program), black_box(&shapes)).unwrap();
            let elapsed = start.elapsed();
            assert_eq!(program.output_slots, vec![0]);
            assert!(program
                .instructions
                .iter()
                .all(|instruction| instruction.input_slots == [0]));
            black_box(program);
            if repetition > 0 {
                println!("{length},{repetition},{}", elapsed.as_nanos());
            }
        }
    }
}
