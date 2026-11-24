//! Diagnostic probes use only pass APIs shared with the original downstream pin.

use super::{conj_sinking, make_exec_instr_with_meta, make_exec_program, DType, DimExpr, ExecOp};

#[test]
#[ignore = "manual release-mode conjugation metadata scaling measurement"]
fn benchmark_conj_sinking_scaling() {
    use std::hint::black_box;
    use std::time::Instant;

    let mut extent = DimExpr::Const(2);
    for _ in 0..16 {
        extent = DimExpr::Add(Box::new(extent), Box::new(DimExpr::Const(0)));
    }
    let shapes = [vec![extent, DimExpr::Const(3)]];
    println!("layout_instructions,repetition,nanoseconds");
    for length in [64, 256, 1024, 4096] {
        let mut instructions = (0..length)
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
                    DType::C64,
                    vec![shapes[0].clone()],
                )
            })
            .collect::<Vec<_>>();
        // Keep an input conjugate without a sinking fixpoint, isolating metadata/redirect traversal.
        instructions.push(make_exec_instr_with_meta(
            ExecOp::Conj,
            vec![0],
            vec![length + 1],
            DType::C64,
            vec![shapes[0].clone()],
        ));
        let template =
            make_exec_program(instructions, vec![0], vec![length, length + 1], length + 2);
        for repetition in 0..21 {
            let mut program = template.clone();
            let start = Instant::now();
            conj_sinking(
                black_box(&mut program),
                black_box(&[DType::C64]),
                black_box(&shapes),
            )
            .unwrap();
            let elapsed = start.elapsed();
            assert_eq!(program.output_slots, [length, length + 1]);
            assert_eq!(program.instructions.len(), length + 1);
            for (index, instruction) in program.instructions[..length].iter().enumerate() {
                assert_eq!(instruction.input_slots, [index]);
                assert_eq!(instruction.output_slots, [index + 1]);
            }
            assert!(matches!(program.instructions[length].op, ExecOp::Conj));
            assert_eq!(program.instructions[length].input_slots, [0]);
            black_box(program);
            if repetition > 0 {
                println!("{length},{repetition},{}", elapsed.as_nanos());
            }
        }
    }
}
