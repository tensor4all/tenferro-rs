use computegraph::compile::{CompiledProgram, Instruction};
use tenferro::compiler::{compile_to_exec, lower_to_stablehlo};
use tenferro::exec::ExecOp;
use tenferro::stablehlo::StableHloOp;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::SliceConfig;

fn make_program(instructions: Vec<Instruction<StdTensorOp>>) -> CompiledProgram<StdTensorOp> {
    CompiledProgram {
        instructions,
        input_slots: vec![0, 1, 2],
        output_slots: vec![3, 4, 5, 6, 7, 8],
        n_slots: 9,
    }
}

fn make_instr(
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
fn lower_to_stablehlo_and_compile_to_exec_wire_remaining_simple_ops() {
    let slice = SliceConfig {
        starts: vec![1, 1],
        limits: vec![3, 3],
        strides: vec![1, 1],
    };
    let program = make_program(vec![
        make_instr(StdTensorOp::ReduceProd { axes: vec![0] }, vec![0], vec![3]),
        make_instr(StdTensorOp::ReduceMax { axes: vec![1] }, vec![1], vec![4]),
        make_instr(
            StdTensorOp::ReduceMin { axes: vec![0, 1] },
            vec![2],
            vec![5],
        ),
        make_instr(StdTensorOp::Slice(slice.clone()), vec![0], vec![6]),
        make_instr(StdTensorOp::Reverse { axes: vec![0] }, vec![1], vec![7]),
        make_instr(StdTensorOp::Concatenate { axis: 0 }, vec![0, 1], vec![8]),
    ]);

    let stablehlo = lower_to_stablehlo(&program);
    assert!(matches!(
        stablehlo.instructions[0].op,
        StableHloOp::ReduceProd { ref axes } if axes == &vec![0]
    ));
    assert!(matches!(
        stablehlo.instructions[1].op,
        StableHloOp::ReduceMax { ref axes } if axes == &vec![1]
    ));
    assert!(matches!(
        stablehlo.instructions[2].op,
        StableHloOp::ReduceMin { ref axes } if axes == &vec![0, 1]
    ));
    assert!(matches!(
        stablehlo.instructions[3].op,
        StableHloOp::Slice(ref config) if config == &slice
    ));
    assert!(matches!(
        stablehlo.instructions[4].op,
        StableHloOp::Reverse { ref axes } if axes == &vec![0]
    ));
    assert!(matches!(
        stablehlo.instructions[5].op,
        StableHloOp::Concatenate { axis: 0 }
    ));

    let exec = compile_to_exec(&stablehlo);
    assert!(matches!(
        exec.instructions[0].op,
        ExecOp::ReduceProd { ref axes } if axes == &vec![0]
    ));
    assert!(matches!(
        exec.instructions[1].op,
        ExecOp::ReduceMax { ref axes } if axes == &vec![1]
    ));
    assert!(matches!(
        exec.instructions[2].op,
        ExecOp::ReduceMin { ref axes } if axes == &vec![0, 1]
    ));
    assert!(matches!(
        exec.instructions[3].op,
        ExecOp::Slice(ref config) if config == &slice
    ));
    assert!(matches!(
        exec.instructions[4].op,
        ExecOp::Reverse { ref axes } if axes == &vec![0]
    ));
    assert!(matches!(
        exec.instructions[5].op,
        ExecOp::Concatenate { axis: 0 }
    ));
}
