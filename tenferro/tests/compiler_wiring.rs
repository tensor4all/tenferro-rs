use computegraph::compile::{CompiledProgram, Instruction};
use num_complex::Complex64;
use tenferro::compiler::{compile_to_exec, lower_to_stablehlo};
use tenferro::exec::ExecOp;
use tenferro::stablehlo::StableHloOp;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{DType, GatherConfig, PadConfig, ScatterConfig, SliceConfig};

fn make_program(instructions: Vec<Instruction<StdTensorOp>>) -> CompiledProgram<StdTensorOp> {
    CompiledProgram {
        instructions,
        input_slots: vec![0, 1, 2],
        output_slots: vec![3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        n_slots: 13,
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
        make_instr(
            StdTensorOp::ReduceProd {
                axes: vec![0],
                input_shape: vec![2, 2],
            },
            vec![0],
            vec![3],
        ),
        make_instr(
            StdTensorOp::ReduceMax {
                axes: vec![1],
                input_shape: vec![2, 2],
            },
            vec![1],
            vec![4],
        ),
        make_instr(
            StdTensorOp::ReduceMin {
                axes: vec![0, 1],
                input_shape: vec![2, 2],
            },
            vec![2],
            vec![5],
        ),
        make_instr(StdTensorOp::Slice(slice.clone()), vec![0], vec![6]),
        make_instr(StdTensorOp::Reverse { axes: vec![0] }, vec![1], vec![7]),
        make_instr(StdTensorOp::Concatenate { axis: 0 }, vec![0, 1], vec![8]),
        make_instr(StdTensorOp::Tril { k: -1 }, vec![2], vec![9]),
        make_instr(StdTensorOp::Mul, vec![0, 1], vec![10]),
        make_instr(
            StdTensorOp::TriangularSolve {
                left_side: true,
                lower: true,
                transpose_a: false,
                unit_diagonal: false,
                lhs_shape: vec![2, 2],
                rhs_shape: vec![2, 1],
            },
            vec![0, 1],
            vec![11],
        ),
        make_instr(StdTensorOp::Triu { k: 1 }, vec![2], vec![12]),
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
    assert!(matches!(
        stablehlo.instructions[6].op,
        StableHloOp::Tril { k: -1 }
    ));
    assert!(matches!(
        stablehlo.instructions[7].op,
        StableHloOp::Multiply
    ));
    assert!(matches!(
        stablehlo.instructions[8].op,
        StableHloOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        }
    ));
    assert!(matches!(
        stablehlo.instructions[9].op,
        StableHloOp::Triu { k: 1 }
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
    assert!(matches!(exec.instructions[6].op, ExecOp::Tril { k: -1 }));
    assert!(matches!(exec.instructions[7].op, ExecOp::Multiply));
    assert!(matches!(
        exec.instructions[8].op,
        ExecOp::TriangularSolve {
            left_side: true,
            lower: true,
            transpose_a: false,
            unit_diagonal: false,
        }
    ));
    assert!(matches!(exec.instructions[9].op, ExecOp::Triu { k: 1 }));
}

#[test]
fn lower_to_stablehlo_and_compile_to_exec_wire_indexing_ops() {
    let gather = GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };
    let scatter = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let pad = PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![2],
        interior_padding: vec![0],
    };

    let program = make_program(vec![
        make_instr(StdTensorOp::Gather(gather.clone()), vec![0, 1], vec![3]),
        make_instr(
            StdTensorOp::Scatter(scatter.clone()),
            vec![0, 1, 2],
            vec![4],
        ),
        make_instr(
            StdTensorOp::DynamicSlice {
                slice_sizes: vec![2, 1],
            },
            vec![0, 1],
            vec![5],
        ),
        make_instr(StdTensorOp::Pad(pad.clone()), vec![2], vec![6]),
    ]);

    let stablehlo = lower_to_stablehlo(&program);
    assert!(matches!(
        stablehlo.instructions[0].op,
        StableHloOp::Gather(ref config) if config == &gather
    ));
    assert!(matches!(
        stablehlo.instructions[1].op,
        StableHloOp::Scatter(ref config) if config == &scatter
    ));
    assert!(matches!(
        stablehlo.instructions[2].op,
        StableHloOp::DynamicSlice { ref slice_sizes } if slice_sizes == &vec![2, 1]
    ));
    assert!(matches!(
        stablehlo.instructions[3].op,
        StableHloOp::Pad(ref config) if config == &pad
    ));

    let exec = compile_to_exec(&stablehlo);
    assert!(matches!(
        exec.instructions[0].op,
        ExecOp::Gather(ref config) if config == &gather
    ));
    assert!(matches!(
        exec.instructions[1].op,
        ExecOp::Scatter(ref config) if config == &scatter
    ));
    assert!(matches!(
        exec.instructions[2].op,
        ExecOp::DynamicSlice { ref slice_sizes } if slice_sizes == &vec![2, 1]
    ));
    assert!(matches!(
        exec.instructions[3].op,
        ExecOp::Pad(ref config) if config == &pad
    ));
}

#[test]
fn lower_to_stablehlo_and_compile_to_exec_wire_constant_ops() {
    let complex = Complex64::new(1.0, -2.0);
    let mut complex_bytes = Vec::new();
    complex_bytes.extend_from_slice(&complex.re.to_le_bytes());
    complex_bytes.extend_from_slice(&complex.im.to_le_bytes());

    let program = CompiledProgram {
        instructions: vec![
            make_instr(StdTensorOp::constant_f64(2.5), vec![], vec![1]),
            make_instr(StdTensorOp::constant_c64(complex), vec![], vec![2]),
        ],
        input_slots: vec![],
        output_slots: vec![1, 2],
        n_slots: 3,
    };

    let stablehlo = lower_to_stablehlo(&program);
    assert!(matches!(
        stablehlo.instructions[0].op,
        StableHloOp::Constant { dtype: DType::F64, ref bytes }
            if bytes == &2.5_f64.to_le_bytes().to_vec()
    ));
    assert!(stablehlo.instructions[0].input_slots.is_empty());
    assert!(matches!(
        stablehlo.instructions[1].op,
        StableHloOp::Constant { dtype: DType::C64, ref bytes } if bytes == &complex_bytes
    ));

    let exec = compile_to_exec(&stablehlo);
    assert!(matches!(
        exec.instructions[0].op,
        ExecOp::Constant { dtype: DType::F64, ref bytes }
            if bytes == &2.5_f64.to_le_bytes().to_vec()
    ));
    assert!(exec.instructions[0].input_slots.is_empty());
    assert!(matches!(
        exec.instructions[1].op,
        ExecOp::Constant { dtype: DType::C64, ref bytes } if bytes == &complex_bytes
    ));
}

#[test]
fn lower_to_stablehlo_and_compile_to_exec_wire_convert_op() {
    let program = CompiledProgram {
        instructions: vec![make_instr(
            StdTensorOp::Convert {
                from: DType::F64,
                to: DType::C64,
            },
            vec![0],
            vec![1],
        )],
        input_slots: vec![0],
        output_slots: vec![1],
        n_slots: 2,
    };

    let stablehlo = lower_to_stablehlo(&program);
    assert!(matches!(
        stablehlo.instructions[0].op,
        StableHloOp::Convert { to: DType::C64 }
    ));
    assert_eq!(stablehlo.instructions[0].dtype, DType::C64);

    let exec = compile_to_exec(&stablehlo);
    assert!(matches!(
        exec.instructions[0].op,
        ExecOp::Convert { to: DType::C64 }
    ));
    assert_eq!(exec.instructions[0].dtype, DType::C64);
}
