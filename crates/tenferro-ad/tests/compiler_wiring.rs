use computegraph::compile::{CompiledProgram, Instruction};
use num_complex::Complex64;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::ShapeExtent;
use tenferro_runtime::extension::compile_std_to_exec;
use tenferro_runtime::extension::ExecOp;
use tenferro_tensor::{DType, GatherConfig, PadConfig, ScatterConfig, SliceConfig};

fn dim_shape(shape: &[usize]) -> Vec<DimExpr> {
    DimExpr::from_concrete(shape)
}

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
        operation: op,
        inputs,
        outputs,
    }
}

#[test]
fn compile_std_to_exec_wires_remaining_simple_ops() {
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
        make_instr(
            StdTensorOp::Concatenate {
                axis: 0,
                input_count: 2,
            },
            vec![0, 1],
            vec![8],
        ),
        make_instr(StdTensorOp::Tril { k: -1 }, vec![2], vec![9]),
        make_instr(StdTensorOp::Mul, vec![0, 1], vec![10]),
        make_instr(StdTensorOp::Triu { k: 1 }, vec![2], vec![12]),
    ]);

    let exec = compile_std_to_exec(
        &program,
        &[DType::F64, DType::F64, DType::F64],
        &[dim_shape(&[2, 2]), dim_shape(&[2, 2]), dim_shape(&[2, 2])],
    );

    assert!(matches!(
        exec.instructions[0].op,
        ExecOp::ReduceProd { ref axes } if axes == &vec![0]
    ));
    assert_eq!(
        exec.instructions[0].output_shapes.as_slice(),
        &[dim_shape(&[2])]
    );
    assert!(matches!(
        exec.instructions[1].op,
        ExecOp::ReduceMax { ref axes } if axes == &vec![1]
    ));
    assert!(matches!(
        exec.instructions[2].op,
        ExecOp::ReduceMin { ref axes } if axes == &vec![0, 1]
    ));
    assert_eq!(
        exec.instructions[2].output_shapes.as_slice(),
        &[Vec::<DimExpr>::new()]
    );
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
    assert!(matches!(exec.instructions[8].op, ExecOp::Triu { k: 1 }));
}

#[test]
fn compile_std_to_exec_wires_indexing_ops() {
    let gather = GatherConfig {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, 3],
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

    let exec = compile_std_to_exec(
        &program,
        &[DType::F64, DType::F64, DType::F64],
        &[dim_shape(&[4, 3]), dim_shape(&[2, 1]), dim_shape(&[3])],
    );

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
    assert_eq!(
        exec.instructions[2].output_shapes.as_slice(),
        &[dim_shape(&[2, 1])]
    );
    assert!(matches!(
        exec.instructions[3].op,
        ExecOp::Pad(ref config) if config == &pad
    ));
}

#[test]
fn compile_std_to_exec_does_not_treat_dynamic_truncate_bound_as_exact() {
    let program = CompiledProgram {
        instructions: vec![
            make_instr(
                StdTensorOp::DynamicTruncate { axis: 0 },
                vec![0, 1],
                vec![2],
            ),
            make_instr(
                StdTensorOp::Reshape {
                    to_shape: vec![DimExpr::InputDim {
                        input_idx: 0,
                        axis: 0,
                    }],
                },
                vec![2],
                vec![3],
            ),
        ],
        input_slots: vec![0, 1],
        output_slots: vec![3],
        n_slots: 4,
    };

    let exec = compile_std_to_exec(
        &program,
        &[DType::F64, DType::F64],
        &[dim_shape(&[5]), Vec::new()],
    );

    assert_eq!(exec.instructions.len(), 1);
    assert_eq!(exec.output_slots, vec![2]);
    assert_eq!(exec.instructions[0].output_slots, vec![2]);
    assert_eq!(
        exec.instructions[0].output_extents[0][0],
        ShapeExtent::upper_bound(DimExpr::Const(5))
    );
}

#[test]
fn compile_std_to_exec_marks_unresolvable_extent_unknown() {
    let program = CompiledProgram {
        instructions: vec![make_instr(
            StdTensorOp::BroadcastInDim {
                shape: vec![DimExpr::InputDim {
                    input_idx: 1,
                    axis: 0,
                }],
                dims: vec![],
            },
            vec![0],
            vec![1],
        )],
        input_slots: vec![0],
        output_slots: vec![1],
        n_slots: 2,
    };

    let exec = compile_std_to_exec(&program, &[DType::F64], &[Vec::new()]);

    assert_eq!(
        exec.instructions[0].output_extents[0][0],
        ShapeExtent::unknown()
    );
}

#[test]
fn compile_std_to_exec_wires_constant_and_convert_ops() {
    let complex = Complex64::new(1.0, -2.0);
    let mut complex_bytes = Vec::new();
    complex_bytes.extend_from_slice(&complex.re.to_le_bytes());
    complex_bytes.extend_from_slice(&complex.im.to_le_bytes());

    let program = CompiledProgram {
        instructions: vec![
            make_instr(StdTensorOp::constant(2.5), vec![], vec![1]),
            make_instr(StdTensorOp::constant(complex), vec![], vec![2]),
            make_instr(
                StdTensorOp::Convert {
                    from: DType::F64,
                    to: DType::C64,
                },
                vec![0],
                vec![3],
            ),
        ],
        input_slots: vec![0],
        output_slots: vec![1, 2, 3],
        n_slots: 4,
    };

    let exec = compile_std_to_exec(&program, &[DType::F64], &[dim_shape(&[2])]);

    assert!(matches!(
        exec.instructions[0].op,
        ExecOp::Constant { dtype: DType::F64, ref bytes }
            if bytes == &2.5_f64.to_le_bytes().to_vec()
    ));
    assert_eq!(
        exec.instructions[0].output_shapes.as_slice(),
        &[Vec::<DimExpr>::new()]
    );
    assert!(matches!(
        exec.instructions[1].op,
        ExecOp::Constant { dtype: DType::C64, ref bytes } if bytes == &complex_bytes
    ));
    assert!(matches!(
        exec.instructions[2].op,
        ExecOp::Convert { to: DType::C64 }
    ));
    assert_eq!(exec.instructions[2].dtype, DType::C64);
    assert_eq!(
        exec.instructions[2].output_shapes.as_slice(),
        &[dim_shape(&[2])]
    );
}
