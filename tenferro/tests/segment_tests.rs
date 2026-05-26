use num_complex::{Complex32, Complex64};
use tenferro::exec::{
    eval_exec_ir, eval_exec_ir_unsegmented, ExecInstruction, ExecOp, ExecProgram,
};
use tenferro::segment::{eval_exec_segmented, segment_exec_program, Segment};
use tenferro::DType;
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{DotGeneralConfig, Tensor, TypedTensor};

#[cfg(feature = "cuda")]
use tenferro_gpu::cubecl::{download_tensor, upload_tensor, CubeclBackend};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

fn scalar_extents(
    n_outputs: usize,
) -> Vec<Vec<tenferro_ops::ShapeExtent<tenferro_ops::dim_expr::DimExpr>>> {
    vec![Vec::new(); n_outputs]
}

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

fn cpu_parity_program() -> ExecProgram {
    ExecProgram {
        instructions: vec![
            ExecInstruction {
                op: ExecOp::Add,
                input_slots: vec![0, 1],
                output_slots: vec![4],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![false, false],
            },
            ExecInstruction {
                op: ExecOp::Exp,
                input_slots: vec![4],
                output_slots: vec![5],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![false],
            },
            ExecInstruction {
                op: ExecOp::ShapeOf { axis: 0 },
                input_slots: vec![5],
                output_slots: vec![6],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![false],
            },
            ExecInstruction {
                op: ExecOp::DynamicTruncate { axis: 0 },
                input_slots: vec![5, 6],
                output_slots: vec![7],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![true, true],
            },
            ExecInstruction {
                op: ExecOp::ReduceSum { axes: vec![1] },
                input_slots: vec![7],
                output_slots: vec![8],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![true],
            },
            ExecInstruction {
                op: ExecOp::DotGeneral(matmul_config()),
                input_slots: vec![0, 1],
                output_slots: vec![9],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![true, true],
            },
            ExecInstruction {
                op: ExecOp::PadToMatch { axis: 0 },
                input_slots: vec![2, 3],
                output_slots: vec![11],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![true, true],
            },
            ExecInstruction {
                op: ExecOp::Constant {
                    dtype: DType::F64,
                    bytes: 1.5_f64.to_le_bytes().to_vec(),
                },
                input_slots: vec![],
                output_slots: vec![12],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![],
            },
        ],
        input_slots: vec![0, 1, 2, 3],
        output_slots: vec![8, 9, 11, 12],
        n_slots: 13,
    }
}

fn cpu_parity_inputs() -> Vec<Tensor> {
    vec![
        f64_tensor(vec![2, 2], vec![4.0, 1.0, 1.0, 3.0]),
        f64_tensor(vec![2, 2], vec![0.5, 0.0, 0.0, 0.5]),
        f64_tensor(vec![2], vec![1.0, 2.0]),
        f64_tensor(vec![4], vec![0.0, 0.0, 0.0, 0.0]),
    ]
}

fn segment_classification_program() -> ExecProgram {
    ExecProgram {
        instructions: vec![
            ExecInstruction {
                op: ExecOp::Add,
                input_slots: vec![0, 1],
                output_slots: vec![2],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![false, false],
            },
            ExecInstruction {
                op: ExecOp::Negate,
                input_slots: vec![2],
                output_slots: vec![3],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![true],
            },
            ExecInstruction {
                op: ExecOp::ShapeOf { axis: 0 },
                input_slots: vec![3],
                output_slots: vec![4],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![true],
            },
            ExecInstruction {
                op: ExecOp::DotGeneral(matmul_config()),
                input_slots: vec![0, 1],
                output_slots: vec![5],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![true, true],
            },
        ],
        input_slots: vec![0, 1],
        output_slots: vec![4, 5],
        n_slots: 6,
    }
}

fn assert_tensor_eq(lhs: &Tensor, rhs: &Tensor) {
    assert_eq!(lhs.shape(), rhs.shape());
    match (lhs, rhs) {
        (Tensor::F32(lhs), Tensor::F32(rhs)) => assert_eq!(lhs.host_data(), rhs.host_data()),
        (Tensor::F64(lhs), Tensor::F64(rhs)) => assert_eq!(lhs.host_data(), rhs.host_data()),
        (Tensor::C32(lhs), Tensor::C32(rhs)) => assert_eq!(lhs.host_data(), rhs.host_data()),
        (Tensor::C64(lhs), Tensor::C64(rhs)) => assert_eq!(lhs.host_data(), rhs.host_data()),
        _ => panic!("dtype mismatch: lhs={lhs:?} rhs={rhs:?}"),
    }
}

fn assert_tensor_vec_eq(lhs: &[Tensor], rhs: &[Tensor]) {
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        assert_tensor_eq(lhs, rhs);
    }
}

#[test]
fn segment_exec_program_groups_fusible_and_boundary_ops() {
    let program = segment_classification_program();
    let segments = segment_exec_program(&program);

    assert_eq!(segments.len(), 3);
    match &segments[0] {
        Segment::Fused {
            instructions,
            input_slots,
            output_slots,
            last_use,
        } => {
            assert_eq!(instructions.len(), 2);
            assert_eq!(input_slots, &vec![0, 1]);
            assert_eq!(output_slots, &vec![3]);
            assert_eq!(last_use, &vec![false, false]);
        }
        other => panic!("expected fused segment, got {other:?}"),
    }
    assert!(matches!(
        &segments[1],
        Segment::Host(ExecInstruction {
            op: ExecOp::ShapeOf { axis: 0 },
            ..
        })
    ));
    assert!(matches!(
        &segments[2],
        Segment::Ffi(ExecInstruction {
            op: ExecOp::DotGeneral(_),
            output_slots,
            ..
        }) if output_slots.len() == 1
    ));
}

#[test]
fn segmented_dispatch_matches_unsegmented_dispatch_on_cpu() {
    let program = cpu_parity_program();
    let inputs = cpu_parity_inputs();

    let mut cpu_unsegmented = CpuBackend::new();
    let unsegmented =
        eval_exec_ir_unsegmented(&mut cpu_unsegmented, &program, inputs.clone()).unwrap();

    let mut cpu_segmented = CpuBackend::new();
    let segmented = eval_exec_segmented(&mut cpu_segmented, &program, inputs.clone()).unwrap();

    let mut cpu_alias = CpuBackend::new();
    let segmented_via_alias = eval_exec_ir(&mut cpu_alias, &program, inputs).unwrap();

    assert_tensor_vec_eq(&unsegmented, &segmented);
    assert_tensor_vec_eq(&segmented, &segmented_via_alias);
}

#[cfg(feature = "cuda")]
fn gpu_host_boundary_program() -> ExecProgram {
    ExecProgram {
        instructions: vec![
            ExecInstruction {
                op: ExecOp::DotGeneral(matmul_config()),
                input_slots: vec![0, 0],
                output_slots: vec![3],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![true],
            },
            ExecInstruction {
                op: ExecOp::ShapeOf { axis: 0 },
                input_slots: vec![3],
                output_slots: vec![4],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![true],
            },
            ExecInstruction {
                op: ExecOp::PadToMatch { axis: 0 },
                input_slots: vec![1, 2],
                output_slots: vec![5],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![true, true],
            },
            ExecInstruction {
                op: ExecOp::Exp,
                input_slots: vec![5],
                output_slots: vec![6],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![false],
            },
            ExecInstruction {
                op: ExecOp::ShapeOf { axis: 0 },
                input_slots: vec![6],
                output_slots: vec![7],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![false],
            },
            ExecInstruction {
                op: ExecOp::DynamicTruncate { axis: 0 },
                input_slots: vec![6, 7],
                output_slots: vec![8],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![true, true],
            },
            ExecInstruction {
                op: ExecOp::Negate,
                input_slots: vec![8],
                output_slots: vec![9],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![true],
            },
            ExecInstruction {
                op: ExecOp::Constant {
                    dtype: DType::F64,
                    bytes: 2.0_f64.to_le_bytes().to_vec(),
                },
                input_slots: vec![],
                output_slots: vec![10],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()],
                output_extents: scalar_extents(1),
                last_use: vec![],
            },
        ],
        input_slots: vec![0, 1, 2],
        output_slots: vec![4, 9, 10],
        n_slots: 11,
    }
}

#[cfg(feature = "cuda")]
fn gpu_host_boundary_inputs() -> Vec<Tensor> {
    vec![
        f64_tensor(vec![2, 2], vec![5.0, 1.0, 1.0, 4.0]),
        f64_tensor(vec![2], vec![1.0, 2.0]),
        f64_tensor(vec![4], vec![0.0, 0.0, 0.0, 0.0]),
    ]
}

#[cfg(feature = "cuda")]
fn upload_all(backend: &CubeclBackend, tensors: &[Tensor]) -> Vec<Tensor> {
    tensors
        .iter()
        .map(|tensor| upload_tensor(backend.runtime(), tensor).unwrap())
        .collect()
}

#[cfg(feature = "cuda")]
fn download_all(backend: &CubeclBackend, tensors: &[Tensor]) -> Vec<Tensor> {
    tensors
        .iter()
        .map(|tensor| download_tensor(backend.runtime(), tensor).unwrap())
        .collect()
}

#[cfg(feature = "cuda")]
#[test]
fn segmented_dispatch_matches_unsegmented_dispatch_on_cubecl_host_boundaries() {
    let program = gpu_host_boundary_program();
    let host_inputs = gpu_host_boundary_inputs();

    let mut gpu_unsegmented = CubeclBackend::new(0).unwrap();
    let unsegmented_inputs = upload_all(&gpu_unsegmented, &host_inputs);
    let unsegmented =
        eval_exec_ir_unsegmented(&mut gpu_unsegmented, &program, unsegmented_inputs).unwrap();
    let unsegmented_host = download_all(&gpu_unsegmented, &unsegmented);

    let mut gpu_segmented = CubeclBackend::new(0).unwrap();
    let segmented_inputs = upload_all(&gpu_segmented, &host_inputs);
    let segmented = eval_exec_segmented(&mut gpu_segmented, &program, segmented_inputs).unwrap();
    let segmented_host = download_all(&gpu_segmented, &segmented);

    assert_tensor_vec_eq(&unsegmented_host, &segmented_host);
}

#[allow(dead_code)]
fn _keep_complex_imports(_: Complex32, _: Complex64) {}
