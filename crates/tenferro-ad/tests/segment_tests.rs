use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::extension::{ExecInstruction, ExecOp, ExecProgram};
use tenferro_runtime::{DType, ExtensionCacheStore, ExtensionExecutionContext, GraphExecutor};
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorValue, TypedTensor};

#[cfg(feature = "cuda")]
use tenferro_gpu::{download_tensor, upload_tensor, CudaBackend};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn scalar_extents(
    output_count: usize,
) -> Vec<Vec<tenferro_ops::ShapeExtent<tenferro_ops::dim_expr::DimExpr>>> {
    vec![Vec::new(); output_count]
}

fn const_shape(shape: &[usize]) -> Vec<tenferro_ops::dim_expr::DimExpr> {
    shape
        .iter()
        .map(|&dim| tenferro_ops::dim_expr::DimExpr::Const(dim))
        .collect()
}

fn exact_extents(
    shape: &[usize],
) -> Vec<tenferro_ops::ShapeExtent<tenferro_ops::dim_expr::DimExpr>> {
    const_shape(shape)
        .into_iter()
        .map(tenferro_ops::ShapeExtent::exact)
        .collect()
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
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![false, false],
            },
            ExecInstruction {
                op: ExecOp::Exp,
                input_slots: vec![4],
                output_slots: vec![5],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![false],
            },
            ExecInstruction {
                op: ExecOp::ShapeOf { axis: 0 },
                input_slots: vec![5],
                output_slots: vec![6],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![false],
            },
            ExecInstruction {
                op: ExecOp::DynamicTruncate { axis: 0 },
                input_slots: vec![5, 6],
                output_slots: vec![7],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![true, true],
            },
            ExecInstruction {
                op: ExecOp::ReduceSum { axes: vec![1] },
                input_slots: vec![7],
                output_slots: vec![8],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![true],
            },
            ExecInstruction {
                op: ExecOp::DotGeneral(matmul_config()),
                input_slots: vec![0, 1],
                output_slots: vec![9],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![true, true],
            },
            ExecInstruction {
                op: ExecOp::PadToMatch { axis: 0 },
                input_slots: vec![2, 3],
                output_slots: vec![11],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
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
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![],
            },
        ],
        input_slots: vec![0, 1, 2, 3],
        output_slots: vec![8, 9, 11, 12],
        n_slots: 13,
        shape_guards: Vec::new(),
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

fn assert_tensor_eq(lhs: &Tensor, rhs: &Tensor) {
    assert_eq!(lhs.shape(), rhs.shape());
    match (lhs, rhs) {
        (Tensor::F32(lhs), Tensor::F32(rhs)) => {
            assert_eq!(lhs.host_data().unwrap(), rhs.host_data().unwrap())
        }
        (Tensor::F64(lhs), Tensor::F64(rhs)) => {
            assert_eq!(lhs.host_data().unwrap(), rhs.host_data().unwrap())
        }
        (Tensor::C32(lhs), Tensor::C32(rhs)) => {
            assert_eq!(lhs.host_data().unwrap(), rhs.host_data().unwrap())
        }
        (Tensor::C64(lhs), Tensor::C64(rhs)) => {
            assert_eq!(lhs.host_data().unwrap(), rhs.host_data().unwrap())
        }
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
fn segmented_dispatch_matches_unsegmented_dispatch_on_cpu() {
    let program = cpu_parity_program();
    let inputs = cpu_parity_inputs();

    let mut cpu_unsegmented = CpuBackend::new();
    let mut extension_caches = ExtensionCacheStore::new();
    let mut unsegmented_context =
        ExtensionExecutionContext::new(&mut cpu_unsegmented, &mut extension_caches);
    let unsegmented = unsegmented_context
        .execute_core_exec_program_unsegmented(&program, inputs.clone())
        .unwrap();

    let mut cpu_segmented = GraphExecutor::new(CpuBackend::new());
    let segmented = cpu_segmented
        .eval_exec_ir(&program, inputs.clone())
        .unwrap();

    let mut cpu_non_consuming = GraphExecutor::new(CpuBackend::new());
    let segmented_non_consuming = cpu_non_consuming
        .eval_exec_ir_non_consuming(&program, &inputs)
        .unwrap();

    assert_tensor_vec_eq(&unsegmented, &segmented);
    assert_tensor_vec_eq(&segmented, &segmented_non_consuming);
}

fn terminal_noncompact_broadcast_multiply_program() -> ExecProgram {
    let output_shape = vec![2, 2, 4, 3];
    let output_exprs = const_shape(&output_shape);
    let output_extents = exact_extents(&output_shape);

    ExecProgram {
        instructions: vec![
            ExecInstruction {
                op: ExecOp::BroadcastInDim {
                    shape: output_exprs.clone(),
                    dims: vec![1, 0, 3],
                },
                input_slots: vec![0],
                output_slots: vec![2],
                dtype: DType::F64,
                output_shapes: vec![output_exprs.clone()].into(),
                output_extents: vec![output_extents.clone()].into(),
                last_use: vec![false],
            },
            ExecInstruction {
                op: ExecOp::BroadcastInDim {
                    shape: output_exprs.clone(),
                    dims: vec![2, 3],
                },
                input_slots: vec![1],
                output_slots: vec![3],
                dtype: DType::F64,
                output_shapes: vec![output_exprs.clone()].into(),
                output_extents: vec![output_extents.clone()].into(),
                last_use: vec![false],
            },
            ExecInstruction {
                op: ExecOp::Multiply,
                input_slots: vec![2, 3],
                output_slots: vec![4],
                dtype: DType::F64,
                output_shapes: vec![output_exprs].into(),
                output_extents: vec![output_extents].into(),
                last_use: vec![true, true],
            },
        ],
        input_slots: vec![0, 1],
        output_slots: vec![4],
        n_slots: 5,
        shape_guards: Vec::new(),
    }
}

fn terminal_noncompact_broadcast_multiply_inputs() -> Vec<Tensor> {
    vec![
        f64_tensor(vec![2, 2, 3], (0..12).map(|idx| idx as f64 + 1.0).collect()),
        f64_tensor(vec![4, 3], (0..12).map(|idx| idx as f64 + 101.0).collect()),
    ]
}

#[test]
fn segmented_value_dispatch_preserves_terminal_lazy_broadcast_multiply_view() {
    let program = terminal_noncompact_broadcast_multiply_program();
    let inputs = terminal_noncompact_broadcast_multiply_inputs();

    let mut executor = GraphExecutor::new(CpuBackend::new());
    let mut outputs = executor
        .eval_exec_ir_non_consuming_values(&program, &inputs)
        .unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(
        matches!(outputs[0], TensorValue::View(_)),
        "terminal broadcast-multiply segment should preserve a lazy output view"
    );

    let output = outputs.pop().unwrap().to_tensor().unwrap();
    assert_eq!(output.shape(), &[2, 2, 4, 3]);
    let Tensor::F64(output) = output else {
        panic!("expected f64 output")
    };
    let lhs = inputs[0].as_slice::<f64>().unwrap();
    let rhs = inputs[1].as_slice::<f64>().unwrap();
    let actual = output.host_data().unwrap();
    for t in 0..3 {
        for o in 0..4 {
            for k in 0..2 {
                for j in 0..2 {
                    let out_idx = j + 2 * k + 4 * o + 16 * t;
                    let lhs_idx = k + 2 * j + 4 * t;
                    let rhs_idx = o + 4 * t;
                    assert_eq!(actual[out_idx], lhs[lhs_idx] * rhs[rhs_idx]);
                }
            }
        }
    }
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
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![true],
            },
            ExecInstruction {
                op: ExecOp::ShapeOf { axis: 0 },
                input_slots: vec![3],
                output_slots: vec![4],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![true],
            },
            ExecInstruction {
                op: ExecOp::PadToMatch { axis: 0 },
                input_slots: vec![1, 2],
                output_slots: vec![5],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![true, true],
            },
            ExecInstruction {
                op: ExecOp::Exp,
                input_slots: vec![5],
                output_slots: vec![6],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![false],
            },
            ExecInstruction {
                op: ExecOp::ShapeOf { axis: 0 },
                input_slots: vec![6],
                output_slots: vec![7],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![false],
            },
            ExecInstruction {
                op: ExecOp::DynamicTruncate { axis: 0 },
                input_slots: vec![6, 7],
                output_slots: vec![8],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![true, true],
            },
            ExecInstruction {
                op: ExecOp::Negate,
                input_slots: vec![8],
                output_slots: vec![9],
                dtype: DType::F64,
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
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
                output_shapes: vec![Vec::new()].into(),
                output_extents: scalar_extents(1).into(),
                last_use: vec![],
            },
        ],
        input_slots: vec![0, 1, 2],
        output_slots: vec![4, 9, 10],
        n_slots: 11,
        shape_guards: Vec::new(),
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
fn upload_all(backend: &CudaBackend, tensors: &[Tensor]) -> Vec<Tensor> {
    tensors
        .iter()
        .map(|tensor| upload_tensor(backend.runtime(), tensor).unwrap())
        .collect()
}

#[cfg(feature = "cuda")]
fn download_all(backend: &CudaBackend, tensors: &[Tensor]) -> Vec<Tensor> {
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

    let mut gpu_unsegmented = CudaBackend::new(0).unwrap();
    let unsegmented_inputs = upload_all(&gpu_unsegmented, &host_inputs);
    let mut extension_caches = ExtensionCacheStore::new();
    let mut unsegmented_context =
        ExtensionExecutionContext::new(&mut gpu_unsegmented, &mut extension_caches);
    let unsegmented = unsegmented_context
        .execute_core_exec_program_unsegmented(&program, unsegmented_inputs)
        .unwrap();
    drop(unsegmented_context);
    let unsegmented_host = download_all(&gpu_unsegmented, &unsegmented);

    let gpu_segmented_backend = CudaBackend::new(0).unwrap();
    let segmented_inputs = upload_all(&gpu_segmented_backend, &host_inputs);
    let mut gpu_segmented = GraphExecutor::new(gpu_segmented_backend);
    let segmented = gpu_segmented
        .eval_exec_ir(&program, segmented_inputs)
        .unwrap();
    let segmented_host = download_all(gpu_segmented.backend(), &segmented);

    assert_tensor_vec_eq(&unsegmented_host, &segmented_host);
}

#[allow(dead_code)]
fn _keep_complex_imports(_: Complex32, _: Complex64) {}
