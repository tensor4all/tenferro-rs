use tenferro_cpu::CpuBackend;
use tenferro_runtime::extension::{ExecInstruction, ExecOp, ExecProgram};
use tenferro_runtime::{DType, GraphExecutor, Tensor, TypedTensor};

#[test]
fn eval_exec_ir_non_consuming_preserves_caller_inputs() {
    let program = ExecProgram {
        instructions: vec![ExecInstruction {
            op: ExecOp::Add,
            input_slots: vec![0, 1],
            output_slots: vec![2],
            dtype: DType::F64,
            output_shapes: vec![vec![]].into(),
            output_extents: vec![vec![]].into(),
            last_use: vec![true, true],
        }],
        input_slots: vec![0, 1],
        output_slots: vec![2],
        n_slots: 3,
        shape_guards: Vec::new(),
    };

    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![2.0]).unwrap());
    let rhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![3.0]).unwrap());
    let inputs = vec![lhs, rhs];

    let mut engine = GraphExecutor::new(CpuBackend::with_threads(1).unwrap());
    let outputs = engine
        .eval_exec_ir_non_consuming(&program, &inputs)
        .expect("non-consuming eval should succeed");

    assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[5.0]);
    assert_eq!(inputs[0].as_slice::<f64>().unwrap(), &[2.0]);
    assert_eq!(inputs[1].as_slice::<f64>().unwrap(), &[3.0]);
}
