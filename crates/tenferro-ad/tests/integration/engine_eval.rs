use tenferro_cpu::CpuBackend;
use tenferro_runtime::{DType, GraphCompiler, GraphExecutor, Tensor, TracedTensor, TypedTensor};

#[test]
fn graph_execution_with_borrowed_inputs_preserves_caller_tensors() {
    let lhs_value = TracedTensor::input_symbolic_shape(DType::F64, 0).unwrap();
    let rhs_value = TracedTensor::input_symbolic_shape(DType::F64, 0).unwrap();
    let sum = (&lhs_value + &rhs_value).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(
            &sum,
            &[(&lhs_value, DType::F64, &[]), (&rhs_value, DType::F64, &[])],
        )
        .unwrap();
    let lhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![2.0]).unwrap());
    let rhs = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![3.0]).unwrap());

    let mut engine = GraphExecutor::new(CpuBackend::with_threads(1).unwrap());
    let output = engine
        .run_with_inputs(&program, &[&lhs, &rhs])
        .expect("borrowed graph execution should succeed");

    assert_eq!(output.as_slice::<f64>().unwrap(), &[5.0]);
    assert_eq!(lhs.as_slice::<f64>().unwrap(), &[2.0]);
    assert_eq!(rhs.as_slice::<f64>().unwrap(), &[3.0]);
}
