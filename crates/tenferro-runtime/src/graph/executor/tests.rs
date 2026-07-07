use tenferro_cpu::CpuBackend;
use tenferro_tensor::{DType, Tensor, TensorRead, TensorValue, TypedTensor};

use super::GraphExecutor;
use crate::{Error, GraphCompiler, TracedTensor};

#[test]
fn borrowed_input_execution_retains_executor_slot_workspace_capacity() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[4])])
        .unwrap();
    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let mut executor = GraphExecutor::new(CpuBackend::new());

    assert_eq!(executor.borrowed_slot_workspace_capacity, 0);

    let outputs = executor
        .run_many_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&input))])
        .unwrap();

    assert_eq!(outputs[0].as_slice::<f64>().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
    assert_eq!(input.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert!(
        executor.borrowed_slot_workspace_capacity >= program.exec.n_slots,
        "borrowed execution should retain reusable slot capacity; capacity={}, n_slots={}",
        executor.borrowed_slot_workspace_capacity,
        program.exec.n_slots
    );
}

#[test]
fn borrowed_input_value_execution_retains_workspace_and_lazy_output() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let y = x.transpose(&[1, 0]).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2, 2])])
        .unwrap();
    let input =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let outputs = executor
        .run_many_values_with_input_reads(&program, &[(&x, TensorRead::from_tensor(&input))])
        .unwrap();

    assert!(matches!(outputs[0], TensorValue::View(_)));
    assert_eq!(input.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert!(
        executor.borrowed_slot_workspace_capacity >= program.exec.n_slots,
        "borrowed value execution should retain reusable slot capacity; capacity={}, n_slots={}",
        executor.borrowed_slot_workspace_capacity,
        program.exec.n_slots
    );
}

#[test]
fn single_value_wrappers_preserve_lazy_outputs_and_debug_state() {
    let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let y = x.transpose(&[1, 0]).unwrap();
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());

    let debug = format!("{executor:?}");
    assert!(debug.contains("GraphExecutor"));
    assert!(debug.contains("backend_type"));

    let value = executor.run_value(&program).unwrap();
    assert!(matches!(value, TensorValue::View(_)));
    assert_eq!(value.shape(), &[2, 2]);

    let placeholder = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    let transposed = placeholder.transpose(&[1, 0]).unwrap();
    let program = compiler
        .compile_with_input_specs(&transposed, &[(&placeholder, DType::F64, &[2, 2])])
        .unwrap();
    let bound =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());

    let value = executor
        .run_value_with_inputs(&program, &[(&placeholder, &bound)])
        .unwrap();
    assert!(matches!(value, TensorValue::View(_)));
    assert_eq!(value.shape(), &[2, 2]);

    let value = executor
        .run_value_with_input_reads(&program, &[(&placeholder, TensorRead::from_tensor(&bound))])
        .unwrap();
    assert!(matches!(value, TensorValue::View(_)));
    assert_eq!(value.shape(), &[2, 2]);

    executor.reclaim_value_outputs(vec![TensorValue::from_tensor(
        Tensor::from_vec_col_major(vec![1], vec![9.0_f64]).unwrap(),
    )]);
}

#[test]
fn deferred_zero_tensor_covers_supported_dtypes_and_overflow() {
    for dtype in [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C32,
        DType::C64,
    ] {
        let zero = super::zeros_tensor(dtype, vec![2]).unwrap();
        assert_eq!(zero.dtype(), dtype);
        assert_eq!(zero.shape(), &[2]);
    }

    let err = super::zeros_tensor(DType::Bool, vec![usize::MAX, 2]).unwrap_err();
    assert!(matches!(err, Error::InvalidCompiledGraph { .. }));
}

#[test]
fn borrowed_input_workspace_does_not_retype_static_slot_vec() {
    let source = include_str!("../executor.rs");

    assert!(
        !source.contains("Vec::from_raw_parts"),
        "borrowed input workspace must not retype Vec allocations across ExecSlot lifetimes"
    );
    assert!(
        !source.contains("BorrowedSlotWorkspace"),
        "borrowed input execution should use a lifetime-local workspace"
    );
}

#[test]
fn compile_with_input_specs_rejects_computed_placeholder_specs() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = (&x + &x).unwrap();
    let mut compiler = GraphCompiler::new();

    let err = compiler
        .compile_with_input_specs(&y, &[(&y, DType::F64, &[2])])
        .unwrap_err();

    assert!(matches!(err, Error::UnexpectedBinding { binding_index: 0 }));
}
