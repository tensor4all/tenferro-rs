use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::{OperationRole, ValueKey, ValueRef};
use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{Tensor, TypedTensor};
use tidu::{ADKey, PrimitiveBuilder, PrimitiveValue};

use crate::extension_runtime::ExtensionExecutor;

use super::{missing_tangent_base_key, zero_like_tensor, EagerPrimitiveBuilder};

#[test]
fn debug_summarizes_builder_without_tensor_payloads() {
    let mut backend = CpuBackend::new();
    let mut builder = EagerPrimitiveBuilder::new(&mut backend);
    let id = builder.push_tensor(Arc::new(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
    ));
    let _tensor = builder.tensor(id);

    let debug = format!("{builder:?}");

    assert!(debug.contains("EagerPrimitiveBuilder"));
    assert!(debug.contains("backend_type"));
    assert!(debug.contains("has_extension_executor: false"));
    assert!(debug.contains("results_len: 1"));
}

#[test]
fn debug_reports_extension_executor_presence() {
    let mut backend = CpuBackend::new();
    let mut executor = ExtensionExecutor::<CpuBackend>::new();
    let builder = EagerPrimitiveBuilder::with_extension_executor(&mut backend, &mut executor);

    let debug = format!("{builder:?}");

    assert!(debug.contains("has_extension_executor: true"));
}

#[test]
fn new_builder_executes_standard_primitives_without_extension_executor() {
    let mut backend = CpuBackend::new();
    let mut builder = EagerPrimitiveBuilder::new(&mut backend);
    let lhs = builder.push_tensor(Arc::new(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
    ));
    let rhs = builder.push_tensor(Arc::new(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
    ));

    let outputs = PrimitiveBuilder::add_primitive(
        &mut builder,
        StdTensorOp::Add,
        vec![PrimitiveValue::Local(lhs), PrimitiveValue::Local(rhs)],
        OperationRole::Primary,
    );

    assert_eq!(outputs.len(), 1);
    assert_eq!(
        builder.tensor(outputs[0]).as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
}

#[test]
fn missing_tangent_external_uses_zero_like_primal_fallback() {
    let mut backend = CpuBackend::new();
    let mut builder = EagerPrimitiveBuilder::new(&mut backend);
    let primal_input = TensorInputKey::User { id: 7 };
    let primal_key = ValueKey::Input(primal_input.clone());
    let tangent_key = ValueKey::Input(primal_input.tangent_of(3));
    builder.external_data.insert(
        primal_key,
        Arc::new(Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0]).unwrap()),
    );

    let outputs = PrimitiveBuilder::add_primitive(
        &mut builder,
        StdTensorOp::Neg,
        vec![PrimitiveValue::External(tangent_key.clone())],
        OperationRole::Primary,
    );

    assert_eq!(outputs.len(), 1);
    assert!(builder.external_data.contains_key(&tangent_key));
    assert_eq!(
        builder.tensor(outputs[0]).as_slice::<f64>().unwrap(),
        &[0.0, 0.0]
    );
}

#[test]
fn missing_tangent_base_key_accepts_only_input_tangent_keys() {
    let primal_input = TensorInputKey::User { id: 11 };
    let primal_key = ValueKey::Input(primal_input.clone());
    assert_eq!(missing_tangent_base_key(&primal_key), None);

    let tangent_key = ValueKey::Input(primal_input.tangent_of(5));
    assert_eq!(missing_tangent_base_key(&tangent_key), Some(primal_key));

    let mut graph = GraphBuilder::<StdTensorOp>::new();
    let input = graph.add_input(TensorInputKey::User { id: 12 });
    let output = graph.add_operation(
        StdTensorOp::Neg,
        vec![ValueRef::Local(input)],
        OperationRole::Primary,
    )[0];
    let derived_key = graph.global_key(output).clone();
    assert_eq!(missing_tangent_base_key(&derived_key), None);
}

#[test]
fn zero_like_tensor_covers_all_dtypes() {
    assert_zero_like_matches(Tensor::F32(
        TypedTensor::from_vec_col_major(vec![2], vec![1.0_f32, -2.0]).unwrap(),
    ));
    assert_zero_like_matches(Tensor::F64(
        TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap(),
    ));
    assert_zero_like_matches(Tensor::I32(
        TypedTensor::from_vec_col_major(vec![2], vec![1_i32, -2]).unwrap(),
    ));
    assert_zero_like_matches(Tensor::I64(
        TypedTensor::from_vec_col_major(vec![2], vec![1_i64, -2]).unwrap(),
    ));
    assert_zero_like_matches(Tensor::Bool(
        TypedTensor::from_vec_col_major(vec![2], vec![true, false]).unwrap(),
    ));
    assert_zero_like_matches(Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(1.0, 2.0), Complex32::new(-3.0, 4.0)],
        )
        .unwrap(),
    ));
    assert_zero_like_matches(Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)],
        )
        .unwrap(),
    ));
}

fn assert_zero_like_matches(input: Tensor) {
    let shape = input.shape().to_vec();
    let mut backend = CpuBackend::new();
    let zero = zero_like_tensor(&input, &mut backend).unwrap();

    assert_eq!(zero.shape(), shape.as_slice());
    match zero {
        Tensor::F32(tensor) => assert_eq!(tensor.as_slice().unwrap(), &[0.0_f32, 0.0]),
        Tensor::F64(tensor) => assert_eq!(tensor.as_slice().unwrap(), &[0.0_f64, 0.0]),
        Tensor::I32(tensor) => assert_eq!(tensor.as_slice().unwrap(), &[0_i32, 0]),
        Tensor::I64(tensor) => assert_eq!(tensor.as_slice().unwrap(), &[0_i64, 0]),
        Tensor::Bool(tensor) => assert_eq!(tensor.as_slice().unwrap(), &[false, false]),
        Tensor::C32(tensor) => assert_eq!(
            tensor.as_slice().unwrap(),
            &[Complex32::new(0.0, 0.0), Complex32::new(0.0, 0.0)]
        ),
        Tensor::C64(tensor) => assert_eq!(
            tensor.as_slice().unwrap(),
            &[Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
        ),
    }
}
