use chainrules_core::AutodiffError;
use num_complex::Complex64;
use tenferro_tensor::{MemoryOrder, Tensor};

use super::*;
use crate::structured::StructuredTensor;
use crate::AdTensor;

fn dense_structured<T: tenferro_algebra::Scalar>(tensor: Tensor<T>) -> StructuredTensor<T> {
    StructuredTensor(tenferro_tensor::StructuredTensor::from_dense(tensor))
}

fn f64_scalar(value: f64) -> DynTensor {
    DynTensor::from(dense_structured(
        Tensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap(),
    ))
}

fn f64_tensor_scalar(value: f64) -> Tensor<f64> {
    Tensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

struct EchoTensorRule {
    input_node_ids: Vec<NodeId>,
}

impl chainrules_core::ReverseRule<Tensor<f64>> for EchoTensorRule {
    fn pullback(
        &self,
        cotangent: &Tensor<f64>,
    ) -> chainrules_core::AdResult<Vec<(NodeId, Tensor<f64>)>> {
        let seed = cotangent.buffer().as_slice().unwrap()[0];
        Ok(self
            .input_node_ids
            .iter()
            .map(|&node| (node, f64_tensor_scalar(seed * 2.0)))
            .collect())
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.input_node_ids.clone()
    }

    fn forward_tangents<'t>(
        &self,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t Tensor<f64>>,
    ) -> chainrules_core::AdResult<Option<Tensor<f64>>>
    where
        Tensor<f64>: 't,
    {
        Ok(self
            .input_node_ids
            .iter()
            .find_map(|&node| input_tangents(node))
            .map(|tangent| {
                let value = tangent.buffer().as_slice().unwrap()[0];
                f64_tensor_scalar(value * 3.0)
            }))
    }

    fn pullback_with_tangents<'t>(
        &self,
        cotangent: &Tensor<f64>,
        cotangent_tangent: &Tensor<f64>,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t Tensor<f64>>,
    ) -> chainrules_core::AdResult<Vec<(NodeId, Tensor<f64>, Tensor<f64>)>>
    where
        Tensor<f64>: 't,
    {
        let seed = cotangent.buffer().as_slice().unwrap()[0];
        let seed_tangent = cotangent_tangent.buffer().as_slice().unwrap()[0];
        Ok(self
            .input_node_ids
            .iter()
            .map(|&node| {
                let tangent = input_tangents(node)
                    .map(|value| value.buffer().as_slice().unwrap()[0])
                    .unwrap_or(0.0);
                (
                    node,
                    f64_tensor_scalar(seed * 2.0),
                    f64_tensor_scalar(seed_tangent + tangent),
                )
            })
            .collect())
    }
}

#[test]
fn closure_rule_adapter_rejects_cotangent_dtype_mismatch_and_reports_empty_inputs() {
    let adapter = ClosureRuleAdapter::<f64> {
        pullback_fn: Box::new(|_| Ok(Vec::new())),
        input_node_ids: vec![],
    };
    let cotangent = DynTensor::from(dense_structured(
        Tensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, -2.0)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    ));

    match adapter.pullback(&cotangent) {
        Err(AutodiffError::InvalidArgument(message)) => {
            assert!(message.contains("cotangent dtype did not match"));
        }
        Err(err) => panic!("unexpected adapter error: {err}"),
        Ok(_) => panic!("dtype mismatch should be rejected"),
    }
    assert!(adapter.inputs().is_empty());
}

#[test]
fn closure_rule_adapter_pullback_with_tangents_reports_hvp_not_supported() {
    let adapter = ClosureRuleAdapter::<f64> {
        pullback_fn: Box::new(|_| Ok(Vec::new())),
        input_node_ids: vec![],
    };

    let input_tangents_fn = |_: NodeId| -> Option<&DynTensor> { None };
    match adapter.pullback_with_tangents(&f64_scalar(1.0), &f64_scalar(0.5), &input_tangents_fn) {
        Err(AutodiffError::HvpNotSupported) => {}
        Err(err) => panic!("unexpected HVP error: {err}"),
        Ok(_) => panic!("VJP-only rule should not claim HVP support"),
    }
}

#[test]
fn mixed_tensor_rule_adapter_rejects_cotangent_dtype_mismatch_and_reports_empty_inputs() {
    let adapter = MixedTensorRuleAdapter::<f64, Complex64> {
        rule: Box::new(|_| Ok(Vec::new())),
        input_node_ids: vec![],
    };
    let cotangent = DynTensor::from(dense_structured(
        Tensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, -2.0)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    ));

    match adapter.pullback(&cotangent) {
        Err(AutodiffError::InvalidArgument(message)) => {
            assert!(message.contains("cotangent dtype did not match"));
        }
        Err(err) => panic!("unexpected adapter error: {err}"),
        Ok(_) => panic!("dtype mismatch should be rejected"),
    }
    assert!(adapter.inputs().is_empty());
}

#[test]
fn mixed_tensor_rule_adapter_pullback_converts_gradient_dtype() {
    let adapter = MixedTensorRuleAdapter::<f64, Complex64> {
        rule: Box::new(|cotangent| {
            let seed = cotangent.payload().buffer().as_slice().unwrap()[0];
            Ok(vec![(
                NodeId::new(7),
                dense_structured(
                    Tensor::<Complex64>::from_slice(
                        &[Complex64::new(seed, -seed)],
                        &[],
                        MemoryOrder::ColumnMajor,
                    )
                    .unwrap(),
                ),
            )])
        }),
        input_node_ids: vec![NodeId::new(7)],
    };

    let cotangent = f64_scalar(2.5);
    let grads = adapter.pullback(&cotangent).unwrap();
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].0, NodeId::new(7));
    assert_eq!(grads[0].1.scalar_type(), crate::ScalarType::C64);
    assert_eq!(
        grads[0]
            .1
            .as_c64()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(2.5, -2.5)]
    );
}

#[test]
fn tensor_rule_adapter_pullback_converts_tensor_gradients_to_dyn_tensor() {
    let node = NodeId::new(11);
    let adapter = TensorRuleAdapter::<f64> {
        rule: Box::new(EchoTensorRule {
            input_node_ids: vec![node],
        }),
    };

    let grads = adapter.pullback(&f64_scalar(2.5)).unwrap();
    assert_eq!(adapter.inputs(), vec![node]);
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].0, node);
    assert_eq!(grads[0].1.scalar_type(), crate::ScalarType::F64);
    assert_eq!(
        grads[0]
            .1
            .as_f64()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[5.0]
    );
}

#[test]
fn tensor_rule_adapter_forward_tangents_converts_dyn_tensor_inputs() {
    let node = NodeId::new(13);
    let adapter = TensorRuleAdapter::<f64> {
        rule: Box::new(EchoTensorRule {
            input_node_ids: vec![node],
        }),
    };
    let tangent = f64_scalar(1.5);

    let tangent_out = adapter
        .forward_tangents(&|candidate| (candidate == node).then_some(&tangent))
        .unwrap()
        .expect("rule should produce a tangent");

    assert_eq!(tangent_out.scalar_type(), crate::ScalarType::F64);
    assert_eq!(
        tangent_out
            .as_f64()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[4.5]
    );
}

#[test]
fn tensor_rule_adapter_pullback_with_tangents_converts_both_outputs() {
    let node = NodeId::new(17);
    let adapter = TensorRuleAdapter::<f64> {
        rule: Box::new(EchoTensorRule {
            input_node_ids: vec![node],
        }),
    };
    let input_tangent = f64_scalar(0.25);

    let grads = adapter
        .pullback_with_tangents(&f64_scalar(2.0), &f64_scalar(0.5), &|candidate| {
            (candidate == node).then_some(&input_tangent)
        })
        .unwrap();

    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].0, node);
    assert_eq!(
        grads[0]
            .1
            .as_f64()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[4.0]
    );
    assert_eq!(
        grads[0]
            .2
            .as_f64()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[0.75]
    );
}

#[test]
fn register_rule_attaches_tensor_rule_to_reverse_tape() {
    let tape = Tape::<crate::DynTensor>::new();
    let x = AdTensor::new_reverse_leaf(f64_tensor_scalar(2.0), &tape).unwrap();
    let y = AdTensor::new_reverse_output(f64_tensor_scalar(3.0), &tape, None).unwrap();
    let x_node = x.node_id().unwrap();
    let y_node = y.node_id().unwrap();

    register_rule::<f64>(
        &tape,
        y_node,
        Box::new(EchoTensorRule {
            input_node_ids: vec![x_node],
        }),
    );

    let grads = tape
        .pullback_with_seed(
            &y.as_tracked()
                .expect("reverse output should expose tracked value"),
            crate::DynTensor::from(dense_structured(f64_tensor_scalar(1.25))),
        )
        .unwrap();
    let grad = grads
        .entries()
        .iter()
        .find_map(|(node, grad)| (*node == x_node).then_some(grad))
        .expect("registered rule should emit a grad");
    assert_eq!(grad.scalar_type(), crate::ScalarType::F64);
    assert_eq!(
        grad.as_f64()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[2.5]
    );
}

#[test]
fn register_mixed_rule_attaches_dtype_converting_rule_to_reverse_tape() {
    let tape = Tape::<crate::DynTensor>::new();
    let x = AdTensor::new_reverse_leaf(
        Tensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, -1.0)],
            &[],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
        &tape,
    )
    .unwrap();
    let y = AdTensor::new_reverse_output(f64_tensor_scalar(3.0), &tape, None).unwrap();
    let x_node = x.node_id().unwrap();
    let y_node = y.node_id().unwrap();

    register_mixed_rule::<f64, Complex64>(
        &tape,
        y_node,
        vec![x_node],
        Box::new(move |cotangent| {
            let seed = cotangent.payload().buffer().as_slice().unwrap()[0];
            Ok(vec![(
                x_node,
                dense_structured(
                    Tensor::<Complex64>::from_slice(
                        &[Complex64::new(seed, -seed)],
                        &[],
                        MemoryOrder::ColumnMajor,
                    )
                    .unwrap(),
                ),
            )])
        }),
    );

    let grads = tape
        .pullback_with_seed(
            &y.as_tracked()
                .expect("reverse output should expose tracked value"),
            crate::DynTensor::from(dense_structured(f64_tensor_scalar(0.75))),
        )
        .unwrap();
    let grad = grads
        .entries()
        .iter()
        .find_map(|(node, grad)| (*node == x_node).then_some(grad))
        .expect("mixed rule should emit a grad");
    assert_eq!(grad.scalar_type(), crate::ScalarType::C64);
    assert_eq!(
        grad.as_c64()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(0.75, -0.75)]
    );
}
