use chainrules_core::{AdResult, AutodiffError, NodeId, ReverseRule};
use tenferro_internal_error::Error;
use tenferro_internal_frontend_core::{DynTensor, StructuredTensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

use super::{ClosureRuleAdapter, MixedTensorRuleAdapter, TensorRuleAdapter};

fn rank0_f64(value: f64) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn rank0_f32(value: f32) -> DenseTensor<f32> {
    DenseTensor::<f32>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn structured_rank0_f64(value: f64) -> StructuredTensor<f64> {
    StructuredTensor::from(rank0_f64(value))
}

fn structured_rank0_f32(value: f32) -> StructuredTensor<f32> {
    StructuredTensor::from(rank0_f32(value))
}

struct EchoTensorRule {
    input_node_ids: Vec<NodeId>,
}

impl ReverseRule<DenseTensor<f64>> for EchoTensorRule {
    fn pullback(&self, cotangent: &DenseTensor<f64>) -> AdResult<Vec<(NodeId, DenseTensor<f64>)>> {
        let seed = cotangent.buffer().as_slice().unwrap()[0];
        Ok(self
            .input_node_ids
            .iter()
            .map(|&node| (node, rank0_f64(seed * 2.0)))
            .collect())
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.input_node_ids.clone()
    }

    fn forward_tangents<'t>(
        &self,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t DenseTensor<f64>>,
    ) -> AdResult<Option<DenseTensor<f64>>>
    where
        DenseTensor<f64>: 't,
    {
        Ok(self
            .input_node_ids
            .iter()
            .find_map(|&node| input_tangents(node))
            .cloned())
    }
}

#[test]
fn closure_rule_adapter_exposes_inputs_and_rejects_dtype_mismatch() {
    let adapter = ClosureRuleAdapter::<f64> {
        pullback_fn: Box::new(|_| Ok(Vec::new())),
        input_node_ids: vec![NodeId::new(1), NodeId::new(2)],
    };

    assert_eq!(adapter.inputs(), vec![NodeId::new(1), NodeId::new(2)]);
    match adapter.pullback(&DynTensor::from(structured_rank0_f32(1.0))) {
        Err(AutodiffError::InvalidArgument(message)) => {
            assert!(message.contains("cotangent dtype did not match"));
        }
        Err(err) => panic!("unexpected adapter error: {err}"),
        Ok(_) => panic!("closure adapter should reject mismatched cotangent dtypes"),
    }
}

#[test]
fn tensor_rule_adapter_covers_inputs_none_tangent_and_dtype_guards() {
    let adapter = TensorRuleAdapter::<f64> {
        rule: Box::new(EchoTensorRule {
            input_node_ids: vec![NodeId::new(3)],
        }),
    };

    assert_eq!(adapter.inputs(), vec![NodeId::new(3)]);
    assert!(adapter.forward_tangents(&|_| None).unwrap().is_none());

    match adapter.pullback(&DynTensor::from(structured_rank0_f32(1.0))) {
        Err(AutodiffError::InvalidArgument(message)) => {
            assert!(message.contains("cotangent dtype did not match"));
        }
        Err(err) => panic!("unexpected adapter error: {err}"),
        Ok(_) => panic!("tensor adapter should reject mismatched cotangent dtypes"),
    }

    match adapter.pullback_with_tangents(
        &DynTensor::from(structured_rank0_f32(1.0)),
        &DynTensor::from(structured_rank0_f64(1.0)),
        &|_| None,
    ) {
        Err(AutodiffError::InvalidArgument(message)) => {
            assert!(message.contains("cotangent dtype did not match"));
        }
        Err(err) => panic!("unexpected adapter error: {err}"),
        Ok(_) => panic!("tensor adapter should reject mismatched cotangent dtypes"),
    }

    match adapter.pullback_with_tangents(
        &DynTensor::from(structured_rank0_f64(1.0)),
        &DynTensor::from(structured_rank0_f32(1.0)),
        &|_| None,
    ) {
        Err(AutodiffError::InvalidArgument(message)) => {
            assert!(message.contains("cotangent_tangent dtype did not match"));
        }
        Err(err) => panic!("unexpected adapter error: {err}"),
        Ok(_) => panic!("tensor adapter should reject mismatched cotangent tangent dtypes"),
    }
}

#[test]
fn mixed_rule_adapter_exposes_inputs_and_maps_errors() {
    let adapter = MixedTensorRuleAdapter::<f64, f32> {
        rule: Box::new(|_| {
            Err(Error::InvalidAdTensor {
                message: "synthetic mixed failure".to_string(),
            })
        }),
        input_node_ids: vec![NodeId::new(4)],
    };

    assert_eq!(adapter.inputs(), vec![NodeId::new(4)]);

    match adapter.pullback(&DynTensor::from(structured_rank0_f32(1.0))) {
        Err(AutodiffError::InvalidArgument(message)) => {
            assert!(message.contains("cotangent dtype did not match"));
        }
        Err(err) => panic!("unexpected mixed adapter error: {err}"),
        Ok(_) => panic!("mixed adapter should reject mismatched output cotangent dtypes"),
    }

    match adapter.pullback(&DynTensor::from(structured_rank0_f64(1.0))) {
        Err(AutodiffError::InvalidArgument(message)) => {
            assert!(message.contains("synthetic mixed failure"));
        }
        Err(err) => panic!("unexpected mixed adapter error: {err}"),
        Ok(_) => panic!("mixed adapter should map closure failures into InvalidArgument"),
    }
}
