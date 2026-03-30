use chainrules_core::{AutodiffError, NodeId};
use tenferro_internal_frontend_core::{DynTensor, StructuredTensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
use tidu::expert::Tape;

use crate::{register_closure_rule, register_mixed_rule, register_rule, AdTensor};

mod core_value;
mod core_value_organization;
mod core_value_reverse_api;
mod dyn_ad_tensor;
mod tape_frontend;
mod tape_organization;

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

impl chainrules_core::ReverseRule<DenseTensor<f64>> for EchoTensorRule {
    fn pullback(
        &self,
        cotangent: &DenseTensor<f64>,
    ) -> chainrules_core::AdResult<Vec<(NodeId, DenseTensor<f64>)>> {
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
    ) -> chainrules_core::AdResult<Option<DenseTensor<f64>>>
    where
        DenseTensor<f64>: 't,
    {
        Ok(self
            .input_node_ids
            .iter()
            .find_map(|&node| input_tangents(node))
            .map(|tangent| {
                let value = tangent.buffer().as_slice().unwrap()[0];
                rank0_f64(value * 3.0)
            }))
    }

    fn pullback_with_tangents<'t>(
        &self,
        cotangent: &DenseTensor<f64>,
        cotangent_tangent: &DenseTensor<f64>,
        input_tangents: &dyn Fn(NodeId) -> Option<&'t DenseTensor<f64>>,
    ) -> chainrules_core::AdResult<Vec<(NodeId, DenseTensor<f64>, DenseTensor<f64>)>>
    where
        DenseTensor<f64>: 't,
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
                    rank0_f64(seed * 2.0),
                    rank0_f64(seed_tangent + tangent),
                )
            })
            .collect())
    }
}

#[test]
fn reverse_leaf_creation_preserves_tape_identity() {
    let tape = Tape::<DynTensor>::new();
    let value = AdTensor::new_reverse_leaf(rank0_f64(1.0), &tape).unwrap();

    assert!(value.tape().unwrap().same_tape(&tape));
    assert!(value.node_id().is_some());
}

#[test]
fn mixed_reverse_tapes_are_rejected() {
    let tape_a = Tape::<DynTensor>::new();
    let tape_b = Tape::<DynTensor>::new();
    let value = AdTensor::new_reverse_leaf(rank0_f64(2.0), &tape_a).unwrap();

    let err = value.ensure_reverse_leaf_on(&tape_b).unwrap_err();
    assert!(matches!(
        err,
        tenferro_internal_error::Error::MixedReverseTape { .. }
    ));
}

#[test]
fn snapshot_roundtrip_preserves_reverse_attachment() {
    let tape = Tape::<DynTensor>::new();
    let value =
        AdTensor::new_reverse_leaf_with_tangent(rank0_f64(1.0), rank0_f64(0.5), &tape).unwrap();
    let snapshot = value.snapshot().unwrap();
    let restored = AdTensor::try_from(snapshot).unwrap();

    assert!(restored.tape().unwrap().same_tape(&tape));
    assert_eq!(restored.node_id(), value.node_id());
    assert_eq!(
        restored.tangent().unwrap().buffer().as_slice().unwrap(),
        &[0.5]
    );
}

#[test]
fn reverse_leaf_zero_grad_clears_accumulated_gradients_and_hvps() {
    let tape = Tape::<DynTensor>::new();
    let value = AdTensor::new_reverse_leaf(rank0_f64(1.0), &tape).unwrap();

    value
        .accumulate_leaf_grad(structured_rank0_f64(2.0))
        .unwrap();
    value
        .accumulate_leaf_grad(structured_rank0_f64(-0.5))
        .unwrap();
    value
        .accumulate_leaf_hvp(structured_rank0_f64(3.0))
        .unwrap();
    value
        .accumulate_leaf_hvp(structured_rank0_f64(1.5))
        .unwrap();

    assert_eq!(
        value.grad().unwrap().payload().buffer().as_slice().unwrap(),
        &[1.5]
    );
    assert_eq!(
        value.hvp().unwrap().payload().buffer().as_slice().unwrap(),
        &[4.5]
    );

    value.zero_grad().unwrap();
    assert!(value.grad().is_none());
    assert!(value.hvp().is_none());
}

#[test]
fn register_rule_attaches_tensor_rule_to_reverse_tape() {
    let tape = Tape::<DynTensor>::new();
    let x = AdTensor::new_reverse_leaf(rank0_f64(2.0), &tape).unwrap();
    let y = AdTensor::new_reverse_output(rank0_f64(3.0), &tape, None).unwrap();
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
            DynTensor::from(structured_rank0_f64(1.25)),
        )
        .unwrap();
    let grad = grads
        .entries()
        .iter()
        .find_map(|(node, grad)| (*node == x_node).then_some(grad))
        .expect("registered rule should emit a grad");

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
fn registered_tensor_rule_supports_hvp_tangent_paths() {
    let tape = Tape::<DynTensor>::new();
    let x =
        AdTensor::new_reverse_leaf_with_tangent(rank0_f64(2.0), rank0_f64(0.25), &tape).unwrap();
    let y = AdTensor::new_reverse_output(rank0_f64(3.0), &tape, None).unwrap();
    let x_node = x.node_id().unwrap();
    let y_node = y.node_id().unwrap();

    register_rule::<f64>(
        &tape,
        y_node,
        Box::new(EchoTensorRule {
            input_node_ids: vec![x_node],
        }),
    );

    let mut leaf_tangents = std::collections::HashMap::new();
    leaf_tangents.insert(x_node, DynTensor::from(structured_rank0_f64(0.25)));

    let grads = tape
        .hvp(
            &y.as_tracked()
                .expect("reverse output should expose tracked value"),
            &leaf_tangents,
        )
        .unwrap();

    let grad = grads
        .gradients
        .entries()
        .iter()
        .find_map(|(node, grad)| (*node == x_node).then_some(grad))
        .expect("registered rule should emit a grad");
    let grad_tangent = grads
        .hvp
        .entries()
        .iter()
        .find_map(|(node, tangent)| (*node == x_node).then_some(tangent))
        .expect("registered rule should emit a grad tangent");

    assert_eq!(
        grad.as_f64()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[2.0]
    );
    assert_eq!(
        grad_tangent
            .as_f64()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[0.25]
    );
}

#[test]
fn registered_closure_rule_reports_hvp_not_supported() {
    let tape = Tape::<DynTensor>::new();
    let y = AdTensor::new_reverse_output(rank0_f64(3.0), &tape, None).unwrap();
    let y_node = y.node_id().unwrap();

    register_closure_rule::<f64>(&tape, y_node, vec![], Box::new(|_| Ok(Vec::new())));

    match tape.hvp(
        &y.as_tracked()
            .expect("reverse output should expose tracked value"),
        &std::collections::HashMap::new(),
    ) {
        Err(AutodiffError::HvpNotSupported) => {}
        Err(err) => panic!("unexpected HVP error: {err}"),
        Ok(_) => panic!("closure-backed VJP-only rule should not claim HVP support"),
    }
}

#[test]
fn register_mixed_rule_maps_output_dtype_to_input_dtype() {
    let tape = Tape::<DynTensor>::new();
    let x = AdTensor::new_reverse_leaf(rank0_f32(2.0), &tape).unwrap();
    let y = AdTensor::new_reverse_output(rank0_f64(3.0), &tape, None).unwrap();
    let x_node = x.node_id().unwrap();
    let y_node = y.node_id().unwrap();

    register_mixed_rule::<f64, f32>(
        &tape,
        y_node,
        vec![x_node],
        Box::new(move |cotangent| {
            let seed = cotangent.payload().buffer().as_slice().unwrap()[0] as f32;
            Ok(vec![(x_node, structured_rank0_f32(seed * 2.0))])
        }),
    );

    let grads = tape
        .pullback_with_seed(
            &y.as_tracked()
                .expect("reverse output should expose tracked value"),
            DynTensor::from(structured_rank0_f64(1.25)),
        )
        .unwrap();
    let grad = grads
        .entries()
        .iter()
        .find_map(|(node, grad)| (*node == x_node).then_some(grad))
        .expect("registered mixed rule should emit a grad");

    assert_eq!(
        grad.as_f32()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[2.5]
    );
}
