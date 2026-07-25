//! Coverage tests for extension identity, hashing, and arity.

use std::any::Any;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::types::ValueRef;
use computegraph::GraphOperation;

use crate::ext_op::{
    ExtensionLoweringError, ExtensionLoweringResult, ExtensionOp, ExtensionStandardLowering,
};
use crate::input_key::TensorInputKey;
use crate::std_tensor_op::StdTensorOp;
use crate::{ExtensionFamilyId, SymDim};
use tenferro_tensor::{DType, ErrorKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum WindowMode {
    Valid,
    Same,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PayloadOp {
    axis: usize,
    mode: WindowMode,
    tensor_inputs: usize,
}

#[derive(ExtensionFamilyId)]
#[tenferro_extension(namespace = "covtest", name = "macro_rule", version = 1)]
struct MacroRuleFamily;

impl ExtensionOp for PayloadOp {
    fn family_id(&self) -> &'static str {
        "covtest.payload.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.axis);
        hasher.write_u8(match self.mode {
            WindowMode::Valid => 0,
            WindowMode::Same => 1,
        });
        hasher.write_usize(self.tensor_inputs);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|op| op == self)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        self.tensor_inputs
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut crate::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct LoweringOp;

impl ExtensionOp for LoweringOp {
    fn family_id(&self) -> &'static str {
        "covtest.lowering.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut crate::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }

    fn lower_to_standard_ops(
        &self,
        _builder: &mut GraphBuilder<StdTensorOp>,
        inputs: &[ValueRef<StdTensorOp>],
        _input_dtypes: &[DType],
        _input_shapes: &[&[SymDim]],
    ) -> ExtensionLoweringResult {
        Ok(ExtensionStandardLowering::Lowered(vec![inputs[0].clone()]))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct FailingLoweringOp;

impl ExtensionOp for FailingLoweringOp {
    fn family_id(&self) -> &'static str {
        "covtest.lowering_fail.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut crate::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }

    fn lower_to_standard_ops(
        &self,
        _builder: &mut GraphBuilder<StdTensorOp>,
        _inputs: &[ValueRef<StdTensorOp>],
        _input_dtypes: &[DType],
        _input_shapes: &[&[SymDim]],
    ) -> ExtensionLoweringResult {
        Err(ExtensionLoweringError::new_with_kind(
            ErrorKind::Unsupported,
            "no standard lowering",
        ))
    }
}

fn payload_op(axis: usize, mode: WindowMode, tensor_inputs: usize) -> StdTensorOp {
    StdTensorOp::Extension(Arc::new(PayloadOp {
        axis,
        mode,
        tensor_inputs,
    }))
}

fn hash_std_tensor_op(op: &StdTensorOp) -> u64 {
    let mut hasher = DefaultHasher::new();
    op.hash(&mut hasher);
    hasher.finish()
}

fn lowering_fixture() -> (
    GraphBuilder<StdTensorOp>,
    ValueRef<StdTensorOp>,
    [DType; 1],
    [SymDim; 1],
) {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let input = ValueRef::Local(builder.add_input(TensorInputKey::User { id: 0 }));
    (builder, input, [DType::F64], [SymDim::from(2usize)])
}

#[test]
fn extension_family_id_macro_generates_stable_const() {
    assert_eq!(MacroRuleFamily::FAMILY_ID, "covtest.macro_rule.v1");
}

#[test]
fn extension_payload_equal_when_semantic_fields_match() {
    assert_eq!(
        payload_op(1, WindowMode::Valid, 1),
        payload_op(1, WindowMode::Valid, 1)
    );
}

#[test]
fn extension_payload_not_equal_when_axis_differs() {
    assert_ne!(
        payload_op(1, WindowMode::Valid, 1),
        payload_op(2, WindowMode::Valid, 1)
    );
}

#[test]
fn extension_payload_not_equal_when_mode_differs() {
    assert_ne!(
        payload_op(1, WindowMode::Valid, 1),
        payload_op(1, WindowMode::Same, 1)
    );
}

#[test]
fn extension_payload_hash_matches_for_equal_payloads() {
    let lhs = payload_op(1, WindowMode::Valid, 1);
    let rhs = payload_op(1, WindowMode::Valid, 1);
    assert_eq!(hash_std_tensor_op(&lhs), hash_std_tensor_op(&rhs));
}

#[test]
fn extension_payload_hash_changes_for_distinct_payloads() {
    let lhs = payload_op(1, WindowMode::Valid, 1);
    let rhs = payload_op(2, WindowMode::Valid, 1);
    assert_ne!(hash_std_tensor_op(&lhs), hash_std_tensor_op(&rhs));
}

#[test]
fn extension_payload_does_not_affect_tensor_input_arity() {
    assert_eq!(payload_op(1, WindowMode::Valid, 2).input_count(), 2);
    assert_eq!(payload_op(2, WindowMode::Same, 2).input_count(), 2);
}

#[test]
fn lower_to_standard_ops_reports_default_unsupported_without_error() {
    let op = PayloadOp {
        axis: 0,
        mode: WindowMode::Valid,
        tensor_inputs: 1,
    };
    let (mut builder, input, dtypes, shape) = lowering_fixture();

    let lowered = op
        .lower_to_standard_ops(&mut builder, &[input], &dtypes, &[shape.as_slice()])
        .unwrap();

    assert_eq!(lowered, ExtensionStandardLowering::Unsupported);
}

#[test]
fn lower_to_standard_ops_returns_standard_outputs() {
    let op = LoweringOp;
    let (mut builder, input, dtypes, shape) = lowering_fixture();

    let lowered = op
        .lower_to_standard_ops(
            &mut builder,
            std::slice::from_ref(&input),
            &dtypes,
            &[shape.as_slice()],
        )
        .unwrap();

    assert_eq!(lowered, ExtensionStandardLowering::Lowered(vec![input]));
}

#[test]
fn lower_to_standard_ops_preserves_lowering_error_kind() {
    let op = FailingLoweringOp;
    let (mut builder, input, dtypes, shape) = lowering_fixture();

    let error = op
        .lower_to_standard_ops(&mut builder, &[input], &dtypes, &[shape.as_slice()])
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

#[test]
fn extension_standard_lowering_has_no_legacy_option_shim() {
    let source = include_str!("../ext_op.rs");
    assert!(
        !source.contains("from_legacy"),
        "standard lowering must expose an explicit Unsupported outcome, not a legacy Option shim"
    );
    assert!(
        !source.contains("lower_to_standard_ops_typed"),
        "lower_to_standard_ops is the canonical typed lowering hook"
    );
    assert!(
        !source.contains("Option<Vec<ValueRef<StdTensorOp>>>"),
        "standard lowering must not encode unsupported as Ok(None)"
    );
}

#[test]
fn extension_op_contract_has_no_host_reference_execution_hook() {
    let source = include_str!("../ext_op.rs");
    assert!(
        !source.contains("pub trait HostReference"),
        "host-reference execution must be owned by ExtensionModule implementations, not ExtensionOp"
    );
    assert!(
        !source.contains("fn host_reference("),
        "ExtensionOp must not expose an execution fallback hook"
    );
}

#[test]
fn extension_operation_ownership_is_ad_engine_independent() {
    let source = include_str!("../ext_op.rs");
    for forbidden in [
        "ExtensionAdDispatcher",
        "dispatch_extension_linearize",
        "dispatch_extension_transpose",
        "LocalValueId",
        "OperationRole",
        "PrimitiveTransposeInput",
        "ValueKey",
        "use tidu",
    ] {
        assert!(
            !source.contains(forbidden),
            "extension-op ownership must not mention AD-engine surface {forbidden}"
        );
    }
}
