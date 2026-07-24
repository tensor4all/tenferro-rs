//! Coverage tests for extension identity, hashing, and arity.

use std::any::Any;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use computegraph::GraphOperation;

use crate::ext_op::ExtensionOp;
use crate::std_tensor_op::StdTensorOp;
use crate::{ExtensionFamilyId, SymDim};
use tenferro_tensor::DType;

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
