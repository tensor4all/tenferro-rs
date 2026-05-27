use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_core_ops::{all_primitive_descriptors, PrimitiveOpKind};
use tenferro_tensor::{CompareDir, DType, Tensor};

use crate::ext_op::ExtensionOp;
use crate::std_tensor_op::StdTensorOp;
use crate::SymDim;

#[derive(Clone, Debug)]
struct CatalogExt;

impl ExtensionOp for CatalogExt {
    fn family_id(&self) -> &'static str {
        "catalog.ext.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<CatalogExt>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn n_outputs(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        vec![(input_dtypes[0], input_shapes[0].to_vec())]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        Ok(vec![inputs[0].clone()])
    }
}

#[test]
fn std_tensor_op_maps_core_ops_to_catalog_kinds() {
    assert_eq!(
        StdTensorOp::Add.primitive_kind(),
        Some(PrimitiveOpKind::Add)
    );
    assert_eq!(
        StdTensorOp::Compare(CompareDir::Lt).primitive_kind(),
        Some(PrimitiveOpKind::Compare)
    );
    assert_eq!(
        StdTensorOp::Convert {
            from: DType::F32,
            to: DType::F64,
        }
        .primitive_kind(),
        Some(PrimitiveOpKind::Convert)
    );
}

#[test]
fn extension_ops_do_not_claim_core_kind() {
    let op = StdTensorOp::Extension(Arc::new(CatalogExt));

    assert_eq!(op.primitive_kind(), None);
}

#[test]
fn every_catalog_descriptor_has_a_std_tensor_op_variant() {
    for descriptor in all_primitive_descriptors() {
        let op = StdTensorOp::sample_from_kind(descriptor.kind);
        assert_eq!(
            op.primitive_kind(),
            Some(descriptor.kind),
            "StdTensorOp variant for {:?} disagrees with descriptor",
            descriptor.kind
        );
    }
}

#[test]
fn extension_ad_api_uses_linearize_and_transpose_terminology() {
    let source = include_str!("../ext_op.rs");

    for forbidden in [
        "ExtensionChainRule",
        "FruleBuilder",
        "RRuleBuilder",
        "register_extension_chain_rule",
        "frule",
        "rrule",
    ] {
        assert!(
            !source.contains(forbidden),
            "extension AD API should not expose ChainRules-style `{forbidden}` terminology"
        );
    }
}
