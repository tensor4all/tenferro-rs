use super::dot_general_config_retained_bytes;
use tenferro_tensor::DotGeneralConfig;

#[test]
fn dot_general_retained_bytes_exclude_inline_axes() {
    for count in [0, 4, 5, 9] {
        let config = DotGeneralConfig {
            lhs_contracting_dims: (0..count).collect(),
            rhs_contracting_dims: (0..count).rev().collect(),
            lhs_batch_dims: (count..2 * count).collect(),
            rhs_batch_dims: (count..2 * count).rev().collect(),
        };
        let expected = if count <= 4 {
            0
        } else {
            4 * count * size_of::<usize>()
        };
        assert_eq!(dot_general_config_retained_bytes(&config), Some(expected));
    }
}

/// An extension whose inference copies its input shape and then forces one axis to a
/// constant, so a zero extent can be either a copied empty batch or a derived zero.
#[derive(Clone, Debug)]
struct AxisRuleExtension {
    /// Output axis set to a literal zero, if any.
    derived_zero_axis: Option<usize>,
}

impl tenferro_ops::ext_op::ExtensionOp for AxisRuleExtension {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.axis-rule.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        hasher.write_u8(u8::from(self.derived_zero_axis.is_some()));
    }

    fn payload_eq(&self, other: &dyn tenferro_ops::ext_op::ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.derived_zero_axis == self.derived_zero_axis)
    }

    fn clone_arc(&self) -> std::sync::Arc<dyn tenferro_ops::ext_op::ExtensionOp> {
        std::sync::Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
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
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(tenferro_tensor::DType, Vec<tenferro_ops::SymDim>)>> {
        let dtype = ctx.input_dtype(0)?;
        let mut shape = ctx.input_shape(0)?.to_vec();
        if let Some(axis) = self.derived_zero_axis
            && axis < shape.len()
        {
            shape[axis] = tenferro_ops::SymDim::from(0usize);
        }
        Ok(vec![(dtype, shape)])
    }

    fn semantic_effects(&self) -> tenferro_ops::ext_op::ExtensionEffectDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> tenferro_ops::ext_op::ExtensionAliasDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionAliasDeclaration::AllFresh
    }
}

fn resolved_inputs(shape: &[usize]) -> Vec<super::ResolvedExtensionInput> {
    vec![super::ResolvedExtensionInput {
        dtype: tenferro_tensor::DType::F64,
        shape: shape.to_vec(),
    }]
}

#[test]
fn an_empty_input_axis_explains_a_copied_zero_output_axis() {
    let extension = AxisRuleExtension {
        derived_zero_axis: None,
    };
    // The caller only consults this for an extent that resolved to zero; axis 0 is
    // that extent here, and raising the empty batch axis to 1 moves it to 1.
    let inputs = resolved_inputs(&[0, 3]);
    assert!(super::empty_batch_explains_zero(&inputs, &extension, 0, 0));
}

#[test]
fn a_derived_zero_is_not_explained_by_any_empty_input_axis() {
    // Derived on axis 1: raising the empty batch axis to 1 leaves that axis at 0.
    let extension = AxisRuleExtension {
        derived_zero_axis: Some(1),
    };
    let inputs = resolved_inputs(&[0, 3]);
    assert!(!super::empty_batch_explains_zero(&inputs, &extension, 0, 1));

    // Derived with no empty input at all: the #2434 shape.
    let inputs = resolved_inputs(&[1, 3]);
    assert!(!super::empty_batch_explains_zero(&inputs, &extension, 0, 1));
}
