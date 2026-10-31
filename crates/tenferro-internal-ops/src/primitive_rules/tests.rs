use super::ResidualSpec;

#[test]
fn residual_spec_builders_and_queries_cover_always_on_api() {
    let none = ResidualSpec::none();
    assert!(none.is_empty());
    assert!(!none.declares_input(0));
    assert!(!none.declares_output(0));

    let single = ResidualSpec::input(1).with_output(0);
    assert!(!single.declares_input(0));
    assert!(single.declares_input(1));
    assert!(single.declares_output(0));
    assert!(!single.is_empty());

    let chained = ResidualSpec::output(2).with_input(0).with_input(3);
    assert!(chained.declares_input(0) && chained.declares_input(3));
    assert!(!chained.declares_input(1));
    assert!(chained.declares_output(2));

    let all_in = ResidualSpec::all_inputs();
    assert!(all_in.declares_input(0) && all_in.declares_input(63));
    assert!(!all_in.declares_output(0));
    let all_out = ResidualSpec::all_outputs();
    assert!(all_out.declares_output(0) && all_out.declares_output(63));
    assert!(!all_out.declares_input(0));

    let both = ResidualSpec::all_inputs().with_all_outputs();
    assert!(both.declares_input(63) && both.declares_output(63));
    assert!(!both.declares_input(64) && !both.declares_output(64));

    assert!(ResidualSpec::default().is_empty());
}

#[cfg(feature = "autodiff")]
#[test]
fn residual_spec_builders_and_queries() {
    // `none` declares nothing.
    let none = ResidualSpec::none();
    assert!(none.is_empty());
    assert!(!none.declares_input(0));
    assert!(!none.declares_output(0));

    // Single input / output builders.
    let single = ResidualSpec::input(1).with_output(0);
    assert!(!single.declares_input(0));
    assert!(single.declares_input(1));
    assert!(single.declares_output(0));
    assert!(!single.is_empty());

    // Chained builders accumulate bits.
    let chained = ResidualSpec::output(2).with_input(0).with_input(3);
    assert!(chained.declares_input(0));
    assert!(chained.declares_input(3));
    assert!(!chained.declares_input(1));
    assert!(chained.declares_output(2));

    // All-inputs / all-outputs builders.
    let all_in = ResidualSpec::all_inputs();
    assert!(all_in.declares_input(0));
    assert!(all_in.declares_input(63));
    assert!(!all_in.declares_output(0));
    let all_out = ResidualSpec::all_outputs();
    assert!(all_out.declares_output(0));
    assert!(all_out.declares_output(63));
    assert!(!all_out.declares_input(0));

    // with_all_* additive builders.
    let both = ResidualSpec::all_inputs().with_all_outputs();
    assert!(both.declares_input(0) && both.declares_input(63));
    assert!(both.declares_output(0) && both.declares_output(63));

    // Out-of-range indices are never declared.
    assert!(!both.declares_input(64));
    assert!(!both.declares_output(64));
}

#[cfg(feature = "autodiff")]
#[test]
fn residual_spec_masks_on_representative_rules() {
    // add/sub/neg declare no tensor residuals; mul/div declare all inputs.
    assert!(super::residual_spec::residual_spec_for_kind(PrimitiveOpKind::Add).is_empty());
    assert!(super::residual_spec::residual_spec_for_kind(PrimitiveOpKind::Neg).is_empty());
    assert!(super::residual_spec::residual_spec_for_kind(PrimitiveOpKind::Mul).declares_input(0));
    assert!(super::residual_spec::residual_spec_for_kind(PrimitiveOpKind::Div).declares_input(1));
}

#[cfg(feature = "autodiff")]
#[test]
fn all_residual_masks_match_the_pre_retirement_registry() {
    use tenferro_core_ops::{all_primitive_descriptors, PrimitiveOpKind};
    let expected = [
        (PrimitiveOpKind::Add, 0u64, 0u64),
        (PrimitiveOpKind::Sub, 0u64, 0u64),
        (PrimitiveOpKind::Mul, 18446744073709551615u64, 0u64),
        (PrimitiveOpKind::Neg, 0u64, 0u64),
        (PrimitiveOpKind::Conj, 0u64, 0u64),
        (PrimitiveOpKind::Div, 18446744073709551615u64, 0u64),
        (PrimitiveOpKind::Rem, 0u64, 0u64),
        (PrimitiveOpKind::Abs, 1u64, 0u64),
        (PrimitiveOpKind::Sign, 0u64, 0u64),
        (PrimitiveOpKind::Maximum, 18446744073709551615u64, 0u64),
        (PrimitiveOpKind::Minimum, 18446744073709551615u64, 0u64),
        (PrimitiveOpKind::Compare, 0u64, 0u64),
        (PrimitiveOpKind::Select, 1u64, 0u64),
        (PrimitiveOpKind::Clamp, 18446744073709551615u64, 0u64),
        (PrimitiveOpKind::Exp, 1u64, 0u64),
        (PrimitiveOpKind::Log, 1u64, 0u64),
        (PrimitiveOpKind::Sin, 1u64, 0u64),
        (PrimitiveOpKind::Cos, 1u64, 0u64),
        (PrimitiveOpKind::Tanh, 1u64, 0u64),
        (PrimitiveOpKind::Sqrt, 1u64, 0u64),
        (PrimitiveOpKind::Rsqrt, 1u64, 0u64),
        (PrimitiveOpKind::Pow, 18446744073709551615u64, 0u64),
        (PrimitiveOpKind::Expm1, 1u64, 0u64),
        (PrimitiveOpKind::Log1p, 1u64, 0u64),
        (PrimitiveOpKind::Erf, 1u64, 0u64),
        (PrimitiveOpKind::DotGeneral, 18446744073709551615u64, 0u64),
        (PrimitiveOpKind::ReduceSum, 1u64, 0u64),
        (PrimitiveOpKind::ReduceSumSquares, 1u64, 0u64),
        (PrimitiveOpKind::ReduceProd, 1u64, 0u64),
        (PrimitiveOpKind::ReduceMax, 1u64, 0u64),
        (PrimitiveOpKind::ReduceMin, 1u64, 0u64),
        (PrimitiveOpKind::Transpose, 0u64, 0u64),
        (PrimitiveOpKind::Reshape, 1u64, 0u64),
        (PrimitiveOpKind::BroadcastInDim, 1u64, 0u64),
        (PrimitiveOpKind::Convert, 0u64, 0u64),
        (PrimitiveOpKind::ExtractDiag, 1u64, 0u64),
        (PrimitiveOpKind::EmbedDiag, 0u64, 0u64),
        (PrimitiveOpKind::Tril, 0u64, 0u64),
        (PrimitiveOpKind::Triu, 0u64, 0u64),
        (PrimitiveOpKind::Gather, 18446744073709551615u64, 0u64),
        (
            PrimitiveOpKind::GatherDynamicSliceSizes,
            18446744073709551615u64,
            0u64,
        ),
        (PrimitiveOpKind::Scatter, 18446744073709551615u64, 0u64),
        (PrimitiveOpKind::Slice, 1u64, 0u64),
        (PrimitiveOpKind::DynamicSlice, 18446744073709551615u64, 0u64),
        (
            PrimitiveOpKind::DynamicUpdateSlice,
            18446744073709551615u64,
            0u64,
        ),
        (PrimitiveOpKind::Pad, 1u64, 0u64),
        (PrimitiveOpKind::Concatenate, 18446744073709551615u64, 0u64),
        (PrimitiveOpKind::Reverse, 0u64, 0u64),
        (PrimitiveOpKind::ShapeOf, 0u64, 0u64),
        (
            PrimitiveOpKind::DynamicTruncate,
            18446744073709551615u64,
            0u64,
        ),
        (PrimitiveOpKind::PadToMatch, 18446744073709551615u64, 0u64),
        (PrimitiveOpKind::Constant, 0u64, 0u64),
    ];
    assert_eq!(expected.len(), PrimitiveOpKind::COUNT);
    for (descriptor, (kind, input_mask, output_mask)) in
        all_primitive_descriptors().iter().zip(expected)
    {
        assert_eq!(descriptor.kind, kind);
        let actual = super::residual_spec::residual_spec_for_kind(kind);
        assert_eq!(
            (actual.input_mask, actual.output_mask),
            (input_mask, output_mask),
            "{kind:?}"
        );
    }
}

#[cfg(feature = "autodiff")]
use tenferro_core_ops::PrimitiveOpKind;

#[cfg(feature = "autodiff")]
#[test]
fn rule_errors_retain_kind_and_diagnostics() {
    use super::{ADRuleError, ADRuleKind};

    let unsupported = ADRuleError::unsupported("extension.family", ADRuleKind::Jvp);
    assert_eq!(unsupported.rule(), ADRuleKind::Jvp);
    assert_eq!(
        unsupported.to_string(),
        "unsupported AD rule Jvp for extension.family"
    );
    let invalid =
        ADRuleError::invalid_input("mul", ADRuleKind::Transpose, "missing primal metadata");
    assert_eq!(invalid.rule(), ADRuleKind::Transpose);
    assert_eq!(
        invalid.to_string(),
        "invalid AD rule input for mul (Transpose): missing primal metadata"
    );
}

#[cfg(feature = "autodiff")]
mod extension_residual {
    use crate::{ext_op::ExtensionOp, std_tensor_op::StdTensorOp, SymDim};
    use std::{any::Any, hash::Hasher, sync::Arc};
    use tenferro_tensor::DType;
    #[derive(Clone, Debug)]
    struct RuleOnlyExt {
        family: &'static str,
    }

    impl ExtensionOp for RuleOnlyExt {
        fn family_id(&self) -> &'static str {
            self.family
        }

        fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

        fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
            other
                .as_any()
                .downcast_ref::<RuleOnlyExt>()
                .is_some_and(|rhs| rhs.family == self.family)
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
    }

    #[test]
    fn extensions_have_no_core_residual_declaration() {
        let op = StdTensorOp::Extension(Arc::new(RuleOnlyExt {
            family: "stdtensor.missing_rule.v1",
        }));
        assert_eq!(crate::ad::primitive_residual_spec(&op), None);
    }
}
