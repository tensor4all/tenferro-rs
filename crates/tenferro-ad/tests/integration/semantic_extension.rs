use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_ad::semantic_extension::{
    AdValue, ResidualSpec, SemanticAdError, SemanticExtensionRegistryError,
    SemanticExtensionRuleSet, SemanticLinearTransposeRequest, SemanticLinearTransposeRule,
    SemanticLinearizeRequest, SemanticLinearizeResult, SemanticLinearizeRule,
    SemanticPrimalVjpRequest, SemanticPrimalVjpRule,
};
use tenferro_ad::AdContext;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{
    ExtensionAliasDeclaration, ExtensionEffect, ExtensionEffectAccess, ExtensionEffectDeclaration,
    ExtensionOp,
};
use tenferro_ops::{ExtensionShapeContext, SymDim};
use tenferro_runtime::program::{
    CoreSemanticOp, ProgramInputSpec, SemanticOpRef, SemanticProgramBuilder,
};
use tenferro_tensor::DType;

#[derive(Clone, Debug)]
struct IdentityExtension {
    effectful: bool,
}

impl ExtensionOp for IdentityExtension {
    fn family_id(&self) -> &'static str {
        "tenferro-ad.semantic-identity.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(u8::from(self.effectful));
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.effectful == self.effectful)
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

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        static EFFECTS: [ExtensionEffect; 1] = [ExtensionEffect {
            family: "tenferro-ad.semantic-test-state.v1",
            key: 0,
            access: ExtensionEffectAccess::Read,
        }];
        if self.effectful {
            ExtensionEffectDeclaration::Declared(&EFFECTS)
        } else {
            ExtensionEffectDeclaration::Declared(&[])
        }
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        context: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(
            context.input_dtype(0)?,
            context.input_shape(0)?.to_vec(),
        )])
    }
}

#[derive(Debug)]
struct IdentityRule;

impl SemanticLinearizeRule for IdentityRule {
    fn family_id(&self) -> &'static str {
        "tenferro-ad.semantic-identity.v1"
    }

    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        assert_eq!(request.primal_inputs().len(), 1);
        assert_eq!(request.primal_outputs().len(), 1);
        assert_eq!(request.active_outputs(), &[true]);
        assert_eq!(
            request.provenance().label(),
            Some("tenferro-ad.semantic-identity.v1")
        );
        let tangent = match request.tangent_inputs()[0] {
            AdValue::Absent => AdValue::Absent,
            AdValue::Value(value) => {
                AdValue::Value(builder.add_op(CoreSemanticOp::Neg, &[value])?[0])
            }
        };
        Ok(SemanticLinearizeResult::new(
            [tangent],
            request.primal_outputs().iter().copied(),
        ))
    }
}

impl SemanticLinearTransposeRule for IdentityRule {
    fn family_id(&self) -> &'static str {
        "tenferro-ad.semantic-identity.v1"
    }

    fn residual_mask(&self) -> ResidualSpec {
        // The identity transpose only forwards cotangents; it reads no primal
        // tensor, though linearize may save the output as a residual.
        ResidualSpec::none()
    }

    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        _builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        assert_eq!(request.active_inputs(), &[true]);
        assert_eq!(request.residuals().len(), 1);
        Ok(request.cotangent_outputs().into())
    }
}

impl SemanticPrimalVjpRule for IdentityRule {
    fn family_id(&self) -> &'static str {
        "tenferro-ad.semantic-identity.v1"
    }

    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::none()
    }

    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        _builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        assert_eq!(request.active_inputs(), &[true]);
        Ok(request.cotangent_outputs().into())
    }
}

fn extension_view(
    effectful: bool,
) -> (
    Arc<tenferro_runtime::program::SemanticProgram>,
    tenferro_runtime::program::ProgramValue,
    tenferro_runtime::program::ProgramValue,
) {
    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let output = builder
        .add_extension(Arc::new(IdentityExtension { effectful }), &[input])
        .unwrap()[0];
    let frozen = builder.finish(&[output]).unwrap();
    (frozen.program, input, output)
}

#[test]
fn semantic_rule_traits_are_object_safe_and_registry_is_role_specific() {
    fn assert_object_safe(
        _: Arc<dyn SemanticLinearizeRule>,
        _: Arc<dyn SemanticLinearTransposeRule>,
        _: Arc<dyn SemanticPrimalVjpRule>,
    ) {
    }
    assert_object_safe(
        Arc::new(IdentityRule),
        Arc::new(IdentityRule),
        Arc::new(IdentityRule),
    );

    let mut rules = SemanticExtensionRuleSet::new();
    rules.register_linearize(Arc::new(IdentityRule)).unwrap();
    rules
        .register_linear_transpose(Arc::new(IdentityRule))
        .unwrap();
    rules.register_primal_vjp(Arc::new(IdentityRule)).unwrap();
    assert!(rules
        .lookup_linearize("tenferro-ad.semantic-identity.v1")
        .is_some());
    assert!(matches!(
        rules.register_linearize(Arc::new(IdentityRule)),
        Err(SemanticExtensionRegistryError::DuplicateRule { .. })
    ));
}

#[test]
fn semantic_linearize_validates_order_activity_residuals_and_builder_ownership() {
    let (program, _, _) = extension_view(false);
    let operation = program.operations().next().unwrap();
    assert!(matches!(operation.op(), SemanticOpRef::Extension(_)));

    let mut builder = SemanticProgramBuilder::new();
    let primal_input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let primal_output = builder
        .add_op(CoreSemanticOp::Neg, &[primal_input])
        .unwrap()[0];
    let tangent_input = builder
        .add_op(CoreSemanticOp::Neg, &[primal_input])
        .unwrap()[0];

    let rules = SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(IdentityRule))
        .unwrap();
    let result = rules
        .linearize_operation(
            operation,
            &[primal_input],
            &[primal_output],
            &[AdValue::Value(tangent_input)],
            &[true],
            &mut builder,
        )
        .unwrap();
    assert!(matches!(result.tangent_outputs()[0], AdValue::Value(_)));
    assert_eq!(result.residuals(), &[primal_output]);

    let mut foreign_builder = SemanticProgramBuilder::new();
    let foreign = foreign_builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    assert!(matches!(
        rules.linearize_operation(
            operation,
            &[foreign],
            &[primal_output],
            &[AdValue::Value(tangent_input)],
            &[true],
            &mut builder,
        ),
        Err(SemanticAdError::ForeignValue {
            field: "primal_inputs",
            index: 0,
        })
    ));
}

#[test]
fn semantic_dispatch_rejects_effects_before_calling_rules() {
    let (program, _, _) = extension_view(true);
    let operation = program.operations().next().unwrap();
    let mut builder = SemanticProgramBuilder::new();
    let value = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    let rules = SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(IdentityRule))
        .unwrap();

    assert!(matches!(
        rules.linearize_operation(
            operation,
            &[value],
            &[value],
            &[AdValue::Absent],
            &[true],
            &mut builder,
        ),
        Err(SemanticAdError::EffectfulExtension { .. })
    ));
}

#[test]
fn ad_context_owns_cloned_semantic_rule_sets_and_rejects_duplicate_inputs_atomically() {
    let rules = SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(IdentityRule))
        .unwrap();
    let ad = AdContext::builder()
        .with_semantic_extension_rules(rules.clone())
        .unwrap()
        .build()
        .unwrap();
    assert!(ad
        .semantic_extension_rules()
        .lookup_linearize("tenferro-ad.semantic-identity.v1")
        .is_some());

    let error = AdContext::builder()
        .with_semantic_extension_rules(rules.clone())
        .unwrap()
        .with_semantic_extension_rules(rules)
        .unwrap_err();
    assert!(matches!(
        error,
        SemanticExtensionRegistryError::DuplicateRule { .. }
    ));
}
