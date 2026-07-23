use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_ad::semantic_extension::{
    AdValue, SemanticAdError, SemanticExtensionRuleSet, SemanticLinearTransposeRequest,
    SemanticLinearTransposeRule, SemanticLinearizeRequest, SemanticLinearizeResult,
    SemanticLinearizeRule, SemanticPrimalVjpRequest, SemanticPrimalVjpRule,
};
use tenferro_ad::semantic_transform::SemanticAdTransformError;
use tenferro_ad::AdContext;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp};
use tenferro_ops::{ExtensionShapeContext, SymDim};
use tenferro_runtime::program::{CoreSemanticOp, ProgramInputSpec, SemanticProgramBuilder};
use tenferro_tensor::{DType, Tensor};

const FAMILY: &str = "tenferro-ad.semantic-transform-test.v1";

#[derive(Clone, Debug)]
struct AddInputsExtension;

impl ExtensionOp for AddInputsExtension {
    fn family_id(&self) -> &'static str {
        FAMILY
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(1);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().is::<Self>()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        2
    }

    fn output_count(&self) -> usize {
        1
    }

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        context: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        context.require_same_shape(0, 1)?;
        Ok(vec![(
            context.input_dtype(0)?,
            context.input_shape(0)?.to_vec(),
        )])
    }
}

#[derive(Debug)]
struct AddInputsRule;

impl SemanticLinearizeRule for AddInputsRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }

    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        assert_eq!(request.active_outputs(), &[true]);
        assert_eq!(request.provenance().label(), Some(FAMILY));
        let tangent = match request.tangent_inputs() {
            [AdValue::Value(lhs), AdValue::Value(rhs)] => {
                AdValue::Value(builder.add_op(CoreSemanticOp::Add, &[*lhs, *rhs])?[0])
            }
            [AdValue::Value(value), AdValue::Absent] | [AdValue::Absent, AdValue::Value(value)] => {
                AdValue::Value(*value)
            }
            [AdValue::Absent, AdValue::Absent] => AdValue::Absent,
            _ => unreachable!(),
        };
        Ok(SemanticLinearizeResult::new([tangent], []))
    }
}

impl SemanticLinearTransposeRule for AddInputsRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }

    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        _builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        assert_eq!(request.provenance().label(), Some(FAMILY));
        let cotangent = request.cotangent_outputs()[0];
        Ok(request
            .active_inputs()
            .iter()
            .map(|active| if *active { cotangent } else { AdValue::Absent })
            .collect())
    }
}

impl SemanticPrimalVjpRule for AddInputsRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }

    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        _builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        assert_eq!(request.provenance().label(), Some(FAMILY));
        let cotangent = request.cotangent_outputs()[0];
        Ok(request
            .active_inputs()
            .iter()
            .map(|active| if *active { cotangent } else { AdValue::Absent })
            .collect())
    }
}

fn repeated_input_program() -> tenferro_runtime::program::FrozenProgram {
    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [DimExpr::Const(2)]))
        .unwrap();
    builder
        .bind_input(
            input,
            Arc::new(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0_f64]).unwrap()),
        )
        .unwrap();
    let output = builder
        .add_extension(Arc::new(AddInputsExtension), &[input, input])
        .unwrap()[0];
    builder.finish(&[output]).unwrap()
}

fn rules() -> SemanticExtensionRuleSet {
    let mut rules = SemanticExtensionRuleSet::new();
    rules.register_linearize(Arc::new(AddInputsRule)).unwrap();
    rules.register_primal_vjp(Arc::new(AddInputsRule)).unwrap();
    rules
}

fn transpose_rules() -> SemanticExtensionRuleSet {
    let mut rules = SemanticExtensionRuleSet::new();
    rules.register_linearize(Arc::new(AddInputsRule)).unwrap();
    rules
        .register_linear_transpose(Arc::new(AddInputsRule))
        .unwrap();
    rules
}

fn ad_context() -> AdContext {
    AdContext::builder()
        .with_semantic_extension_rules(rules())
        .unwrap()
        .build()
        .unwrap()
}

#[test]
fn semantic_jvp_appends_ordered_tangent_inputs_and_returns_only_tangents() {
    let source = repeated_input_program();
    let transformed = ad_context().jvp_program(&source, &[true]).unwrap();

    assert_eq!(transformed.frozen().program.inputs().len(), 2);
    assert_eq!(transformed.frozen().program.outputs().len(), 1);
    assert_eq!(transformed.derivative_input_indices(), &[Some(1)]);
    assert_eq!(transformed.derivative_output_indices(), &[Some(0)]);
    assert_eq!(transformed.frozen().program.operations().count(), 2);
    assert_eq!(transformed.frozen().bindings.len(), 1);
}

#[test]
fn semantic_vjp_accumulates_repeated_input_cotangents() {
    let source = repeated_input_program();
    let transformed = ad_context().vjp_program(&source, &[true], &[true]).unwrap();

    assert_eq!(transformed.frozen().program.inputs().len(), 2);
    assert_eq!(transformed.derivative_input_indices(), &[Some(1)]);
    assert_eq!(transformed.derivative_output_indices(), &[Some(0)]);
    assert_eq!(transformed.frozen().program.operations().count(), 2);
    assert!(matches!(
        transformed
            .frozen()
            .program
            .operations()
            .last()
            .unwrap()
            .op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Add)
    ));
    assert_eq!(transformed.frozen().bindings.len(), 1);
}

#[test]
fn semantic_vjp_falls_back_to_linearize_then_transpose_explicitly() {
    let source = repeated_input_program();
    let ad = AdContext::builder()
        .with_semantic_extension_rules(transpose_rules())
        .unwrap()
        .build()
        .unwrap();
    let transformed = ad.vjp_program(&source, &[true], &[true]).unwrap();

    assert_eq!(transformed.derivative_output_indices(), &[Some(0)]);
    assert!(matches!(
        transformed
            .frozen()
            .program
            .operations()
            .last()
            .unwrap()
            .op(),
        tenferro_runtime::program::SemanticOpRef::Core(CoreSemanticOp::Add)
    ));
}

#[test]
fn semantic_activity_is_ordered_typed_and_preserves_inactive_values() {
    let source = repeated_input_program();
    let ad = ad_context();

    let inactive = ad.jvp_program(&source, &[false]).unwrap();
    assert_eq!(inactive.derivative_input_indices(), &[None]);
    assert_eq!(inactive.derivative_output_indices(), &[None]);
    assert!(inactive.frozen().program.outputs().is_empty());

    assert!(matches!(
        ad.jvp_program(&source, &[]),
        Err(SemanticAdTransformError::ActivityArity {
            field: "active_inputs",
            expected: 1,
            actual: 0,
            ..
        })
    ));
    assert!(matches!(
        ad.vjp_program(&source, &[true], &[]),
        Err(SemanticAdTransformError::ActivityArity {
            field: "active_outputs",
            expected: 1,
            actual: 0,
            ..
        })
    ));
}
