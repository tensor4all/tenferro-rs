use super::*;
use std::any::Any;
use std::hash::Hasher;
use tenferro_ops::{ExtensionShapeContext, SymDim};
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_tensor::DType;

const FAMILY: &str = "tenferro-ad.checked-request-test.v1";

#[derive(Clone, Debug)]
struct TestOp;

impl ExtensionOp for TestOp {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn payload_hash(&self, state: &mut dyn Hasher) {
        state.write_u8(0);
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
        1
    }
    fn output_count(&self) -> usize {
        1
    }
    fn semantic_effects(&self) -> tenferro_ops::ext_op::ExtensionEffectDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionEffectDeclaration::Declared(&[])
    }
    fn semantic_aliases(&self) -> tenferro_ops::ext_op::ExtensionAliasDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionAliasDeclaration::AllFresh
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
struct TestRule(&'static str);

impl SemanticLinearizeRule for TestRule {
    fn family_id(&self) -> &'static str {
        self.0
    }
    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        _builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        Ok(SemanticLinearizeResult::new(
            std::iter::repeat_n(AdValue::Absent, request.primal_outputs().len()),
            [],
        ))
    }
}

impl SemanticLinearTransposeRule for TestRule {
    fn family_id(&self) -> &'static str {
        self.0
    }
    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::none()
    }
    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        _builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        Ok(vec![AdValue::Absent; request.primal_input_count()].into_boxed_slice())
    }
}

impl SemanticPrimalVjpRule for TestRule {
    fn family_id(&self) -> &'static str {
        self.0
    }
    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::none()
    }
    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        _builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        Ok(vec![AdValue::Absent; request.primal_input_count()].into_boxed_slice())
    }
}

#[test]
fn checked_request_accessors_cover_declared_metadata_and_errors() {
    let mut builder = SemanticProgramBuilder::new();
    let input = builder
        .input(ProgramInputSpec::new(DType::F64, [1usize.into()]))
        .unwrap();
    let output = builder.add_extension(Arc::new(TestOp), &[input]).unwrap()[0];
    let frozen = builder.finish(&[output]).unwrap();
    let operation = frozen.program.operations().next().unwrap();
    let op = extension_for_dispatch(operation).unwrap();
    let inputs = [input];
    let outputs = [output];
    let metadata = [ProgramValueMetadata::new(DType::F64, [1usize.into()])];
    let cotangents = [AdValue::Absent];
    let active = [true];
    let residuals = [output];
    let mask = ResidualSpec::all_inputs().with_all_outputs();

    let transpose = SemanticLinearTransposeRequest {
        op,
        primal_inputs: &inputs,
        primal_outputs: &outputs,
        primal_input_metadata: metadata.clone().into(),
        primal_output_metadata: metadata.clone().into(),
        cotangent_outputs: &cotangents,
        active_inputs: &active,
        residuals: &residuals,
        residual_mask: mask,
        provenance: operation.provenance(),
    };
    assert_eq!(transpose.op().family_id(), FAMILY);
    assert_eq!(transpose.primal_input_value(0).unwrap(), input);
    assert_eq!(transpose.primal_output_value(0).unwrap(), output);
    assert_eq!(transpose.primal_input_meta(0).unwrap().dtype(), DType::F64);
    assert_eq!(transpose.primal_output_meta(0).unwrap().dtype(), DType::F64);
    assert_eq!(transpose.primal_input_count(), 1);
    assert_eq!(transpose.primal_output_count(), 1);
    assert_eq!(transpose.cotangent_outputs(), &cotangents);
    assert_eq!(transpose.active_inputs(), &active);
    assert_eq!(transpose.residuals(), &residuals);
    assert_eq!(transpose.residual_mask(), mask);
    assert_eq!(transpose.provenance().label(), Some(FAMILY));
    assert!(matches!(
        transpose.primal_input_value(2),
        Err(SemanticAdError::PrimalIndexOutOfBounds {
            kind: PrimalValueKind::Input,
            ..
        })
    ));
    assert!(matches!(
        checked_primal_value(
            FAMILY,
            PrimalValueKind::Output,
            0,
            &outputs,
            ResidualSpec::none()
        ),
        Err(SemanticAdError::UndeclaredResidualValue { .. })
    ));
    assert!(transpose.primal_output_meta(2).is_err());

    let direct = SemanticPrimalVjpRequest {
        op,
        primal_inputs: &inputs,
        primal_outputs: &outputs,
        primal_input_metadata: metadata.clone().into(),
        primal_output_metadata: metadata.into(),
        cotangent_outputs: &cotangents,
        active_inputs: &active,
        residual_mask: mask,
        provenance: operation.provenance(),
    };
    assert_eq!(direct.op().family_id(), FAMILY);
    assert_eq!(direct.primal_input_value(0).unwrap(), input);
    assert_eq!(direct.primal_output_value(0).unwrap(), output);
    assert_eq!(direct.primal_input_meta(0).unwrap().dtype(), DType::F64);
    assert_eq!(direct.primal_output_meta(0).unwrap().dtype(), DType::F64);
    assert_eq!(direct.primal_input_count(), 1);
    assert_eq!(direct.primal_output_count(), 1);
    assert_eq!(direct.cotangent_outputs(), &cotangents);
    assert_eq!(direct.active_inputs(), &active);
    assert_eq!(direct.residual_mask(), mask);
    assert_eq!(direct.provenance().label(), Some(FAMILY));
    assert!(direct.primal_input_meta(2).is_err());
    assert_eq!(AdValue::Absent.value(), None);
    assert_eq!(AdValue::Value(input).value(), Some(input));
}

#[test]
fn rule_set_registration_lookup_debug_and_validation_are_covered() {
    let mut rules = SemanticExtensionRuleSet::new();
    rules
        .register_linearize(Arc::new(TestRule(FAMILY)))
        .unwrap();
    rules
        .register_linear_transpose(Arc::new(TestRule(FAMILY)))
        .unwrap();
    rules
        .register_primal_vjp(Arc::new(TestRule(FAMILY)))
        .unwrap();
    assert!(rules.lookup_linearize(FAMILY).is_some());
    assert!(rules.lookup_linear_transpose(FAMILY).is_some());
    assert!(rules.lookup_primal_vjp(FAMILY).is_some());
    assert!(format!("{rules:?}").contains(FAMILY));
    assert!(matches!(
        rules.register_linearize(Arc::new(TestRule(FAMILY))),
        Err(SemanticExtensionRegistryError::DuplicateRule {
            role: SemanticAdRuleRole::Linearize,
            ..
        })
    ));
    assert!(matches!(
        SemanticExtensionRuleSet::new().with_linearize(Arc::new(TestRule("invalid"))),
        Err(SemanticExtensionRegistryError::MalformedFamilyId { .. })
    ));
    assert!(SemanticExtensionRuleSet::new()
        .with_linear_transpose(Arc::new(TestRule(FAMILY)))
        .unwrap()
        .lookup_linear_transpose(FAMILY)
        .is_some());
    assert!(SemanticExtensionRuleSet::new()
        .with_primal_vjp(Arc::new(TestRule(FAMILY)))
        .unwrap()
        .lookup_primal_vjp(FAMILY)
        .is_some());
    for invalid in [
        "crate.op",
        ".op.v1",
        "crate..v1",
        "crate.op.v",
        "crate op.v1",
    ] {
        assert!(!is_valid_family_id(invalid));
    }
}
