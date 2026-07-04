//! Coverage tests for `ext_op` rule-set validation and extension dispatch.

use std::any::Any;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use computegraph::GraphOperation;
use tidu::{ADRuleKind, ADRuleResult};

use crate::ad::context::ShapeGuardContext;
use crate::ad::PrimitiveRuleBuilder;
use crate::ext_op::{
    linearize_extension_rule, transpose_extension_rule, ExtensionLinearTransposeRule,
    ExtensionLinearizeRule, ExtensionOp, ExtensionPrimalVjpRule, ExtensionRegistryError,
    ExtensionRuleRole,
};
use crate::input_key::TensorInputKey;
use crate::std_tensor_op::StdTensorOp;
use crate::{ExtensionFamilyId, ExtensionRuleSet, SymDim};
use tenferro_tensor::DType;

#[derive(Debug)]
struct CoverageRule {
    family: &'static str,
}

#[derive(Clone, Debug)]
struct NoInlineRuleOp;

#[derive(Clone, Debug)]
struct FamilyOnlyOp {
    family: &'static str,
}

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

impl ExtensionOp for NoInlineRuleOp {
    fn family_id(&self) -> &'static str {
        "covtest.no_inline_rule.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<NoInlineRuleOp>().is_some()
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
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(input_dtypes[0], input_shapes[0].to_vec())])
    }
}

impl ExtensionOp for FamilyOnlyOp {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write(self.family.as_bytes());
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<FamilyOnlyOp>()
            .is_some_and(|op| op.family == self.family)
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
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(input_dtypes[0], input_shapes[0].to_vec())])
    }
}

impl ExtensionOp for PayloadOp {
    fn family_id(&self) -> &'static str {
        "covtest.payload.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.axis);
        match self.mode {
            WindowMode::Valid => hasher.write_u8(0),
            WindowMode::Same => hasher.write_u8(1),
        }
        hasher.write_usize(self.tensor_inputs);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<PayloadOp>()
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
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(input_dtypes[0], input_shapes[0].to_vec())])
    }
}

impl ExtensionLinearizeRule for CoverageRule {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn linearize(
        &self,
        _op: &dyn ExtensionOp,
        _builder: &mut dyn PrimitiveRuleBuilder,
        _primal_in: &[ValueKey<StdTensorOp>],
        _primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        Ok(vec![tangent_in[0]])
    }
}

#[derive(Debug)]
struct CoverageLinearizeRule {
    family: &'static str,
}

#[derive(Debug)]
struct CoverageLinearTransposeRule {
    family: &'static str,
}

#[derive(Debug)]
struct CoveragePrimalVjpRule {
    family: &'static str,
}

impl ExtensionLinearizeRule for CoverageLinearizeRule {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn linearize(
        &self,
        _op: &dyn ExtensionOp,
        _builder: &mut dyn PrimitiveRuleBuilder,
        _primal_in: &[ValueKey<StdTensorOp>],
        _primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        Ok(vec![tangent_in[0]])
    }
}

impl ExtensionLinearTransposeRule for CoverageLinearTransposeRule {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn linear_transpose(
        &self,
        _op: &dyn ExtensionOp,
        _builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        _inputs: &[ValueRef<StdTensorOp>],
        active_mask: &[bool],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        assert_eq!(active_mask, &[true]);
        Ok(vec![cotangent_out[0]])
    }
}

impl ExtensionPrimalVjpRule for CoveragePrimalVjpRule {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn primal_vjp(
        &self,
        _op: &dyn ExtensionOp,
        _builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        _inputs: &[ValueRef<StdTensorOp>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        Ok(vec![cotangent_out[0]])
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
    let lhs = payload_op(1, WindowMode::Valid, 1);
    let rhs = payload_op(1, WindowMode::Valid, 1);

    assert_eq!(lhs, rhs);
}

#[test]
fn extension_payload_not_equal_when_axis_differs() {
    let lhs = payload_op(1, WindowMode::Valid, 1);
    let rhs = payload_op(2, WindowMode::Valid, 1);

    assert_ne!(lhs, rhs);
}

#[test]
fn extension_payload_not_equal_when_mode_differs() {
    let lhs = payload_op(1, WindowMode::Valid, 1);
    let rhs = payload_op(1, WindowMode::Same, 1);

    assert_ne!(lhs, rhs);
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
    let lhs = payload_op(1, WindowMode::Valid, 2);
    let rhs = payload_op(2, WindowMode::Same, 2);

    assert_eq!(lhs.input_count(), 2);
    assert_eq!(rhs.input_count(), 2);
}

#[test]
fn register_and_lookup_rule_roundtrips() {
    let family = "covtest.register_rule.v1";
    let rule: Arc<dyn ExtensionLinearizeRule> = Arc::new(CoverageRule { family });
    let mut rules = ExtensionRuleSet::new();
    rules
        .register_linearize(rule)
        .expect("first rule registration should succeed");

    assert!(rules.is_linearize_registered(family));
    let looked_up = rules
        .lookup_linearize(family)
        .expect("rule should be registered");
    assert_eq!(looked_up.family_id(), family);
}

#[test]
fn role_split_rule_set_accepts_one_rule_per_role_for_same_family() {
    let family = "covtest.role_split.v1";
    let mut rules = ExtensionRuleSet::new();

    rules
        .register_linearize(Arc::new(CoverageLinearizeRule { family }))
        .expect("linearize registration");
    rules
        .register_linear_transpose(Arc::new(CoverageLinearTransposeRule { family }))
        .expect("linear transpose registration");
    rules
        .register_primal_vjp(Arc::new(CoveragePrimalVjpRule { family }))
        .expect("primal VJP registration");

    assert!(rules.lookup_linearize(family).is_some());
    assert!(rules.lookup_linear_transpose(family).is_some());
    assert!(rules.lookup_primal_vjp(family).is_some());
}

#[test]
fn role_split_rule_set_rejects_duplicate_family_per_role() {
    let family = "covtest.duplicate_linearize.v1";
    let mut rules = ExtensionRuleSet::new()
        .with_linearize(Arc::new(CoverageLinearizeRule { family }))
        .expect("first linearize registration");

    match rules.register_linearize(Arc::new(CoverageLinearizeRule { family })) {
        Err(ExtensionRegistryError::DuplicateRule {
            family_id,
            role: ExtensionRuleRole::Linearize,
        }) => assert_eq!(family_id, family),
        other => panic!("expected duplicate linearize rule, got {other:?}"),
    }
}

#[test]
fn transpose_dispatch_uses_linear_transpose_for_linearized_role() {
    let family = "covtest.linear-transpose-dispatch.v1";
    let op = FamilyOnlyOp { family };
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let ct = builder.add_input(TensorInputKey::User { id: 10_301 });
    let rules = ExtensionRuleSet::new()
        .with_linear_transpose(Arc::new(CoverageLinearTransposeRule { family }))
        .expect("linear transpose registration");
    let mut ctx = ShapeGuardContext::default().with_extension_rules(rules);

    let result = transpose_extension_rule(
        &op,
        &mut builder,
        &[Some(ct)],
        &[],
        &OperationRole::Linearized {
            active_mask: vec![true],
        },
        &mut ctx,
    )
    .expect("linearized transpose dispatch");

    assert_eq!(result, vec![Some(ct)]);
}

#[test]
fn transpose_dispatch_uses_primal_vjp_for_primary_role() {
    let family = "covtest.primal-vjp-dispatch.v1";
    let op = FamilyOnlyOp { family };
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let ct = builder.add_input(TensorInputKey::User { id: 10_302 });
    let rules = ExtensionRuleSet::new()
        .with_primal_vjp(Arc::new(CoveragePrimalVjpRule { family }))
        .expect("primal VJP registration");
    let mut ctx = ShapeGuardContext::default().with_extension_rules(rules);

    let result = transpose_extension_rule(
        &op,
        &mut builder,
        &[Some(ct)],
        &[],
        &OperationRole::Primary,
        &mut ctx,
    )
    .expect("primary transpose dispatch");

    assert_eq!(result, vec![Some(ct)]);
}

#[test]
fn register_rule_rejects_duplicate_family_id() {
    let family = "covtest.duplicate_rule.v1";
    let first: Arc<dyn ExtensionLinearizeRule> = Arc::new(CoverageRule { family });
    let mut rules = ExtensionRuleSet::new();
    rules
        .register_linearize(first)
        .expect("first rule registration should succeed");

    let second: Arc<dyn ExtensionLinearizeRule> = Arc::new(CoverageRule { family });
    match rules.register_linearize(second) {
        Err(ExtensionRegistryError::DuplicateRule {
            family_id,
            role: ExtensionRuleRole::Linearize,
        }) => {
            assert_eq!(family_id, family);
        }
        other => panic!("expected DuplicateRule for {family:?}, got {other:?}"),
    }
}

#[test]
fn register_rule_rejects_malformed_family_id() {
    let bad = "covtest.bad_rule";
    let rule: Arc<dyn ExtensionLinearizeRule> = Arc::new(CoverageRule { family: bad });

    match ExtensionRuleSet::new().with_linearize(rule) {
        Err(ExtensionRegistryError::MalformedFamilyId { family_id }) => {
            assert_eq!(family_id, bad);
        }
        other => panic!("expected MalformedFamilyId for {bad:?}, got {other:?}"),
    }
}

#[test]
fn empty_rule_set_does_not_use_process_global_fallback() {
    let family = "covtest.global_only.v1";
    let op = FamilyOnlyOp { family };
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let dx = builder.add_input(TensorInputKey::User { id: 10_201 });
    let mut ctx = ShapeGuardContext::default().with_extension_rules(ExtensionRuleSet::new());
    let err = linearize_extension_rule(&op, &mut builder, &[], &[], &[Some(dx)], &mut ctx)
        .expect_err("explicit empty rule set must not consult global registry");
    assert_eq!(err.rule(), ADRuleKind::Jvp);
    assert!(err.to_string().contains(family));
}

#[test]
fn owned_rule_set_merge_is_atomic_on_duplicate_family() {
    let family_a = "covtest.merge_a.v1";
    let family_b = "covtest.merge_b.v1";
    let mut base = ExtensionRuleSet::new()
        .with_linearize(Arc::new(CoverageRule { family: family_a }))
        .expect("base rule should register");
    let other = ExtensionRuleSet::new()
        .with_linearize(Arc::new(CoverageRule { family: family_b }))
        .expect("other rule should register")
        .with_linearize(Arc::new(CoverageRule { family: family_a }))
        .expect("duplicate is only relative to base");

    let err = base
        .merge(other)
        .expect_err("merge should reject duplicate family");
    assert!(matches!(
        err,
        ExtensionRegistryError::DuplicateRule {
            family_id: "covtest.merge_a.v1",
            role: ExtensionRuleRole::Linearize
        }
    ));
    assert!(base.is_linearize_registered(family_a));
    assert!(!base.is_linearize_registered(family_b));
}

#[test]
fn owned_rule_set_rejects_duplicate_and_malformed_rules() {
    let family = "covtest.owned_duplicate.v1";
    let mut rules = ExtensionRuleSet::new()
        .with_linearize(Arc::new(CoverageRule { family }))
        .expect("first owned rule should register");
    let duplicate_err = rules
        .register_linearize(Arc::new(CoverageRule { family }))
        .expect_err("duplicate owned rule should be rejected");
    assert!(matches!(
        duplicate_err,
        ExtensionRegistryError::DuplicateRule {
            family_id: "covtest.owned_duplicate.v1",
            role: ExtensionRuleRole::Linearize
        }
    ));
    assert!(rules.is_linearize_registered(family));

    let malformed = "covtest.owned_malformed";
    let malformed_err = ExtensionRuleSet::new()
        .with_linearize(Arc::new(CoverageRule { family: malformed }))
        .expect_err("malformed owned rule should be rejected");
    assert!(matches!(
        malformed_err,
        ExtensionRegistryError::MalformedFamilyId {
            family_id: "covtest.owned_malformed"
        }
    ));
}

#[test]
fn missing_registered_rule_helpers_return_ad_rule_errors() {
    let op = NoInlineRuleOp;

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let linearize_err =
        linearize_extension_rule(&op, &mut builder, &[], &[], &[], &mut ctx).unwrap_err();
    assert_eq!(linearize_err.rule(), ADRuleKind::Jvp);
    assert!(linearize_err.to_string().contains(op.family_id()));

    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let transpose_err = transpose_extension_rule(
        &op,
        &mut builder,
        &[],
        &[],
        &OperationRole::Primary,
        &mut ctx,
    )
    .unwrap_err();
    assert_eq!(transpose_err.rule(), ADRuleKind::Transpose);
    assert!(transpose_err.to_string().contains(op.family_id()));
}

#[test]
fn register_rule_rejects_malformed_family_ids() {
    // Each case targets a different reject branch in `is_valid_family_id`.
    let cases = [
        "noversion",            // rsplitn(2, '.').next() returns None on second call
        "foo.v1",               // prefix has no '.', split_once returns None
        "foo.bar",              // version segment "bar" fails starts_with('v')
        "foo.bar.v",            // empty digit string after 'v'
        "foo.bar.vabc",         // non-digit version
        ".operation.v1",        // empty crate name
        "foo..v1",              // empty op name
        "foo bar.operation.v1", // whitespace in crate
        "fooあ.operation.v1",   // non-ASCII in crate
    ];
    for bad in cases {
        let rule: Arc<dyn ExtensionLinearizeRule> = Arc::new(CoverageRule { family: bad });
        match ExtensionRuleSet::new().with_linearize(rule) {
            Err(ExtensionRegistryError::MalformedFamilyId { family_id }) => {
                assert_eq!(family_id, bad);
            }
            other => panic!("expected MalformedFamilyId for {bad:?}, got {other:?}"),
        }
    }
}
