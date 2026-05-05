//! Coverage tests for `ext_op` registry + validation.

use std::any::Any;
use std::hash::Hasher;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use chainrules_core::{ADRuleKind, ADRuleResult};
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;

use crate::ad::context::ShapeGuardContext;
use crate::ext_op::{
    is_extension_registered, is_extension_rule_registered, linearize_extension_rule,
    lookup_extension_factory, lookup_extension_rule, register_extension, register_extension_rule,
    transpose_extension_rule, ExtensionAdRule, ExtensionFactory, ExtensionOp,
    ExtensionRegistryError,
};
use crate::std_tensor_op::StdTensorOp;
use crate::{ExtensionFamilyId, SymDim};
use tenferro_tensor::{DType, Tensor};

#[derive(Debug)]
struct CoverageFamily {
    family: &'static str,
}

impl ExtensionFactory for CoverageFamily {
    fn family_id(&self) -> &'static str {
        self.family
    }
    fn version(&self) -> u32 {
        1
    }
    // `instantiate_default` intentionally left as the default (`None`) to
    // exercise the default-impl body.
}

#[derive(Debug)]
struct CoverageRule {
    family: &'static str,
}

#[derive(Clone, Debug)]
struct NoInlineRuleOp;

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

impl ExtensionAdRule for CoverageRule {
    fn family_id(&self) -> &'static str {
        self.family
    }

    fn linearize(
        &self,
        _op: &dyn ExtensionOp,
        _builder: &mut FragmentBuilder<StdTensorOp>,
        _primal_in: &[GlobalValKey<StdTensorOp>],
        _primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        Ok(vec![tangent_in[0]])
    }

    fn transpose_rule(
        &self,
        _op: &dyn ExtensionOp,
        _emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<StdTensorOp>],
        _mode: &OpMode,
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        Ok(vec![cotangent_out[0]])
    }
}

#[test]
fn default_instantiate_returns_none() {
    let factory: Arc<dyn ExtensionFactory> = Arc::new(CoverageFamily {
        family: "covtest.instantiate.v1",
    });
    assert!(factory.instantiate_default().is_none());
}

#[test]
fn extension_family_id_macro_generates_stable_const() {
    assert_eq!(MacroRuleFamily::FAMILY_ID, "covtest.macro_rule.v1");
}

#[test]
fn register_and_lookup_rule_roundtrips() {
    let family = "covtest.register_rule.v1";
    let rule: Arc<dyn ExtensionAdRule> = Arc::new(CoverageRule { family });
    register_extension_rule(rule).expect("first rule registration should succeed");

    assert!(is_extension_rule_registered(family));
    let looked_up = lookup_extension_rule(family).expect("rule should be registered");
    assert_eq!(looked_up.family_id(), family);
}

#[test]
fn register_rule_rejects_duplicate_family_id() {
    let family = "covtest.duplicate_rule.v1";
    let first: Arc<dyn ExtensionAdRule> = Arc::new(CoverageRule { family });
    register_extension_rule(first).expect("first rule registration should succeed");

    let second: Arc<dyn ExtensionAdRule> = Arc::new(CoverageRule { family });
    match register_extension_rule(second) {
        Err(ExtensionRegistryError::DuplicateRule { family_id }) => {
            assert_eq!(family_id, family);
        }
        other => panic!("expected DuplicateRule for {family:?}, got {other:?}"),
    }
}

#[test]
fn register_rule_rejects_malformed_family_id() {
    let bad = "covtest.bad_rule";
    let rule: Arc<dyn ExtensionAdRule> = Arc::new(CoverageRule { family: bad });

    match register_extension_rule(rule) {
        Err(ExtensionRegistryError::MalformedFamilyId { family_id }) => {
            assert_eq!(family_id, bad);
        }
        other => panic!("expected MalformedFamilyId for {bad:?}, got {other:?}"),
    }
}

#[test]
fn default_inline_rules_panic_with_registration_guidance() {
    let op = NoInlineRuleOp;

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let linearize_panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = op.linearize(&mut builder, &[], &[], &[], &mut ctx);
    }));
    assert!(linearize_panic.is_err());

    let mut emitter = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let transpose_panic = catch_unwind(AssertUnwindSafe(|| {
        let _ = op.transpose_rule(&mut emitter, &[], &[], &OpMode::Primal, &mut ctx);
    }));
    assert!(transpose_panic.is_err());
}

#[test]
fn missing_registered_rule_helpers_return_ad_rule_errors() {
    let op = NoInlineRuleOp;

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let linearize_err =
        linearize_extension_rule(&op, &mut builder, &[], &[], &[], &mut ctx).unwrap_err();
    assert_eq!(linearize_err.rule(), ADRuleKind::Linearize);
    assert!(linearize_err.to_string().contains(op.family_id()));

    let mut emitter = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let transpose_err =
        transpose_extension_rule(&op, &mut emitter, &[], &[], &OpMode::Primal, &mut ctx)
            .unwrap_err();
    assert_eq!(transpose_err.rule(), ADRuleKind::Transpose);
    assert!(transpose_err.to_string().contains(op.family_id()));
}

#[test]
fn register_rejects_malformed_family_ids() {
    // Each case targets a different reject branch in `is_valid_family_id`.
    let cases = [
        "noversion",     // rsplitn(2, '.').next() returns None on second call
        "foo.v1",        // prefix has no '.', split_once returns None
        "foo.bar",       // version segment "bar" fails starts_with('v')
        "foo.bar.v",     // empty digit string after 'v'
        "foo.bar.vabc",  // non-digit version
        ".op.v1",        // empty crate name
        "foo..v1",       // empty op name
        "foo bar.op.v1", // whitespace in crate
        "fooあ.op.v1",   // non-ASCII in crate
    ];
    for bad in cases {
        let factory: Arc<dyn ExtensionFactory> = Arc::new(CoverageFamily { family: bad });
        match register_extension(factory) {
            Err(ExtensionRegistryError::MalformedFamilyId { family_id }) => {
                assert_eq!(family_id, bad);
            }
            other => panic!("expected MalformedFamilyId for {bad:?}, got {other:?}"),
        }
    }
}

#[test]
fn register_and_lookup_roundtrips() {
    let family = "covtest.register_lookup.v1";
    let factory: Arc<dyn ExtensionFactory> = Arc::new(CoverageFamily { family });
    register_extension(factory).expect("first registration should succeed");

    assert!(is_extension_registered(family));
    let looked_up = lookup_extension_factory(family).expect("factory should be registered");
    assert_eq!(looked_up.family_id(), family);
    assert_eq!(looked_up.version(), 1);
}

#[test]
fn register_rejects_duplicate_family_id() {
    let family = "covtest.duplicate.v1";
    let first: Arc<dyn ExtensionFactory> = Arc::new(CoverageFamily { family });
    register_extension(first).expect("first registration should succeed");

    let second: Arc<dyn ExtensionFactory> = Arc::new(CoverageFamily { family });
    match register_extension(second) {
        Err(ExtensionRegistryError::Duplicate { family_id }) => {
            assert_eq!(family_id, family);
        }
        other => panic!("expected Duplicate for {family:?}, got {other:?}"),
    }
}

#[test]
fn lookup_unregistered_family_returns_none() {
    assert!(!is_extension_registered("covtest.absent.v999"));
    assert!(lookup_extension_factory("covtest.absent.v999").is_none());
}
