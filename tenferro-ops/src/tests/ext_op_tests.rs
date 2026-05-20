//! Coverage tests for `ext_op` registry + validation.

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use chainrules_core::{ADRuleKind, ADRuleResult};
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;

use crate::ad::context::ShapeGuardContext;
use crate::ext_op::{
    is_extension_rule_registered, linearize_extension_rule, lookup_extension_rule,
    register_extension_chain_rule, register_extension_rule, transpose_extension_rule,
    ExtensionAdRule, ExtensionChainRule, ExtensionOp, ExtensionRegistryError, FruleBuilder,
    RRuleBuilder,
};
use crate::input_key::TensorInputKey;
use crate::std_tensor_op::StdTensorOp;
use crate::{ExtensionFamilyId, SymDim, TensorMeta};
use tenferro_tensor::{DType, Tensor};

#[derive(Debug)]
struct CoverageRule {
    family: &'static str,
}

#[derive(Clone, Debug)]
struct NoInlineRuleOp;

#[derive(Clone, Debug)]
struct ChainScaleOp;

#[derive(Clone, Debug)]
struct BuilderCoverageOp;

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

impl ExtensionOp for ChainScaleOp {
    fn family_id(&self) -> &'static str {
        "covtest.chain_scale.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<ChainScaleOp>().is_some()
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

impl ExtensionOp for BuilderCoverageOp {
    fn family_id(&self) -> &'static str {
        "covtest.builder_coverage.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<BuilderCoverageOp>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn n_outputs(&self) -> usize {
        2
    }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        vec![
            (input_dtypes[0], input_shapes[0].to_vec()),
            (input_dtypes[1], input_shapes[1].to_vec()),
        ]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        Ok(vec![inputs[0].clone(), inputs[1].clone()])
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

#[derive(Debug)]
struct ChainScaleRule;

impl ExtensionChainRule for ChainScaleRule {
    fn family_id(&self) -> &'static str {
        "covtest.chain_scale.v1"
    }

    fn frule(&self, _op: &dyn ExtensionOp, cx: &mut FruleBuilder<'_>) -> ADRuleResult<()> {
        let Some(dx) = cx.tangent(0)? else {
            cx.set_output_tangent(0, None)?;
            return Ok(());
        };
        let doubled = cx.add(dx.clone(), dx)?;
        cx.set_output_tangent(0, Some(doubled))
    }

    fn rrule(&self, _op: &dyn ExtensionOp, cx: &mut RRuleBuilder<'_>) -> ADRuleResult<()> {
        let Some(cot_y) = cx.cotangent(0)? else {
            cx.set_input_cotangent(0, None)?;
            return Ok(());
        };
        let doubled = cx.add(cot_y.clone(), cot_y)?;
        cx.set_input_cotangent(0, Some(doubled))
    }
}

#[derive(Debug)]
struct BuilderCoverageRule;

impl ExtensionChainRule for BuilderCoverageRule {
    fn family_id(&self) -> &'static str {
        "covtest.builder_coverage.v1"
    }

    fn frule(&self, _op: &dyn ExtensionOp, cx: &mut FruleBuilder<'_>) -> ADRuleResult<()> {
        let x0 = cx.primal(0)?;
        let _y0 = cx.output(0)?;
        assert_eq!(cx.input_rank(0)?, 2);

        let primal_sum = cx.add(x0.clone(), x0)?;
        let primal_product = cx.mul(primal_sum.clone(), primal_sum)?;

        let Some(dx0) = cx.tangent(0)? else {
            cx.set_output_tangent(0, None)?;
            cx.set_output_tangent(1, None)?;
            return Ok(());
        };
        let squared = cx.mul(dx0.clone(), dx0)?;
        let reduced = cx.reduce_sum_all(squared, 2)?;
        let broadcast = cx.broadcast_scalar_to_input(reduced, 0)?;
        let mut ext_outputs = cx.apply_extension(Arc::new(NoInlineRuleOp), vec![broadcast])?;
        let ext_out = ext_outputs.remove(0);
        let dy1 = match cx.tangent(1)? {
            Some(dx1) => cx.add(ext_out.clone(), dx1)?,
            None => cx.add(ext_out.clone(), primal_product)?,
        };

        cx.set_output_tangent(0, Some(ext_out))?;
        cx.set_output_tangent(1, Some(dy1))
    }

    fn rrule(&self, _op: &dyn ExtensionOp, cx: &mut RRuleBuilder<'_>) -> ADRuleResult<()> {
        let x1 = cx.primal(1)?;
        assert_eq!(cx.input_rank(0)?, 2);

        let primal_product = cx.mul(x1.clone(), x1)?;

        let Some(cot_y0) = cx.cotangent(0)? else {
            cx.set_input_cotangent(0, None)?;
            cx.set_input_cotangent(1, None)?;
            return Ok(());
        };
        let squared = cx.mul(cot_y0.clone(), cot_y0)?;
        let reduced = cx.reduce_sum_all(squared, 2)?;
        let broadcast = cx.broadcast_scalar_to_input(reduced, 0)?;
        let mut ext_outputs = cx.apply_extension(Arc::new(NoInlineRuleOp), vec![broadcast])?;
        let ext_out = ext_outputs.remove(0);
        let dx1 = match cx.cotangent(1)? {
            Some(cot_y1) => cx.add(ext_out.clone(), cot_y1)?,
            None => cx.add(ext_out.clone(), primal_product)?,
        };

        cx.set_input_cotangent(0, Some(ext_out))?;
        cx.set_input_cotangent(1, Some(dx1))
    }
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
fn register_chain_rule_adapts_frule_and_rrule() {
    register_extension_chain_rule(Arc::new(ChainScaleRule))
        .expect("first chain rule registration should succeed");

    let op = ChainScaleOp;
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let dx = builder.add_input(TensorInputKey::User { id: 10_001 });
    let mut ctx = ShapeGuardContext::default();
    let jvp = linearize_extension_rule(&op, &mut builder, &[], &[], &[Some(dx)], &mut ctx)
        .expect("chain frule should adapt to linearize");
    assert_ne!(jvp, vec![Some(dx)]);
    let fragment = builder.build();
    assert_eq!(fragment.ops().len(), 1);
    assert_eq!(fragment.ops()[0].op, StdTensorOp::Add);
    assert_eq!(jvp, vec![Some(fragment.ops()[0].outputs[0])]);

    let mut emitter = FragmentBuilder::<StdTensorOp>::new();
    let cot_y = emitter.add_input(TensorInputKey::User { id: 10_002 });
    let mut ctx = ShapeGuardContext::default();
    let vjp = transpose_extension_rule(
        &op,
        &mut emitter,
        &[Some(cot_y)],
        &[],
        &OpMode::Linear {
            active_mask: vec![true],
        },
        &mut ctx,
    )
    .expect("chain rrule should adapt to transpose_rule");
    assert_ne!(vjp, vec![Some(cot_y)]);
    let fragment = emitter.build();
    assert_eq!(fragment.ops().len(), 1);
    assert_eq!(fragment.ops()[0].op, StdTensorOp::Add);
    assert_eq!(vjp, vec![Some(fragment.ops()[0].outputs[0])]);
}

#[test]
fn chain_rule_builders_cover_core_helpers() {
    register_extension_chain_rule(Arc::new(BuilderCoverageRule))
        .expect("first builder coverage rule registration should succeed");

    let x0 = GlobalValKey::Input(TensorInputKey::User { id: 20_001 });
    let x1 = GlobalValKey::Input(TensorInputKey::User { id: 20_002 });
    let y0 = GlobalValKey::Input(TensorInputKey::User { id: 20_003 });
    let y1 = GlobalValKey::Input(TensorInputKey::User { id: 20_004 });
    let mut ctx = ShapeGuardContext::default();
    ctx.insert_metadata(
        x0.clone(),
        TensorMeta::exact(DType::F64, vec![SymDim::from(2usize), SymDim::from(3usize)]),
    );
    ctx.insert_metadata(
        x1.clone(),
        TensorMeta::exact(DType::F64, vec![SymDim::from(2usize), SymDim::from(3usize)]),
    );

    let op = BuilderCoverageOp;
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let dx0 = builder.add_input(TensorInputKey::User { id: 20_005 });
    let dx1 = builder.add_input(TensorInputKey::User { id: 20_006 });
    let jvp = linearize_extension_rule(
        &op,
        &mut builder,
        &[x0.clone(), x1.clone()],
        &[y0, y1],
        &[Some(dx0), Some(dx1)],
        &mut ctx,
    )
    .expect("builder frule should emit helper ops");
    assert_eq!(jvp.len(), 2);
    assert!(jvp.iter().all(Option::is_some));
    let fragment = builder.build();
    assert!(fragment
        .ops()
        .iter()
        .any(|node| node.op == StdTensorOp::Mul));
    assert!(fragment.ops().iter().any(|node| {
        matches!(
            node.op,
            StdTensorOp::BroadcastInDim {
                dims: ref broadcast_dims,
                ..
            } if broadcast_dims.is_empty()
        )
    }));
    assert!(fragment
        .ops()
        .iter()
        .any(|node| matches!(node.op, StdTensorOp::Extension(_))));
    assert!(fragment
        .ops()
        .iter()
        .any(|node| matches!(node.mode, OpMode::Primal)));

    let mut emitter = FragmentBuilder::<StdTensorOp>::new();
    let cot_y0 = emitter.add_input(TensorInputKey::User { id: 20_007 });
    let cot_y1 = emitter.add_input(TensorInputKey::User { id: 20_008 });
    let mut ctx = ShapeGuardContext::default();
    ctx.insert_metadata(
        x0.clone(),
        TensorMeta::exact(DType::F64, vec![SymDim::from(2usize), SymDim::from(3usize)]),
    );
    ctx.insert_metadata(
        x1.clone(),
        TensorMeta::exact(DType::F64, vec![SymDim::from(2usize), SymDim::from(3usize)]),
    );
    let vjp = transpose_extension_rule(
        &op,
        &mut emitter,
        &[Some(cot_y0), Some(cot_y1)],
        &[ValRef::External(x0), ValRef::External(x1)],
        &OpMode::Linear {
            active_mask: vec![true, true],
        },
        &mut ctx,
    )
    .expect("builder rrule should emit helper ops");
    assert_eq!(vjp.len(), 2);
    assert!(vjp.iter().all(Option::is_some));
    let fragment = emitter.build();
    assert!(fragment
        .ops()
        .iter()
        .any(|node| node.op == StdTensorOp::Mul));
    assert!(fragment
        .ops()
        .iter()
        .any(|node| matches!(node.op, StdTensorOp::Extension(_))));
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
fn register_rule_rejects_malformed_family_ids() {
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
        let rule: Arc<dyn ExtensionAdRule> = Arc::new(CoverageRule { family: bad });
        match register_extension_rule(rule) {
            Err(ExtensionRegistryError::MalformedFamilyId { family_id }) => {
                assert_eq!(family_id, bad);
            }
            other => panic!("expected MalformedFamilyId for {bad:?}, got {other:?}"),
        }
    }
}
