//! Smoke tests for the Stage 6 `ExtensionOp` mechanism.
//!
//! Exercises two end-to-end paths using only the public `tenferro` and
//! `tenferro-internal-ops` APIs:
//!
//! - `TestScaleBy2`: single-input, single-output. Forward computes
//!   `input * 2.0` via the eager backend; its registered AD rule emits core
//!   `StdTensorOp::Add` ops so the extension participates in AD through the
//!   closure invariant.
//! - `TestSwap`: two inputs, two outputs (input-swap). Exercises
//!   multi-input / multi-output plumbing.
//!
//! The tests verify forward results and backward (`grad`) results against
//! hand-computed expected values.

use tenferro_ad::TracedTensorAdExt;
use tenferro_ad::{AdContext, AdContextBuilder, EagerRuntime, EagerTensor};
#[path = "extension_op/api_and_registry.rs"]
mod api_and_registry;
mod support;
use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;
use support::RunTraced;

use computegraph::types::{LocalValueId, OperationRole, ValueKey, ValueRef};
use num_complex::{Complex32, Complex64};
use tenferro_ad::extension::{apply_eager, ExtensionAdRule, ExtensionRuleSet};
use tenferro_cpu::CpuBackend;
use tenferro_ops::ad::PrimitiveRuleBuilder;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeGuardContext, SymDim};
use tenferro_runtime::extension::{apply, ExtensionExecutionContext, ExtensionRuntime};
use tenferro_runtime::{GraphExecutor, Tensor, TracedTensor};
use tenferro_tensor::{DType, TensorBackend, TensorRead, TypedTensor};
use tidu::ADRuleResult;

// ----------------------------------------------------------------------
// TestScaleBy2: single-input, single-output. y = x + x (= 2x).
// ----------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct TestScaleBy2;

impl ExtensionOp for TestScaleBy2 {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.scale_by_2.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {
        // No payload to hash. The carrier's family_id contribution is
        // enough to distinguish the op.
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        // Only equal to another TestScaleBy2 instance.
        other.as_any().downcast_ref::<TestScaleBy2>().is_some()
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
    ) -> Vec<(DType, Vec<SymDim>)> {
        vec![(input_dtypes[0], input_shapes[0].to_vec())]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        let input = inputs[0];
        // Reuse the same strategy the AD rules use: output = input + input.
        match input {
            Tensor::F64(inner) => {
                let data = inner.host_data().unwrap();
                let sum: Vec<f64> = data.iter().map(|&v| v + v).collect();
                Ok(vec![Tensor::F64(
                    TypedTensor::from_vec_col_major(input.shape().to_vec(), sum).unwrap(),
                )])
            }
            other => Err(tenferro_tensor::Error::backend_failure(
                "extension",
                format!(
                    "TestScaleBy2 only supports F64 in tests; got dtype {:?}",
                    other.dtype()
                ),
            )),
        }
    }
}

#[derive(Debug)]
struct TestScaleBy2Rule;

impl ExtensionAdRule for TestScaleBy2Rule {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.scale_by_2.v1"
    }

    fn linearize(
        &self,
        _op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        _primal_in: &[ValueKey<StdTensorOp>],
        _primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        match tangent_in[0] {
            Some(dx) => {
                let sum = builder.add_operation(
                    StdTensorOp::Add,
                    vec![ValueRef::Local(dx), ValueRef::Local(dx)],
                    OperationRole::Linearized {
                        active_mask: vec![true, true],
                    },
                );
                Ok(vec![Some(sum[0])])
            }
            None => Ok(vec![None]),
        }
    }

    fn transpose_rule(
        &self,
        _op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        _inputs: &[ValueRef<StdTensorOp>],
        _mode: &OperationRole,
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        match cotangent_out[0] {
            Some(ct) => {
                let sum = builder.add_operation(
                    StdTensorOp::Add,
                    vec![ValueRef::Local(ct), ValueRef::Local(ct)],
                    OperationRole::Linearized {
                        active_mask: vec![true, true],
                    },
                );
                Ok(vec![Some(sum[0])])
            }
            None => Ok(vec![None]),
        }
    }
}

fn scale_by_2_rules() -> ExtensionRuleSet {
    ExtensionRuleSet::new()
        .with_rule(Arc::new(TestScaleBy2Rule))
        .expect("scale_by_2 rule registration")
}

fn scale_by_2_ad_context() -> AdContext {
    AdContext::builder()
        .with_extension_rules(scale_by_2_rules())
        .build()
        .expect("scale_by_2 AD context")
}

// ----------------------------------------------------------------------
// TestSwap: two inputs, two outputs. (a, b) -> (b, a).
// ----------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct TestSwap;

impl ExtensionOp for TestSwap {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.swap.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<TestSwap>().is_some()
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
        2
    }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        // (a, b) -> (b, a). We report the swapped meta so downstream
        // consumers see the correct shape for each output slot.
        assert_eq!(
            input_dtypes[0], input_dtypes[1],
            "TestSwap expects matching dtypes"
        );
        vec![
            (input_dtypes[1], input_shapes[1].to_vec()),
            (input_dtypes[0], input_shapes[0].to_vec()),
        ]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        Ok(vec![inputs[1].clone(), inputs[0].clone()])
    }
}

#[derive(Debug)]
struct TestSwapRule;

impl ExtensionAdRule for TestSwapRule {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.swap.v1"
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
        Ok(vec![tangent_in[1], tangent_in[0]])
    }

    fn transpose_rule(
        &self,
        _op: &dyn ExtensionOp,
        _builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        _inputs: &[ValueRef<StdTensorOp>],
        _mode: &OperationRole,
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        Ok(vec![cotangent_out[1], cotangent_out[0]])
    }
}

fn swap_rules() -> ExtensionRuleSet {
    ExtensionRuleSet::new()
        .with_rule(Arc::new(TestSwapRule))
        .expect("swap rule registration")
}

fn swap_ad_context() -> AdContext {
    AdContext::builder()
        .with_extension_rules(swap_rules())
        .build()
        .expect("swap AD context")
}

// ----------------------------------------------------------------------
// TestNoAd: forward-only extension. Missing AD must be reported as Error.
// ----------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct TestNoAd;

impl ExtensionOp for TestNoAd {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.no_ad.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<TestNoAd>().is_some()
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
    ) -> Vec<(DType, Vec<SymDim>)> {
        vec![(input_dtypes[0], input_shapes[0].to_vec())]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        Ok(vec![inputs[0].clone()])
    }
}

// ----------------------------------------------------------------------
// TestProbeIdentity: identity op whose linearization carries a second
// non-differentiated input so eager transpose materializes missing tangents.
// ----------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct TestProbeIdentity {
    probe_shape: Vec<usize>,
}

impl ExtensionOp for TestProbeIdentity {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.probe_identity.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hash_shape(&self.probe_shape, hasher);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<TestProbeIdentity>() == Some(self)
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

#[derive(Clone, Debug, PartialEq)]
struct TestProbeLinear {
    probe_shape: Vec<usize>,
}

impl ExtensionOp for TestProbeLinear {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.probe_linear.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hash_shape(&self.probe_shape, hasher);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<TestProbeLinear>() == Some(self)
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

#[derive(Debug)]
struct TestProbeIdentityRule;

impl ExtensionAdRule for TestProbeIdentityRule {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.probe_identity.v1"
    }

    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        _primal_in: &[ValueKey<StdTensorOp>],
        _primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let op = op
            .as_any()
            .downcast_ref::<TestProbeIdentity>()
            .expect("TestProbeIdentityRule received a different op");
        let Some(dx) = tangent_in[0] else {
            return Ok(vec![None]);
        };
        let Some(dprobe) = tangent_in[1] else {
            return Ok(vec![Some(dx)]);
        };

        let out = builder.add_operation(
            StdTensorOp::Extension(Arc::new(TestProbeLinear {
                probe_shape: op.probe_shape.clone(),
            })),
            vec![ValueRef::Local(dx), ValueRef::Local(dprobe)],
            OperationRole::Linearized {
                active_mask: vec![true, true],
            },
        );
        Ok(vec![Some(out[0])])
    }

    fn transpose_rule(
        &self,
        _op: &dyn ExtensionOp,
        _builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        _inputs: &[ValueRef<StdTensorOp>],
        _mode: &OperationRole,
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        Ok(vec![cotangent_out[0], None])
    }
}

#[derive(Debug)]
struct TestProbeLinearRule;

impl ExtensionAdRule for TestProbeLinearRule {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.probe_linear.v1"
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

    fn transpose_rule(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[ValueRef<StdTensorOp>],
        _mode: &OperationRole,
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let Some(ct) = cotangent_out[0] else {
            return Ok(vec![None, None]);
        };
        let op = op
            .as_any()
            .downcast_ref::<TestProbeLinear>()
            .expect("TestProbeLinearRule received a different op");
        let _probe_value = builder.add_operation(
            StdTensorOp::Reshape {
                to_shape: DimExpr::from_concrete(&op.probe_shape),
            },
            vec![inputs[1].clone()],
            OperationRole::Primary,
        );
        Ok(vec![Some(ct), None])
    }
}

// ----------------------------------------------------------------------
// TestBadOutputCount: malformed extension for facade validation paths.
// ----------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct TestBadOutputCount;

impl ExtensionOp for TestBadOutputCount {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.bad_output_count.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<TestBadOutputCount>()
            .is_some()
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
        2
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

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn probe_rules() -> ExtensionRuleSet {
    ExtensionRuleSet::new()
        .with_rule(Arc::new(TestProbeIdentityRule))
        .expect("probe identity rule registration")
        .with_rule(Arc::new(TestProbeLinearRule))
        .expect("probe linear rule registration")
}

fn probe_ad_context() -> AdContext {
    AdContext::builder()
        .with_extension_rules(probe_rules())
        .build()
        .expect("probe AD context")
}

fn hash_shape(shape: &[usize], hasher: &mut dyn Hasher) {
    hasher.write_usize(shape.len());
    for &dim in shape {
        hasher.write_usize(dim);
    }
}

fn f64_slice(tensor: &Tensor) -> &[f64] {
    match tensor {
        Tensor::F64(inner) => inner.host_data().unwrap(),
        _ => panic!("expected F64 tensor"),
    }
}

#[derive(Debug)]
struct TestRuntime {
    family_id: &'static str,
}

impl<B: TensorBackend + 'static> ExtensionRuntime<B> for TestRuntime {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn execute(
        &self,
        op: &dyn ExtensionOp,
        inputs: &[&Tensor],
        _ctx: &mut ExtensionExecutionContext<'_, B>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        op.eager_execute(inputs)
    }

    fn execute_reads(
        &self,
        op: &dyn ExtensionOp,
        inputs: &[TensorRead<'_>],
        ctx: &mut ExtensionExecutionContext<'_, B>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let materialized_inputs: Vec<Tensor> = inputs
            .iter()
            .map(TensorRead::to_tensor)
            .collect::<tenferro_tensor::Result<_>>()?;
        let input_refs: Vec<&Tensor> = materialized_inputs.iter().collect();
        self.execute(op, &input_refs, ctx)
    }
}

fn register_test_runtime<B: TensorBackend + 'static>(
    executor: &mut GraphExecutor<B>,
    family_id: &'static str,
) {
    executor
        .register_extension(|extension_executor| {
            extension_executor
                .registry_mut()
                .register(Arc::new(TestRuntime { family_id }))
        })
        .expect("register test extension runtime");
}

fn register_test_eager_runtime(runtime: &EagerRuntime, family_id: &'static str) {
    runtime
        .register_extension(|extension_executor| {
            extension_executor
                .registry_mut()
                .register(Arc::new(TestRuntime { family_id }))
        })
        .expect("register test eager extension runtime");
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

#[test]
fn scale_by_2_forward_roundtrip() {
    let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let outputs = apply(Arc::new(TestScaleBy2), &[&x]).unwrap();
    assert_eq!(outputs.len(), 1);
    let y = outputs.into_iter().next().unwrap();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    register_test_runtime(&mut engine, "tenferro-tests.scale_by_2.v1");
    let result = y.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3]);
    assert_eq!(f64_slice(&result), &[2.0, 4.0, 6.0]);
}

#[test]
fn scale_by_2_grad_against_reduce_sum() {
    // loss = sum(scale_by_2(x))    =>   dloss/dx = [2, 2, 2, 2]
    let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let scaled = apply(Arc::new(TestScaleBy2), &[&x])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let loss = scaled.reduce_sum(&[0]).unwrap();

    let g = scale_by_2_ad_context().grad(&loss, &x).expect("grad build");
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let grad_out = g.run_with(&mut engine).unwrap();

    assert_eq!(grad_out.shape(), &[4]);
    assert_eq!(f64_slice(&grad_out), &[2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn scale_by_2_eager_backward_uses_registered_rule() {
    let ad = scale_by_2_ad_context();
    let ctx = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad);
    register_test_eager_runtime(&ctx, "tenferro-tests.scale_by_2.v1");
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        ctx,
    );
    let scaled = apply_eager(Arc::new(TestScaleBy2), &[&x])
        .expect("eager extension apply")
        .into_iter()
        .next()
        .expect("single extension output");
    let loss = scaled.reduce_sum(&[0]).expect("loss");

    let _ = loss.backward().expect("eager backward");

    assert_eq!(
        x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 2.0, 2.0, 2.0]
    );
}

#[test]
fn ad_context_uses_owned_extension_rules_without_global_fallback() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let scaled = apply(Arc::new(TestScaleBy2), &[&x])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let loss = scaled.reduce_sum(&[0]).unwrap();

    let empty_ad = AdContext::builder().build().unwrap();
    let err = match empty_ad.grad(&loss, &x) {
        Ok(_) => panic!("explicit empty rule set should not use global fallback"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("tenferro-tests.scale_by_2.v1"));

    let rules = ExtensionRuleSet::new()
        .with_rule(Arc::new(TestScaleBy2Rule))
        .expect("owned scale_by_2 rule registration");
    let ad = AdContext::builder()
        .with_extension_rules(rules)
        .build()
        .unwrap();
    assert!(ad
        .extension_rules()
        .is_rule_registered("tenferro-tests.scale_by_2.v1"));

    let grad = ad
        .grad_optional(&loss, &x)
        .expect("grad should build")
        .expect("scale_by_2 is active");
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let grad_out = grad.run_with(&mut engine).unwrap();
    assert_eq!(f64_slice(&grad_out), &[2.0, 2.0]);

    let dx = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0]).unwrap();
    let jvp = ad.jvp(&scaled, &x, &dx).expect("jvp should build");
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let jvp_out = jvp.run_with(&mut engine).unwrap();
    assert_eq!(f64_slice(&jvp_out), &[6.0, 10.0]);

    let jvp = ad
        .jvp_optional(&scaled, &x, &dx)
        .expect("jvp should build")
        .expect("scale_by_2 output is active");
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let jvp_out = jvp.run_with(&mut engine).unwrap();
    assert_eq!(f64_slice(&jvp_out), &[6.0, 10.0]);

    let dy = TracedTensor::from_vec_col_major(vec![2], vec![7.0_f64, 11.0]).unwrap();
    let vjp = ad.vjp(&scaled, &x, &dy).expect("vjp should build");
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let vjp_out = vjp.run_with(&mut engine).unwrap();
    assert_eq!(f64_slice(&vjp_out), &[14.0, 22.0]);

    let vjp = ad
        .vjp_optional(&scaled, &x, &dy)
        .expect("vjp should build")
        .expect("scale_by_2 input is active");
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let vjp_out = vjp.run_with(&mut engine).unwrap();
    assert_eq!(f64_slice(&vjp_out), &[14.0, 22.0]);
}

#[test]
fn ad_context_builder_rejects_duplicate_extension_rule_sets() {
    let first = ExtensionRuleSet::new()
        .with_rule(Arc::new(TestScaleBy2Rule))
        .expect("first rule set");
    let second = ExtensionRuleSet::new()
        .with_rule(Arc::new(TestScaleBy2Rule))
        .expect("second rule set");

    let err = AdContextBuilder::new()
        .with_extension_rules(first)
        .with_extension_rules(second)
        .build()
        .expect_err("duplicate extension rule family should fail");
    assert!(matches!(
        err,
        tenferro_ad::extension::ExtensionRegistryError::DuplicateRule {
            family_id: "tenferro-tests.scale_by_2.v1"
        }
    ));
}

#[test]
fn eager_runtime_ad_context_uses_owned_extension_rules_without_global_fallback() {
    let empty_ad = AdContext::builder().build().unwrap();
    let empty_ctx = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &empty_ad);
    register_test_eager_runtime(&empty_ctx, "tenferro-tests.scale_by_2.v1");
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        empty_ctx,
    );
    let scaled = apply_eager(Arc::new(TestScaleBy2), &[&x])
        .expect("eager extension apply")
        .into_iter()
        .next()
        .expect("single extension output");
    let loss = scaled.reduce_sum(&[0]).expect("loss");
    let err = loss
        .backward()
        .expect_err("explicit empty rule set should not use global fallback");
    assert!(err.to_string().contains("tenferro-tests.scale_by_2.v1"));

    let rules = ExtensionRuleSet::new()
        .with_rule(Arc::new(TestScaleBy2Rule))
        .expect("owned scale_by_2 rule registration");
    let ad = AdContext::builder()
        .with_extension_rules(rules)
        .build()
        .unwrap();
    let ctx = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad);
    register_test_eager_runtime(&ctx, "tenferro-tests.scale_by_2.v1");
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx,
    );
    let scaled = apply_eager(Arc::new(TestScaleBy2), &[&x])
        .expect("eager extension apply")
        .into_iter()
        .next()
        .expect("single extension output");
    let loss = scaled.reduce_sum(&[0]).expect("loss");

    let _ = loss.backward().expect("eager backward");

    assert_eq!(
        x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 2.0]
    );
}

fn assert_probe_identity_eager_backward(probe: Tensor) {
    let ad = probe_ad_context();
    let ctx = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad);
    register_test_eager_runtime(&ctx, "tenferro-tests.probe_identity.v1");
    register_test_eager_runtime(&ctx, "tenferro-tests.probe_linear.v1");
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0]).unwrap(),
        ctx.clone(),
    );
    let probe_shape = probe.shape().to_vec();
    let probe = EagerTensor::from_tensor_in(probe, ctx);
    let y = apply_eager(Arc::new(TestProbeIdentity { probe_shape }), &[&x, &probe])
        .expect("probe identity eager apply")
        .into_iter()
        .next()
        .expect("single probe identity output");
    let loss = y.reduce_sum(&[0]).expect("loss");

    let _ = loss.backward().expect("eager backward");

    assert_eq!(
        x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[1.0, 1.0]
    );
}

#[test]
fn eager_backward_materializes_missing_tangent_zeros_for_all_probe_dtypes() {
    assert_probe_identity_eager_backward(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap(),
    );
    assert_probe_identity_eager_backward(
        Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap(),
    );
    assert_probe_identity_eager_backward(
        Tensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap(),
    );
    assert_probe_identity_eager_backward(
        Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap(),
    );
    assert_probe_identity_eager_backward(
        Tensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(1.0, -1.0), Complex32::new(2.0, -2.0)],
        )
        .unwrap(),
    );
    assert_probe_identity_eager_backward(
        Tensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, -1.0), Complex64::new(2.0, -2.0)],
        )
        .unwrap(),
    );
}

#[test]
fn missing_extension_rule_errors_in_traced_grad() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = apply(Arc::new(TestNoAd), &[&x])
        .unwrap()
        .into_iter()
        .next()
        .expect("single output");
    let loss = y.reduce_sum(&[0]).unwrap();

    let err = match loss.grad(&x) {
        Ok(_) => panic!("missing extension AD rule unexpectedly succeeded"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("tenferro-tests.no_ad.v1"));
}

#[test]
fn missing_extension_rule_errors_in_eager_backward() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    register_test_eager_runtime(&ctx, "tenferro-tests.no_ad.v1");
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx,
    );
    let y = apply_eager(Arc::new(TestNoAd), &[&x])
        .expect("forward-only eager extension apply")
        .into_iter()
        .next()
        .expect("single output");
    let loss = y.reduce_sum(&[0]).expect("loss");

    let err = loss
        .backward()
        .expect_err("missing extension AD rule should error");

    assert!(err.to_string().contains("tenferro-tests.no_ad.v1"));
}
