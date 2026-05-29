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
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};
use support::RunTraced;

use chainrules_core::ADRuleResult;
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use num_complex::{Complex32, Complex64};
use tenferro_ad::extension::{
    apply_eager, register_extension_rule, ExtensionAdRuleTrait, ExtensionRuleSet,
};
use tenferro_cpu::CpuBackend;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeGuardContext, SymDim};
use tenferro_runtime::extension::{apply, ExtensionExecutionContext, ExtensionRuntime};
use tenferro_runtime::{GraphExecutor, Tensor, TracedTensor};
use tenferro_tensor::{DType, TensorBackend, TypedTensor};

// ----------------------------------------------------------------------
// Test-only AD rule registration guard. Integration tests may share a process,
// so register each family at most once per process.
// ----------------------------------------------------------------------
fn register_rule_once(rule: Arc<dyn ExtensionAdRuleTrait>) {
    static REGISTERED: OnceLock<Mutex<Vec<&'static str>>> = OnceLock::new();
    let guard = REGISTERED.get_or_init(|| Mutex::new(Vec::new()));
    let mut ids = guard.lock().expect("test rule registry mutex");
    let family_id = rule.family_id();
    if ids.contains(&family_id) {
        return;
    }
    if let Err(err) = register_extension_rule(rule) {
        match err {
            tenferro_ad::extension::ExtensionRegistryError::DuplicateRule { .. } => {}
            other => panic!("register_extension_rule failed: {other}"),
        }
    }
    ids.push(family_id);
}

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
        let input = inputs[0];
        // Reuse the same strategy the AD rules use: output = input + input.
        match input {
            Tensor::F64(inner) => {
                let data = inner.host_data();
                let sum: Vec<f64> = data.iter().map(|&v| v + v).collect();
                Ok(vec![Tensor::F64(TypedTensor::from_vec_col_major(
                    input.shape().to_vec(),
                    sum,
                ))])
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

fn ensure_scale_by_2_registered() {
    register_rule_once(Arc::new(TestScaleBy2Rule));
}

#[derive(Debug)]
struct TestScaleBy2Rule;

impl ExtensionAdRuleTrait for TestScaleBy2Rule {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.scale_by_2.v1"
    }

    fn linearize(
        &self,
        _op: &dyn ExtensionOp,
        builder: &mut FragmentBuilder<StdTensorOp>,
        _primal_in: &[GlobalValKey<StdTensorOp>],
        _primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        match tangent_in[0] {
            Some(dx) => {
                let sum = builder.add_op(
                    StdTensorOp::Add,
                    vec![ValRef::Local(dx), ValRef::Local(dx)],
                    OpMode::Linear {
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
        emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<StdTensorOp>],
        _mode: &OpMode,
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        match cotangent_out[0] {
            Some(ct) => {
                let sum = emitter.add_op(
                    StdTensorOp::Add,
                    vec![ValRef::Local(ct), ValRef::Local(ct)],
                    OpMode::Linear {
                        active_mask: vec![true, true],
                    },
                );
                Ok(vec![Some(sum[0])])
            }
            None => Ok(vec![None]),
        }
    }
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

fn ensure_swap_registered() {
    register_rule_once(Arc::new(TestSwapRule));
}

#[derive(Debug)]
struct TestSwapRule;

impl ExtensionAdRuleTrait for TestSwapRule {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.swap.v1"
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
        Ok(vec![tangent_in[1], tangent_in[0]])
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
        Ok(vec![cotangent_out[1], cotangent_out[0]])
    }
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

    fn n_inputs(&self) -> usize {
        2
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

    fn n_inputs(&self) -> usize {
        2
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

fn ensure_probe_registered() {
    register_rule_once(Arc::new(TestProbeIdentityRule));
    register_rule_once(Arc::new(TestProbeLinearRule));
}

#[derive(Debug)]
struct TestProbeIdentityRule;

impl ExtensionAdRuleTrait for TestProbeIdentityRule {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.probe_identity.v1"
    }

    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut FragmentBuilder<StdTensorOp>,
        _primal_in: &[GlobalValKey<StdTensorOp>],
        _primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
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

        let out = builder.add_op(
            StdTensorOp::Extension(Arc::new(TestProbeLinear {
                probe_shape: op.probe_shape.clone(),
            })),
            vec![ValRef::Local(dx), ValRef::Local(dprobe)],
            OpMode::Linear {
                active_mask: vec![true, true],
            },
        );
        Ok(vec![Some(out[0])])
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
        Ok(vec![cotangent_out[0], None])
    }
}

#[derive(Debug)]
struct TestProbeLinearRule;

impl ExtensionAdRuleTrait for TestProbeLinearRule {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.probe_linear.v1"
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
        op: &dyn ExtensionOp,
        emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<StdTensorOp>],
        _mode: &OpMode,
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        let Some(ct) = cotangent_out[0] else {
            return Ok(vec![None, None]);
        };
        let op = op
            .as_any()
            .downcast_ref::<TestProbeLinear>()
            .expect("TestProbeLinearRule received a different op");
        let _probe_value = emitter.add_op(
            StdTensorOp::Reshape {
                to_shape: DimExpr::from_concrete(&op.probe_shape),
            },
            vec![inputs[1].clone()],
            OpMode::Primal,
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

    fn n_inputs(&self) -> usize {
        1
    }

    fn n_outputs(&self) -> usize {
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

fn hash_shape(shape: &[usize], hasher: &mut dyn Hasher) {
    hasher.write_usize(shape.len());
    for &dim in shape {
        hasher.write_usize(dim);
    }
}

fn f64_slice(tensor: &Tensor) -> &[f64] {
    match tensor {
        Tensor::F64(inner) => inner.host_data(),
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

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

#[test]
fn scale_by_2_forward_roundtrip() {
    ensure_scale_by_2_registered();

    let x = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let outputs = apply(Arc::new(TestScaleBy2), &[&x]);
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
    ensure_scale_by_2_registered();

    // loss = sum(scale_by_2(x))    =>   dloss/dx = [2, 2, 2, 2]
    let x = TracedTensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let scaled = apply(Arc::new(TestScaleBy2), &[&x])
        .into_iter()
        .next()
        .unwrap();
    let loss = scaled.reduce_sum(&[0]);

    let g = loss.grad(&x).expect("grad build");
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let grad_out = g.run_with(&mut engine).unwrap();

    assert_eq!(grad_out.shape(), &[4]);
    assert_eq!(f64_slice(&grad_out), &[2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn scale_by_2_eager_backward_uses_registered_rule() {
    ensure_scale_by_2_registered();

    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]),
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
        x.grad().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 2.0, 2.0, 2.0]
    );
}

#[test]
fn ad_context_uses_owned_extension_rules_without_global_fallback() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let scaled = apply(Arc::new(TestScaleBy2), &[&x])
        .into_iter()
        .next()
        .unwrap();
    let loss = scaled.reduce_sum(&[0]);

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
        .with_core_rules()
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

    let dx = TracedTensor::from_vec_col_major(vec![2], vec![3.0_f64, 5.0]);
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

    let dy = TracedTensor::from_vec_col_major(vec![2], vec![7.0_f64, 11.0]);
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
        .with_core_rules()
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
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
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
        .with_core_rules()
        .with_extension_rules(rules)
        .build()
        .unwrap();
    let ctx = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad);
    let x =
        EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx);
    let scaled = apply_eager(Arc::new(TestScaleBy2), &[&x])
        .expect("eager extension apply")
        .into_iter()
        .next()
        .expect("single extension output");
    let loss = scaled.reduce_sum(&[0]).expect("loss");

    let _ = loss.backward().expect("eager backward");

    assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[2.0, 2.0]);
}

fn assert_probe_identity_eager_backward(probe: Tensor) {
    ensure_probe_registered();

    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0]),
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

    assert_eq!(x.grad().unwrap().as_slice::<f64>().unwrap(), &[1.0, 1.0]);
}

#[test]
fn eager_backward_materializes_missing_tangent_zeros_for_all_probe_dtypes() {
    assert_probe_identity_eager_backward(Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]));
    assert_probe_identity_eager_backward(Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]));
    assert_probe_identity_eager_backward(Tensor::from_vec_col_major(vec![2], vec![1_i64, 2]));
    assert_probe_identity_eager_backward(Tensor::from_vec_col_major(vec![2], vec![true, false]));
    assert_probe_identity_eager_backward(Tensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(1.0, -1.0), Complex32::new(2.0, -2.0)],
    ));
    assert_probe_identity_eager_backward(Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, -1.0), Complex64::new(2.0, -2.0)],
    ));
}

#[test]
fn missing_extension_rule_errors_in_traced_grad() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let y = apply(Arc::new(TestNoAd), &[&x])
        .into_iter()
        .next()
        .expect("single output");
    let loss = y.reduce_sum(&[0]);

    let err = match loss.grad(&x) {
        Ok(_) => panic!("missing extension AD rule unexpectedly succeeded"),
        Err(err) => err,
    };

    assert!(err.to_string().contains("tenferro-tests.no_ad.v1"));
}

#[test]
fn missing_extension_rule_errors_in_eager_backward() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x =
        EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]), ctx);
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
