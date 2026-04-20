//! Smoke tests for the Stage 6 `ExtensionOp` mechanism.
//!
//! Exercises two end-to-end paths using only the public `tenferro` and
//! `tenferro-ops` facades:
//!
//! - `TestScaleBy2`: single-input, single-output. Forward computes
//!   `input * 2.0` via the eager backend; `linearize` and
//!   `transpose_rule` emit core `StdTensorOp::Add` ops so the extension
//!   participates in AD through the closure invariant.
//! - `TestSwap`: two inputs, two outputs (input-swap). Exercises
//!   multi-input / multi-output plumbing.
//!
//! The tests verify forward results and backward (`grad`) results against
//! hand-computed expected values.

use std::any::Any;
use std::hash::Hasher;
use std::sync::{Arc, Mutex, OnceLock};

use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro::extension::{apply, register_extension, ExtensionFactory};
use tenferro::{CpuBackend, Engine, Tensor, TracedTensor};
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeGuardContext, SymDim};
use tenferro_tensor::{DType, TypedTensor};

// ----------------------------------------------------------------------
// Test-only registration guard. Integration tests may share a process,
// so register each family at most once per process.
// ----------------------------------------------------------------------
fn register_once(factory: Arc<dyn ExtensionFactory>) {
    static REGISTERED: OnceLock<Mutex<Vec<&'static str>>> = OnceLock::new();
    let guard = REGISTERED.get_or_init(|| Mutex::new(Vec::new()));
    let mut ids = guard.lock().expect("test registry mutex");
    let family_id = factory.family_id();
    if ids.contains(&family_id) {
        return;
    }
    if let Err(err) = register_extension(factory) {
        match err {
            tenferro::extension::ExtensionRegistryError::Duplicate { .. } => {
                // Already registered by another parallel test binary — fine.
            }
            other => panic!("register_extension failed: {other}"),
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
                Ok(vec![Tensor::F64(TypedTensor::from_vec(
                    input.shape().to_vec(),
                    sum,
                ))])
            }
            other => Err(tenferro_tensor::Error::BackendFailure {
                op: "extension",
                message: format!(
                    "TestScaleBy2 only supports F64 in tests; got dtype {:?}",
                    other.dtype()
                ),
            }),
        }
    }

    fn linearize(
        &self,
        builder: &mut FragmentBuilder<StdTensorOp>,
        _primal_in: &[GlobalValKey<StdTensorOp>],
        _primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut ShapeGuardContext,
    ) -> Vec<Option<LocalValId>> {
        // y = x + x  =>  y_dot = x_dot + x_dot  (linear in x_dot).
        match tangent_in[0] {
            Some(dx) => {
                let sum = builder.add_op(
                    StdTensorOp::Add,
                    vec![ValRef::Local(dx), ValRef::Local(dx)],
                    OpMode::Linear {
                        active_mask: vec![true, true],
                    },
                );
                vec![Some(sum[0])]
            }
            None => vec![None],
        }
    }

    fn transpose_rule(
        &self,
        emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<StdTensorOp>],
        _mode: &OpMode,
        _ctx: &mut ShapeGuardContext,
    ) -> Vec<Option<LocalValId>> {
        // cot_x = cot_y + cot_y (dual of y_dot = x_dot + x_dot)
        match cotangent_out[0] {
            Some(ct) => {
                let sum = emitter.add_op(
                    StdTensorOp::Add,
                    vec![ValRef::Local(ct), ValRef::Local(ct)],
                    OpMode::Linear {
                        active_mask: vec![true, true],
                    },
                );
                vec![Some(sum[0])]
            }
            None => vec![None],
        }
    }
}

struct TestScaleBy2Factory;

impl ExtensionFactory for TestScaleBy2Factory {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.scale_by_2.v1"
    }

    fn version(&self) -> u32 {
        1
    }
}

fn ensure_scale_by_2_registered() {
    register_once(Arc::new(TestScaleBy2Factory));
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

    fn linearize(
        &self,
        _builder: &mut FragmentBuilder<StdTensorOp>,
        _primal_in: &[GlobalValKey<StdTensorOp>],
        _primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut ShapeGuardContext,
    ) -> Vec<Option<LocalValId>> {
        // Tangent rule is the same swap: [ta, tb] -> [tb, ta].
        vec![tangent_in[1], tangent_in[0]]
    }

    fn transpose_rule(
        &self,
        _emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<StdTensorOp>],
        _mode: &OpMode,
        _ctx: &mut ShapeGuardContext,
    ) -> Vec<Option<LocalValId>> {
        // Transpose of swap is swap.
        vec![cotangent_out[1], cotangent_out[0]]
    }
}

struct TestSwapFactory;

impl ExtensionFactory for TestSwapFactory {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.swap.v1"
    }

    fn version(&self) -> u32 {
        1
    }
}

fn ensure_swap_registered() {
    register_once(Arc::new(TestSwapFactory));
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

fn f64_slice(tensor: &Tensor) -> &[f64] {
    match tensor {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected F64 tensor"),
    }
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

#[test]
fn scale_by_2_forward_roundtrip() {
    ensure_scale_by_2_registered();

    let x = TracedTensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let outputs = apply(Arc::new(TestScaleBy2), &[&x]);
    assert_eq!(outputs.len(), 1);
    let mut y = outputs.into_iter().next().unwrap();

    let mut engine = Engine::new(CpuBackend::new());
    let result = y.eval(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3]);
    assert_eq!(f64_slice(result), &[2.0, 4.0, 6.0]);
}

#[test]
fn scale_by_2_grad_against_reduce_sum() {
    ensure_scale_by_2_registered();

    // loss = sum(scale_by_2(x))    =>   dloss/dx = [2, 2, 2, 2]
    let x = TracedTensor::from_vec(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let scaled = apply(Arc::new(TestScaleBy2), &[&x])
        .into_iter()
        .next()
        .unwrap();
    let loss = scaled.reduce_sum(&[0]);

    let mut g = loss.grad(&x).expect("grad build");
    let mut engine = Engine::new(CpuBackend::new());
    let grad_out = g.eval(&mut engine).unwrap();

    assert_eq!(grad_out.shape(), &[4]);
    assert_eq!(f64_slice(grad_out), &[2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn scale_by_2_grad_through_symbolic_placeholder() {
    ensure_scale_by_2_registered();

    // Same loss but with an input_symbolic_shape placeholder so the AD
    // path exercises the deferred zero-tangent policy (spec
    // Section 10).
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let scaled = apply(Arc::new(TestScaleBy2), &[&x])
        .into_iter()
        .next()
        .unwrap();
    let loss = scaled.reduce_sum(&[0]);

    let mut g = loss.grad(&x).expect("grad build");
    let bound = Tensor::from_vec(vec![5], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]);

    let mut engine = Engine::new(CpuBackend::new());
    let grad_out = g
        .eval_with_inputs(&mut engine, &[(&x, &bound)])
        .expect("grad eval");

    assert_eq!(grad_out.shape(), &[5]);
    assert_eq!(f64_slice(grad_out), &[2.0, 2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn swap_forward_roundtrip() {
    ensure_swap_registered();

    let a = TracedTensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
    let b = TracedTensor::from_vec(vec![2], vec![100.0_f64, 200.0]);
    let outputs = apply(Arc::new(TestSwap), &[&a, &b]);
    assert_eq!(outputs.len(), 2);
    let mut iter = outputs.into_iter();
    let mut out_first = iter.next().unwrap();
    let mut out_second = iter.next().unwrap();

    let mut engine = Engine::new(CpuBackend::new());
    let t0 = out_first.eval(&mut engine).unwrap().clone();
    let t1 = out_second.eval(&mut engine).unwrap().clone();

    // (a, b) -> (b, a)
    assert_eq!(f64_slice(&t0), &[100.0, 200.0]);
    assert_eq!(f64_slice(&t1), &[1.0, 2.0]);
}

#[test]
fn swap_grad_routes_cotangents_across_inputs() {
    ensure_swap_registered();

    // loss = sum(out0 + out1)   where (out0, out1) = swap(a, b)
    //       = sum(b + a)
    // => dloss/da = ones_like(a),  dloss/db = ones_like(b)
    let a = TracedTensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let b = TracedTensor::from_vec(vec![3], vec![10.0_f64, 20.0, 30.0]);
    let swapped = apply(Arc::new(TestSwap), &[&a, &b]);
    let mut iter = swapped.into_iter();
    let out0 = iter.next().unwrap();
    let out1 = iter.next().unwrap();
    let combined = &out0 + &out1;
    let loss = combined.reduce_sum(&[0]);

    let mut grad_a = loss.grad(&a).expect("grad a");
    let mut grad_b = loss.grad(&b).expect("grad b");

    let mut engine = Engine::new(CpuBackend::new());
    let ga = grad_a.eval(&mut engine).unwrap().clone();
    let gb = grad_b.eval(&mut engine).unwrap().clone();

    assert_eq!(f64_slice(&ga), &[1.0, 1.0, 1.0]);
    assert_eq!(f64_slice(&gb), &[1.0, 1.0, 1.0]);
}

#[test]
fn swap_grad_routes_only_through_active_output() {
    ensure_swap_registered();

    // loss = sum(out1)  where (out0, out1) = swap(a, b)  (so out1 = a)
    // => dloss/da = ones_like(a)
    let a = TracedTensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let b = TracedTensor::from_vec(vec![3], vec![10.0_f64, 20.0, 30.0]);
    let swapped = apply(Arc::new(TestSwap), &[&a, &b]);
    let mut iter = swapped.into_iter();
    let _out0 = iter.next().unwrap();
    let out1 = iter.next().unwrap();
    let loss = out1.reduce_sum(&[0]);

    let mut grad_a = loss.grad(&a).expect("grad a");
    let mut engine = Engine::new(CpuBackend::new());
    let ga = grad_a.eval(&mut engine).unwrap().clone();
    assert_eq!(f64_slice(&ga), &[1.0, 1.0, 1.0]);

    // grad wrt b: sum(out1) does not depend on b; try_grad returns None.
    let maybe = loss.try_grad(&b).expect("try_grad b");
    assert!(
        maybe.is_none(),
        "expected sum(swap(a,b)[1]) to have no gradient wrt b"
    );
}

#[test]
fn extension_carrier_hash_and_eq_are_stable() {
    // Structural identity: two extension carriers with the same payload
    // hash to equal values and compare equal; different payloads (here
    // different families) compare unequal.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let a: StdTensorOp = StdTensorOp::Extension(Arc::new(TestScaleBy2) as Arc<dyn ExtensionOp>);
    let b: StdTensorOp = StdTensorOp::Extension(Arc::new(TestScaleBy2) as Arc<dyn ExtensionOp>);
    let c: StdTensorOp = StdTensorOp::Extension(Arc::new(TestSwap) as Arc<dyn ExtensionOp>);

    assert_eq!(a, b);
    assert_ne!(a, c);

    let mut ha = DefaultHasher::new();
    a.hash(&mut ha);
    let mut hb = DefaultHasher::new();
    b.hash(&mut hb);
    assert_eq!(ha.finish(), hb.finish());
}

#[test]
fn duplicate_registration_is_rejected() {
    use tenferro::extension::ExtensionRegistryError;

    ensure_scale_by_2_registered();
    let err = register_extension(Arc::new(TestScaleBy2Factory))
        .expect_err("second registration must error");
    assert!(matches!(
        err,
        ExtensionRegistryError::Duplicate {
            family_id: "tenferro-tests.scale_by_2.v1"
        }
    ));
}

#[test]
fn malformed_family_id_is_rejected() {
    use tenferro::extension::ExtensionRegistryError;

    struct BadFactory;
    impl ExtensionFactory for BadFactory {
        fn family_id(&self) -> &'static str {
            "no-version-suffix"
        }
        fn version(&self) -> u32 {
            1
        }
    }

    let err = register_extension(Arc::new(BadFactory)).expect_err("malformed family_id must error");
    assert!(matches!(
        err,
        ExtensionRegistryError::MalformedFamilyId { .. }
    ));
}
