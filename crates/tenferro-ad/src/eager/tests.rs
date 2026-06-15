use std::any::Any;
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::Arc;
use std::time::Duration;

use computegraph::graph::{Graph, GraphBuilder};
use computegraph::types::ValueKey;
use computegraph::{OperationRole, ValueRef};
use tenferro_cpu::CpuBackend;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::SymDim;
use tenferro_runtime::{ExtensionExecutionContext, ExtensionExecutor, ExtensionRuntime};
use tenferro_tensor::Tensor;
use tenferro_tensor::{DType, DotGeneralConfig, TensorBackend};
use tenferro_tensor::{TensorFusion, TensorRead};
use tidu::ADKey;

use crate::eager_backend::EagerBackend;
use crate::eager_exec::exec_op_on_tensor_reads_with_extension_executor;

use super::backward::{
    eager_forward_input_metadata, eager_forward_value, missing_tangent_base_key,
};
use super::{
    eager_op_profile_enabled, maybe_print_eager_op_profile, print_and_reset_eager_op_profile,
    profile_eager_op_section, record_eager_op_profile, zero_like_tensor, EagerRuntime, EagerTensor,
};

fn build_add_mul_reduce_graph(keys: &[TensorInputKey]) -> Arc<Graph<StdTensorOp>> {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let lhs = builder.add_input(keys[0].clone());
    let rhs = builder.add_input(keys[1].clone());
    let sum = builder.add_operation(
        StdTensorOp::Add,
        vec![ValueRef::Local(lhs), ValueRef::Local(rhs)],
        OperationRole::Primary,
    )[0];
    let product = builder.add_operation(
        StdTensorOp::Mul,
        vec![ValueRef::Local(sum), ValueRef::Local(rhs)],
        OperationRole::Primary,
    )[0];
    let loss = builder.add_operation(
        StdTensorOp::ReduceSum { axes: vec![0] },
        vec![ValueRef::Local(product)],
        OperationRole::Primary,
    )[0];
    builder.set_outputs(vec![loss]);
    Arc::new(builder.build())
}

struct EagerOpProfileOverrideGuard;

impl EagerOpProfileOverrideGuard {
    fn set(enabled: bool, print_every: Option<usize>) -> Self {
        super::EAGER_OP_PROFILE_ENABLED_OVERRIDE.with(|state| {
            *state.borrow_mut() = Some(enabled);
        });
        super::EAGER_OP_PROFILE_PRINT_EVERY_OVERRIDE.with(|state| {
            *state.borrow_mut() = Some(print_every);
        });
        super::EAGER_OP_PROFILE_STATE.with(|state| {
            state.borrow_mut().clear();
        });
        Self
    }
}

impl Drop for EagerOpProfileOverrideGuard {
    fn drop(&mut self) {
        super::EAGER_OP_PROFILE_ENABLED_OVERRIDE.with(|state| {
            *state.borrow_mut() = None;
        });
        super::EAGER_OP_PROFILE_PRINT_EVERY_OVERRIDE.with(|state| {
            *state.borrow_mut() = None;
        });
        super::EAGER_OP_PROFILE_STATE.with(|state| {
            state.borrow_mut().clear();
        });
    }
}

#[derive(Clone, Debug)]
struct ReadPathFallbackProbe;

impl ExtensionOp for ReadPathFallbackProbe {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.read-path-fallback-probe.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().is::<ReadPathFallbackProbe>()
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

#[derive(Debug)]
struct ReadPathFallbackRuntime;

impl<B: TensorBackend + 'static> ExtensionRuntime<B> for ReadPathFallbackRuntime {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.read-path-fallback-probe.v1"
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

#[test]
fn tensor_read_extension_path_errors_when_runtime_family_is_missing() {
    let op = StdTensorOp::Extension(Arc::new(ReadPathFallbackProbe));
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let reads = [TensorRead::from_tensor(&input)];
    let mut backend = CpuBackend::new();
    let mut extension_executor = ExtensionExecutor::<CpuBackend>::new();

    let err = exec_op_on_tensor_reads_with_extension_executor(
        &op,
        &reads,
        &mut backend,
        Some(&mut extension_executor),
    )
    .expect_err("registered runtime owner with missing family must not eager fallback");

    let message = err.to_string();
    assert!(message.contains("missing runtime"), "{message}");
    assert!(
        message.contains("tenferro-tests.read-path-fallback-probe.v1"),
        "{message}"
    );
}

#[test]
fn eager_extension_dispatch_does_not_initialize_lazy_view_materialization_cache() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    ctx.register_extension(|executor| {
        executor
            .registry_mut()
            .register(Arc::new(ReadPathFallbackRuntime))
    })
    .expect("register read-path fallback runtime");

    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ctx,
    );
    let x_t = x.transpose(&[1, 0]).unwrap();
    assert!(matches!(x_t.tensor_read(), TensorRead::View(_)));
    assert!(!x_t.materialized_cache_is_initialized());

    let outputs = crate::extension::apply_eager(Arc::new(ReadPathFallbackProbe), &[&x_t])
        .expect("eager extension dispatch");

    assert!(!x_t.materialized_cache_is_initialized());
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        outputs[0].data().as_slice::<f64>().unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn eager_op_profile_helpers_cover_enabled_paths() {
    let _guard = EagerOpProfileOverrideGuard::set(true, Some(2));

    assert!(eager_op_profile_enabled());
    assert_eq!(profile_eager_op_section("coverage.profile", || 7), 7);
    record_eager_op_profile("nary_op.total", Duration::from_micros(3));
    record_eager_op_profile("nary_op.total", Duration::from_micros(5));
    maybe_print_eager_op_profile();
    print_and_reset_eager_op_profile();
}

#[test]
fn eager_forward_helpers_synthesize_tangent_values_from_primal_data() {
    let user = TensorInputKey::User { id: 7 };
    let base_key = ValueKey::Input(user.clone());
    let tangent_key = ValueKey::Input(user.tangent_of(11));
    let base = Arc::new(Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 5.0]));
    let initial_data = HashMap::from([(base_key.clone(), Arc::clone(&base))]);

    assert_eq!(missing_tangent_base_key(&base_key), None);
    assert_eq!(
        missing_tangent_base_key(&tangent_key),
        Some(base_key.clone())
    );
    assert_eq!(
        eager_forward_input_metadata(&tangent_key, &initial_data).shape,
        vec![SymDim::from(2usize)]
    );

    let mut all_values = HashMap::new();
    let mut backend = CpuBackend::new();
    let tangent = eager_forward_value(&mut all_values, &tangent_key, &initial_data, &mut backend);
    assert_eq!(tangent.as_slice::<f64>().unwrap(), &[0.0, 0.0]);
}

#[test]
fn standard_graph_op_executes_untracked_outputs_without_trace() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        ctx.clone(),
    );
    let rhs =
        EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]), ctx);

    let outputs =
        EagerTensor::standard_graph_op(&[&lhs, &rhs], |keys| Ok(build_add_mul_reduce_graph(keys)))
            .expect("standard graph op");

    assert_eq!(outputs.len(), 1);
    assert!(!outputs[0].tracks_grad());
    assert_eq!(outputs[0].debug_trace_saved_value_count(), None);
    assert_eq!(outputs[0].data().shape(), &[] as &[usize]);
    assert_eq!(outputs[0].data().as_slice::<f64>().unwrap(), &[36.0]);
}

#[test]
fn standard_graph_op_records_one_tracked_graph_and_backpropagates() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let lhs = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]),
        ctx.clone(),
    );
    let rhs =
        EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]), ctx);

    let mut outputs =
        EagerTensor::standard_graph_op(&[&lhs, &rhs], |keys| Ok(build_add_mul_reduce_graph(keys)))
            .expect("standard graph op");
    let loss = outputs.pop().expect("one output");

    assert!(loss.tracks_grad());
    assert_eq!(loss.data().as_slice::<f64>().unwrap(), &[36.0]);
    assert_eq!(loss.debug_trace_saved_value_count(), Some(5));

    loss.backward().expect("backward through recorded graph");
    assert_eq!(lhs.grad().unwrap().as_slice::<f64>().unwrap(), &[3.0, 4.0]);
    assert_eq!(rhs.grad().unwrap().as_slice::<f64>().unwrap(), &[7.0, 10.0]);
}

#[test]
fn standard_graph_op_rejects_empty_and_cross_context_inputs() {
    let err = match EagerTensor::standard_graph_op(&[], |_| {
        panic!("empty inputs should fail before graph construction")
    }) {
        Ok(_) => panic!("empty graph op must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("requires at least one input tensor"),
        "{err}"
    );

    let lhs_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let rhs_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let lhs =
        EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]), lhs_ctx);
    let rhs =
        EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![1], vec![2.0_f64]), rhs_ctx);

    let err = match EagerTensor::standard_graph_op(&[&lhs, &rhs], |_| {
        panic!("cross-context inputs should fail before graph construction")
    }) {
        Ok(_) => panic!("cross-context graph op must be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, crate::error::Error::ContextMismatch { .. }));
}

#[test]
fn zero_like_tensor_covers_non_f64_dtypes() {
    let mut backend = CpuBackend::new();
    let cases = [
        Tensor::from_vec_col_major(vec![1], vec![1.0_f32]),
        Tensor::from_vec_col_major(vec![1], vec![1_i32]),
        Tensor::from_vec_col_major(vec![1], vec![1_i64]),
        Tensor::from_vec_col_major(vec![2], vec![true, false]),
        Tensor::from_vec_col_major(vec![1], vec![num_complex::Complex32::new(1.0, -1.0)]),
        Tensor::from_vec_col_major(vec![1], vec![num_complex::Complex64::new(1.0, -1.0)]),
    ];

    for input in cases {
        let zero = zero_like_tensor(&input, &mut backend);
        assert_eq!(zero.shape(), input.shape());
    }
}

#[test]
fn eager_backend_delegates_broadcast_multiply_fusion_to_cpu_backend() {
    let mut backend = EagerBackend::cpu(CpuBackend::new());
    let lhs = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]);
    let rhs = Tensor::from_vec_col_major(vec![3], vec![5.0_f64, 7.0, 11.0]);

    let out = backend
        .execute_broadcast_multiply(
            TensorRead::from_tensor(&lhs),
            &[2, 3],
            &[0],
            TensorRead::from_tensor(&rhs),
            &[2, 3],
            &[1],
        )
        .unwrap()
        .expect("eager backend should delegate CPU broadcast multiply fusion");

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(
        out.as_slice::<f64>().unwrap(),
        &[10.0, 15.0, 14.0, 21.0, 22.0, 33.0]
    );
}

#[test]
fn untracked_nary_ops_consume_lazy_views_without_materializing_inputs() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ctx,
    );
    let x_t = x.transpose(&[1, 0]).unwrap();
    assert!(matches!(x_t.tensor_read(), TensorRead::View(_)));
    assert!(!x_t.materialized_cache_is_initialized());

    let doubled = x_t.add(&x_t).unwrap();
    assert!(!x_t.materialized_cache_is_initialized());
    assert_eq!(
        doubled.data().as_slice::<f64>().unwrap(),
        &[2.0, 6.0, 10.0, 4.0, 8.0, 12.0]
    );

    let reduced = x_t.reduce_sum(&[0]).unwrap();
    assert!(!x_t.materialized_cache_is_initialized());
    assert_eq!(reduced.data().as_slice::<f64>().unwrap(), &[9.0, 12.0]);

    let dot = x_t
        .dot_general(
            &x,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();
    assert!(!x_t.materialized_cache_is_initialized());
    assert_eq!(
        dot.data().as_slice::<f64>().unwrap(),
        &[5.0, 11.0, 17.0, 11.0, 25.0, 39.0, 17.0, 39.0, 61.0]
    );
}
