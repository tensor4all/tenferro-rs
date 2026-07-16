use std::any::Any;
use std::collections::HashMap;
use std::hash::Hasher;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use computegraph::graph::{Graph, GraphBuilder};
use computegraph::resolve::resolve;
use computegraph::types::ValueKey;
use computegraph::{OperationRole, ValueRef};
use tenferro_cpu::CpuBackend;
use tenferro_ops::ext_op::{ExtensionOp, HostReference};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::{ShapeExtent, ShapeGuardContext, SymDim, TensorMeta};
use tenferro_runtime::ExtensionCacheLimits;
use tenferro_runtime::{Error, ExtensionExecutionContext, ExtensionExecutor, ExtensionRuntime};
use tenferro_tensor::Tensor;
use tenferro_tensor::{DType, DotGeneralConfig, TensorBackend};
use tenferro_tensor::{TensorFusion, TensorRead};
use tidu::eager::BackwardExecutor;
use tidu::{linearize, ADKey};

use crate::eager_backend::EagerBackend;
use crate::eager_exec::exec_op_on_tensor_reads_with_extension_executor;
use crate::metadata::{register_scoped_metadata_batch, tensor_meta_from_tensor};

use super::backward::{
    eager_forward_input_metadata, eager_forward_value, eager_residual_value,
    missing_tangent_base_key, prefill_linear_residual_values, zero_from_exact_metadata,
    TenferroBackwardCallbacks,
};
use super::{
    eager_op_profile_enabled, eager_op_profile_per_call_us, maybe_print_eager_op_profile,
    one_like_tensor, print_and_reset_eager_op_profile, profile_eager_op_section,
    record_eager_op_profile, zero_like_tensor, EagerOpProfileEntry, EagerRuntime, EagerTensor,
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

#[test]
fn eager_runtime_synchronize_reports_poisoned_backend_lock() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = ctx.backend.lock().unwrap();
        panic!("poison eager backend lock");
    }));
    assert!(poisoned.is_err());

    let err = ctx.synchronize().unwrap_err();

    assert!(matches!(
        err,
        Error::Internal(ref message) if message.contains("backend lock poisoned")
    ));
}

#[test]
fn eager_materialization_uses_backend() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();

    let compact = x.to_tensor().unwrap();
    assert_eq!(compact.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(ctx.recorded_to_contiguous_reads(), 0);

    let view = x.transpose(&[1, 0]).unwrap();
    let compact = view.to_tensor().unwrap();
    assert_eq!(compact.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    assert_eq!(ctx.recorded_to_contiguous_reads(), 1);
}

#[test]
fn eager_runtime_register_extension_reports_poisoned_executor_lock() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = ctx.extension_executor.lock().unwrap();
        panic!("poison extension executor lock");
    }));
    assert!(poisoned.is_err());

    let err = ctx.register_extension(|_| Ok(())).unwrap_err();

    assert!(err.to_string().contains("extension executor lock poisoned"));
}

#[test]
fn eager_runtime_public_helpers_do_not_unwrap_poisoned_locks() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = ctx.extension_executor.lock().unwrap();
        panic!("poison extension executor lock");
    }));
    assert!(poisoned.is_err());

    assert!(ctx.clear_extension_caches().is_err());
    assert!(ctx.cache_stats().is_err());
    assert!(ctx.extension_cache_limits().is_err());
    assert!(ctx
        .set_extension_cache_limits(ExtensionCacheLimits::new(NonZeroUsize::new(1).unwrap(),))
        .is_err());
    assert!(ctx.with_extension_caches_mut(|_| ()).is_err());

    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = ctx.clear_extension_caches();
        let _ = ctx.clear_caches();
        let _ = ctx.cache_stats();
        let _ = ctx.extension_cache_limits();
        let _ = ctx.set_extension_cache_limits(ExtensionCacheLimits::default());
    }))
    .is_ok());
}

#[test]
fn eager_runtime_backend_closure_reports_poisoned_backend_lock() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = ctx.backend.lock().unwrap();
        panic!("poison eager backend lock");
    }));
    assert!(poisoned.is_err());

    let err = ctx.with_backend_mut(|_| ()).unwrap_err();

    assert!(matches!(
        err,
        Error::Internal(ref message) if message.contains("backend lock poisoned")
    ));
}

#[test]
fn eager_index_select_reports_poisoned_backend_lock() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = ctx.backend.lock().unwrap();
        panic!("poison eager backend lock");
    }));
    assert!(poisoned.is_err());

    let err = x.index_select(0, &[0]).unwrap_err();

    assert!(matches!(
        err,
        Error::Internal(ref message) if message.contains("backend lock poisoned")
    ));
}

#[test]
fn eager_runtime_gradient_helpers_do_not_unwrap_poisoned_locks() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let poisoned_slot = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = x.grad_slot.lock().unwrap();
        panic!("poison eager gradient slot");
    }));
    assert!(poisoned_slot.is_err());

    assert!(x.grad().is_err());
    assert!(x.clear_grad().is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = x.grad();
        let _ = x.clear_grad();
    }))
    .is_ok());

    let poisoned_registry = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = ctx.grad_slots.lock().unwrap();
        panic!("poison eager gradient registry");
    }));
    assert!(poisoned_registry.is_err());

    assert!(ctx.clear_grads().is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = ctx.clear_grads();
    }))
    .is_ok());
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
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }

    fn host_reference(&self) -> Option<&dyn HostReference> {
        Some(self)
    }
}

impl HostReference for ReadPathFallbackProbe {
    fn execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
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
        op.host_reference()
            .ok_or(tenferro_tensor::Error::NoHostReference {
                family_id: op.family_id(),
            })?
            .execute(inputs)
    }

    fn execute_reads(
        &self,
        op: &dyn ExtensionOp,
        inputs: &[TensorRead<'_>],
        ctx: &mut ExtensionExecutionContext<'_, B>,
    ) -> tenferro_tensor::Result<Vec<Tensor>> {
        let materialized_inputs = ctx.backend_mut().with_backend_session(|exec| {
            inputs
                .iter()
                .cloned()
                .map(|input| exec.to_contiguous_read(input))
                .collect::<tenferro_tensor::Result<Vec<_>>>()
        })?;
        let input_refs: Vec<&Tensor> = materialized_inputs.iter().collect();
        self.execute(op, &input_refs, ctx)
    }
}

#[test]
fn tensor_read_extension_path_errors_when_runtime_family_is_missing() {
    let op = StdTensorOp::Extension(Arc::new(ReadPathFallbackProbe));
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
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
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx,
    )
    .unwrap();
    let x_t = x.transpose(&[1, 0]).unwrap();
    assert!(matches!(x_t.tensor_read(), TensorRead::View(_)));
    assert!(!x_t.materialized_cache_is_initialized());

    let outputs = crate::extension::apply_eager(Arc::new(ReadPathFallbackProbe), &[&x_t])
        .expect("eager extension dispatch");

    assert!(!x_t.materialized_cache_is_initialized());
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        outputs[0]
            .materialized()
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn eager_op_profile_helpers_cover_enabled_paths() {
    let _guard = EagerOpProfileOverrideGuard::set(true, Some(2));

    assert!(eager_op_profile_enabled());
    assert_eq!(profile_eager_op_section("coverage.profile", || 7), 7);
    assert_eq!(
        eager_op_profile_per_call_us(&EagerOpProfileEntry::default()),
        None
    );
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
    let base = Arc::new(Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 5.0]).unwrap());
    let initial_data = HashMap::from([(base_key.clone(), Arc::clone(&base))]);

    assert_eq!(missing_tangent_base_key(&base_key), None);
    assert_eq!(
        missing_tangent_base_key(&tangent_key),
        Some(base_key.clone())
    );
    assert_eq!(
        eager_forward_input_metadata(&tangent_key, &initial_data)
            .unwrap()
            .exact_shape(),
        Some(vec![SymDim::from(2usize)])
    );

    let mut all_values = HashMap::new();
    let mut backend = CpuBackend::new();
    let tangent =
        eager_forward_value(&mut all_values, &tangent_key, &initial_data, &mut backend).unwrap();
    assert_eq!(tangent.as_slice::<f64>().unwrap(), &[0.0, 0.0]);
}

#[test]
fn eager_forward_helpers_report_missing_values_without_panicking() {
    let user = TensorInputKey::User { id: 8 };
    let base_key = ValueKey::Input(user.clone());
    let tangent_key = ValueKey::Input(user.tangent_of(12));
    let initial_data = HashMap::new();

    let err = eager_forward_input_metadata(&base_key, &initial_data).unwrap_err();
    assert!(
        err.to_string().contains("missing concrete eager value"),
        "{err}"
    );
    let err = eager_forward_input_metadata(&tangent_key, &initial_data).unwrap_err();
    assert!(
        err.to_string().contains("missing base eager value"),
        "{err}"
    );

    let mut all_values = HashMap::new();
    let mut backend = CpuBackend::new();
    let err =
        eager_forward_value(&mut all_values, &base_key, &initial_data, &mut backend).unwrap_err();
    assert!(
        err.to_string().contains("missing concrete eager value"),
        "{err}"
    );
    let err = eager_forward_value(&mut all_values, &tangent_key, &initial_data, &mut backend)
        .unwrap_err();
    assert!(
        err.to_string().contains("missing base eager value"),
        "{err}"
    );
}

#[test]
fn eager_backward_zero_from_exact_metadata_covers_dtypes_and_dynamic_none() {
    let mut backend = CpuBackend::new();
    let dynamic = TensorMeta::with_extents(
        DType::F64,
        vec![ShapeExtent::upper_bound(SymDim::from(2usize))],
    );
    assert!(zero_from_exact_metadata(&dynamic, &mut backend)
        .unwrap()
        .is_none());
    let symbolic_exact = TensorMeta::exact(DType::F64, vec![SymDim::tensor_axis(9, 0)]);
    assert!(zero_from_exact_metadata(&symbolic_exact, &mut backend)
        .unwrap()
        .is_none());
    let overflow = TensorMeta::exact(
        DType::Bool,
        vec![SymDim::from(usize::MAX), SymDim::from(2usize)],
    );
    let err = zero_from_exact_metadata(&overflow, &mut backend).unwrap_err();
    assert!(
        err.to_string()
            .contains("zero tensor shape product overflows"),
        "{err}"
    );

    for dtype in [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C32,
        DType::C64,
    ] {
        let meta = TensorMeta::exact(dtype, vec![SymDim::from(2usize)]);
        let zero = zero_from_exact_metadata(&meta, &mut backend)
            .unwrap()
            .expect("exact metadata should produce zero tensor");
        assert_eq!(zero.dtype(), dtype);
        assert_eq!(zero.shape(), &[2]);
    }
}

#[test]
fn eager_backward_callbacks_record_add_errors_without_panicking() {
    let mut backend = CpuBackend::new();
    let mut callbacks = TenferroBackwardCallbacks::new(&mut backend, None, Vec::new());
    let a = Arc::new(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap());
    let b = Arc::new(Tensor::from_vec_col_major(vec![3], vec![3.0_f64, 4.0, 5.0]).unwrap());

    let first = callbacks.add_operands(&a, &b);
    assert!(Arc::ptr_eq(&first, &a));
    let second = callbacks.add_operands(&a, &b);
    assert!(Arc::ptr_eq(&second, &a));

    let err = callbacks
        .take_error()
        .expect("shape mismatch should be recorded");
    assert!(
        err.to_string().contains("eager cotangent add failed"),
        "{err}"
    );
    assert!(callbacks.take_error().is_none());
}

#[test]
fn eager_backward_transpose_runs_without_extension_executor() {
    let input_key = TensorInputKey::User { id: 123 };
    let mut graph_builder = GraphBuilder::<StdTensorOp>::new();
    let x = graph_builder.add_input(input_key.clone());
    let y = graph_builder.add_operation(
        StdTensorOp::Mul,
        vec![ValueRef::Local(x), ValueRef::Local(x)],
        OperationRole::Primary,
    )[0];
    graph_builder.set_outputs(vec![y]);
    let graph = Arc::new(graph_builder.build());
    let output_key = graph.values()[y].key.clone();
    let x_tensor = Arc::new(Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap());
    let _primal_scope = register_scoped_metadata_batch(vec![(
        ValueKey::Input(input_key.clone()),
        tensor_meta_from_tensor(x_tensor.as_ref()),
    )])
    .unwrap();

    let view = resolve(vec![Arc::clone(&graph)]);
    let mut ad_ctx = ShapeGuardContext::with_global_metadata();
    let linear = linearize(
        &view,
        &[output_key],
        std::slice::from_ref(&input_key),
        0,
        &mut ad_ctx,
        &HashMap::new(),
    )
    .unwrap();
    let tangent_input_key = match &linear.as_graph().values()[linear.tangent_inputs()[0].1].key {
        ValueKey::Input(key) => key.clone(),
        other => panic!("expected tangent input key, got {other:?}"),
    };
    let _linear_scope = register_scoped_metadata_batch(vec![(
        ValueKey::Input(tangent_input_key),
        tensor_meta_from_tensor(x_tensor.as_ref()),
    )])
    .unwrap();
    ad_ctx.refresh_global_metadata();

    let cotangent = Arc::new(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap());
    let mut backend = CpuBackend::new();
    let mut callbacks = TenferroBackwardCallbacks::new(&mut backend, None, Vec::new());
    let external_data = HashMap::from([(ValueKey::Input(input_key), x_tensor)]);
    let gradients = callbacks
        .run_transposed_linear(&linear, &[Some(cotangent)], &external_data, &mut ad_ctx)
        .unwrap();

    let grad = gradients[0].as_ref().expect("active gradient");
    assert_eq!(grad.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
}

#[test]
fn eager_backward_prefills_split_linear_residual_graph_values() {
    let base_key = TensorInputKey::User { id: 124 };
    let exponent_key = TensorInputKey::User { id: 125 };
    let mut graph_builder = GraphBuilder::<StdTensorOp>::new();
    let x = graph_builder.add_input(base_key.clone());
    let exponent = graph_builder.add_input(exponent_key.clone());
    let y = graph_builder.add_operation(
        StdTensorOp::Pow,
        vec![ValueRef::Local(x), ValueRef::Local(exponent)],
        OperationRole::Primary,
    )[0];
    graph_builder.set_outputs(vec![y]);
    let graph = Arc::new(graph_builder.build());
    let output_key = graph.values()[y].key.clone();
    let x_tensor = Arc::new(Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap());
    let exponent_tensor =
        Arc::new(Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 2.0]).unwrap());
    let output_tensor = Arc::new(Tensor::from_vec_col_major(vec![2], vec![8.0_f64, 9.0]).unwrap());
    let _primal_scope = register_scoped_metadata_batch(vec![
        (
            ValueKey::Input(base_key.clone()),
            tensor_meta_from_tensor(x_tensor.as_ref()),
        ),
        (
            ValueKey::Input(exponent_key.clone()),
            tensor_meta_from_tensor(exponent_tensor.as_ref()),
        ),
        (
            output_key.clone(),
            tensor_meta_from_tensor(output_tensor.as_ref()),
        ),
    ])
    .unwrap();

    let view = resolve(vec![Arc::clone(&graph)]);
    let mut ad_ctx = ShapeGuardContext::with_global_metadata();
    let linear = linearize(
        &view,
        std::slice::from_ref(&output_key),
        &[base_key.clone(), exponent_key.clone()],
        0,
        &mut ad_ctx,
        &HashMap::new(),
    )
    .unwrap();
    assert!(
        !linear.residual_graph().operations().is_empty(),
        "Pow linearization should emit fixed residual operations"
    );

    let mut backend = CpuBackend::new();
    let mut external_data = HashMap::from([
        (ValueKey::Input(base_key), x_tensor),
        (ValueKey::Input(exponent_key), exponent_tensor),
        (output_key, output_tensor),
    ]);
    let before_len = external_data.len();
    prefill_linear_residual_values(&linear, &mut external_data, &mut backend, None).unwrap();

    assert!(
        external_data.len() > before_len,
        "residual prefill should add fixed linearization intermediates"
    );
}

#[test]
fn eager_backward_forward_replay_runs_only_live_fixed_linear_ops() {
    let input_key = TensorInputKey::User { id: 126 };
    let external_key = ValueKey::Input(input_key.clone());
    let mut graph_builder = GraphBuilder::<StdTensorOp>::new();
    let x = graph_builder.add_input(input_key.clone());
    graph_builder.add_operation(
        StdTensorOp::Neg,
        vec![ValueRef::Local(x)],
        OperationRole::Linearized {
            active_mask: vec![true],
        },
    );
    graph_builder.add_operation(
        StdTensorOp::Neg,
        vec![ValueRef::Local(x)],
        OperationRole::Primary,
    );
    let sum = graph_builder.add_operation(
        StdTensorOp::Add,
        vec![ValueRef::Local(x), ValueRef::External(external_key.clone())],
        OperationRole::Linearized {
            active_mask: vec![false, false],
        },
    )[0];
    graph_builder.set_outputs(vec![sum]);
    let graph = graph_builder.build();

    let x_tensor = Arc::new(Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap());
    let initial_data = HashMap::from([(external_key, x_tensor)]);
    let mut backend = CpuBackend::new();
    let mut callbacks = TenferroBackwardCallbacks::new(&mut backend, None, Vec::new());

    let all_values = callbacks.execute_forward_graph(&graph, &initial_data);

    let out_key = &graph.values()[sum].key;
    let out = all_values
        .get(out_key)
        .expect("live fixed op output should be replayed");
    assert_eq!(out.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    assert!(
        callbacks.take_error().is_none(),
        "forward replay should not record errors"
    );
}

#[test]
fn eager_backward_residual_value_synthesizes_missing_tangent_zero() {
    let user = TensorInputKey::User { id: 127 };
    let base_key = ValueKey::Input(user.clone());
    let tangent_key = ValueKey::Input(user.tangent_of(128));
    let base = Arc::new(Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 6.0]).unwrap());
    let mut all_values = HashMap::from([(base_key, base)]);
    let mut backend = CpuBackend::new();

    let tangent = eager_residual_value(&mut all_values, &tangent_key, &mut backend).unwrap();

    assert_eq!(tangent.as_slice::<f64>().unwrap(), &[0.0, 0.0]);
    assert!(Arc::ptr_eq(
        &tangent,
        all_values
            .get(&tangent_key)
            .expect("synthesized tangent should be cached")
    ));
}

#[test]
fn standard_graph_op_executes_untracked_outputs_without_trace() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
        ctx,
    )
    .unwrap();

    let outputs =
        EagerTensor::standard_graph_op(&[&lhs, &rhs], |keys| Ok(build_add_mul_reduce_graph(keys)))
            .expect("standard graph op");

    assert_eq!(outputs.len(), 1);
    assert!(!outputs[0].tracks_grad());
    assert_eq!(outputs[0].debug_trace_saved_value_count(), None);
    assert_eq!(outputs[0].shape(), &[] as &[usize]);
    assert_eq!(
        outputs[0]
            .materialized()
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[36.0]
    );
}

#[test]
fn standard_graph_op_records_one_tracked_graph_and_backpropagates() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let lhs = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
        ctx,
    )
    .unwrap();

    let mut outputs =
        EagerTensor::standard_graph_op(&[&lhs, &rhs], |keys| Ok(build_add_mul_reduce_graph(keys)))
            .expect("standard graph op");
    let loss = outputs.pop().expect("one output");

    assert!(loss.tracks_grad());
    assert_eq!(
        loss.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[36.0]
    );
    assert_eq!(loss.debug_trace_saved_value_count(), Some(5));

    loss.backward().expect("backward through recorded graph");
    assert_eq!(
        lhs.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[3.0, 4.0]
    );
    assert_eq!(
        rhs.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[7.0, 10.0]
    );
}

#[test]
fn eager_backward_with_accepts_vector_cotangent_seed() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let y = x.mul(&x).unwrap();

    y.backward_with(&seed).unwrap();

    assert_eq!(
        x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 12.0]
    );
}

#[test]
fn eager_backward_with_rejects_mismatched_seed_shape() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let y = x.mul(&x).unwrap();

    let err = y.backward_with(&seed).unwrap_err();

    assert!(
        err.to_string().contains("shape mismatch"),
        "unexpected error: {err}"
    );
    assert!(x.grad().unwrap().is_none());
}

#[test]
fn eager_runtime_vjp_returns_composable_tensor_without_touching_grad_slot() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let y = x.mul(&x).unwrap();

    let dx = ctx.vjp(&y, &x, &seed).unwrap();

    assert!(dx.tracks_grad());
    assert_eq!(
        dx.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 12.0]
    );
    assert!(x.grad().unwrap().is_none());
}

#[test]
fn eager_runtime_jvp_uses_forward_walker() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let tangent = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let y = x.mul(&x).unwrap();

    let dy = ctx.jvp(&y, &x, &tangent).unwrap();

    assert!(dy.tracks_grad());
    assert_eq!(
        dy.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
}

#[test]
fn eager_runtime_ad_transform_cache_reuses_recorded_graph_linearization() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let y = x.mul(&x).unwrap();

    assert_eq!(ctx.cache_stats().unwrap().ad_transforms.entries, 0);

    let first = ctx.vjp(&y, &x, &seed).unwrap();
    assert_eq!(
        first.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
    let after_first = ctx.cache_stats().unwrap().ad_transforms;
    assert!(after_first.entries > 0);

    let second = ctx.vjp(&y, &x, &seed).unwrap();
    assert_eq!(
        second.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
    let after_second = ctx.cache_stats().unwrap().ad_transforms;
    assert_eq!(after_second.entries, after_first.entries);
    assert_eq!(after_second.retained_bytes, after_first.retained_bytes);

    ctx.clear_caches().unwrap();
    assert_eq!(ctx.cache_stats().unwrap().ad_transforms.entries, 0);

    let tangent = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let dy = ctx.jvp(&y, &x, &tangent).unwrap();
    assert_eq!(
        dy.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
    assert!(ctx.cache_stats().unwrap().ad_transforms.entries > 0);
}

#[test]
fn eager_functional_grad_can_feed_jvp() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let tangent = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let loss = x.mul(&x).unwrap();

    let grad = ctx.grad(&loss, &x).unwrap();
    let hvp = ctx.jvp(&grad, &x, &tangent).unwrap();

    assert!(grad.tracks_grad());
    assert_eq!(
        grad.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[6.0]
    );
    assert_eq!(
        hvp.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[2.0]
    );
}

#[test]
fn eager_jvp_of_functional_grad_matches_cubic_hvp() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let tangent = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![], vec![1.0_f64]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let x2 = x.mul(&x).unwrap();
    let loss = x2.mul(&x).unwrap();

    let grad = ctx.grad(&loss, &x).unwrap();
    let hvp = ctx.jvp(&grad, &x, &tangent).unwrap();

    assert_eq!(
        grad.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[27.0]
    );
    assert_eq!(
        hvp.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[18.0]
    );
}

#[test]
fn eager_no_grad_scope_suppresses_operation_recording() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();

    let y = {
        let _guard = ctx.no_grad();
        x.mul(&x).unwrap()
    };
    let z = x.mul(&x).unwrap();

    assert!(!y.tracks_grad());
    assert!(z.tracks_grad());
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
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        lhs_ctx,
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
        rhs_ctx,
    )
    .unwrap();

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
        Tensor::from_vec_col_major(vec![1], vec![1.0_f32]).unwrap(),
        Tensor::from_vec_col_major(vec![1], vec![1_i32]).unwrap(),
        Tensor::from_vec_col_major(vec![1], vec![1_i64]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap(),
        Tensor::from_vec_col_major(vec![1], vec![num_complex::Complex32::new(1.0, -1.0)]).unwrap(),
        Tensor::from_vec_col_major(vec![1], vec![num_complex::Complex64::new(1.0, -1.0)]).unwrap(),
    ];

    for input in cases {
        let zero = zero_like_tensor(&input, &mut backend).unwrap();
        assert_eq!(zero.shape(), input.shape());
    }
}

#[test]
fn one_like_tensor_covers_integer_and_bool_dtypes_without_analytic_backend_ops() {
    let mut backend = CpuBackend::new();
    let cases = [
        Tensor::from_vec_col_major(vec![1], vec![1_i32]).unwrap(),
        Tensor::from_vec_col_major(vec![1], vec![1_i64]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap(),
    ];

    for input in cases {
        let one = one_like_tensor(&input, &mut backend).unwrap();
        assert_eq!(one.shape(), input.shape());
        assert_eq!(one.dtype(), input.dtype());
    }
}

#[test]
fn eager_backend_delegates_broadcast_multiply_fusion_to_cpu_backend() {
    let mut backend = EagerBackend::cpu(CpuBackend::new());
    let lhs = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3], vec![5.0_f64, 7.0, 11.0]).unwrap();

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
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx,
    )
    .unwrap();
    let x_t = x.transpose(&[1, 0]).unwrap();
    assert!(matches!(x_t.tensor_read(), TensorRead::View(_)));
    assert!(!x_t.materialized_cache_is_initialized());

    let doubled = x_t.add(&x_t).unwrap();
    assert!(!x_t.materialized_cache_is_initialized());
    assert_eq!(
        doubled.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 6.0, 10.0, 4.0, 8.0, 12.0]
    );

    let reduced = x_t.reduce_sum(&[0]).unwrap();
    assert!(!x_t.materialized_cache_is_initialized());
    assert_eq!(
        reduced.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[9.0, 12.0]
    );

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
        dot.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[5.0, 11.0, 17.0, 11.0, 25.0, 39.0, 17.0, 39.0, 61.0]
    );
}
