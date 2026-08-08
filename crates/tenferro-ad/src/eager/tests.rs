use std::any::Any;
use std::collections::HashMap;
use std::hash::Hasher;
use std::num::NonZeroUsize;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

use crate::context::AdContext;
use computegraph::graph::{Graph, GraphBuilder};
use computegraph::{OperationRole, ValueKey, ValueRef};
use tenferro_cpu::CpuBackend;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::SymDim;
use tenferro_runtime::ExtensionCacheLimits;
use tenferro_runtime::{
    Error, ExecutionContextIdentity, ExtensionEngine, ExtensionModule, ExtensionModuleError,
    ExtensionModuleId, ExtensionModuleRegistrar, ExtensionPlanningConfig, ExtensionPrepareRequest,
    GraphCompiler, PrepareCapability, PrepareError, Runtime, UnsupportedReason,
};
use tenferro_tensor::TypedTensorView;
use tenferro_tensor::{AllocationGroup, DescriptorSlot, GroupError, Tensor};
use tenferro_tensor::{
    BackendSession, BackendSessionHost, DType, DotGeneralConfig, TensorElementwise,
};
use tenferro_tensor::{ErrorKind, ValidationKind};
use tenferro_tensor::{TensorFusion, TensorRead, TensorStructural, TensorView, TensorWrite};

use crate::eager_backend::EagerBackend;
use crate::eager_exec::exec_op_on_tensor_reads_with_runtime;

mod placement_bound;
mod runtime_snapshot;

use super::{
    eager_op_profile_enabled, eager_op_profile_per_call_us, eager_op_profile_start,
    maybe_print_eager_op_profile, one_like_tensor, print_and_reset_eager_op_profile,
    profile_eager_op_section, record_eager_op_profile, zero_like_tensor, EagerOpProfileEntry,
    EagerRuntime, EagerTensor,
};

#[test]
fn gradients_take_grad_preserves_group_error_source() {
    let (group, bindings) = AllocationGroup::from_tensors(Vec::new()).unwrap();
    assert!(bindings.is_empty());
    let missing = DescriptorSlot::from_index(0).unwrap();
    let key = ValueKey::Input(TensorInputKey::User { id: 991 });
    let mut gradients = super::Gradients {
        group,
        slots: HashMap::from([(key.clone(), missing)]),
    };

    let error = gradients.take_grad(&key).unwrap_err();
    let mut current: &(dyn std::error::Error + 'static) = &error;
    let mut found = false;
    while let Some(source) = current.source() {
        if source.downcast_ref::<GroupError>().is_some() {
            found = true;
            break;
        }
        current = source;
    }
    assert!(found, "group error was not retained as an error source");
}

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
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = ctx.backend.lock().unwrap();
        panic!("poison eager backend lock");
    }));
    assert!(poisoned.is_err());

    let err = ctx.synchronize().unwrap_err();

    assert_eq!(err.kind(), ErrorKind::RuntimeState);
    assert!(matches!(
        err,
        Error::RuntimeState {
            phase: tenferro_runtime::ErrorPhase::Execution,
            ..
        }
    ));
}

#[test]
fn eager_runtime_execution_session_runs_cpu_operation() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();

    let output = runtime
        .with_execution_session(|session| TensorElementwise::add(session, &lhs, &rhs))
        .unwrap()
        .unwrap();

    assert_eq!(output.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
}

#[test]
fn eager_backend_session_identity_projects_to_owner() {
    let mut backend = EagerBackend::cpu(CpuBackend::new());
    let owner = (&mut backend as *mut EagerBackend).cast::<()>();
    let identity = backend.session_type_id();
    let projected = unsafe { backend.session_data_mut() };
    assert_eq!(identity, backend.session_type_id());
    assert_eq!(projected, owner);

    let materializations = Arc::new(AtomicUsize::new(0));
    let mut recording = EagerBackend::recording_cpu(materializations);
    let recording_owner = recording.recording_session_owner().unwrap() as usize;
    let recording_identity = recording.with_backend_session(|session| {
        let identity = session.session_type_id();
        let projected = unsafe { session.session_data_mut() };
        assert_eq!(projected as usize, recording_owner);
        identity
    });
    assert_ne!(recording_identity, identity);
    assert_eq!(
        recording_identity,
        recording.with_backend_session(|session| { session.session_type_id() })
    );
    assert!(backend.recording_session_owner().is_none());
}

#[test]
fn eager_materialization_uses_backend() {
    let mut cpu_backend = EagerBackend::cpu(CpuBackend::new());
    assert!(format!("{cpu_backend:?}").contains("Cpu"));
    cpu_backend.synchronize().unwrap();

    let materializations = Arc::new(AtomicUsize::new(0));
    let mut backend = EagerBackend::recording_cpu(Arc::clone(&materializations));
    assert!(format!("{backend:?}").contains("Recording"));
    backend.synchronize().unwrap();
    let probe = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let sum = TensorElementwise::add(&mut backend, &probe, &probe).unwrap();
    assert_eq!(sum.as_slice::<f64>().unwrap(), &[4.0]);
    assert_eq!(materializations.load(Ordering::Relaxed), 0);

    let view_data = [1.0_f64, 2.0, 3.0, 4.0];
    let view = TensorView::F64(
        TypedTensorView::from_col_major(&[2, 2], &view_data)
            .unwrap()
            .transpose_view([1, 0])
            .unwrap(),
    );
    let direct =
        TensorStructural::to_contiguous_read(&mut backend, TensorRead::from_view(view)).unwrap();
    assert_eq!(direct.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    assert_eq!(materializations.swap(0, Ordering::Relaxed), 1);

    let mut destination = Tensor::from_vec_col_major(vec![1], vec![0.0_f64]).unwrap();
    TensorStructural::copy_read_into(
        &mut backend,
        TensorRead::from_tensor(&probe),
        TensorWrite::from_tensor(&mut destination),
    )
    .unwrap();
    assert_eq!(destination.as_slice::<f64>().unwrap(), &[2.0]);
    let ctx = Arc::new(EagerRuntime::from_backend(backend).unwrap());
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();

    let compact = x.to_tensor().unwrap();
    assert_eq!(compact.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(materializations.load(Ordering::Relaxed), 1);

    let view = x.transpose(&[1, 0]).unwrap();
    let compact = view.to_tensor().unwrap();
    assert_eq!(compact.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    assert_eq!(materializations.load(Ordering::Relaxed), 2);
}

#[test]
fn untracked_standard_op_results_do_not_enter_value_record_registry() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f64]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1, 1], vec![3.0_f64]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let records_before = ctx.value_records.lock().unwrap().len();

    let output = lhs
        .dot_general(
            &rhs,
            DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap();

    assert!(!output.tracks_grad());
    assert_eq!(ctx.value_records.lock().unwrap().len(), records_before);
}

#[test]
fn eager_runtime_public_helpers_do_not_unwrap_poisoned_locks() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = ctx.extension_caches.lock().unwrap();
        panic!("poison extension cache lock");
    }));
    assert!(poisoned.is_err());

    assert!(ctx.clear_extension_caches().is_err());
    assert!(ctx.cache_stats().is_err());
    assert!(ctx.extension_cache_limits().is_err());
    assert!(ctx
        .set_extension_cache_limits(ExtensionCacheLimits::new(NonZeroUsize::new(1).unwrap(),))
        .is_err());
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
fn eager_runtime_execution_session_reports_poisoned_backend_lock() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = ctx.backend.lock().unwrap();
        panic!("poison eager backend lock");
    }));
    assert!(poisoned.is_err());

    let err = ctx.with_execution_session(|_| ()).unwrap_err();

    assert_eq!(err.kind(), ErrorKind::RuntimeState);
    assert!(matches!(
        err,
        Error::RuntimeState {
            phase: tenferro_runtime::ErrorPhase::Execution,
            ..
        }
    ));
}

#[test]
fn eager_index_select_reports_poisoned_backend_lock() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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

    assert_eq!(err.kind(), ErrorKind::RuntimeState);
    assert!(matches!(
        err,
        Error::RuntimeState {
            phase: tenferro_runtime::ErrorPhase::Execution,
            ..
        }
    ));
}

#[test]
fn eager_runtime_gradient_helpers_do_not_unwrap_poisoned_locks() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
}

#[test]
fn tensor_read_extension_path_errors_when_runtime_family_is_missing() {
    let op = StdTensorOp::Extension(Arc::new(ReadPathFallbackProbe));
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let reads = [TensorRead::from_tensor(&input)];
    let mut backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
        .unwrap();
    let runtime = builder.build().unwrap();

    let err = exec_op_on_tensor_reads_with_runtime(&op, &reads, &mut backend, Some(&runtime))
        .expect_err("runtime owner with missing extension module must not eager fallback");

    let message = err.to_string();
    assert!(message.contains("missing extension module"), "{message}");
    assert!(
        message.contains("tenferro-tests.read-path-fallback-probe.v1"),
        "{message}"
    );
}

#[derive(Debug)]
struct ReadPathFallbackConfig;

impl ExtensionPlanningConfig for ReadPathFallbackConfig {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.read-path-fallback-probe.v1"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn payload_hash(&self, _state: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionPlanningConfig) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

#[derive(Debug)]
struct ReadPathFallbackEngine {
    engine_id: tenferro_runtime::EngineId,
}

impl ExtensionEngine for ReadPathFallbackEngine {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.read-path-fallback-probe.v1"
    }

    fn engine_id(&self) -> &tenferro_runtime::EngineId {
        &self.engine_id
    }

    fn context_identity(&self) -> ExecutionContextIdentity {
        ExecutionContextIdentity::of::<CpuBackend>()
    }

    fn prepare(
        &self,
        _request: ExtensionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        Ok(PrepareCapability::Unsupported(
            UnsupportedReason::Operation {
                operation: "read-path-fallback-test",
            },
        ))
    }
}

#[derive(Debug)]
struct ReadPathFallbackModule {
    module_id: ExtensionModuleId,
}

impl ReadPathFallbackModule {
    fn module() -> Arc<dyn ExtensionModule> {
        Arc::new(Self {
            module_id: ExtensionModuleId::new("tenferro-tests.read-path-fallback.module").unwrap(),
        })
    }
}

impl ExtensionModule for ReadPathFallbackModule {
    fn module_id(&self) -> &ExtensionModuleId {
        &self.module_id
    }

    fn configure(
        &self,
        registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> std::result::Result<(), ExtensionModuleError> {
        let engine_id = tenferro_cpu::runtime_engine_id().unwrap();
        registrar.register_engine(Arc::new(ReadPathFallbackEngine {
            engine_id: engine_id.clone(),
        }))?;
        registrar.register_planning_config(engine_id, Arc::new(ReadPathFallbackConfig))
    }
}

#[test]
fn eager_extension_dispatch_does_not_initialize_lazy_view_materialization_cache() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();

    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx,
    )
    .unwrap();
    let x_t = x.transpose(&[1, 0]).unwrap();
    assert!(matches!(x_t.tensor_read(), TensorRead::View(_)));

    let outputs = crate::extension::apply_eager_with_extension_session(
        Arc::new(ReadPathFallbackProbe),
        &[&x_t],
        |_target| Ok(ReadPathFallbackModule::module()),
        |op, inputs, ctx| {
            op.as_any()
                .downcast_ref::<ReadPathFallbackProbe>()
                .expect("test op payload");
            let backend = ctx.backend_mut();
            let materialized_inputs = inputs
                .iter()
                .cloned()
                .map(|input| TensorStructural::to_contiguous_read(backend, input))
                .collect::<tenferro_tensor::Result<Vec<_>>>();
            let materialized_inputs = materialized_inputs?;
            let input_refs: Vec<&Tensor> = materialized_inputs.iter().collect();
            Ok(vec![input_refs[0].duplicate()?])
        },
    )
    .expect("eager extension dispatch");

    assert_eq!(outputs.len(), 1);
    assert_eq!(
        outputs[0].to_tensor().unwrap().as_slice::<f64>().unwrap(),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn eager_op_profile_helpers_cover_enabled_paths() {
    let _guard = EagerOpProfileOverrideGuard::set(true, Some(2));

    assert!(eager_op_profile_enabled());
    assert!(eager_op_profile_start().is_some());
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
fn eager_op_profile_start_respects_enabled_gate() {
    let _guard = EagerOpProfileOverrideGuard::set(false, None);

    assert!(!eager_op_profile_enabled());
    assert!(eager_op_profile_start().is_none());
}

#[test]
fn standard_graph_op_executes_untracked_outputs_without_trace() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
        outputs[0].to_tensor().unwrap().as_slice::<f64>().unwrap(),
        &[36.0]
    );
}

#[test]
fn standard_graph_op_records_one_tracked_graph_and_backpropagates() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
    assert_eq!(loss.value().unwrap().as_slice::<f64>().unwrap(), &[36.0]);
    assert_eq!(loss.debug_trace_saved_value_count(), None);

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
fn eager_recording_retains_symbolic_semantic_trace_for_shape_churn() {
    fn compile_square(values: Vec<f64>) -> tenferro_runtime::CompiledGraph {
        let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
        let len = values.len();
        let x = EagerTensor::requires_grad_in(
            Tensor::from_vec_col_major(vec![len], values).unwrap(),
            ctx,
        )
        .unwrap();
        let y = x.mul(&x).unwrap();
        let semantic_trace = y
            .semantic_trace
            .as_ref()
            .expect("tracked eager output should retain a semantic trace");
        assert!(
            !semantic_trace.is_concrete_shape(),
            "eager semantic recording should keep input extents symbolic"
        );

        let mut compiler = GraphCompiler::new();
        compiler.compile(semantic_trace).unwrap()
    }

    let program2 = compile_square(vec![2.0, 3.0]);
    let program3 = compile_square(vec![5.0, 7.0, 11.0]);

    assert_eq!(program2.input_count(), 1);
    assert_eq!(program2.output_count(), 1);
    assert_eq!(
        program2.program().semantic_fingerprint(),
        program3.program().semantic_fingerprint()
    );
    assert!(program2.program().semantic_eq(program3.program()));
    assert_eq!(program2.bindings().iter().next().unwrap().1.shape(), &[2]);
    assert_eq!(program3.bindings().iter().next().unwrap().1.shape(), &[3]);
}

#[test]
fn eager_runtime_vjp_can_use_semantic_trace_when_gate_enabled() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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

    let vjp = ctx.vjp(&y, &x, &seed).unwrap();

    assert_eq!(
        vjp.value().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 12.0]
    );
}

#[test]
fn eager_runtime_vjp_uses_semantic_trace_for_multi_input_graph_when_gate_enabled() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let output = x.mul(&y).unwrap();

    let dx = ctx.vjp(&output, &x, &seed).unwrap();
    let dy = ctx.vjp(&output, &y, &seed).unwrap();

    assert_eq!(dx.value().unwrap().as_slice::<f64>().unwrap(), &[5.0, 14.0]);
    assert_eq!(dy.value().unwrap().as_slice::<f64>().unwrap(), &[2.0, 6.0]);
}

#[test]
fn eager_backward_with_accepts_vector_cotangent_seed() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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

    assert_eq!(
        err.kind(),
        ErrorKind::Validation(ValidationKind::ShapeMismatch)
    );
    assert!(matches!(
        err,
        Error::TensorRuntime(tenferro_tensor::Error::Validation {
            source: tenferro_tensor::ValidationError::ShapeMismatch(_),
            ..
        })
    ));
    assert!(x.grad().unwrap().is_none());
}

#[test]
fn eager_runtime_vjp_returns_composable_tensor_without_touching_grad_slot() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
    assert_eq!(dx.value().unwrap().as_slice::<f64>().unwrap(), &[4.0, 12.0]);
    assert!(x.grad().unwrap().is_none());
}

#[test]
fn eager_functional_ad_reports_inactive_inputs_and_accepts_explicit_rule_context() {
    let ad = AdContext::builder().build().unwrap();
    let ctx = EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::new(), &ad).unwrap();
    let active = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let inactive = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let loss = active.mul(&active).unwrap();

    let inactive_vjp = ctx.vjp_optional(&loss, &inactive, &seed).unwrap();
    let inactive_jvp = ctx.jvp_optional(&loss, &inactive, &seed).unwrap();
    assert!(inactive_vjp.is_none());
    assert!(inactive_jvp.is_none());

    let active_vjp = ctx.vjp_optional(&loss, &active, &seed).unwrap().unwrap();
    let active_jvp = ctx.jvp_optional(&loss, &active, &seed).unwrap().unwrap();
    assert_eq!(
        active_vjp.to_tensor().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
    assert_eq!(
        active_jvp.to_tensor().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
}

#[test]
fn eager_runtime_jvp_returns_composable_semantic_trace() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
    assert_eq!(dy.value().unwrap().as_slice::<f64>().unwrap(), &[4.0, 6.0]);
}

#[test]
fn eager_runtime_ad_transform_cache_reuses_recorded_graph_linearization() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
        first.value().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
    let after_first = ctx.cache_stats().unwrap().ad_transforms;
    assert!(after_first.entries > 0);

    let second = ctx.vjp(&y, &x, &seed).unwrap();
    assert_eq!(
        second.value().unwrap().as_slice::<f64>().unwrap(),
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
    assert_eq!(dy.value().unwrap().as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    assert!(ctx.cache_stats().unwrap().ad_transforms.entries > 0);
}

#[test]
fn eager_prepared_derivative_cache_reuses_runtime_preparation() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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

    assert_eq!(ctx.runtime.cache_stats().unwrap().prepared_plans.entries, 0);

    let first = ctx.vjp(&y, &x, &seed).unwrap();
    assert_eq!(
        first.value().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
    let after_first = ctx.runtime.cache_stats().unwrap().prepared_plans;
    assert!(after_first.entries > 0);

    let second = ctx.vjp(&y, &x, &seed).unwrap();
    assert_eq!(
        second.value().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
    let after_second = ctx.runtime.cache_stats().unwrap().prepared_plans;
    assert_eq!(after_second.entries, after_first.entries);
    assert_eq!(after_second.hits, after_first.hits);
    assert_eq!(after_second.misses, after_first.misses);
    assert_eq!(after_second.preparations, after_first.preparations);
}

#[test]
fn eager_prepared_derivative_cache_is_visible_and_clearable() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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

    assert_eq!(ctx.cache_stats().unwrap().prepared_derivatives.entries, 0);

    let first = ctx.vjp(&y, &x, &seed).unwrap();
    assert_eq!(
        first.value().unwrap().as_slice::<f64>().unwrap(),
        &[4.0, 6.0]
    );
    let after_first = ctx.cache_stats().unwrap().prepared_derivatives;
    assert_eq!(after_first.entries, 1);
    assert!(after_first.retained_bytes > 0);

    ctx.clear_prepared_derivative_cache().unwrap();
    let after_clear = ctx.cache_stats().unwrap().prepared_derivatives;
    assert_eq!(after_clear.entries, 0);
    assert_eq!(after_clear.retained_bytes, 0);
    assert_eq!(after_clear.clears, after_first.clears + 1);
}

#[test]
fn eager_prepared_derivative_cache_limit_evicts_lru_entries() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    ctx.set_prepared_derivative_cache_limits(crate::AdTransformCacheLimits::new(
        NonZeroUsize::new(1).unwrap(),
    ))
    .unwrap();
    assert_eq!(
        ctx.prepared_derivative_cache_limits()
            .unwrap()
            .max_entries()
            .get(),
        1
    );

    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let y = x.mul(&x).unwrap();
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 1.0]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let _ = ctx.vjp(&y, &x, &seed).unwrap();
    assert_eq!(ctx.cache_stats().unwrap().prepared_derivatives.entries, 1);

    let z = x.add(&x).unwrap();
    let _ = ctx.vjp(&z, &x, &seed).unwrap();
    let stats = ctx.cache_stats().unwrap().prepared_derivatives;
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.evictions, 1);
}

#[test]
fn eager_functional_grad_can_feed_jvp() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
    assert_eq!(grad.value().unwrap().as_slice::<f64>().unwrap(), &[6.0]);
    assert_eq!(hvp.value().unwrap().as_slice::<f64>().unwrap(), &[2.0]);
}

#[test]
fn eager_jvp_of_functional_grad_matches_cubic_hvp() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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

    assert_eq!(grad.value().unwrap().as_slice::<f64>().unwrap(), &[27.0]);
    assert_eq!(hvp.value().unwrap().as_slice::<f64>().unwrap(), &[18.0]);
}

#[test]
fn eager_no_grad_scope_suppresses_operation_recording() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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

    let lhs_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let rhs_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
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
fn eager_backend_delegates_elementwise_into_hook_to_cpu_variants() {
    let materializations = Arc::new(AtomicUsize::new(0));
    let backends = [
        EagerBackend::cpu(CpuBackend::new()),
        EagerBackend::recording_cpu(Arc::clone(&materializations)),
    ];

    for mut backend in backends {
        let lhs = Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 3.0, 5.0]).unwrap();
        let rhs = Tensor::from_vec_col_major(vec![3], vec![7.0_f64, 11.0, 13.0]).unwrap();
        let mut out = Tensor::from_vec_col_major(vec![3], vec![0.0_f64; 3]).unwrap();

        backend
            .add_read_into(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                TensorWrite::from_tensor(&mut out),
            )
            .unwrap();

        assert_eq!(out.as_slice::<f64>().unwrap(), &[9.0, 14.0, 18.0]);
    }
}

#[test]
fn untracked_nary_ops_consume_lazy_views_without_materializing_inputs() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx,
    )
    .unwrap();
    let x_t = x.transpose(&[1, 0]).unwrap();
    assert!(matches!(x_t.tensor_read(), TensorRead::View(_)));

    let doubled = x_t.add(&x_t).unwrap();
    assert_eq!(
        doubled.value().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 6.0, 10.0, 4.0, 8.0, 12.0]
    );

    let reduced = x_t.reduce_sum(Some(&[0])).unwrap();
    assert_eq!(
        reduced.value().unwrap().as_slice::<f64>().unwrap(),
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
    assert_eq!(
        dot.value().unwrap().as_slice::<f64>().unwrap(),
        &[5.0, 11.0, 17.0, 11.0, 25.0, 39.0, 17.0, 39.0, 61.0]
    );
}

#[test]
fn eager_gradients_bundle_borrows_and_extracts_one_owner() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap(),
        ctx,
    )
    .unwrap();
    let loss = x.mul(&x).unwrap().reduce_sum(Some(&[0])).unwrap();
    let mut gradients = loss.backward().unwrap();

    let view = gradients.grad(&x.key).unwrap();
    assert_eq!(view.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    let extracted = gradients.take_grad(&x.key).unwrap().unwrap();
    assert_eq!(extracted.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    assert!(gradients.grad(&x.key).is_none());
}

#[test]
fn eager_retention_exposes_borrowed_values_and_explicit_duplication() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        ctx,
    )
    .unwrap();
    let sibling = x.clone();

    let value = x.value().unwrap();
    assert_eq!(value.shape(), &[2]);
    drop(value);

    let duplicate = x.duplicate_value().unwrap();
    assert_eq!(duplicate.as_slice::<f64>().unwrap(), &[1.0, 2.0]);

    let error = x.into_value().unwrap_err();
    assert!(matches!(error, crate::IntoValueError::NotUnique(_)));
    drop(sibling);
}
