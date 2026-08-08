use std::any::Any;
use std::hash::Hasher;
use std::sync::{Arc, Mutex};

use super::{
    apply_eager_with_extension_session, EagerExtensionBackendKind, EagerExtensionTarget,
    ExtensionOp,
};
use crate::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_ops::{ExtensionShapeContext, SymDim};
use tenferro_runtime::{
    EngineId, Error, ErrorPhase, ExecutionContextIdentity, ExtensionEngine, ExtensionModule,
    ExtensionModuleError, ExtensionModuleId, ExtensionModuleRegistrar, ExtensionPlanningConfig,
    ExtensionPrepareRequest, PrepareCapability, PrepareError, Runtime, RuntimeConfigError,
    UnsupportedReason,
};
#[cfg(any(feature = "cuda", feature = "webgpu"))]
use tenferro_tensor::TensorRead;
use tenferro_tensor::{DType, Tensor, TensorStructural};

#[derive(Clone, Debug)]
struct BridgeProbe {
    inputs: usize,
}

impl BridgeProbe {
    fn one() -> Self {
        Self { inputs: 1 }
    }

    fn two() -> Self {
        Self { inputs: 2 }
    }
}

impl ExtensionOp for BridgeProbe {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.eager-extension-bridge.v1"
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.inputs == self.inputs)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        self.inputs
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }
}

#[derive(Debug)]
struct BridgeConfig {
    family_id: &'static str,
}

impl ExtensionPlanningConfig for BridgeConfig {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn payload_hash(&self, state: &mut dyn Hasher) {
        state.write(self.family_id.as_bytes());
    }

    fn payload_eq(&self, other: &dyn ExtensionPlanningConfig) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self.family_id == other.family_id)
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

#[derive(Debug)]
struct BridgeEngine {
    family_id: &'static str,
    engine_id: EngineId,
}

impl ExtensionEngine for BridgeEngine {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn engine_id(&self) -> &EngineId {
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
                operation: "eager-extension-bridge-test",
            },
        ))
    }
}

#[derive(Debug)]
struct BridgeModule {
    module_id: ExtensionModuleId,
    engine_id: Option<EngineId>,
}

impl BridgeModule {
    fn without_engine() -> Arc<dyn ExtensionModule> {
        Arc::new(Self {
            module_id: ExtensionModuleId::new("tenferro-tests.eager-extension-bridge.module")
                .unwrap(),
            engine_id: None,
        })
    }

    fn for_engine(engine_id: EngineId) -> Arc<dyn ExtensionModule> {
        Arc::new(Self {
            module_id: ExtensionModuleId::new("tenferro-tests.eager-extension-bridge.module")
                .unwrap(),
            engine_id: Some(engine_id),
        })
    }
}

impl ExtensionModule for BridgeModule {
    fn module_id(&self) -> &ExtensionModuleId {
        &self.module_id
    }

    fn configure(
        &self,
        registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> Result<(), ExtensionModuleError> {
        let Some(engine_id) = &self.engine_id else {
            return Ok(());
        };
        registrar.register_engine(Arc::new(BridgeEngine {
            family_id: BridgeProbe::one().family_id(),
            engine_id: engine_id.clone(),
        }))?;
        registrar.register_planning_config(
            engine_id.clone(),
            Arc::new(BridgeConfig {
                family_id: BridgeProbe::one().family_id(),
            }),
        )
    }
}

fn input(ctx: &Arc<EagerRuntime>) -> EagerTensor {
    EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
        Arc::clone(ctx),
    )
    .unwrap()
}

fn execute_probe(
    op: Arc<dyn ExtensionOp>,
    input: &EagerTensor,
    factory: impl FnOnce(EagerExtensionTarget) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>>,
) -> tenferro_runtime::Result<Vec<EagerTensor>> {
    apply_eager_with_extension_session(op, &[input], factory, |_op, inputs, ctx| {
        let output = TensorStructural::to_contiguous_read(ctx.backend_mut(), inputs[0].clone())?;
        Ok(vec![output])
    })
}

#[test]
fn eager_extension_factory_receives_exact_cpu_target() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let input = input(&ctx);
    let seen = Arc::new(Mutex::new(None));
    let seen_by_factory = Arc::clone(&seen);
    let expected_engine = tenferro_cpu::runtime_engine_id().unwrap();

    execute_probe(Arc::new(BridgeProbe::one()), &input, move |target| {
        *seen_by_factory.lock().unwrap() = Some(target.clone());
        Ok(BridgeModule::for_engine(target.engine_id))
    })
    .unwrap();

    assert_eq!(
        *seen.lock().unwrap(),
        Some(EagerExtensionTarget {
            engine_id: expected_engine,
            backend_kind: EagerExtensionBackendKind::Cpu,
        })
    );
}

#[test]
fn eager_extension_input_context_mismatch_is_reported_before_factory() {
    let lhs_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let rhs_ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let lhs = input(&lhs_ctx);
    let rhs = input(&rhs_ctx);
    let called = Arc::new(Mutex::new(false));
    let called_by_factory = Arc::clone(&called);

    let error = apply_eager_with_extension_session(
        Arc::new(BridgeProbe::two()),
        &[&lhs, &rhs],
        move |_target| {
            *called_by_factory.lock().unwrap() = true;
            Ok(BridgeModule::without_engine())
        },
        |_op, _inputs, _ctx| unreachable!("input validation must run first"),
    )
    .unwrap_err();

    assert!(matches!(error, Error::ContextMismatch { .. }));
    assert!(!*called.lock().unwrap());
}

#[test]
fn eager_extension_missing_selected_engine_retains_typed_runtime_source() {
    let runtime = Runtime::builder().build().unwrap();
    let target = EagerExtensionTarget {
        engine_id: EngineId::new("tenferro-tests.missing-eager-engine").unwrap(),
        backend_kind: EagerExtensionBackendKind::Cpu,
    };

    let error = super::validate_eager_extension_target(&runtime, &target).unwrap_err();
    assert!(matches!(error, Error::RuntimeStateSource { .. }));
    let source = std::error::Error::source(&error).unwrap();
    let source = source
        .downcast_ref::<RuntimeConfigError>()
        .expect("missing engine source should retain RuntimeConfigError");
    assert!(matches!(
        source,
        RuntimeConfigError::MissingEngine { engine_id }
            if engine_id == &target.engine_id
    ));
}

#[test]
fn eager_extension_target_rejects_extension_engine_without_direct_engine() {
    let target = EagerExtensionTarget {
        engine_id: EngineId::new("tenferro-tests.placement-engine").unwrap(),
        backend_kind: EagerExtensionBackendKind::Cpu,
    };
    let mut builder = Runtime::builder();
    builder
        .install_extension_module(BridgeModule::for_engine(target.engine_id.clone()))
        .unwrap();
    let runtime = builder.build().unwrap();

    let error = super::validate_eager_extension_target(&runtime, &target).unwrap_err();
    let Error::RuntimeStateSource { source, .. } = error else {
        panic!("expected structured missing direct engine error");
    };
    let source = source
        .downcast_ref::<RuntimeConfigError>()
        .expect("missing direct engine source");
    assert!(matches!(
        source,
        RuntimeConfigError::MissingEngine { engine_id } if engine_id == &target.engine_id
    ));
}

#[test]
fn eager_extension_module_without_target_engine_is_rejected_before_backend_session() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let input = input(&ctx);
    let error = execute_probe(Arc::new(BridgeProbe::one()), &input, |_target| {
        Ok(BridgeModule::without_engine())
    })
    .unwrap_err();
    let Error::RuntimeStateSource { source, .. } = error else {
        panic!("expected structured missing extension engine error");
    };
    let source = source
        .downcast_ref::<RuntimeConfigError>()
        .expect("missing extension engine source");
    let expected_engine = tenferro_cpu::runtime_engine_id().unwrap();
    assert!(matches!(
        source,
        RuntimeConfigError::MissingExtensionEngine { family_id, engine_id }
            if *family_id == BridgeProbe::one().family_id()
                && engine_id == &expected_engine
    ));
}

#[test]
fn eager_extension_module_for_different_engine_is_rejected_before_backend_session() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let input = input(&ctx);
    let expected_engine = tenferro_cpu::runtime_engine_id().unwrap();
    let different_engine = EngineId::new("tenferro-tests.other-eager-engine").unwrap();

    let error = execute_probe(Arc::new(BridgeProbe::one()), &input, |_target| {
        Ok(BridgeModule::for_engine(different_engine))
    })
    .unwrap_err();
    let Error::RuntimeStateSource { source, .. } = error else {
        panic!("expected structured mismatched extension engine error");
    };
    let source = source
        .downcast_ref::<RuntimeConfigError>()
        .expect("mismatched extension engine source");
    assert!(matches!(
        source,
        RuntimeConfigError::MissingExtensionEngine { family_id, engine_id }
            if *family_id == BridgeProbe::one().family_id()
                && engine_id == &expected_engine
    ));
}

#[test]
fn eager_extension_missing_module_is_rejected_for_constructible_runtime() {
    let mut backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap())
        .unwrap();
    let runtime = builder.build().unwrap();
    let target = EagerExtensionTarget {
        engine_id: tenferro_cpu::runtime_engine_id().unwrap(),
        backend_kind: EagerExtensionBackendKind::Cpu,
    };

    let error =
        super::validate_eager_extension_module(&runtime, BridgeProbe::one().family_id(), &target)
            .unwrap_err();
    let Error::RuntimeStateSource { source, .. } = error else {
        panic!("expected structured missing extension engine error");
    };
    let source = source
        .downcast_ref::<RuntimeConfigError>()
        .expect("missing module source");
    assert!(matches!(
        source,
        RuntimeConfigError::MissingExtensionEngine { family_id, engine_id }
            if *family_id == BridgeProbe::one().family_id()
                && engine_id == &target.engine_id
    ));
    let _ = &mut backend;
}

#[test]
fn eager_extension_factory_errors_are_returned_without_entering_backend_session() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let input = input(&ctx);
    let missing_engine = EngineId::new("tenferro-tests.factory-missing-engine").unwrap();
    let error = execute_probe(Arc::new(BridgeProbe::one()), &input, |_target| {
        Err(Error::runtime_state_source(
            "test-extension-factory",
            ErrorPhase::Execution,
            RuntimeConfigError::MissingEngine {
                engine_id: missing_engine.clone(),
            },
        ))
    })
    .unwrap_err();

    let Error::RuntimeStateSource { op, phase, source } = error else {
        panic!("factory error should retain its exact runtime error kind");
    };
    assert_eq!(op, "test-extension-factory");
    assert_eq!(phase, ErrorPhase::Execution);
    let source = source
        .downcast_ref::<RuntimeConfigError>()
        .expect("factory source should retain RuntimeConfigError");
    assert!(matches!(
        source,
        RuntimeConfigError::MissingEngine { engine_id } if engine_id == &missing_engine
    ));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA device initialization"]
fn eager_extension_factory_receives_exact_cuda_target() {
    use tenferro_gpu::cuda::{cuda_devices, CudaBackend};

    let device = cuda_devices().unwrap().into_iter().next().unwrap();
    let backend = CudaBackend::new(device.id()).unwrap();
    let ctx = EagerRuntime::with_cuda_backend(backend).unwrap();
    let host = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let device_input = ctx
        .with_execution_session(|session| {
            session.upload_host_tensor(TensorRead::from_tensor(&host))
        })
        .unwrap()
        .unwrap();
    let input = EagerTensor::from_tensor_in(device_input, ctx).unwrap();
    let seen = Arc::new(Mutex::new(None));
    let seen_by_factory = Arc::clone(&seen);

    execute_probe(Arc::new(BridgeProbe::one()), &input, move |target| {
        *seen_by_factory.lock().unwrap() = Some(target.clone());
        Ok(BridgeModule::for_engine(target.engine_id))
    })
    .unwrap();

    assert_eq!(
        seen.lock()
            .unwrap()
            .as_ref()
            .map(|target| target.backend_kind),
        Some(EagerExtensionBackendKind::Cuda)
    );
    assert_eq!(
        seen.lock()
            .unwrap()
            .as_ref()
            .map(|target| target.engine_id.as_str()),
        Some("tenferro-ad.cuda.default.v1")
    );
}

#[cfg(feature = "webgpu")]
#[test]
#[ignore = "requires a WebGPU adapter"]
fn eager_extension_factory_receives_exact_webgpu_target() {
    use tenferro_gpu::webgpu::{webgpu_runtime_engine_id, WebGpuBackend};

    let Ok(backend) = WebGpuBackend::new_default() else {
        return;
    };
    let ctx = EagerRuntime::with_webgpu_backend(backend).unwrap();
    let host = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let device_input = ctx
        .with_execution_session(|session| {
            session.upload_host_tensor(TensorRead::from_tensor(&host))
        })
        .unwrap()
        .unwrap();
    let input = EagerTensor::from_tensor_in(device_input, ctx).unwrap();
    let seen = Arc::new(Mutex::new(None));
    let seen_by_factory = Arc::clone(&seen);

    execute_probe(Arc::new(BridgeProbe::one()), &input, move |target| {
        *seen_by_factory.lock().unwrap() = Some(target.clone());
        Ok(BridgeModule::for_engine(target.engine_id))
    })
    .unwrap();

    assert_eq!(
        seen.lock()
            .unwrap()
            .as_ref()
            .map(|target| target.backend_kind),
        Some(EagerExtensionBackendKind::WebGpu)
    );
    assert_eq!(
        seen.lock()
            .unwrap()
            .as_ref()
            .map(|target| target.engine_id.as_str()),
        Some(webgpu_runtime_engine_id().unwrap().as_str())
    );
}
