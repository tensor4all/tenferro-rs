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
    EngineId, Error, ErrorPhase, ExtensionModule, ExtensionModuleError, ExtensionModuleId,
    ExtensionModuleRegistrar, Runtime,
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
struct BridgeModule {
    module_id: ExtensionModuleId,
}

impl BridgeModule {
    fn new() -> Arc<dyn ExtensionModule> {
        Arc::new(Self {
            module_id: ExtensionModuleId::new("tenferro-tests.eager-extension-bridge.module")
                .unwrap(),
        })
    }
}

impl ExtensionModule for BridgeModule {
    fn module_id(&self) -> &ExtensionModuleId {
        &self.module_id
    }

    fn configure(
        &self,
        _registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> Result<(), ExtensionModuleError> {
        Ok(())
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
        Ok(BridgeModule::new())
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
            Ok(BridgeModule::new())
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
    let source = std::error::Error::source(&error);
    assert!(source
        .as_deref()
        .is_some_and(|source| source.to_string().contains("is not registered")));
}

#[test]
fn eager_extension_factory_errors_are_returned_without_entering_backend_session() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let input = input(&ctx);
    let error = execute_probe(Arc::new(BridgeProbe::one()), &input, |_target| {
        Err(Error::runtime_state(
            "test-extension-factory",
            ErrorPhase::Execution,
            "missing extension module",
        ))
    })
    .unwrap_err();

    assert!(error.to_string().contains("missing extension module"));
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
        Ok(BridgeModule::new())
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
        Ok(BridgeModule::new())
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
