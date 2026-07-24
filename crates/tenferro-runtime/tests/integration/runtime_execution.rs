use std::error::Error as StdError;
use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{
    CoreCapabilityBundle, DType, DotGeneralPreparation, ElementwiseRuntime, EngineId,
    EngineRegistration, ExecutionContextIdentity, GraphCompiler, HardwareClassId, IndexingRuntime,
    LayoutRuntime, ReductionRuntime, Runtime, RuntimeCacheOwner, RuntimeConfigError, StorageClass,
    TracedTensor,
};
use tenferro_tensor::Tensor;

const CPU_ENGINE_ID: &str = "tenferro-cpu.default.v1";
const CPU_HARDWARE_CLASS_ID: &str = "tenferro-cpu.host.v1";
const CPU_STORAGE_CLASS_ID: &str = "tenferro-cpu.host.v1";

fn cpu_registration(
    backend: &CpuBackend,
    include_execution_bridge: bool,
) -> Result<EngineRegistration, RuntimeConfigError> {
    let backend = Arc::new(backend.clone());
    let elementwise: Arc<dyn ElementwiseRuntime> = backend.clone();
    let reduction: Arc<dyn ReductionRuntime> = backend.clone();
    let indexing: Arc<dyn IndexingRuntime> = backend.clone();
    let dot_general: Arc<dyn DotGeneralPreparation> = backend.clone();
    let layout: Arc<dyn LayoutRuntime> = backend.clone();
    let cache_owner: Arc<dyn RuntimeCacheOwner> = backend.clone();

    let mut capabilities = CoreCapabilityBundle::builder();
    capabilities
        .elementwise(elementwise)
        .reduction(reduction)
        .indexing(indexing)
        .dot_general(dot_general)
        .layout(layout);

    let storage = StorageClass::new(CPU_STORAGE_CLASS_ID).map_err(RuntimeConfigError::from)?;
    EngineRegistration::new(
        EngineId::new(CPU_ENGINE_ID).map_err(RuntimeConfigError::from)?,
        ExecutionContextIdentity::of::<CpuBackend>(),
        HardwareClassId::new(CPU_HARDWARE_CLASS_ID).map_err(RuntimeConfigError::from)?,
        Arc::from(vec![storage.clone()]),
        storage,
        capabilities.build(),
    )
    .map(|registration| {
        let registration = registration.with_cache_owner(cache_owner);
        if include_execution_bridge {
            registration.with_tensor_backend_executor(backend.as_ref().clone())
        } else {
            registration
        }
    })
}

fn runtime_with_cpu(backend: &CpuBackend) -> Result<Runtime, RuntimeConfigError> {
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration(backend, true)?)?;
    builder.build()
}

#[test]
fn runtime_run_compiled_uses_prepared_cache_on_second_call() -> Result<(), Box<dyn StdError>> {
    let backend = CpuBackend::new();
    let runtime = runtime_with_cpu(&backend)?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = (&x + &x)?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;

    let first = runtime.run_compiled(&program, &[&input])?;
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].as_slice::<f64>()?, &[2.0, 4.0]);
    let after_first = runtime.cache_stats()?.prepared_plans;

    let second = runtime.run_compiled(&program, &[&input])?;
    assert_eq!(second[0].as_slice::<f64>()?, &[2.0, 4.0]);
    let after_second = runtime.cache_stats()?.prepared_plans;

    assert!(after_first.misses >= 1);
    assert_eq!(after_second.misses, after_first.misses);
    assert!(after_second.hits > after_first.hits);
    assert_eq!(after_second.entries, after_first.entries);

    Ok(())
}

#[test]
fn runtime_run_compiled_reports_missing_execution_bridge() -> Result<(), Box<dyn StdError>> {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder.register_engine(cpu_registration(&backend, false)?)?;
    let runtime = builder.build()?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = x.neg()?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;

    let error = runtime.run_compiled(&program, &[&input]).unwrap_err();

    assert!(error.to_string().contains("execution bridge"));
    Ok(())
}

#[test]
fn runtime_run_compiled_reprepares_after_engine_reconfiguration() -> Result<(), Box<dyn StdError>> {
    let backend = CpuBackend::new();
    let runtime = runtime_with_cpu(&backend)?;
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1)?;
    let y = x.neg()?;
    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])?;
    let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;

    let first = runtime.run_compiled(&program, &[&input])?;
    assert_eq!(first[0].as_slice::<f64>()?, &[-1.0, -2.0]);
    let after_first = runtime.cache_stats()?.prepared_plans;

    runtime.reconfigure(|edit| {
        edit.replace_engine(cpu_registration(&CpuBackend::new(), true)?)?;
        Ok(())
    })?;

    let second = runtime.run_compiled(&program, &[&input])?;
    assert_eq!(second[0].as_slice::<f64>()?, &[-1.0, -2.0]);
    let after_second = runtime.cache_stats()?.prepared_plans;

    assert!(after_second.misses > after_first.misses);
    Ok(())
}
