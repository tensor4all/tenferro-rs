//! Runtime-owned compiled-graph execution boundary.
//!
//! This module owns the private execution bridge used by
//! `Runtime::run_compiled*`.

use std::error::Error as StdError;
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, ThreadId};

use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_tensor::{Tensor, TensorBackend, TensorRead, TensorValue};

use crate::error::ErrorPhase;
use crate::exec::{ExecProgram, ExecSlot, ExtensionExecutionDispatch};
use crate::extension_cache::{ExtensionCacheSelector, ExtensionCacheStore};
use crate::graph::CompiledGraph;
use crate::runtime::{
    CacheOwnerError, CacheStats, InputSignature, PrepareError, PrepareOptions,
    PreparedOperationHandle, Runtime, RuntimeCacheOwner,
};
use crate::{Error, Result};

#[allow(
    dead_code,
    reason = "Phase 5 runtime execution task adds erased dispatch methods"
)]
pub(super) trait ErasedTensorBackendExecutor: fmt::Debug + Send + Sync {
    fn backend_type_name(&self) -> &'static str;
    fn extension_cache_stats(&self) -> std::result::Result<CacheStats, CacheOwnerError>;
    fn clear_extension_caches(&self) -> std::result::Result<(), CacheOwnerError>;
    fn execute(
        &self,
        program: &ExecProgram,
        operations: &[PreparedOperationHandle],
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>>;
    fn execute_values(
        &self,
        program: &ExecProgram,
        operations: &[PreparedOperationHandle],
        inputs: Vec<Tensor>,
    ) -> Result<Vec<TensorValue>>;
}

pub(super) fn erased_tensor_backend_executor<B>(backend: B) -> Arc<dyn ErasedTensorBackendExecutor>
where
    B: TensorBackend + Send + Sync + 'static,
{
    Arc::new(TensorBackendExecutor::<B>::new(backend))
}

pub(super) fn extension_cache_owner(
    executor: Arc<dyn ErasedTensorBackendExecutor>,
) -> Arc<dyn RuntimeCacheOwner> {
    Arc::new(TensorBackendExtensionCacheOwner { executor })
}

#[derive(Debug)]
struct TensorBackendExtensionCacheOwner {
    executor: Arc<dyn ErasedTensorBackendExecutor>,
}

impl RuntimeCacheOwner for TensorBackendExtensionCacheOwner {
    fn cache_stats(&self) -> std::result::Result<CacheStats, CacheOwnerError> {
        self.executor.extension_cache_stats()
    }

    fn clear_caches(&self) -> std::result::Result<(), CacheOwnerError> {
        self.executor.clear_extension_caches()
    }
}

#[allow(
    dead_code,
    reason = "Phase 5 runtime execution task consumes backend execution state"
)]
struct TensorBackendExecutorState<B: TensorBackend + 'static> {
    backend: B,
    backend_cache: B::RuntimeCache,
    extension_caches: ExtensionCacheStore,
    slot_workspace: Vec<Option<ExecSlot>>,
}

struct TensorBackendExecutor<B: TensorBackend + 'static> {
    state: Mutex<TensorBackendExecutorSlot<B>>,
    available: Condvar,
}

struct TensorBackendExecutorSlot<B: TensorBackend + 'static> {
    state: Option<TensorBackendExecutorState<B>>,
    active_thread: Option<ThreadId>,
    execution_poisoned: bool,
}

impl<B> TensorBackendExecutor<B>
where
    B: TensorBackend + Send + Sync + 'static,
{
    fn new(backend: B) -> Self {
        Self {
            state: Mutex::new(TensorBackendExecutorSlot {
                state: Some(TensorBackendExecutorState {
                    backend,
                    backend_cache: B::RuntimeCache::default(),
                    extension_caches: ExtensionCacheStore::new(),
                    slot_workspace: Vec::new(),
                }),
                active_thread: None,
                execution_poisoned: false,
            }),
            available: Condvar::new(),
        }
    }

    fn lease_state(&self, caller: &'static str) -> Result<TensorBackendExecutorLease<'_, B>> {
        let current = thread::current().id();
        let mut slot = self.lock_slot(caller)?;
        loop {
            if slot.execution_poisoned {
                return Err(Error::runtime_state(
                    caller,
                    ErrorPhase::Execution,
                    "tensor backend executor state poisoned by panic during prior execution",
                ));
            }
            if let Some(state) = slot.state.take() {
                slot.active_thread = Some(current);
                return Ok(TensorBackendExecutorLease {
                    executor: self,
                    state: Some(state),
                });
            }
            if slot.active_thread == Some(current) {
                return Err(Error::runtime_state(
                    caller,
                    ErrorPhase::Execution,
                    "reentrant tensor backend executor call would deadlock",
                ));
            }
            slot = self.available.wait(slot).map_err(|_| {
                Error::runtime_state(
                    caller,
                    ErrorPhase::Execution,
                    "tensor backend executor state lock poisoned while waiting",
                )
            })?;
        }
    }

    fn lock_slot(
        &self,
        caller: &'static str,
    ) -> Result<MutexGuard<'_, TensorBackendExecutorSlot<B>>> {
        self.state.lock().map_err(|_| {
            Error::runtime_state(
                caller,
                ErrorPhase::Execution,
                "tensor backend executor state lock poisoned",
            )
        })
    }
}

struct TensorBackendExecutorLease<'a, B: TensorBackend + 'static> {
    executor: &'a TensorBackendExecutor<B>,
    state: Option<TensorBackendExecutorState<B>>,
}

impl<B: TensorBackend + 'static> TensorBackendExecutorLease<'_, B> {
    fn state_mut(&mut self) -> &mut TensorBackendExecutorState<B> {
        self.state
            .as_mut()
            .expect("executor state lease always owns state before drop")
    }
}

impl<B: TensorBackend + 'static> Drop for TensorBackendExecutorLease<'_, B> {
    fn drop(&mut self) {
        let Some(state) = self.state.take() else {
            return;
        };
        let panicking = thread::panicking();
        let mut wake_all_waiters = panicking;
        match self.executor.state.lock() {
            Ok(mut slot) => {
                slot.state = Some(state);
                slot.active_thread = None;
                slot.execution_poisoned |= panicking;
            }
            Err(poisoned_lock) => {
                let mut slot = poisoned_lock.into_inner();
                slot.state = Some(state);
                slot.active_thread = None;
                slot.execution_poisoned = true;
                wake_all_waiters = true;
            }
        }
        if wake_all_waiters {
            self.executor.available.notify_all();
        } else {
            self.executor.available.notify_one();
        }
    }
}

impl<B: TensorBackend + 'static> fmt::Debug for TensorBackendExecutor<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorBackendExecutor")
            .field("backend_type", &std::any::type_name::<B>())
            .field("state_lock_poisoned", &self.state.is_poisoned())
            .finish_non_exhaustive()
    }
}

impl<B> ErasedTensorBackendExecutor for TensorBackendExecutor<B>
where
    B: TensorBackend + Send + Sync + 'static,
{
    fn backend_type_name(&self) -> &'static str {
        std::any::type_name::<B>()
    }

    fn extension_cache_stats(&self) -> std::result::Result<CacheStats, CacheOwnerError> {
        let lease = self
            .lease_state("Runtime::extension_cache_stats")
            .map_err(cache_owner_error)?;
        let state = lease
            .state
            .as_ref()
            .expect("executor state lease always owns state before drop");
        Ok(cache_stats_from_tensor_stats(
            state.extension_caches.stats(ExtensionCacheSelector::All),
        ))
    }

    fn clear_extension_caches(&self) -> std::result::Result<(), CacheOwnerError> {
        let mut lease = self
            .lease_state("Runtime::clear_extension_caches")
            .map_err(cache_owner_error)?;
        lease.state_mut().extension_caches.clear();
        Ok(())
    }

    fn execute(
        &self,
        program: &ExecProgram,
        operations: &[PreparedOperationHandle],
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>> {
        validate_exec_input_count(program, inputs.len())?;
        let mut lease = self.lease_state("Runtime::run_compiled")?;
        let TensorBackendExecutorState {
            backend,
            backend_cache,
            extension_caches,
            slot_workspace,
        } = lease.state_mut();
        let mut extension_dispatch = ExtensionExecutionDispatch {
            operations,
            caches: extension_caches,
        };
        crate::segment::eval_exec_segmented_with_cache_and_workspace(
            backend,
            program,
            inputs,
            slot_workspace,
            backend_cache,
            Some(&mut extension_dispatch),
        )
    }

    fn execute_values(
        &self,
        program: &ExecProgram,
        operations: &[PreparedOperationHandle],
        inputs: Vec<Tensor>,
    ) -> Result<Vec<TensorValue>> {
        validate_exec_input_count(program, inputs.len())?;
        let inputs = inputs.into_iter().map(ExecSlot::Owned).collect();
        let mut lease = self.lease_state("Runtime::run_compiled_values")?;
        let TensorBackendExecutorState {
            backend,
            backend_cache,
            extension_caches,
            slot_workspace,
        } = lease.state_mut();
        let mut extension_dispatch = ExtensionExecutionDispatch {
            operations,
            caches: extension_caches,
        };
        crate::segment::eval_exec_segmented_slot_values_with_cache_and_workspace(
            backend,
            program,
            inputs,
            slot_workspace,
            backend_cache,
            Some(&mut extension_dispatch),
        )
    }
}

fn cache_owner_error(error: Error) -> CacheOwnerError {
    CacheOwnerError::new(Arc::new(error))
}

fn cache_stats_from_tensor_stats(stats: tenferro_tensor::CacheStats) -> CacheStats {
    CacheStats {
        entries: stats.entries,
        retained_bytes: stats.retained_bytes,
        hits: stats.hits,
        misses: stats.misses,
        evictions: stats.evictions,
        clears: stats.clears,
    }
}

pub(super) fn run_compiled(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    let inputs = resolve_input_tensors(program, inputs)?;
    let signature = input_signature(&inputs)?;
    let prepared = prepare(runtime, program, &signature)?;
    let executor = execution_engine(runtime, prepared.root().as_ref())?;
    executor.execute(prepared.root().staging(), prepared.operations(), inputs)
}

pub(super) fn run_compiled_values(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> Result<Vec<TensorValue>> {
    let inputs = resolve_input_tensors(program, inputs)?;
    let signature = input_signature(&inputs)?;
    let prepared = prepare(runtime, program, &signature)?;
    let executor = execution_engine(runtime, prepared.root().as_ref())?;
    executor.execute_values(prepared.root().staging(), prepared.operations(), inputs)
}

fn prepare(
    runtime: &Runtime,
    program: &CompiledGraph,
    signature: &InputSignature,
) -> Result<Arc<super::preparation::PreparedProgram>> {
    runtime
        .prepare_compiled_for(program, signature, &PrepareOptions::new())
        .map_err(prepare_error)
}

fn execution_engine(
    runtime: &Runtime,
    root: &super::preparation::PreparedProgramRoot,
) -> Result<Arc<dyn ErasedTensorBackendExecutor>> {
    let snapshot = runtime.snapshot().map_err(|source| {
        Error::runtime_state_source("Runtime::run_compiled", ErrorPhase::Execution, source)
    })?;
    if snapshot.epoch() != root.epoch() {
        return Err(Error::runtime_state(
            "Runtime::run_compiled",
            ErrorPhase::Execution,
            format!(
                "prepared epoch {:?} does not match current epoch {:?}",
                root.epoch(),
                snapshot.epoch()
            ),
        ));
    }
    let engine = snapshot.engine(root.engine_id()).ok_or_else(|| {
        Error::runtime_state(
            "Runtime::run_compiled",
            ErrorPhase::Execution,
            format!(
                "prepared engine {:?} is no longer registered",
                root.engine_id()
            ),
        )
    })?;
    engine.execution_engine().cloned().ok_or_else(|| {
        Error::runtime_state(
            "Runtime::run_compiled",
            ErrorPhase::Execution,
            format!(
                "engine {:?} has no runtime execution bridge",
                root.engine_id()
            ),
        )
    })
}

fn input_signature(inputs: &[Tensor]) -> Result<InputSignature> {
    let reads: Vec<_> = inputs.iter().map(TensorRead::from_tensor).collect();
    InputSignature::from_reads(&reads).map_err(|source| prepare_error(Arc::new(source)))
}

fn resolve_input_tensors(program: &CompiledGraph, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    let resolved = if inputs.is_empty() {
        semantic_default_inputs(program)?
            .into_iter()
            .cloned()
            .collect::<Vec<_>>()
    } else {
        inputs.iter().map(|tensor| (*tensor).clone()).collect()
    };
    validate_ordered_input_metadata(program, &resolved)?;
    Ok(resolved)
}

fn semantic_default_inputs(program: &CompiledGraph) -> Result<Vec<&Tensor>> {
    program
        .program()
        .inputs()
        .iter()
        .enumerate()
        .map(|(input_index, value)| {
            program
                .bindings()
                .tensor_ref_for_input(*value)
                .map(AsRef::as_ref)
                .ok_or_else(|| Error::UnboundPlaceholder {
                    input_key: format!("semantic input {input_index}"),
                })
        })
        .collect()
}

fn validate_ordered_input_metadata(program: &CompiledGraph, inputs: &[Tensor]) -> Result<()> {
    let expected = program.input_count();
    if inputs.len() != expected {
        return Err(Error::GraphInputCountMismatch {
            expected,
            actual: inputs.len(),
        });
    }
    let input_shapes: Vec<_> = inputs.iter().map(Tensor::shape).collect();
    for (input_value, actual) in program.program().inputs().iter().zip(inputs) {
        let metadata = program
            .program()
            .value_metadata(*input_value)
            .map_err(|source| {
                Error::runtime_state_source("Runtime::run_compiled", ErrorPhase::Execution, source)
            })?;
        if metadata.dtype() != actual.dtype() {
            return Err(Error::PlaceholderDtypeMismatch {
                expected: metadata.dtype(),
                actual: actual.dtype(),
            });
        }
        if metadata.shape().len() != actual.shape().len() {
            return Err(Error::PlaceholderRankMismatch {
                expected: metadata.shape().len(),
                actual: actual.shape().len(),
            });
        }
        let mut expected_shape = actual.shape().to_vec();
        let mut exact_mismatch = false;
        for (axis, (extent, actual_size)) in metadata.shape().iter().zip(actual.shape()).enumerate()
        {
            match extent {
                ShapeExtent::Exact(expression) => {
                    let expected = expression.eval(&input_shapes).map_err(|source| {
                        Error::runtime_state_source(
                            "Runtime::run_compiled",
                            ErrorPhase::Execution,
                            source,
                        )
                    })?;
                    expected_shape[axis] = expected;
                    exact_mismatch |= expected != *actual_size;
                }
                ShapeExtent::UpperBound(expression) => {
                    let bound = expression.eval(&input_shapes).map_err(|source| {
                        Error::runtime_state_source(
                            "Runtime::run_compiled",
                            ErrorPhase::Execution,
                            source,
                        )
                    })?;
                    if *actual_size > bound {
                        return Err(Error::PlaceholderShapeBoundExceeded {
                            axis,
                            bound,
                            actual: *actual_size,
                        });
                    }
                }
                ShapeExtent::Unknown => {}
            }
        }
        if exact_mismatch {
            return Err(Error::PlaceholderShapeMismatch {
                expected: expected_shape,
                actual: actual.shape().to_vec(),
            });
        }
    }
    Ok(())
}

fn validate_exec_input_count(program: &ExecProgram, actual: usize) -> Result<()> {
    let expected = program.input_slots.len();
    if actual != expected {
        return Err(Error::runtime_state(
            "Runtime::run_compiled",
            ErrorPhase::Execution,
            format!("expected {expected} inputs for execution program, got {actual}"),
        ));
    }
    Ok(())
}

fn prepare_error(source: Arc<PrepareError>) -> Error {
    Error::runtime_state_source(
        "Runtime::run_compiled",
        ErrorPhase::Execution,
        SharedPrepareError(source),
    )
}

#[derive(Clone, Debug)]
struct SharedPrepareError(Arc<PrepareError>);

impl fmt::Display for SharedPrepareError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

impl StdError for SharedPrepareError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(self.0.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::hash::Hasher;
    use std::num::NonZeroU64;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, TryLockError, Weak};

    use tenferro_cpu::CpuBackend;
    use tenferro_ops::dim_expr::DimExpr;
    use tenferro_ops::ext_op::ExtensionOp;
    use tenferro_ops::SymDim;
    use tenferro_tensor::{
        BackendSessionHost, DType, Tensor, TensorBackend, TensorRead, TypedTensor,
    };

    use crate::exec::{ExecInstruction, ExecOp, ExecProgram};
    use crate::runtime::{
        EngineId, ErasedExecutionContext, ExecutionContextIdentity, HardwareClassId,
        InputSignature, PreparedOperation, PreparedOperationBinding, PreparedOperationHandle,
        RegistrationIdentity, RuntimeEpoch, RuntimeId, SpecializationProjection,
        SpecializationRequirements,
    };
    use crate::{Error, ErrorPhase, ExtensionCacheStore, Result};

    use super::{ErasedTensorBackendExecutor, TensorBackendExecutor};

    const LOCK_PROBE_FAMILY: &str = "runtime.lock-probe.v1";
    const REENTRANT_PROBE_FAMILY: &str = "runtime.reentrant-probe.v1";

    #[derive(Clone, Debug)]
    struct LockProbeOp;

    impl ExtensionOp for LockProbeOp {
        fn family_id(&self) -> &'static str {
            LOCK_PROBE_FAMILY
        }

        fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

        fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
            other.as_any().downcast_ref::<Self>().is_some()
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

    #[derive(Clone, Debug)]
    struct ReentrantProbeOp;

    impl ExtensionOp for ReentrantProbeOp {
        fn family_id(&self) -> &'static str {
            REENTRANT_PROBE_FAMILY
        }

        fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

        fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
            other.as_any().downcast_ref::<Self>().is_some()
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

    #[derive(Debug)]
    struct LockProbePreparedOperation {
        binding: PreparedOperationBinding,
        specialization: SpecializationProjection,
        executor: Weak<TensorBackendExecutor<CpuBackend>>,
        observed_unlocked_state: Arc<AtomicBool>,
    }

    impl PreparedOperation for LockProbePreparedOperation {
        fn binding(&self) -> &PreparedOperationBinding {
            &self.binding
        }

        fn specialization(&self) -> &SpecializationProjection {
            &self.specialization
        }

        fn retained_bytes(&self) -> usize {
            0
        }

        fn execute(
            &self,
            context: &mut ErasedExecutionContext<'_>,
            _extension_caches: &mut ExtensionCacheStore,
            inputs: &[TensorRead<'_>],
        ) -> Result<Vec<Tensor>> {
            let executor = self.executor.upgrade().expect("executor still alive");
            let unlocked = match executor.state.try_lock() {
                Ok(_guard) => true,
                Err(TryLockError::WouldBlock) => false,
                Err(TryLockError::Poisoned(_)) => false,
            };
            self.observed_unlocked_state
                .store(unlocked, Ordering::SeqCst);
            let backend = context
                .downcast_mut::<CpuBackend>(self.binding.context_identity())
                .map_err(|source| {
                    Error::runtime_state_source("lock_probe", ErrorPhase::Execution, source)
                })?;
            Ok(vec![backend.with_backend_session(|exec| {
                exec.to_contiguous_read(inputs[0].clone())
            })?])
        }
    }

    #[derive(Debug)]
    struct ReentrantProbePreparedOperation {
        binding: PreparedOperationBinding,
        specialization: SpecializationProjection,
        executor: Weak<TensorBackendExecutor<CpuBackend>>,
        observed_reentrant_error: Arc<AtomicBool>,
    }

    impl ReentrantProbePreparedOperation {
        fn probe_reentrant_call(&self, input: Tensor) {
            let executor = self.executor.upgrade().expect("executor still alive");
            let nested = ErasedTensorBackendExecutor::execute(
                executor.as_ref(),
                &passthrough_program(),
                &[],
                vec![input],
            );
            let observed = nested.is_err_and(|error| {
                error
                    .to_string()
                    .contains("reentrant tensor backend executor call would deadlock")
            });
            self.observed_reentrant_error
                .store(observed, Ordering::SeqCst);
        }
    }

    impl PreparedOperation for ReentrantProbePreparedOperation {
        fn binding(&self) -> &PreparedOperationBinding {
            &self.binding
        }

        fn specialization(&self) -> &SpecializationProjection {
            &self.specialization
        }

        fn retained_bytes(&self) -> usize {
            0
        }

        fn execute(
            &self,
            context: &mut ErasedExecutionContext<'_>,
            _extension_caches: &mut ExtensionCacheStore,
            inputs: &[TensorRead<'_>],
        ) -> Result<Vec<Tensor>> {
            let backend = context
                .downcast_mut::<CpuBackend>(self.binding.context_identity())
                .map_err(|source| {
                    Error::runtime_state_source("reentrant_probe", ErrorPhase::Execution, source)
                })?;
            let materialized =
                backend.with_backend_session(|exec| exec.to_contiguous_read(inputs[0].clone()))?;
            self.probe_reentrant_call(materialized.clone());
            Ok(vec![materialized])
        }
    }

    fn lock_probe_program() -> ExecProgram {
        ExecProgram {
            instructions: vec![ExecInstruction {
                op: ExecOp::Extension(Arc::new(LockProbeOp)),
                semantic_operation_index: Some(0),
                input_slots: vec![0],
                output_slots: vec![1],
                dtype: DType::F64,
                output_shapes: vec![vec![DimExpr::InputDim {
                    input_idx: 0,
                    axis: 0,
                }]]
                .into(),
                output_extents: vec![vec![]].into(),
                last_use: vec![false],
            }],
            input_slots: vec![0],
            output_slots: vec![1],
            n_slots: 2,
            shape_guards: vec![],
        }
    }

    fn reentrant_probe_program() -> ExecProgram {
        ExecProgram {
            instructions: vec![ExecInstruction {
                op: ExecOp::Extension(Arc::new(ReentrantProbeOp)),
                semantic_operation_index: Some(0),
                input_slots: vec![0],
                output_slots: vec![1],
                dtype: DType::F64,
                output_shapes: vec![vec![DimExpr::InputDim {
                    input_idx: 0,
                    axis: 0,
                }]]
                .into(),
                output_extents: vec![vec![]].into(),
                last_use: vec![false],
            }],
            input_slots: vec![0],
            output_slots: vec![1],
            n_slots: 2,
            shape_guards: vec![],
        }
    }

    fn passthrough_program() -> ExecProgram {
        ExecProgram {
            instructions: vec![],
            input_slots: vec![0],
            output_slots: vec![0],
            n_slots: 1,
            shape_guards: vec![],
        }
    }

    fn f64_zeros(shape: Vec<usize>) -> Tensor {
        Tensor::F64(TypedTensor::zeros(shape).unwrap())
    }

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap_or(NonZeroU64::MIN)
    }

    fn probe_binding() -> PreparedOperationBinding {
        PreparedOperationBinding::new(
            RuntimeId::from_nonzero(nz(1)),
            RuntimeEpoch::from_nonzero(nz(2)),
            EngineId::new("tenferro.cpu").unwrap(),
            RegistrationIdentity::new(nz(3), nz(4)),
            ExecutionContextIdentity::of::<CpuBackend>(),
            HardwareClassId::new("tenferro.cpu.host").unwrap(),
        )
    }

    fn probe_specialization() -> SpecializationProjection {
        SpecializationRequirements::polymorphic(0)
            .project(&InputSignature::new(Vec::new()))
            .unwrap()
    }

    #[test]
    fn tensor_backend_executor_releases_state_lock_during_extension_execution() {
        let executor = Arc::new(TensorBackendExecutor::<CpuBackend>::new(CpuBackend::new()));
        let observed_unlocked_state = Arc::new(AtomicBool::new(false));
        let operations: Vec<PreparedOperationHandle> = vec![Arc::new(LockProbePreparedOperation {
            binding: probe_binding(),
            specialization: probe_specialization(),
            executor: Arc::downgrade(&executor),
            observed_unlocked_state: Arc::clone(&observed_unlocked_state),
        })];

        let output = ErasedTensorBackendExecutor::execute(
            executor.as_ref(),
            &lock_probe_program(),
            &operations,
            vec![f64_zeros(vec![2])],
        )
        .expect("extension executes");

        assert_eq!(output[0].shape(), &[2]);
        assert!(
            observed_unlocked_state.load(Ordering::SeqCst),
            "executor state lock must not be held while extension runtime callbacks execute"
        );
    }

    #[test]
    fn tensor_backend_executor_reentrant_call_returns_error_instead_of_deadlocking() {
        let executor = Arc::new(TensorBackendExecutor::<CpuBackend>::new(CpuBackend::new()));
        let observed_reentrant_error = Arc::new(AtomicBool::new(false));
        let operations: Vec<PreparedOperationHandle> =
            vec![Arc::new(ReentrantProbePreparedOperation {
                binding: probe_binding(),
                specialization: probe_specialization(),
                executor: Arc::downgrade(&executor),
                observed_reentrant_error: Arc::clone(&observed_reentrant_error),
            })];

        let output = ErasedTensorBackendExecutor::execute(
            executor.as_ref(),
            &reentrant_probe_program(),
            &operations,
            vec![f64_zeros(vec![2])],
        )
        .expect("outer extension executes");

        assert_eq!(output[0].shape(), &[2]);
        assert!(
            observed_reentrant_error.load(Ordering::SeqCst),
            "same-thread reentrant executor call must fail immediately instead of deadlocking"
        );
    }

    #[test]
    fn tensor_backend_executor_bridge_does_not_require_clone_source_contract() {
        fn factory_accepts_backend_without_clone_bound<B>()
        where
            B: TensorBackend + Send + Sync + 'static,
        {
            let _factory: fn(B) -> Arc<dyn ErasedTensorBackendExecutor> =
                super::erased_tensor_backend_executor::<B>;
        }

        fn registration_accepts_backend_without_clone_bound<B>()
        where
            B: TensorBackend + Send + Sync + 'static,
        {
            let _method: fn(
                crate::runtime::EngineRegistration,
                B,
            ) -> crate::runtime::EngineRegistration =
                crate::runtime::EngineRegistration::with_tensor_backend_executor::<B>;
        }

        factory_accepts_backend_without_clone_bound::<CpuBackend>();
        registration_accepts_backend_without_clone_bound::<CpuBackend>();
    }
}
