//! Runtime-owned compiled-graph execution boundary.
//!
//! This module owns the private execution bridge used by
//! `Runtime::run_compiled*`.

use std::error::Error as StdError;
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, ThreadId};

use smallvec::SmallVec;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_tensor::{Tensor, TensorBackend, TensorRead, TensorValue};

use crate::error::ErrorPhase;
use crate::exec::{
    DispatchMode, ExecInstruction, ExecProgram, ExecSlot, ExtensionExecutionDispatch,
};
use crate::extension_cache::{ExtensionCacheSelector, ExtensionCacheStore};
use crate::graph::CompiledGraph;
use crate::runtime::schedule::{ScheduledGraph, ScheduledNode};
use crate::runtime::{
    CacheOwnerError, CacheStats, InputSignature, PrepareError, PrepareOptions,
    PreparedOperationHandle, Runtime, RuntimeCacheOwner,
};
use crate::{Error, Result};

type RuntimeInputRefs<'a> = SmallVec<[&'a Tensor; 8]>;
type RuntimeInputReads<'a> = SmallVec<[TensorRead<'a>; 8]>;
type RuntimeInputShapes<'a> = SmallVec<[&'a [usize]; 8]>;
type RuntimeShapeScratch = SmallVec<[usize; 8]>;

#[derive(Clone, Copy, Debug)]
pub(super) enum RuntimeOutputMode {
    Tensor,
    Value,
}

/// Runtime-prepared compiled graph execution handle.
///
/// This handle keeps the prepared execution staging and selected engine bridge
/// outside steady-state execution. It is tied to the runtime epoch that created
/// it and becomes stale after runtime reconfiguration.
#[derive(Clone)]
pub struct PreparedCompiledGraph {
    runtime_id: super::RuntimeId,
    epoch: super::RuntimeEpoch,
    program: CompiledGraph,
    prepared: Arc<super::preparation::PreparedProgram>,
    execution: Arc<PreparedExecutionEngines>,
}

impl fmt::Debug for PreparedCompiledGraph {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedCompiledGraph")
            .field("runtime_id", &self.runtime_id)
            .field("epoch", &self.epoch)
            .field("program", &self.program)
            .finish_non_exhaustive()
    }
}

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
    fn execute_tensor_refs(
        &self,
        program: &ExecProgram,
        operations: &[PreparedOperationHandle],
        inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>>;
    fn execute_values(
        &self,
        program: &ExecProgram,
        operations: &[PreparedOperationHandle],
        inputs: Vec<Tensor>,
    ) -> Result<Vec<TensorValue>>;
    fn execute_value_refs(
        &self,
        program: &ExecProgram,
        operations: &[PreparedOperationHandle],
        inputs: &[&Tensor],
    ) -> Result<Vec<TensorValue>>;
    fn execute_slot_instruction<'input>(
        &self,
        instruction_index: usize,
        instruction: &ExecInstruction,
        operations: &[PreparedOperationHandle],
        slots: &mut [Option<ExecSlot<'input>>],
        output_mode: RuntimeOutputMode,
        terminal_slots: &[bool],
    ) -> Result<()>;
    fn collect_slot_outputs<'input>(
        &self,
        program: &ExecProgram,
        slots: &mut [Option<ExecSlot<'input>>],
    ) -> Result<Vec<Tensor>>;
    fn collect_slot_output_values<'input>(
        &self,
        program: &ExecProgram,
        slots: &mut [Option<ExecSlot<'input>>],
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
    slot_workspace: Vec<Option<ExecSlot<'static>>>,
    borrowed_slot_workspace_capacity: usize,
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
                    borrowed_slot_workspace_capacity: 0,
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
            borrowed_slot_workspace_capacity: _,
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

    fn execute_tensor_refs(
        &self,
        program: &ExecProgram,
        operations: &[PreparedOperationHandle],
        inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>> {
        validate_exec_input_count(program, inputs.len())?;
        let inputs = inputs
            .iter()
            .map(|tensor| ExecSlot::Read(TensorRead::from_tensor(tensor)))
            .collect();
        let mut lease = self.lease_state("Runtime::run_compiled")?;
        let TensorBackendExecutorState {
            backend,
            backend_cache,
            extension_caches,
            borrowed_slot_workspace_capacity,
            ..
        } = lease.state_mut();
        let mut extension_dispatch = ExtensionExecutionDispatch {
            operations,
            caches: extension_caches,
        };
        let mut slot_workspace = Vec::with_capacity(*borrowed_slot_workspace_capacity);
        let result = crate::segment::eval_exec_segmented_slots_with_cache_and_workspace(
            backend,
            program,
            inputs,
            &mut slot_workspace,
            backend_cache,
            Some(&mut extension_dispatch),
        );
        *borrowed_slot_workspace_capacity = slot_workspace.capacity();
        result
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
            borrowed_slot_workspace_capacity: _,
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

    fn execute_value_refs(
        &self,
        program: &ExecProgram,
        operations: &[PreparedOperationHandle],
        inputs: &[&Tensor],
    ) -> Result<Vec<TensorValue>> {
        validate_exec_input_count(program, inputs.len())?;
        let inputs = inputs
            .iter()
            .map(|tensor| ExecSlot::Read(TensorRead::from_tensor(tensor)))
            .collect();
        let mut lease = self.lease_state("Runtime::run_compiled_values")?;
        let TensorBackendExecutorState {
            backend,
            backend_cache,
            extension_caches,
            borrowed_slot_workspace_capacity,
            ..
        } = lease.state_mut();
        let mut extension_dispatch = ExtensionExecutionDispatch {
            operations,
            caches: extension_caches,
        };
        let mut slot_workspace = Vec::with_capacity(*borrowed_slot_workspace_capacity);
        let result = crate::segment::eval_exec_segmented_slot_values_with_cache_and_workspace(
            backend,
            program,
            inputs,
            &mut slot_workspace,
            backend_cache,
            Some(&mut extension_dispatch),
        );
        *borrowed_slot_workspace_capacity = slot_workspace.capacity();
        result
    }

    fn execute_slot_instruction<'input>(
        &self,
        instruction_index: usize,
        instruction: &ExecInstruction,
        operations: &[PreparedOperationHandle],
        slots: &mut [Option<ExecSlot<'input>>],
        output_mode: RuntimeOutputMode,
        terminal_slots: &[bool],
    ) -> Result<()> {
        let mut lease = self.lease_state("Runtime::run_compiled scheduled instruction")?;
        let TensorBackendExecutorState {
            backend,
            backend_cache,
            extension_caches,
            ..
        } = lease.state_mut();
        let mut extension_dispatch = ExtensionExecutionDispatch {
            operations,
            caches: extension_caches,
        };

        if matches!(output_mode, RuntimeOutputMode::Value)
            && backend.with_backend_session(|exec| {
                crate::exec::try_execute_terminal_value_instruction(
                    exec,
                    slots,
                    instruction,
                    terminal_slots,
                )
            })?
        {
            // Already handled as a metadata-only TensorValue.
        } else if crate::exec::is_host_instruction(instruction) {
            crate::exec::execute_host_instruction(backend, slots, instruction)?;
        } else if crate::exec::is_ffi_instruction(instruction) {
            crate::exec::execute_ffi_instruction_cached(
                backend,
                backend_cache,
                slots,
                instruction,
                DispatchMode::Unsegmented,
                Some(instruction_index),
                Some(&mut extension_dispatch),
            )?;
        } else {
            let result = backend.with_backend_session(|exec| {
                crate::exec::execute_backend_op(exec, slots, instruction)
            })?;
            slots[instruction.output_slots[0]] = Some(ExecSlot::Owned(result));
        }
        crate::exec::reclaim_last_use_inputs_backend(slots, instruction, backend);
        Ok(())
    }

    fn collect_slot_outputs<'input>(
        &self,
        program: &ExecProgram,
        slots: &mut [Option<ExecSlot<'input>>],
    ) -> Result<Vec<Tensor>> {
        let mut lease = self.lease_state("Runtime::run_compiled collect outputs")?;
        let backend = &mut lease.state_mut().backend;
        backend.with_backend_session(|exec| crate::exec::collect_outputs_from(program, slots, exec))
    }

    fn collect_slot_output_values<'input>(
        &self,
        program: &ExecProgram,
        slots: &mut [Option<ExecSlot<'input>>],
    ) -> Result<Vec<TensorValue>> {
        let mut lease = self.lease_state("Runtime::run_compiled_values collect outputs")?;
        let backend = &mut lease.state_mut().backend;
        backend.with_backend_session(|exec| {
            crate::exec::collect_output_values_from(program, slots, exec)
        })
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

#[derive(Debug)]
struct PreparedExecutionEngines {
    root: Arc<dyn ErasedTensorBackendExecutor>,
    operations: Box<[Arc<dyn ErasedTensorBackendExecutor>]>,
}

pub(super) fn run_compiled(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    let inputs = resolve_input_refs(program, inputs)?;
    let signature = input_signature(&inputs)?;
    let prepared = prepare(runtime, program, &signature)?;
    let execution = execution_engines(runtime, &prepared)?;
    execute_scheduled_tensor_refs(
        &execution,
        prepared.root().staging(),
        prepared.root().schedule(),
        prepared.operations(),
        &inputs,
    )
}

pub(super) fn prepare_compiled(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> Result<PreparedCompiledGraph> {
    let inputs = resolve_input_refs(program, inputs)?;
    let signature = input_signature(&inputs)?;
    let prepared = prepare(runtime, program, &signature)?;
    let execution = Arc::new(execution_engines(runtime, &prepared)?);
    Ok(PreparedCompiledGraph {
        runtime_id: runtime.id(),
        epoch: prepared.root().epoch(),
        program: program.clone(),
        prepared,
        execution,
    })
}

pub(super) fn run_prepared(
    runtime: &Runtime,
    prepared: &PreparedCompiledGraph,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    validate_prepared_runtime(runtime, prepared, "Runtime::run_prepared")?;
    let inputs = resolve_input_refs(&prepared.program, inputs)?;
    execute_scheduled_tensor_refs(
        &prepared.execution,
        prepared.prepared.root().staging(),
        prepared.prepared.root().schedule(),
        prepared.prepared.operations(),
        &inputs,
    )
}

pub(super) fn run_compiled_values(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[&Tensor],
) -> Result<Vec<TensorValue>> {
    let inputs = resolve_input_refs(program, inputs)?;
    let signature = input_signature(&inputs)?;
    let prepared = prepare(runtime, program, &signature)?;
    let execution = execution_engines(runtime, &prepared)?;
    execute_scheduled_value_refs(
        &execution,
        prepared.root().staging(),
        prepared.root().schedule(),
        prepared.operations(),
        &inputs,
    )
}

fn validate_prepared_runtime(
    runtime: &Runtime,
    prepared: &PreparedCompiledGraph,
    caller: &'static str,
) -> Result<()> {
    if runtime.id() != prepared.runtime_id {
        return Err(Error::runtime_state(
            caller,
            ErrorPhase::Execution,
            "prepared compiled graph belongs to a different runtime",
        ));
    }
    let epoch = runtime
        .epoch()
        .map_err(|source| Error::runtime_state_source(caller, ErrorPhase::Execution, source))?;
    if epoch != prepared.epoch {
        return Err(Error::runtime_state(
            caller,
            ErrorPhase::Execution,
            format!(
                "prepared epoch {:?} does not match current epoch {:?}",
                prepared.epoch, epoch
            ),
        ));
    }
    Ok(())
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

fn execution_engines(
    runtime: &Runtime,
    prepared: &Arc<super::preparation::PreparedProgram>,
) -> Result<PreparedExecutionEngines> {
    let snapshot = runtime.snapshot().map_err(|source| {
        Error::runtime_state_source("Runtime::run_compiled", ErrorPhase::Execution, source)
    })?;
    let root = prepared.root();
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
    let storage_class = root.resolved_placement().storage_class();
    let root_executor = execution_engine_from_snapshot(&snapshot, root.engine_id())?;
    let mut operation_executors = Vec::with_capacity(prepared.operations().len());
    for operation in prepared.operations() {
        let engine_id = operation.binding().engine_id();
        let engine = snapshot.engine(engine_id).ok_or_else(|| {
            Error::runtime_state(
                "Runtime::run_compiled",
                ErrorPhase::Execution,
                format!("prepared operation engine {engine_id:?} is no longer registered"),
            )
        })?;
        if !engine.storage_classes().contains(storage_class) {
            return Err(Error::runtime_state(
                "Runtime::run_compiled",
                ErrorPhase::Execution,
                format!(
                    "prepared operation engine {engine_id:?} uses storage outside {:?}; transfer execution is not implemented yet",
                    storage_class.as_str()
                ),
            ));
        }
        operation_executors.push(execution_engine_from_snapshot(&snapshot, engine_id)?);
    }
    Ok(PreparedExecutionEngines {
        root: root_executor,
        operations: operation_executors.into_boxed_slice(),
    })
}

fn execution_engine_from_snapshot(
    snapshot: &super::RuntimeConfigSnapshot,
    engine_id: &super::EngineId,
) -> Result<Arc<dyn ErasedTensorBackendExecutor>> {
    let engine = snapshot.engine(engine_id).ok_or_else(|| {
        Error::runtime_state(
            "Runtime::run_compiled",
            ErrorPhase::Execution,
            format!("prepared engine {engine_id:?} is no longer registered"),
        )
    })?;
    engine.execution_engine().cloned().ok_or_else(|| {
        Error::runtime_state(
            "Runtime::run_compiled",
            ErrorPhase::Execution,
            format!("engine {engine_id:?} has no runtime execution bridge"),
        )
    })
}

fn execute_scheduled_tensor_refs(
    execution: &PreparedExecutionEngines,
    program: &ExecProgram,
    schedule: &ScheduledGraph,
    operations: &[PreparedOperationHandle],
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    validate_exec_input_count(program, inputs.len())?;
    crate::exec::validate_exec_program(program, "scheduled tensor executor")?;
    schedule.validate_for_runtime()?;
    let inputs = inputs
        .iter()
        .map(|tensor| ExecSlot::Read(TensorRead::from_tensor(tensor)))
        .collect();
    execute_scheduled_slots(
        execution,
        program,
        schedule,
        operations,
        inputs,
        RuntimeOutputMode::Tensor,
    )
    .and_then(|mut slots| execution.root.collect_slot_outputs(program, &mut slots))
}

fn execute_scheduled_value_refs(
    execution: &PreparedExecutionEngines,
    program: &ExecProgram,
    schedule: &ScheduledGraph,
    operations: &[PreparedOperationHandle],
    inputs: &[&Tensor],
) -> Result<Vec<TensorValue>> {
    validate_exec_input_count(program, inputs.len())?;
    crate::exec::validate_exec_program(program, "scheduled value executor")?;
    schedule.validate_for_runtime()?;
    let inputs = inputs
        .iter()
        .map(|tensor| ExecSlot::Read(TensorRead::from_tensor(tensor)))
        .collect();
    execute_scheduled_slots(
        execution,
        program,
        schedule,
        operations,
        inputs,
        RuntimeOutputMode::Value,
    )
    .and_then(|mut slots| {
        execution
            .root
            .collect_slot_output_values(program, &mut slots)
    })
}

fn execute_scheduled_slots<'input>(
    execution: &PreparedExecutionEngines,
    program: &ExecProgram,
    schedule: &ScheduledGraph,
    operations: &[PreparedOperationHandle],
    inputs: Vec<ExecSlot<'input>>,
    output_mode: RuntimeOutputMode,
) -> Result<Vec<Option<ExecSlot<'input>>>> {
    let terminal_slots = if matches!(output_mode, RuntimeOutputMode::Value) {
        crate::exec::terminal_output_slots(program)
    } else {
        Vec::new()
    };
    let mut slots = Vec::new();
    let result = (|| {
        crate::exec::initialize_exec_slots_in(program, inputs, &mut slots)?;
        if schedule.nodes().len() != program.instructions.len() {
            return Err(Error::runtime_state(
                "Runtime::run_compiled",
                ErrorPhase::Execution,
                format!(
                    "scheduled graph has {} nodes for {} execution instructions",
                    schedule.nodes().len(),
                    program.instructions.len()
                ),
            ));
        }
        for (instruction_index, node) in schedule.nodes().iter().enumerate() {
            match node {
                ScheduledNode::Operation(_) => {
                    let instruction = &program.instructions[instruction_index];
                    instruction_executor(execution, instruction)?.execute_slot_instruction(
                        instruction_index,
                        instruction,
                        operations,
                        &mut slots,
                        output_mode,
                        &terminal_slots,
                    )?;
                }
                ScheduledNode::Transfer(_) => {
                    return Err(Error::runtime_state(
                        "Runtime::run_compiled",
                        ErrorPhase::Execution,
                        "transfer node execution is not implemented before U3",
                    ));
                }
                ScheduledNode::Collective(_) => {
                    return Err(Error::runtime_state(
                        "Runtime::run_compiled",
                        ErrorPhase::Execution,
                        "collective node execution is not implemented",
                    ));
                }
                ScheduledNode::Barrier(_) => {}
            }
        }
        Ok(())
    })();
    if let Err(error) = result {
        slots.clear();
        return Err(error);
    }
    Ok(slots)
}

fn instruction_executor<'a>(
    execution: &'a PreparedExecutionEngines,
    instruction: &ExecInstruction,
) -> Result<&'a Arc<dyn ErasedTensorBackendExecutor>> {
    let Some(operation_index) = instruction.semantic_operation_index else {
        return Ok(&execution.root);
    };
    execution.operations.get(operation_index).ok_or_else(|| {
        Error::runtime_state(
            "Runtime::run_compiled",
            ErrorPhase::Execution,
            format!(
                "instruction references semantic operation {operation_index}, but prepared program has {} operations",
                execution.operations.len()
            ),
        )
    })
}

fn input_signature(inputs: &[&Tensor]) -> Result<InputSignature> {
    let reads: RuntimeInputReads<'_> = inputs
        .iter()
        .map(|tensor| TensorRead::from_tensor(tensor))
        .collect();
    InputSignature::from_reads(&reads).map_err(|source| prepare_error(Arc::new(source)))
}

fn resolve_input_refs<'a>(
    program: &'a CompiledGraph,
    inputs: &'a [&'a Tensor],
) -> Result<RuntimeInputRefs<'a>> {
    let resolved = if inputs.is_empty() {
        semantic_default_inputs(program)?
    } else {
        inputs.iter().copied().collect()
    };
    validate_ordered_input_metadata(program, &resolved)?;
    Ok(resolved)
}

fn semantic_default_inputs(program: &CompiledGraph) -> Result<RuntimeInputRefs<'_>> {
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

fn validate_ordered_input_metadata(program: &CompiledGraph, inputs: &[&Tensor]) -> Result<()> {
    let expected = program.input_count();
    if inputs.len() != expected {
        return Err(Error::GraphInputCountMismatch {
            expected,
            actual: inputs.len(),
        });
    }
    let input_shapes: RuntimeInputShapes<'_> = inputs.iter().map(|tensor| tensor.shape()).collect();
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
        let mut expected_shape: RuntimeShapeScratch = actual.shape().iter().copied().collect();
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
                expected: expected_shape.into_vec(),
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
