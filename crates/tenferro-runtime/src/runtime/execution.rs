//! Runtime-owned compiled-graph execution boundary.
//!
//! This module owns the private execution bridge used by
//! `Runtime::run_compiled*`.

use std::collections::{HashMap, HashSet};
use std::error::Error as StdError;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, ThreadId};

use smallvec::SmallVec;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_tensor::{
    AllocationGroup, DescriptorSlot, GroupError, Tensor, TensorBackend, TensorRead, TensorValue,
    TensorView,
};

use crate::error::ErrorPhase;
use crate::exec::{
    DispatchMode, ExecInstruction, ExecProgram, ExecSlot, ExtensionExecutionDispatch,
};
use crate::extension_cache::{ExtensionCacheSelector, ExtensionCacheStore};
use crate::graph::CompiledGraph;
use crate::runtime::schedule::{
    EventDependency, ExecutionLocation, ScheduledGraph, ScheduledNode, ScheduledNodeKind,
    ScheduledTransfer, UnsupportedScheduledNodeError,
};
use crate::runtime::{
    CacheOwnerError, CacheStats, EventDomainError, EventDomainOperation, EventDomainRun,
    EventToken, InputSignature, PrepareError, PrepareOptions, PreparedOperationPlan, Runtime,
    RuntimeCacheOwner, SubmissionError, TransferError, TransferProviderContractError,
    TransferRequest,
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
/// This handle keeps the prepared execution staging and its direct provider
/// witnesses inside the immutable schedule. It is tied to the runtime epoch
/// that created it and becomes stale after runtime reconfiguration.
#[derive(Clone)]
pub struct PreparedCompiledGraph {
    runtime_id: super::RuntimeId,
    epoch: super::RuntimeEpoch,
    program: CompiledGraph,
    prepared: Arc<super::preparation::PreparedProgram>,
}

/// Asynchronous runtime execution handle returned by [`Runtime::submit`].
pub struct ExecutionHandle {
    submission: Arc<InFlightSubmission>,
}

/// Move-only input package for detached runtime submission.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::runtime::ExecutionInputs;
///
/// let inputs = ExecutionInputs::new(Vec::new())?;
/// assert!(format!("{inputs:?}").contains("ExecutionInputs"));
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
pub struct ExecutionInputs {
    group: AllocationGroup,
    bindings: Box<[DescriptorSlot]>,
}

impl ExecutionInputs {
    /// Construct a package from already-owned tensors.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RuntimeState`] when a tensor binding cannot be
    /// registered in the allocation group.
    pub fn new(tensors: Vec<Tensor>) -> Result<Self> {
        let (group, bindings) = AllocationGroup::from_tensors(tensors).map_err(|error| {
            Error::runtime_state(
                "ExecutionInputs::new",
                ErrorPhase::Execution,
                error.to_string(),
            )
        })?;
        Ok(Self { group, bindings })
    }

    pub(crate) fn as_reads(&self) -> Result<Vec<TensorRead<'_>>> {
        self.group.read_views(&self.bindings).map_err(|error| {
            Error::runtime_state(
                "ExecutionInputs::as_reads",
                ErrorPhase::Execution,
                error.to_string(),
            )
        })
    }
}

impl fmt::Debug for ExecutionInputs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionInputs")
            .field("len", &self.bindings.len())
            .finish()
    }
}

/// A detached execution result retaining one allocation group for all outputs.
#[derive(Debug)]
pub struct ExecutionBundle {
    group: AllocationGroup,
    outputs: Box<[DescriptorSlot]>,
}

/// A borrowed output view from an [`ExecutionBundle`].
#[derive(Debug)]
pub enum OutputRef<'a> {
    Tensor(TensorView<'a>),
}

#[derive(Debug, thiserror::Error)]
pub enum OutputAccessError {
    #[error("execution output index {index} is outside the output set")]
    InvalidOutput { index: usize },
    #[error("execution output group is invalid: {0}")]
    Group(#[from] GroupError),
}

#[derive(Debug, thiserror::Error)]
pub enum OutputExtractError {
    #[error("execution output index {index} is outside the output set")]
    InvalidOutput { index: usize },
    #[error("execution output cannot be extracted: {0}")]
    Group(#[from] GroupError),
}

impl ExecutionBundle {
    fn from_inputs_and_outputs(inputs: ExecutionInputs, outputs: Vec<Tensor>) -> Result<Self> {
        let mut group = inputs.group;
        let mut output_slots = Vec::with_capacity(outputs.len());
        for output in outputs {
            let slot = group.append_tensor(output).map_err(|error| {
                Error::runtime_state(
                    "ExecutionBundle::from_inputs_and_outputs",
                    ErrorPhase::Execution,
                    error.to_string(),
                )
            })?;
            output_slots.push(slot);
        }
        Ok(Self {
            group,
            outputs: output_slots.into_boxed_slice(),
        })
    }

    #[cfg(test)]
    pub(super) fn from_outputs(outputs: Vec<Tensor>) -> Result<Self> {
        let (group, bindings) = AllocationGroup::from_tensors(outputs).map_err(|error| {
            Error::runtime_state(
                "ExecutionBundle::from_outputs",
                ErrorPhase::Execution,
                error.to_string(),
            )
        })?;
        Ok(Self {
            group,
            outputs: bindings,
        })
    }

    pub fn output(&self, index: usize) -> std::result::Result<OutputRef<'_>, OutputAccessError> {
        let slot = *self
            .outputs
            .get(index)
            .ok_or(OutputAccessError::InvalidOutput { index })?;
        let mut views = self.group.read_views(std::slice::from_ref(&slot))?;
        let view = views
            .pop()
            .ok_or(OutputAccessError::InvalidOutput { index })?;
        match view {
            TensorRead::View(view) => Ok(OutputRef::Tensor(view)),
            TensorRead::Tensor(_) => Err(OutputAccessError::InvalidOutput { index }),
        }
    }

    // INVARIANT: extraction errors return the unchanged move-only bundle so
    // callers can retry or inspect it without a hidden copy.
    #[allow(clippy::result_large_err)]
    pub fn into_output(
        self,
        index: usize,
    ) -> std::result::Result<Tensor, (Self, OutputExtractError)> {
        let slot = match self.outputs.get(index).copied() {
            Some(slot) => slot,
            None => return Err((self, OutputExtractError::InvalidOutput { index })),
        };
        let ExecutionBundle { group, outputs } = self;
        match group.into_tensor(slot) {
            Ok(tensor) => Ok(tensor),
            Err((group, error)) => Err((Self { group, outputs }, OutputExtractError::Group(error))),
        }
    }
}

/// The result of a detached submission after the provider retirement point.
#[derive(Debug)]
pub enum ExecutionOutcome {
    /// The scheduled graph completed and produced one alias-safe output bundle.
    Completed(Box<ExecutionBundle>),
    /// The graph retired with an ordinary execution error; the input owner is
    /// available again because retirement was observed.
    RetiredFailed {
        error: Error,
        inputs: Box<ExecutionInputs>,
    },
    /// Completion could not be proven. No owner is exposed and the private
    /// in-flight record retains all provider state permanently.
    CompletionUnproven {
        error: Error,
        diagnostic_keys: Box<[String]>,
    },
}

/// A detached submission failed before ownership could be transferred to the
/// in-flight worker.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{Error, runtime::{ExecutionInputs, SubmitError}};
///
/// let error = SubmitError::PreAdmission {
///     source: Box::new(Error::Internal("input rejected".into())),
///     inputs: Box::new(ExecutionInputs::new(Vec::new())?),
/// };
/// assert!(error.to_string().contains("before admission"));
/// # Ok::<(), tenferro_runtime::Error>(())
/// ```
#[derive(Debug)]
pub enum SubmitError {
    /// Preparation/admission rejected the request. The exact input package is
    /// returned unchanged to the caller.
    PreAdmission {
        source: Box<Error>,
        inputs: Box<ExecutionInputs>,
    },
}

impl SubmitError {
    /// Recover the unchanged package from a pre-admission rejection.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{Error, runtime::{ExecutionInputs, SubmitError}};
    ///
    /// let error = SubmitError::PreAdmission {
    ///     source: Box::new(Error::Internal("input rejected".into())),
    ///     inputs: Box::new(ExecutionInputs::new(Vec::new())?),
    /// };
    /// assert!(error.into_pre_admission().is_some());
    /// # Ok::<(), tenferro_runtime::Error>(())
    /// ```
    pub fn into_pre_admission(self) -> Option<(Error, ExecutionInputs)> {
        match self {
            Self::PreAdmission { source, inputs } => Some((*source, *inputs)),
        }
    }
}

impl fmt::Display for SubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PreAdmission { source, .. } => {
                write!(formatter, "submission rejected before admission: {source}")
            }
        }
    }
}

impl StdError for SubmitError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::PreAdmission { source, .. } => {
                // Keep the pre-admission typed source at the same depth as the
                // historical runtime error chain while the owner travels in
                // this wrapper.
                source.source().or(Some(source.as_ref()))
            }
        }
    }
}

impl From<SubmitError> for Error {
    fn from(error: SubmitError) -> Self {
        match error {
            SubmitError::PreAdmission { source, .. } => *source,
        }
    }
}

impl ExecutionHandle {
    /// Wait for submitted work to finish and return its tensor outputs.
    ///
    /// # Errors
    ///
    /// Returns the submitted runtime execution [`Error`], including
    /// [`ErrorKind::RuntimeState`](tenferro_tensor::ErrorKind::RuntimeState)
    /// when the handle was already consumed or the worker panicked.
    pub fn wait(self) -> Result<ExecutionOutcome> {
        self.submission.wait()
    }
}

impl fmt::Debug for ExecutionHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionHandle")
            .field("pending", &self.submission.is_pending())
            .finish_non_exhaustive()
    }
}

pub(super) struct InFlightSubmission {
    work: Mutex<Option<InFlightWork>>,
    completion: Mutex<Option<Result<ExecutionOutcome>>>,
    completed: Condvar,
}

struct AdmittedExecution {
    prepared: PreparedCompiledGraph,
    inputs: ExecutionInputs,
}

enum InFlightWork {
    Admitted(Box<AdmittedExecution>),
    #[cfg(test)]
    Test {
        inputs: Box<ExecutionInputs>,
        work: Box<dyn FnOnce() -> Result<Vec<Tensor>> + Send>,
    },
}

impl InFlightSubmission {
    fn new(prepared: PreparedCompiledGraph, inputs: ExecutionInputs) -> Self {
        Self {
            work: Mutex::new(Some(InFlightWork::Admitted(Box::new(AdmittedExecution {
                prepared,
                inputs,
            })))),
            completion: Mutex::new(None),
            completed: Condvar::new(),
        }
    }

    #[cfg(test)]
    pub(super) fn for_test(work: impl FnOnce() -> Result<Vec<Tensor>> + Send + 'static) -> Self {
        Self {
            work: Mutex::new(Some(InFlightWork::Test {
                inputs: Box::new(ExecutionInputs::new(Vec::new()).expect("empty test inputs")),
                work: Box::new(work),
            })),
            completion: Mutex::new(None),
            completed: Condvar::new(),
        }
    }

    fn into_unstarted_inputs(self) -> ExecutionInputs {
        let work = match self.work.into_inner() {
            Ok(work) => work,
            Err(poisoned) => poisoned.into_inner(),
        };
        match work {
            Some(InFlightWork::Admitted(admitted)) => admitted.inputs,
            #[cfg(test)]
            Some(InFlightWork::Test { inputs, .. }) => *inputs,
            None => unreachable!("unstarted submission must still contain its owner"),
        }
    }

    pub(super) fn run(&self) {
        let work = match self.work.lock() {
            Ok(mut work) => work.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        let result = match work {
            Some(InFlightWork::Admitted(admitted)) => run_admitted_work(admitted),
            #[cfg(test)]
            Some(InFlightWork::Test { work, .. }) => {
                let result = catch_unwind(AssertUnwindSafe(work));
                match result {
                    Ok(Ok(outputs)) => ExecutionBundle::from_outputs(outputs)
                        .map(|bundle| ExecutionOutcome::Completed(Box::new(bundle))),
                    Ok(Err(error)) => Err(error),
                    Err(payload) => Ok(ExecutionOutcome::CompletionUnproven {
                        error: Error::runtime_state(
                            "ExecutionHandle::wait",
                            ErrorPhase::Execution,
                            panic_payload_message(payload),
                        ),
                        diagnostic_keys: Box::from(["execution.retirement-unproven".to_owned()]),
                    }),
                }
            }
            None => Err(Error::runtime_state(
                "Runtime::submit",
                ErrorPhase::Execution,
                "in-flight submission work was already consumed",
            )),
        };
        match self.completion.lock() {
            Ok(mut completion) => {
                *completion = Some(result);
            }
            Err(poisoned) => {
                *poisoned.into_inner() = Some(Err(Error::runtime_state(
                    "ExecutionHandle::wait",
                    ErrorPhase::Execution,
                    "in-flight completion lock poisoned",
                )));
            }
        }
        self.completed.notify_all();
    }

    fn wait(&self) -> Result<ExecutionOutcome> {
        let mut completion = self.completion.lock().map_err(|_| {
            Error::runtime_state(
                "ExecutionHandle::wait",
                ErrorPhase::Execution,
                "in-flight completion lock poisoned",
            )
        })?;
        loop {
            if let Some(result) = completion.take() {
                return result;
            }
            completion = self.completed.wait(completion).map_err(|_| {
                Error::runtime_state(
                    "ExecutionHandle::wait",
                    ErrorPhase::Execution,
                    "in-flight completion lock poisoned while waiting",
                )
            })?;
        }
    }

    fn is_pending(&self) -> bool {
        self.completion
            .lock()
            .map_or(true, |completion| completion.is_none())
    }
}

fn run_admitted_work(admitted: Box<AdmittedExecution>) -> Result<ExecutionOutcome> {
    let execution = catch_unwind(AssertUnwindSafe(|| {
        let input_reads = admitted.inputs.as_reads()?;
        execute_admitted(&admitted.prepared, &input_reads)
    }));
    match execution {
        Ok(Ok(outputs)) => {
            let AdmittedExecution { inputs, .. } = *admitted;
            match ExecutionBundle::from_inputs_and_outputs(inputs, outputs) {
                Ok(bundle) => Ok(ExecutionOutcome::Completed(Box::new(bundle))),
                Err(error) => Ok(ExecutionOutcome::CompletionUnproven {
                    error,
                    diagnostic_keys: Box::from(["execution.bundle-build".to_owned()]),
                }),
            }
        }
        Ok(Err(error)) => {
            let AdmittedExecution { inputs, .. } = *admitted;
            Ok(ExecutionOutcome::RetiredFailed {
                error,
                inputs: Box::new(inputs),
            })
        }
        Err(payload) => {
            let error = Error::runtime_state(
                "ExecutionHandle::wait",
                ErrorPhase::Execution,
                panic_payload_message(payload),
            );
            // The completion witness was lost after admission. Leak the
            // private record deliberately: no owner or provider state may be
            // exposed while retirement is unproven.
            Box::leak(admitted);
            Ok(ExecutionOutcome::CompletionUnproven {
                error,
                diagnostic_keys: Box::from(["execution.retirement-unproven".to_owned()]),
            })
        }
    }
}

pub(super) trait SubmissionSpawner {
    fn spawn(&self, submission: Arc<InFlightSubmission>) -> std::io::Result<()>;
}

pub(super) fn spawn_in_flight(
    submission: Arc<InFlightSubmission>,
    spawner: &dyn SubmissionSpawner,
) -> std::result::Result<ExecutionHandle, Box<(ExecutionInputs, Error)>> {
    if let Err(source) = spawner.spawn(Arc::clone(&submission)) {
        // INVARIANT: a SubmissionSpawner error means it did not take or start
        // the submission, so the sole Arc can be unwrapped and the exact
        // pre-admission owner recovered.
        let submission = match Arc::try_unwrap(submission) {
            Ok(submission) => submission,
            Err(_) => unreachable!("failed spawner must not retain submission"),
        };
        let inputs = submission.into_unstarted_inputs();
        let error = Error::runtime_state_source(
            "Runtime::submit",
            ErrorPhase::Execution,
            SubmissionError::WorkerSpawn { source },
        );
        return Err(Box::new((inputs, error)));
    }
    Ok(ExecutionHandle { submission })
}

pub(super) struct OsThreadSpawner;

impl SubmissionSpawner for OsThreadSpawner {
    fn spawn(&self, submission: Arc<InFlightSubmission>) -> std::io::Result<()> {
        thread::Builder::new()
            .name("tenferro-runtime-submit".to_string())
            .spawn(move || submission.run())
            .map(drop)
    }
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

fn panic_payload_message(payload: Box<dyn std::any::Any + Send + 'static>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        format!("submitted execution panicked: {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("submitted execution panicked: {message}")
    } else {
        "submitted execution panicked".to_string()
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
        operations: &[PreparedOperationPlan],
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>>;
    fn execute_tensor_refs(
        &self,
        program: &ExecProgram,
        operations: &[PreparedOperationPlan],
        inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>>;
    fn execute_values(
        &self,
        program: &ExecProgram,
        operations: &[PreparedOperationPlan],
        inputs: Vec<Tensor>,
    ) -> Result<Vec<TensorValue>>;
    fn execute_value_refs(
        &self,
        program: &ExecProgram,
        operations: &[PreparedOperationPlan],
        inputs: &[&Tensor],
    ) -> Result<Vec<TensorValue>>;
    fn execute_slot_instruction<'input>(
        &self,
        instruction_index: usize,
        instruction: &ExecInstruction,
        operations: &[PreparedOperationPlan],
        slots: &mut [Option<ExecSlot<'input>>],
        output_mode: RuntimeOutputMode,
        terminal_slots: &[bool],
    ) -> Result<()>;
    fn materialize_slot<'input>(&self, slot: ExecSlot<'input>) -> Result<Tensor>;
    fn materialize_slot_value<'input>(&self, slot: ExecSlot<'input>) -> Result<TensorValue>;
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
        operations: &[PreparedOperationPlan],
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
        operations: &[PreparedOperationPlan],
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
        operations: &[PreparedOperationPlan],
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
        operations: &[PreparedOperationPlan],
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
        operations: &[PreparedOperationPlan],
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

    fn materialize_slot<'input>(&self, slot: ExecSlot<'input>) -> Result<Tensor> {
        let mut lease = self.lease_state("Runtime::run_compiled collect outputs")?;
        let backend = &mut lease.state_mut().backend;
        backend.with_backend_session(|exec| slot.into_tensor(exec))
    }

    fn materialize_slot_value<'input>(&self, slot: ExecSlot<'input>) -> Result<TensorValue> {
        let mut lease = self.lease_state("Runtime::run_compiled_values collect outputs")?;
        let backend = &mut lease.state_mut().backend;
        backend.with_backend_session(|exec| slot.into_value(exec))
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
    let inputs = resolve_input_refs(program, inputs)?;
    let signature = input_signature(&inputs)?;
    let prepared = prepare(runtime, program, &signature)?;
    validate_prepared_epoch(runtime, prepared.root().epoch(), "Runtime::run_compiled")?;
    execute_scheduled_tensor_refs(
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
    validate_prepared_epoch(
        runtime,
        prepared.root().epoch(),
        "Runtime::prepare_compiled",
    )?;
    Ok(PreparedCompiledGraph {
        runtime_id: runtime.id(),
        epoch: prepared.root().epoch(),
        program: program.clone(),
        prepared,
    })
}

pub(super) fn submit(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: ExecutionInputs,
) -> std::result::Result<ExecutionHandle, SubmitError> {
    submit_with_spawner(runtime, program, inputs, &OsThreadSpawner)
}

pub(super) fn submit_with_spawner(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: ExecutionInputs,
    spawner: &dyn SubmissionSpawner,
) -> std::result::Result<ExecutionHandle, SubmitError> {
    let prepared = match prepare_submission(runtime, program, &inputs) {
        Ok(prepared) => prepared,
        Err(source) => {
            return Err(SubmitError::PreAdmission {
                source: Box::new(source),
                inputs: Box::new(inputs),
            })
        }
    };
    let submission = Arc::new(InFlightSubmission::new(prepared, inputs));
    match spawn_in_flight(submission, spawner) {
        Ok(handle) => Ok(handle),
        Err(failure) => {
            let (inputs, source) = *failure;
            Err(SubmitError::PreAdmission {
                source: Box::new(source),
                inputs: Box::new(inputs),
            })
        }
    }
}

pub(super) fn run_prepared(
    runtime: &Runtime,
    prepared: &PreparedCompiledGraph,
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    validate_prepared_runtime(runtime, prepared, "Runtime::run_prepared")?;
    let inputs = resolve_input_refs(&prepared.program, inputs)?;
    execute_scheduled_tensor_refs(
        prepared.prepared.root().staging(),
        prepared.prepared.root().schedule(),
        prepared.prepared.operations(),
        &inputs,
    )
}

fn prepare_submission(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &ExecutionInputs,
) -> Result<PreparedCompiledGraph> {
    let input_reads = inputs.as_reads()?;
    prepare_compiled_reads(runtime, program, &input_reads)
}

fn prepare_compiled_reads(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[TensorRead<'_>],
) -> Result<PreparedCompiledGraph> {
    validate_ordered_input_metadata_reads(program, inputs)?;
    let signature = input_signature_reads(inputs)?;
    let prepared = prepare(runtime, program, &signature)?;
    validate_prepared_epoch(runtime, prepared.root().epoch(), "Runtime::submit")?;
    Ok(PreparedCompiledGraph {
        runtime_id: runtime.id(),
        epoch: prepared.root().epoch(),
        program: program.clone(),
        prepared,
    })
}

fn execute_admitted(
    prepared: &PreparedCompiledGraph,
    inputs: &[TensorRead<'_>],
) -> Result<Vec<Tensor>> {
    execute_scheduled_reads(
        prepared.prepared.root().staging(),
        prepared.prepared.root().schedule(),
        prepared.prepared.operations(),
        inputs,
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
    validate_prepared_epoch(
        runtime,
        prepared.root().epoch(),
        "Runtime::run_compiled_values",
    )?;
    execute_scheduled_value_refs(
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

fn validate_prepared_epoch(
    runtime: &Runtime,
    prepared_epoch: super::RuntimeEpoch,
    caller: &'static str,
) -> Result<()> {
    let epoch = runtime
        .epoch()
        .map_err(|source| Error::runtime_state_source(caller, ErrorPhase::Execution, source))?;
    if epoch != prepared_epoch {
        return Err(Error::runtime_state(
            caller,
            ErrorPhase::Execution,
            format!(
                "prepared epoch {:?} does not match current epoch {:?}",
                prepared_epoch, epoch
            ),
        ));
    }
    Ok(())
}

fn execute_scheduled_tensor_refs(
    program: &ExecProgram,
    schedule: &ScheduledGraph,
    operations: &[PreparedOperationPlan],
    inputs: &[&Tensor],
) -> Result<Vec<Tensor>> {
    let inputs = inputs
        .iter()
        .map(|tensor| TensorRead::from_tensor(tensor))
        .collect::<Vec<_>>();
    execute_scheduled_reads(program, schedule, operations, &inputs)
}

fn execute_scheduled_reads(
    program: &ExecProgram,
    schedule: &ScheduledGraph,
    operations: &[PreparedOperationPlan],
    inputs: &[TensorRead<'_>],
) -> Result<Vec<Tensor>> {
    validate_exec_input_count(program, inputs.len())?;
    crate::exec::validate_exec_program(program, "scheduled tensor executor")?;
    let inputs = inputs.iter().cloned().map(ExecSlot::Read).collect();
    execute_scheduled_slots(
        program,
        schedule,
        operations,
        inputs,
        RuntimeOutputMode::Tensor,
    )
    .and_then(|mut slots| {
        collect_tensor_outputs_with(program, &mut slots, |location, slot| {
            location.witness().executor().materialize_slot(slot)
        })
    })
}

fn execute_scheduled_value_refs(
    program: &ExecProgram,
    schedule: &ScheduledGraph,
    operations: &[PreparedOperationPlan],
    inputs: &[&Tensor],
) -> Result<Vec<TensorValue>> {
    validate_exec_input_count(program, inputs.len())?;
    crate::exec::validate_exec_program(program, "scheduled value executor")?;
    let inputs = inputs
        .iter()
        .map(|tensor| ExecSlot::Read(TensorRead::from_tensor(tensor)))
        .collect();
    execute_scheduled_slots(
        program,
        schedule,
        operations,
        inputs,
        RuntimeOutputMode::Value,
    )
    .and_then(|mut slots| {
        collect_value_outputs_with(program, &mut slots, |location, slot| {
            location.witness().executor().materialize_slot_value(slot)
        })
    })
}

fn execute_scheduled_slots<'input>(
    program: &ExecProgram,
    schedule: &ScheduledGraph,
    operations: &[PreparedOperationPlan],
    inputs: Vec<ExecSlot<'input>>,
    output_mode: RuntimeOutputMode,
) -> Result<Vec<Option<LocatedExecSlot<'input>>>> {
    let terminal_slots = if matches!(output_mode, RuntimeOutputMode::Value) {
        crate::exec::terminal_output_slots(program)
    } else {
        Vec::new()
    };
    let mut staged = Vec::new();
    let mut located = (0..program.n_slots)
        .map(|_| Vec::new())
        .collect::<Vec<Vec<LocatedExecSlot<'input>>>>();
    // INVARIANT: the immutable schedule was validated during preparation;
    // runs are declared after the value stores so unwinding drops
    // and drains native domain work before any tensor storage is released.
    // Driver preflight completes before input ingress or any operation launch.
    let mut event_domains = ScheduledEventDomains::new(schedule)?;
    let result = (|| {
        crate::exec::initialize_exec_slots_in(program, inputs, &mut staged)?;
        if schedule.input_locations().len() != program.input_slots.len() {
            return Err(Error::runtime_state(
                "Runtime::run_compiled",
                ErrorPhase::Execution,
                format!(
                    "prepared schedule has {} input locations for {} inputs",
                    schedule.input_locations().len(),
                    program.input_slots.len()
                ),
            ));
        }
        for (&slot, location) in program.input_slots.iter().zip(schedule.input_locations()) {
            let value = staged
                .get_mut(slot)
                .and_then(Option::take)
                .ok_or(tenferro_tensor::Error::MissingValue { slot })?;
            validate_runtime_input_ingress(location, &value.as_read(), slot)?;
            located[slot].push(LocatedExecSlot {
                location: location.clone(),
                value,
            });
        }
        for (node_index, node) in schedule.nodes().iter().enumerate() {
            match node {
                ScheduledNode::Operation(operation_node) => {
                    let mut launch = || {
                        let instruction_index = operation_node.instruction_index();
                        let instruction =
                            program.instructions.get(instruction_index).ok_or_else(|| {
                                Error::runtime_state(
                                    "Runtime::run_compiled",
                                    ErrorPhase::Execution,
                                    format!(
                                        "scheduled operation references instruction \
                                         {instruction_index}, but the execution program has {} \
                                         instructions",
                                        program.instructions.len()
                                    ),
                                )
                            })?;
                        let operation = instruction_execution(schedule, instruction)?;
                        if operation.location() != operation_node.location() {
                            return Err(Error::runtime_state(
                                "Runtime::run_compiled",
                                ErrorPhase::Execution,
                                format!(
                                    "scheduled instruction {instruction_index} location does not \
                                     match its prepared executor"
                                ),
                            ));
                        }
                        stage_instruction_inputs(
                            instruction,
                            operation.location(),
                            &mut located,
                            &mut staged,
                        )?;
                        operation.executor().execute_slot_instruction(
                            instruction_index,
                            instruction,
                            operations,
                            &mut staged,
                            output_mode,
                            &terminal_slots,
                        )?;
                        validate_instruction_outputs(
                            instruction_index,
                            instruction,
                            operation.location(),
                            &staged,
                        )?;
                        retain_instruction_results(
                            instruction,
                            operation.location(),
                            &mut located,
                            &mut staged,
                        )
                    };
                    event_domains.enqueue(node_index, node, &mut launch)?;
                }
                ScheduledNode::Transfer(transfer) => {
                    let mut launch = || execute_scheduled_transfer(transfer, &mut located);
                    event_domains.enqueue(node_index, node, &mut launch)?;
                }
                ScheduledNode::Collective(_) => {
                    return Err(Error::runtime_state_source(
                        "Runtime::run_compiled",
                        ErrorPhase::Execution,
                        UnsupportedScheduledNodeError {
                            node_index,
                            node_kind: ScheduledNodeKind::Collective,
                        },
                    ));
                }
                ScheduledNode::Barrier(_) => {
                    let mut launch = || Ok(());
                    event_domains.enqueue(node_index, node, &mut launch)?;
                }
            }
        }
        Ok(())
    })();
    let drain = event_domains.drain();
    match (result, drain) {
        (Ok(()), Ok(())) => collect_located_outputs(program, &mut located),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => {
            staged.clear();
            located.clear();
            Err(error)
        }
        (Err(primary), Err(cleanup)) => {
            staged.clear();
            located.clear();
            Err(scheduled_execution_cleanup_error(primary, cleanup))
        }
    }
}

#[derive(Debug)]
pub(crate) struct ScheduledEventDomains {
    runs: Vec<RuntimeOwnedEventDomainRun>,
    completions: HashMap<EventDependency, Arc<dyn EventToken>>,
}

#[derive(Debug, thiserror::Error)]
#[error(
    "scheduled node {node_index} depends on completion {dependency:?}, but no completion token was recorded"
)]
pub(crate) struct MissingScheduledDependencyCompletionError {
    pub(crate) dependency: EventDependency,
    pub(crate) node_index: usize,
}

#[derive(Debug, thiserror::Error)]
#[error("scheduled transfer destination {destination:?} already contains value slot {value_slot}")]
pub(crate) struct DuplicateTransferDestinationError {
    pub(crate) value_slot: usize,
    pub(crate) destination: ExecutionLocation,
}

impl ScheduledEventDomains {
    pub(crate) fn new(schedule: &ScheduledGraph) -> Result<Self> {
        schedule.preflight().map_err(|source| {
            Error::runtime_state_source("Runtime::run_compiled", ErrorPhase::Execution, source)
        })?;
        let mut drivers = Vec::new();
        let mut seen_domains = HashSet::new();
        for node in schedule.nodes() {
            let domain = node.completion().domain();
            if !seen_domains.insert(domain) {
                continue;
            }
            let witness = node.event_domain_witness();
            debug_assert_eq!(witness.event_domain_id(), domain);
            drivers.push((domain, witness.event_domain_driver().clone()));
        }
        let mut runs = Vec::with_capacity(drivers.len());
        for (domain, driver) in drivers {
            let run = RuntimeOwnedEventDomainRun::new(domain, driver.begin_run(domain)?);
            let actual = run.domain(EventDomainOperation::BeginRun)?;
            if actual != domain {
                return Err(event_domain_error(EventDomainError::RunDomainMismatch {
                    operation: EventDomainOperation::BeginRun,
                    node_index: None,
                    expected: domain,
                    actual,
                }));
            }
            runs.push(run);
        }
        Ok(Self {
            runs,
            completions: HashMap::new(),
        })
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        drivers: Vec<(super::EventDomainId, Arc<dyn super::EventDomainDriver>)>,
    ) -> Result<Self> {
        let mut runs = Vec::with_capacity(drivers.len());
        for (domain, driver) in drivers {
            let run = RuntimeOwnedEventDomainRun::new(domain, driver.begin_run(domain)?);
            let actual = run.domain(EventDomainOperation::BeginRun)?;
            if actual != domain {
                return Err(event_domain_error(EventDomainError::RunDomainMismatch {
                    operation: EventDomainOperation::BeginRun,
                    node_index: None,
                    expected: domain,
                    actual,
                }));
            }
            runs.push(run);
        }
        Ok(Self {
            runs,
            completions: HashMap::new(),
        })
    }

    pub(crate) fn enqueue(
        &mut self,
        node_index: usize,
        node: &ScheduledNode,
        launch: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let completion = node.completion();
        let destination = completion.domain();
        let run_index = self.run_index(completion.domain())?;
        let actual_preflight_domain = self.runs[run_index].domain(EventDomainOperation::Enqueue)?;
        if actual_preflight_domain != destination {
            return Err(event_domain_error(EventDomainError::RunDomainMismatch {
                operation: EventDomainOperation::Enqueue,
                node_index: Some(node_index),
                expected: destination,
                actual: actual_preflight_domain,
            }));
        }
        let dependencies = self.classify_dependencies(node_index, node, destination)?;
        let actual_run_domain = self.runs[run_index].domain(EventDomainOperation::Enqueue)?;
        if actual_run_domain != destination {
            return Err(event_domain_error(EventDomainError::RunDomainMismatch {
                operation: EventDomainOperation::Enqueue,
                node_index: Some(node_index),
                expected: destination,
                actual: actual_run_domain,
            }));
        }
        let completion_event = self.runs[run_index].enqueue(&dependencies, launch)?;
        let actual = completion_event.origin();
        if actual != completion.domain() {
            return Err(event_domain_error(
                EventDomainError::CompletionTokenDomainMismatch {
                    operation: EventDomainOperation::ValidateCompletion,
                    node_index: Some(node_index),
                    expected: completion.domain(),
                    actual,
                },
            ));
        }
        self.completions.insert(
            EventDependency::from_completion(completion),
            completion_event,
        );
        Ok(())
    }

    fn classify_dependencies(
        &self,
        node_index: usize,
        node: &ScheduledNode,
        destination: super::EventDomainId,
    ) -> Result<SmallVec<[Arc<dyn EventToken>; 4]>> {
        let mut admitted = SmallVec::with_capacity(node.dependencies().len());
        for dependency in node.dependencies() {
            let dependency_completion =
                self.completions.get(dependency).cloned().ok_or_else(|| {
                    Error::runtime_state_source(
                        "Runtime::run_compiled",
                        ErrorPhase::Execution,
                        MissingScheduledDependencyCompletionError {
                            dependency: *dependency,
                            node_index,
                        },
                    )
                })?;
            let actual = dependency_completion.origin();
            if actual != dependency.domain() {
                return Err(event_domain_error(
                    EventDomainError::DependencyDomainMismatch {
                        operation: match node {
                            ScheduledNode::Transfer(_) => EventDomainOperation::TransferBridge,
                            ScheduledNode::Operation(_)
                            | ScheduledNode::Collective(_)
                            | ScheduledNode::Barrier(_) => EventDomainOperation::Enqueue,
                        },
                        node_index: Some(node_index),
                        expected: dependency.domain(),
                        actual,
                    },
                ));
            }
            match node {
                ScheduledNode::Transfer(transfer) => {
                    let source = transfer.source_event_domain();
                    if actual == destination {
                        admitted.push(dependency_completion);
                    } else if actual == source {
                        dependency_completion.wait().map_err(|source_error| {
                            event_domain_error(EventDomainError::DependencyWaitFailed {
                                operation: EventDomainOperation::TransferBridge,
                                node_index: Some(node_index),
                                expected: destination,
                                actual,
                                source: Box::new(source_error),
                            })
                        })?;
                    } else {
                        return Err(event_domain_error(
                            EventDomainError::DependencyDomainMismatch {
                                operation: EventDomainOperation::TransferBridge,
                                node_index: Some(node_index),
                                expected: source,
                                actual,
                            },
                        ));
                    }
                }
                ScheduledNode::Operation(_)
                | ScheduledNode::Collective(_)
                | ScheduledNode::Barrier(_) => {
                    if actual != destination {
                        return Err(event_domain_error(
                            EventDomainError::DependencyDomainMismatch {
                                operation: EventDomainOperation::Enqueue,
                                node_index: Some(node_index),
                                expected: destination,
                                actual,
                            },
                        ));
                    }
                    admitted.push(dependency_completion);
                }
            }
        }
        Ok(admitted)
    }

    fn run_index(&self, domain: super::EventDomainId) -> Result<usize> {
        if let Some(index) = self
            .runs
            .iter()
            .position(|run| run.requested_domain() == domain)
        {
            return Ok(index);
        }
        Err(missing_event_domain_driver(domain))
    }

    pub(crate) fn drain(&mut self) -> Result<()> {
        let mut failures = Vec::new();
        for run in &mut self.runs {
            if let Err(error) = run.drain() {
                failures.push(error);
            }
        }
        let mut failures = failures.into_iter();
        let Some(mut error) = failures.next() else {
            return Ok(());
        };
        for failure in failures {
            error = Error::with_suppressed(error, failure);
        }
        Err(error)
    }
}

#[derive(Debug)]
enum RuntimeOwnedEventDomainRunState {
    Pending(Box<dyn EventDomainRun>),
    Retired,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EventDomainRunTerminalState {
    Retired,
    Failed,
}

impl fmt::Display for EventDomainRunTerminalState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Retired => formatter.write_str("retired"),
            Self::Failed => formatter.write_str("failed"),
        }
    }
}

#[derive(Debug)]
struct RuntimeOwnedEventDomainRun {
    requested_domain: super::EventDomainId,
    state: RuntimeOwnedEventDomainRunState,
}

impl RuntimeOwnedEventDomainRun {
    fn new(requested_domain: super::EventDomainId, inner: Box<dyn EventDomainRun>) -> Self {
        Self {
            requested_domain,
            state: RuntimeOwnedEventDomainRunState::Pending(inner),
        }
    }

    fn requested_domain(&self) -> super::EventDomainId {
        self.requested_domain
    }

    fn domain(&self, operation: EventDomainOperation) -> Result<super::EventDomainId> {
        match &self.state {
            RuntimeOwnedEventDomainRunState::Pending(run) => Ok(run.domain()),
            RuntimeOwnedEventDomainRunState::Retired => Err(event_domain_run_state_error(
                operation,
                self.requested_domain,
                EventDomainRunTerminalState::Retired,
            )),
            RuntimeOwnedEventDomainRunState::Failed => Err(event_domain_run_state_error(
                operation,
                self.requested_domain,
                EventDomainRunTerminalState::Failed,
            )),
        }
    }

    fn enqueue(
        &mut self,
        dependencies: &[Arc<dyn EventToken>],
        launch: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Arc<dyn EventToken>> {
        match &mut self.state {
            RuntimeOwnedEventDomainRunState::Pending(run) => run.enqueue(dependencies, launch),
            RuntimeOwnedEventDomainRunState::Retired => Err(event_domain_run_state_error(
                EventDomainOperation::Enqueue,
                self.requested_domain,
                EventDomainRunTerminalState::Retired,
            )),
            RuntimeOwnedEventDomainRunState::Failed => Err(event_domain_run_state_error(
                EventDomainOperation::Enqueue,
                self.requested_domain,
                EventDomainRunTerminalState::Failed,
            )),
        }
    }

    fn drain(&mut self) -> Result<()> {
        let run = match std::mem::replace(&mut self.state, RuntimeOwnedEventDomainRunState::Failed)
        {
            RuntimeOwnedEventDomainRunState::Pending(run) => run,
            RuntimeOwnedEventDomainRunState::Retired => {
                self.state = RuntimeOwnedEventDomainRunState::Retired;
                return Err(event_domain_run_state_error(
                    EventDomainOperation::Drain,
                    self.requested_domain,
                    EventDomainRunTerminalState::Retired,
                ));
            }
            RuntimeOwnedEventDomainRunState::Failed => {
                self.state = RuntimeOwnedEventDomainRunState::Failed;
                return Err(event_domain_run_state_error(
                    EventDomainOperation::Drain,
                    self.requested_domain,
                    EventDomainRunTerminalState::Failed,
                ));
            }
        };
        let domain = self.requested_domain;
        let mut run = run;
        let drain_result = catch_unwind(AssertUnwindSafe(|| run.drain()));
        drop_event_domain_run(run);
        match drain_result {
            Ok(Ok(())) => {
                self.state = RuntimeOwnedEventDomainRunState::Retired;
                Ok(())
            }
            Ok(Err(error)) => {
                self.state = RuntimeOwnedEventDomainRunState::Failed;
                Err(error)
            }
            Err(payload) => {
                self.state = RuntimeOwnedEventDomainRunState::Failed;
                Err(event_domain_error(EventDomainError::DrainPanicked {
                    operation: EventDomainOperation::Drain,
                    domain,
                    message: safe_event_domain_panic_message(payload),
                }))
            }
        }
    }
}

/// Internal error returned when a runtime-owned event run is used after
/// explicit retirement or failure.
#[derive(Debug, thiserror::Error)]
#[error("{operation} used event-domain run {domain:?} after it reached terminal state {state}")]
pub(crate) struct EventDomainRunLifecycleError {
    operation: EventDomainOperation,
    domain: super::EventDomainId,
    state: EventDomainRunTerminalState,
}

fn event_domain_run_state_error(
    operation: EventDomainOperation,
    domain: super::EventDomainId,
    state: EventDomainRunTerminalState,
) -> Error {
    Error::runtime_state_source(
        "Runtime::run_compiled",
        ErrorPhase::Execution,
        EventDomainRunLifecycleError {
            operation,
            domain,
            state,
        },
    )
}

fn drop_event_domain_run(run: Box<dyn EventDomainRun>) {
    if catch_unwind(AssertUnwindSafe(|| drop(run))).is_err() {
        // The provider run has been consumed; its cleanup panic is contained.
    }
}

impl Drop for RuntimeOwnedEventDomainRun {
    fn drop(&mut self) {
        let state = std::mem::replace(&mut self.state, RuntimeOwnedEventDomainRunState::Failed);
        if let RuntimeOwnedEventDomainRunState::Pending(run) = state {
            drop_event_domain_run(run);
        }
    }
}

fn missing_event_domain_driver(domain: super::EventDomainId) -> Error {
    Error::from(EventDomainError::MissingDriver { domain })
}

fn event_domain_error(source: EventDomainError) -> Error {
    Error::from(source)
}

fn safe_event_domain_panic_message(payload: Box<dyn std::any::Any + Send + 'static>) -> String {
    match payload.downcast::<&'static str>() {
        Ok(message) => (*message).to_owned(),
        Err(payload) => match payload.downcast::<String>() {
            Ok(message) => *message,
            Err(_) => "non-string panic payload".to_owned(),
        },
    }
}

fn scheduled_execution_cleanup_error(primary: Error, cleanup: Error) -> Error {
    Error::with_suppressed(primary, cleanup)
}

fn instruction_execution<'a>(
    schedule: &'a ScheduledGraph,
    instruction: &ExecInstruction,
) -> Result<InstructionExecution<'a>> {
    let Some(operation_index) = instruction.semantic_operation_index else {
        let location = schedule.root_location();
        return Ok(InstructionExecution {
            witness: location.witness(),
            location,
        });
    };
    let location = schedule
        .operation_locations()
        .get(operation_index)
        .ok_or_else(|| {
            Error::runtime_state(
                "Runtime::run_compiled",
                ErrorPhase::Execution,
                format!(
                    "instruction references semantic operation {operation_index}, but prepared schedule has {} operations",
                    schedule.operation_locations().len()
                ),
            )
        })?;
    Ok(InstructionExecution {
        witness: location.witness(),
        location,
    })
}

struct InstructionExecution<'a> {
    witness: &'a super::snapshot::ExecutableEngineSnapshot,
    location: &'a ExecutionLocation,
}

impl InstructionExecution<'_> {
    fn executor(&self) -> &Arc<dyn ErasedTensorBackendExecutor> {
        self.witness.executor()
    }

    fn location(&self) -> &ExecutionLocation {
        self.location
    }
}

pub(crate) struct LocatedExecSlot<'input> {
    pub(crate) location: ExecutionLocation,
    pub(crate) value: ExecSlot<'input>,
}

fn stage_instruction_inputs<'input>(
    instruction: &ExecInstruction,
    location: &ExecutionLocation,
    located: &mut [Vec<LocatedExecSlot<'input>>],
    staged: &mut [Option<ExecSlot<'input>>],
) -> Result<()> {
    for &slot in &instruction.input_slots {
        if staged
            .get(slot)
            .ok_or(tenferro_tensor::Error::MissingValue { slot })?
            .is_some()
        {
            continue;
        }
        let values = located
            .get_mut(slot)
            .ok_or(tenferro_tensor::Error::MissingValue { slot })?;
        let value_index = values
            .iter()
            .position(|value| &value.location == location)
            .ok_or(tenferro_tensor::Error::MissingValue { slot })?;
        staged[slot] = Some(values.swap_remove(value_index).value);
    }
    Ok(())
}

fn validate_instruction_outputs(
    instruction_index: usize,
    instruction: &ExecInstruction,
    location: &ExecutionLocation,
    staged: &[Option<ExecSlot<'_>>],
) -> Result<()> {
    for &output_slot in &instruction.output_slots {
        let output = staged
            .get(output_slot)
            .and_then(Option::as_ref)
            .ok_or(tenferro_tensor::Error::MissingValue { slot: output_slot })?;
        let output = output.as_read();
        if !location
            .witness()
            .owns_resident_tensor(&output, location.storage_class())
        {
            return Err(Error::runtime_state_source(
                "Runtime::run_compiled",
                ErrorPhase::Execution,
                super::EngineExecutionContractError::OutputResidencyMismatch {
                    instruction_index,
                    output_slot,
                    engine_id: location.engine_id().clone(),
                    storage_class: location.storage_class().clone(),
                    backend_family: output.backend_family(),
                    allocation_domain: output.allocation_domain(),
                },
            ));
        }
    }
    Ok(())
}

pub(crate) fn retain_instruction_results<'input>(
    instruction: &ExecInstruction,
    location: &ExecutionLocation,
    located: &mut [Vec<LocatedExecSlot<'input>>],
    staged: &mut [Option<ExecSlot<'input>>],
) -> Result<()> {
    for &slot in &instruction.input_slots {
        let is_output = instruction.output_slots.contains(&slot);
        let is_last_use = instruction
            .input_slots
            .iter()
            .enumerate()
            .any(|(index, &candidate)| {
                candidate == slot && instruction.last_use.get(index).copied().unwrap_or(false)
            });
        if is_last_use {
            located[slot].clear();
            if !is_output {
                staged[slot].take();
            }
        } else if !is_output {
            if let Some(value) = staged[slot].take() {
                located[slot].push(LocatedExecSlot {
                    location: location.clone(),
                    value,
                });
            }
        }
    }

    for &slot in &instruction.output_slots {
        let value = staged
            .get_mut(slot)
            .and_then(Option::take)
            .ok_or(tenferro_tensor::Error::MissingValue { slot })?;
        let values = located
            .get_mut(slot)
            .ok_or(tenferro_tensor::Error::MissingValue { slot })?;
        values.clear();
        values.push(LocatedExecSlot {
            location: location.clone(),
            value,
        });
    }
    Ok(())
}

fn validate_runtime_input_ingress(
    location: &ExecutionLocation,
    input: &TensorRead<'_>,
    slot: usize,
) -> Result<()> {
    let accepted = location
        .witness()
        .accepts_runtime_input(input, location.storage_class());
    if accepted {
        return Ok(());
    }
    Err(Error::runtime_state_source(
        "Runtime::run_compiled",
        ErrorPhase::Execution,
        super::InputIngressContractError::ResidencyMismatch {
            input_slot: slot,
            ingress_engine_id: location.engine_id().clone(),
            ingress_storage_class: location.storage_class().clone(),
            placement: input.placement().clone(),
            backend_family: input.backend_family(),
            allocation_domain: input.allocation_domain(),
        },
    ))
}

fn execute_scheduled_transfer<'input>(
    transfer: &ScheduledTransfer,
    located: &mut [Vec<LocatedExecSlot<'input>>],
) -> Result<()> {
    let source = transfer.source_location();
    let destination = transfer.destination_location();
    let provider = transfer.provider();
    let values =
        located
            .get_mut(transfer.value_slot())
            .ok_or(tenferro_tensor::Error::MissingValue {
                slot: transfer.value_slot(),
            })?;
    if values.iter().any(|value| &value.location == destination) {
        return Err(Error::runtime_state_source(
            "Runtime::run_compiled",
            ErrorPhase::Execution,
            DuplicateTransferDestinationError {
                value_slot: transfer.value_slot(),
                destination: destination.clone(),
            },
        ));
    }
    let transferred = {
        let source_value = values
            .iter()
            .find(|value| &value.location == source)
            .ok_or(tenferro_tensor::Error::MissingValue {
                slot: transfer.value_slot(),
            })?;
        let source_read = source_value.value.as_read();
        let expected_dtype = source_read.dtype();
        let expected_shape = source_read.shape().to_vec();
        let transferred =
            provider.transfer_blocking(TransferRequest::new(source, destination, source_read))?;
        validate_transfer_output(destination, expected_dtype, &expected_shape, &transferred)?;
        transferred
    };
    values.push(LocatedExecSlot {
        location: destination.clone(),
        value: ExecSlot::Owned(transferred),
    });
    Ok(())
}

fn validate_transfer_output(
    destination: &ExecutionLocation,
    expected_dtype: tenferro_tensor::DType,
    expected_shape: &[usize],
    output: &Tensor,
) -> Result<()> {
    let expected_elements = checked_transfer_element_count(expected_shape).map_err(|source| {
        Error::runtime_state_source(
            "Runtime::run_compiled",
            ErrorPhase::Execution,
            TransferError::ProviderContract { source },
        )
    })?;
    let contract_error = if output.dtype() != expected_dtype {
        Some(TransferProviderContractError::DTypeMismatch {
            expected: expected_dtype,
            actual: output.dtype(),
        })
    } else if output.shape() != expected_shape {
        Some(TransferProviderContractError::ShapeMismatch {
            expected: expected_shape.to_vec(),
            actual: output.shape().to_vec(),
        })
    } else if tensor_buffer_len(output) != expected_elements {
        Some(TransferProviderContractError::InvalidBufferLength {
            expected: expected_elements,
            actual: tensor_buffer_len(output),
        })
    } else if !destination
        .witness()
        .accepts_input_placement(output.placement(), destination.storage_class())
    {
        Some(
            TransferProviderContractError::DestinationPlacementMismatch {
                destination_engine_id: destination.engine_id().clone(),
                destination_storage_class: destination.storage_class().clone(),
                actual: output.placement().clone(),
            },
        )
    } else if !destination.witness().owns_resident_tensor(
        &TensorRead::from_tensor(output),
        destination.storage_class(),
    ) {
        Some(
            TransferProviderContractError::DestinationResidencyMismatch {
                destination_engine_id: destination.engine_id().clone(),
                destination_storage_class: destination.storage_class().clone(),
                actual_backend_family: TensorRead::from_tensor(output).backend_family(),
                actual_allocation_domain: TensorRead::from_tensor(output).allocation_domain(),
            },
        )
    } else {
        None
    };
    match contract_error {
        None => Ok(()),
        Some(source) => Err(Error::runtime_state_source(
            "Runtime::run_compiled",
            ErrorPhase::Execution,
            TransferError::ProviderContract { source },
        )),
    }
}

fn checked_transfer_element_count(
    shape: &[usize],
) -> std::result::Result<usize, TransferProviderContractError> {
    tenferro_tensor::validate::checked_shape_product(
        "Runtime::run_compiled",
        "transfer source shape",
        shape,
    )
    .map_err(|source| TransferProviderContractError::LogicalElementCount { source })
}

fn tensor_buffer_len(tensor: &Tensor) -> usize {
    match tensor {
        Tensor::F32(tensor) => tensor.buffer().len(),
        Tensor::F64(tensor) => tensor.buffer().len(),
        Tensor::I32(tensor) => tensor.buffer().len(),
        Tensor::I64(tensor) => tensor.buffer().len(),
        Tensor::Bool(tensor) => tensor.buffer().len(),
        Tensor::C32(tensor) => tensor.buffer().len(),
        Tensor::C64(tensor) => tensor.buffer().len(),
    }
}

#[cfg(test)]
mod transfer_validation_tests {
    use std::error::Error as _;

    use super::checked_transfer_element_count;
    use crate::TransferProviderContractError;

    #[test]
    fn transfer_element_count_overflow_is_typed_and_preserves_source() {
        let error = checked_transfer_element_count(&[usize::MAX, 2]).unwrap_err();

        assert!(matches!(
            error,
            TransferProviderContractError::LogicalElementCount { .. }
        ));
        assert!(error.source().is_some());
    }
}

fn collect_located_outputs<'input>(
    program: &ExecProgram,
    located: &mut [Vec<LocatedExecSlot<'input>>],
) -> Result<Vec<Option<LocatedExecSlot<'input>>>> {
    let mut outputs = (0..program.n_slots).map(|_| None).collect::<Vec<_>>();
    for &slot in &program.output_slots {
        if outputs[slot].is_some() {
            continue;
        }
        let values = located
            .get_mut(slot)
            .ok_or(tenferro_tensor::Error::MissingValue { slot })?;
        let value = values
            .pop()
            .ok_or(tenferro_tensor::Error::MissingValue { slot })?;
        values.clear();
        outputs[slot] = Some(value);
    }
    located.iter_mut().for_each(Vec::clear);
    Ok(outputs)
}

pub(super) fn collect_tensor_outputs_with<'input>(
    program: &ExecProgram,
    outputs: &mut [Option<LocatedExecSlot<'input>>],
    mut materialize: impl FnMut(&ExecutionLocation, ExecSlot<'input>) -> Result<Tensor>,
) -> Result<Vec<Tensor>> {
    program
        .output_slots
        .iter()
        .map(|&slot| {
            let located = outputs
                .get_mut(slot)
                .and_then(Option::take)
                .ok_or(tenferro_tensor::Error::MissingValue { slot })?;
            materialize(&located.location, located.value)
        })
        .collect()
}

fn collect_value_outputs_with<'input>(
    program: &ExecProgram,
    outputs: &mut [Option<LocatedExecSlot<'input>>],
    mut materialize: impl FnMut(&ExecutionLocation, ExecSlot<'input>) -> Result<TensorValue>,
) -> Result<Vec<TensorValue>> {
    program
        .output_slots
        .iter()
        .map(|&slot| {
            let located = outputs
                .get_mut(slot)
                .and_then(Option::take)
                .ok_or(tenferro_tensor::Error::MissingValue { slot })?;
            materialize(&located.location, located.value)
        })
        .collect()
}

fn input_signature(inputs: &[&Tensor]) -> Result<InputSignature> {
    let reads: RuntimeInputReads<'_> = inputs
        .iter()
        .map(|tensor| TensorRead::from_tensor(tensor))
        .collect();
    input_signature_reads(&reads)
}

fn input_signature_reads(inputs: &[TensorRead<'_>]) -> Result<InputSignature> {
    InputSignature::from_reads(inputs).map_err(|source| prepare_error(Arc::new(source)))
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
    let actuals = inputs
        .iter()
        .map(|tensor| (tensor.dtype(), tensor.shape().to_vec()))
        .collect::<Vec<_>>();
    validate_ordered_input_metadata_values(program, &actuals, "Runtime::run_compiled")
}

fn validate_ordered_input_metadata_reads(
    program: &CompiledGraph,
    inputs: &[TensorRead<'_>],
) -> Result<()> {
    let actuals = inputs
        .iter()
        .map(|input| (input.dtype(), input.shape().to_vec()))
        .collect::<Vec<_>>();
    validate_ordered_input_metadata_values(program, &actuals, "Runtime::submit")
}

fn validate_ordered_input_metadata_values(
    program: &CompiledGraph,
    actuals: &[(tenferro_tensor::DType, Vec<usize>)],
    caller: &'static str,
) -> Result<()> {
    let expected = program.input_count();
    if actuals.len() != expected {
        return Err(Error::GraphInputCountMismatch {
            expected,
            actual: actuals.len(),
        });
    }
    let input_shapes: RuntimeInputShapes<'_> =
        actuals.iter().map(|(_, shape)| shape.as_slice()).collect();
    for (input_value, (actual_dtype, actual_shape)) in
        program.program().inputs().iter().zip(actuals)
    {
        let metadata = program
            .program()
            .value_metadata(*input_value)
            .map_err(|source| Error::runtime_state_source(caller, ErrorPhase::Execution, source))?;
        if metadata.dtype() != *actual_dtype {
            return Err(Error::PlaceholderDtypeMismatch {
                expected: metadata.dtype(),
                actual: *actual_dtype,
            });
        }
        if metadata.shape().len() != actual_shape.len() {
            return Err(Error::PlaceholderRankMismatch {
                expected: metadata.shape().len(),
                actual: actual_shape.len(),
            });
        }
        let mut expected_shape: RuntimeShapeScratch = actual_shape.iter().copied().collect();
        let mut exact_mismatch = false;
        for (axis, (extent, actual_size)) in metadata.shape().iter().zip(actual_shape).enumerate() {
            match extent {
                ShapeExtent::Exact(expression) => {
                    let expected = expression.eval(&input_shapes).map_err(|source| {
                        Error::runtime_state_source(caller, ErrorPhase::Execution, source)
                    })?;
                    expected_shape[axis] = expected;
                    exact_mismatch |= expected != *actual_size;
                }
                ShapeExtent::UpperBound(expression) => {
                    let bound = expression.eval(&input_shapes).map_err(|source| {
                        Error::runtime_state_source(caller, ErrorPhase::Execution, source)
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
                actual: actual_shape.clone(),
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
    use std::error::Error as StdError;
    use std::hash::Hasher;
    use std::num::NonZeroU64;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, TryLockError, Weak};

    use tenferro_cpu::CpuBackend;
    use tenferro_ops::dim_expr::DimExpr;
    use tenferro_ops::ext_op::ExtensionOp;
    use tenferro_ops::SymDim;
    use tenferro_tensor::{
        BackendSession, BackendSessionHost, DType, Tensor, TensorBackend, TensorRead, TypedTensor,
    };

    use crate::exec::{ExecInstruction, ExecOp, ExecProgram, ExecSlot};
    use crate::runtime::{
        CoreCapabilityBundle, EngineId, ErasedExecutionContext, EventDomainDriver, EventDomainId,
        ExecutableEngineContract, ExecutionContextIdentity, HardwareClassId, InputIngressContract,
        InputSignature, PreparedOperation, PreparedOperationBinding, PreparedOperationExecutor,
        PreparedOperationPlan, ProviderDeviceIdentity, ProviderId, RegistrationIdentity,
        RuntimeCacheOwner, RuntimeEpoch, RuntimeId, SpecializationProjection,
        SpecializationRequirements, StorageClass,
    };
    use crate::{Error, ErrorPhase, ExtensionCacheStore, Result};

    use super::{
        DuplicateTransferDestinationError, ErasedTensorBackendExecutor, LocatedExecSlot,
        TensorBackendExecutor,
    };
    use crate::runtime::schedule::{
        EventCompletion, EventSlotId, ExecutionLocation, ScheduledTransfer,
    };

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
    }

    impl PreparedOperationExecutor for LockProbePreparedOperation {
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

    #[derive(Debug)]
    struct SessionProbePreparedOperation {
        binding: PreparedOperationBinding,
        specialization: SpecializationProjection,
        observed_session: Arc<AtomicBool>,
    }

    impl PreparedOperation for SessionProbePreparedOperation {
        fn binding(&self) -> &PreparedOperationBinding {
            &self.binding
        }

        fn specialization(&self) -> &SpecializationProjection {
            &self.specialization
        }

        fn retained_bytes(&self) -> usize {
            0
        }
    }

    impl PreparedOperationExecutor for SessionProbePreparedOperation {
        fn execute(
            &self,
            _context: &mut ErasedExecutionContext<'_>,
            _extension_caches: &mut ExtensionCacheStore,
            _inputs: &[TensorRead<'_>],
        ) -> Result<Vec<Tensor>> {
            Err(Error::unsupported(
                "session_probe",
                ErrorPhase::Execution,
                "session probe must use the scheduler-owned session path",
            ))
        }

        fn supports_session(&self) -> bool {
            true
        }

        fn execute_in_session(
            &self,
            session: &mut dyn BackendSession,
            _extension_caches: &mut ExtensionCacheStore,
            inputs: &[TensorRead<'_>],
        ) -> Result<Vec<Tensor>> {
            self.observed_session.store(true, Ordering::SeqCst);
            Ok(vec![session.to_contiguous_read(inputs[0].clone())?])
        }
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
    }

    impl PreparedOperationExecutor for ReentrantProbePreparedOperation {
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
            self.probe_reentrant_call(materialized.duplicate()?);
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

    fn partial_session_probe_program() -> ExecProgram {
        let output_shape = || {
            vec![DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            }]
        };
        ExecProgram {
            instructions: vec![
                ExecInstruction {
                    op: ExecOp::Extension(Arc::new(LockProbeOp)),
                    semantic_operation_index: Some(0),
                    input_slots: vec![0],
                    output_slots: vec![1],
                    dtype: DType::F64,
                    output_shapes: vec![output_shape()].into(),
                    output_extents: vec![vec![]].into(),
                    last_use: vec![false],
                },
                ExecInstruction {
                    op: ExecOp::Extension(Arc::new(LockProbeOp)),
                    semantic_operation_index: Some(1),
                    input_slots: vec![1],
                    output_slots: vec![2],
                    dtype: DType::F64,
                    output_shapes: vec![output_shape()].into(),
                    output_extents: vec![vec![]].into(),
                    last_use: vec![false],
                },
            ],
            input_slots: vec![0],
            output_slots: vec![2],
            n_slots: 3,
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
        let prepared = Arc::new(LockProbePreparedOperation {
            binding: probe_binding(),
            specialization: probe_specialization(),
            executor: Arc::downgrade(&executor),
            observed_unlocked_state: Arc::clone(&observed_unlocked_state),
        });
        let operations = vec![PreparedOperationPlan::executable(
            prepared.clone(),
            prepared,
        )];

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
    fn scheduled_execution_cleanup_error_preserves_primary_error() {
        let primary = Error::runtime_state(
            "primary-execution",
            ErrorPhase::Execution,
            "primary execution failure",
        );
        let cleanup = Error::runtime_state(
            "event-domain-cleanup",
            ErrorPhase::Execution,
            "cleanup failure",
        );
        let primary_display = primary.to_string();

        let combined = super::scheduled_execution_cleanup_error(primary, cleanup);
        let primary_error = combined.primary().expect("primary execution error");
        let cleanup_error = combined.suppressed().expect("cleanup error");
        assert_eq!(primary_error.to_string(), primary_display);
        assert_eq!(
            cleanup_error.to_string(),
            "event-domain-cleanup (Execution): runtime state failure: cleanup failure"
        );
        assert_eq!(
            StdError::source(&combined)
                .expect("primary error in the standard source chain")
                .to_string(),
            primary_display
        );
    }

    #[test]
    fn duplicate_scheduled_transfer_destination_reports_typed_fields() {
        let domain = EventDomainId::runtime_created_for_test(
            RuntimeId::from_nonzero(nz(1)),
            RuntimeEpoch::from_nonzero(nz(1)),
            RegistrationIdentity::new(nz(1), nz(1)),
        );
        let source = ExecutionLocation::new(
            EngineId::new("tenferro-test.transfer-source").expect("source engine"),
            ProviderDeviceIdentity::new(
                ProviderId::new("tenferro-test.transfer").expect("provider id"),
                "source",
            )
            .expect("source provider target"),
            domain,
            StorageClass::new("tenferro-test.transfer-source").expect("source storage"),
        );
        let destination = ExecutionLocation::new(
            EngineId::new("tenferro-test.transfer-destination").expect("destination engine"),
            ProviderDeviceIdentity::new(
                ProviderId::new("tenferro-test.transfer").expect("provider id"),
                "destination",
            )
            .expect("destination provider target"),
            domain,
            StorageClass::new("tenferro-test.transfer-destination").expect("destination storage"),
        );
        let transfer = ScheduledTransfer::new(
            3,
            source,
            destination.clone(),
            [],
            EventCompletion::new(domain, EventSlotId::new(0), 0),
        );
        let mut located = vec![
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![LocatedExecSlot {
                location: destination.clone(),
                value: ExecSlot::Owned(f64_zeros(vec![1])),
            }],
        ];

        let error = super::execute_scheduled_transfer(&transfer, &mut located)
            .expect_err("duplicate transfer destination");
        let Error::RuntimeStateSource { source, .. } = error else {
            panic!("duplicate transfer destination must retain a typed source");
        };
        let duplicate = source
            .downcast_ref::<DuplicateTransferDestinationError>()
            .expect("typed duplicate transfer destination source");
        assert_eq!(duplicate.value_slot, 3);
        assert_eq!(duplicate.destination, destination);
    }

    #[test]
    fn tensor_backend_executor_reentrant_call_returns_error_instead_of_deadlocking() {
        let executor = Arc::new(TensorBackendExecutor::<CpuBackend>::new(CpuBackend::new()));
        let observed_reentrant_error = Arc::new(AtomicBool::new(false));
        let prepared = Arc::new(ReentrantProbePreparedOperation {
            binding: probe_binding(),
            specialization: probe_specialization(),
            executor: Arc::downgrade(&executor),
            observed_reentrant_error: Arc::clone(&observed_reentrant_error),
        });
        let operations = vec![PreparedOperationPlan::executable(
            prepared.clone(),
            prepared,
        )];

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
    fn tensor_backend_executor_dispatches_session_capable_extension_in_one_session_path() {
        let executor = TensorBackendExecutor::<CpuBackend>::new(CpuBackend::new());
        let observed_session = Arc::new(AtomicBool::new(false));
        let prepared = Arc::new(SessionProbePreparedOperation {
            binding: probe_binding(),
            specialization: probe_specialization(),
            observed_session: Arc::clone(&observed_session),
        });
        let operations = vec![PreparedOperationPlan::executable(
            prepared.clone(),
            prepared,
        )];

        let output = ErasedTensorBackendExecutor::execute(
            &executor,
            &lock_probe_program(),
            &operations,
            vec![f64_zeros(vec![2])],
        )
        .expect("session-capable extension executes");

        assert_eq!(output[0].shape(), &[2]);
        assert!(observed_session.load(Ordering::SeqCst));
    }

    #[test]
    fn tensor_backend_executor_batches_session_capable_region_after_boundary() {
        let executor = Arc::new(TensorBackendExecutor::<CpuBackend>::new(CpuBackend::new()));
        let observed_unlocked_state = Arc::new(AtomicBool::new(false));
        let observed_session = Arc::new(AtomicBool::new(false));
        let ordinary = Arc::new(LockProbePreparedOperation {
            binding: probe_binding(),
            specialization: probe_specialization(),
            executor: Arc::downgrade(&executor),
            observed_unlocked_state: Arc::clone(&observed_unlocked_state),
        });
        let session = Arc::new(SessionProbePreparedOperation {
            binding: probe_binding(),
            specialization: probe_specialization(),
            observed_session: Arc::clone(&observed_session),
        });
        let operations = vec![
            PreparedOperationPlan::executable(ordinary.clone(), ordinary),
            PreparedOperationPlan::executable(session.clone(), session),
        ];

        let output = ErasedTensorBackendExecutor::execute(
            executor.as_ref(),
            &partial_session_probe_program(),
            &operations,
            vec![f64_zeros(vec![2])],
        )
        .expect("mixed session regions execute");

        assert_eq!(output[0].shape(), &[2]);
        assert!(observed_unlocked_state.load(Ordering::SeqCst));
        assert!(observed_session.load(Ordering::SeqCst));

        let values = ErasedTensorBackendExecutor::execute_values(
            executor.as_ref(),
            &partial_session_probe_program(),
            &operations,
            vec![f64_zeros(vec![2])],
        )
        .expect("mixed session regions execute in value mode");
        assert_eq!(values[0].shape(), &[2]);
    }

    #[test]
    fn tensor_backend_executor_bridge_does_not_require_clone_source_contract() {
        type ContractConstructor<B> = fn(
            ProviderDeviceIdentity,
            CoreCapabilityBundle,
            B,
            Arc<dyn EventDomainDriver>,
            InputIngressContract,
            Option<Arc<dyn RuntimeCacheOwner>>,
        ) -> ExecutableEngineContract;

        fn factory_accepts_backend_without_clone_bound<B>()
        where
            B: TensorBackend + Send + Sync + 'static,
        {
            let _factory: fn(B) -> Arc<dyn ErasedTensorBackendExecutor> =
                super::erased_tensor_backend_executor::<B>;
        }

        fn contract_accepts_backend_without_clone_bound<B>()
        where
            B: TensorBackend + Send + Sync + 'static,
        {
            let _constructor: ContractConstructor<B> = ExecutableEngineContract::new::<B>;
        }

        factory_accepts_backend_without_clone_bound::<CpuBackend>();
        contract_accepts_backend_without_clone_bound::<CpuBackend>();
    }
}
