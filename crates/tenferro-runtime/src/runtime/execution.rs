//! Runtime-owned compiled-graph execution boundary.
//!
//! Phase 5 fills this module with the private execution bridge used by
//! `Runtime::run_compiled*`. The legacy `GraphExecutor` compatibility facade
//! must not grow new runtime ownership after this boundary exists.

use std::error::Error as StdError;
use std::fmt;
use std::sync::{Arc, Mutex};

use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_tensor::{Tensor, TensorBackend, TensorRead, TensorValue};

use crate::error::ErrorPhase;
use crate::exec::{ExecProgram, ExecSlot};
use crate::extension_runtime::ExtensionExecutor;
use crate::graph::CompiledGraph;
use crate::runtime::{InputSignature, PrepareError, PrepareOptions, Runtime};
use crate::{Error, Result};

#[allow(
    dead_code,
    reason = "Phase 5 runtime execution task adds erased dispatch methods"
)]
pub(super) trait ErasedTensorBackendExecutor: fmt::Debug + Send + Sync {
    fn backend_type_name(&self) -> &'static str;
    fn execute(&self, program: &ExecProgram, inputs: Vec<Tensor>) -> Result<Vec<Tensor>>;
    fn execute_values(
        &self,
        program: &ExecProgram,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<TensorValue>>;
}

pub(super) fn erased_tensor_backend_executor<B>(backend: B) -> Arc<dyn ErasedTensorBackendExecutor>
where
    B: TensorBackend + Clone + Send + Sync + 'static,
{
    Arc::new(TensorBackendExecutor::<B>::new(backend))
}

#[allow(
    dead_code,
    reason = "Phase 5 runtime execution task consumes backend execution state"
)]
struct TensorBackendExecutorState<B: TensorBackend + 'static> {
    backend: B,
    backend_cache: B::RuntimeCache,
    extension_executor: ExtensionExecutor<B>,
    slot_workspace: Vec<Option<ExecSlot<'static>>>,
}

struct TensorBackendExecutor<B: TensorBackend + 'static> {
    state: Mutex<TensorBackendExecutorState<B>>,
}

impl<B> TensorBackendExecutor<B>
where
    B: TensorBackend + Clone + Send + Sync + 'static,
{
    fn new(backend: B) -> Self {
        Self {
            state: Mutex::new(TensorBackendExecutorState {
                backend,
                backend_cache: B::RuntimeCache::default(),
                extension_executor: ExtensionExecutor::new(),
                slot_workspace: Vec::new(),
            }),
        }
    }
}

impl<B: TensorBackend + 'static> fmt::Debug for TensorBackendExecutor<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorBackendExecutor")
            .field("backend_type", &std::any::type_name::<B>())
            .field("state_poisoned", &self.state.is_poisoned())
            .finish_non_exhaustive()
    }
}

impl<B> ErasedTensorBackendExecutor for TensorBackendExecutor<B>
where
    B: TensorBackend + Clone + Send + Sync + 'static,
{
    fn backend_type_name(&self) -> &'static str {
        std::any::type_name::<B>()
    }

    fn execute(&self, program: &ExecProgram, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        validate_exec_input_count(program, inputs.len())?;
        let input_shapes: Vec<&[usize]> = inputs.iter().map(Tensor::shape).collect();
        crate::exec::validate_shape_guards(program, &input_shapes)?;
        let mut state = self.state.lock().map_err(|_| {
            Error::runtime_state(
                "Runtime::run_compiled",
                ErrorPhase::Execution,
                "tensor backend executor state poisoned",
            )
        })?;
        let TensorBackendExecutorState {
            backend,
            backend_cache,
            extension_executor,
            slot_workspace,
        } = &mut *state;
        crate::segment::eval_exec_segmented_with_cache_and_workspace(
            backend,
            program,
            inputs,
            slot_workspace,
            backend_cache,
            Some(extension_executor),
        )
    }

    fn execute_values(
        &self,
        program: &ExecProgram,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<TensorValue>> {
        validate_exec_input_count(program, inputs.len())?;
        let input_shapes: Vec<&[usize]> = inputs.iter().map(Tensor::shape).collect();
        crate::exec::validate_shape_guards(program, &input_shapes)?;
        let inputs = inputs.into_iter().map(ExecSlot::Owned).collect();
        let mut state = self.state.lock().map_err(|_| {
            Error::runtime_state(
                "Runtime::run_compiled_values",
                ErrorPhase::Execution,
                "tensor backend executor state poisoned",
            )
        })?;
        let TensorBackendExecutorState {
            backend,
            backend_cache,
            extension_executor,
            slot_workspace,
        } = &mut *state;
        crate::segment::eval_exec_segmented_slot_values_with_cache_and_workspace(
            backend,
            program,
            inputs,
            slot_workspace,
            backend_cache,
            Some(extension_executor),
        )
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
    executor.execute(prepared.root().staging(), inputs)
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
    executor.execute_values(prepared.root().staging(), inputs)
}

fn prepare(
    runtime: &Runtime,
    program: &CompiledGraph,
    signature: &InputSignature,
) -> Result<Arc<super::preparation::PreparedProgram>> {
    let prepared = runtime
        .prepare_compiled_for(program, signature, &PrepareOptions::new())
        .map_err(prepare_error)?;
    let projected = prepared
        .specialization()
        .requirements()
        .project(signature)
        .map_err(|source| prepare_error(Arc::new(source)))?;
    if &projected != prepared.specialization() {
        return Err(Error::runtime_state(
            "Runtime::run_compiled",
            ErrorPhase::Execution,
            "prepared specialization does not match input signature",
        ));
    }
    Ok(prepared)
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
