//! Runtime-owned compiled-graph execution boundary.
//!
//! [`crate::Runtime::prepare`] turns a compiled graph into a [`Prepared`]
//! program for a set of engine targets, and [`crate::Runtime::run`] runs it on
//! the engines the caller passes in. The runtime holds no engine and no
//! execution state between calls.

use num_complex::{Complex32, Complex64};
use std::collections::{HashMap, HashSet};
use std::error::Error as StdError;
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use smallvec::SmallVec;
use tenferro_ops::shape_extent::ShapeExtent;
use tenferro_tensor::{DType, Engine, Tensor, TensorRead};
use tinyvec::TinyVec;

use crate::error::ErrorPhase;
use crate::exec::{ExecInstruction, ExecProgram, ExecSlot, ExtensionExecutionDispatch};
use crate::extension_cache::ExtensionCacheStore;
use crate::graph::CompiledGraph;
use crate::runtime::schedule::{
    EventDependency, ExecutionLocation, ScheduledGraph, ScheduledNode, ScheduledNodeKind,
    ScheduledTransfer, UnsupportedScheduledNodeError,
};
use crate::runtime::{
    EngineId, EngineTarget, EventDomainDriver, EventDomainError, EventDomainOperation,
    EventDomainRun, EventToken, ExecutionEngine, InputSignature, Output, PrepareError,
    PrepareOptions, PreparedOperationPlan, ProgramPlacementConstraint, Runtime, TransferError,
    TransferProviderContractError, TransferRequest,
};
use crate::{Error, Result};

// `TensorRead` has no `Default`, so `tinyvec::Array` cannot hold it inline.
// Keep the short input-read list in a `SmallVec`, which does not require
// `Default` and stays inline for the common case.
type RuntimeInputReads<'a> = SmallVec<[TensorRead<'a>; 8]>;
type RuntimeInputShapes<'a> = TinyVec<[&'a [usize]; 8]>;
type RuntimeShapeScratch = TinyVec<[usize; 8]>;

/// A compiled graph prepared for a set of engine targets.
///
/// A prepared program holds the execution plan and the requirements each
/// target slot puts on its engine; it holds no engine, stream or event state,
/// so any engines that report the same targets can run it, and several callers
/// can run it at the same time on their own engines. It is tied to the runtime
/// epoch that created it and becomes stale after runtime reconfiguration.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::Prepared;
///
/// fn target_count(prepared: &Prepared) -> usize {
///     prepared.targets().len()
/// }
/// ```
#[derive(Clone)]
pub struct Prepared {
    runtime_id: super::RuntimeId,
    epoch: super::RuntimeEpoch,
    program: CompiledGraph,
    prepared: Arc<super::preparation::PreparedProgram>,
    /// Whether this handle's preparation ran the metadata check on the graph's
    /// *retained* inputs, i.e. on the values an empty-input run would reuse.
    ///
    /// Preparation that replaced every input validated the replacements only,
    /// so its prepared handle must re-check the defaults at execution time.
    defaults_validated: bool,
    /// The engine targets, in slot order, that a run must bind.
    targets: Arc<[EngineTarget]>,
}

impl fmt::Debug for Prepared {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Prepared")
            .field("runtime_id", &self.runtime_id)
            .field("epoch", &self.epoch)
            .field("targets", &self.targets)
            .finish_non_exhaustive()
    }
}

impl Prepared {
    /// The engine targets this program was prepared for, in slot order.
    ///
    /// [`crate::Runtime::run`] binds slot `i` to the `i`-th engine it receives.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::Prepared;
    ///
    /// fn first_engine(prepared: &Prepared) -> Option<&str> {
    ///     prepared.targets().first().map(|target| target.engine_id().as_str())
    /// }
    /// ```
    #[must_use]
    pub fn targets(&self) -> &[EngineTarget] {
        &self.targets
    }

    /// How executions must treat this handle's retained inputs.
    ///
    /// Skipping the check is sound only when preparation itself ran it on those
    /// values; preparation with a complete replacement input list did not.
    fn bound_input_check(&self) -> BoundInputCheck {
        if self.defaults_validated {
            BoundInputCheck::AlreadyValidatedAtPrepare
        } else {
            BoundInputCheck::Validate
        }
    }

    /// Number of planned elementwise regions and the instructions they cover,
    /// for execution-path parity tests.
    #[doc(hidden)]
    #[must_use]
    pub fn elementwise_region_summary(&self) -> (usize, usize) {
        super::region::region_summary(self.prepared.root().regions())
    }

    /// Number of planned elementwise regions executed as one fused command, and
    /// the number that fell back to per-instruction dispatch.
    #[doc(hidden)]
    #[must_use]
    pub fn elementwise_region_execution_counts(&self) -> (usize, usize) {
        let counters = self.prepared.root().region_counters();
        (counters.fused(), counters.fallbacks())
    }

    /// Number of successful scheduled event-domain submissions observed by this
    /// prepared program's production executor.
    ///
    /// This is runtime submission evidence, rather than a plan-derived census.
    #[doc(hidden)]
    #[must_use]
    pub fn execution_submission_count(&self) -> usize {
        self.prepared.root().region_counters().submissions()
    }

    /// Plan-derived command count for the production execution path, for
    /// execution-path tests: scheduled operations that no region covers, plus
    /// one command per planned region.
    ///
    /// This is a plan census, not a runtime submission count.
    #[doc(hidden)]
    #[must_use]
    pub fn execution_command_count(&self) -> usize {
        let root = self.prepared.root();
        super::region::command_count(root.schedule(), root.regions())
    }
}

/// Run one scheduled instruction on `exec` without entering a session.
///
/// This is the shared body of the leased scheduler path and the
/// caller-borrowed `Runtime::run_prepared_in` path; only the session source
/// differs. The owner extension fallback is not reachable here: `run_prepared_in`
/// preflight rejects every non-session-capable prepared operation, and the leased
/// path handles that fallback before entering this function.
#[allow(clippy::too_many_arguments)]
fn execute_slot_instruction_on_engine<'input>(
    engine: &mut dyn ExecutionEngine,
    extension_caches: &mut ExtensionCacheStore,
    operations: &[PreparedOperationPlan],
    instruction_index: usize,
    instruction: &ExecInstruction,
    slots: &mut [Option<ExecSlot<'input>>],
) -> Result<()> {
    if crate::exec::is_ffi_instruction(instruction)
        && crate::exec::needs_owner_extension_fallback(
            instruction,
            Some(&ExtensionExecutionDispatch {
                operations,
                caches: extension_caches,
            }),
        )
    {
        // A prepared operation that executes against the concrete engine type
        // instead of the engine operations receives the engine itself.
        let identity = engine.target().context_identity();
        let mut extension_dispatch = ExtensionExecutionDispatch {
            operations,
            caches: extension_caches,
        };
        crate::exec::execute_owner_extension_fallback(
            engine.as_any_mut(),
            identity,
            slots,
            instruction,
            Some(&mut extension_dispatch),
        )?;
        crate::exec::reclaim_last_use_inputs_exec(slots, instruction, engine);
        return Ok(());
    }
    let exec: &mut dyn Engine = engine;
    let mut extension_dispatch = ExtensionExecutionDispatch {
        operations,
        caches: extension_caches,
    };
    if crate::exec::is_host_instruction(instruction) {
        crate::exec::execute_host_instruction_exec(exec, slots, instruction)?;
    } else if crate::exec::is_ffi_instruction(instruction) {
        crate::exec::execute_ffi_instruction_exec(
            exec,
            slots,
            instruction,
            Some(instruction_index),
            Some(&mut extension_dispatch),
        )?;
    } else {
        let result = crate::exec::execute_backend_op(exec, slots, instruction)?;
        slots[instruction.output_slots[0]] = Some(ExecSlot::Owned(result));
    }
    crate::exec::reclaim_last_use_inputs_exec(slots, instruction, exec);
    Ok(())
}

/// Run one planned elementwise region on `exec`.
fn execute_elementwise_fusion_slots_on_engine<'input>(
    exec: &mut dyn Engine,
    input_slots: &[usize],
    instruction_count: usize,
    plan: &tenferro_tensor::backend::ElementwiseFusionPlan,
    slots: &mut [Option<ExecSlot<'input>>],
    output_slots: &[usize],
) -> Result<bool> {
    // The fused entry point takes owned tensors. A borrowed view input is
    // materialized once when the region is long enough to amortize that
    // copy; otherwise the per-instruction path, which resolves views itself,
    // stays in charge.
    let mut views: Vec<usize> = Vec::new();
    for &slot in input_slots {
        let value = slots
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| crate::Error::from(tenferro_tensor::Error::MissingValue { slot }))?;
        if value.as_tensor("elementwise_region").is_err() {
            views.push(slot);
        }
    }
    if !views.is_empty() {
        if instruction_count < super::region::VIEW_COPY_MIN_INSTRUCTIONS {
            return Ok(false);
        }
        for &slot in &views {
            let tensor = {
                let read = slots[slot].as_ref().map(ExecSlot::as_read).ok_or_else(|| {
                    crate::Error::from(tenferro_tensor::Error::MissingValue { slot })
                })?;
                exec.to_contiguous_read(read)?
            };
            slots[slot] = Some(ExecSlot::Owned(tensor));
        }
    }

    let mut inputs: Vec<&Tensor> = Vec::with_capacity(input_slots.len());
    for &slot in input_slots {
        let value = slots
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| crate::Error::from(tenferro_tensor::Error::MissingValue { slot }))?;
        inputs.push(value.as_tensor("elementwise_region")?);
    }

    if !crate::segment::plan_matches_input_dtypes(plan, &inputs) {
        return Ok(false);
    }

    let outputs = exec.execute_elementwise_fusion(&inputs, plan)?;
    let Some(outputs) = outputs else {
        return Ok(false);
    };
    if outputs.len() != output_slots.len() {
        return Err(crate::Error::Internal(format!(
            "fused elementwise region produced {} outputs for {} slots",
            outputs.len(),
            output_slots.len()
        )));
    }
    for (&slot, tensor) in output_slots.iter().zip(outputs) {
        slots[slot] = Some(ExecSlot::Owned(tensor));
    }
    Ok(true)
}

/// Prepare `program` for the engines described by `targets`.
pub(super) fn prepare(
    runtime: &Runtime,
    program: &CompiledGraph,
    inputs: &[TensorRead<'_>],
    targets: &[EngineTarget],
) -> Result<Prepared> {
    const CALLER: &str = "Runtime::prepare";
    if targets.is_empty() {
        return Err(Error::invalid_argument(
            CALLER,
            ErrorPhase::Execution,
            "targets",
            "prepare needs at least one engine target",
        ));
    }
    let snapshot = runtime
        .snapshot()
        .map_err(|source| Error::runtime_state_source(CALLER, ErrorPhase::Execution, source))?;
    let mut engine_ids = Vec::with_capacity(targets.len());
    for (index, target) in targets.iter().enumerate() {
        if targets[..index]
            .iter()
            .any(|earlier| earlier.engine_id() == target.engine_id())
        {
            return Err(Error::invalid_argument(
                CALLER,
                ErrorPhase::Execution,
                "targets",
                format!(
                    "engine {:?} is named by more than one target",
                    target.engine_id().as_str()
                ),
            ));
        }
        let view = snapshot.engine(target.engine_id()).ok_or_else(|| {
            Error::runtime_state(
                CALLER,
                ErrorPhase::Execution,
                format!(
                    "no engine {:?} is registered with this runtime",
                    target.engine_id().as_str()
                ),
            )
        })?;
        if view.context_identity() != target.context_identity() {
            return Err(Error::runtime_state(
                CALLER,
                ErrorPhase::Execution,
                format!(
                    "engine {:?} is registered for context {:?}, but the target reports {:?}",
                    target.engine_id().as_str(),
                    view.context_identity(),
                    target.context_identity()
                ),
            ));
        }
        engine_ids.push(target.engine_id().clone());
    }
    let defaults_validated = retained_inputs_validated(program, inputs.len());
    let inputs = resolve_input_reads(program, inputs, BoundInputCheck::Validate)?;
    let signature = input_signature_reads(&inputs)?;
    let placement = ProgramPlacementConstraint::new(engine_ids, None)
        .map_err(|source| Error::runtime_state_source(CALLER, ErrorPhase::Execution, source))?;
    let options = PrepareOptions::new().with_placement(placement);
    let prepared = runtime
        .prepare_compiled_for(program, &signature, &options)
        .map_err(prepare_error)?;
    validate_prepared_epoch(runtime, prepared.root().epoch(), CALLER)?;
    Ok(Prepared {
        runtime_id: runtime.id(),
        epoch: prepared.root().epoch(),
        program: program.clone(),
        prepared,
        defaults_validated,
        targets: targets.into(),
    })
}

/// Whether resolving `input_count` inputs makes [`resolve_input_reads`] check
/// the graph's retained inputs.
///
/// Empty inputs resolve to the retained values, and a partial list still
/// resolves every bound input, so both validate them. A list that covers every
/// input replaces all of them, which leaves the retained values unchecked.
fn retained_inputs_validated(program: &CompiledGraph, input_count: usize) -> bool {
    let expected = program.program().inputs().len();
    input_count != expected || expected == 0
}

/// Run `prepared` on the engines the caller passes in.
pub(super) fn run<'a>(
    runtime: &Runtime,
    engines: &mut [&mut dyn ExecutionEngine],
    prepared: &'a Prepared,
    inputs: &[TensorRead<'a>],
) -> Result<Vec<Output<'a>>> {
    const CALLER: &str = "Runtime::run";
    validate_prepared_runtime(runtime, prepared, CALLER)?;
    let mut engines = EngineSet::bind(engines, &prepared.targets, CALLER)?;
    let inputs = resolve_input_reads(&prepared.program, inputs, prepared.bound_input_check())?;
    let root = prepared.prepared.root();
    engines.take_caches();
    let result = execute_scheduled(
        root.staging(),
        root.schedule(),
        root.regions(),
        root.region_counters(),
        prepared.prepared.operations(),
        inputs,
        &mut engines,
    );
    engines.restore_caches();
    result
}

fn validate_prepared_runtime(
    runtime: &Runtime,
    prepared: &Prepared,
    caller: &'static str,
) -> Result<()> {
    if runtime.id() != prepared.runtime_id {
        return Err(Error::runtime_state(
            caller,
            ErrorPhase::Execution,
            "prepared program belongs to a different runtime",
        ));
    }
    validate_prepared_epoch(runtime, prepared.epoch, caller)
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

/// The engines of one run, bound to the prepared targets.
///
/// Slot `i` of the prepared program runs on `engines[i]`; [`Self::bind`]
/// checks once, before any launch, that every engine reports the target it is
/// bound to. Extension caches are moved out of the engines for the duration of
/// the run so an instruction can borrow an engine and its caches together.
struct EngineSet<'e, 'r> {
    engines: &'e mut [&'r mut dyn ExecutionEngine],
    targets: &'e [EngineTarget],
    caches: Vec<ExtensionCacheStore>,
}

impl<'e, 'r> EngineSet<'e, 'r> {
    fn bind(
        engines: &'e mut [&'r mut dyn ExecutionEngine],
        targets: &'e [EngineTarget],
        caller: &'static str,
    ) -> Result<Self> {
        if engines.len() != targets.len() {
            return Err(Error::invalid_argument(
                caller,
                ErrorPhase::Execution,
                "engines",
                format!(
                    "the program was prepared for {} engines, but {} were passed",
                    targets.len(),
                    engines.len()
                ),
            ));
        }
        for (index, (engine, target)) in engines.iter().zip(targets).enumerate() {
            let actual = engine.target();
            if &actual != target {
                return Err(Error::invalid_argument(
                    caller,
                    ErrorPhase::Execution,
                    "engines",
                    format!(
                        "engine {index} reports target {actual:?}, but the program was prepared for {target:?}"
                    ),
                ));
            }
        }
        Ok(Self {
            engines,
            targets,
            caches: Vec::new(),
        })
    }

    fn take_caches(&mut self) {
        self.caches = self
            .engines
            .iter_mut()
            .map(|engine| engine.take_extension_caches())
            .collect();
    }

    fn restore_caches(&mut self) {
        for (engine, caches) in self.engines.iter_mut().zip(self.caches.drain(..)) {
            engine.restore_extension_caches(caches);
        }
    }

    fn index_of(&self, engine_id: &EngineId) -> Result<usize> {
        self.targets
            .iter()
            .position(|target| target.engine_id() == engine_id)
            .ok_or_else(|| {
                Error::runtime_state(
                    "Runtime::run",
                    ErrorPhase::Execution,
                    format!(
                        "the prepared schedule uses engine {:?}, which is not among the run's engines",
                        engine_id.as_str()
                    ),
                )
            })
    }

    fn driver(&self, engine_id: &EngineId) -> Result<Arc<dyn EventDomainDriver>> {
        let index = self.index_of(engine_id)?;
        Ok(self.engines[index].event_domain_driver())
    }

    fn execute_slot_instruction<'input>(
        &mut self,
        location: &ExecutionLocation,
        instruction_index: usize,
        instruction: &ExecInstruction,
        operations: &[PreparedOperationPlan],
        slots: &mut [Option<ExecSlot<'input>>],
    ) -> Result<()> {
        let index = self.index_of(location.engine_id())?;
        let engine = &mut *self.engines[index];
        let caches = match self.caches.get_mut(index) {
            Some(caches) => caches,
            None => {
                return Err(Error::runtime_state(
                    "Runtime::run",
                    ErrorPhase::Execution,
                    "engine extension caches were not taken before execution",
                ))
            }
        };
        execute_slot_instruction_on_engine(
            engine,
            caches,
            operations,
            instruction_index,
            instruction,
            slots,
        )
    }

    fn execute_elementwise_fusion_slots<'input>(
        &mut self,
        location: &ExecutionLocation,
        input_slots: &[usize],
        instruction_count: usize,
        plan: &tenferro_tensor::backend::ElementwiseFusionPlan,
        slots: &mut [Option<ExecSlot<'input>>],
        output_slots: &[usize],
    ) -> Result<bool> {
        let index = self.index_of(location.engine_id())?;
        execute_elementwise_fusion_slots_on_engine(
            &mut *self.engines[index],
            input_slots,
            instruction_count,
            plan,
            slots,
            output_slots,
        )
    }

    /// Hand every event-domain run to the engine that owns its domain.
    fn retire(&mut self, event_domains: ScheduledEventDomains) -> Result<()> {
        let mut failures = Vec::new();
        for (engine_id, run) in event_domains.into_pending_runs() {
            let result = match self.index_of(&engine_id) {
                Ok(index) => self.engines[index].retire_run(run),
                Err(error) => {
                    drop_event_domain_run(run);
                    Err(error)
                }
            };
            if let Err(error) = result {
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

fn execute_scheduled<'input>(
    program: &ExecProgram,
    schedule: &ScheduledGraph,
    regions: &[super::region::ElementwiseRegion],
    region_counters: &super::region::RegionExecutionCounters,
    operations: &[PreparedOperationPlan],
    inputs: RuntimeInputReads<'input>,
    engines: &mut EngineSet<'_, '_>,
) -> Result<Vec<Output<'input>>> {
    validate_exec_input_count(program, inputs.len())?;
    crate::exec::validate_exec_program(program, "scheduled tensor executor")?;
    let mut event_domains = Some(ScheduledEventDomains::new(schedule, engines)?);
    let inputs = inputs.into_iter().map(ExecSlot::Read).collect();
    let executed = execute_scheduled_slots_into(
        program,
        schedule,
        regions,
        region_counters,
        operations,
        inputs,
        engines,
        &mut event_domains,
    );
    let Some(event_domains) = event_domains.take() else {
        return executed.and_then(|_| {
            Err(Error::runtime_state(
                "Runtime::run",
                ErrorPhase::Execution,
                "scheduled execution consumed its event domains before collecting outputs",
            ))
        });
    };
    let collected = executed.and_then(|mut outputs| collect_outputs(program, &mut outputs));
    match collected {
        Ok(outputs) => {
            engines.retire(event_domains)?;
            Ok(outputs)
        }
        Err(primary) => {
            // On failure, nothing the run produced is handed out; retire the
            // run now so its device work cannot outlive the error.
            let mut event_domains = event_domains;
            match event_domains.drain() {
                Ok(()) => Err(primary),
                Err(cleanup) => Err(scheduled_execution_cleanup_error(primary, cleanup)),
            }
        }
    }
}

/// Turn the located output slots into [`Output`]s.
///
/// A slot that still reads one of the run's inputs (the input itself or a view
/// of it) is returned borrowed, without a copy. Owned results, which are
/// always compact, are returned as they are.
fn collect_outputs<'input>(
    program: &ExecProgram,
    outputs: &mut [Option<LocatedExecSlot<'input>>],
) -> Result<Vec<Output<'input>>> {
    program
        .output_slots
        .iter()
        .map(|&slot| {
            let located = outputs
                .get_mut(slot)
                .and_then(Option::take)
                .ok_or(tenferro_tensor::Error::MissingValue { slot })?;
            match located.value {
                ExecSlot::Read(read) => Ok(Output::Borrowed(read)),
                ExecSlot::Owned(tensor) => Ok(Output::Owned(tensor)),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn execute_scheduled_slots_into<'input>(
    program: &ExecProgram,
    schedule: &ScheduledGraph,
    regions: &[super::region::ElementwiseRegion],
    region_counters: &super::region::RegionExecutionCounters,
    operations: &[PreparedOperationPlan],
    inputs: Vec<ExecSlot<'input>>,
    engines: &mut EngineSet<'_, '_>,
    event_domains: &mut Option<ScheduledEventDomains>,
) -> Result<Vec<Option<LocatedExecSlot<'input>>>> {
    let domains = event_domains.as_mut().ok_or_else(|| {
        Error::runtime_state(
            "Runtime::run_compiled",
            ErrorPhase::Execution,
            "scheduled execution has no event-domain run to submit into",
        )
    })?;
    let mut staged = Vec::new();
    let mut located = (0..program.n_slots)
        .map(|_| Vec::new())
        .collect::<Vec<Vec<LocatedExecSlot<'input>>>>();
    // INVARIANT: the immutable schedule was validated during preparation;
    // driver preflight completes before input ingress or any operation launch.
    // A provider panic is caught after this body and drops the event-domain
    // run before this frame's staged tensor storage drops.
    // Planned elementwise regions: the start node executes the whole region as
    // one fused command and the covered nodes are skipped.
    let mut region_starts: HashMap<usize, usize> = HashMap::new();
    let mut region_covered: HashSet<usize> = HashSet::new();
    for (region_index, region) in regions.iter().enumerate() {
        if let Some(&first) = region.node_indices.first() {
            region_starts.insert(first, region_index);
        }
        region_covered.extend(region.node_indices.iter().copied());
    }

    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<Vec<Option<LocatedExecSlot<'input>>>> {
            (|| {
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
                for (&slot, location) in program.input_slots.iter().zip(schedule.input_locations())
                {
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
                            if let Some(&region_index) = region_starts.get(&node_index) {
                                let region = &regions[region_index];
                                let first_instruction = program
                            .instructions
                            .get(region.instruction_range.start)
                            .ok_or_else(|| {
                                Error::runtime_state(
                                    "Runtime::run_prepared",
                                    ErrorPhase::Execution,
                                    format!(
                                        "elementwise region references instruction {}, but the                                          execution program has {} instructions",
                                        region.instruction_range.start,
                                        program.instructions.len()
                                    ),
                                )
                            })?;
                                let operation = instruction_execution(schedule, first_instruction)?;
                                if operation.location() != operation_node.location() {
                                    return Err(Error::runtime_state(
                                "Runtime::run_prepared",
                                ErrorPhase::Execution,
                                "elementwise region location does not match its first prepared                                  operation"
                                    .to_string(),
                            ));
                                }
                                let location = operation.location().clone();
                                let mut launch = || {
                                    for &slot in &region.input_slots {
                                        stage_slot_input(
                                            slot,
                                            &location,
                                            &mut located,
                                            &mut staged,
                                        )?;
                                    }
                                    let applied = engines.execute_elementwise_fusion_slots(
                                        &location,
                                        &region.input_slots,
                                        region.instructions.len(),
                                        &region.plan,
                                        &mut staged,
                                        &region.output_slots,
                                    )?;
                                    if applied {
                                        region_counters.record_fused();
                                        validate_region_outputs(region, &location, &staged)?;
                                        return retain_region_results(
                                            region,
                                            &location,
                                            &mut located,
                                            &mut staged,
                                        );
                                    }
                                    // The backend rejected the fusion: execute the
                                    // region's instructions one by one.
                                    region_counters.record_fallback();
                                    for (offset, instruction) in
                                        region.instructions.iter().enumerate()
                                    {
                                        let instruction_index =
                                            region.instruction_range.start + offset;
                                        let member = instruction_execution(schedule, instruction)?;
                                        if member.location() != operation_node.location() {
                                            return Err(Error::runtime_state(
                                        "Runtime::run_prepared",
                                        ErrorPhase::Execution,
                                        "elementwise region member location does not match the                                          region location"
                                            .to_string(),
                                    ));
                                        }
                                        stage_instruction_inputs(
                                            instruction,
                                            member.location(),
                                            &mut located,
                                            &mut staged,
                                        )?;
                                        engines.execute_slot_instruction(
                                            member.location(),
                                            instruction_index,
                                            instruction,
                                            operations,
                                            &mut staged,
                                        )?;
                                        validate_instruction_outputs(
                                            instruction_index,
                                            instruction,
                                            member.location(),
                                            &staged,
                                        )?;
                                        retain_instruction_results(
                                            instruction,
                                            member.location(),
                                            &mut located,
                                            &mut staged,
                                        )?;
                                    }
                                    Ok(())
                                };
                                domains.enqueue(node_index, node, &mut launch)?;
                                region_counters.record_submission();
                                // Every covered node's completion resolves to the region's
                                // completion token, which was recorded after the region's
                                // last command.
                                let completion =
                                    EventDependency::from_completion(node.completion());
                                if let Some(token) = domains.completions.get(&completion).cloned() {
                                    for &covered in &region.node_indices {
                                        if covered == node_index {
                                            continue;
                                        }
                                        let covered_node = schedule.nodes().get(covered).ok_or_else(|| {
                                    Error::runtime_state(
                                        "Runtime::run_prepared",
                                        ErrorPhase::Execution,
                                        format!(
                                            "elementwise region references schedule node {covered},                                              but the schedule has {} nodes",
                                            schedule.nodes().len()
                                        ),
                                    )
                                })?;
                                        domains.completions.insert(
                                            EventDependency::from_completion(
                                                covered_node.completion(),
                                            ),
                                            Arc::clone(&token),
                                        );
                                    }
                                }
                                continue;
                            }
                            if region_covered.contains(&node_index) {
                                // Executed as part of its region.
                                continue;
                            }
                            let mut launch = || {
                                let instruction_index = operation_node.instruction_index();
                                let instruction = program
                                    .instructions
                                    .get(instruction_index)
                                    .ok_or_else(|| {
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
                                engines.execute_slot_instruction(
                                    operation.location(),
                                    instruction_index,
                                    instruction,
                                    operations,
                                    &mut staged,
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
                            domains.enqueue(node_index, node, &mut launch)?;
                            region_counters.record_submission();
                        }
                        ScheduledNode::Transfer(transfer) => {
                            let mut launch = || execute_scheduled_transfer(transfer, &mut located);
                            domains.enqueue(node_index, node, &mut launch)?;
                            region_counters.record_submission();
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
                            domains.enqueue(node_index, node, &mut launch)?;
                            region_counters.record_submission();
                        }
                    }
                }
                Ok(())
            })()?;
            collect_located_outputs(program, &mut located)
        },
    ));
    match result {
        Ok(outputs) => outputs,
        Err(payload) => {
            // A provider launch may unwind through the scheduler. Drop the
            // event-domain run before this frame's staged tensor storage drops,
            // preserving the same cleanup order as the synchronous wrapper.
            event_domains.take();
            std::panic::resume_unwind(payload)
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

/// Prove that same-stream submission order can replace per-node scheduler
/// events for `schedule`.
///
/// True only when every scheduled location and node names one event domain and
/// no transfer node bridges providers. The provider run must still accept the
/// mode; this predicate is necessary, not sufficient.
fn single_domain_coalescing_eligible(schedule: &ScheduledGraph) -> bool {
    let domain = schedule.root_location().event_domain_id();
    let same_domain = |location: &ExecutionLocation| location.event_domain_id() == domain;
    !schedule
        .nodes()
        .iter()
        .any(|node| matches!(node, ScheduledNode::Transfer(_)))
        && schedule.input_locations().iter().all(same_domain)
        && schedule.operation_locations().iter().all(same_domain)
        && schedule.nodes().iter().all(|node| {
            node.completion().domain() == domain
                && node
                    .dependencies()
                    .iter()
                    .all(|dependency| dependency.domain() == domain)
        })
}

impl ScheduledEventDomains {
    fn new(schedule: &ScheduledGraph, engines: &EngineSet<'_, '_>) -> Result<Self> {
        schedule.preflight().map_err(|source| {
            Error::runtime_state_source("Runtime::run_compiled", ErrorPhase::Execution, source)
        })?;
        // A coalescing-eligible schedule names exactly one event domain: every
        // location, completion, and dependency carries the root's, and no
        // transfer node exists. Seed that single run directly instead of
        // scanning every location into a driver table only to find one entry.
        if single_domain_coalescing_eligible(schedule) {
            let domain = schedule.root_location().event_domain_id();
            let engine_id = schedule.root_location().engine_id().clone();
            let mut run = RuntimeOwnedEventDomainRun::new(
                domain,
                engine_id.clone(),
                engines.driver(&engine_id)?.begin_run(domain)?,
            );
            let actual = run.domain(EventDomainOperation::BeginRun)?;
            if actual != domain {
                return Err(event_domain_error(EventDomainError::RunDomainMismatch {
                    operation: EventDomainOperation::BeginRun,
                    node_index: None,
                    expected: domain,
                    actual,
                }));
            }
            run.enable_single_domain_coalescing();
            return Ok(Self {
                runs: vec![run],
                completions: HashMap::new(),
            });
        }
        fn add_domain(
            drivers: &mut Vec<(super::EventDomainId, EngineId)>,
            seen_domains: &mut HashSet<super::EventDomainId>,
            domain: super::EventDomainId,
            engine_id: &EngineId,
        ) {
            if seen_domains.insert(domain) {
                drivers.push((domain, engine_id.clone()));
            }
        }
        let mut drivers = Vec::new();
        let mut seen_domains = HashSet::new();
        // Seed a run for every scheduled location, not only for scheduled nodes.
        // Output materialization launches work on the output location's stream
        // after the nodes have run, and a zero-instruction graph (an identity
        // graph) has no node to seed a run at all; without the location runs
        // there would be nothing to retire that launch.
        add_domain(
            &mut drivers,
            &mut seen_domains,
            schedule.root_location().event_domain_id(),
            schedule.root_location().engine_id(),
        );
        for location in schedule.input_locations() {
            add_domain(
                &mut drivers,
                &mut seen_domains,
                location.event_domain_id(),
                location.engine_id(),
            );
        }
        for location in schedule.operation_locations() {
            add_domain(
                &mut drivers,
                &mut seen_domains,
                location.event_domain_id(),
                location.engine_id(),
            );
        }
        for node in schedule.nodes() {
            add_domain(
                &mut drivers,
                &mut seen_domains,
                node.completion().domain(),
                node.event_domain_witness().engine_id(),
            );
        }
        let mut runs = Vec::with_capacity(drivers.len());
        for (domain, engine_id) in drivers {
            let driver = engines.driver(&engine_id)?;
            let run = RuntimeOwnedEventDomainRun::new(domain, engine_id, driver.begin_run(domain)?);
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
            let engine_id =
                EngineId::new("tenferro-test.event-domain-engine").expect("test engine id");
            let run = RuntimeOwnedEventDomainRun::new(domain, engine_id, driver.begin_run(domain)?);
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
        // `Arc<dyn EventToken>` has no `Default`; `tinyvec::Array` cannot hold it.
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

    /// Release every pending run, each with the engine that owns it.
    fn into_pending_runs(self) -> Vec<(EngineId, Box<dyn EventDomainRun>)> {
        self.runs
            .into_iter()
            .filter_map(RuntimeOwnedEventDomainRun::into_pending)
            .collect()
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
    /// The engine whose driver opened this run; it takes the run over once the
    /// outputs are collected.
    engine_id: EngineId,
    state: RuntimeOwnedEventDomainRunState,
}

impl RuntimeOwnedEventDomainRun {
    fn new(
        requested_domain: super::EventDomainId,
        engine_id: EngineId,
        inner: Box<dyn EventDomainRun>,
    ) -> Self {
        Self {
            requested_domain,
            engine_id,
            state: RuntimeOwnedEventDomainRunState::Pending(inner),
        }
    }

    /// Release the pending run to its engine, or `None` once it is terminal.
    fn into_pending(mut self) -> Option<(EngineId, Box<dyn EventDomainRun>)> {
        match std::mem::replace(&mut self.state, RuntimeOwnedEventDomainRunState::Retired) {
            RuntimeOwnedEventDomainRunState::Pending(run) => Some((self.engine_id.clone(), run)),
            RuntimeOwnedEventDomainRunState::Retired | RuntimeOwnedEventDomainRunState::Failed => {
                None
            }
        }
    }

    fn requested_domain(&self) -> super::EventDomainId {
        self.requested_domain
    }

    /// Ask the provider run to accept coalesced single-domain submission.
    ///
    /// A run that keeps the default per-node contract ignores this, and a
    /// retired or failed run is never asked.
    fn enable_single_domain_coalescing(&mut self) {
        match &mut self.state {
            RuntimeOwnedEventDomainRunState::Pending(run) => run.enable_single_domain_coalescing(),
            RuntimeOwnedEventDomainRunState::Retired | RuntimeOwnedEventDomainRunState::Failed => {}
        }
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
        let drain_result = std::panic::catch_unwind(AssertUnwindSafe(|| run.drain()));
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
    if std::panic::catch_unwind(AssertUnwindSafe(|| drop(run))).is_err() {
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
        return Ok(InstructionExecution { location });
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
    Ok(InstructionExecution { location })
}

struct InstructionExecution<'a> {
    location: &'a ExecutionLocation,
}

impl InstructionExecution<'_> {
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
        stage_slot_input(slot, location, located, staged)?;
    }
    Ok(())
}

/// Move one slot's value from `located` into `staged`.
fn stage_slot_input<'input>(
    slot: usize,
    location: &ExecutionLocation,
    located: &mut [Vec<LocatedExecSlot<'input>>],
    staged: &mut [Option<ExecSlot<'input>>],
) -> Result<()> {
    if staged
        .get(slot)
        .ok_or(tenferro_tensor::Error::MissingValue { slot })?
        .is_some()
    {
        return Ok(());
    }
    let values = located
        .get_mut(slot)
        .ok_or(tenferro_tensor::Error::MissingValue { slot })?;
    let value_index = values
        .iter()
        .position(|value| &value.location == location)
        .ok_or(tenferro_tensor::Error::MissingValue { slot })?;
    staged[slot] = Some(values.swap_remove(value_index).value);
    Ok(())
}

/// Validate every live-out of a fused elementwise region: present and resident
/// on the region's location.
fn validate_region_outputs(
    region: &super::region::ElementwiseRegion,
    location: &ExecutionLocation,
    staged: &[Option<ExecSlot<'_>>],
) -> Result<()> {
    for &output_slot in &region.output_slots {
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
                "Runtime::run_prepared",
                ErrorPhase::Execution,
                super::EngineExecutionContractError::OutputResidencyMismatch {
                    instruction_index: region.instruction_range.start,
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

/// Publish a fused region's results: release region inputs whose last use is
/// inside the region and move every live-out into `located`.
fn retain_region_results<'input>(
    region: &super::region::ElementwiseRegion,
    location: &ExecutionLocation,
    located: &mut [Vec<LocatedExecSlot<'input>>],
    staged: &mut [Option<ExecSlot<'input>>],
) -> Result<()> {
    for (&slot, &last_use) in region.input_slots.iter().zip(region.input_last_use.iter()) {
        if last_use {
            located[slot].clear();
            staged[slot].take();
        }
    }
    for &slot in &region.output_slots {
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
        } else if !is_output && let Some(value) = staged[slot].take() {
            located[slot].push(LocatedExecSlot {
                location: location.clone(),
                value,
            });
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
    match tensor.dtype() {
        DType::F32 => tensor.as_typed::<f32>().map_or(0, |t| t.buffer().len()),
        DType::F64 => tensor.as_typed::<f64>().map_or(0, |t| t.buffer().len()),
        DType::I32 => tensor.as_typed::<i32>().map_or(0, |t| t.buffer().len()),
        DType::I64 => tensor.as_typed::<i64>().map_or(0, |t| t.buffer().len()),
        DType::Bool => tensor.as_typed::<bool>().map_or(0, |t| t.buffer().len()),
        DType::C32 => tensor
            .as_typed::<Complex32>()
            .map_or(0, |t| t.buffer().len()),
        DType::C64 => tensor
            .as_typed::<Complex64>()
            .map_or(0, |t| t.buffer().len()),
        // A caller-owned payload has no runtime buffer length to report, which is
        // also what the accessor falls back to when the tag table and the runtime
        // dtype disagree.
        DType::External(_) => 0,
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

fn input_signature_reads(inputs: &[TensorRead<'_>]) -> Result<InputSignature> {
    InputSignature::from_reads(inputs).map_err(|source| prepare_error(Arc::new(source)))
}

/// Whether [`resolve_input_reads`] must re-validate the graph's retained inputs.
///
/// A retained input's value is the program's own tensor and cannot change
/// between calls, so a check that already ran on it need not run again. That is
/// a fact about the resolution, not about preparedness:
/// `Prepared::defaults_validated` records whether preparation
/// itself checked these values, and only then may an execution skip the check.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BoundInputCheck {
    Validate,
    AlreadyValidatedAtPrepare,
}

fn resolve_input_reads<'a>(
    program: &'a CompiledGraph,
    inputs: &[TensorRead<'a>],
    bound_inputs: BoundInputCheck,
) -> Result<RuntimeInputReads<'a>> {
    let expected = program.program().inputs().len();
    if inputs.len() > expected {
        return Err(Error::GraphInputCountMismatch {
            expected,
            actual: inputs.len(),
        });
    }
    if inputs.is_empty() {
        let resolved = semantic_default_inputs(program)?;
        if matches!(bound_inputs, BoundInputCheck::AlreadyValidatedAtPrepare) {
            return Ok(resolved);
        }
        validate_ordered_input_metadata_reads(program, &resolved)?;
        return Ok(resolved);
    }
    let resolved = if inputs.len() == expected {
        inputs.iter().cloned().collect()
    } else {
        let mut explicit = inputs.iter();
        let mut resolved = RuntimeInputReads::new();
        for value in program.program().inputs() {
            if let Some(retained) = program.bindings().tensor_ref_for_input(*value) {
                resolved.push(retained.tensor_read()?);
            } else {
                let read = explicit.next().ok_or_else(|| {
                    Error::invalid_argument(
                        "Runtime::run",
                        ErrorPhase::Execution,
                        "inputs",
                        format!(
                            "expected {expected} inputs, or one explicit input for each unbound placeholder"
                        ),
                    )
                })?;
                resolved.push(read.clone());
            }
        }
        if explicit.next().is_some() {
            return Err(Error::invalid_argument(
                "Runtime::run",
                ErrorPhase::Execution,
                "inputs",
                format!("expected {expected} inputs, received {}", inputs.len()),
            ));
        }
        resolved
    };
    validate_ordered_input_metadata_reads(program, &resolved)?;
    Ok(resolved)
}

fn semantic_default_inputs(program: &CompiledGraph) -> Result<RuntimeInputReads<'_>> {
    program
        .program()
        .inputs()
        .iter()
        .enumerate()
        .map(|(input_index, value)| {
            let retained = program
                .bindings()
                .tensor_ref_for_input(*value)
                .ok_or_else(|| Error::UnboundPlaceholder {
                    input_key: format!("semantic input {input_index}"),
                })?;
            retained.tensor_read()
        })
        .collect()
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
