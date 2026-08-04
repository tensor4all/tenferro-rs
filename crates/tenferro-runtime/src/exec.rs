use std::sync::Arc;

use crate::error::{Error, ErrorPhase, Result};
use num_complex::{Complex32, Complex64};
use smallvec::SmallVec;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::{dim_expr::DimExpr, ShapeExtent};
use tenferro_tensor::backend::ElementwiseFusionOp;
use tenferro_tensor::Error as TensorError;
use tenferro_tensor::{
    BackendSession, CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig,
    SliceConfig, Tensor, TensorBackend, TensorRead, TensorValue, TypedTensor, ValidationError,
};

use crate::extension_cache::ExtensionCacheStore;
use crate::runtime::{ErasedExecutionContext, PreparedOperationPlan};

mod dispatch;

tenferro_core_ops::define_exec_op!();

#[derive(Clone, Debug)]
pub struct ExecInstruction {
    pub op: ExecOp,
    /// Semantic operation index that produced this execution instruction.
    ///
    /// Runtime-owned prepared execution uses this to associate staged extension
    /// instructions with their prepared operation handles. Instructions created
    /// by lower-level compiler rewrites or unit-test fixtures leave this unset.
    pub semantic_operation_index: Option<usize>,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub dtype: tenferro_tensor::DType,
    pub output_shapes: ExecOutputShapes,
    pub output_extents: ExecOutputExtents,
    pub last_use: Vec<bool>,
}

pub type ExecOutputShapes = SmallVec<[Vec<DimExpr>; 1]>;
pub type ExecOutputExtents = SmallVec<[Vec<ShapeExtent<DimExpr>>; 1]>;

#[derive(Clone, Debug)]
pub struct ExecProgram {
    pub instructions: Vec<ExecInstruction>,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub n_slots: usize,
    /// Normalized symbolic shape obligations retained during compilation.
    ///
    /// Guard internals are opaque. Execution validates them before backend,
    /// host, or extension work. Public callers obtain guarded programs through
    /// [`crate::GraphCompiler`]; this execution-stage representation remains
    /// crate-private.
    #[doc(hidden)]
    pub shape_guards: Vec<crate::ShapeGuard>,
}

pub(crate) struct ExtensionExecutionDispatch<'a> {
    pub(crate) operations: &'a [PreparedOperationPlan],
    pub(crate) caches: &'a mut ExtensionCacheStore,
}

#[derive(Debug, thiserror::Error)]
#[error(
    "prepared extension operation for family_id {family_id:?} at semantic operation {operation_index} has no executor bridge"
)]
pub(crate) struct MissingPreparedOperationExecutorError {
    pub(crate) family_id: &'static str,
    pub(crate) operation_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DispatchMode {
    Unsegmented,
    Segmented,
}

// INVARIANT: execution slots carry either a move-only tensor or a borrowed
// read; the enum is the single dispatch boundary and is not cloned.
#[allow(clippy::large_enum_variant)]
pub(crate) enum ExecSlot<'a> {
    Owned(Tensor),
    Value(TensorValue),
    Read(TensorRead<'a>),
}

impl<'a> ExecSlot<'a> {
    pub(crate) fn as_read<'slot>(&'slot self) -> TensorRead<'slot>
    where
        'a: 'slot,
    {
        match self {
            Self::Owned(tensor) => TensorRead::from_tensor(tensor),
            Self::Value(value) => value.tensor_read(),
            Self::Read(read) => read.clone(),
        }
    }

    pub(crate) fn as_tensor<'slot>(&'slot self, op: &'static str) -> Result<&'slot Tensor>
    where
        'a: 'slot,
    {
        match self {
            Self::Owned(tensor) => Ok(tensor),
            Self::Value(value) => value.as_tensor().ok_or_else(|| {
                Error::Internal(format!(
                    "{op}: owned TensorValue view reached an owned-only execution boundary"
                ))
            }),
            Self::Read(read) => read.as_tensor().ok_or_else(|| {
                Error::Internal(format!(
                    "{op}: borrowed TensorView reached an owned-only execution boundary"
                ))
            }),
        }
    }

    pub(crate) fn shape(&self) -> &[usize] {
        match self {
            Self::Owned(tensor) => tensor.shape(),
            Self::Value(value) => value.shape(),
            Self::Read(read) => read.shape(),
        }
    }

    pub(crate) fn into_tensor(self, exec: &mut dyn BackendSession) -> Result<Tensor> {
        match self {
            Self::Owned(tensor) => Ok(tensor),
            Self::Value(value) => {
                if let Some(tensor) = value.as_tensor() {
                    Ok(tensor.duplicate()?)
                } else {
                    Ok(exec.to_contiguous_read(value.tensor_read())?)
                }
            }
            Self::Read(TensorRead::Tensor(tensor)) => Ok(tensor.duplicate()?),
            Self::Read(read @ TensorRead::View(_)) => Ok(exec.to_contiguous_read(read)?),
        }
    }

    pub(crate) fn into_value(self, exec: &mut dyn BackendSession) -> Result<TensorValue> {
        match self {
            Self::Owned(tensor) => Ok(TensorValue::from_tensor(tensor)),
            Self::Value(value) => Ok(value),
            Self::Read(TensorRead::Tensor(tensor)) => {
                Ok(TensorValue::from_tensor(tensor.duplicate()?))
            }
            Self::Read(read @ TensorRead::View(_)) => {
                Ok(TensorValue::from_tensor(exec.to_contiguous_read(read)?))
            }
        }
    }
}

fn invalid_compiled_graph(message: impl Into<String>) -> Error {
    Error::Internal(message.into())
}

fn checked_input_slot(input_slots: &[usize], idx: usize, caller: &'static str) -> Result<usize> {
    input_slots.get(idx).copied().ok_or_else(|| {
        invalid_compiled_graph(format!(
            "{caller}: input index {idx} out of range for {} input slots",
            input_slots.len()
        ))
    })
}

fn checked_slot_index(
    slot: usize,
    slots_len: usize,
    role: &'static str,
    caller: &'static str,
) -> Result<usize> {
    if slot < slots_len {
        Ok(slot)
    } else {
        Err(invalid_compiled_graph(format!(
            "{caller}: {role} slot {slot} out of range for {slots_len} slots"
        )))
    }
}

fn slot_ref<'slot, 'input>(
    slots: &'slot [Option<ExecSlot<'input>>],
    slot: usize,
    role: &'static str,
    caller: &'static str,
) -> Result<&'slot ExecSlot<'input>>
where
    'input: 'slot,
{
    let slot = checked_slot_index(slot, slots.len(), role, caller)?;
    slots[slot]
        .as_ref()
        .ok_or_else(|| TensorError::MissingValue { slot }.into())
}

fn take_slot<'input>(
    slots: &mut [Option<ExecSlot<'input>>],
    slot: usize,
    role: &'static str,
    caller: &'static str,
) -> Result<ExecSlot<'input>> {
    let slot = checked_slot_index(slot, slots.len(), role, caller)?;
    slots[slot]
        .take()
        .ok_or_else(|| TensorError::MissingValue { slot }.into())
}

pub(crate) fn validate_exec_program(program: &ExecProgram, caller: &str) -> Result<()> {
    for (idx, &slot) in program.input_slots.iter().enumerate() {
        if slot >= program.n_slots {
            return Err(invalid_compiled_graph(format!(
                "{caller}: input slot {idx} points to slot {slot}, but program has {} slots",
                program.n_slots
            )));
        }
    }
    for (idx, &slot) in program.output_slots.iter().enumerate() {
        if slot >= program.n_slots {
            return Err(invalid_compiled_graph(format!(
                "{caller}: output slot {idx} points to slot {slot}, but program has {} slots",
                program.n_slots
            )));
        }
    }
    for (inst_idx, inst) in program.instructions.iter().enumerate() {
        for (idx, &slot) in inst.input_slots.iter().enumerate() {
            if slot >= program.n_slots {
                return Err(invalid_compiled_graph(format!(
                    "{caller}: instruction {inst_idx} input slot {idx} points to slot {slot}, but program has {} slots",
                    program.n_slots
                )));
            }
        }
        let expected_inputs = exec_op_input_arity_bounds(&inst.op);
        if let Some((min_inputs, max_inputs)) = expected_inputs {
            let actual = inst.input_slots.len();
            if actual < min_inputs || actual > max_inputs {
                let expected = if min_inputs == max_inputs {
                    min_inputs.to_string()
                } else {
                    format!("{min_inputs}..={max_inputs}")
                };
                return Err(invalid_compiled_graph(format!(
                    "{caller}: instruction {inst_idx} declares {actual} input slots for {:?}, expected {expected}",
                    inst.op
                )));
            }
        }
        if inst.output_slots.is_empty() {
            return Err(invalid_compiled_graph(format!(
                "{caller}: instruction {inst_idx} has no output slots"
            )));
        }
        for (idx, &slot) in inst.output_slots.iter().enumerate() {
            if slot >= program.n_slots {
                return Err(invalid_compiled_graph(format!(
                    "{caller}: instruction {inst_idx} output slot {idx} points to slot {slot}, but program has {} slots",
                    program.n_slots
                )));
            }
        }

        let expected_outputs = match &inst.op {
            ExecOp::Extension(ext) => ext.output_count(),
            _ => 1,
        };
        if inst.output_slots.len() != expected_outputs {
            return Err(invalid_compiled_graph(format!(
                "{caller}: instruction {inst_idx} declares {} output slots for {:?}, expected {expected_outputs}",
                inst.output_slots.len(),
                inst.op
            )));
        }
        if inst.output_shapes.len() != inst.output_slots.len() {
            return Err(invalid_compiled_graph(format!(
                "{caller}: instruction {inst_idx} has {} output shape entries for {} output slots",
                inst.output_shapes.len(),
                inst.output_slots.len()
            )));
        }
        if inst.output_extents.len() != inst.output_slots.len() {
            return Err(invalid_compiled_graph(format!(
                "{caller}: instruction {inst_idx} has {} output extent entries for {} output slots",
                inst.output_extents.len(),
                inst.output_slots.len()
            )));
        }
        if inst.last_use.len() > inst.input_slots.len() {
            return Err(invalid_compiled_graph(format!(
                "{caller}: instruction {inst_idx} has {} last-use flags for {} input slots",
                inst.last_use.len(),
                inst.input_slots.len()
            )));
        }
    }
    Ok(())
}

fn exec_op_input_arity_bounds(op: &ExecOp) -> Option<(usize, usize)> {
    fn n_inputs_from_dim_exprs(min_inputs: usize, exprs: &[&[DimExpr]]) -> usize {
        exprs
            .iter()
            .flat_map(|exprs| exprs.iter())
            .filter_map(DimExpr::max_input_idx)
            .max()
            .map_or(min_inputs, |max_idx| (max_idx + 1).max(min_inputs))
    }

    match op {
        ExecOp::Extension(ext) => Some((ext.input_count(), ext.input_count())),
        ExecOp::Reshape { shape } => {
            let input_count = n_inputs_from_dim_exprs(1, &[shape]);
            Some((input_count, input_count))
        }
        ExecOp::BroadcastInDim { shape, .. } => {
            let input_count = n_inputs_from_dim_exprs(1, &[shape]);
            Some((input_count, input_count))
        }
        ExecOp::GatherDynamicSliceSizes { slice_sizes, .. } => {
            let input_count = n_inputs_from_dim_exprs(2, &[slice_sizes]);
            Some((input_count, input_count))
        }
        _ => op.primitive_kind().and_then(|kind| {
            tenferro_core_ops::all_primitive_descriptors()
                .iter()
                .find(|descriptor| descriptor.kind == kind)
                .map(|descriptor| {
                    (
                        descriptor.min_inputs as usize,
                        descriptor.max_inputs as usize,
                    )
                })
        }),
    }
}

pub(crate) fn get<'slot, 'input>(
    slots: &'slot [Option<ExecSlot<'input>>],
    input_slots: &[usize],
    idx: usize,
) -> Result<&'slot Tensor>
where
    'input: 'slot,
{
    let slot = checked_input_slot(input_slots, idx, "get")?;
    slot_ref(slots, slot, "input", "get").and_then(|value| value.as_tensor("get"))
}

pub(crate) fn ensure_owned<'input>(
    exec: &mut dyn BackendSession,
    slots: &mut [Option<ExecSlot<'input>>],
    input_slots: &[usize],
    idx: usize,
) -> Result<()> {
    let slot = checked_input_slot(input_slots, idx, "ensure_owned")?;
    let needs_materialization = matches!(
        slot_ref(slots, slot, "input", "ensure_owned")?,
        ExecSlot::Value(_) | ExecSlot::Read(TensorRead::View(_))
    );
    if needs_materialization {
        let read = slot_ref(slots, slot, "input", "ensure_owned")?.as_read();
        let tensor = exec.to_contiguous_read(read)?;
        slots[slot] = Some(ExecSlot::Owned(tensor));
    }
    Ok(())
}

pub(crate) fn get_read<'slot, 'input>(
    slots: &'slot [Option<ExecSlot<'input>>],
    input_slots: &[usize],
    idx: usize,
) -> Result<TensorRead<'slot>>
where
    'input: 'slot,
{
    let slot = checked_input_slot(input_slots, idx, "get_read")?;
    slot_ref(slots, slot, "input", "get_read").map(ExecSlot::as_read)
}

pub(crate) fn initialize_exec_slots_in<'input>(
    program: &ExecProgram,
    inputs: Vec<ExecSlot<'input>>,
    slots: &mut Vec<Option<ExecSlot<'input>>>,
) -> Result<()> {
    validate_exec_input_count(program, inputs.len(), "initialize_exec_slots_in")?;
    slots.clear();
    slots.resize_with(program.n_slots, || None);
    for (i, input) in inputs.into_iter().enumerate() {
        let slot = program.input_slots[i];
        checked_slot_index(slot, slots.len(), "input", "initialize_exec_slots_in")?;
        slots[slot] = Some(input);
    }
    Ok(())
}

pub(crate) fn collect_outputs_from<'input>(
    program: &ExecProgram,
    slots: &mut [Option<ExecSlot<'input>>],
    exec: &mut dyn BackendSession,
) -> Result<Vec<Tensor>> {
    program
        .output_slots
        .iter()
        .map(|&slot| {
            let value = take_slot(slots, slot, "output", "collect_outputs_from")?;
            value.into_tensor(exec)
        })
        .collect()
}

pub(crate) fn collect_output_values_from<'input>(
    program: &ExecProgram,
    slots: &mut [Option<ExecSlot<'input>>],
    exec: &mut dyn BackendSession,
) -> Result<Vec<TensorValue>> {
    program
        .output_slots
        .iter()
        .map(|&slot| {
            let value = take_slot(slots, slot, "output", "collect_output_values_from")?;
            value.into_value(exec)
        })
        .collect()
}

pub(crate) fn terminal_output_slots(program: &ExecProgram) -> Vec<bool> {
    let mut consumed = vec![false; program.n_slots];
    for inst in &program.instructions {
        for &slot in &inst.input_slots {
            if let Some(consumed) = consumed.get_mut(slot) {
                *consumed = true;
            }
        }
    }

    let mut terminal = vec![false; program.n_slots];
    for &slot in &program.output_slots {
        if let Some(terminal) = terminal.get_mut(slot) {
            if !consumed.get(slot).copied().unwrap_or(true) {
                *terminal = true;
            }
        }
    }
    terminal
}

pub(crate) fn try_execute_terminal_value_instruction<'input>(
    exec: &mut dyn BackendSession,
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    terminal_slots: &[bool],
) -> Result<bool> {
    if inst.output_slots.len() != 1
        || !terminal_slots
            .get(inst.output_slots[0])
            .copied()
            .unwrap_or(false)
    {
        return Ok(false);
    }

    let output = match &inst.op {
        // `validate_exec_program` enforces unary arity for these terminal lazy
        // view ops before execution; direct slot 0 indexing is an internal
        // compiled-program invariant here.
        ExecOp::Transpose { perm } => {
            let input_slot = inst.input_slots[0];
            let consume_input = inst.last_use.first().copied().unwrap_or(false);
            let input = tensor_value_for_lazy_view(exec, slots, input_slot, consume_input)?;
            input.transpose_view(perm).map_err(Error::TensorRuntime)?
        }
        ExecOp::Reshape { shape } => {
            let input_slot = inst.input_slots[0];
            let consume_input = inst.last_use.first().copied().unwrap_or(false);
            let shape = resolve_tensor_shape_exprs(slots, &inst.input_slots, shape)?;
            let input = tensor_value_for_lazy_view(exec, slots, input_slot, consume_input)?;
            match input.try_reshape_view(&shape) {
                Ok(value) => value,
                Err(error) => {
                    let (input, _) = error.into_parts();
                    if consume_input {
                        slots[input_slot] = Some(ExecSlot::Value(input));
                    }
                    return Ok(false);
                }
            }
        }
        ExecOp::BroadcastInDim { shape, dims } => {
            let input_slot = inst.input_slots[0];
            let consume_input = inst.last_use.first().copied().unwrap_or(false);
            let shape = resolve_tensor_shape_exprs(slots, &inst.input_slots, shape)?;
            let input = tensor_value_for_lazy_view(exec, slots, input_slot, consume_input)?;
            input
                .broadcast_in_dim_view(&shape, dims)
                .map_err(Error::TensorRuntime)?
        }
        ExecOp::Slice(config) => {
            let input_slot = inst.input_slots[0];
            let consume_input = inst.last_use.first().copied().unwrap_or(false);
            let input = tensor_value_for_lazy_view(exec, slots, input_slot, consume_input)?;
            input.slice_view(config).map_err(Error::TensorRuntime)?
        }
        _ => return Ok(false),
    };
    slots[inst.output_slots[0]] = Some(ExecSlot::Value(output));
    Ok(true)
}

fn tensor_value_for_lazy_view<'input>(
    exec: &mut dyn BackendSession,
    slots: &mut [Option<ExecSlot<'input>>],
    slot: usize,
    consume: bool,
) -> Result<TensorValue> {
    if consume {
        let value = take_slot(slots, slot, "input", "tensor_value_for_lazy_view")?;
        return value.into_value(exec);
    }

    let slot = checked_slot_index(slot, slots.len(), "input", "tensor_value_for_lazy_view")?;
    match slots[slot].take() {
        Some(ExecSlot::Owned(tensor)) => {
            let value = TensorValue::from_tensor(tensor);
            let output = value.duplicate()?;
            slots[slot] = Some(ExecSlot::Value(value));
            Ok(output)
        }
        Some(ExecSlot::Value(value)) => {
            let output = value.duplicate()?;
            slots[slot] = Some(ExecSlot::Value(value));
            Ok(output)
        }
        Some(ExecSlot::Read(read)) => {
            let output = match read.clone() {
                TensorRead::Tensor(tensor) => TensorValue::from_tensor(tensor.duplicate()?),
                read @ TensorRead::View(_) => {
                    TensorValue::from_tensor(exec.to_contiguous_read(read)?)
                }
            };
            slots[slot] = Some(ExecSlot::Read(read));
            Ok(output)
        }
        None => Err(TensorError::MissingValue { slot }.into()),
    }
}

pub(crate) fn is_host_instruction(inst: &ExecInstruction) -> bool {
    dispatch::is_host_op(&inst.op)
}

pub(crate) fn is_ffi_instruction(inst: &ExecInstruction) -> bool {
    dispatch::is_ffi_op(&inst.op)
}

pub(crate) fn is_exec_session_ffi_instruction(inst: &ExecInstruction) -> bool {
    dispatch::is_exec_session_ffi_op(&inst.op)
}

pub(crate) fn resolve_tensor_shape_exprs<'input>(
    slots: &[Option<ExecSlot<'input>>],
    input_slots: &[usize],
    exprs: &[DimExpr],
) -> Result<Vec<usize>> {
    let mut input_shapes = Vec::with_capacity(input_slots.len());
    for &slot in input_slots {
        input_shapes.push(slot_ref(slots, slot, "input", "resolve_tensor_shape_exprs")?.shape());
    }
    DimExpr::eval_all(exprs, &input_shapes).map_err(|cause| Error::ShapeExpressionEvaluation {
        expression: format!("{} expression(s)", exprs.len()),
        cause: cause.into(),
    })
}

#[cfg(test)]
pub(crate) fn validate_shape_guards(
    program: &ExecProgram,
    input_shapes: &[&[usize]],
) -> Result<()> {
    for guard in &program.shape_guards {
        guard.evaluate(input_shapes)?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn eval_exec_ir_unsegmented_with_cache<B: TensorBackend + 'static>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    let mut slots = Vec::new();
    eval_exec_ir_unsegmented_with_cache_and_workspace(backend, program, inputs, &mut slots, None)
}

#[cfg(test)]
pub(crate) fn eval_exec_ir_unsegmented_with_cache_and_workspace<
    'input,
    B: TensorBackend + 'static,
>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    slots: &mut Vec<Option<ExecSlot<'input>>>,
    extension_dispatch: Option<&mut ExtensionExecutionDispatch<'_>>,
) -> Result<Vec<Tensor>> {
    validate_exec_input_count(program, inputs.len(), "initialize_exec_slots_in")?;
    let input_shapes: SmallVec<[&[usize]; 8]> = inputs.iter().map(Tensor::shape).collect();
    validate_shape_guards(program, &input_shapes)?;
    drop(input_shapes);
    let inputs = inputs.into_iter().map(ExecSlot::Owned).collect();
    eval_exec_ir_unsegmented_slots_with_cache_and_workspace(
        backend,
        program,
        inputs,
        slots,
        extension_dispatch,
    )
}

fn validate_exec_input_count(program: &ExecProgram, actual: usize, caller: &str) -> Result<()> {
    let expected = program.input_slots.len();
    if actual != expected {
        return Err(invalid_compiled_graph(format!(
            "{caller}: expected {expected} inputs, got {actual}"
        )));
    }
    Ok(())
}

pub(crate) fn eval_exec_ir_unsegmented_slots_with_cache_and_workspace<
    'input,
    B: TensorBackend + 'static,
>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<ExecSlot<'input>>,
    slots: &mut Vec<Option<ExecSlot<'input>>>,
    mut extension_dispatch: Option<&mut ExtensionExecutionDispatch<'_>>,
) -> Result<Vec<Tensor>> {
    let result = (|| {
        validate_exec_program(program, "unsegmented executor")?;
        initialize_exec_slots_in(program, inputs, slots)?;

        for inst in &program.instructions {
            if is_host_instruction(inst) {
                execute_host_instruction(backend, slots, inst)?;
            } else if is_ffi_instruction(inst) {
                execute_ffi_instruction(
                    backend,
                    slots,
                    inst,
                    DispatchMode::Unsegmented,
                    extension_dispatch.as_deref_mut(),
                )?;
            } else {
                let result =
                    backend.with_backend_session(|exec| execute_backend_op(exec, slots, inst))?;
                slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
            }
            reclaim_last_use_inputs_backend(slots, inst, backend);
        }

        backend.with_backend_session(|exec| collect_outputs_from(program, slots, exec))
    })();
    slots.clear();
    result
}

pub(crate) fn eval_exec_ir_unsegmented_slot_values_with_cache_and_workspace<
    'input,
    B: TensorBackend + 'static,
>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<ExecSlot<'input>>,
    slots: &mut Vec<Option<ExecSlot<'input>>>,
    backend_cache: &mut B::RuntimeCache,
    mut extension_dispatch: Option<&mut ExtensionExecutionDispatch<'_>>,
) -> Result<Vec<TensorValue>> {
    let result = (|| {
        validate_exec_program(program, "unsegmented value executor")?;
        initialize_exec_slots_in(program, inputs, slots)?;
        let terminal_slots = terminal_output_slots(program);

        for (inst_idx, inst) in program.instructions.iter().enumerate() {
            if backend.with_backend_session(|exec| {
                try_execute_terminal_value_instruction(exec, slots, inst, &terminal_slots)
            })? {
                // Already handled as a metadata-only TensorValue.
            } else if is_host_instruction(inst) {
                execute_host_instruction(backend, slots, inst)?;
            } else if is_ffi_instruction(inst) {
                execute_ffi_instruction_cached(
                    backend,
                    backend_cache,
                    slots,
                    inst,
                    DispatchMode::Unsegmented,
                    Some(inst_idx),
                    extension_dispatch.as_deref_mut(),
                )?;
            } else {
                let result =
                    backend.with_backend_session(|exec| execute_backend_op(exec, slots, inst))?;
                slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
            }
            reclaim_last_use_inputs_backend(slots, inst, backend);
        }

        backend.with_backend_session(|exec| collect_output_values_from(program, slots, exec))
    })();
    slots.clear();
    result
}

pub(crate) fn can_run_in_single_exec_session_with_extensions(
    program: &ExecProgram,
    extension_dispatch: Option<&ExtensionExecutionDispatch<'_>>,
) -> bool {
    program
        .instructions
        .iter()
        .all(|inst| is_session_compatible_instruction(inst, extension_dispatch))
}

pub(crate) fn has_session_capable_extension(
    program: &ExecProgram,
    extension_dispatch: Option<&ExtensionExecutionDispatch<'_>>,
) -> bool {
    program.instructions.iter().any(|inst| {
        matches!(inst.op, ExecOp::Extension(_))
            && is_session_compatible_instruction(inst, extension_dispatch)
    })
}

pub(crate) fn is_session_compatible_instruction(
    inst: &ExecInstruction,
    extension_dispatch: Option<&ExtensionExecutionDispatch<'_>>,
) -> bool {
    if !is_ffi_instruction(inst) || is_exec_session_ffi_instruction(inst) {
        return true;
    }
    let ExecOp::Extension(_) = &inst.op else {
        return false;
    };
    let Some(dispatch) = extension_dispatch else {
        return false;
    };
    let Some(operation_index) = inst.semantic_operation_index else {
        return false;
    };
    dispatch
        .operations
        .get(operation_index)
        .and_then(PreparedOperationPlan::executor)
        .is_some_and(|executor| executor.supports_session())
}

pub(crate) fn eval_exec_ir_single_session_slots_with_workspace<'input, B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<ExecSlot<'input>>,
    slots: &mut Vec<Option<ExecSlot<'input>>>,
    backend_cache: &mut B::RuntimeCache,
    mut extension_dispatch: Option<&mut ExtensionExecutionDispatch<'_>>,
) -> Result<Vec<Tensor>> {
    let result = (|| {
        validate_exec_program(program, "single-session executor")?;
        initialize_exec_slots_in(program, inputs, slots)?;

        backend.with_backend_session_cached(backend_cache, |exec| {
            for (inst_idx, inst) in program.instructions.iter().enumerate() {
                if is_host_instruction(inst) {
                    execute_host_instruction_exec(exec, slots, inst)?;
                } else if is_ffi_instruction(inst) {
                    execute_ffi_instruction_exec(
                        exec,
                        slots,
                        inst,
                        Some(inst_idx),
                        extension_dispatch.as_deref_mut(),
                    )?;
                } else {
                    let result = execute_backend_op(exec, slots, inst)?;
                    slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
                }
                reclaim_last_use_inputs_exec(slots, inst, exec);
            }
            collect_outputs_from(program, slots, exec)
        })
    })();
    slots.clear();
    result
}

pub(crate) fn execute_backend_op<'input>(
    exec: &mut dyn BackendSession,
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    dispatch::execute_backend_dispatch(exec, slots, inst)
}

pub(crate) fn execute_host_instruction<'input, B: TensorBackend>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
) -> Result<()> {
    dispatch::execute_host_dispatch(backend, slots, inst)
}

pub(crate) fn execute_host_instruction_exec<'input>(
    exec: &mut dyn BackendSession,
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
) -> Result<()> {
    dispatch::execute_host_dispatch(exec, slots, inst)
}

pub(crate) fn execute_ffi_instruction<'input, B: TensorBackend + 'static>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    mode: DispatchMode,
    extension_dispatch: Option<&mut ExtensionExecutionDispatch<'_>>,
) -> Result<()> {
    dispatch::execute_ffi_dispatch(backend, slots, inst, mode, extension_dispatch)
}

pub(crate) fn execute_ffi_instruction_cached<'input, B: TensorBackend + 'static>(
    backend: &mut B,
    backend_cache: &mut B::RuntimeCache,
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    mode: DispatchMode,
    cache_slot: Option<usize>,
    extension_dispatch: Option<&mut ExtensionExecutionDispatch<'_>>,
) -> Result<()> {
    match &inst.op {
        ExecOp::DotGeneral(config) => {
            let result = backend.dot_general_read_cached(
                backend_cache,
                cache_slot,
                get_read(slots, &inst.input_slots, 0)?,
                get_read(slots, &inst.input_slots, 1)?,
                config,
            )?;
            slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
            Ok(())
        }
        ExecOp::DotGeneralWithConj {
            config,
            lhs_conj,
            rhs_conj,
        } => {
            let result = backend.dot_general_with_conj_read_cached(
                backend_cache,
                cache_slot,
                get_read(slots, &inst.input_slots, 0)?,
                get_read(slots, &inst.input_slots, 1)?,
                config,
                *lhs_conj,
                *rhs_conj,
            )?;
            slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
            Ok(())
        }
        _ => execute_ffi_instruction(backend, slots, inst, mode, extension_dispatch),
    }
}

pub(crate) fn execute_ffi_instruction_exec<'input>(
    exec: &mut dyn BackendSession,
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    cache_slot: Option<usize>,
    extension_dispatch: Option<&mut ExtensionExecutionDispatch<'_>>,
) -> Result<()> {
    match &inst.op {
        ExecOp::DotGeneral(config) => {
            let result = exec.dot_general_read_cached(
                cache_slot,
                get_read(slots, &inst.input_slots, 0)?,
                get_read(slots, &inst.input_slots, 1)?,
                config,
            )?;
            slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
        }
        ExecOp::DotGeneralWithConj {
            config,
            lhs_conj,
            rhs_conj,
        } => {
            let result = exec.dot_general_with_conj_read_cached(
                cache_slot,
                get_read(slots, &inst.input_slots, 0)?,
                get_read(slots, &inst.input_slots, 1)?,
                config,
                *lhs_conj,
                *rhs_conj,
            )?;
            slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
        }
        ExecOp::Extension(ext) => {
            let Some(extension_dispatch) = extension_dispatch else {
                return Err(Error::validation(
                    "extension",
                    ErrorPhase::Execution,
                    ValidationError::InvalidArgument {
                        argument: "prepared_operation",
                        message: format!(
                            "extension instruction for family_id {:?} requires a prepared operation execution handle",
                            ext.family_id()
                        ),
                    },
                ));
            };
            let outputs = execute_prepared_extension_instruction_in_session(
                exec,
                slots,
                inst,
                ext.as_ref(),
                extension_dispatch.operations,
                extension_dispatch.caches,
            )?;
            if outputs.len() != inst.output_slots.len() {
                return Err(Error::validation(
                    "extension",
                    ErrorPhase::Execution,
                    ValidationError::InvalidArgument {
                        argument: "outputs",
                        message: format!(
                            "family_id={:?}: extension runtime returned {} outputs for {} slots",
                            ext.family_id(),
                            outputs.len(),
                            inst.output_slots.len()
                        ),
                    },
                ));
            }
            for (slot, tensor) in inst.output_slots.iter().copied().zip(outputs) {
                slots[slot] = Some(ExecSlot::Owned(tensor));
            }
        }
        other => {
            return Err(Error::Internal(format!(
                "unsupported single-session FFI op: {other:?}"
            )))
        }
    }
    Ok(())
}

fn execute_prepared_extension_instruction_in_session<'input>(
    session: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    ext: &dyn ExtensionOp,
    operations: &[PreparedOperationPlan],
    caches: &mut ExtensionCacheStore,
) -> Result<Vec<Tensor>> {
    let operation_index = inst.semantic_operation_index.ok_or_else(|| {
        Error::runtime_state(
            "extension",
            ErrorPhase::Execution,
            format!(
                "extension instruction for family_id {:?} is missing semantic operation origin",
                ext.family_id()
            ),
        )
    })?;
    let operation = operations.get(operation_index).ok_or_else(|| {
        Error::runtime_state(
            "extension",
            ErrorPhase::Execution,
            format!(
                "extension instruction for family_id {:?} references semantic operation {operation_index}, but prepared program has {} operations",
                ext.family_id(),
                operations.len()
            ),
        )
    })?;
    let executor = operation.executor().ok_or_else(|| {
        Error::runtime_state_source(
            "extension",
            ErrorPhase::Execution,
            MissingPreparedOperationExecutorError {
                family_id: ext.family_id(),
                operation_index,
            },
        )
    })?;
    if !executor.supports_session() {
        return Err(Error::unsupported(
            "extension",
            ErrorPhase::Execution,
            format!(
                "prepared extension family_id {:?} does not support the scheduler-owned backend session",
                ext.family_id()
            ),
        ));
    }
    let inputs = collect_tensor_reads(slots, &inst.input_slots)?;
    executor.execute_in_session(session, caches, &inputs)
}

/// Dispatch a compiled `ExecOp::Extension` instruction.
///
/// Per spec Section 8, the compiled pipeline owns metadata lowering and
/// input resolution; the extension runtime owns the actual forward computation.
/// Errors are wrapped in [`Error::BackendFailure`] with `op: "extension"`
/// and the `family_id` included in the message.
fn execute_extension_instruction<'input, B: TensorBackend + 'static>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    ext: &dyn ExtensionOp,
    extension_dispatch: Option<&mut ExtensionExecutionDispatch<'_>>,
) -> Result<()> {
    let Some(extension_dispatch) = extension_dispatch else {
        return Err(Error::validation(
            "extension",
            ErrorPhase::Execution,
            ValidationError::InvalidArgument {
                argument: "prepared_operation",
                message: format!(
                    "extension instruction for family_id {:?} requires a prepared operation execution handle",
                    ext.family_id()
                ),
            },
        ));
    };
    let outputs = execute_prepared_extension_instruction(
        backend,
        slots,
        inst,
        ext,
        extension_dispatch.operations,
        extension_dispatch.caches,
    )?;
    if outputs.len() != inst.output_slots.len() {
        return Err(Error::validation(
            "extension",
            ErrorPhase::Execution,
            ValidationError::InvalidArgument {
                argument: "outputs",
                message: format!(
                    "family_id={:?}: extension runtime returned {} outputs for {} slots",
                    ext.family_id(),
                    outputs.len(),
                    inst.output_slots.len()
                ),
            },
        ));
    }
    for (slot, tensor) in inst.output_slots.iter().copied().zip(outputs) {
        slots[slot] = Some(ExecSlot::Owned(tensor));
    }
    Ok(())
}

fn execute_prepared_extension_instruction<'input, B: TensorBackend + 'static>(
    backend: &mut B,
    slots: &[Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    ext: &dyn ExtensionOp,
    operations: &[PreparedOperationPlan],
    caches: &mut ExtensionCacheStore,
) -> Result<Vec<Tensor>> {
    let operation_index = inst.semantic_operation_index.ok_or_else(|| {
        Error::runtime_state(
            "extension",
            ErrorPhase::Execution,
            format!(
                "extension instruction for family_id {:?} is missing semantic operation origin",
                ext.family_id()
            ),
        )
    })?;
    let operation = operations.get(operation_index).ok_or_else(|| {
        Error::runtime_state(
            "extension",
            ErrorPhase::Execution,
            format!(
                "extension instruction for family_id {:?} references semantic operation {operation_index}, but prepared program has {} operations",
                ext.family_id(),
                operations.len()
            ),
        )
    })?;
    let mut context = ErasedExecutionContext::new(backend);
    if operation.binding().context_identity() != context.identity() {
        return Err(Error::runtime_state(
            "extension",
            ErrorPhase::Execution,
            format!(
                "prepared operation context {:?} does not match execution context {:?}",
                operation.binding().context_identity(),
                context.identity()
            ),
        ));
    }
    let executor = operation.executor().ok_or_else(|| {
        Error::runtime_state_source(
            "extension",
            ErrorPhase::Execution,
            MissingPreparedOperationExecutorError {
                family_id: ext.family_id(),
                operation_index,
            },
        )
    })?;
    let inputs = collect_tensor_reads(slots, &inst.input_slots)?;
    executor.execute(&mut context, caches, &inputs)
}

fn collect_tensor_reads<'slot, 'input>(
    slots: &'slot [Option<ExecSlot<'input>>],
    input_slots: &[usize],
) -> Result<Vec<TensorRead<'slot>>>
where
    'input: 'slot,
{
    let mut inputs = Vec::with_capacity(input_slots.len());
    for idx in 0..input_slots.len() {
        inputs.push(get_read(slots, input_slots, idx)?);
    }
    Ok(inputs)
}

fn collect_tensor_refs<'slot, 'input>(
    slots: &'slot [Option<ExecSlot<'input>>],
    input_slots: &[usize],
) -> Result<Vec<&'slot Tensor>>
where
    'input: 'slot,
{
    let mut inputs = Vec::with_capacity(input_slots.len());
    for &slot in input_slots {
        inputs.push(slot_ref(slots, slot, "input", "extension")?.as_tensor("extension")?);
    }
    Ok(inputs)
}

pub(crate) fn constant_tensor(dtype: DType, bytes: &[u8]) -> Result<Tensor> {
    match dtype {
        DType::F64 => Ok(Tensor::F64(TypedTensor::from_vec_col_major(
            vec![],
            vec![f64::from_le_bytes(exact_bytes::<8>(dtype, bytes)?)],
        )?)),
        DType::F32 => Ok(Tensor::F32(TypedTensor::from_vec_col_major(
            vec![],
            vec![f32::from_le_bytes(exact_bytes::<4>(dtype, bytes)?)],
        )?)),
        DType::I32 => Ok(Tensor::I32(TypedTensor::from_vec_col_major(
            vec![],
            vec![i32::from_le_bytes(exact_bytes::<4>(dtype, bytes)?)],
        )?)),
        DType::I64 => Ok(Tensor::I64(TypedTensor::from_vec_col_major(
            vec![],
            vec![i64::from_le_bytes(exact_bytes::<8>(dtype, bytes)?)],
        )?)),
        DType::Bool => Ok(Tensor::Bool(TypedTensor::from_vec_col_major(
            vec![],
            vec![exact_bytes::<1>(dtype, bytes)?[0] != 0],
        )?)),
        DType::C64 => {
            let data = exact_bytes::<16>(dtype, bytes)?;
            let mut re_bytes = [0u8; 8];
            let mut im_bytes = [0u8; 8];
            re_bytes.copy_from_slice(&data[..8]);
            im_bytes.copy_from_slice(&data[8..]);
            let re = f64::from_le_bytes(re_bytes);
            let im = f64::from_le_bytes(im_bytes);
            Ok(Tensor::C64(TypedTensor::from_vec_col_major(
                vec![],
                vec![Complex64::new(re, im)],
            )?))
        }
        DType::C32 => {
            let data = exact_bytes::<8>(dtype, bytes)?;
            let mut re_bytes = [0u8; 4];
            let mut im_bytes = [0u8; 4];
            re_bytes.copy_from_slice(&data[..4]);
            im_bytes.copy_from_slice(&data[4..]);
            let re = f32::from_le_bytes(re_bytes);
            let im = f32::from_le_bytes(im_bytes);
            Ok(Tensor::C32(TypedTensor::from_vec_col_major(
                vec![],
                vec![Complex32::new(re, im)],
            )?))
        }
    }
}

fn exact_bytes<const N: usize>(dtype: DType, bytes: &[u8]) -> Result<[u8; N]> {
    if bytes.len() != N {
        return Err(invalid_compiled_graph(format!(
            "constant {dtype:?} expected {N} bytes, got {}",
            bytes.len()
        )));
    }
    let mut out = [0u8; N];
    out.copy_from_slice(bytes);
    Ok(out)
}

pub(crate) fn reclaim_last_use_inputs_exec<'input>(
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    exec: &mut dyn BackendSession,
) {
    for (i, &is_last) in inst.last_use.iter().enumerate() {
        if is_last {
            if let Some(slot) = slots[inst.input_slots[i]].take() {
                reclaim_exec_slot_with_session(slot, exec);
            }
        }
    }
}

pub(crate) fn reclaim_last_use_inputs_backend<'input, B: TensorBackend>(
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    backend: &mut B,
) {
    for (i, &is_last) in inst.last_use.iter().enumerate() {
        if is_last {
            if let Some(slot) = slots[inst.input_slots[i]].take() {
                reclaim_exec_slot_with_backend(slot, backend);
            }
        }
    }
}

fn reclaim_exec_slot_with_session(slot: ExecSlot<'_>, exec: &mut dyn BackendSession) {
    match slot {
        ExecSlot::Owned(tensor) => exec.reclaim_buffer(tensor),
        ExecSlot::Value(value) => {
            if let Ok(tensor) = value.into_tensor() {
                exec.reclaim_buffer(tensor);
            }
        }
        ExecSlot::Read(_) => {}
    }
}

fn reclaim_exec_slot_with_backend<B: TensorBackend>(slot: ExecSlot<'_>, backend: &mut B) {
    match slot {
        ExecSlot::Owned(tensor) => backend.reclaim_buffer(tensor),
        ExecSlot::Value(value) => {
            if let Ok(tensor) = value.into_tensor() {
                backend.reclaim_buffer(tensor);
            }
        }
        ExecSlot::Read(_) => {}
    }
}

#[cfg(test)]
mod tests;
