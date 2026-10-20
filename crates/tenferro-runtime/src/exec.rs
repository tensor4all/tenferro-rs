use std::sync::Arc;

use crate::error::{Error, ErrorPhase, Result};
use num_complex::{Complex32, Complex64};
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::{dim_expr::DimExpr, ShapeExtent};
use tenferro_tensor::backend::ElementwiseFusionOp;
use tenferro_tensor::Error as TensorError;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, Engine, GatherConfig, PadConfig, ScatterConfig,
    SliceConfig, Tensor, TensorRead, TypedTensor, ValidationError,
};
use tinyvec::TinyVec;

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

pub type ExecOutputShapes = TinyVec<[Vec<DimExpr>; 1]>;
pub type ExecOutputExtents = TinyVec<[Vec<ShapeExtent<DimExpr>>; 1]>;

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

// INVARIANT: execution slots carry either a move-only tensor or a borrowed
// read; the enum is the single dispatch boundary and is not cloned.
#[allow(clippy::large_enum_variant)]
pub(crate) enum ExecSlot<'a> {
    Owned(Tensor),
    Read(TensorRead<'a>),
}

impl<'a> ExecSlot<'a> {
    pub(crate) fn as_read<'slot>(&'slot self) -> TensorRead<'slot>
    where
        'a: 'slot,
    {
        match self {
            Self::Owned(tensor) => TensorRead::from_tensor(tensor),
            Self::Read(read) => read.clone(),
        }
    }

    pub(crate) fn as_tensor<'slot>(&'slot self, op: &'static str) -> Result<&'slot Tensor>
    where
        'a: 'slot,
    {
        match self {
            Self::Owned(tensor) => Ok(tensor),
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
            Self::Read(read) => read.shape(),
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
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'input>>],
    input_slots: &[usize],
    idx: usize,
) -> Result<()> {
    let slot = checked_input_slot(input_slots, idx, "ensure_owned")?;
    let needs_materialization = matches!(
        slot_ref(slots, slot, "input", "ensure_owned")?,
        ExecSlot::Read(TensorRead::View(_))
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

fn validate_exec_input_count(program: &ExecProgram, actual: usize, caller: &str) -> Result<()> {
    let expected = program.input_slots.len();
    if actual != expected {
        return Err(invalid_compiled_graph(format!(
            "{caller}: expected {expected} inputs, got {actual}"
        )));
    }
    Ok(())
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

pub(crate) fn execute_backend_op<'input>(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    dispatch::execute_backend_dispatch(exec, slots, inst)
}

pub(crate) fn execute_host_instruction_exec<'input>(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
) -> Result<()> {
    dispatch::execute_host_dispatch(exec, slots, inst)
}

/// True when `inst` cannot run inside a session and needs the extension owner
/// fallback, which forms the session itself.
///
/// `is_session_compatible_instruction` returns `true` for every non-FFI op and
/// for the session-capable FFI ops, so a `false` result can only come from an
/// extension whose `supports_session()` is false.
pub(crate) fn needs_owner_extension_fallback(
    inst: &ExecInstruction,
    extension_dispatch: Option<&ExtensionExecutionDispatch<'_>>,
) -> bool {
    !is_session_compatible_instruction(inst, extension_dispatch)
}

/// Run one non-session-compatible extension instruction through its owner entry.
///
/// The entry is the extension runtime's own `execute`, which forms the session
/// and runs the operation through it; no operation runs on the owner.
pub(crate) fn execute_owner_extension_fallback<'input>(
    engine: &mut dyn std::any::Any,
    identity: crate::runtime::ExecutionContextIdentity,
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    extension_dispatch: Option<&mut ExtensionExecutionDispatch<'_>>,
) -> Result<()> {
    let ExecOp::Extension(ext) = &inst.op else {
        return Err(Error::Internal(format!(
            "non-session-compatible instruction is not an extension op: {:?}",
            inst.op
        )));
    };
    execute_extension_instruction(
        engine,
        identity,
        slots,
        inst,
        ext.as_ref(),
        extension_dispatch,
    )
}

pub(crate) fn execute_ffi_instruction_exec<'input>(
    exec: &mut dyn Engine,
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
    session: &mut dyn Engine,
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
fn execute_extension_instruction<'input>(
    engine: &mut dyn std::any::Any,
    identity: crate::runtime::ExecutionContextIdentity,
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
        engine,
        identity,
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

fn execute_prepared_extension_instruction<'input>(
    engine: &mut dyn std::any::Any,
    identity: crate::runtime::ExecutionContextIdentity,
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
    let mut context = ErasedExecutionContext::from_dyn(identity, engine);
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
        DType::F64 => Ok(Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(
            vec![],
            vec![f64::from_le_bytes(exact_bytes::<8>(dtype, bytes)?)],
        )?)),
        DType::F32 => Ok(Tensor::from_typed::<f32>(TypedTensor::from_vec_col_major(
            vec![],
            vec![f32::from_le_bytes(exact_bytes::<4>(dtype, bytes)?)],
        )?)),
        DType::I32 => Ok(Tensor::from_typed::<i32>(TypedTensor::from_vec_col_major(
            vec![],
            vec![i32::from_le_bytes(exact_bytes::<4>(dtype, bytes)?)],
        )?)),
        DType::I64 => Ok(Tensor::from_typed::<i64>(TypedTensor::from_vec_col_major(
            vec![],
            vec![i64::from_le_bytes(exact_bytes::<8>(dtype, bytes)?)],
        )?)),
        DType::Bool => Ok(Tensor::from_typed::<bool>(TypedTensor::from_vec_col_major(
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
            Ok(Tensor::from_typed::<tenferro_tensor::Complex64>(
                TypedTensor::from_vec_col_major(vec![], vec![Complex64::new(re, im)])?,
            ))
        }
        DType::C32 => {
            let data = exact_bytes::<8>(dtype, bytes)?;
            let mut re_bytes = [0u8; 4];
            let mut im_bytes = [0u8; 4];
            re_bytes.copy_from_slice(&data[..4]);
            im_bytes.copy_from_slice(&data[4..]);
            let re = f32::from_le_bytes(re_bytes);
            let im = f32::from_le_bytes(im_bytes);
            Ok(Tensor::from_typed::<tenferro_tensor::Complex32>(
                TypedTensor::from_vec_col_major(vec![], vec![Complex32::new(re, im)])?,
            ))
        }
        // An externally defined scalar has no runtime constant representation, so
        // the compiled program rejects it rather than decoding a guessed layout.
        DType::External(_) => Err(Error::invalid_argument(
            "constant_tensor",
            ErrorPhase::GraphBuild,
            "dtype",
            "an externally defined scalar has no runtime constant",
        )),
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
    exec: &mut dyn Engine,
) {
    for (i, &is_last) in inst.last_use.iter().enumerate() {
        if is_last && let Some(slot) = slots[inst.input_slots[i]].take() {
            reclaim_exec_slot_with_session(slot, exec);
        }
    }
}

fn reclaim_exec_slot_with_session(slot: ExecSlot<'_>, exec: &mut dyn Engine) {
    match slot {
        ExecSlot::Owned(tensor) => exec.reclaim_buffer(tensor),
        ExecSlot::Read(_) => {}
    }
}

/// Execute `program` instruction by instruction on `engine`.
///
/// Test-only: production execution goes through the scheduled executor. A
/// compiler or traced-surface test that holds only an [`ExecProgram`] replays
/// it here so its numerical end-to-end check keeps running on a real engine.
#[cfg(test)]
pub(crate) fn run_exec_program_for_test(
    engine: &mut dyn tenferro_tensor::Engine,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    validate_exec_program(program, "test exec program replay")?;
    let mut slots = Vec::new();
    initialize_exec_slots_in(
        program,
        inputs.into_iter().map(ExecSlot::Owned).collect(),
        &mut slots,
    )?;
    for instruction in &program.instructions {
        if is_host_instruction(instruction) {
            execute_host_instruction_exec(engine, &mut slots, instruction)?;
            continue;
        }
        if is_ffi_instruction(instruction) {
            execute_ffi_instruction_exec(engine, &mut slots, instruction, Some(0), None)?;
            continue;
        }
        let result = execute_backend_op(engine, &mut slots, instruction)?;
        slots[instruction.output_slots[0]] = Some(ExecSlot::Owned(result));
    }
    let mut outputs = Vec::with_capacity(program.output_slots.len());
    for &slot in &program.output_slots {
        let value = slots
            .get_mut(slot)
            .and_then(Option::take)
            .ok_or(tenferro_tensor::Error::MissingValue { slot })?;
        outputs.push(match value {
            ExecSlot::Owned(tensor) => tensor,
            ExecSlot::Read(read) => {
                tenferro_tensor::TensorStructural::to_contiguous_read(engine, read)?
            }
        });
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests;
