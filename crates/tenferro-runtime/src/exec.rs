use std::sync::Arc;

use crate::error::{Error, ErrorPhase, Result};
use num_complex::{Complex32, Complex64};
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::{dim_expr::DimExpr, ShapeExtent};
use tenferro_tensor::backend::ElementwiseFusionOp;
use tenferro_tensor::Error as TensorError;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, Engine, GatherConfig, PadConfig, ScatterConfig,
    SliceConfig, StridedSliceSpec, Tensor, TensorRead, TensorView, TypedTensor, ValidationError,
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
    /// Metadata-only transforms applied to an input when this instruction
    /// reads it, as `(input index, view)` pairs in application order (#2226).
    ///
    /// A folded `Transpose`, `Reshape`, `BroadcastInDim` or `Slice` is not a
    /// scheduled command: its consumer derives the view from the source slot at
    /// the moment it reads the operand. Empty for every instruction that reads
    /// its inputs as they are, which keeps the common path to one branch.
    pub input_views: Vec<(usize, InputView)>,
}

/// A metadata-only transform of one operand.
///
/// This is the single definition of which execution ops are views: the
/// Read-slot path (`view_backend_op`) and the folded consumer path
/// (`input_views`) both derive their view through [`InputView::apply`].
#[derive(Clone, Debug, PartialEq)]
pub enum InputView {
    Transpose {
        perm: Vec<usize>,
    },
    Reshape {
        shape: Vec<DimExpr>,
    },
    BroadcastInDim {
        shape: Vec<DimExpr>,
        dims: Vec<usize>,
    },
    Slice(SliceConfig),
}

impl InputView {
    /// The view `op` describes, or `None` when `op` moves data.
    pub(crate) fn from_op(op: &ExecOp) -> Option<Self> {
        match op {
            ExecOp::Transpose { perm } => Some(Self::Transpose { perm: perm.clone() }),
            ExecOp::Reshape { shape } => Some(Self::Reshape {
                shape: shape.clone(),
            }),
            ExecOp::BroadcastInDim { shape, dims } => Some(Self::BroadcastInDim {
                shape: shape.clone(),
                dims: dims.clone(),
            }),
            ExecOp::Slice(config) => Some(Self::Slice(config.clone())),
            _ => None,
        }
    }

    /// Whether this view may need a compact source: a reshape of a strided
    /// view has no view form in general, so the fold rule only folds a
    /// `Reshape` whose source is a compact instruction result.
    pub(crate) fn requires_compact_source(&self) -> bool {
        matches!(self, Self::Reshape { .. })
    }

    /// The view of `read` this transform describes.
    ///
    /// Shape expressions are evaluated against the shape `read` has before the
    /// transform, exactly as the removed instruction evaluated them against
    /// its single input.
    pub(crate) fn apply<'a>(&self, read: TensorRead<'a>) -> Result<TensorRead<'a>> {
        // A caller-owned payload has no borrowed view.
        if let DType::External(_) = read.dtype() {
            return Err(Error::unsupported(
                "input_view",
                ErrorPhase::Execution,
                "an externally defined scalar has no metadata view",
            ));
        }
        macro_rules! typed_view_map {
            ($view:expr, $method:ident ( $($arg:expr),* )) => {
                match $view {
                    TensorView::F32(view) => TensorView::F32(view.$method($($arg),*)?),
                    TensorView::F64(view) => TensorView::F64(view.$method($($arg),*)?),
                    TensorView::I32(view) => TensorView::I32(view.$method($($arg),*)?),
                    TensorView::I64(view) => TensorView::I64(view.$method($($arg),*)?),
                    TensorView::Bool(view) => TensorView::Bool(view.$method($($arg),*)?),
                    TensorView::C32(view) => TensorView::C32(view.$method($($arg),*)?),
                    TensorView::C64(view) => TensorView::C64(view.$method($($arg),*)?),
                }
            };
        }
        let view = match self {
            Self::Transpose { perm } => typed_view_map!(read.tensor_view(), transpose_view(perm)),
            Self::Reshape { shape } => {
                let shape = view_shape(shape, read.shape())?;
                typed_view_map!(read.tensor_view(), reshape_view(&shape))
            }
            Self::BroadcastInDim { shape, dims } => {
                let shape = view_shape(shape, read.shape())?;
                return Ok(read.broadcast_in_dim_view(&shape, dims)?);
            }
            Self::Slice(config) => {
                if config.starts.len() != config.limits.len()
                    || config.starts.len() != config.strides.len()
                {
                    return Err(Error::validation(
                        "slice",
                        ErrorPhase::Execution,
                        ValidationError::InvalidArgument {
                            argument: "config",
                            message: "slice starts, limits and strides differ in length".into(),
                        },
                    ));
                }
                let mut specs = Vec::with_capacity(config.starts.len());
                for ((&start, &limit), &step) in config
                    .starts
                    .iter()
                    .zip(&config.limits)
                    .zip(&config.strides)
                {
                    specs.push(StridedSliceSpec::new(
                        slice_bound(start)?,
                        Some(slice_bound(limit)?),
                        slice_bound(step)?,
                    ));
                }
                typed_view_map!(read.tensor_view(), slice_view(&specs))
            }
        };
        Ok(TensorRead::from_view(view))
    }
}

/// Evaluate a view's shape expressions against its source shape.
fn view_shape(exprs: &[DimExpr], source: &[usize]) -> Result<Vec<usize>> {
    DimExpr::eval_all(exprs, &[source]).map_err(|cause| Error::ShapeExpressionEvaluation {
        expression: format!("{} expression(s)", exprs.len()),
        cause: cause.into(),
    })
}

fn slice_bound(value: usize) -> Result<isize> {
    isize::try_from(value).map_err(|_| {
        Error::validation(
            "slice",
            ErrorPhase::Execution,
            ValidationError::InvalidArgument {
                argument: "config",
                message: format!("slice bound {value} does not fit the address space"),
            },
        )
    })
}

/// Whether `inst` reads input `idx` through a folded view.
fn has_input_view(inst: &ExecInstruction, idx: usize) -> bool {
    inst.input_views.iter().any(|(input, _)| *input == idx)
}

fn owned_input_has_view(inst: &ExecInstruction, idx: usize, caller: &'static str) -> Error {
    Error::Internal(format!(
        "{caller}: input {idx} of {:?} carries a folded view, but the operation reads owned inputs",
        inst.op
    ))
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

    #[cfg(test)]
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
        for (input, _) in &inst.input_views {
            if *input >= inst.input_slots.len() {
                return Err(invalid_compiled_graph(format!(
                    "{caller}: instruction {inst_idx} folds a view onto input {input}, but has {} input slots",
                    inst.input_slots.len()
                )));
            }
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

/// The owned tensor in input `idx`.
///
/// An input that carries a folded view has no owned form: the fold rule only
/// folds into consumers that read views, so reaching here with one is a
/// classification defect, reported typed instead of reading the source value.
pub(crate) fn get<'slot, 'input>(
    slots: &'slot [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    idx: usize,
) -> Result<&'slot Tensor>
where
    'input: 'slot,
{
    if has_input_view(inst, idx) {
        return Err(owned_input_has_view(inst, idx, "get"));
    }
    let slot = checked_input_slot(&inst.input_slots, idx, "get")?;
    slot_ref(slots, slot, "input", "get").and_then(|value| value.as_tensor("get"))
}

pub(crate) fn ensure_owned<'input>(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    idx: usize,
) -> Result<()> {
    if has_input_view(inst, idx) {
        return Err(owned_input_has_view(inst, idx, "ensure_owned"));
    }
    let input_slots = &inst.input_slots;
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

/// Input `idx` as a read, with its folded views applied.
pub(crate) fn get_read<'slot, 'input>(
    slots: &'slot [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    idx: usize,
) -> Result<TensorRead<'slot>>
where
    'input: 'slot,
{
    let slot = checked_input_slot(&inst.input_slots, idx, "get_read")?;
    let mut read = slot_ref(slots, slot, "input", "get_read")?.as_read();
    for (input, view) in &inst.input_views {
        if *input == idx {
            read = view.apply(read)?;
        }
    }
    Ok(read)
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

/// Whether the executor of `op` reads input `input_index` through a view.
pub(crate) fn reads_input_view(op: &ExecOp, input_index: usize) -> bool {
    dispatch::reads_input_view(op, input_index)
}

pub(crate) fn is_exec_session_ffi_instruction(inst: &ExecInstruction) -> bool {
    dispatch::is_exec_session_ffi_op(&inst.op)
}

/// Evaluate `exprs` against the shapes `inst` reads, folded views included.
pub(crate) fn resolve_tensor_shape_exprs<'input>(
    slots: &[Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
    exprs: &[DimExpr],
) -> Result<Vec<usize>> {
    let reads = collect_tensor_reads(slots, inst)?;
    let input_shapes: Vec<&[usize]> = reads.iter().map(TensorRead::shape).collect();
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

/// Execute a backend instruction as a borrowed view when the input slot is
/// borrowed and the operation is metadata-only.
pub(crate) fn execute_backend_view_op<'input>(
    slots: &[Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
) -> Option<ExecSlot<'input>> {
    dispatch::view_backend_op(slots, inst)
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
                get_read(slots, inst, 0)?,
                get_read(slots, inst, 1)?,
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
                get_read(slots, inst, 0)?,
                get_read(slots, inst, 1)?,
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
    let inputs = collect_tensor_reads(slots, inst)?;
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
    let inputs = collect_tensor_reads(slots, inst)?;
    executor.execute(&mut context, caches, &inputs)
}

pub(super) fn collect_tensor_reads<'slot, 'input>(
    slots: &'slot [Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
) -> Result<Vec<TensorRead<'slot>>>
where
    'input: 'slot,
{
    let mut inputs = Vec::with_capacity(inst.input_slots.len());
    for idx in 0..inst.input_slots.len() {
        inputs.push(get_read(slots, inst, idx)?);
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
