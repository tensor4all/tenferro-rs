use crate::error::{Error, Result};
use tenferro_core_ops::PrimitiveOpKind;
use tenferro_tensor::{
    Engine, GatherConfig, PadConfig, SliceConfig, Tensor, TensorBackendOps, TensorDeviceTransfer,
    TensorRead,
};

use super::{
    collect_tensor_reads, constant_tensor, ensure_owned, get, get_read, resolve_tensor_shape_exprs,
    ExecInstruction, ExecOp, ExecSlot, InputView,
};
use crate::scalar_semantics::dynamic_truncate_size;

type BackendDispatchFn =
    for<'a> fn(&mut dyn Engine, &mut [Option<ExecSlot<'_>>], &ExecInstruction) -> Result<Tensor>;

pub(super) trait HostExecution: TensorBackendOps + TensorDeviceTransfer {}

impl<T> HostExecution for T where T: TensorBackendOps + TensorDeviceTransfer + ?Sized {}

type HostDispatchFn<B> = fn(&mut B, &mut [Option<ExecSlot<'_>>], &ExecInstruction) -> Result<()>;

macro_rules! count_keys {
    ($key:ident) => {
        ()
    };
}

macro_rules! input_access {
    (read) => {
        InputAccess::Read
    };
    (owned) => {
        InputAccess::Owned
    };
}

macro_rules! define_backend_dispatch {
    ($( $key:path => $access:ident $execute:ident, )*) => {
        pub(super) static BACKEND_DISPATCH_TABLE: &[BackendDispatchEntry] = &[
            $(
                BackendDispatchEntry {
                    key: $key,
                    inputs: input_access!($access),
                    execute: $execute,
                },
            )*
        ];
    };
}

macro_rules! define_ffi_dispatch {
    ($( $key:ident => $pattern:pat, )*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub(super) enum FfiDispatchKey {
            $( $key, )*
        }

        impl FfiDispatchKey {
            #[cfg(test)]
            pub(super) const COUNT: usize = <[()]>::len(&[$(count_keys!($key)),*]);

            pub(super) fn for_op(op: &ExecOp) -> Option<Self> {
                match op {
                    $( $pattern => Some(Self::$key), )*
                    _ => None,
                }
            }
        }
    };
}

macro_rules! define_host_dispatch {
    ($( $key:ident => $pattern:pat => $execute:ident, )*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub(super) enum HostDispatchKey {
            $( $key, )*
        }

        impl HostDispatchKey {
            #[cfg(test)]
            pub(super) const COUNT: usize = <[()]>::len(&[$(count_keys!($key)),*]);
            const COUNT_FOR_TABLE: usize = <[()]>::len(&[$(count_keys!($key)),*]);

            pub(super) fn for_op(op: &ExecOp) -> Option<Self> {
                match op {
                    $( $pattern => Some(Self::$key), )*
                    _ => None,
                }
            }
        }

        fn host_dispatch_table<B: HostExecution + ?Sized>() -> [HostDispatchEntry<B>; HostDispatchKey::COUNT_FOR_TABLE] {
            [
                $(
                    HostDispatchEntry {
                        key: HostDispatchKey::$key,
                        execute: $execute::<B>,
                    },
                )*
            ]
        }
    };
}

/// How an execute function reads its inputs, declared beside it in the table.
///
/// `Read` entries take every input through `get_read`, so a folded view on an
/// input is applied as the operand is read; `Owned` entries materialize their
/// inputs (`ensure_owned` + `get`) and the fold rule never targets them. The
/// owned accessors reject a folded view typed, so a wrong marker fails loudly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum InputAccess {
    Read,
    Owned,
}

pub(super) struct BackendDispatchEntry {
    pub(super) key: PrimitiveOpKind,
    pub(super) inputs: InputAccess,
    execute: BackendDispatchFn,
}

pub(super) struct HostDispatchEntry<B: HostExecution + ?Sized> {
    pub(super) key: HostDispatchKey,
    execute: HostDispatchFn<B>,
}

impl<B: HostExecution + ?Sized> Copy for HostDispatchEntry<B> {}

impl<B: HostExecution + ?Sized> Clone for HostDispatchEntry<B> {
    fn clone(&self) -> Self {
        *self
    }
}

define_backend_dispatch! {
    PrimitiveOpKind::Transpose => read execute_transpose,
    PrimitiveOpKind::Reshape => read execute_reshape,
    PrimitiveOpKind::BroadcastInDim => read execute_broadcast_in_dim,
    PrimitiveOpKind::Convert => owned execute_convert,
    PrimitiveOpKind::ReduceSum => read execute_reduce_sum,
    PrimitiveOpKind::ReduceSumSquares => read execute_reduce_sum_squares,
    PrimitiveOpKind::ExtractDiag => read execute_extract_diag,
    PrimitiveOpKind::EmbedDiag => read execute_embed_diag,
    PrimitiveOpKind::Tril => read execute_tril,
    PrimitiveOpKind::Triu => read execute_triu,
    PrimitiveOpKind::Add => read execute_add,
    PrimitiveOpKind::Sub => read execute_subtract,
    PrimitiveOpKind::Mul => read execute_multiply,
    PrimitiveOpKind::Neg => read execute_negate,
    PrimitiveOpKind::Conj => read execute_conj,
    PrimitiveOpKind::Div => read execute_divide,
    PrimitiveOpKind::Rem => read execute_remainder,
    PrimitiveOpKind::Abs => read execute_abs,
    PrimitiveOpKind::Sign => read execute_sign,
    PrimitiveOpKind::Maximum => read execute_maximum,
    PrimitiveOpKind::Minimum => read execute_minimum,
    PrimitiveOpKind::Compare => read execute_compare,
    PrimitiveOpKind::Select => read execute_select,
    PrimitiveOpKind::Clamp => read execute_clamp,
    PrimitiveOpKind::Exp => read execute_exp,
    PrimitiveOpKind::Log => read execute_log,
    PrimitiveOpKind::Sin => read execute_sin,
    PrimitiveOpKind::Cos => read execute_cos,
    PrimitiveOpKind::Tanh => read execute_tanh,
    PrimitiveOpKind::Sqrt => read execute_sqrt,
    PrimitiveOpKind::Rsqrt => read execute_rsqrt,
    PrimitiveOpKind::Pow => read execute_pow,
    PrimitiveOpKind::Expm1 => read execute_expm1,
    PrimitiveOpKind::Log1p => read execute_log1p,
    PrimitiveOpKind::Erf => read execute_erf,
    PrimitiveOpKind::Gather => read execute_gather,
    PrimitiveOpKind::GatherDynamicSliceSizes => read execute_gather_dynamic_slice_sizes,
    PrimitiveOpKind::Scatter => read execute_scatter,
    PrimitiveOpKind::Slice => read execute_slice,
    PrimitiveOpKind::DynamicSlice => read execute_dynamic_slice,
    PrimitiveOpKind::DynamicUpdateSlice => read execute_dynamic_update_slice,
    PrimitiveOpKind::Pad => read execute_pad,
    PrimitiveOpKind::Concatenate => read execute_concatenate,
    PrimitiveOpKind::Reverse => read execute_reverse,
    PrimitiveOpKind::ReduceProd => read execute_reduce_prod,
    PrimitiveOpKind::ReduceMax => read execute_reduce_max,
    PrimitiveOpKind::ReduceMin => read execute_reduce_min,
}

define_ffi_dispatch! {
    DotGeneral => ExecOp::DotGeneral(_),
    DotGeneralWithConj => ExecOp::DotGeneralWithConj { .. },
    Extension => ExecOp::Extension(_),
}

define_host_dispatch! {
    ShapeOf => ExecOp::ShapeOf { .. } => execute_shape_of_host,
    DynamicTruncate => ExecOp::DynamicTruncate { .. } => execute_dynamic_truncate_host,
    PadToMatch => ExecOp::PadToMatch { .. } => execute_pad_to_match_host,
    Constant => ExecOp::Constant { .. } => execute_constant_host,
}

pub(super) fn backend_dispatch_entry(op: &ExecOp) -> Option<&'static BackendDispatchEntry> {
    let key = op.primitive_kind()?;
    // INVARIANT: this macro-generated table is fixed at compile time and has
    // one entry per core primitive family, so the scan is constant-bounded.
    BACKEND_DISPATCH_TABLE.iter().find(|entry| entry.key == key)
}

pub(super) fn is_host_op(op: &ExecOp) -> bool {
    HostDispatchKey::for_op(op).is_some()
}

/// Whether the executor of `op` reads input `input_index` as a `TensorRead`,
/// so a metadata-only producer can be folded onto that input (#2226).
///
/// Backend entries declare it in `BACKEND_DISPATCH_TABLE`; the FFI entries
/// (`dot_general_read_cached`, extension executors) and `ShapeOf` take reads;
/// the remaining host ops materialize their inputs.
pub(super) fn reads_input_view(op: &ExecOp, input_index: usize) -> bool {
    if let Some(entry) = backend_dispatch_entry(op) {
        return entry.inputs == InputAccess::Read;
    }
    match FfiDispatchKey::for_op(op) {
        Some(FfiDispatchKey::DotGeneral | FfiDispatchKey::DotGeneralWithConj) => {
            return input_index < 2;
        }
        Some(FfiDispatchKey::Extension) => return true,
        None => {}
    }
    matches!(HostDispatchKey::for_op(op), Some(HostDispatchKey::ShapeOf))
}

pub(super) fn is_ffi_op(op: &ExecOp) -> bool {
    FfiDispatchKey::for_op(op).is_some()
}

pub(super) fn is_exec_session_ffi_op(op: &ExecOp) -> bool {
    matches!(
        FfiDispatchKey::for_op(op),
        Some(FfiDispatchKey::DotGeneral | FfiDispatchKey::DotGeneralWithConj)
    )
}

pub(super) fn host_dispatch_entry<B: HostExecution + ?Sized>(
    op: &ExecOp,
) -> Option<HostDispatchEntry<B>> {
    let key = HostDispatchKey::for_op(op)?;
    // INVARIANT: the host dispatch table is macro-generated and fixed at
    // compile time, so this lookup is constant-bounded.
    host_dispatch_table::<B>()
        .into_iter()
        .find(|entry| entry.key == key)
}

/// Return a metadata-only view of a borrowed input for a structural operation,
/// or `None` when the operation must materialize through the engine.
///
/// A view is derivable only from an `ExecSlot::Read`: an owned intermediate
/// lives in the slot table, so a borrow of it could not carry the run's input
/// lifetime. Computed intermediates therefore keep the materializing path,
/// unless the fold pass removed the instruction altogether (`input_views`).
pub(super) fn view_backend_op<'input>(
    slots: &[Option<ExecSlot<'input>>],
    inst: &ExecInstruction,
) -> Option<ExecSlot<'input>> {
    if inst.input_slots.len() != 1 || inst.output_slots.len() != 1 {
        return None;
    }
    let view = InputView::from_op(&inst.op)?;
    let read = match slots.get(inst.input_slots[0])?.as_ref()? {
        ExecSlot::Read(read) => read.clone(),
        ExecSlot::Owned(_) => return None,
    };
    // The instruction's own folded views come first, then its transform; a
    // layout the view form cannot express (a reshape of a strided view) falls
    // back to the materializing entry.
    let mut read = read;
    for (_, folded) in &inst.input_views {
        read = folded.apply(read).ok()?;
    }
    view.apply(read).ok().map(ExecSlot::Read)
}

pub(super) fn execute_backend_dispatch(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let Some(entry) = backend_dispatch_entry(&inst.op) else {
        return Err(Error::Internal(format!(
            "host or FFI op reached backend executor: {:?}",
            inst.op
        )));
    };
    (entry.execute)(exec, slots, inst)
}

pub(super) fn execute_host_dispatch<B: HostExecution + ?Sized>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<()> {
    let Some(entry) = host_dispatch_entry::<B>(&inst.op) else {
        return Err(Error::Internal(format!(
            "non-host op reached host executor: {:?}",
            inst.op
        )));
    };
    (entry.execute)(backend, slots, inst)
}

fn ensure_backend_inputs(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
    indices: &[usize],
) -> Result<()> {
    for &idx in indices {
        ensure_owned(exec, slots, inst, idx)?;
    }
    Ok(())
}

fn dispatch_mismatch(expected: PrimitiveOpKind, op: &ExecOp) -> Error {
    Error::Internal(format!(
        "backend dispatch table key {expected:?} called with mismatched op: {op:?}"
    ))
}

fn host_dispatch_mismatch(expected: HostDispatchKey, op: &ExecOp) -> Error {
    Error::Internal(format!(
        "host dispatch table key {expected:?} called with mismatched op: {op:?}"
    ))
}

fn ensure_host_owned<B: HostExecution + ?Sized>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
    idx: usize,
) -> Result<()> {
    if inst.input_views.iter().any(|(input, _)| *input == idx) {
        return Err(Error::Internal(format!(
            "host input {idx} of {:?} carries a folded view, but the operation reads owned inputs",
            inst.op
        )));
    }
    let slot = inst
        .input_slots
        .get(idx)
        .copied()
        .ok_or_else(|| Error::Internal(format!("host input slot {idx} is missing")))?;
    let needs_materialization = matches!(
        slots
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::Internal(format!("host input slot {slot} is empty")))?,
        ExecSlot::Read(TensorRead::View(_))
    );
    if needs_materialization {
        let read = slots
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::Internal(format!("host input slot {slot} is empty")))?
            .as_read();
        let tensor = backend.to_contiguous_read(read)?;
        slots[slot] = Some(ExecSlot::Owned(tensor));
    }
    Ok(())
}

fn execute_shape_of_host<B: HostExecution + ?Sized>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<()> {
    let ExecOp::ShapeOf { axis } = &inst.op else {
        return Err(host_dispatch_mismatch(HostDispatchKey::ShapeOf, &inst.op));
    };
    let input = get_read(slots, inst, 0)?;
    if *axis >= input.shape().len() {
        return Err(Error::invalid_argument(
            "shape_of",
            crate::ErrorPhase::Execution,
            "axis",
            format!(
                "axis {axis} is out of bounds for rank {}",
                input.shape().len()
            ),
        ));
    }
    let host = Tensor::from_typed::<f64>(tenferro_tensor::TypedTensor::from_vec_col_major(
        vec![],
        vec![input.shape()[*axis] as f64],
    )?);
    slots[inst.output_slots[0]] = Some(ExecSlot::Owned(
        backend.upload_host_tensor(TensorRead::from_tensor(&host))?,
    ));
    Ok(())
}

fn execute_dynamic_truncate_host<B: HostExecution + ?Sized>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<()> {
    let ExecOp::DynamicTruncate { axis } = &inst.op else {
        return Err(host_dispatch_mismatch(
            HostDispatchKey::DynamicTruncate,
            &inst.op,
        ));
    };
    ensure_host_owned(backend, slots, inst, 0)?;
    ensure_host_owned(backend, slots, inst, 1)?;
    let input = get(slots, inst, 0)?;
    if *axis >= input.shape().len() {
        return Err(Error::invalid_argument(
            "dynamic_truncate",
            crate::ErrorPhase::Execution,
            "axis",
            format!(
                "axis {axis} is out of bounds for rank {}",
                input.shape().len()
            ),
        ));
    }
    let size_tensor = backend.download_to_host(get_read(slots, inst, 1)?)?;
    let axis_extent = input.shape()[*axis];
    let size = dynamic_truncate_size(&size_tensor, axis_extent)?;
    let rank = input.shape().len();
    let mut limits = input.shape().to_vec();
    limits[*axis] = size;
    let config = SliceConfig {
        starts: vec![0; rank],
        limits,
        strides: vec![1; rank],
    };
    slots[inst.output_slots[0]] = Some(ExecSlot::Owned(backend.slice(input, &config)?));
    Ok(())
}

fn execute_pad_to_match_host<B: HostExecution + ?Sized>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<()> {
    let ExecOp::PadToMatch { axis } = &inst.op else {
        return Err(host_dispatch_mismatch(
            HostDispatchKey::PadToMatch,
            &inst.op,
        ));
    };
    ensure_host_owned(backend, slots, inst, 0)?;
    ensure_host_owned(backend, slots, inst, 1)?;
    let input = get(slots, inst, 0)?;
    let reference = get(slots, inst, 1)?;
    if *axis >= input.shape().len() {
        return Err(Error::invalid_argument(
            "pad_to_match",
            crate::ErrorPhase::Execution,
            "axis",
            format!(
                "axis {axis} is out of bounds for rank {}",
                input.shape().len()
            ),
        ));
    }
    let target_size = reference.shape()[*axis];
    let current_size = input.shape()[*axis];
    if current_size >= target_size {
        slots[inst.output_slots[0]] = Some(ExecSlot::Owned(input.duplicate()?));
    } else {
        let rank = input.shape().len();
        let mut high = vec![0i64; rank];
        high[*axis] = (target_size - current_size) as i64;
        let config = PadConfig {
            edge_padding_low: vec![0i64; rank],
            edge_padding_high: high,
            interior_padding: vec![0i64; rank],
        };
        slots[inst.output_slots[0]] = Some(ExecSlot::Owned(backend.pad(input, &config)?));
    }
    Ok(())
}

fn execute_constant_host<B: HostExecution + ?Sized>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<()> {
    let ExecOp::Constant { dtype, bytes } = &inst.op else {
        return Err(host_dispatch_mismatch(HostDispatchKey::Constant, &inst.op));
    };
    let host = constant_tensor(*dtype, bytes)?;
    slots[inst.output_slots[0]] = Some(ExecSlot::Owned(
        backend.upload_host_tensor(TensorRead::from_tensor(&host))?,
    ));
    Ok(())
}

fn execute_transpose(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Transpose { perm } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Transpose, &inst.op));
    };
    Ok(exec.transpose_read(get_read(slots, inst, 0)?, perm)?)
}

fn execute_reshape(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Reshape { shape } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Reshape, &inst.op));
    };
    let shape = resolve_tensor_shape_exprs(slots, inst, shape)?;
    Ok(exec.reshape_read(get_read(slots, inst, 0)?, &shape)?)
}

fn execute_broadcast_in_dim(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::BroadcastInDim { shape, dims } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::BroadcastInDim, &inst.op));
    };
    let shape = resolve_tensor_shape_exprs(slots, inst, shape)?;
    Ok(exec.broadcast_in_dim_read(get_read(slots, inst, 0)?, &shape, dims)?)
}

fn execute_convert(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    ensure_backend_inputs(exec, slots, inst, &[0])?;
    let ExecOp::Convert { to } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Convert, &inst.op));
    };
    Ok(exec.cast(get(slots, inst, 0)?, *to)?)
}

fn execute_reduce_sum(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::ReduceSum { axes } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::ReduceSum, &inst.op));
    };
    Ok(exec.reduce_sum_read(get_read(slots, inst, 0)?, axes)?)
}

fn execute_reduce_sum_squares(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::ReduceSumSquares { axes } = &inst.op else {
        return Err(dispatch_mismatch(
            PrimitiveOpKind::ReduceSumSquares,
            &inst.op,
        ));
    };
    Ok(exec.reduce_sum_squares_read(get_read(slots, inst, 0)?, axes)?)
}

fn execute_extract_diag(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::ExtractDiag { axis_a, axis_b } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::ExtractDiag, &inst.op));
    };
    Ok(exec.extract_diagonal_read(get_read(slots, inst, 0)?, *axis_a, *axis_b)?)
}

fn execute_embed_diag(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::EmbedDiag { axis_a, axis_b } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::EmbedDiag, &inst.op));
    };
    Ok(exec.embed_diagonal_read(get_read(slots, inst, 0)?, *axis_a, *axis_b)?)
}

fn execute_tril(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Tril { k } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Tril, &inst.op));
    };
    Ok(exec.tril_read(get_read(slots, inst, 0)?, *k)?)
}

fn execute_triu(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Triu { k } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Triu, &inst.op));
    };
    Ok(exec.triu_read(get_read(slots, inst, 0)?, *k)?)
}

fn execute_add(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.add_read(get_read(slots, inst, 0)?, get_read(slots, inst, 1)?)?)
}

fn execute_subtract(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.sub_read(get_read(slots, inst, 0)?, get_read(slots, inst, 1)?)?)
}

fn execute_multiply(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.mul_read(get_read(slots, inst, 0)?, get_read(slots, inst, 1)?)?)
}

fn execute_negate(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.neg_read(get_read(slots, inst, 0)?)?)
}

fn execute_conj(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.conj_read(get_read(slots, inst, 0)?)?)
}

fn execute_divide(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.div_read(get_read(slots, inst, 0)?, get_read(slots, inst, 1)?)?)
}

fn execute_remainder(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.rem_read(get_read(slots, inst, 0)?, get_read(slots, inst, 1)?)?)
}

fn execute_abs(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.abs_read(get_read(slots, inst, 0)?)?)
}

fn execute_sign(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.sign_read(get_read(slots, inst, 0)?)?)
}

fn execute_maximum(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.maximum_read(get_read(slots, inst, 0)?, get_read(slots, inst, 1)?)?)
}

fn execute_minimum(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.minimum_read(get_read(slots, inst, 0)?, get_read(slots, inst, 1)?)?)
}

fn execute_compare(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Compare(dir) = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Compare, &inst.op));
    };
    Ok(exec.compare_read(get_read(slots, inst, 0)?, get_read(slots, inst, 1)?, dir)?)
}

fn execute_select(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.select_read(
        get_read(slots, inst, 0)?,
        get_read(slots, inst, 1)?,
        get_read(slots, inst, 2)?,
    )?)
}

fn execute_clamp(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.clamp_read(
        get_read(slots, inst, 0)?,
        get_read(slots, inst, 1)?,
        get_read(slots, inst, 2)?,
    )?)
}

fn execute_exp(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.exp_read(get_read(slots, inst, 0)?)?)
}

fn execute_log(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.log_read(get_read(slots, inst, 0)?)?)
}

fn execute_sin(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.sin_read(get_read(slots, inst, 0)?)?)
}

fn execute_cos(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.cos_read(get_read(slots, inst, 0)?)?)
}

fn execute_tanh(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.tanh_read(get_read(slots, inst, 0)?)?)
}

fn execute_sqrt(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.sqrt_read(get_read(slots, inst, 0)?)?)
}

fn execute_rsqrt(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.rsqrt_read(get_read(slots, inst, 0)?)?)
}

fn execute_pow(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.pow_read(get_read(slots, inst, 0)?, get_read(slots, inst, 1)?)?)
}

fn execute_expm1(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.expm1_read(get_read(slots, inst, 0)?)?)
}

fn execute_log1p(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.log1p_read(get_read(slots, inst, 0)?)?)
}

fn execute_erf(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.erf_read(get_read(slots, inst, 0)?)?)
}

fn execute_gather(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Gather(config) = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Gather, &inst.op));
    };
    Ok(exec.gather_read(get_read(slots, inst, 0)?, get_read(slots, inst, 1)?, config)?)
}

fn execute_gather_dynamic_slice_sizes(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::GatherDynamicSliceSizes {
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        index_vector_dim,
        slice_sizes,
    } = &inst.op
    else {
        return Err(dispatch_mismatch(
            PrimitiveOpKind::GatherDynamicSliceSizes,
            &inst.op,
        ));
    };
    let slice_sizes = resolve_tensor_shape_exprs(slots, inst, slice_sizes)?;
    let config = GatherConfig {
        offset_dims: offset_dims.clone(),
        collapsed_slice_dims: collapsed_slice_dims.clone(),
        start_index_map: start_index_map.clone(),
        index_vector_dim: *index_vector_dim,
        slice_sizes,
    };
    Ok(exec.gather_read(
        get_read(slots, inst, 0)?,
        get_read(slots, inst, 1)?,
        &config,
    )?)
}

fn execute_scatter(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Scatter(config) = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Scatter, &inst.op));
    };
    Ok(exec.scatter_read(
        get_read(slots, inst, 0)?,
        get_read(slots, inst, 1)?,
        get_read(slots, inst, 2)?,
        config,
    )?)
}

fn execute_slice(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Slice(config) = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Slice, &inst.op));
    };
    Ok(exec.slice_read(get_read(slots, inst, 0)?, config)?)
}

fn execute_dynamic_slice(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::DynamicSlice { slice_sizes } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::DynamicSlice, &inst.op));
    };
    Ok(exec.dynamic_slice_read(
        get_read(slots, inst, 0)?,
        get_read(slots, inst, 1)?,
        slice_sizes,
    )?)
}

fn execute_dynamic_update_slice(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.dynamic_update_slice_read(
        get_read(slots, inst, 0)?,
        get_read(slots, inst, 1)?,
        get_read(slots, inst, 2)?,
    )?)
}

fn execute_pad(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Pad(config) = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Pad, &inst.op));
    };
    Ok(exec.pad_read(get_read(slots, inst, 0)?, config)?)
}

fn execute_concatenate(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Concatenate { axis } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Concatenate, &inst.op));
    };
    let inputs = collect_tensor_reads(slots, inst)?;
    Ok(exec.concatenate_read(&inputs, *axis)?)
}

fn execute_reverse(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Reverse { axes } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Reverse, &inst.op));
    };
    Ok(exec.reverse_read(get_read(slots, inst, 0)?, axes)?)
}

fn execute_reduce_prod(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::ReduceProd { axes } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::ReduceProd, &inst.op));
    };
    Ok(exec.reduce_prod_read(get_read(slots, inst, 0)?, axes)?)
}

fn execute_reduce_max(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::ReduceMax { axes } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::ReduceMax, &inst.op));
    };
    Ok(exec.reduce_max_read(get_read(slots, inst, 0)?, axes)?)
}

fn execute_reduce_min(
    exec: &mut dyn Engine,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::ReduceMin { axes } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::ReduceMin, &inst.op));
    };
    Ok(exec.reduce_min_read(get_read(slots, inst, 0)?, axes)?)
}
