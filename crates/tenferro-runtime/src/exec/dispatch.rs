use crate::error::{Error, Result};
use tenferro_core_ops::PrimitiveOpKind;
use tenferro_tensor::{
    BackendSession, GatherConfig, PadConfig, SliceConfig, Tensor, TensorBackend,
};

use super::{
    collect_tensor_refs, constant_tensor, execute_extension_instruction, get, get_read,
    resolve_tensor_shape_exprs, DispatchMode, ExecInstruction, ExecOp, ExecSlot,
};
use crate::extension_runtime::ExtensionExecutor;
use crate::scalar_semantics::dynamic_truncate_size;

type BackendDispatchFn = for<'a> fn(
    &mut dyn BackendSession,
    &[Option<ExecSlot<'a>>],
    &ExecInstruction,
) -> Result<Tensor>;

type FfiDispatchFn<B> = fn(
    &mut B,
    &mut [Option<ExecSlot<'_>>],
    &ExecInstruction,
    DispatchMode,
    Option<&mut ExtensionExecutor<B>>,
) -> Result<()>;

type HostDispatchFn<B> = fn(&mut B, &mut [Option<ExecSlot<'_>>], &ExecInstruction) -> Result<()>;

macro_rules! count_keys {
    ($key:ident) => {
        ()
    };
}

macro_rules! define_backend_dispatch {
    ($( $key:path => $execute:ident, )*) => {
        pub(super) static BACKEND_DISPATCH_TABLE: &[BackendDispatchEntry] = &[
            $(
                BackendDispatchEntry {
                    key: $key,
                    execute: $execute,
                },
            )*
        ];
    };
}

macro_rules! define_ffi_dispatch {
    ($( $key:ident => $pattern:pat => $execute:ident, )*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub(super) enum FfiDispatchKey {
            $( $key, )*
        }

        impl FfiDispatchKey {
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

        fn ffi_dispatch_table<B: TensorBackend + 'static>() -> [FfiDispatchEntry<B>; FfiDispatchKey::COUNT_FOR_TABLE] {
            [
                $(
                    FfiDispatchEntry {
                        key: FfiDispatchKey::$key,
                        execute: $execute::<B>,
                    },
                )*
            ]
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

        fn host_dispatch_table<B: TensorBackend>() -> [HostDispatchEntry<B>; HostDispatchKey::COUNT_FOR_TABLE] {
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

pub(super) struct BackendDispatchEntry {
    pub(super) key: PrimitiveOpKind,
    execute: BackendDispatchFn,
}

pub(super) struct FfiDispatchEntry<B: TensorBackend + 'static> {
    pub(super) key: FfiDispatchKey,
    execute: FfiDispatchFn<B>,
}

pub(super) struct HostDispatchEntry<B: TensorBackend> {
    pub(super) key: HostDispatchKey,
    execute: HostDispatchFn<B>,
}

impl<B: TensorBackend + 'static> Copy for FfiDispatchEntry<B> {}

impl<B: TensorBackend + 'static> Clone for FfiDispatchEntry<B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: TensorBackend> Copy for HostDispatchEntry<B> {}

impl<B: TensorBackend> Clone for HostDispatchEntry<B> {
    fn clone(&self) -> Self {
        *self
    }
}

define_backend_dispatch! {
    PrimitiveOpKind::Transpose => execute_transpose,
    PrimitiveOpKind::Reshape => execute_reshape,
    PrimitiveOpKind::BroadcastInDim => execute_broadcast_in_dim,
    PrimitiveOpKind::Convert => execute_convert,
    PrimitiveOpKind::ReduceSum => execute_reduce_sum,
    PrimitiveOpKind::ExtractDiag => execute_extract_diag,
    PrimitiveOpKind::EmbedDiag => execute_embed_diag,
    PrimitiveOpKind::Tril => execute_tril,
    PrimitiveOpKind::Triu => execute_triu,
    PrimitiveOpKind::Add => execute_add,
    PrimitiveOpKind::Mul => execute_multiply,
    PrimitiveOpKind::Neg => execute_negate,
    PrimitiveOpKind::Conj => execute_conj,
    PrimitiveOpKind::Div => execute_divide,
    PrimitiveOpKind::Abs => execute_abs,
    PrimitiveOpKind::Sign => execute_sign,
    PrimitiveOpKind::Maximum => execute_maximum,
    PrimitiveOpKind::Minimum => execute_minimum,
    PrimitiveOpKind::Compare => execute_compare,
    PrimitiveOpKind::Select => execute_select,
    PrimitiveOpKind::Clamp => execute_clamp,
    PrimitiveOpKind::Exp => execute_exp,
    PrimitiveOpKind::Log => execute_log,
    PrimitiveOpKind::Sin => execute_sin,
    PrimitiveOpKind::Cos => execute_cos,
    PrimitiveOpKind::Tanh => execute_tanh,
    PrimitiveOpKind::Sqrt => execute_sqrt,
    PrimitiveOpKind::Rsqrt => execute_rsqrt,
    PrimitiveOpKind::Pow => execute_pow,
    PrimitiveOpKind::Expm1 => execute_expm1,
    PrimitiveOpKind::Log1p => execute_log1p,
    PrimitiveOpKind::Gather => execute_gather,
    PrimitiveOpKind::GatherDynamicSliceSizes => execute_gather_dynamic_slice_sizes,
    PrimitiveOpKind::Scatter => execute_scatter,
    PrimitiveOpKind::Slice => execute_slice,
    PrimitiveOpKind::DynamicSlice => execute_dynamic_slice,
    PrimitiveOpKind::DynamicUpdateSlice => execute_dynamic_update_slice,
    PrimitiveOpKind::Pad => execute_pad,
    PrimitiveOpKind::Concatenate => execute_concatenate,
    PrimitiveOpKind::Reverse => execute_reverse,
    PrimitiveOpKind::ReduceProd => execute_reduce_prod,
    PrimitiveOpKind::ReduceMax => execute_reduce_max,
    PrimitiveOpKind::ReduceMin => execute_reduce_min,
}

define_ffi_dispatch! {
    DotGeneral => ExecOp::DotGeneral(_) => execute_dot_general_ffi,
    DotGeneralWithConj => ExecOp::DotGeneralWithConj { .. } => execute_dot_general_with_conj_ffi,
    Extension => ExecOp::Extension(_) => execute_extension_ffi,
}

define_host_dispatch! {
    ShapeOf => ExecOp::ShapeOf { .. } => execute_shape_of_host,
    DynamicTruncate => ExecOp::DynamicTruncate { .. } => execute_dynamic_truncate_host,
    PadToMatch => ExecOp::PadToMatch { .. } => execute_pad_to_match_host,
    Constant => ExecOp::Constant { .. } => execute_constant_host,
}

pub(super) fn backend_dispatch_entry(op: &ExecOp) -> Option<&'static BackendDispatchEntry> {
    let key = op.primitive_kind()?;
    BACKEND_DISPATCH_TABLE.iter().find(|entry| entry.key == key)
}

pub(super) fn ffi_dispatch_entry<B: TensorBackend + 'static>(
    op: &ExecOp,
) -> Option<FfiDispatchEntry<B>> {
    let key = FfiDispatchKey::for_op(op)?;
    ffi_dispatch_table::<B>()
        .into_iter()
        .find(|entry| entry.key == key)
}

pub(super) fn is_host_op(op: &ExecOp) -> bool {
    HostDispatchKey::for_op(op).is_some()
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

pub(super) fn host_dispatch_entry<B: TensorBackend>(op: &ExecOp) -> Option<HostDispatchEntry<B>> {
    let key = HostDispatchKey::for_op(op)?;
    host_dispatch_table::<B>()
        .into_iter()
        .find(|entry| entry.key == key)
}

pub(super) fn execute_backend_dispatch(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
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

pub(super) fn execute_host_dispatch<B: TensorBackend>(
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

pub(super) fn execute_ffi_dispatch<B: TensorBackend + 'static>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
    mode: DispatchMode,
    extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<()> {
    let Some(entry) = ffi_dispatch_entry::<B>(&inst.op) else {
        return Err(Error::Internal(format!(
            "non-ffi op reached ffi executor: {:?}",
            inst.op
        )));
    };
    (entry.execute)(backend, slots, inst, mode, extension_executor)
}

fn dispatch_mismatch(expected: PrimitiveOpKind, op: &ExecOp) -> Error {
    Error::Internal(format!(
        "backend dispatch table key {expected:?} called with mismatched op: {op:?}"
    ))
}

fn ffi_dispatch_mismatch(expected: FfiDispatchKey, op: &ExecOp) -> Error {
    Error::Internal(format!(
        "FFI dispatch table key {expected:?} called with mismatched op: {op:?}"
    ))
}

fn host_dispatch_mismatch(expected: HostDispatchKey, op: &ExecOp) -> Error {
    Error::Internal(format!(
        "host dispatch table key {expected:?} called with mismatched op: {op:?}"
    ))
}

fn execute_shape_of_host<B: TensorBackend>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<()> {
    let ExecOp::ShapeOf { axis } = &inst.op else {
        return Err(host_dispatch_mismatch(HostDispatchKey::ShapeOf, &inst.op));
    };
    let input = get_read(slots, &inst.input_slots, 0)?;
    if *axis >= input.shape().len() {
        return Err(Error::Internal(format!(
            "ShapeOf: axis {} out of bounds for rank {}",
            axis,
            input.shape().len()
        )));
    }
    let host = Tensor::F64(tenferro_tensor::TypedTensor::from_vec_col_major(
        vec![],
        vec![input.shape()[*axis] as f64],
    ));
    slots[inst.output_slots[0]] = Some(ExecSlot::Owned(backend.upload_host_tensor(&host)?));
    Ok(())
}

fn execute_dynamic_truncate_host<B: TensorBackend>(
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
    let input = get(slots, &inst.input_slots, 0)?;
    if *axis >= input.shape().len() {
        return Err(Error::Internal(format!(
            "DynamicTruncate: axis {} out of bounds for rank {}",
            axis,
            input.shape().len()
        )));
    }
    let size_tensor = backend.download_to_host(get(slots, &inst.input_slots, 1)?)?;
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

fn execute_pad_to_match_host<B: TensorBackend>(
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
    let input = get(slots, &inst.input_slots, 0)?;
    let reference = get(slots, &inst.input_slots, 1)?;
    if *axis >= input.shape().len() {
        return Err(Error::Internal(format!(
            "PadToMatch: axis {} out of bounds for rank {}",
            axis,
            input.shape().len()
        )));
    }
    let target_size = reference.shape()[*axis];
    let current_size = input.shape()[*axis];
    if current_size >= target_size {
        slots[inst.output_slots[0]] = Some(ExecSlot::Owned(input.clone()));
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

fn execute_constant_host<B: TensorBackend>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<()> {
    let ExecOp::Constant { dtype, bytes } = &inst.op else {
        return Err(host_dispatch_mismatch(HostDispatchKey::Constant, &inst.op));
    };
    let host = constant_tensor(*dtype, bytes);
    slots[inst.output_slots[0]] = Some(ExecSlot::Owned(backend.upload_host_tensor(&host)?));
    Ok(())
}

fn execute_dot_general_ffi<B: TensorBackend + 'static>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
    _mode: DispatchMode,
    _extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<()> {
    let ExecOp::DotGeneral(config) = &inst.op else {
        return Err(ffi_dispatch_mismatch(FfiDispatchKey::DotGeneral, &inst.op));
    };
    let result = backend.dot_general_read(
        get_read(slots, &inst.input_slots, 0)?,
        get_read(slots, &inst.input_slots, 1)?,
        config,
    )?;
    slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
    Ok(())
}

fn execute_dot_general_with_conj_ffi<B: TensorBackend + 'static>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
    _mode: DispatchMode,
    _extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<()> {
    let ExecOp::DotGeneralWithConj {
        config,
        lhs_conj,
        rhs_conj,
    } = &inst.op
    else {
        return Err(ffi_dispatch_mismatch(
            FfiDispatchKey::DotGeneralWithConj,
            &inst.op,
        ));
    };
    let result = backend.dot_general_with_conj_read(
        get_read(slots, &inst.input_slots, 0)?,
        get_read(slots, &inst.input_slots, 1)?,
        config,
        *lhs_conj,
        *rhs_conj,
    )?;
    slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
    Ok(())
}

fn execute_extension_ffi<B: TensorBackend + 'static>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
    _mode: DispatchMode,
    extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<()> {
    let ExecOp::Extension(ext) = &inst.op else {
        return Err(ffi_dispatch_mismatch(FfiDispatchKey::Extension, &inst.op));
    };
    execute_extension_instruction(backend, slots, inst, ext.as_ref(), extension_executor)
}

fn execute_transpose(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Transpose { perm } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Transpose, &inst.op));
    };
    Ok(exec.transpose_read(get_read(slots, &inst.input_slots, 0)?, perm)?)
}

fn execute_reshape(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Reshape { shape } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Reshape, &inst.op));
    };
    let shape = resolve_tensor_shape_exprs(slots, &inst.input_slots, shape)?;
    Ok(exec.reshape_read(get_read(slots, &inst.input_slots, 0)?, &shape)?)
}

fn execute_broadcast_in_dim(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::BroadcastInDim { shape, dims } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::BroadcastInDim, &inst.op));
    };
    let shape = resolve_tensor_shape_exprs(slots, &inst.input_slots, shape)?;
    Ok(exec.broadcast_in_dim_read(get_read(slots, &inst.input_slots, 0)?, &shape, dims)?)
}

fn execute_convert(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Convert { to } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Convert, &inst.op));
    };
    Ok(exec.convert(get(slots, &inst.input_slots, 0)?, *to)?)
}

fn execute_reduce_sum(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::ReduceSum { axes } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::ReduceSum, &inst.op));
    };
    Ok(exec.reduce_sum_read(get_read(slots, &inst.input_slots, 0)?, axes)?)
}

fn execute_extract_diag(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::ExtractDiag { axis_a, axis_b } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::ExtractDiag, &inst.op));
    };
    Ok(exec.extract_diagonal(get(slots, &inst.input_slots, 0)?, *axis_a, *axis_b)?)
}

fn execute_embed_diag(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::EmbedDiag { axis_a, axis_b } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::EmbedDiag, &inst.op));
    };
    Ok(exec.embed_diagonal(get(slots, &inst.input_slots, 0)?, *axis_a, *axis_b)?)
}

fn execute_tril(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Tril { k } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Tril, &inst.op));
    };
    Ok(exec.tril(get(slots, &inst.input_slots, 0)?, *k)?)
}

fn execute_triu(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Triu { k } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Triu, &inst.op));
    };
    Ok(exec.triu(get(slots, &inst.input_slots, 0)?, *k)?)
}

fn execute_add(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.add_read(
        get_read(slots, &inst.input_slots, 0)?,
        get_read(slots, &inst.input_slots, 1)?,
    )?)
}

fn execute_multiply(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.mul_read(
        get_read(slots, &inst.input_slots, 0)?,
        get_read(slots, &inst.input_slots, 1)?,
    )?)
}

fn execute_negate(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.neg_read(get_read(slots, &inst.input_slots, 0)?)?)
}

fn execute_conj(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.conj_read(get_read(slots, &inst.input_slots, 0)?)?)
}

fn execute_divide(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.div_read(
        get_read(slots, &inst.input_slots, 0)?,
        get_read(slots, &inst.input_slots, 1)?,
    )?)
}

fn execute_abs(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.abs_read(get_read(slots, &inst.input_slots, 0)?)?)
}

fn execute_sign(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.sign_read(get_read(slots, &inst.input_slots, 0)?)?)
}

fn execute_maximum(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.maximum_read(
        get_read(slots, &inst.input_slots, 0)?,
        get_read(slots, &inst.input_slots, 1)?,
    )?)
}

fn execute_minimum(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.minimum_read(
        get_read(slots, &inst.input_slots, 0)?,
        get_read(slots, &inst.input_slots, 1)?,
    )?)
}

fn execute_compare(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Compare(dir) = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Compare, &inst.op));
    };
    Ok(exec.compare_read(
        get_read(slots, &inst.input_slots, 0)?,
        get_read(slots, &inst.input_slots, 1)?,
        dir,
    )?)
}

fn execute_select(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.select_read(
        get_read(slots, &inst.input_slots, 0)?,
        get_read(slots, &inst.input_slots, 1)?,
        get_read(slots, &inst.input_slots, 2)?,
    )?)
}

fn execute_clamp(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.clamp_read(
        get_read(slots, &inst.input_slots, 0)?,
        get_read(slots, &inst.input_slots, 1)?,
        get_read(slots, &inst.input_slots, 2)?,
    )?)
}

fn execute_exp(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.exp_read(get_read(slots, &inst.input_slots, 0)?)?)
}

fn execute_log(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.log_read(get_read(slots, &inst.input_slots, 0)?)?)
}

fn execute_sin(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.sin_read(get_read(slots, &inst.input_slots, 0)?)?)
}

fn execute_cos(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.cos_read(get_read(slots, &inst.input_slots, 0)?)?)
}

fn execute_tanh(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.tanh_read(get_read(slots, &inst.input_slots, 0)?)?)
}

fn execute_sqrt(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.sqrt_read(get_read(slots, &inst.input_slots, 0)?)?)
}

fn execute_rsqrt(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.rsqrt_read(get_read(slots, &inst.input_slots, 0)?)?)
}

fn execute_pow(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.pow_read(
        get_read(slots, &inst.input_slots, 0)?,
        get_read(slots, &inst.input_slots, 1)?,
    )?)
}

fn execute_expm1(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.expm1_read(get_read(slots, &inst.input_slots, 0)?)?)
}

fn execute_log1p(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.log1p_read(get_read(slots, &inst.input_slots, 0)?)?)
}

fn execute_gather(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Gather(config) = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Gather, &inst.op));
    };
    Ok(exec.gather(
        get(slots, &inst.input_slots, 0)?,
        get(slots, &inst.input_slots, 1)?,
        config,
    )?)
}

fn execute_gather_dynamic_slice_sizes(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
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
    let slice_sizes = resolve_tensor_shape_exprs(slots, &inst.input_slots, slice_sizes)?;
    let config = GatherConfig {
        offset_dims: offset_dims.clone(),
        collapsed_slice_dims: collapsed_slice_dims.clone(),
        start_index_map: start_index_map.clone(),
        index_vector_dim: *index_vector_dim,
        slice_sizes,
    };
    Ok(exec.gather(
        get(slots, &inst.input_slots, 0)?,
        get(slots, &inst.input_slots, 1)?,
        &config,
    )?)
}

fn execute_scatter(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Scatter(config) = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Scatter, &inst.op));
    };
    Ok(exec.scatter(
        get(slots, &inst.input_slots, 0)?,
        get(slots, &inst.input_slots, 1)?,
        get(slots, &inst.input_slots, 2)?,
        config,
    )?)
}

fn execute_slice(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Slice(config) = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Slice, &inst.op));
    };
    Ok(exec.slice(get(slots, &inst.input_slots, 0)?, config)?)
}

fn execute_dynamic_slice(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::DynamicSlice { slice_sizes } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::DynamicSlice, &inst.op));
    };
    Ok(exec.dynamic_slice(
        get(slots, &inst.input_slots, 0)?,
        get(slots, &inst.input_slots, 1)?,
        slice_sizes,
    )?)
}

fn execute_dynamic_update_slice(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    Ok(exec.dynamic_update_slice(
        get(slots, &inst.input_slots, 0)?,
        get(slots, &inst.input_slots, 1)?,
        get(slots, &inst.input_slots, 2)?,
    )?)
}

fn execute_pad(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Pad(config) = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Pad, &inst.op));
    };
    Ok(exec.pad(get(slots, &inst.input_slots, 0)?, config)?)
}

fn execute_concatenate(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Concatenate { axis } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Concatenate, &inst.op));
    };
    let inputs = collect_tensor_refs(slots, &inst.input_slots)?;
    Ok(exec.concatenate(&inputs, *axis)?)
}

fn execute_reverse(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::Reverse { axes } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::Reverse, &inst.op));
    };
    Ok(exec.reverse(get(slots, &inst.input_slots, 0)?, axes)?)
}

fn execute_reduce_prod(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::ReduceProd { axes } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::ReduceProd, &inst.op));
    };
    Ok(exec.reduce_prod_read(get_read(slots, &inst.input_slots, 0)?, axes)?)
}

fn execute_reduce_max(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::ReduceMax { axes } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::ReduceMax, &inst.op));
    };
    Ok(exec.reduce_max_read(get_read(slots, &inst.input_slots, 0)?, axes)?)
}

fn execute_reduce_min(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let ExecOp::ReduceMin { axes } = &inst.op else {
        return Err(dispatch_mismatch(PrimitiveOpKind::ReduceMin, &inst.op));
    };
    Ok(exec.reduce_min_read(get_read(slots, &inst.input_slots, 0)?, axes)?)
}
