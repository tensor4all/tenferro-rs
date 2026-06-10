use std::sync::Arc;

use crate::error::{Error, Result};
use num_complex::{Complex32, Complex64};
use smallvec::SmallVec;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::{dim_expr::DimExpr, ShapeExtent};
use tenferro_tensor::backend::ElementwiseFusionOp;
use tenferro_tensor::Error as TensorError;
use tenferro_tensor::{
    BackendSession, CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig,
    SliceConfig, Tensor, TensorBackend, TensorRead, TensorValue, TypedTensor,
};

use crate::extension_runtime::ExtensionExecutor;

mod dispatch;

tenferro_core_ops::define_exec_op!();

#[derive(Clone, Debug)]
pub struct ExecInstruction {
    pub op: ExecOp,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DispatchMode {
    Unsegmented,
    Segmented,
}

pub(crate) enum ExecSlot<'a> {
    Owned(Tensor),
    Value(TensorValue),
    Read(TensorRead<'a>),
}

impl<'a> ExecSlot<'a> {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::Owned(tensor) => tensor.dtype(),
            Self::Value(value) => value.dtype(),
            Self::Read(read) => read.dtype(),
        }
    }

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
            Self::Value(TensorValue::Tensor(tensor)) => Ok(tensor.as_ref()),
            Self::Value(TensorValue::View(_)) => Err(Error::Internal(format!(
                "{op}: owned TensorValue view reached an owned-only execution boundary"
            ))),
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

    pub(crate) fn into_tensor(self) -> Tensor {
        match self {
            Self::Owned(tensor) => tensor,
            Self::Value(value) => value.to_tensor(),
            Self::Read(read) => read.to_tensor(),
        }
    }

    pub(crate) fn into_value(self) -> TensorValue {
        match self {
            Self::Owned(tensor) => TensorValue::from_tensor(tensor),
            Self::Value(value) => value,
            Self::Read(read) => TensorValue::from_tensor(read.to_tensor()),
        }
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
    let slot = input_slots[idx];
    slots[slot]
        .as_ref()
        .ok_or_else(|| TensorError::MissingValue { slot }.into())
        .and_then(|value| value.as_tensor("get"))
}

pub(crate) fn get_read<'slot, 'input>(
    slots: &'slot [Option<ExecSlot<'input>>],
    input_slots: &[usize],
    idx: usize,
) -> Result<TensorRead<'slot>>
where
    'input: 'slot,
{
    let slot = input_slots[idx];
    slots[slot]
        .as_ref()
        .ok_or_else(|| TensorError::MissingValue { slot }.into())
        .map(ExecSlot::as_read)
}

pub(crate) fn initialize_exec_slots_in<'input>(
    program: &ExecProgram,
    inputs: Vec<ExecSlot<'input>>,
    slots: &mut Vec<Option<ExecSlot<'input>>>,
) {
    slots.clear();
    slots.resize_with(program.n_slots, || None);
    for (i, input) in inputs.into_iter().enumerate() {
        slots[program.input_slots[i]] = Some(input);
    }
}

pub(crate) fn collect_outputs_from<'input>(
    program: &ExecProgram,
    slots: &mut [Option<ExecSlot<'input>>],
) -> Result<Vec<Tensor>> {
    program
        .output_slots
        .iter()
        .map(|&slot| {
            slots[slot]
                .take()
                .map(ExecSlot::into_tensor)
                .ok_or(TensorError::MissingValue { slot }.into())
        })
        .collect()
}

pub(crate) fn collect_output_values_from<'input>(
    program: &ExecProgram,
    slots: &mut [Option<ExecSlot<'input>>],
) -> Result<Vec<TensorValue>> {
    program
        .output_slots
        .iter()
        .map(|&slot| {
            slots[slot]
                .take()
                .map(ExecSlot::into_value)
                .ok_or(TensorError::MissingValue { slot }.into())
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
        if !consumed.get(slot).copied().unwrap_or(true) {
            terminal[slot] = true;
        }
    }
    terminal
}

pub(crate) fn try_execute_terminal_value_instruction<'input>(
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
        ExecOp::Transpose { perm } => {
            let input_slot = inst.input_slots[0];
            let consume_input = inst.last_use.first().copied().unwrap_or(false);
            let input = tensor_value_for_lazy_view(slots, input_slot, consume_input)?;
            input.transpose_view(perm).map_err(Error::TensorRuntime)?
        }
        ExecOp::Reshape { shape } => {
            let input_slot = inst.input_slots[0];
            let consume_input = inst.last_use.first().copied().unwrap_or(false);
            let shape = resolve_tensor_shape_exprs(slots, &inst.input_slots, shape)?;
            let input = tensor_value_for_lazy_view(slots, input_slot, consume_input)?;
            match input.reshape_view(&shape) {
                Ok(value) => value,
                Err(_) => {
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
            let input = tensor_value_for_lazy_view(slots, input_slot, consume_input)?;
            input
                .broadcast_in_dim_view(&shape, dims)
                .map_err(Error::TensorRuntime)?
        }
        ExecOp::Slice(config) => {
            let input_slot = inst.input_slots[0];
            let consume_input = inst.last_use.first().copied().unwrap_or(false);
            let input = tensor_value_for_lazy_view(slots, input_slot, consume_input)?;
            input.slice_view(config).map_err(Error::TensorRuntime)?
        }
        _ => return Ok(false),
    };
    slots[inst.output_slots[0]] = Some(ExecSlot::Value(output));
    Ok(true)
}

fn tensor_value_for_lazy_view<'input>(
    slots: &mut [Option<ExecSlot<'input>>],
    slot: usize,
    consume: bool,
) -> Result<TensorValue> {
    if consume {
        return slots[slot]
            .take()
            .map(ExecSlot::into_value)
            .ok_or(TensorError::MissingValue { slot }.into());
    }

    match slots[slot].take() {
        Some(ExecSlot::Owned(tensor)) => {
            let tensor = Arc::new(tensor);
            let value = TensorValue::from_tensor_arc(Arc::clone(&tensor));
            slots[slot] = Some(ExecSlot::Value(TensorValue::from_tensor_arc(tensor)));
            Ok(value)
        }
        Some(ExecSlot::Value(value)) => {
            let output = value.clone();
            slots[slot] = Some(ExecSlot::Value(value));
            Ok(output)
        }
        Some(ExecSlot::Read(read)) => {
            let output = TensorValue::from_tensor(read.to_tensor());
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

pub(crate) fn resolve_tensor_shape_exprs(
    slots: &[Option<ExecSlot<'_>>],
    input_slots: &[usize],
    exprs: &[DimExpr],
) -> Result<Vec<usize>> {
    let mut input_shapes = Vec::with_capacity(input_slots.len());
    for &slot in input_slots {
        input_shapes.push(
            slots[slot]
                .as_ref()
                .ok_or(TensorError::MissingValue { slot })?
                .shape(),
        );
    }
    Ok(DimExpr::eval_all(exprs, &input_shapes))
}

pub(crate) fn ensure_core_exec_program(program: &ExecProgram, caller: &str) -> Result<()> {
    for (idx, inst) in program.instructions.iter().enumerate() {
        if let ExecOp::Extension(ext) = &inst.op {
            return Err(Error::TensorRuntime(TensorError::InvalidConfig {
                op: "extension",
                message: format!(
                    "{caller} can execute only core ExecProgram instructions; instruction {idx} uses extension family_id {:?}",
                    ext.family_id()
                ),
            }));
        }
    }
    Ok(())
}

/// Evaluate an [`ExecProgram`] with caller-owned backend runtime cache state.
pub(crate) fn eval_exec_ir_with_backend_cache<B: TensorBackend + 'static>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    backend_cache: &mut B::RuntimeCache,
) -> Result<Vec<Tensor>> {
    let mut slots = Vec::new();
    crate::segment::eval_exec_segmented_with_cache_and_workspace(
        backend,
        program,
        inputs,
        &mut slots,
        backend_cache,
        None,
    )
}

pub(crate) fn eval_exec_ir_unsegmented_with_cache<B: TensorBackend + 'static>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    let mut slots = Vec::new();
    eval_exec_ir_unsegmented_with_cache_and_workspace(backend, program, inputs, &mut slots, None)
}

pub(crate) fn eval_exec_ir_unsegmented_with_cache_and_workspace<B: TensorBackend + 'static>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    slots: &mut Vec<Option<ExecSlot<'static>>>,
    extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<Vec<Tensor>> {
    let inputs = inputs.into_iter().map(ExecSlot::Owned).collect();
    eval_exec_ir_unsegmented_slots_with_cache_and_workspace(
        backend,
        program,
        inputs,
        slots,
        extension_executor,
    )
}

pub(crate) fn eval_exec_ir_unsegmented_slots_with_cache_and_workspace<
    'input,
    B: TensorBackend + 'static,
>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<ExecSlot<'input>>,
    slots: &mut Vec<Option<ExecSlot<'input>>>,
    mut extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<Vec<Tensor>> {
    let result = (|| {
        initialize_exec_slots_in(program, inputs, slots);

        for inst in &program.instructions {
            if is_host_instruction(inst) {
                execute_host_instruction(backend, slots, inst)?;
            } else if is_ffi_instruction(inst) {
                execute_ffi_instruction(
                    backend,
                    slots,
                    inst,
                    DispatchMode::Unsegmented,
                    extension_executor.as_deref_mut(),
                )?;
            } else {
                let result =
                    backend.with_backend_session(|exec| execute_backend_op(exec, slots, inst))?;
                slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
            }
            reclaim_last_use_inputs_backend(slots, inst, backend);
        }

        collect_outputs_from(program, slots)
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
    mut extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<Vec<TensorValue>> {
    let result = (|| {
        initialize_exec_slots_in(program, inputs, slots);
        let terminal_slots = terminal_output_slots(program);

        for (inst_idx, inst) in program.instructions.iter().enumerate() {
            if try_execute_terminal_value_instruction(slots, inst, &terminal_slots)? {
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
                    extension_executor.as_deref_mut(),
                )?;
            } else {
                let result =
                    backend.with_backend_session(|exec| execute_backend_op(exec, slots, inst))?;
                slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
            }
            reclaim_last_use_inputs_backend(slots, inst, backend);
        }

        collect_output_values_from(program, slots)
    })();
    slots.clear();
    result
}

pub(crate) fn can_run_in_single_exec_session(program: &ExecProgram) -> bool {
    program.instructions.iter().all(|inst| {
        !is_host_instruction(inst)
            && (!is_ffi_instruction(inst) || is_exec_session_ffi_instruction(inst))
    })
}

pub(crate) fn eval_exec_ir_single_session_slots_with_workspace<'input, B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<ExecSlot<'input>>,
    slots: &mut Vec<Option<ExecSlot<'input>>>,
    backend_cache: &mut B::RuntimeCache,
) -> Result<Vec<Tensor>> {
    let result = (|| {
        initialize_exec_slots_in(program, inputs, slots);

        backend.with_backend_session_cached(backend_cache, |exec| -> Result<()> {
            for (inst_idx, inst) in program.instructions.iter().enumerate() {
                if is_host_instruction(inst) {
                    return Err(Error::Internal(
                        "host instruction reached single-session executor".into(),
                    ));
                } else if is_ffi_instruction(inst) {
                    execute_ffi_instruction_exec(exec, slots, inst, Some(inst_idx))?;
                } else {
                    let result = execute_backend_op(exec, slots, inst)?;
                    slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
                }
                reclaim_last_use_inputs_exec(slots, inst, exec);
            }
            Ok(())
        })?;

        collect_outputs_from(program, slots)
    })();
    slots.clear();
    result
}

pub(crate) fn execute_backend_op(
    exec: &mut dyn BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    dispatch::execute_backend_dispatch(exec, slots, inst)
}

pub(crate) fn execute_host_instruction<B: TensorBackend>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
) -> Result<()> {
    dispatch::execute_host_dispatch(backend, slots, inst)
}

pub(crate) fn execute_ffi_instruction<B: TensorBackend + 'static>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
    mode: DispatchMode,
    extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<()> {
    dispatch::execute_ffi_dispatch(backend, slots, inst, mode, extension_executor)
}

pub(crate) fn execute_ffi_instruction_cached<B: TensorBackend + 'static>(
    backend: &mut B,
    backend_cache: &mut B::RuntimeCache,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
    mode: DispatchMode,
    cache_slot: Option<usize>,
    extension_executor: Option<&mut ExtensionExecutor<B>>,
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
        _ => execute_ffi_instruction(backend, slots, inst, mode, extension_executor),
    }
}

pub(crate) fn execute_ffi_instruction_exec(
    exec: &mut dyn BackendSession,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
    cache_slot: Option<usize>,
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
        other => {
            return Err(Error::Internal(format!(
                "unsupported single-session FFI op: {other:?}"
            )))
        }
    }
    Ok(())
}

/// Dispatch a compiled `ExecOp::Extension` instruction.
///
/// Per spec Section 8, the compiled pipeline owns metadata lowering and
/// input resolution; the extension runtime owns the actual forward computation.
/// Errors are wrapped in [`Error::BackendFailure`] with `op: "extension"`
/// and the `family_id` included in the message.
fn execute_extension_instruction<B: TensorBackend + 'static>(
    backend: &mut B,
    slots: &mut [Option<ExecSlot<'_>>],
    inst: &ExecInstruction,
    ext: &dyn ExtensionOp,
    extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<()> {
    let inputs = collect_tensor_refs(slots, &inst.input_slots)?;
    let Some(extension_executor) = extension_executor else {
        return Err(Error::TensorRuntime(TensorError::InvalidConfig {
            op: "extension",
            message: format!(
                "extension instruction for family_id {:?} requires an ExtensionExecutor; execute compiled programs through GraphExecutor and register the extension runtime on that executor",
                ext.family_id()
            ),
        }));
    };
    let outputs = extension_executor
        .execute(backend, ext, &inputs)
        .map_err(|err| {
            Error::TensorRuntime(tenferro_tensor::Error::backend_failure(
                "extension",
                format!("family_id={:?}: {err}", ext.family_id()),
            ))
        })?;
    if outputs.len() != inst.output_slots.len() {
        return Err(Error::TensorRuntime(
            tenferro_tensor::Error::InvalidConfig {
                op: "extension",
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

fn collect_tensor_refs<'slot, 'input>(
    slots: &'slot [Option<ExecSlot<'input>>],
    input_slots: &[usize],
) -> Result<Vec<&'slot Tensor>>
where
    'input: 'slot,
{
    let mut inputs = Vec::with_capacity(input_slots.len());
    for &slot in input_slots {
        inputs.push(
            slots[slot]
                .as_ref()
                .ok_or(TensorError::MissingValue { slot })?
                .as_tensor("extension")?,
        );
    }
    Ok(inputs)
}

pub(crate) fn constant_tensor(dtype: DType, bytes: &[u8]) -> Tensor {
    match dtype {
        DType::F64 => Tensor::F64(TypedTensor::from_vec_col_major(
            vec![],
            vec![f64::from_le_bytes(exact_bytes::<8>(dtype, bytes))],
        )),
        DType::F32 => Tensor::F32(TypedTensor::from_vec_col_major(
            vec![],
            vec![f32::from_le_bytes(exact_bytes::<4>(dtype, bytes))],
        )),
        DType::I32 => Tensor::I32(TypedTensor::from_vec_col_major(
            vec![],
            vec![i32::from_le_bytes(exact_bytes::<4>(dtype, bytes))],
        )),
        DType::I64 => Tensor::I64(TypedTensor::from_vec_col_major(
            vec![],
            vec![i64::from_le_bytes(exact_bytes::<8>(dtype, bytes))],
        )),
        DType::Bool => Tensor::Bool(TypedTensor::from_vec_col_major(
            vec![],
            vec![exact_bytes::<1>(dtype, bytes)[0] != 0],
        )),
        DType::C64 => {
            let data = exact_bytes::<16>(dtype, bytes);
            let mut re_bytes = [0u8; 8];
            let mut im_bytes = [0u8; 8];
            re_bytes.copy_from_slice(&data[..8]);
            im_bytes.copy_from_slice(&data[8..]);
            let re = f64::from_le_bytes(re_bytes);
            let im = f64::from_le_bytes(im_bytes);
            Tensor::C64(TypedTensor::from_vec_col_major(
                vec![],
                vec![Complex64::new(re, im)],
            ))
        }
        DType::C32 => {
            let data = exact_bytes::<8>(dtype, bytes);
            let mut re_bytes = [0u8; 4];
            let mut im_bytes = [0u8; 4];
            re_bytes.copy_from_slice(&data[..4]);
            im_bytes.copy_from_slice(&data[4..]);
            let re = f32::from_le_bytes(re_bytes);
            let im = f32::from_le_bytes(im_bytes);
            Tensor::C32(TypedTensor::from_vec_col_major(
                vec![],
                vec![Complex32::new(re, im)],
            ))
        }
    }
}

fn exact_bytes<const N: usize>(dtype: DType, bytes: &[u8]) -> [u8; N] {
    if bytes.len() != N {
        panic!(
            "constant {:?} expected {} bytes, got {}",
            dtype,
            N,
            bytes.len()
        );
    }
    let mut out = [0u8; N];
    out.copy_from_slice(bytes);
    out
}

pub(crate) fn reclaim_last_use_inputs_exec(
    slots: &mut [Option<ExecSlot<'_>>],
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

pub(crate) fn reclaim_last_use_inputs_backend<B: TensorBackend>(
    slots: &mut [Option<ExecSlot<'_>>],
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
        ExecSlot::Value(TensorValue::Tensor(tensor)) => {
            if let Ok(tensor) = Arc::try_unwrap(tensor) {
                exec.reclaim_buffer(tensor);
            }
        }
        ExecSlot::Value(TensorValue::View(_)) | ExecSlot::Read(_) => {}
    }
}

fn reclaim_exec_slot_with_backend<B: TensorBackend>(slot: ExecSlot<'_>, backend: &mut B) {
    match slot {
        ExecSlot::Owned(tensor) => backend.reclaim_buffer(tensor),
        ExecSlot::Value(TensorValue::Tensor(tensor)) => {
            if let Ok(tensor) = Arc::try_unwrap(tensor) {
                backend.reclaim_buffer(tensor);
            }
        }
        ExecSlot::Value(TensorValue::View(_)) | ExecSlot::Read(_) => {}
    }
}

#[cfg(test)]
mod tests;
