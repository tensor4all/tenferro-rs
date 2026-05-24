use std::sync::Arc;

use crate::error::{Error, Result};
use num_complex::{Complex32, Complex64};
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::{dim_expr::DimExpr, ShapeExtent};
use tenferro_tensor::Error as TensorError;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
    Tensor, TensorBackend, TensorExec, TypedTensor,
};

use crate::scalar_semantics::dynamic_truncate_size;
use tenferro_runtime::extension_runtime::ExtensionExecutor;

#[derive(Clone, Debug)]
pub enum ExecOp {
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
    Convert {
        to: DType,
    },
    Constant {
        dtype: DType,
        bytes: Vec<u8>,
    },
    DotGeneral(DotGeneralConfig),
    DotGeneralWithConj {
        config: DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    },
    ReduceSum {
        axes: Vec<usize>,
    },
    ExtractDiag {
        axis_a: usize,
        axis_b: usize,
    },
    EmbedDiag {
        axis_a: usize,
        axis_b: usize,
    },
    Tril {
        k: i64,
    },
    Triu {
        k: i64,
    },
    Add,
    Multiply,
    Negate,
    Conj,
    Divide,
    Abs,
    Sign,
    Maximum,
    Minimum,
    Compare(CompareDir),
    Select,
    Clamp,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Rsqrt,
    Pow,
    Expm1,
    Log1p,
    Gather(GatherConfig),
    GatherDynamicSliceSizes {
        offset_dims: Vec<usize>,
        collapsed_slice_dims: Vec<usize>,
        start_index_map: Vec<usize>,
        index_vector_dim: usize,
        slice_sizes: Vec<DimExpr>,
    },
    Scatter(ScatterConfig),
    Slice(SliceConfig),
    DynamicSlice {
        slice_sizes: Vec<usize>,
    },
    DynamicUpdateSlice,
    Pad(PadConfig),
    Concatenate {
        axis: usize,
    },
    Reverse {
        axes: Vec<usize>,
    },
    ShapeOf {
        axis: usize,
    },
    DynamicTruncate {
        axis: usize,
    },
    PadToMatch {
        axis: usize,
    },
    ReduceProd {
        axes: Vec<usize>,
    },
    ReduceMax {
        axes: Vec<usize>,
    },
    ReduceMin {
        axes: Vec<usize>,
    },
    /// Out-of-tree extension carrier in the execution IR.
    ///
    /// Payload and dispatch are defined by the inner [`ExtensionOp`]. The
    /// execution pipeline treats extensions as single-instruction FFI
    /// boundaries (spec Section 8): no elementwise fusion, and dispatch is
    /// routed through the executor's registered extension runtime.
    Extension(Arc<dyn ExtensionOp>),
}

#[derive(Clone, Debug)]
pub struct ExecInstruction {
    pub op: ExecOp,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub dtype: tenferro_tensor::DType,
    pub output_shapes: Vec<Vec<DimExpr>>,
    pub output_extents: Vec<Vec<ShapeExtent<DimExpr>>>,
    pub last_use: Vec<bool>,
}

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

pub(crate) fn get<'a, T>(
    slots: &'a [Option<T>],
    input_slots: &[usize],
    idx: usize,
) -> Result<&'a T> {
    let slot = input_slots[idx];
    slots[slot]
        .as_ref()
        .ok_or(TensorError::MissingValue { slot }.into())
}

pub(crate) fn initialize_slots_in(
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    slots: &mut Vec<Option<Tensor>>,
) {
    slots.clear();
    slots.resize_with(program.n_slots, || None);
    for (i, tensor) in inputs.into_iter().enumerate() {
        slots[program.input_slots[i]] = Some(tensor);
    }
}

pub(crate) fn collect_outputs_from(
    program: &ExecProgram,
    slots: &mut [Option<Tensor>],
) -> Result<Vec<Tensor>> {
    program
        .output_slots
        .iter()
        .map(|&slot| {
            slots[slot]
                .take()
                .ok_or(TensorError::MissingValue { slot }.into())
        })
        .collect()
}

pub(crate) fn is_host_instruction(inst: &ExecInstruction) -> bool {
    matches!(
        &inst.op,
        ExecOp::ShapeOf { .. }
            | ExecOp::DynamicTruncate { .. }
            | ExecOp::PadToMatch { .. }
            | ExecOp::Constant { .. }
    )
}

pub(crate) fn is_ffi_instruction(inst: &ExecInstruction) -> bool {
    matches!(
        &inst.op,
        ExecOp::DotGeneral(_) | ExecOp::DotGeneralWithConj { .. } | ExecOp::Extension(_)
    )
}

pub(crate) fn is_exec_session_ffi_instruction(inst: &ExecInstruction) -> bool {
    matches!(
        &inst.op,
        ExecOp::DotGeneral(_) | ExecOp::DotGeneralWithConj { .. }
    )
}

pub(crate) fn resolve_tensor_shape_exprs(
    slots: &[Option<Tensor>],
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

/// Evaluate an [`ExecProgram`] using segmented dispatch.
///
/// Consecutive fusible ops are executed within one backend execution session.
///
/// # Examples
///
/// ```
/// use tenferro::exec::{eval_exec_ir, ExecProgram};
/// use tenferro::CpuBackend;
///
/// let _eval: fn(&mut CpuBackend, &ExecProgram, Vec<tenferro::Tensor>) -> tenferro::error::Result<Vec<tenferro::Tensor>> =
///     eval_exec_ir::<CpuBackend>;
/// ```
pub fn eval_exec_ir<B: TensorBackend + 'static>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    crate::segment::eval_exec_segmented(backend, program, inputs)
}

/// Evaluate an [`ExecProgram`] one instruction at a time.
///
/// This is retained for parity tests against segmented dispatch.
///
/// # Examples
///
/// ```
/// use tenferro::exec::{eval_exec_ir_unsegmented, ExecProgram};
/// use tenferro::CpuBackend;
///
/// let _eval: fn(&mut CpuBackend, &ExecProgram, Vec<tenferro::Tensor>) -> tenferro::error::Result<Vec<tenferro::Tensor>> =
///     eval_exec_ir_unsegmented::<CpuBackend>;
/// ```
pub fn eval_exec_ir_unsegmented<B: TensorBackend + 'static>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    eval_exec_ir_unsegmented_with_cache(backend, program, inputs)
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
    slots: &mut Vec<Option<Tensor>>,
    mut extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<Vec<Tensor>> {
    let result = (|| {
        initialize_slots_in(program, inputs, slots);

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
                    backend.with_exec_session(|exec| execute_backend_op(exec, slots, inst))?;
                slots[inst.output_slots[0]] = Some(result);
            }
            reclaim_last_use_inputs_backend(slots, inst, backend);
        }

        collect_outputs_from(program, slots)
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

pub(crate) fn eval_exec_ir_single_session_with_workspace<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    slots: &mut Vec<Option<Tensor>>,
    backend_cache: &mut B::RuntimeCache,
) -> Result<Vec<Tensor>> {
    let result = (|| {
        initialize_slots_in(program, inputs, slots);

        backend.with_exec_session_cached(backend_cache, |exec| -> Result<()> {
            for (inst_idx, inst) in program.instructions.iter().enumerate() {
                if is_host_instruction(inst) {
                    return Err(Error::Internal(
                        "host instruction reached single-session executor".into(),
                    ));
                } else if is_ffi_instruction(inst) {
                    execute_ffi_instruction_exec(exec, slots, inst, Some(inst_idx))?;
                } else {
                    let result = execute_backend_op(exec, slots, inst)?;
                    slots[inst.output_slots[0]] = Some(result);
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
    exec: &mut dyn TensorExec,
    slots: &[Option<Tensor>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let result = match &inst.op {
        ExecOp::Transpose { perm } => exec.transpose(get(slots, &inst.input_slots, 0)?, perm)?,
        ExecOp::Reshape { shape } => {
            let shape = resolve_tensor_shape_exprs(slots, &inst.input_slots, shape)?;
            exec.reshape(get(slots, &inst.input_slots, 0)?, &shape)?
        }
        ExecOp::BroadcastInDim { shape, dims } => {
            let shape = resolve_tensor_shape_exprs(slots, &inst.input_slots, shape)?;
            exec.broadcast_in_dim(get(slots, &inst.input_slots, 0)?, &shape, dims)?
        }
        ExecOp::Convert { to } => exec.convert(get(slots, &inst.input_slots, 0)?, *to)?,
        ExecOp::ReduceSum { axes } => exec.reduce_sum(get(slots, &inst.input_slots, 0)?, axes)?,
        ExecOp::ExtractDiag { axis_a, axis_b } => {
            exec.extract_diagonal(get(slots, &inst.input_slots, 0)?, *axis_a, *axis_b)?
        }
        ExecOp::EmbedDiag { axis_a, axis_b } => {
            exec.embed_diagonal(get(slots, &inst.input_slots, 0)?, *axis_a, *axis_b)?
        }
        ExecOp::Tril { k } => exec.tril(get(slots, &inst.input_slots, 0)?, *k)?,
        ExecOp::Triu { k } => exec.triu(get(slots, &inst.input_slots, 0)?, *k)?,
        ExecOp::Add => exec.add(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
        )?,
        ExecOp::Multiply => exec.mul(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
        )?,
        ExecOp::Negate => exec.neg(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Conj => exec.conj(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Divide => exec.div(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
        )?,
        ExecOp::Abs => exec.abs(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Sign => exec.sign(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Maximum => exec.maximum(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
        )?,
        ExecOp::Minimum => exec.minimum(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
        )?,
        ExecOp::Compare(dir) => exec.compare(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
            dir,
        )?,
        ExecOp::Select => exec.select(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
            get(slots, &inst.input_slots, 2)?,
        )?,
        ExecOp::Clamp => exec.clamp(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
            get(slots, &inst.input_slots, 2)?,
        )?,
        ExecOp::Exp => exec.exp(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Log => exec.log(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Sin => exec.sin(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Cos => exec.cos(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Tanh => exec.tanh(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Sqrt => exec.sqrt(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Rsqrt => exec.rsqrt(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Pow => exec.pow(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
        )?,
        ExecOp::Expm1 => exec.expm1(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Log1p => exec.log1p(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Gather(config) => exec.gather(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
            config,
        )?,
        ExecOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            index_vector_dim,
            slice_sizes,
        } => {
            let slice_sizes = resolve_tensor_shape_exprs(slots, &inst.input_slots, slice_sizes)?;
            let config = GatherConfig {
                offset_dims: offset_dims.clone(),
                collapsed_slice_dims: collapsed_slice_dims.clone(),
                start_index_map: start_index_map.clone(),
                index_vector_dim: *index_vector_dim,
                slice_sizes,
            };
            exec.gather(
                get(slots, &inst.input_slots, 0)?,
                get(slots, &inst.input_slots, 1)?,
                &config,
            )?
        }
        ExecOp::Scatter(config) => exec.scatter(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
            get(slots, &inst.input_slots, 2)?,
            config,
        )?,
        ExecOp::Slice(config) => exec.slice(get(slots, &inst.input_slots, 0)?, config)?,
        ExecOp::DynamicSlice { slice_sizes } => exec.dynamic_slice(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
            slice_sizes,
        )?,
        ExecOp::DynamicUpdateSlice => exec.dynamic_update_slice(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
            get(slots, &inst.input_slots, 2)?,
        )?,
        ExecOp::Pad(config) => exec.pad(get(slots, &inst.input_slots, 0)?, config)?,
        ExecOp::Concatenate { axis } => {
            let inputs = collect_tensor_refs(slots, &inst.input_slots)?;
            exec.concatenate(&inputs, *axis)?
        }
        ExecOp::Reverse { axes } => exec.reverse(get(slots, &inst.input_slots, 0)?, axes)?,
        ExecOp::ReduceProd { axes } => exec.reduce_prod(get(slots, &inst.input_slots, 0)?, axes)?,
        ExecOp::ReduceMax { axes } => exec.reduce_max(get(slots, &inst.input_slots, 0)?, axes)?,
        ExecOp::ReduceMin { axes } => exec.reduce_min(get(slots, &inst.input_slots, 0)?, axes)?,
        other => {
            return Err(Error::Internal(format!(
                "host or FFI op reached backend executor: {other:?}"
            )))
        }
    };
    Ok(result)
}

pub(crate) fn execute_host_instruction<B: TensorBackend>(
    backend: &mut B,
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
) -> Result<()> {
    match &inst.op {
        ExecOp::ShapeOf { axis } => {
            let input = get(slots, &inst.input_slots, 0)?;
            if *axis >= input.shape().len() {
                return Err(Error::Internal(format!(
                    "ShapeOf: axis {} out of bounds for rank {}",
                    axis,
                    input.shape().len()
                )));
            }
            let host = Tensor::F64(TypedTensor::from_vec_col_major(
                vec![],
                vec![input.shape()[*axis] as f64],
            ));
            slots[inst.output_slots[0]] = Some(backend.upload_host_tensor(&host)?);
        }
        ExecOp::DynamicTruncate { axis } => {
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
            slots[inst.output_slots[0]] = Some(backend.slice(input, &config)?);
        }
        ExecOp::PadToMatch { axis } => {
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
                slots[inst.output_slots[0]] = Some(input.clone());
            } else {
                let rank = input.shape().len();
                let mut high = vec![0i64; rank];
                high[*axis] = (target_size - current_size) as i64;
                let config = PadConfig {
                    edge_padding_low: vec![0i64; rank],
                    edge_padding_high: high,
                    interior_padding: vec![0i64; rank],
                };
                slots[inst.output_slots[0]] = Some(backend.pad(input, &config)?);
            }
        }
        ExecOp::Constant { dtype, bytes } => {
            let host = constant_tensor(*dtype, bytes);
            slots[inst.output_slots[0]] = Some(backend.upload_host_tensor(&host)?);
        }
        other => {
            return Err(Error::Internal(format!(
                "non-host op reached host executor: {other:?}"
            )))
        }
    }
    Ok(())
}

pub(crate) fn execute_ffi_instruction<B: TensorBackend + 'static>(
    backend: &mut B,
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
    _mode: DispatchMode,
    extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<()> {
    match &inst.op {
        ExecOp::DotGeneral(config) => {
            let result = backend.dot_general(
                get(slots, &inst.input_slots, 0)?,
                get(slots, &inst.input_slots, 1)?,
                config,
            )?;
            slots[inst.output_slots[0]] = Some(result);
        }
        ExecOp::DotGeneralWithConj {
            config,
            lhs_conj,
            rhs_conj,
        } => {
            let result = backend.dot_general_with_conj(
                get(slots, &inst.input_slots, 0)?,
                get(slots, &inst.input_slots, 1)?,
                config,
                *lhs_conj,
                *rhs_conj,
            )?;
            slots[inst.output_slots[0]] = Some(result);
        }
        ExecOp::Extension(ext) => {
            execute_extension_instruction(backend, slots, inst, ext.as_ref(), extension_executor)?
        }
        other => {
            return Err(Error::Internal(format!(
                "non-ffi op reached ffi executor: {other:?}"
            )))
        }
    }
    Ok(())
}

pub(crate) fn execute_ffi_instruction_cached<B: TensorBackend + 'static>(
    backend: &mut B,
    backend_cache: &mut B::RuntimeCache,
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
    mode: DispatchMode,
    cache_slot: Option<usize>,
    extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<()> {
    match &inst.op {
        ExecOp::DotGeneral(config) => {
            let result = backend.dot_general_cached(
                backend_cache,
                cache_slot,
                get(slots, &inst.input_slots, 0)?,
                get(slots, &inst.input_slots, 1)?,
                config,
            )?;
            slots[inst.output_slots[0]] = Some(result);
            Ok(())
        }
        ExecOp::DotGeneralWithConj {
            config,
            lhs_conj,
            rhs_conj,
        } => {
            let result = backend.dot_general_with_conj_cached(
                backend_cache,
                cache_slot,
                get(slots, &inst.input_slots, 0)?,
                get(slots, &inst.input_slots, 1)?,
                config,
                *lhs_conj,
                *rhs_conj,
            )?;
            slots[inst.output_slots[0]] = Some(result);
            Ok(())
        }
        _ => execute_ffi_instruction(backend, slots, inst, mode, extension_executor),
    }
}

pub(crate) fn execute_ffi_instruction_exec(
    exec: &mut dyn TensorExec,
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
    cache_slot: Option<usize>,
) -> Result<()> {
    match &inst.op {
        ExecOp::DotGeneral(config) => {
            let result = exec.dot_general_cached(
                cache_slot,
                get(slots, &inst.input_slots, 0)?,
                get(slots, &inst.input_slots, 1)?,
                config,
            )?;
            slots[inst.output_slots[0]] = Some(result);
        }
        ExecOp::DotGeneralWithConj {
            config,
            lhs_conj,
            rhs_conj,
        } => {
            let result = exec.dot_general_with_conj_cached(
                cache_slot,
                get(slots, &inst.input_slots, 0)?,
                get(slots, &inst.input_slots, 1)?,
                config,
                *lhs_conj,
                *rhs_conj,
            )?;
            slots[inst.output_slots[0]] = Some(result);
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
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
    ext: &dyn ExtensionOp,
    extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<()> {
    let inputs = collect_tensor_refs(slots, &inst.input_slots)?;
    let outputs = if let Some(extension_executor) = extension_executor {
        extension_executor.execute(backend, ext, &inputs)
    } else {
        ext.eager_execute(&inputs)
    }
    .map_err(|err| {
        Error::TensorRuntime(tenferro_tensor::Error::BackendFailure {
            op: "extension",
            message: format!("family_id={:?}: {err}", ext.family_id()),
        })
    })?;
    if outputs.len() != inst.output_slots.len() {
        return Err(Error::TensorRuntime(
            tenferro_tensor::Error::InvalidConfig {
                op: "extension",
                message: format!(
                    "family_id={:?}: eager_execute returned {} outputs for {} slots",
                    ext.family_id(),
                    outputs.len(),
                    inst.output_slots.len()
                ),
            },
        ));
    }
    for (slot, tensor) in inst.output_slots.iter().copied().zip(outputs.into_iter()) {
        slots[slot] = Some(tensor);
    }
    Ok(())
}

fn collect_tensor_refs<'a>(
    slots: &'a [Option<Tensor>],
    input_slots: &[usize],
) -> Result<Vec<&'a Tensor>> {
    let mut inputs = Vec::with_capacity(input_slots.len());
    for &slot in input_slots {
        inputs.push(
            slots[slot]
                .as_ref()
                .ok_or(TensorError::MissingValue { slot })?,
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
        DType::I64 => Tensor::I64(TypedTensor::from_vec_col_major(
            vec![],
            vec![i64::from_le_bytes(exact_bytes::<8>(dtype, bytes))],
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
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
    exec: &mut dyn TensorExec,
) {
    for (i, &is_last) in inst.last_use.iter().enumerate() {
        if is_last {
            if let Some(tensor) = slots[inst.input_slots[i]].take() {
                exec.reclaim_buffer(tensor);
            }
        }
    }
}

pub(crate) fn reclaim_last_use_inputs_backend<B: TensorBackend>(
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
    backend: &mut B,
) {
    for (i, &is_last) in inst.last_use.iter().enumerate() {
        if is_last {
            if let Some(tensor) = slots[inst.input_slots[i]].take() {
                backend.reclaim_buffer(tensor);
            }
        }
    }
}
