//! ExecProgram segmentation and segment-based dispatch.

use std::collections::{HashMap, HashSet};

use crate::error::Result;
use crate::exec::{
    can_run_in_single_exec_session, collect_output_values_from, collect_outputs_from,
    eval_exec_ir_single_session_slots_with_workspace,
    eval_exec_ir_unsegmented_slot_values_with_cache_and_workspace,
    eval_exec_ir_unsegmented_slots_with_cache_and_workspace, execute_backend_op,
    execute_ffi_instruction_cached, execute_host_instruction, get_read, initialize_exec_slots_in,
    is_ffi_instruction, is_host_instruction, reclaim_last_use_inputs_backend,
    reclaim_last_use_inputs_exec, resolve_tensor_shape_exprs, terminal_output_slots,
    try_execute_terminal_value_instruction, validate_exec_program, DispatchMode, ExecInstruction,
    ExecOp, ExecProgram, ExecSlot,
};
use crate::extension_runtime::ExtensionExecutor;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_tensor::backend::{
    ElementwiseFusionInputView, ElementwiseFusionInst, ElementwiseFusionPlan,
};
use tenferro_tensor::{Tensor, TensorBackend, TensorRead, TensorValue};

/// A compiled execution segment.
///
/// Fused segments group consecutive non-host, non-FFI instructions that can
/// share one backend execution session. FFI and host segments remain
/// single-instruction boundaries in Phase 4.
#[derive(Clone, Debug)]
pub(crate) enum Segment {
    Fused {
        instructions: Vec<ExecInstruction>,
        input_slots: Vec<usize>,
        output_slots: Vec<usize>,
        last_use: Vec<bool>,
    },
    Ffi(ExecInstruction),
    Host(ExecInstruction),
}

/// Compile an [`ExecProgram`] into execution segments.
pub(crate) fn segment_exec_program(program: &ExecProgram) -> Vec<Segment> {
    let future_uses = SegmentUseSummary::new(program);
    let mut segments = Vec::new();
    let mut fused_start: Option<usize> = None;
    let mut idx = 0usize;

    while idx < program.instructions.len() {
        let inst = &program.instructions[idx];
        if is_host_instruction(inst) {
            flush_fused_segment(
                program,
                &future_uses,
                &mut segments,
                fused_start.take(),
                idx,
            );
            segments.push(Segment::Host(inst.clone()));
            idx += 1;
        } else if is_ffi_instruction(inst) {
            flush_fused_segment(
                program,
                &future_uses,
                &mut segments,
                fused_start.take(),
                idx,
            );
            segments.push(Segment::Ffi(inst.clone()));
            idx += 1;
        } else if is_broadcast_multiply_triplet_at(program, idx)
            && !segment_outputs_have_future_uses(program, &future_uses, idx, idx + 3)
        {
            flush_fused_segment(
                program,
                &future_uses,
                &mut segments,
                fused_start.take(),
                idx,
            );
            segments.push(build_fused_segment(program, &future_uses, idx, idx + 3));
            idx += 3;
        } else if is_single_broadcast_multiply_pair_at(program, idx)
            && !segment_outputs_have_future_uses(program, &future_uses, idx, idx + 2)
        {
            flush_fused_segment(
                program,
                &future_uses,
                &mut segments,
                fused_start.take(),
                idx,
            );
            segments.push(build_fused_segment(program, &future_uses, idx, idx + 2));
            idx += 2;
        } else if fused_start.is_none() {
            fused_start = Some(idx);
            idx += 1;
        } else {
            idx += 1;
        }
    }

    flush_fused_segment(
        program,
        &future_uses,
        &mut segments,
        fused_start.take(),
        program.instructions.len(),
    );
    segments
}

fn segment_outputs_have_future_uses(
    program: &ExecProgram,
    use_summary: &SegmentUseSummary,
    start: usize,
    end: usize,
) -> bool {
    program.instructions[start..end]
        .iter()
        .flat_map(|inst| inst.output_slots.iter().copied())
        .any(|slot| use_summary.is_used_at_or_after(slot, end))
}

fn is_broadcast_multiply_triplet_at(program: &ExecProgram, idx: usize) -> bool {
    let Some([lhs_bc, rhs_bc, multiply]) = program.instructions.get(idx..idx + 3) else {
        return false;
    };
    matches!(lhs_bc.op, ExecOp::BroadcastInDim { .. })
        && matches!(rhs_bc.op, ExecOp::BroadcastInDim { .. })
        && matches!(multiply.op, ExecOp::Multiply)
        && lhs_bc.output_slots.len() == 1
        && rhs_bc.output_slots.len() == 1
        && multiply.input_slots.len() == 2
        && multiply.input_slots[0] == lhs_bc.output_slots[0]
        && multiply.input_slots[1] == rhs_bc.output_slots[0]
}

fn is_single_broadcast_multiply_pair_at(program: &ExecProgram, idx: usize) -> bool {
    let Some([broadcast, multiply]) = program.instructions.get(idx..idx + 2) else {
        return false;
    };
    if !matches!(broadcast.op, ExecOp::BroadcastInDim { .. })
        || !matches!(multiply.op, ExecOp::Multiply)
        || broadcast.output_slots.len() != 1
        || multiply.output_slots.len() != 1
        || multiply.input_slots.len() != 2
    {
        return false;
    }

    let broadcast_output = broadcast.output_slots[0];
    multiply.input_slots[0] == broadcast_output || multiply.input_slots[1] == broadcast_output
}

/// Evaluate an [`ExecProgram`] via segment-based dispatch.
#[cfg(test)]
pub(crate) fn eval_exec_segmented<B: TensorBackend + 'static>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    eval_exec_segmented_with_cache(backend, program, inputs)
}

#[cfg(test)]
pub(crate) fn eval_exec_segmented_with_cache<B: TensorBackend + 'static>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    let mut slots = Vec::new();
    let mut backend_cache = B::RuntimeCache::default();
    eval_exec_segmented_with_cache_and_workspace(
        backend,
        program,
        inputs,
        &mut slots,
        &mut backend_cache,
        None,
    )
}

pub(crate) fn eval_exec_segmented_with_cache_and_workspace<B: TensorBackend + 'static>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    slots: &mut Vec<Option<ExecSlot<'static>>>,
    backend_cache: &mut B::RuntimeCache,
    extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<Vec<Tensor>> {
    let inputs = inputs.into_iter().map(ExecSlot::Owned).collect();
    eval_exec_segmented_slots_with_cache_and_workspace(
        backend,
        program,
        inputs,
        slots,
        backend_cache,
        extension_executor,
    )
}

pub(crate) fn eval_exec_segmented_slots_with_cache_and_workspace<
    'input,
    B: TensorBackend + 'static,
>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<ExecSlot<'input>>,
    slots: &mut Vec<Option<ExecSlot<'input>>>,
    backend_cache: &mut B::RuntimeCache,
    mut extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<Vec<Tensor>> {
    validate_exec_program(program, "segmented executor")?;
    let has_fused_segment = has_multi_instruction_fused_segment(program);
    if !has_fused_segment && can_run_in_single_exec_session(program) {
        return eval_exec_ir_single_session_slots_with_workspace(
            backend,
            program,
            inputs,
            slots,
            backend_cache,
        );
    }

    if !has_fused_segment {
        return eval_exec_ir_unsegmented_slots_with_cache_and_workspace(
            backend,
            program,
            inputs,
            slots,
            extension_executor,
        );
    }

    let result = (|| {
        initialize_exec_slots_in(program, inputs, slots)?;

        let segments = segment_exec_program(program);
        let mut inst_idx = 0usize;
        for segment in &segments {
            match segment {
                Segment::Fused {
                    instructions,
                    input_slots,
                    output_slots,
                    last_use,
                } => {
                    backend.with_backend_session(|exec| -> Result<()> {
                        if let Some(plan) =
                            build_elementwise_fusion_plan(instructions, input_slots, output_slots)
                        {
                            let inputs = collect_segment_inputs(slots, input_slots)?;
                            if let Some(outputs) =
                                exec.execute_elementwise_fusion(&inputs, &plan)?
                            {
                                if outputs.len() != output_slots.len() {
                                    return Err(crate::error::Error::Internal(format!(
                                        "fused elementwise kernel produced {} outputs for {} slots",
                                        outputs.len(),
                                        output_slots.len()
                                    )));
                                }
                                for (slot, tensor) in output_slots.iter().copied().zip(outputs) {
                                    slots[slot] = Some(ExecSlot::Owned(tensor));
                                }
                                reclaim_segment_inputs_exec(slots, input_slots, last_use, exec);
                                return Ok(());
                            }
                        }
                        if let Some(output) = try_execute_broadcast_multiply_segment(
                            exec,
                            slots,
                            instructions,
                            output_slots,
                        )? {
                            slots[output_slots[0]] = Some(ExecSlot::Owned(output));
                            reclaim_segment_inputs_exec(slots, input_slots, last_use, exec);
                            return Ok(());
                        }
                        if let Some(output) = try_execute_single_broadcast_multiply_segment(
                            exec,
                            slots,
                            instructions,
                            output_slots,
                        )? {
                            slots[output_slots[0]] = Some(ExecSlot::Owned(output));
                            reclaim_segment_inputs_exec(slots, input_slots, last_use, exec);
                            return Ok(());
                        }

                        for inst in instructions {
                            let result = execute_backend_op(exec, slots, inst)?;
                            slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
                            reclaim_last_use_inputs_exec(slots, inst, exec);
                        }
                        Ok(())
                    })?;
                    inst_idx += instructions.len();
                }
                Segment::Ffi(inst) => {
                    execute_ffi_instruction_cached(
                        backend,
                        backend_cache,
                        slots,
                        inst,
                        DispatchMode::Segmented,
                        Some(inst_idx),
                        extension_executor.as_deref_mut(),
                    )?;
                    reclaim_last_use_inputs_backend(slots, inst, backend);
                    inst_idx += 1;
                }
                Segment::Host(inst) => {
                    execute_host_instruction(backend, slots, inst)?;
                    reclaim_last_use_inputs_backend(slots, inst, backend);
                    inst_idx += 1;
                }
            }
        }

        collect_outputs_from(program, slots)
    })();
    slots.clear();
    result
}

pub(crate) fn eval_exec_segmented_slot_values_with_cache_and_workspace<
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
    validate_exec_program(program, "segmented value executor")?;
    let has_fused_segment = has_multi_instruction_fused_segment(program);
    if !has_fused_segment {
        return eval_exec_ir_unsegmented_slot_values_with_cache_and_workspace(
            backend,
            program,
            inputs,
            slots,
            backend_cache,
            extension_executor,
        );
    }

    let result = (|| {
        initialize_exec_slots_in(program, inputs, slots)?;
        let terminal_slots = terminal_output_slots(program);

        let segments = segment_exec_program(program);
        let mut inst_idx = 0usize;
        for segment in &segments {
            match segment {
                Segment::Fused {
                    instructions,
                    input_slots,
                    output_slots,
                    last_use,
                } => {
                    backend.with_backend_session(|exec| -> Result<()> {
                        if let Some(plan) =
                            build_elementwise_fusion_plan(instructions, input_slots, output_slots)
                        {
                            let inputs = collect_segment_inputs(slots, input_slots)?;
                            if let Some(outputs) =
                                exec.execute_elementwise_fusion(&inputs, &plan)?
                            {
                                if outputs.len() != output_slots.len() {
                                    return Err(crate::error::Error::Internal(format!(
                                        "fused elementwise kernel produced {} outputs for {} slots",
                                        outputs.len(),
                                        output_slots.len()
                                    )));
                                }
                                for (slot, tensor) in output_slots.iter().copied().zip(outputs) {
                                    slots[slot] = Some(ExecSlot::Owned(tensor));
                                }
                                reclaim_segment_inputs_exec(slots, input_slots, last_use, exec);
                                return Ok(());
                            }
                        }
                        if let Some(output) = try_execute_terminal_broadcast_multiply_value_segment(
                            exec,
                            slots,
                            instructions,
                            output_slots,
                            &terminal_slots,
                        )? {
                            slots[output_slots[0]] = Some(ExecSlot::Value(output));
                            reclaim_segment_inputs_exec(slots, input_slots, last_use, exec);
                            return Ok(());
                        }
                        if let Some(output) =
                            try_execute_terminal_single_broadcast_multiply_value_segment(
                                exec,
                                slots,
                                instructions,
                                output_slots,
                                &terminal_slots,
                            )?
                        {
                            slots[output_slots[0]] = Some(ExecSlot::Value(output));
                            reclaim_segment_inputs_exec(slots, input_slots, last_use, exec);
                            return Ok(());
                        }
                        if let Some(output) = try_execute_broadcast_multiply_segment(
                            exec,
                            slots,
                            instructions,
                            output_slots,
                        )? {
                            slots[output_slots[0]] = Some(ExecSlot::Owned(output));
                            reclaim_segment_inputs_exec(slots, input_slots, last_use, exec);
                            return Ok(());
                        }
                        if let Some(output) = try_execute_single_broadcast_multiply_segment(
                            exec,
                            slots,
                            instructions,
                            output_slots,
                        )? {
                            slots[output_slots[0]] = Some(ExecSlot::Owned(output));
                            reclaim_segment_inputs_exec(slots, input_slots, last_use, exec);
                            return Ok(());
                        }

                        for inst in instructions {
                            if try_execute_terminal_value_instruction(slots, inst, &terminal_slots)?
                            {
                                // Already handled as a metadata-only TensorValue.
                            } else {
                                let result = execute_backend_op(exec, slots, inst)?;
                                slots[inst.output_slots[0]] = Some(ExecSlot::Owned(result));
                            }
                            reclaim_last_use_inputs_exec(slots, inst, exec);
                        }
                        Ok(())
                    })?;
                    inst_idx += instructions.len();
                }
                Segment::Ffi(inst) => {
                    if try_execute_terminal_value_instruction(slots, inst, &terminal_slots)? {
                        // Already handled as a metadata-only TensorValue.
                    } else {
                        execute_ffi_instruction_cached(
                            backend,
                            backend_cache,
                            slots,
                            inst,
                            DispatchMode::Segmented,
                            Some(inst_idx),
                            extension_executor.as_deref_mut(),
                        )?;
                    }
                    reclaim_last_use_inputs_backend(slots, inst, backend);
                    inst_idx += 1;
                }
                Segment::Host(inst) => {
                    if try_execute_terminal_value_instruction(slots, inst, &terminal_slots)? {
                        // Already handled as a metadata-only TensorValue.
                    } else {
                        execute_host_instruction(backend, slots, inst)?;
                    }
                    reclaim_last_use_inputs_backend(slots, inst, backend);
                    inst_idx += 1;
                }
            }
        }

        collect_output_values_from(program, slots)
    })();
    slots.clear();
    result
}

fn has_multi_instruction_fused_segment(program: &ExecProgram) -> bool {
    let mut run_len = 0usize;
    for inst in &program.instructions {
        if is_host_instruction(inst) || is_ffi_instruction(inst) {
            run_len = 0;
        } else {
            run_len += 1;
            if run_len > 1 {
                return true;
            }
        }
    }
    false
}

fn try_execute_broadcast_multiply_segment(
    exec: &mut dyn tenferro_tensor::BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    instructions: &[ExecInstruction],
    output_slots: &[usize],
) -> Result<Option<Tensor>> {
    let [lhs_bc, rhs_bc, multiply] = instructions else {
        return Ok(None);
    };
    let ExecOp::BroadcastInDim {
        shape: lhs_shape_exprs,
        dims: lhs_dims,
    } = &lhs_bc.op
    else {
        return Ok(None);
    };
    let ExecOp::BroadcastInDim {
        shape: rhs_shape_exprs,
        dims: rhs_dims,
    } = &rhs_bc.op
    else {
        return Ok(None);
    };
    if !matches!(multiply.op, ExecOp::Multiply)
        || lhs_bc.output_slots.len() != 1
        || rhs_bc.output_slots.len() != 1
        || multiply.output_slots.len() != 1
        || multiply.input_slots.len() != 2
        || multiply.input_slots[0] != lhs_bc.output_slots[0]
        || multiply.input_slots[1] != rhs_bc.output_slots[0]
        || output_slots != multiply.output_slots.as_slice()
    {
        return Ok(None);
    }

    let lhs_shape = resolve_tensor_shape_exprs(slots, &lhs_bc.input_slots, lhs_shape_exprs)?;
    let rhs_shape = resolve_tensor_shape_exprs(slots, &rhs_bc.input_slots, rhs_shape_exprs)?;
    Ok(exec.execute_broadcast_multiply(
        get_read(slots, &lhs_bc.input_slots, 0)?,
        &lhs_shape,
        lhs_dims,
        get_read(slots, &rhs_bc.input_slots, 0)?,
        &rhs_shape,
        rhs_dims,
    )?)
}

#[derive(Clone, Copy)]
enum SingleBroadcastMultiplyPair<'a> {
    WithOther {
        broadcast: &'a ExecInstruction,
        other_slot: usize,
        broadcast_is_lhs: bool,
    },
    ReusedBroadcast {
        broadcast: &'a ExecInstruction,
    },
}

fn single_broadcast_multiply_pair<'a>(
    instructions: &'a [ExecInstruction],
    output_slots: &[usize],
) -> Option<SingleBroadcastMultiplyPair<'a>> {
    let [broadcast, multiply] = instructions else {
        return None;
    };
    if !matches!(broadcast.op, ExecOp::BroadcastInDim { .. })
        || !matches!(multiply.op, ExecOp::Multiply)
        || broadcast.output_slots.len() != 1
        || multiply.output_slots.len() != 1
        || multiply.input_slots.len() != 2
        || output_slots != multiply.output_slots.as_slice()
    {
        return None;
    }

    let broadcast_output = broadcast.output_slots[0];
    match (
        multiply.input_slots[0] == broadcast_output,
        multiply.input_slots[1] == broadcast_output,
    ) {
        (true, true) => Some(SingleBroadcastMultiplyPair::ReusedBroadcast { broadcast }),
        (true, false) => Some(SingleBroadcastMultiplyPair::WithOther {
            broadcast,
            other_slot: multiply.input_slots[1],
            broadcast_is_lhs: true,
        }),
        (false, true) => Some(SingleBroadcastMultiplyPair::WithOther {
            broadcast,
            other_slot: multiply.input_slots[0],
            broadcast_is_lhs: false,
        }),
        (false, false) => None,
    }
}

// Single-broadcast multiply still has Mul's implicit right-aligned broadcasting
// on the non-BroadcastInDim operand, so its backend dims must come from shape.
fn aligned_broadcast_dims(source_shape: &[usize], target_shape: &[usize]) -> Option<Vec<usize>> {
    if source_shape.len() > target_shape.len() {
        return None;
    }
    let offset = target_shape.len() - source_shape.len();
    let mut dims = Vec::with_capacity(source_shape.len());
    for (source_axis, &source_dim) in source_shape.iter().enumerate() {
        let target_axis = offset + source_axis;
        let target_dim = target_shape[target_axis];
        if source_dim != target_dim && source_dim != 1 {
            return None;
        }
        dims.push(target_axis);
    }
    Some(dims)
}

fn read_slot<'slot, 'input>(
    slots: &'slot [Option<ExecSlot<'input>>],
    slot: usize,
) -> Result<TensorRead<'slot>>
where
    'input: 'slot,
{
    slots
        .get(slot)
        .and_then(Option::as_ref)
        .ok_or(tenferro_tensor::Error::MissingValue { slot }.into())
        .map(|value| value.as_read())
}

fn try_execute_single_broadcast_multiply_segment(
    exec: &mut dyn tenferro_tensor::BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    instructions: &[ExecInstruction],
    output_slots: &[usize],
) -> Result<Option<Tensor>> {
    let Some(pair) = single_broadcast_multiply_pair(instructions, output_slots) else {
        return Ok(None);
    };
    let broadcast = match pair {
        SingleBroadcastMultiplyPair::WithOther { broadcast, .. }
        | SingleBroadcastMultiplyPair::ReusedBroadcast { broadcast } => broadcast,
    };
    let ExecOp::BroadcastInDim { shape, dims } = &broadcast.op else {
        return Ok(None);
    };

    let target_shape = resolve_tensor_shape_exprs(slots, &broadcast.input_slots, shape)?;
    let broadcast_read = get_read(slots, &broadcast.input_slots, 0)?;

    match pair {
        SingleBroadcastMultiplyPair::ReusedBroadcast { .. } => Ok(exec
            .execute_broadcast_multiply(
                broadcast_read.clone(),
                &target_shape,
                dims,
                broadcast_read,
                &target_shape,
                dims,
            )?),
        SingleBroadcastMultiplyPair::WithOther {
            other_slot,
            broadcast_is_lhs,
            ..
        } if broadcast_is_lhs => {
            let other_read = read_slot(slots, other_slot)?;
            let Some(other_dims) = aligned_broadcast_dims(other_read.shape(), &target_shape) else {
                return Ok(None);
            };
            Ok(exec.execute_broadcast_multiply(
                broadcast_read,
                &target_shape,
                dims,
                other_read,
                &target_shape,
                &other_dims,
            )?)
        }
        SingleBroadcastMultiplyPair::WithOther { other_slot, .. } => {
            let other_read = read_slot(slots, other_slot)?;
            let Some(other_dims) = aligned_broadcast_dims(other_read.shape(), &target_shape) else {
                return Ok(None);
            };
            Ok(exec.execute_broadcast_multiply(
                other_read,
                &target_shape,
                &other_dims,
                broadcast_read,
                &target_shape,
                dims,
            )?)
        }
    }
}

fn try_execute_terminal_broadcast_multiply_value_segment(
    exec: &mut dyn tenferro_tensor::BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    instructions: &[ExecInstruction],
    output_slots: &[usize],
    terminal_slots: &[bool],
) -> Result<Option<TensorValue>> {
    let [output_slot] = output_slots else {
        return Ok(None);
    };
    if !terminal_slots.get(*output_slot).copied().unwrap_or(false) {
        return Ok(None);
    }

    let [lhs_bc, rhs_bc, multiply] = instructions else {
        return Ok(None);
    };
    let ExecOp::BroadcastInDim {
        shape: lhs_shape_exprs,
        dims: lhs_dims,
    } = &lhs_bc.op
    else {
        return Ok(None);
    };
    let ExecOp::BroadcastInDim {
        shape: rhs_shape_exprs,
        dims: rhs_dims,
    } = &rhs_bc.op
    else {
        return Ok(None);
    };
    if !matches!(multiply.op, ExecOp::Multiply)
        || lhs_bc.output_slots.len() != 1
        || rhs_bc.output_slots.len() != 1
        || multiply.output_slots.len() != 1
        || multiply.input_slots.len() != 2
        || multiply.input_slots[0] != lhs_bc.output_slots[0]
        || multiply.input_slots[1] != rhs_bc.output_slots[0]
        || output_slots != multiply.output_slots.as_slice()
    {
        return Ok(None);
    }

    let lhs_shape = resolve_tensor_shape_exprs(slots, &lhs_bc.input_slots, lhs_shape_exprs)?;
    let rhs_shape = resolve_tensor_shape_exprs(slots, &rhs_bc.input_slots, rhs_shape_exprs)?;
    Ok(exec.execute_broadcast_multiply_value(
        get_read(slots, &lhs_bc.input_slots, 0)?,
        &lhs_shape,
        lhs_dims,
        get_read(slots, &rhs_bc.input_slots, 0)?,
        &rhs_shape,
        rhs_dims,
    )?)
}

fn try_execute_terminal_single_broadcast_multiply_value_segment(
    exec: &mut dyn tenferro_tensor::BackendSession,
    slots: &[Option<ExecSlot<'_>>],
    instructions: &[ExecInstruction],
    output_slots: &[usize],
    terminal_slots: &[bool],
) -> Result<Option<TensorValue>> {
    let [output_slot] = output_slots else {
        return Ok(None);
    };
    if !terminal_slots.get(*output_slot).copied().unwrap_or(false) {
        return Ok(None);
    }

    let Some(pair) = single_broadcast_multiply_pair(instructions, output_slots) else {
        return Ok(None);
    };
    let broadcast = match pair {
        SingleBroadcastMultiplyPair::WithOther { broadcast, .. }
        | SingleBroadcastMultiplyPair::ReusedBroadcast { broadcast } => broadcast,
    };
    let ExecOp::BroadcastInDim { shape, dims } = &broadcast.op else {
        return Ok(None);
    };

    let target_shape = resolve_tensor_shape_exprs(slots, &broadcast.input_slots, shape)?;
    let broadcast_read = get_read(slots, &broadcast.input_slots, 0)?;

    match pair {
        SingleBroadcastMultiplyPair::ReusedBroadcast { .. } => Ok(exec
            .execute_broadcast_multiply_value(
                broadcast_read.clone(),
                &target_shape,
                dims,
                broadcast_read,
                &target_shape,
                dims,
            )?),
        SingleBroadcastMultiplyPair::WithOther {
            other_slot,
            broadcast_is_lhs,
            ..
        } if broadcast_is_lhs => {
            let other_read = read_slot(slots, other_slot)?;
            let Some(other_dims) = aligned_broadcast_dims(other_read.shape(), &target_shape) else {
                return Ok(None);
            };
            Ok(exec.execute_broadcast_multiply_value(
                broadcast_read,
                &target_shape,
                dims,
                other_read,
                &target_shape,
                &other_dims,
            )?)
        }
        SingleBroadcastMultiplyPair::WithOther { other_slot, .. } => {
            let other_read = read_slot(slots, other_slot)?;
            let Some(other_dims) = aligned_broadcast_dims(other_read.shape(), &target_shape) else {
                return Ok(None);
            };
            Ok(exec.execute_broadcast_multiply_value(
                other_read,
                &target_shape,
                &other_dims,
                broadcast_read,
                &target_shape,
                dims,
            )?)
        }
    }
}

fn flush_fused_segment(
    program: &ExecProgram,
    use_summary: &SegmentUseSummary,
    segments: &mut Vec<Segment>,
    start: Option<usize>,
    end: usize,
) {
    let Some(start) = start else {
        return;
    };
    if start == end {
        return;
    }
    segments.push(build_fused_segment(program, use_summary, start, end));
}

struct SegmentUseSummary {
    program_outputs: Vec<bool>,
    overflow_program_outputs: HashSet<usize>,
    last_input_use: Vec<Option<usize>>,
    overflow_last_input_use: HashMap<usize, usize>,
}

impl SegmentUseSummary {
    fn new(program: &ExecProgram) -> Self {
        let mut program_outputs = vec![false; program.n_slots];
        let mut overflow_program_outputs = HashSet::new();
        for &slot in &program.output_slots {
            if let Some(output) = program_outputs.get_mut(slot) {
                *output = true;
            } else {
                overflow_program_outputs.insert(slot);
            }
        }

        let mut last_input_use = vec![None; program.n_slots];
        let mut overflow_last_input_use = HashMap::new();
        for (idx, inst) in program.instructions.iter().enumerate().rev() {
            for &slot in &inst.input_slots {
                if let Some(last_use) = last_input_use.get_mut(slot) {
                    if last_use.is_none() {
                        *last_use = Some(idx);
                    }
                } else {
                    overflow_last_input_use.entry(slot).or_insert(idx);
                }
            }
        }

        Self {
            program_outputs,
            overflow_program_outputs,
            last_input_use,
            overflow_last_input_use,
        }
    }

    fn is_program_output(&self, slot: usize) -> bool {
        self.program_outputs.get(slot).copied().unwrap_or(false)
            || self.overflow_program_outputs.contains(&slot)
    }

    fn is_used_at_or_after(&self, slot: usize, inst_idx: usize) -> bool {
        let last_use = self
            .last_input_use
            .get(slot)
            .copied()
            .flatten()
            .or_else(|| self.overflow_last_input_use.get(&slot).copied());
        last_use.is_some_and(|last_use| last_use >= inst_idx)
    }
}

fn build_fused_segment(
    program: &ExecProgram,
    use_summary: &SegmentUseSummary,
    start: usize,
    end: usize,
) -> Segment {
    let instructions = program.instructions[start..end].to_vec();
    let mut produced = HashSet::new();
    let mut seen_inputs = HashSet::new();
    let mut input_slots = Vec::new();
    let mut produced_order = Vec::new();

    for inst in &instructions {
        for &slot in &inst.input_slots {
            if !produced.contains(&slot) && seen_inputs.insert(slot) {
                input_slots.push(slot);
            }
        }
        for &slot in &inst.output_slots {
            if produced.insert(slot) {
                produced_order.push(slot);
            }
        }
    }

    let output_slots: Vec<usize> = produced_order
        .into_iter()
        .filter(|&slot| {
            use_summary.is_program_output(slot) || use_summary.is_used_at_or_after(slot, end)
        })
        .collect();

    let last_use = input_slots
        .iter()
        .map(|slot| {
            !use_summary.is_program_output(*slot) && !use_summary.is_used_at_or_after(*slot, end)
        })
        .collect();

    Segment::Fused {
        instructions,
        input_slots,
        output_slots,
        last_use,
    }
}

fn collect_segment_inputs<'a>(
    slots: &'a [Option<ExecSlot<'_>>],
    input_slots: &[usize],
) -> Result<Vec<&'a Tensor>> {
    input_slots
        .iter()
        .map(|&slot| {
            slots[slot]
                .as_ref()
                .ok_or(tenferro_tensor::Error::MissingValue { slot }.into())
                .and_then(|value| value.as_tensor("elementwise_fusion"))
        })
        .collect()
}

fn reclaim_segment_inputs_exec(
    slots: &mut [Option<ExecSlot<'_>>],
    input_slots: &[usize],
    last_use: &[bool],
    exec: &mut dyn tenferro_tensor::BackendSession,
) {
    for (&slot, &is_last_use) in input_slots.iter().zip(last_use.iter()) {
        if is_last_use {
            if let Some(ExecSlot::Owned(tensor)) = slots[slot].take() {
                exec.reclaim_buffer(tensor);
            }
        }
    }
}

fn build_elementwise_fusion_plan(
    instructions: &[ExecInstruction],
    input_slots: &[usize],
    output_slots: &[usize],
) -> Option<ElementwiseFusionPlan> {
    let first = instructions.first()?;
    let dtype = first.dtype;
    let mut slot_to_value = HashMap::with_capacity(input_slots.len() + instructions.len());
    // Keep this as Vec to match ElementwiseFusionPlan metadata; A/B
    // benchmarks showed SmallVec made the broadcast metadata path slower.
    let mut input_views = Vec::with_capacity(input_slots.len());
    input_views.resize(input_slots.len(), ElementwiseFusionInputView::Identity);
    for (value, &slot) in input_slots.iter().enumerate() {
        slot_to_value.insert(slot, value);
    }

    let mut ops = Vec::with_capacity(instructions.len());
    let mut next_value = input_slots.len();
    for inst in instructions {
        if inst.dtype != dtype || inst.output_slots.len() != 1 {
            return None;
        }
        if let ExecOp::BroadcastInDim { shape, dims } = &inst.op {
            let input_slot = *inst.input_slots.first()?;
            if inst.input_slots.len() != 1
                || source_slot_is_used_directly_by_elementwise(instructions, input_slot)
            {
                return None;
            }
            let input_value = *slot_to_value.get(&input_slot)?;
            if input_value >= input_views.len()
                || !matches!(
                    input_views[input_value],
                    ElementwiseFusionInputView::Identity
                )
            {
                return None;
            }
            let shape = const_dim_expr_shape(shape)?;
            input_views[input_value] =
                ElementwiseFusionInputView::broadcast_in_dim(shape, dims.clone());
            slot_to_value.insert(inst.output_slots[0], input_value);
            continue;
        }

        let op = inst.op.elementwise_fusion_op()?;
        let inputs = inst
            .input_slots
            .iter()
            .map(|slot| slot_to_value.get(slot).copied())
            .collect::<Option<Vec<_>>>()?;
        ops.push(ElementwiseFusionInst::new(op, inputs));
        slot_to_value.insert(inst.output_slots[0], next_value);
        next_value += 1;
    }

    let outputs = output_slots
        .iter()
        .map(|slot| slot_to_value.get(slot).copied())
        .collect::<Option<Vec<_>>>()?;

    Some(ElementwiseFusionPlan::with_input_views(
        dtype,
        input_views,
        outputs,
        ops,
    ))
}

fn source_slot_is_used_directly_by_elementwise(
    instructions: &[ExecInstruction],
    source_slot: usize,
) -> bool {
    instructions.iter().any(|inst| {
        !matches!(inst.op, ExecOp::BroadcastInDim { .. }) && inst.input_slots.contains(&source_slot)
    })
}

fn const_dim_expr_shape(shape: &[DimExpr]) -> Option<Vec<usize>> {
    shape
        .iter()
        .map(|dim| match dim {
            DimExpr::Const(value) => Some(*value),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests;
