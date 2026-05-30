//! ExecProgram segmentation and segment-based dispatch.

use std::collections::{HashMap, HashSet};

use crate::error::Result;
use crate::exec::{
    can_run_in_single_exec_session, collect_outputs_from,
    eval_exec_ir_single_session_with_workspace, eval_exec_ir_unsegmented_with_cache_and_workspace,
    execute_backend_op, execute_ffi_instruction_cached, execute_host_instruction,
    initialize_slots_in, is_ffi_instruction, is_host_instruction, reclaim_last_use_inputs_backend,
    reclaim_last_use_inputs_exec, DispatchMode, ExecInstruction, ExecProgram,
};
use crate::extension_runtime::ExtensionExecutor;
use tenferro_tensor::{ElementwiseFusionInst, ElementwiseFusionPlan, Tensor, TensorBackend};

/// A compiled execution segment.
///
/// Fused segments group consecutive non-host, non-FFI instructions that can
/// share one backend execution session. FFI and host segments remain
/// single-instruction boundaries in Phase 4.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::segment::{segment_exec_program, Segment};
/// use tenferro_runtime::exec::{ExecInstruction, ExecOp, ExecProgram};
/// use tenferro_runtime::DType;
///
/// let program = ExecProgram {
///     instructions: vec![
///         ExecInstruction {
///             op: ExecOp::Add,
///             input_slots: vec![0, 1],
///             output_slots: vec![2],
///             dtype: DType::F64,
///             output_shapes: vec![vec![]],
///             output_extents: vec![vec![]],
///             last_use: vec![false, true],
///         },
///         ExecInstruction {
///             op: ExecOp::Negate,
///             input_slots: vec![2],
///             output_slots: vec![3],
///             dtype: DType::F64,
///             output_shapes: vec![vec![]],
///             output_extents: vec![vec![]],
///             last_use: vec![true],
///         },
///     ],
///     input_slots: vec![0, 1],
///     output_slots: vec![3],
///     n_slots: 4,
/// };
///
/// let segments = segment_exec_program(&program);
/// assert!(matches!(&segments[0], Segment::Fused { instructions, .. } if instructions.len() == 2));
/// ```
#[derive(Clone, Debug)]
pub enum Segment {
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
///
/// # Examples
///
/// ```
/// use tenferro_runtime::segment::{segment_exec_program, Segment};
/// use tenferro_runtime::exec::{ExecInstruction, ExecOp, ExecProgram};
/// use tenferro_runtime::DType;
///
/// let program = ExecProgram {
///     instructions: vec![
///         ExecInstruction {
///             op: ExecOp::Add,
///             input_slots: vec![0, 1],
///             output_slots: vec![2],
///             dtype: DType::F64,
///             output_shapes: vec![vec![]],
///             output_extents: vec![vec![]],
///             last_use: vec![false, true],
///         },
///         ExecInstruction {
///             op: ExecOp::ShapeOf { axis: 0 },
///             input_slots: vec![2],
///             output_slots: vec![3],
///             dtype: DType::F64,
///             output_shapes: vec![vec![]],
///             output_extents: vec![vec![]],
///             last_use: vec![true],
///         },
///     ],
///     input_slots: vec![0, 1],
///     output_slots: vec![2, 3],
///     n_slots: 4,
/// };
///
/// let segments = segment_exec_program(&program);
/// assert!(matches!(&segments[0], Segment::Fused { .. }));
/// assert!(matches!(&segments[1], Segment::Host(_)));
/// ```
pub fn segment_exec_program(program: &ExecProgram) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut fused_start: Option<usize> = None;

    for (idx, inst) in program.instructions.iter().enumerate() {
        if is_host_instruction(inst) {
            flush_fused_segment(program, &mut segments, fused_start.take(), idx);
            segments.push(Segment::Host(inst.clone()));
        } else if is_ffi_instruction(inst) {
            flush_fused_segment(program, &mut segments, fused_start.take(), idx);
            segments.push(Segment::Ffi(inst.clone()));
        } else if fused_start.is_none() {
            fused_start = Some(idx);
        }
    }

    flush_fused_segment(
        program,
        &mut segments,
        fused_start.take(),
        program.instructions.len(),
    );
    segments
}

/// Evaluate an [`ExecProgram`] via segment-based dispatch.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::segment::eval_exec_segmented;
/// use tenferro_runtime::exec::ExecProgram;
/// use tenferro_cpu::CpuBackend;
///
/// let _eval: fn(&mut CpuBackend, &ExecProgram, Vec<tenferro_runtime::Tensor>) -> tenferro_runtime::error::Result<Vec<tenferro_runtime::Tensor>> =
///     eval_exec_segmented::<CpuBackend>;
/// ```
pub fn eval_exec_segmented<B: TensorBackend + 'static>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    eval_exec_segmented_with_cache(backend, program, inputs)
}

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
    slots: &mut Vec<Option<Tensor>>,
    backend_cache: &mut B::RuntimeCache,
    mut extension_executor: Option<&mut ExtensionExecutor<B>>,
) -> Result<Vec<Tensor>> {
    let has_fused_segment = has_multi_instruction_fused_segment(program);
    if !has_fused_segment && can_run_in_single_exec_session(program) {
        return eval_exec_ir_single_session_with_workspace(
            backend,
            program,
            inputs,
            slots,
            backend_cache,
        );
    }

    if !has_fused_segment {
        return eval_exec_ir_unsegmented_with_cache_and_workspace(
            backend,
            program,
            inputs,
            slots,
            extension_executor,
        );
    }

    let result = (|| {
        initialize_slots_in(program, inputs, slots);

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
                                    slots[slot] = Some(tensor);
                                }
                                reclaim_segment_inputs_exec(slots, input_slots, last_use, exec);
                                return Ok(());
                            }
                        }

                        for inst in instructions {
                            let result = execute_backend_op(exec, slots, inst)?;
                            slots[inst.output_slots[0]] = Some(result);
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

fn flush_fused_segment(
    program: &ExecProgram,
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
    segments.push(build_fused_segment(program, start, end));
}

fn build_fused_segment(program: &ExecProgram, start: usize, end: usize) -> Segment {
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

    let later_instructions = &program.instructions[end..];
    let output_slots: Vec<usize> = produced_order
        .into_iter()
        .filter(|slot| {
            program.output_slots.contains(slot)
                || later_instructions
                    .iter()
                    .any(|later| later.input_slots.contains(slot))
        })
        .collect();

    let last_use = input_slots
        .iter()
        .map(|slot| {
            !program.output_slots.contains(slot)
                && !later_instructions
                    .iter()
                    .any(|later| later.input_slots.contains(slot))
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
    slots: &'a [Option<Tensor>],
    input_slots: &[usize],
) -> Result<Vec<&'a Tensor>> {
    input_slots
        .iter()
        .map(|&slot| {
            slots[slot]
                .as_ref()
                .ok_or(tenferro_tensor::Error::MissingValue { slot }.into())
        })
        .collect()
}

fn reclaim_segment_inputs_exec(
    slots: &mut [Option<Tensor>],
    input_slots: &[usize],
    last_use: &[bool],
    exec: &mut dyn tenferro_tensor::BackendSession,
) {
    for (&slot, &is_last_use) in input_slots.iter().zip(last_use.iter()) {
        if is_last_use {
            if let Some(tensor) = slots[slot].take() {
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
    for (value, &slot) in input_slots.iter().enumerate() {
        slot_to_value.insert(slot, value);
    }

    let mut ops = Vec::with_capacity(instructions.len());
    for (next_value, inst) in (input_slots.len()..).zip(instructions.iter()) {
        if inst.dtype != dtype || inst.output_slots.len() != 1 {
            return None;
        }
        let op = inst.op.elementwise_fusion_op()?;
        let inputs = inst
            .input_slots
            .iter()
            .map(|slot| slot_to_value.get(slot).copied())
            .collect::<Option<Vec<_>>>()?;
        ops.push(ElementwiseFusionInst { op, inputs });
        slot_to_value.insert(inst.output_slots[0], next_value);
    }

    let outputs = output_slots
        .iter()
        .map(|slot| slot_to_value.get(slot).copied())
        .collect::<Option<Vec<_>>>()?;

    Some(ElementwiseFusionPlan {
        dtype,
        n_inputs: input_slots.len(),
        outputs,
        ops,
    })
}
