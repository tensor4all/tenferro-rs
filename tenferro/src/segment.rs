//! ExecProgram segmentation and segment-based dispatch.

use std::collections::HashSet;

use crate::error::Result;
use crate::exec::{
    collect_outputs, execute_ffi_instruction, execute_fusible_instruction, execute_host_instruction,
    initialize_slots, is_ffi_instruction, is_host_instruction, reclaim_last_use_inputs_backend,
    reclaim_last_use_inputs_exec, DispatchMode, ExecInstruction, ExecProgram,
};
use tenferro_tensor::{Tensor, TensorBackend};

/// A compiled execution segment.
///
/// Fused segments group consecutive non-host, non-FFI instructions that can
/// share one backend execution session. FFI and host segments remain
/// single-instruction boundaries in Phase 4.
///
/// # Examples
///
/// ```
/// use tenferro::segment::{segment_exec_program, Segment};
/// use tenferro::exec::{ExecInstruction, ExecOp, ExecProgram};
/// use tenferro::DType;
///
/// let program = ExecProgram {
///     instructions: vec![
///         ExecInstruction {
///             op: ExecOp::Add,
///             input_slots: vec![0, 1],
///             output_slots: vec![2],
///             dtype: DType::F64,
///             last_use: vec![false, true],
///         },
///         ExecInstruction {
///             op: ExecOp::Negate,
///             input_slots: vec![2],
///             output_slots: vec![3],
///             dtype: DType::F64,
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
/// use tenferro::segment::{segment_exec_program, Segment};
/// use tenferro::exec::{ExecInstruction, ExecOp, ExecProgram};
/// use tenferro::DType;
///
/// let program = ExecProgram {
///     instructions: vec![
///         ExecInstruction {
///             op: ExecOp::Add,
///             input_slots: vec![0, 1],
///             output_slots: vec![2],
///             dtype: DType::F64,
///             last_use: vec![false, true],
///         },
///         ExecInstruction {
///             op: ExecOp::ShapeOf { axis: 0 },
///             input_slots: vec![2],
///             output_slots: vec![3],
///             dtype: DType::F64,
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
/// use tenferro::segment::eval_exec_segmented;
/// use tenferro::exec::ExecProgram;
/// use tenferro::CpuBackend;
///
/// let _eval: fn(&mut CpuBackend, &ExecProgram, Vec<tenferro::Tensor>) -> tenferro::error::Result<Vec<tenferro::Tensor>> =
///     eval_exec_segmented::<CpuBackend>;
/// ```
pub fn eval_exec_segmented<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    let segments = segment_exec_program(program);
    let mut slots = initialize_slots(program, inputs);

    for segment in &segments {
        match segment {
            Segment::Fused { instructions, .. } => {
                backend.with_exec_session(|exec| -> Result<()> {
                    for inst in instructions {
                        let result = execute_fusible_instruction(exec, &slots, inst)?;
                        slots[inst.output_slots[0]] = Some(result);
                        reclaim_last_use_inputs_exec(&mut slots, inst, exec);
                    }
                    Ok(())
                })?;
            }
            Segment::Ffi(inst) => {
                execute_ffi_instruction(backend, &mut slots, inst, DispatchMode::Segmented)?;
                reclaim_last_use_inputs_backend(&mut slots, inst, backend);
            }
            Segment::Host(inst) => {
                execute_host_instruction(backend, &mut slots, inst)?;
                reclaim_last_use_inputs_backend(&mut slots, inst, backend);
            }
        }
    }

    collect_outputs(program, slots)
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
