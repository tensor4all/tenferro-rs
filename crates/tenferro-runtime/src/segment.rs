//! ExecProgram segmentation used to plan fused elementwise regions.

use std::collections::{HashMap, HashSet};

use crate::exec::{
    is_ffi_instruction, is_host_instruction, ExecInstruction, ExecOp, ExecProgram, InputView,
};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_tensor::backend::{
    ElementwiseFusionInputView, ElementwiseFusionInst, ElementwiseFusionPlan,
};
use tenferro_tensor::Tensor;

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
    Ffi,
    Host,
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
            segments.push(Segment::Host);
            idx += 1;
        } else if is_ffi_instruction(inst) {
            flush_fused_segment(
                program,
                &future_uses,
                &mut segments,
                fused_start.take(),
                idx,
            );
            segments.push(Segment::Ffi);
            idx += 1;
        } else if let Some(len) = broadcast_multiply_span_at(program, idx)
            && !segment_outputs_have_future_uses(program, &future_uses, idx, idx + len)
        {
            flush_fused_segment(
                program,
                &future_uses,
                &mut segments,
                fused_start.take(),
                idx,
            );
            segments.push(build_fused_segment(program, &future_uses, idx, idx + len));
            idx += len;
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

/// Length of the broadcast-multiply special segment starting at `idx`.
///
/// Three forms are the same computation: `broadcast, broadcast, multiply`,
/// `broadcast, multiply`, and the folded multiply whose input views carry the
/// broadcasts (#2226). The folded form keeps the segmentation boundary the
/// unfolded form had, so a neighbouring candidate is not merged with it.
fn broadcast_multiply_span_at(program: &ExecProgram, idx: usize) -> Option<usize> {
    if is_broadcast_multiply_triplet_at(program, idx) {
        Some(3)
    } else if is_single_broadcast_multiply_pair_at(program, idx) {
        Some(2)
    } else if is_folded_broadcast_multiply_at(program, idx) {
        Some(1)
    } else {
        None
    }
}

fn is_folded_broadcast_multiply_at(program: &ExecProgram, idx: usize) -> bool {
    let Some(multiply) = program.instructions.get(idx) else {
        return false;
    };
    matches!(multiply.op, ExecOp::Multiply)
        && !multiply.input_views.is_empty()
        && multiply
            .input_views
            .iter()
            .all(|(_, view)| matches!(view, InputView::BroadcastInDim { .. }))
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

pub(crate) fn build_elementwise_fusion_plan(
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
    // A slot read through a folded view (#2226) and also read as it is would
    // need two layouts for one region input; such a region keeps the
    // per-instruction path, which reads views itself.
    let mut viewed_slots = HashSet::new();
    let mut direct_slots = HashSet::new();
    for inst in instructions {
        for (position, &slot) in inst.input_slots.iter().enumerate() {
            let viewed = inst
                .input_views
                .iter()
                .any(|(viewed, _)| *viewed == position);
            if viewed {
                viewed_slots.insert(slot);
            } else if !matches!(inst.op, ExecOp::BroadcastInDim { .. }) {
                direct_slots.insert(slot);
            }
        }
    }
    if viewed_slots.iter().any(|slot| direct_slots.contains(slot)) {
        return None;
    }
    for inst in instructions {
        if inst.dtype != dtype || inst.output_slots.len() != 1 {
            return None;
        }
        if !inst.input_views.is_empty() {
            // A folded `BroadcastInDim` on a region input is the same broadcast
            // view the planner derives from a broadcast instruction; any other
            // folded view keeps the region on the per-instruction path.
            if inst.op.elementwise_fusion_op().is_none() {
                return None;
            }
            for (position, view) in &inst.input_views {
                let InputView::BroadcastInDim { shape, dims } = view else {
                    return None;
                };
                if inst
                    .input_views
                    .iter()
                    .filter(|(other, _)| other == position)
                    .count()
                    != 1
                {
                    return None;
                }
                let input_slot = *inst.input_slots.get(*position)?;
                let input_value = *slot_to_value.get(&input_slot)?;
                if input_value >= input_views.len() {
                    return None;
                }
                let shape = const_dim_expr_shape(shape)?;
                let expected = ElementwiseFusionInputView::broadcast_in_dim(shape, dims.clone());
                match &input_views[input_value] {
                    ElementwiseFusionInputView::Identity => input_views[input_value] = expected,
                    current if *current == expected => {}
                    _ => return None,
                }
            }
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

/// Return whether every region input shares the fusion plan's single dtype.
///
/// [`build_elementwise_fusion_plan`] can only compare instruction dtypes, and an
/// instruction records its inferred *output* dtype. An op whose output dtype is
/// not its input dtype — `Abs` on a complex input, for example — therefore
/// leaves an all-real instruction chain that still reads a complex region
/// input. Such a region is a valid per-instruction program but not one
/// uniform-dtype fused kernel, so the runtime declines the fusion and executes
/// the region's instructions one by one instead.
pub(crate) fn plan_matches_input_dtypes(plan: &ElementwiseFusionPlan, inputs: &[&Tensor]) -> bool {
    inputs.iter().all(|input| input.dtype() == plan.dtype())
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
