//! Elementwise regions planned at prepare time.
//!
//! The prepared executor dispatches one command per scheduled operation. This
//! module plans the regions that may instead execute as one fused command, so
//! the decision is made once (outside the timed region) with the same
//! segmentation and eligibility the unprepared path uses.
//!
//! A region is only planned when the segmented executor would also accept it:
//! `segment_exec_program` extracts candidates and `build_elementwise_fusion_plan`
//! proves eligibility. The region keeps the original instruction range, the
//! consecutive scheduled nodes it covers, its external inputs, **all live-outs**
//! (the segment's outputs are exactly the values used at or after the segment or
//! returned by the program), and the input liveness the boundary needs.

use std::collections::HashMap;
use std::ops::Range;

use tenferro_tensor::backend::ElementwiseFusionPlan;

use crate::exec::{ExecInstruction, ExecProgram};
use crate::runtime::schedule::{ScheduledGraph, ScheduledNode};
use crate::segment::{build_elementwise_fusion_plan, segment_exec_program, Segment};

/// Region length from which a borrowed view input is materialized once.
///
/// The fused entry point takes owned tensors, so a view input needs one copy.
/// That copy is a full pass over the input, which only pays off when the region
/// replaces enough commands: below this length the region keeps the
/// per-instruction path, which resolves views itself.
pub(crate) const VIEW_COPY_MIN_INSTRUCTIONS: usize = 3;

/// One elementwise region that may execute as a single fused command.
#[derive(Debug)]
pub(crate) struct ElementwiseRegion {
    /// Instruction indices in the staging program.
    #[allow(
        dead_code,
        reason = "consumed by the region executor in the prepared execution path"
    )]
    pub(crate) instruction_range: Range<usize>,
    /// Scheduled operation nodes covered by the region, consecutive and in
    /// schedule order.
    pub(crate) node_indices: Vec<usize>,
    /// Instructions the region executes, in order.
    pub(crate) instructions: Vec<ExecInstruction>,
    /// Slots read by the region and produced outside it.
    pub(crate) input_slots: Vec<usize>,
    /// Every live-out of the region: produced inside, used at or after the
    /// region, or returned by the program.
    pub(crate) output_slots: Vec<usize>,
    /// For each `input_slots` entry, whether the region holds the last use.
    pub(crate) input_last_use: Vec<bool>,
    /// Fusion plan built by the shared eligibility check.
    pub(crate) plan: ElementwiseFusionPlan,
}

impl ElementwiseRegion {
    pub(crate) fn retained_bytes(&self) -> Option<usize> {
        let instructions = self
            .instructions
            .len()
            .checked_mul(std::mem::size_of::<ExecInstruction>())?;
        let nodes = self
            .node_indices
            .len()
            .checked_mul(std::mem::size_of::<usize>())?;
        let inputs = self
            .input_slots
            .len()
            .checked_mul(std::mem::size_of::<usize>())?;
        let outputs = self
            .output_slots
            .len()
            .checked_mul(std::mem::size_of::<usize>())?;
        let last_use = self
            .input_last_use
            .len()
            .checked_mul(std::mem::size_of::<bool>())?;
        let plan = self
            .plan
            .ops()
            .len()
            .checked_mul(std::mem::size_of::<
                tenferro_tensor::backend::ElementwiseFusionInst,
            >())?
            .checked_add(
                self.plan
                    .outputs()
                    .len()
                    .checked_mul(std::mem::size_of::<usize>())?,
            )?;
        instructions
            .checked_add(nodes)?
            .checked_add(inputs)?
            .checked_add(outputs)?
            .checked_add(last_use)?
            .checked_add(plan)?
            .checked_add(std::mem::size_of::<Self>())
    }
}

/// Plan the elementwise regions of a prepared program.
///
/// Regions are skipped when the segmentation only yields a candidate that the
/// shared eligibility check rejects, or when the candidate does not map to
/// consecutive scheduled operation nodes: those keep the per-instruction
/// dispatch path. A candidate reading more than `max_inputs` external inputs
/// (the root engine's fused-kernel limit, when it reports one) is skipped too,
/// because the backend would decline that fusion on every execution.
pub(crate) fn plan_elementwise_regions(
    staging: &ExecProgram,
    schedule: &ScheduledGraph,
    max_inputs: Option<usize>,
) -> Vec<ElementwiseRegion> {
    let node_of_instruction: HashMap<usize, usize> = schedule
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(node_index, node)| match node {
            ScheduledNode::Operation(operation) => {
                Some((operation.instruction_index(), node_index))
            }
            _ => None,
        })
        .collect();

    let mut regions = Vec::new();
    let mut cursor = 0usize;
    for segment in segment_exec_program(staging) {
        let Segment::Fused {
            instructions,
            input_slots,
            output_slots,
            last_use,
        } = segment
        else {
            cursor += 1;
            continue;
        };
        let start = cursor;
        cursor += instructions.len();
        if instructions.len() < 2 || max_inputs.is_some_and(|limit| input_slots.len() > limit) {
            continue;
        }
        let Some(plan) = build_elementwise_fusion_plan(&instructions, &input_slots, &output_slots)
        else {
            continue;
        };
        let node_indices: Option<Vec<usize>> = (start..start + instructions.len())
            .map(|instruction| node_of_instruction.get(&instruction).copied())
            .collect();
        let Some(node_indices) = node_indices else {
            continue;
        };
        if !node_indices.windows(2).all(|pair| pair[1] == pair[0] + 1) {
            continue;
        }
        regions.push(ElementwiseRegion {
            instruction_range: start..start + instructions.len(),
            node_indices,
            instructions,
            input_slots,
            output_slots,
            input_last_use: last_use,
            plan,
        });
    }
    regions
}

/// How one scheduled node takes part in the planned regions.
///
/// Computed once at prepare so the executor neither builds nor probes a
/// region map per run (#2226).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NodeRegion {
    /// Dispatched as its own command.
    Command,
    /// Starts the region with this index, which executes as one command.
    Start(usize),
    /// Executed as part of the region that starts at an earlier node.
    Covered,
}

/// The region role of every scheduled node.
pub(crate) fn node_regions(
    schedule: &ScheduledGraph,
    regions: &[ElementwiseRegion],
) -> Box<[NodeRegion]> {
    let mut roles = vec![NodeRegion::Command; schedule.nodes().len()];
    for (region_index, region) in regions.iter().enumerate() {
        for (position, &node) in region.node_indices.iter().enumerate() {
            if let Some(role) = roles.get_mut(node) {
                *role = if position == 0 {
                    NodeRegion::Start(region_index)
                } else {
                    NodeRegion::Covered
                };
            }
        }
    }
    roles.into_boxed_slice()
}

/// Execution counters for planned elementwise regions.
///
/// `fused` counts regions executed as one fused command; `fallbacks` counts
/// regions whose fusion the backend rejected and which therefore dispatched
/// their instructions one by one. `submissions` counts successful scheduled
/// event-domain submissions, which is the runtime submission boundary shared by
/// the production backends. Together they are the runtime evidence for
/// execution-path parity tests.
#[derive(Debug, Default)]
pub(crate) struct RegionExecutionCounters {
    fused: std::sync::atomic::AtomicUsize,
    fallbacks: std::sync::atomic::AtomicUsize,
    submissions: std::sync::atomic::AtomicUsize,
}

impl RegionExecutionCounters {
    pub(crate) fn record_fused(&self) {
        self.fused
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn record_fallback(&self) {
        self.fallbacks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn record_submission(&self) {
        self.submissions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn fused(&self) -> usize {
        self.fused.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn fallbacks(&self) -> usize {
        self.fallbacks.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn submissions(&self) -> usize {
        self.submissions.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Plan-derived command count for the production execution path.
///
/// The scheduled executor is the only production executor (`Runtime::run` and
/// admitted execution both reach `execute_scheduled_slots`). It submits one
/// command per scheduled
/// operation, except that a planned region replaces its covered operations with
/// one fused command. The legacy segmented executor in `segment.rs` is
/// exercised only by tests, so its segment count is not a production command
/// count and is not compared here.
pub(crate) fn command_count(schedule: &ScheduledGraph, regions: &[ElementwiseRegion]) -> usize {
    let operations = schedule
        .nodes()
        .iter()
        .filter(|node| matches!(node, ScheduledNode::Operation(_)))
        .count();
    let covered: usize = regions.iter().map(|region| region.node_indices.len()).sum();
    operations
        .saturating_sub(covered)
        .saturating_add(regions.len())
}

/// Number of regions and the instructions they cover, for parity tests.
pub(crate) fn region_summary(regions: &[ElementwiseRegion]) -> (usize, usize) {
    (
        regions.len(),
        regions.iter().map(|region| region.instructions.len()).sum(),
    )
}
