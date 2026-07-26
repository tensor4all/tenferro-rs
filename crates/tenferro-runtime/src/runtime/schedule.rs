//! Runtime-owned scheduled graph boundary.
//!
//! Phase 5 keeps this representation crate-private. Later phases attach native
//! GPU/XLA dispatch and asynchronous completion to the same node families.

#![allow(
    dead_code,
    reason = "Phase 5 establishes schedule metadata consumed incrementally by later phases"
)]

use std::error::Error as StdError;
use std::fmt;

use crate::error::ErrorPhase;
use crate::exec::ExecProgram;
use crate::Error;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct EventDomainId(u32);

impl EventDomainId {
    pub(crate) const CPU_BLOCKING: Self = Self(0);

    #[cfg(test)]
    pub(crate) fn runtime_created_for_test(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct EventSlotId(u32);

impl EventSlotId {
    pub(crate) const fn new(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct EventDependency {
    domain: EventDomainId,
    slot: EventSlotId,
    generation: u64,
}

impl EventDependency {
    pub(crate) fn new(domain: EventDomainId, slot: EventSlotId, generation: u64) -> Self {
        Self {
            domain,
            slot,
            generation,
        }
    }

    pub(crate) fn domain(&self) -> EventDomainId {
        self.domain
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct EventCompletion {
    domain: EventDomainId,
    slot: EventSlotId,
    generation: u64,
}

impl EventCompletion {
    pub(crate) fn new(domain: EventDomainId, slot: EventSlotId, generation: u64) -> Self {
        Self {
            domain,
            slot,
            generation,
        }
    }

    pub(crate) fn domain(&self) -> EventDomainId {
        self.domain
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScheduledOperation {
    input_values: Box<[usize]>,
    output_values: Box<[usize]>,
    completion: EventCompletion,
}

impl ScheduledOperation {
    pub(crate) fn new(
        input_values: impl Into<Box<[usize]>>,
        output_values: impl Into<Box<[usize]>>,
        completion: EventCompletion,
    ) -> Self {
        Self {
            input_values: input_values.into(),
            output_values: output_values.into(),
            completion,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(domain: EventDomainId) -> Self {
        Self::new([], [], EventCompletion::new(domain, EventSlotId::new(0), 0))
    }

    pub(crate) fn completion(&self) -> EventCompletion {
        self.completion
    }

    fn retained_bytes(&self) -> Option<usize> {
        checked_sum([
            self.input_values
                .len()
                .checked_mul(std::mem::size_of::<usize>())?,
            self.output_values
                .len()
                .checked_mul(std::mem::size_of::<usize>())?,
        ])
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScheduledTransfer {
    source_event_domain: EventDomainId,
    destination_event_domain: EventDomainId,
    dependencies: Box<[EventDependency]>,
    completion: EventCompletion,
}

impl ScheduledTransfer {
    pub(crate) fn new(
        source_event_domain: EventDomainId,
        destination_event_domain: EventDomainId,
        dependencies: impl Into<Box<[EventDependency]>>,
        completion: EventCompletion,
    ) -> Self {
        Self {
            source_event_domain,
            destination_event_domain,
            dependencies: dependencies.into(),
            completion,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        source_event_domain: EventDomainId,
        destination_event_domain: EventDomainId,
    ) -> Self {
        Self::new(
            source_event_domain,
            destination_event_domain,
            [EventDependency::new(
                source_event_domain,
                EventSlotId::new(0),
                0,
            )],
            EventCompletion::new(destination_event_domain, EventSlotId::new(0), 0),
        )
    }

    pub(crate) fn source_event_domain(&self) -> EventDomainId {
        self.source_event_domain
    }

    pub(crate) fn destination_event_domain(&self) -> EventDomainId {
        self.destination_event_domain
    }

    pub(crate) fn dependencies(&self) -> &[EventDependency] {
        &self.dependencies
    }

    pub(crate) fn completion(&self) -> EventCompletion {
        self.completion
    }

    fn retained_bytes(&self) -> Option<usize> {
        self.dependencies
            .len()
            .checked_mul(std::mem::size_of::<EventDependency>())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScheduledCollective {
    dependencies: Box<[EventDependency]>,
    completion: EventCompletion,
    supported: bool,
}

impl ScheduledCollective {
    #[cfg(test)]
    pub(crate) fn unsupported_for_test() -> Self {
        Self {
            dependencies: Box::new([]),
            completion: EventCompletion::new(EventDomainId::CPU_BLOCKING, EventSlotId::new(0), 0),
            supported: false,
        }
    }

    pub(crate) fn completion(&self) -> EventCompletion {
        self.completion
    }

    fn retained_bytes(&self) -> Option<usize> {
        self.dependencies
            .len()
            .checked_mul(std::mem::size_of::<EventDependency>())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScheduledBarrier {
    dependencies: Box<[EventDependency]>,
    completion: EventCompletion,
}

impl ScheduledBarrier {
    pub(crate) fn completion(&self) -> EventCompletion {
        self.completion
    }

    fn retained_bytes(&self) -> Option<usize> {
        self.dependencies
            .len()
            .checked_mul(std::mem::size_of::<EventDependency>())
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ScheduledNode {
    Operation(ScheduledOperation),
    Transfer(ScheduledTransfer),
    Collective(ScheduledCollective),
    Barrier(ScheduledBarrier),
}

impl ScheduledNode {
    pub(crate) fn completion(&self) -> EventCompletion {
        match self {
            Self::Operation(node) => node.completion(),
            Self::Transfer(node) => node.completion(),
            Self::Collective(node) => node.completion(),
            Self::Barrier(node) => node.completion(),
        }
    }

    fn retained_bytes(&self) -> Option<usize> {
        match self {
            Self::Operation(node) => node.retained_bytes(),
            Self::Transfer(node) => node.retained_bytes(),
            Self::Collective(node) => node.retained_bytes(),
            Self::Barrier(node) => node.retained_bytes(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BufferPlan {
    value_count: usize,
    output_slots: Box<[usize]>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct RunAdmissionSummary {
    node_count: usize,
    value_count: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NodeAdmissionSummary {
    input_count: usize,
    output_count: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct ScheduleSegment {
    node_range: std::ops::Range<usize>,
    event_domain: EventDomainId,
}

#[derive(Clone, Debug)]
pub(crate) struct ScheduledGraph {
    nodes: Box<[ScheduledNode]>,
    input_slots: Box<[usize]>,
    output_slots: Box<[usize]>,
    value_count: usize,
    buffer_plan: BufferPlan,
    segments: Box<[ScheduleSegment]>,
}

impl ScheduledGraph {
    pub(crate) fn from_exec_program(program: &ExecProgram) -> Self {
        let nodes = program
            .instructions
            .iter()
            .enumerate()
            .map(|(index, instruction)| {
                ScheduledNode::Operation(ScheduledOperation::new(
                    instruction.input_slots.clone(),
                    instruction.output_slots.clone(),
                    EventCompletion::new(
                        EventDomainId::CPU_BLOCKING,
                        EventSlotId::new(index as u32),
                        0,
                    ),
                ))
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let node_count = nodes.len();
        Self {
            nodes,
            input_slots: program.input_slots.clone().into_boxed_slice(),
            output_slots: program.output_slots.clone().into_boxed_slice(),
            value_count: program.n_slots,
            buffer_plan: BufferPlan {
                value_count: program.n_slots,
                output_slots: program.output_slots.clone().into_boxed_slice(),
            },
            segments: Box::new([ScheduleSegment {
                node_range: 0..node_count,
                event_domain: EventDomainId::CPU_BLOCKING,
            }]),
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(nodes: Vec<ScheduledNode>) -> Self {
        let value_count = nodes.len();
        Self {
            nodes: nodes.into_boxed_slice(),
            input_slots: Box::new([]),
            output_slots: Box::new([]),
            value_count,
            buffer_plan: BufferPlan::default(),
            segments: Box::new([]),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), ScheduleValidationError> {
        for (index, node) in self.nodes.iter().enumerate() {
            match node {
                ScheduledNode::Transfer(transfer) => {
                    if transfer.source_event_domain == transfer.destination_event_domain {
                        return Err(ScheduleValidationError::TransferSameEventDomain { index });
                    }
                    if transfer.dependencies.is_empty() {
                        return Err(ScheduleValidationError::MissingDependency { index });
                    }
                }
                ScheduledNode::Collective(_)
                | ScheduledNode::Operation(_)
                | ScheduledNode::Barrier(_) => {}
            }
        }
        Ok(())
    }

    pub(crate) fn validate_for_runtime(&self) -> crate::Result<()> {
        self.validate().map_err(|source| {
            Error::runtime_state_source("ScheduledGraph::validate", ErrorPhase::Execution, source)
        })
    }

    pub(crate) fn contains_collective(&self) -> bool {
        self.nodes
            .iter()
            .any(|node| matches!(node, ScheduledNode::Collective(_)))
    }

    pub(crate) fn nodes(&self) -> &[ScheduledNode] {
        &self.nodes
    }

    pub(crate) fn retained_bytes(&self) -> Option<usize> {
        let node_payload_bytes = self
            .nodes
            .iter()
            .try_fold(0usize, |sum, node| sum.checked_add(node.retained_bytes()?))?;
        checked_sum([
            std::mem::size_of::<ScheduledGraph>(),
            self.nodes
                .len()
                .checked_mul(std::mem::size_of::<ScheduledNode>())?,
            node_payload_bytes,
            self.input_slots
                .len()
                .checked_mul(std::mem::size_of::<usize>())?,
            self.output_slots
                .len()
                .checked_mul(std::mem::size_of::<usize>())?,
            self.value_count.checked_mul(std::mem::size_of::<usize>())?,
            std::mem::size_of::<BufferPlan>(),
            self.buffer_plan
                .output_slots
                .len()
                .checked_mul(std::mem::size_of::<usize>())?,
            self.segments
                .len()
                .checked_mul(std::mem::size_of::<ScheduleSegment>())?,
        ])
    }

    #[cfg(test)]
    pub(crate) fn execute_for_test(&self) -> Result<(), ScheduleExecutionError> {
        if self.contains_collective() {
            return Err(ScheduleExecutionError::UnsupportedCollective);
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn transfers_for_test(&self) -> impl Iterator<Item = &ScheduledTransfer> {
        self.nodes.iter().filter_map(|node| match node {
            ScheduledNode::Transfer(transfer) => Some(transfer),
            _ => None,
        })
    }

    #[cfg(test)]
    pub(crate) fn nodes_for_test(&self) -> &[ScheduledNode] {
        &self.nodes
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ScheduleValidationError {
    #[error("transfer node {index} uses the same source and destination event domain")]
    TransferSameEventDomain { index: usize },
    #[error("schedule node {index} has no event dependency")]
    MissingDependency { index: usize },
}

#[derive(Debug)]
pub(crate) enum ScheduleExecutionError {
    UnsupportedCollective,
}

impl fmt::Display for ScheduleExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedCollective => formatter.write_str("collective execution unsupported"),
        }
    }
}

impl StdError for ScheduleExecutionError {}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Option<usize> {
    values
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value))
}
