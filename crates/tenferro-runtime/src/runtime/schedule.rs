//! Runtime-owned scheduled graph boundary.
//!
//! This representation remains crate-private. Later phases attach native
//! GPU/XLA dispatch and asynchronous completion to the same node families.

use std::collections::BTreeSet;
#[cfg(test)]
use std::error::Error as StdError;
#[cfg(test)]
use std::fmt;

use crate::error::ErrorPhase;
use crate::exec::ExecProgram;
use crate::{EngineId, Error, StorageClass};

pub(crate) type TransferReachability = BTreeSet<(StorageClass, StorageClass)>;

/// Opaque runtime event-domain identifier.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EventDomainId(u32);

impl EventDomainId {
    pub(crate) fn runtime_allocated(value: u32) -> Self {
        Self(value)
    }

    #[cfg(test)]
    pub(crate) fn runtime_created_for_test(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ExecutionLocation {
    engine_id: EngineId,
    event_domain_id: EventDomainId,
    storage_class: StorageClass,
}

impl ExecutionLocation {
    pub(crate) fn new(
        engine_id: EngineId,
        event_domain_id: EventDomainId,
        storage_class: StorageClass,
    ) -> Self {
        Self {
            engine_id,
            event_domain_id,
            storage_class,
        }
    }

    pub(crate) fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    pub(crate) fn event_domain_id(&self) -> EventDomainId {
        self.event_domain_id
    }

    pub(crate) fn storage_class(&self) -> &StorageClass {
        &self.storage_class
    }

    #[cfg(test)]
    fn for_test(domain: EventDomainId) -> Self {
        Self::new(
            EngineId::new("tenferro-test.schedule-engine").expect("test engine id"),
            domain,
            StorageClass::new("tenferro-test.schedule-storage").expect("test storage class"),
        )
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

    fn from_completion(completion: EventCompletion) -> Self {
        Self::new(completion.domain, completion.slot, completion.generation)
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
    instruction_index: usize,
    location: ExecutionLocation,
    input_values: Box<[usize]>,
    output_values: Box<[usize]>,
    completion: EventCompletion,
}

impl ScheduledOperation {
    pub(crate) fn new(
        instruction_index: usize,
        location: ExecutionLocation,
        input_values: impl Into<Box<[usize]>>,
        output_values: impl Into<Box<[usize]>>,
        completion: EventCompletion,
    ) -> Self {
        Self {
            instruction_index,
            location,
            input_values: input_values.into(),
            output_values: output_values.into(),
            completion,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(domain: EventDomainId) -> Self {
        Self::new(
            0,
            ExecutionLocation::for_test(domain),
            [],
            [],
            EventCompletion::new(domain, EventSlotId::new(0), 0),
        )
    }

    pub(crate) fn instruction_index(&self) -> usize {
        self.instruction_index
    }

    pub(crate) fn location(&self) -> &ExecutionLocation {
        &self.location
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
    value_slot: usize,
    source_location: ExecutionLocation,
    destination_location: ExecutionLocation,
    dependencies: Box<[EventDependency]>,
    completion: EventCompletion,
}

impl ScheduledTransfer {
    pub(crate) fn new(
        value_slot: usize,
        source_location: ExecutionLocation,
        destination_location: ExecutionLocation,
        dependencies: impl Into<Box<[EventDependency]>>,
        completion: EventCompletion,
    ) -> Self {
        Self {
            value_slot,
            source_location,
            destination_location,
            dependencies: dependencies.into(),
            completion,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        source_event_domain: EventDomainId,
        destination_event_domain: EventDomainId,
    ) -> Self {
        let source_location = ExecutionLocation::for_test(source_event_domain);
        let destination_location = ExecutionLocation::for_test(destination_event_domain);
        Self::new(
            0,
            source_location,
            destination_location,
            [EventDependency::new(
                source_event_domain,
                EventSlotId::new(0),
                0,
            )],
            EventCompletion::new(destination_event_domain, EventSlotId::new(0), 0),
        )
    }

    #[cfg(test)]
    pub(crate) fn source_event_domain(&self) -> EventDomainId {
        self.source_location.event_domain_id()
    }

    #[cfg(test)]
    pub(crate) fn destination_event_domain(&self) -> EventDomainId {
        self.destination_location.event_domain_id()
    }

    pub(crate) fn value_slot(&self) -> usize {
        self.value_slot
    }

    pub(crate) fn source_location(&self) -> &ExecutionLocation {
        &self.source_location
    }

    pub(crate) fn destination_location(&self) -> &ExecutionLocation {
        &self.destination_location
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
}

impl ScheduledCollective {
    #[cfg(test)]
    pub(crate) fn unsupported_for_test() -> Self {
        Self {
            dependencies: Box::new([]),
            completion: EventCompletion::new(
                EventDomainId::runtime_created_for_test(0),
                EventSlotId::new(0),
                0,
            ),
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
    // INVARIANT: collective nodes remain representation-only until the
    // explicitly deferred collective scheduler work lands.
    #[allow(
        dead_code,
        reason = "collective scheduling remains representation-only in this scoped change"
    )]
    Collective(ScheduledCollective),
    // INVARIANT: barrier nodes remain representation-only until the explicitly
    // deferred asynchronous event scheduler work lands.
    #[allow(
        dead_code,
        reason = "barrier scheduling remains representation-only in this scoped change"
    )]
    Barrier(ScheduledBarrier),
}

impl ScheduledNode {
    #[cfg(test)]
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

#[derive(Clone, Debug)]
pub(crate) struct ScheduledGraph {
    nodes: Box<[ScheduledNode]>,
    input_slots: Box<[usize]>,
    output_slots: Box<[usize]>,
    value_count: usize,
}

impl ScheduledGraph {
    pub(crate) fn from_exec_program(
        program: &ExecProgram,
        root_location: ExecutionLocation,
        input_locations: &[ExecutionLocation],
        operation_locations: &[ExecutionLocation],
        transfer_reachability: &TransferReachability,
    ) -> Result<Self, ScheduleBuildError> {
        let mut nodes = Vec::with_capacity(program.instructions.len());
        let mut available = vec![Vec::<AvailableValue>::new(); program.n_slots];
        if input_locations.len() != program.input_slots.len() {
            return Err(ScheduleBuildError::InputLocationCountMismatch {
                expected: program.input_slots.len(),
                actual: input_locations.len(),
            });
        }
        for (&slot, location) in program.input_slots.iter().zip(input_locations) {
            let values =
                available
                    .get_mut(slot)
                    .ok_or(ScheduleBuildError::ValueSlotOutOfBounds {
                        slot,
                        value_count: program.n_slots,
                    })?;
            values.push(AvailableValue {
                location: location.clone(),
                completion: None,
            });
        }

        for (instruction_index, instruction) in program.instructions.iter().enumerate() {
            let location = match instruction.semantic_operation_index {
                Some(operation_index) => operation_locations.get(operation_index).cloned().ok_or(
                    ScheduleBuildError::MissingOperationLocation {
                        instruction_index,
                        operation_index,
                    },
                )?,
                None => root_location.clone(),
            };

            for &slot in &instruction.input_slots {
                let values =
                    available
                        .get(slot)
                        .ok_or(ScheduleBuildError::ValueSlotOutOfBounds {
                            slot,
                            value_count: program.n_slots,
                        })?;
                if values.iter().any(|value| value.location == location) {
                    continue;
                }
                let source = values
                    .iter()
                    .find(|value| {
                        transfer_reachability.contains(&(
                            value.location.storage_class().clone(),
                            location.storage_class().clone(),
                        ))
                    })
                    .cloned()
                    .ok_or_else(|| {
                        if values.is_empty() {
                            ScheduleBuildError::ValueUnavailable {
                                instruction_index,
                                slot,
                            }
                        } else {
                            ScheduleBuildError::MissingTransferProvider {
                                instruction_index,
                                slot,
                                destination_storage_class: location.storage_class().clone(),
                                available_storage_classes: values
                                    .iter()
                                    .map(|value| value.location.storage_class().clone())
                                    .collect(),
                            }
                        }
                    })?;
                let completion = event_completion(&nodes, location.event_domain_id())?;
                let dependencies = source
                    .completion
                    .map(EventDependency::from_completion)
                    .into_iter()
                    .collect::<Vec<_>>();
                nodes.push(ScheduledNode::Transfer(ScheduledTransfer::new(
                    slot,
                    source.location,
                    location.clone(),
                    dependencies,
                    completion,
                )));
                available[slot].push(AvailableValue {
                    location: location.clone(),
                    completion: Some(completion),
                });
            }

            let completion = event_completion(&nodes, location.event_domain_id())?;
            nodes.push(ScheduledNode::Operation(ScheduledOperation::new(
                instruction_index,
                location.clone(),
                instruction.input_slots.clone(),
                instruction.output_slots.clone(),
                completion,
            )));

            for (input_index, &slot) in instruction.input_slots.iter().enumerate() {
                if instruction
                    .last_use
                    .get(input_index)
                    .copied()
                    .unwrap_or(false)
                {
                    available[slot].clear();
                }
            }
            for &slot in &instruction.output_slots {
                let values =
                    available
                        .get_mut(slot)
                        .ok_or(ScheduleBuildError::ValueSlotOutOfBounds {
                            slot,
                            value_count: program.n_slots,
                        })?;
                values.clear();
                values.push(AvailableValue {
                    location: location.clone(),
                    completion: Some(completion),
                });
            }
        }

        Ok(Self {
            nodes: nodes.into_boxed_slice(),
            input_slots: program.input_slots.clone().into_boxed_slice(),
            output_slots: program.output_slots.clone().into_boxed_slice(),
            value_count: program.n_slots,
        })
    }

    #[cfg(test)]
    pub(crate) fn for_test(nodes: Vec<ScheduledNode>) -> Self {
        let value_count = nodes.len();
        Self {
            nodes: nodes.into_boxed_slice(),
            input_slots: Box::new([]),
            output_slots: Box::new([]),
            value_count,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), ScheduleValidationError> {
        for (index, node) in self.nodes.iter().enumerate() {
            match node {
                ScheduledNode::Operation(operation) => {
                    if operation.completion().domain() != operation.location().event_domain_id() {
                        return Err(ScheduleValidationError::CompletionEventDomainMismatch {
                            index,
                        });
                    }
                }
                ScheduledNode::Transfer(transfer) => {
                    if transfer.source_location == transfer.destination_location {
                        return Err(ScheduleValidationError::TransferSameLocation { index });
                    }
                    if transfer.completion().domain()
                        != transfer.destination_location().event_domain_id()
                    {
                        return Err(ScheduleValidationError::CompletionEventDomainMismatch {
                            index,
                        });
                    }
                    if transfer.dependencies().iter().any(|dependency| {
                        dependency.domain() != transfer.source_location().event_domain_id()
                    }) {
                        return Err(ScheduleValidationError::DependencyEventDomainMismatch {
                            index,
                        });
                    }
                }
                ScheduledNode::Collective(collective) => {
                    let _ = collective.completion().domain();
                }
                ScheduledNode::Barrier(barrier) => {
                    let _ = barrier.completion().domain();
                }
            }
        }
        Ok(())
    }

    pub(crate) fn validate_for_runtime(&self) -> crate::Result<()> {
        self.validate().map_err(|source| {
            Error::runtime_state_source("ScheduledGraph::validate", ErrorPhase::Execution, source)
        })
    }

    #[cfg(test)]
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
    #[error("transfer node {index} uses the same source and destination location")]
    TransferSameLocation { index: usize },
    #[error("schedule node {index} completion uses the wrong event domain")]
    CompletionEventDomainMismatch { index: usize },
    #[error("schedule node {index} dependency uses the wrong event domain")]
    DependencyEventDomainMismatch { index: usize },
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ScheduleBuildError {
    #[error("schedule has {actual} input locations for {expected} program inputs")]
    InputLocationCountMismatch { expected: usize, actual: usize },
    #[error(
        "instruction {instruction_index} references semantic operation {operation_index}, \
         but that operation has no execution location"
    )]
    MissingOperationLocation {
        instruction_index: usize,
        operation_index: usize,
    },
    #[error("instruction {instruction_index} requires unavailable value slot {slot}")]
    ValueUnavailable {
        instruction_index: usize,
        slot: usize,
    },
    #[error(
        "instruction {instruction_index} has no direct transfer provider for value slot {slot} \
         from {available_storage_classes:?} to {destination_storage_class:?}"
    )]
    MissingTransferProvider {
        instruction_index: usize,
        slot: usize,
        destination_storage_class: StorageClass,
        available_storage_classes: Vec<StorageClass>,
    },
    #[error("value slot {slot} is outside schedule value count {value_count}")]
    ValueSlotOutOfBounds { slot: usize, value_count: usize },
    #[error("scheduled node count exceeds the event-slot identity space")]
    EventSlotExhausted,
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) enum ScheduleExecutionError {
    UnsupportedCollective,
}

#[cfg(test)]
impl fmt::Display for ScheduleExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedCollective => formatter.write_str("collective execution unsupported"),
        }
    }
}

#[cfg(test)]
impl StdError for ScheduleExecutionError {}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Option<usize> {
    values
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value))
}

#[derive(Clone)]
struct AvailableValue {
    location: ExecutionLocation,
    completion: Option<EventCompletion>,
}

fn event_completion(
    nodes: &[ScheduledNode],
    domain: EventDomainId,
) -> Result<EventCompletion, ScheduleBuildError> {
    let slot = u32::try_from(nodes.len()).map_err(|_| ScheduleBuildError::EventSlotExhausted)?;
    Ok(EventCompletion::new(domain, EventSlotId::new(slot), 0))
}
