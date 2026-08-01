//! Runtime-owned scheduled graph boundary.
//!
//! This representation remains crate-private. Later phases attach native
//! GPU/XLA dispatch and asynchronous completion to the same node families.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
#[cfg(test)]
use std::num::NonZeroU64;
use std::sync::Arc;

use super::snapshot::ExecutableEngineSnapshot;
use super::{
    FrozenTransferRegistry, RegistrationIdentity, ResolvedTransferEndpoint, ResolvedTransferRoute,
    RuntimeEpoch, RuntimeId,
};
use crate::exec::ExecProgram;
use crate::{EngineId, StorageClass, TransferEndpoint};

/// Provenance-qualified identity of one frozen direct-engine event domain.
///
/// An event domain is identified by the runtime that owns it, the immutable
/// configuration epoch that published it, and the frozen registration identity
/// of the direct engine slot. It is not a reusable slot number.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::EventDomainId;
///
/// fn inspect(domain: EventDomainId) {
///     let _runtime = domain.runtime_id();
///     let _epoch = domain.epoch();
///     let _registration = domain.registration_identity();
/// }
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EventDomainId {
    runtime_id: RuntimeId,
    epoch: RuntimeEpoch,
    registration_identity: RegistrationIdentity,
}

impl EventDomainId {
    pub(crate) const fn new(
        runtime_id: RuntimeId,
        epoch: RuntimeEpoch,
        registration_identity: RegistrationIdentity,
    ) -> Self {
        Self {
            runtime_id,
            epoch,
            registration_identity,
        }
    }

    /// Return the runtime that owns this event domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::EventDomainId;
    ///
    /// fn inspect(domain: EventDomainId) {
    ///     let _runtime = domain.runtime_id();
    /// }
    /// ```
    #[must_use]
    pub const fn runtime_id(self) -> RuntimeId {
        self.runtime_id
    }

    /// Return the immutable configuration epoch that published this domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::EventDomainId;
    ///
    /// fn inspect(domain: EventDomainId) {
    ///     let _epoch = domain.epoch();
    /// }
    /// ```
    #[must_use]
    pub const fn epoch(self) -> RuntimeEpoch {
        self.epoch
    }

    /// Return the frozen direct-engine registration identity for this domain.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::EventDomainId;
    ///
    /// fn inspect(domain: EventDomainId) {
    ///     let _registration = domain.registration_identity();
    /// }
    /// ```
    #[must_use]
    pub const fn registration_identity(self) -> RegistrationIdentity {
        self.registration_identity
    }

    #[cfg(test)]
    pub(crate) const fn runtime_created_for_test(
        runtime_id: RuntimeId,
        epoch: RuntimeEpoch,
        registration_identity: RegistrationIdentity,
    ) -> Self {
        Self::new(runtime_id, epoch, registration_identity)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ExecutionLocation {
    resolved_endpoint: ResolvedTransferEndpoint,
    witness: Arc<ExecutableEngineSnapshot>,
}

impl ExecutionLocation {
    pub(super) fn from_witness(
        witness: Arc<ExecutableEngineSnapshot>,
        storage_class: StorageClass,
    ) -> Self {
        Self {
            resolved_endpoint: ResolvedTransferEndpoint::new(
                TransferEndpoint::new(witness.engine_id().clone(), storage_class),
                witness.provider_device_identity().clone(),
                witness.event_domain_id(),
            ),
            witness,
        }
    }

    #[cfg(test)]
    pub(crate) fn new(
        engine_id: EngineId,
        provider_device_identity: super::ProviderDeviceIdentity,
        event_domain_id: EventDomainId,
        storage_class: StorageClass,
    ) -> Self {
        Self::from_witness(
            super::snapshot::ExecutableEngineSnapshot::for_test(
                engine_id,
                provider_device_identity,
                event_domain_id,
                storage_class.clone(),
            ),
            storage_class,
        )
    }

    pub(crate) fn engine_id(&self) -> &EngineId {
        self.resolved_endpoint.logical().engine_id()
    }

    pub(crate) fn endpoint(&self) -> &TransferEndpoint {
        self.resolved_endpoint.logical()
    }

    pub(crate) fn event_domain_id(&self) -> EventDomainId {
        self.resolved_endpoint.event_domain_id()
    }

    pub(crate) fn provider_device_identity(&self) -> &super::ProviderDeviceIdentity {
        self.resolved_endpoint.provider_device_identity()
    }

    pub(crate) fn resolved_endpoint(&self) -> &ResolvedTransferEndpoint {
        &self.resolved_endpoint
    }

    pub(crate) fn storage_class(&self) -> &StorageClass {
        self.resolved_endpoint.logical().storage_class()
    }

    pub(super) fn witness(&self) -> &Arc<ExecutableEngineSnapshot> {
        &self.witness
    }

    #[cfg(test)]
    fn for_test(
        domain: EventDomainId,
        provider_device_identity: super::ProviderDeviceIdentity,
    ) -> Self {
        Self::new(
            EngineId::new("tenferro-test.schedule-engine").expect("test engine id"),
            provider_device_identity,
            domain,
            StorageClass::new("tenferro-test.schedule-storage").expect("test storage class"),
        )
    }
}

impl PartialEq for ExecutionLocation {
    fn eq(&self, other: &Self) -> bool {
        self.resolved_endpoint == other.resolved_endpoint
    }
}

impl Eq for ExecutionLocation {}

impl Hash for ExecutionLocation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.resolved_endpoint.hash(state);
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

    pub(crate) fn from_completion(completion: EventCompletion) -> Self {
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
    dependencies: Box<[EventDependency]>,
    completion: EventCompletion,
}

impl ScheduledOperation {
    pub(crate) fn new(
        instruction_index: usize,
        location: ExecutionLocation,
        input_values: impl Into<Box<[usize]>>,
        output_values: impl Into<Box<[usize]>>,
        dependencies: impl Into<Box<[EventDependency]>>,
        completion: EventCompletion,
    ) -> Self {
        Self {
            instruction_index,
            location,
            input_values: input_values.into(),
            output_values: output_values.into(),
            dependencies: dependencies.into(),
            completion,
        }
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        domain: EventDomainId,
        provider_device_identity: super::ProviderDeviceIdentity,
    ) -> Self {
        Self::new(
            0,
            ExecutionLocation::for_test(domain, provider_device_identity),
            [],
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

    pub(crate) fn dependencies(&self) -> &[EventDependency] {
        &self.dependencies
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
            self.dependencies
                .len()
                .checked_mul(std::mem::size_of::<EventDependency>())?,
        ])
    }
}

#[derive(Clone)]
pub(crate) struct ScheduledTransfer {
    value_slot: usize,
    source_location: ExecutionLocation,
    destination_location: ExecutionLocation,
    provider: Arc<dyn super::TransferProvider>,
    dependencies: Box<[EventDependency]>,
    completion: EventCompletion,
}

impl ScheduledTransfer {
    pub(crate) fn with_provider(
        value_slot: usize,
        source_location: ExecutionLocation,
        destination_location: ExecutionLocation,
        provider: Arc<dyn super::TransferProvider>,
        dependencies: impl Into<Box<[EventDependency]>>,
        completion: EventCompletion,
    ) -> Self {
        Self {
            value_slot,
            source_location,
            destination_location,
            provider,
            dependencies: dependencies.into(),
            completion,
        }
    }

    #[cfg(test)]
    pub(crate) fn new(
        value_slot: usize,
        source_location: ExecutionLocation,
        destination_location: ExecutionLocation,
        dependencies: impl Into<Box<[EventDependency]>>,
        completion: EventCompletion,
    ) -> Self {
        Self::with_provider(
            value_slot,
            source_location,
            destination_location,
            Arc::new(TestScheduledTransferProvider),
            dependencies,
            completion,
        )
    }

    #[cfg(test)]
    pub(crate) fn for_test(
        source_event_domain: EventDomainId,
        destination_event_domain: EventDomainId,
        source_provider_device_identity: super::ProviderDeviceIdentity,
        destination_provider_device_identity: super::ProviderDeviceIdentity,
    ) -> Self {
        let source_location =
            ExecutionLocation::for_test(source_event_domain, source_provider_device_identity);
        let destination_location = ExecutionLocation::for_test(
            destination_event_domain,
            destination_provider_device_identity,
        );
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

    pub(crate) fn provider(&self) -> &Arc<dyn super::TransferProvider> {
        &self.provider
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

impl std::fmt::Debug for ScheduledTransfer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ScheduledTransfer")
            .field("value_slot", &self.value_slot)
            .field("source_location", &self.source_location)
            .field("destination_location", &self.destination_location)
            .field("dependencies", &self.dependencies)
            .field("completion", &self.completion)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScheduledCollective {
    location: ExecutionLocation,
    dependencies: Box<[EventDependency]>,
    completion: EventCompletion,
}

impl ScheduledCollective {
    #[cfg(test)]
    pub(crate) fn unsupported_for_test() -> Self {
        let domain = EventDomainId::runtime_created_for_test(
            RuntimeId::from_nonzero(NonZeroU64::new(1).expect("runtime id")),
            RuntimeEpoch::from_nonzero(NonZeroU64::new(1).expect("runtime epoch")),
            RegistrationIdentity::new(
                NonZeroU64::new(1).expect("registration issuer"),
                NonZeroU64::new(1).expect("registration ordinal"),
            ),
        );
        let location = ExecutionLocation::for_test(
            domain,
            super::ProviderDeviceIdentity::new(
                super::ProviderId::new("tenferro.test.schedule").expect("test provider id"),
                "collective",
            )
            .expect("test provider target"),
        );
        Self {
            location,
            dependencies: Box::new([]),
            completion: EventCompletion::new(domain, EventSlotId::new(0), 0),
        }
    }

    pub(crate) fn location(&self) -> &ExecutionLocation {
        &self.location
    }

    pub(crate) fn completion(&self) -> EventCompletion {
        self.completion
    }

    fn dependencies(&self) -> &[EventDependency] {
        &self.dependencies
    }

    fn retained_bytes(&self) -> Option<usize> {
        self.dependencies
            .len()
            .checked_mul(std::mem::size_of::<EventDependency>())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ScheduledBarrier {
    location: ExecutionLocation,
    dependencies: Box<[EventDependency]>,
    completion: EventCompletion,
}

impl ScheduledBarrier {
    pub(crate) fn location(&self) -> &ExecutionLocation {
        &self.location
    }

    fn dependencies(&self) -> &[EventDependency] {
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
    // INVARIANT: execution supports explicit barriers, but schedule construction
    // does not emit them until a later scheduling-policy change requires one.
    #[allow(
        dead_code,
        reason = "barrier construction is deferred until scheduling policy emits explicit barriers"
    )]
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

    pub(crate) fn dependencies(&self) -> &[EventDependency] {
        match self {
            Self::Operation(node) => node.dependencies(),
            Self::Transfer(node) => node.dependencies(),
            Self::Collective(node) => node.dependencies(),
            Self::Barrier(node) => node.dependencies(),
        }
    }

    pub(super) fn event_domain_witness(&self) -> &Arc<ExecutableEngineSnapshot> {
        match self {
            Self::Operation(node) => node.location().witness(),
            Self::Transfer(node) => node.destination_location().witness(),
            Self::Collective(node) => node.location().witness(),
            Self::Barrier(node) => node.location().witness(),
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
    root_location: ExecutionLocation,
    input_locations: Box<[ExecutionLocation]>,
    operation_locations: Box<[ExecutionLocation]>,
}

impl ScheduledGraph {
    pub(crate) fn from_exec_program(
        program: &ExecProgram,
        root_location: ExecutionLocation,
        input_locations: &[ExecutionLocation],
        operation_locations: &[ExecutionLocation],
        transfer_registry: &FrozenTransferRegistry,
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
                    .find_map(|value| {
                        let route = ResolvedTransferRoute::new(
                            value.location.resolved_endpoint().clone(),
                            location.resolved_endpoint().clone(),
                        );
                        transfer_registry
                            .get(&route)
                            .map(|provider| (value.clone(), Arc::clone(provider)))
                    })
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
                                destination_endpoint: location.endpoint().clone(),
                                available_source_endpoints: values
                                    .iter()
                                    .map(|value| value.location.endpoint().clone())
                                    .collect(),
                            }
                        }
                    })?;
                let completion = event_completion(&nodes, location.event_domain_id())?;
                let dependencies = source
                    .0
                    .completion
                    .map(EventDependency::from_completion)
                    .into_iter()
                    .collect::<Vec<_>>();
                nodes.push(ScheduledNode::Transfer(ScheduledTransfer::with_provider(
                    slot,
                    source.0.location,
                    location.clone(),
                    source.1,
                    dependencies,
                    completion,
                )));
                available[slot].push(AvailableValue {
                    location: location.clone(),
                    completion: Some(completion),
                });
            }

            let dependencies = operation_dependencies(
                &available,
                instruction_index,
                &instruction.input_slots,
                &location,
            )?;
            let completion = event_completion(&nodes, location.event_domain_id())?;
            nodes.push(ScheduledNode::Operation(ScheduledOperation::new(
                instruction_index,
                location.clone(),
                instruction.input_slots.clone(),
                instruction.output_slots.clone(),
                dependencies,
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

        let graph = Self {
            nodes: nodes.into_boxed_slice(),
            input_slots: program.input_slots.clone().into_boxed_slice(),
            output_slots: program.output_slots.clone().into_boxed_slice(),
            value_count: program.n_slots,
            root_location,
            input_locations: input_locations.to_vec().into_boxed_slice(),
            operation_locations: operation_locations.to_vec().into_boxed_slice(),
        };
        graph
            .validate()
            .map_err(|source| ScheduleBuildError::InvalidSchedule { source })?;
        Ok(graph)
    }

    #[cfg(test)]
    pub(crate) fn for_test(nodes: Vec<ScheduledNode>) -> Self {
        let value_count = nodes.len();
        let root_location = nodes
            .iter()
            .find_map(|node| match node {
                ScheduledNode::Operation(operation) => Some(operation.location().clone()),
                ScheduledNode::Transfer(transfer) => Some(transfer.destination_location().clone()),
                ScheduledNode::Collective(_) | ScheduledNode::Barrier(_) => None,
            })
            .unwrap_or_else(|| {
                ExecutionLocation::new(
                    EngineId::new("tenferro-test.schedule-fallback-engine")
                        .expect("test engine id"),
                    super::ProviderDeviceIdentity::new(
                        super::ProviderId::new("tenferro.test.schedule").expect("test provider id"),
                        "fallback",
                    )
                    .expect("test provider target"),
                    EventDomainId::runtime_created_for_test(
                        RuntimeId::from_nonzero(NonZeroU64::new(1).expect("runtime id")),
                        RuntimeEpoch::from_nonzero(NonZeroU64::new(1).expect("runtime epoch")),
                        RegistrationIdentity::new(
                            NonZeroU64::new(1).expect("registration issuer"),
                            NonZeroU64::new(1).expect("registration ordinal"),
                        ),
                    ),
                    StorageClass::new("tenferro-test.schedule-fallback-storage")
                        .expect("test storage class"),
                )
            });
        Self {
            nodes: nodes.into_boxed_slice(),
            input_slots: Box::new([]),
            output_slots: Box::new([]),
            value_count,
            root_location,
            input_locations: Box::new([]),
            operation_locations: Box::new([]),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), ScheduleValidationError> {
        let mut known_completions = HashSet::with_capacity(self.nodes.len());
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
                            && dependency.domain()
                                != transfer.destination_location().event_domain_id()
                    }) {
                        return Err(ScheduleValidationError::DependencyEventDomainMismatch {
                            index,
                        });
                    }
                }
                ScheduledNode::Collective(collective) => {
                    if collective.completion().domain() != collective.location().event_domain_id() {
                        return Err(ScheduleValidationError::CompletionEventDomainMismatch {
                            index,
                        });
                    }
                }
                ScheduledNode::Barrier(barrier) => {
                    if barrier.completion().domain() != barrier.location().event_domain_id() {
                        return Err(ScheduleValidationError::CompletionEventDomainMismatch {
                            index,
                        });
                    }
                }
            }
            if node
                .dependencies()
                .iter()
                .any(|dependency| !known_completions.contains(dependency))
            {
                return Err(ScheduleValidationError::DependencyNotPriorCompletion { index });
            }
            if !known_completions.insert(EventDependency::from_completion(node.completion())) {
                return Err(ScheduleValidationError::DuplicateCompletion { index });
            }
        }
        Ok(())
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

    pub(crate) fn root_location(&self) -> &ExecutionLocation {
        &self.root_location
    }

    pub(crate) fn input_locations(&self) -> &[ExecutionLocation] {
        &self.input_locations
    }

    pub(crate) fn operation_locations(&self) -> &[ExecutionLocation] {
        &self.operation_locations
    }

    /// Reject non-executable node kinds before any provider event-domain run
    /// is acquired. Preparation currently emits operations, transfers, and
    /// no-op barriers; collective execution remains unsupported.
    pub(crate) fn preflight(&self) -> Result<(), SchedulePreflightError> {
        if let Some(index) = self
            .nodes
            .iter()
            .position(|node| matches!(node, ScheduledNode::Collective(_)))
        {
            return Err(SchedulePreflightError::UnsupportedCollective { index });
        }
        Ok(())
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
            self.input_locations
                .len()
                .checked_mul(std::mem::size_of::<ExecutionLocation>())?,
            self.operation_locations
                .len()
                .checked_mul(std::mem::size_of::<ExecutionLocation>())?,
        ])
    }

    #[cfg(test)]
    pub(crate) fn execute_for_test(&self) -> Result<(), ScheduleExecutionError> {
        self.preflight()
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
    #[error("schedule node {index} dependency does not refer to a prior completion")]
    DependencyNotPriorCompletion { index: usize },
    #[error("schedule node {index} reuses a prior completion identity")]
    DuplicateCompletion { index: usize },
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ScheduleBuildError {
    #[error("schedule construction produced an invalid immutable schedule")]
    InvalidSchedule {
        #[source]
        source: ScheduleValidationError,
    },
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
         from {available_source_endpoints:?} to {destination_endpoint:?}"
    )]
    MissingTransferProvider {
        instruction_index: usize,
        slot: usize,
        destination_endpoint: TransferEndpoint,
        available_source_endpoints: Vec<TransferEndpoint>,
    },
    #[error("value slot {slot} is outside schedule value count {value_count}")]
    ValueSlotOutOfBounds { slot: usize, value_count: usize },
    #[error("scheduled node count exceeds the event-slot identity space")]
    EventSlotExhausted,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum SchedulePreflightError {
    #[error("scheduled node {index} is a collective, but collective execution is unsupported")]
    UnsupportedCollective { index: usize },
}

#[cfg(test)]
type ScheduleExecutionError = SchedulePreflightError;

#[cfg(test)]
#[derive(Debug)]
struct TestScheduledTransferProvider;

#[cfg(test)]
impl super::TransferProvider for TestScheduledTransferProvider {
    fn transfer_blocking(
        &self,
        _request: super::TransferRequest<'_>,
    ) -> crate::Result<tenferro_tensor::Tensor> {
        Err(crate::Error::Internal(
            "scheduled test transfer provider".into(),
        ))
    }
}

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

fn operation_dependencies(
    available: &[Vec<AvailableValue>],
    instruction_index: usize,
    input_slots: &[usize],
    location: &ExecutionLocation,
) -> Result<Vec<EventDependency>, ScheduleBuildError> {
    let mut dependencies = Vec::with_capacity(input_slots.len());
    let mut seen = HashSet::with_capacity(input_slots.len());
    for &slot in input_slots {
        let values = available
            .get(slot)
            .ok_or(ScheduleBuildError::ValueSlotOutOfBounds {
                slot,
                value_count: available.len(),
            })?;
        let value = values
            .iter()
            .find(|value| &value.location == location)
            .ok_or(ScheduleBuildError::ValueUnavailable {
                instruction_index,
                slot,
            })?;
        if let Some(completion) = value.completion {
            let dependency = EventDependency::from_completion(completion);
            if seen.insert(dependency) {
                dependencies.push(dependency);
            }
        }
    }
    Ok(dependencies)
}

fn event_completion(
    nodes: &[ScheduledNode],
    domain: EventDomainId,
) -> Result<EventCompletion, ScheduleBuildError> {
    let slot = u32::try_from(nodes.len()).map_err(|_| ScheduleBuildError::EventSlotExhausted)?;
    Ok(EventCompletion::new(domain, EventSlotId::new(slot), 0))
}
