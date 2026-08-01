use super::super::schedule::{
    EventDependency, EventDomainId, EventSlotId, ExecutionLocation, ScheduledCollective,
    ScheduledGraph, ScheduledNode, ScheduledOperation, ScheduledTransfer, TransferReachability,
};
use super::super::{EngineId, StorageClass, TransferRoute};
use crate::exec::{ExecInstruction, ExecOp, ExecProgram};
use crate::DType;

fn location(engine: &str, domain: u32, storage: &str) -> ExecutionLocation {
    ExecutionLocation::new(
        EngineId::new(engine).expect("engine id"),
        EventDomainId::runtime_created_for_test(domain),
        StorageClass::new(storage).expect("storage class"),
    )
}

fn instruction(
    semantic_operation_index: usize,
    input_slot: usize,
    output_slot: usize,
    last_use: bool,
) -> ExecInstruction {
    ExecInstruction {
        op: ExecOp::Negate,
        semantic_operation_index: Some(semantic_operation_index),
        input_slots: vec![input_slot],
        output_slots: vec![output_slot],
        dtype: DType::F64,
        output_shapes: Default::default(),
        output_extents: Default::default(),
        last_use: vec![last_use],
    }
}

#[test]
fn schedule_emits_location_transfer_and_retains_source_for_split_use() {
    let source = location(
        "tenferro-test.engine.source",
        1,
        "tenferro-test.storage.shared",
    );
    let destination = location(
        "tenferro-test.engine.destination",
        2,
        "tenferro-test.storage.shared",
    );
    let program = ExecProgram {
        instructions: vec![
            instruction(0, 0, 1, true),
            instruction(1, 1, 2, false),
            instruction(2, 1, 3, true),
        ],
        input_slots: vec![0],
        output_slots: vec![2, 3],
        n_slots: 4,
        shape_guards: Vec::new(),
    };

    let graph = ScheduledGraph::from_exec_program(
        &program,
        source.clone(),
        std::slice::from_ref(&source),
        &[source.clone(), destination.clone(), source.clone()],
        &TransferReachability::from([TransferRoute::new(
            source.endpoint().clone(),
            destination.endpoint().clone(),
        )]),
    )
    .expect("schedule");

    assert_eq!(graph.nodes_for_test().len(), 4);
    assert!(matches!(
        &graph.nodes_for_test()[0],
        ScheduledNode::Operation(operation)
            if operation.instruction_index() == 0 && operation.location() == &source
    ));
    assert!(matches!(
        &graph.nodes_for_test()[1],
        ScheduledNode::Transfer(transfer)
            if transfer.value_slot() == 1
                && transfer.source_location() == &source
                && transfer.destination_location() == &destination
    ));
    assert!(matches!(
        &graph.nodes_for_test()[2],
        ScheduledNode::Operation(operation)
            if operation.instruction_index() == 1
                && operation.location() == &destination
                && operation.dependencies()
                    == [EventDependency::new(
                        destination.event_domain_id(),
                        EventSlotId::new(1),
                        0,
                    )]
    ));
    assert!(matches!(
        &graph.nodes_for_test()[3],
        ScheduledNode::Operation(operation)
            if operation.instruction_index() == 2 && operation.location() == &source
    ));
}

#[test]
fn schedule_uses_a_copy_with_a_direct_route_to_the_destination() {
    let first = location(
        "tenferro-test.engine.first",
        1,
        "tenferro-test.storage.first",
    );
    let reachable = location(
        "tenferro-test.engine.reachable",
        2,
        "tenferro-test.storage.reachable",
    );
    let destination = location(
        "tenferro-test.engine.destination",
        3,
        "tenferro-test.storage.destination",
    );
    let program = ExecProgram {
        instructions: vec![
            instruction(0, 0, 1, true),
            instruction(1, 1, 2, false),
            instruction(2, 1, 3, true),
        ],
        input_slots: vec![0],
        output_slots: vec![2, 3],
        n_slots: 4,
        shape_guards: Vec::new(),
    };

    let graph = ScheduledGraph::from_exec_program(
        &program,
        first.clone(),
        std::slice::from_ref(&first),
        &[first.clone(), reachable.clone(), destination.clone()],
        &TransferReachability::from([
            TransferRoute::new(first.endpoint().clone(), reachable.endpoint().clone()),
            TransferRoute::new(reachable.endpoint().clone(), destination.endpoint().clone()),
        ]),
    )
    .expect("schedule");
    let transfers = graph.transfers_for_test().collect::<Vec<_>>();

    assert_eq!(transfers.len(), 2);
    assert_eq!(transfers[1].source_location(), &reachable);
    assert_eq!(transfers[1].destination_location(), &destination);
}

#[test]
fn operation_dependencies_follow_input_order_and_are_deduplicated() {
    let execution = location(
        "tenferro-test.engine.execution",
        1,
        "tenferro-test.storage.execution",
    );
    let program = ExecProgram {
        instructions: vec![
            instruction(0, 0, 1, false),
            instruction(1, 0, 2, true),
            ExecInstruction {
                op: ExecOp::Negate,
                semantic_operation_index: Some(2),
                input_slots: vec![2, 1, 2],
                output_slots: vec![3],
                dtype: DType::F64,
                output_shapes: Default::default(),
                output_extents: Default::default(),
                last_use: vec![true, true, true],
            },
        ],
        input_slots: vec![0],
        output_slots: vec![3],
        n_slots: 4,
        shape_guards: Vec::new(),
    };

    let graph = ScheduledGraph::from_exec_program(
        &program,
        execution.clone(),
        std::slice::from_ref(&execution),
        &[execution.clone(), execution.clone(), execution.clone()],
        &TransferReachability::new(),
    )
    .expect("schedule");
    let ScheduledNode::Operation(operation) = &graph.nodes_for_test()[2] else {
        panic!("third node should be an operation");
    };

    assert_eq!(
        operation.dependencies(),
        &[
            EventDependency::new(
                EventDomainId::runtime_created_for_test(1),
                EventSlotId::new(1),
                0,
            ),
            EventDependency::new(
                EventDomainId::runtime_created_for_test(1),
                EventSlotId::new(0),
                0,
            ),
        ]
    );
}

#[test]
fn schedule_validation_rejects_dependency_without_prior_completion() {
    let source = EventDomainId::runtime_created_for_test(1);
    let destination = EventDomainId::runtime_created_for_test(2);
    let source_location = location(
        "tenferro-test.engine.source",
        1,
        "tenferro-test.storage.source",
    );
    let destination_location = location(
        "tenferro-test.engine.destination",
        2,
        "tenferro-test.storage.destination",
    );
    let graph = ScheduledGraph::for_test(vec![ScheduledNode::Transfer(ScheduledTransfer::new(
        0,
        source_location,
        destination_location,
        [EventDependency::new(source, EventSlotId::new(99), 0)],
        super::super::schedule::EventCompletion::new(destination, EventSlotId::new(0), 0),
    ))]);

    assert!(graph.validate().is_err());
}

#[test]
fn schedule_validation_rejects_duplicate_completion_identity() {
    let domain = EventDomainId::runtime_created_for_test(1);
    let graph = ScheduledGraph::for_test(vec![
        ScheduledNode::Operation(ScheduledOperation::for_test(domain)),
        ScheduledNode::Operation(ScheduledOperation::for_test(domain)),
    ]);

    assert!(matches!(
        graph.validate(),
        Err(super::super::schedule::ScheduleValidationError::DuplicateCompletion { index: 1 })
    ));
}

#[test]
fn retained_bytes_include_operation_dependencies() {
    let execution = location(
        "tenferro-test.engine.execution",
        1,
        "tenferro-test.storage.execution",
    );
    let program = ExecProgram {
        instructions: vec![instruction(0, 0, 1, true), instruction(1, 1, 2, true)],
        input_slots: vec![0],
        output_slots: vec![2],
        n_slots: 3,
        shape_guards: Vec::new(),
    };
    let graph = ScheduledGraph::from_exec_program(
        &program,
        execution.clone(),
        std::slice::from_ref(&execution),
        &[execution.clone(), execution.clone()],
        &TransferReachability::new(),
    )
    .expect("schedule");
    let expected = std::mem::size_of::<ScheduledGraph>()
        + 2 * std::mem::size_of::<ScheduledNode>()
        + 4 * std::mem::size_of::<usize>()
        + std::mem::size_of::<EventDependency>()
        + std::mem::size_of::<usize>()
        + std::mem::size_of::<usize>()
        + 3 * std::mem::size_of::<usize>();

    assert_eq!(graph.retained_bytes(), Some(expected));
}

#[test]
fn transfer_node_bridges_distinct_event_domains() {
    let source = EventDomainId::runtime_created_for_test(1);
    let destination = EventDomainId::runtime_created_for_test(2);
    let transfer = ScheduledTransfer::for_test(source, destination);

    assert_ne!(
        transfer.source_event_domain(),
        transfer.destination_event_domain()
    );
    assert_eq!(transfer.dependencies()[0].domain(), source);
    assert_eq!(transfer.completion().domain(), destination);
}

#[test]
fn collective_node_is_representable_but_execution_is_unsupported() {
    let graph = ScheduledGraph::for_test(vec![ScheduledNode::Collective(
        ScheduledCollective::unsupported_for_test(),
    )]);

    assert!(graph.contains_collective());
    assert!(graph.validate().is_ok());
    assert!(graph
        .execute_for_test()
        .unwrap_err()
        .to_string()
        .contains("collective"));
}

#[test]
fn mock_transfer_does_not_reuse_source_event_domain_as_destination_completion() {
    let source = EventDomainId::runtime_created_for_test(7);
    let destination = EventDomainId::runtime_created_for_test(8);
    let graph = ScheduledGraph::for_test(vec![
        ScheduledNode::Operation(ScheduledOperation::for_test(source)),
        ScheduledNode::Transfer(ScheduledTransfer::for_test(source, destination)),
        ScheduledNode::Operation(ScheduledOperation::new(
            1,
            location(
                "tenferro-test.engine.destination",
                8,
                "tenferro-test.storage.destination",
            ),
            [],
            [],
            [EventDependency::new(destination, EventSlotId::new(0), 0)],
            super::super::schedule::EventCompletion::new(destination, EventSlotId::new(1), 0),
        )),
    ]);

    let transfer = graph
        .transfers_for_test()
        .next()
        .expect("scheduled transfer");

    assert_eq!(transfer.dependencies()[0].domain(), source);
    assert_eq!(transfer.completion().domain(), destination);
    assert_ne!(
        graph.nodes_for_test()[0].completion().domain(),
        transfer.completion().domain()
    );
    assert!(graph.validate().is_ok());
}
