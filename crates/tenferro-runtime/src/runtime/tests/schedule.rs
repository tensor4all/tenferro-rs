use super::super::schedule::{
    EventDomainId, ExecutionLocation, ScheduledCollective, ScheduledGraph, ScheduledNode,
    ScheduledOperation, ScheduledTransfer,
};
use super::super::{EngineId, StorageClass};
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
            if operation.instruction_index() == 1 && operation.location() == &destination
    ));
    assert!(matches!(
        &graph.nodes_for_test()[3],
        ScheduledNode::Operation(operation)
            if operation.instruction_index() == 2 && operation.location() == &source
    ));
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
        ScheduledNode::Operation(ScheduledOperation::for_test(destination)),
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
