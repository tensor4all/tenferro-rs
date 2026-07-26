use super::super::schedule::{
    EventDomainId, ScheduledCollective, ScheduledGraph, ScheduledNode, ScheduledOperation,
    ScheduledTransfer,
};

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
