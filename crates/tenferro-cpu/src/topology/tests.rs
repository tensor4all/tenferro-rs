use super::*;

#[test]
fn cpu_sets_are_sorted_and_deduplicated() {
    let cpus = CpuSet::new([CpuId::new(11), CpuId::new(8), CpuId::new(11), CpuId::new(9)]).unwrap();

    assert_eq!(cpus.as_usize_vec(), vec![8, 9, 11]);
    assert!(cpus.contains(CpuId::new(9)));
    assert!(!cpus.contains(CpuId::new(10)));
}

#[test]
fn empty_cpu_sets_are_rejected() {
    assert_eq!(CpuSet::new([]), Err(CpuSetError::Empty));
}

#[test]
fn topology_intersects_nodes_with_allowed_cpus_without_renumbering() {
    let allowed = cpu_set([8, 9, 10, 11, 12, 13, 14, 15]);
    let topology = CpuTopology::from_discovered(
        allowed.clone(),
        [
            (NumaNodeId::new(2), cpu_set([0, 8, 9, 10, 11])),
            (NumaNodeId::new(7), cpu_set([12, 13, 14, 15, 16])),
            (NumaNodeId::new(9), cpu_set([20, 21, 22, 23, 24])),
        ],
    )
    .unwrap();

    assert_eq!(topology.allowed_cpus(), &allowed);
    assert_eq!(
        topology.node_ids(),
        vec![NumaNodeId::new(2), NumaNodeId::new(7)]
    );
    assert_eq!(
        topology
            .node(NumaNodeId::new(2))
            .unwrap()
            .cpus()
            .as_usize_vec(),
        vec![8, 9, 10, 11]
    );
}

#[test]
fn topology_rejects_overlapping_usable_nodes() {
    let result = CpuTopology::from_discovered(
        cpu_set([0, 1, 2]),
        [
            (NumaNodeId::new(2), cpu_set([0, 1])),
            (NumaNodeId::new(7), cpu_set([1, 2])),
        ],
    );

    assert!(matches!(
        result,
        Err(CpuTopologyError::OverlappingNodes {
            first,
            second,
            ..
        }) if first == NumaNodeId::new(2) && second == NumaNodeId::new(7)
    ));
}

fn cpu_set<const N: usize>(cpus: [usize; N]) -> CpuSet {
    CpuSet::new(cpus.map(CpuId::new)).unwrap()
}
