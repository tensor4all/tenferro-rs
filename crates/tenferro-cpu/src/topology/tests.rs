use super::*;

#[test]
fn cpu_sets_are_sorted_and_deduplicated() {
    let cpus = CpuSet::new([CpuId::new(11), CpuId::new(8), CpuId::new(11), CpuId::new(9)]).unwrap();

    assert_eq!(cpus.as_usize_vec(), vec![8, 9, 11]);
    assert!(cpus.contains(CpuId::new(9)));
    assert!(!cpus.contains(CpuId::new(10)));
}

#[test]
fn cpu_set_clones_share_the_immutable_cpu_domain() {
    let original = cpu_set([2, 5, 8]);
    let cloned = original.clone();

    assert!(Arc::ptr_eq(&original.cpus, &cloned.cpus));
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

#[test]
fn topology_rejects_duplicate_node_ids_even_when_one_copy_is_unusable() {
    let result = CpuTopology::from_discovered(
        cpu_set([0, 1]),
        [
            (NumaNodeId::new(2), cpu_set([8, 9])),
            (NumaNodeId::new(2), cpu_set([0, 1])),
        ],
    );

    assert_eq!(
        result,
        Err(CpuTopologyError::DuplicateNode {
            node: NumaNodeId::new(2)
        })
    );
}

#[test]
fn linux_cpu_list_parser_handles_ranges_gaps_and_whitespace() {
    assert_eq!(
        parse_linux_cpu_list("0-3,8,10-11\n")
            .unwrap()
            .as_usize_vec(),
        vec![0, 1, 2, 3, 8, 10, 11],
    );
}

#[test]
fn linux_cpu_list_parser_rejects_reversed_or_malformed_ranges() {
    assert!(parse_linux_cpu_list("4-2").is_err());
    assert!(parse_linux_cpu_list("0,word,2").is_err());
    assert!(parse_linux_cpu_list("0-").is_err());
}

#[test]
fn discovery_falls_back_to_all_allowed_when_node_files_are_unavailable() {
    let source = FixtureTopologySource {
        allowed: cpu_set([4, 5]),
        nodes: None,
    };
    let topology = discover_from(&source).unwrap();

    assert!(topology.nodes().is_empty());
    assert_eq!(topology.allowed_cpus().as_usize_vec(), vec![4, 5]);
}

#[test]
fn discovery_parses_sparse_nodes_then_applies_process_affinity() {
    let source = FixtureTopologySource {
        allowed: cpu_set([8, 9, 12, 13]),
        nodes: Some(vec![
            (NumaNodeId::new(2), "0-9".to_owned()),
            (NumaNodeId::new(7), "12-15".to_owned()),
            (NumaNodeId::new(9), "20-23".to_owned()),
        ]),
    };
    let topology = discover_from(&source).unwrap();

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
        vec![8, 9]
    );
}

#[test]
fn live_discovery_stays_within_the_process_affinity_mask() {
    let topology = discover_cpu_topology().unwrap();
    if let Some(process_cpus) = crate::process_cpu_affinity() {
        assert_eq!(topology.allowed_cpus(), &process_cpus);
    }
    for node in topology.nodes() {
        assert!(node
            .cpus()
            .as_slice()
            .iter()
            .all(|cpu| topology.allowed_cpus().contains(*cpu)));
    }
}

struct FixtureTopologySource {
    allowed: CpuSet,
    nodes: Option<Vec<(NumaNodeId, String)>>,
}

impl TopologySource for FixtureTopologySource {
    fn allowed_cpus(&self) -> Result<CpuSet, CpuTopologyError> {
        Ok(self.allowed.clone())
    }

    fn numa_node_cpu_lists(&self) -> Result<Option<Vec<(NumaNodeId, String)>>, CpuTopologyError> {
        Ok(self.nodes.clone())
    }
}

fn cpu_set<const N: usize>(cpus: [usize; N]) -> CpuSet {
    CpuSet::new(cpus.map(CpuId::new)).unwrap()
}
