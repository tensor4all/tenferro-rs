use super::*;
use crate::{CpuBackendKind, CpuId, CpuSet, CpuTopology, NumaNodeId};

#[test]
fn faer_resolves_auto_and_explicit_managed_placements() {
    let topology = two_node_fixture();

    assert!(matches!(
        resolve_placement(CpuBackendKind::Faer, CpuPlacement::Auto, &topology).unwrap(),
        ResolvedCpuExecution::Managed(ResolvedCpuPlacement::AllAllowed { .. })
    ));
    assert!(matches!(
        resolve_placement(
            CpuBackendKind::Faer,
            CpuPlacement::NumaNode(NumaNodeId::new(7)),
            &topology,
        )
        .unwrap(),
        ResolvedCpuExecution::Managed(ResolvedCpuPlacement::NumaNode { id, .. })
            if id == NumaNodeId::new(7)
    ));
}

#[test]
fn explicit_external_provider_placement_never_falls_back() {
    let topology = two_node_fixture();
    assert!(matches!(
        resolve_placement(
            CpuBackendKind::Blas,
            CpuPlacement::NumaNode(NumaNodeId::new(2)),
            &topology,
        ),
        Err(CpuPlacementError::ExternalProviderAffinityUnmanaged { .. })
    ));
    assert!(matches!(
        resolve_placement(CpuBackendKind::Blas, CpuPlacement::AllAllowed, &topology),
        Err(CpuPlacementError::ExternalProviderAffinityUnmanaged { .. })
    ));
    assert_eq!(
        resolve_placement(CpuBackendKind::Blas, CpuPlacement::Auto, &topology).unwrap(),
        ResolvedCpuExecution::ProviderDefaultExclusive,
    );
}

#[test]
fn explicit_unknown_node_is_an_error_without_fallback() {
    let topology = two_node_fixture();
    assert!(matches!(
        resolve_placement(
            CpuBackendKind::Faer,
            CpuPlacement::NumaNode(NumaNodeId::new(9)),
            &topology,
        ),
        Err(CpuPlacementError::UnknownNumaNode { node, .. })
            if node == NumaNodeId::new(9)
    ));
}

#[test]
fn explicit_node_requires_discovered_numa_domains() {
    let topology = CpuTopology::from_discovered(cpu_set([4, 5]), []).unwrap();
    assert!(matches!(
        resolve_placement(
            CpuBackendKind::Faer,
            CpuPlacement::NumaNode(NumaNodeId::new(0)),
            &topology,
        ),
        Err(CpuPlacementError::NumaDiscoveryUnavailable { .. })
    ));
}

fn two_node_fixture() -> CpuTopology {
    CpuTopology::from_discovered(
        cpu_set([8, 9, 12, 13]),
        [
            (NumaNodeId::new(2), cpu_set([8, 9])),
            (NumaNodeId::new(7), cpu_set([12, 13])),
        ],
    )
    .unwrap()
}

fn cpu_set<const N: usize>(cpus: [usize; N]) -> CpuSet {
    CpuSet::new(cpus.map(CpuId::new)).unwrap()
}
