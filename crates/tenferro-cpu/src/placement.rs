use thiserror::Error;

use crate::{CpuBackendKind, CpuSet, CpuTopology, CpuTopologyError, NumaNodeId};

/// Requested CPU execution placement.
///
/// `AllAllowed` means all logical CPUs permitted by the process affinity mask,
/// not every CPU installed in the host.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuPlacement, NumaNodeId};
///
/// let placement = CpuPlacement::NumaNode(NumaNodeId::new(2));
/// assert!(matches!(placement, CpuPlacement::NumaNode(_)));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CpuPlacement {
    /// Let the selected provider choose its compatible default policy.
    #[default]
    Auto,
    /// Restrict managed tenferro/faer execution to one usable OS NUMA node.
    NumaNode(NumaNodeId),
    /// Use the complete CPU set permitted to the process.
    AllAllowed,
}

/// Concrete managed placement after topology resolution.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuId, CpuSet, ResolvedCpuPlacement};
///
/// let placement = ResolvedCpuPlacement::AllAllowed {
///     cpus: CpuSet::new([CpuId::new(0)])?,
/// };
/// assert_eq!(placement.cpus().len(), 1);
/// # Ok::<(), tenferro_cpu::CpuSetError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedCpuPlacement {
    /// A selected usable OS NUMA node.
    NumaNode {
        /// The sparse OS NUMA node ID.
        id: NumaNodeId,
        /// The node CPUs permitted by the process affinity mask.
        cpus: CpuSet,
    },
    /// The complete process affinity CPU set.
    AllAllowed {
        /// Logical CPUs permitted by the process affinity mask.
        cpus: CpuSet,
    },
}

impl ResolvedCpuPlacement {
    /// Return the concrete logical CPU set used for pinning and arbitration.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet, ResolvedCpuPlacement};
    ///
    /// let placement = ResolvedCpuPlacement::AllAllowed {
    ///     cpus: CpuSet::new([CpuId::new(1), CpuId::new(2)])?,
    /// };
    /// assert_eq!(placement.cpus().as_usize_vec(), vec![1, 2]);
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn cpus(&self) -> &CpuSet {
        match self {
            Self::NumaNode { cpus, .. } | Self::AllAllowed { cpus } => cpus,
        }
    }

    /// Return the OS NUMA node ID for a node placement.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet, NumaNodeId, ResolvedCpuPlacement};
    ///
    /// let placement = ResolvedCpuPlacement::NumaNode {
    ///     id: NumaNodeId::new(7),
    ///     cpus: CpuSet::new([CpuId::new(3)])?,
    /// };
    /// assert_eq!(placement.node_id(), Some(NumaNodeId::new(7)));
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn node_id(&self) -> Option<NumaNodeId> {
        match self {
            Self::NumaNode { id, .. } => Some(*id),
            Self::AllAllowed { .. } => None,
        }
    }
}

/// Failure to resolve a CPU placement for the selected public provider kind.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuBackendKind, CpuPlacement, CpuPlacementError};
///
/// let error = CpuPlacementError::ExternalProviderAffinityUnmanaged {
///     requested: CpuPlacement::AllAllowed,
///     backend: CpuBackendKind::Blas,
/// };
/// assert!(error.to_string().contains("affinity"));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CpuPlacementError {
    /// Process-visible topology discovery failed before placement resolution.
    #[error("cannot resolve {requested:?} for {backend:?}: topology discovery failed: {source}")]
    TopologyDiscovery {
        /// The placement requested by the caller.
        requested: CpuPlacement,
        /// The selected public backend kind.
        backend: CpuBackendKind,
        /// The preserved topology failure category.
        #[source]
        source: CpuTopologyError,
    },
    /// The current platform cannot construct verified pinned worker pools.
    #[error(
        "cannot resolve {requested:?} for {backend:?}: managed worker affinity is unavailable"
    )]
    ManagedAffinityUnavailable {
        /// The explicit placement requested by the caller.
        requested: CpuPlacement,
        /// The selected public backend kind.
        backend: CpuBackendKind,
    },
    /// NUMA-node placement was requested but OS NUMA discovery was unavailable.
    #[error("cannot resolve {requested:?} for {backend:?}: NUMA discovery is unavailable")]
    NumaDiscoveryUnavailable {
        /// The placement requested by the caller.
        requested: CpuPlacement,
        /// The selected public backend kind.
        backend: CpuBackendKind,
    },
    /// The requested OS NUMA node has no usable CPUs in this process.
    #[error("cannot resolve {requested:?} for {backend:?}: NUMA node {node} is unavailable")]
    UnknownNumaNode {
        /// The placement requested by the caller.
        requested: CpuPlacement,
        /// The selected public backend kind.
        backend: CpuBackendKind,
        /// The unknown or process-unavailable OS node ID.
        node: NumaNodeId,
    },
    /// An external provider owns worker affinity, so explicit placement is unsafe.
    #[error(
        "cannot resolve {requested:?} for {backend:?}: external provider worker affinity is unmanaged"
    )]
    ExternalProviderAffinityUnmanaged {
        /// The explicit placement requested by the caller.
        requested: CpuPlacement,
        /// The selected public backend kind.
        backend: CpuBackendKind,
    },
    /// A pinned engine could not be built for an otherwise valid placement.
    #[error("cannot resolve {requested:?} for {backend:?}: engine construction failed: {message}")]
    EngineConstruction {
        /// The placement requested by the caller.
        requested: CpuPlacement,
        /// The selected public backend kind.
        backend: CpuBackendKind,
        /// Worker-pool construction or affinity verification detail.
        message: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ResolvedCpuExecution {
    Compatibility,
    Managed(ResolvedCpuPlacement),
    ProviderDefaultExclusive,
}

pub(crate) fn resolve_placement(
    backend: CpuBackendKind,
    requested: CpuPlacement,
    topology: &CpuTopology,
) -> Result<ResolvedCpuExecution, CpuPlacementError> {
    resolve_placement_with_affinity(
        backend,
        requested,
        topology,
        cfg!(any(target_os = "linux", target_os = "android")),
    )
}

pub(crate) fn resolve_placement_with_affinity(
    backend: CpuBackendKind,
    requested: CpuPlacement,
    topology: &CpuTopology,
    managed_affinity_available: bool,
) -> Result<ResolvedCpuExecution, CpuPlacementError> {
    if backend == CpuBackendKind::Blas {
        return match requested {
            CpuPlacement::Auto => Ok(ResolvedCpuExecution::ProviderDefaultExclusive),
            CpuPlacement::NumaNode(_) | CpuPlacement::AllAllowed => {
                Err(CpuPlacementError::ExternalProviderAffinityUnmanaged { requested, backend })
            }
        };
    }

    if !managed_affinity_available {
        return match requested {
            CpuPlacement::Auto => Ok(ResolvedCpuExecution::Compatibility),
            CpuPlacement::NumaNode(_) | CpuPlacement::AllAllowed => {
                Err(CpuPlacementError::ManagedAffinityUnavailable { requested, backend })
            }
        };
    }

    let placement = match requested {
        CpuPlacement::Auto | CpuPlacement::AllAllowed => ResolvedCpuPlacement::AllAllowed {
            cpus: topology.allowed_cpus().clone(),
        },
        CpuPlacement::NumaNode(node) => {
            if !topology.has_numa_nodes() {
                return Err(CpuPlacementError::NumaDiscoveryUnavailable { requested, backend });
            }
            let cpus = topology
                .node(node)
                .ok_or(CpuPlacementError::UnknownNumaNode {
                    requested,
                    backend,
                    node,
                })?;
            ResolvedCpuPlacement::NumaNode {
                id: node,
                cpus: cpus.cpus().clone(),
            }
        }
    };
    Ok(ResolvedCpuExecution::Managed(placement))
}

#[cfg(test)]
mod tests;
