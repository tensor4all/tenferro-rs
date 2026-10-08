use thiserror::Error;

use crate::{CpuContextError, CpuSet, CpuTopology, CpuTopologyError, NumaNodeId};

/// Typed failure raised while constructing a CPU execution engine.
///
/// The tensor-backed compatibility path and the managed engine path expose
/// different concrete construction errors. This wrapper keeps both sources
/// typed while allowing [`CpuPlacementError`] to present one public error
/// shape.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuContextError, CpuEngineConstructionError};
/// use std::error::Error;
///
/// let error = CpuEngineConstructionError::Context(CpuContextError::InvalidThreadCount);
/// assert!(error.source().is_some());
/// ```
#[derive(Debug, Error)]
pub enum CpuEngineConstructionError {
    /// A managed CPU context or pinned worker engine could not be built.
    #[error("managed CPU engine construction failed: {0}")]
    Context(#[source] CpuContextError),
    /// The tensor-backed compatibility engine could not be built.
    #[error("tensor CPU engine construction failed: {0}")]
    Tensor(#[source] tenferro_tensor::Error),
}

/// Requested CPU placement.
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
    /// Resolve the default placement for this platform.
    #[default]
    Auto,
    /// Restrict tenferro-owned execution to one usable OS NUMA node.
    NumaNode(NumaNodeId),
    /// Use the complete CPU set permitted to the process.
    AllAllowed,
}

/// Concrete CPU placement resolved for a tenferro-owned engine.
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
    /// A concrete OS NUMA-node placement.
    NumaNode {
        /// The sparse OS NUMA node ID.
        id: NumaNodeId,
        /// The logical CPUs resolved for the node.
        cpus: CpuSet,
    },
    /// A resolved complete process-affinity CPU set.
    AllAllowed {
        /// Logical CPUs resolved as process-permitted.
        cpus: CpuSet,
    },
}

impl ResolvedCpuPlacement {
    /// Return the concrete logical CPU set resolved for this placement.
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

/// Failure to resolve a CPU placement on this process topology.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuPlacement, CpuPlacementError, NumaNodeId};
///
/// let error = CpuPlacementError::NumaDiscoveryUnavailable {
///     requested: CpuPlacement::NumaNode(NumaNodeId::new(1)),
/// };
/// assert!(error.to_string().contains("NUMA"));
/// ```
#[derive(Debug, Error)]
pub enum CpuPlacementError {
    /// Process-visible topology discovery failed before placement resolution.
    #[error("cannot resolve {requested:?}: topology discovery failed: {source}")]
    TopologyDiscovery {
        /// The placement requested by the caller.
        requested: CpuPlacement,
        /// The preserved topology failure category.
        #[source]
        source: CpuTopologyError,
    },
    /// The current platform cannot construct verified pinned worker pools.
    #[error("cannot resolve {requested:?}: managed worker affinity is unavailable")]
    ManagedAffinityUnavailable {
        /// The explicit placement requested by the caller.
        requested: CpuPlacement,
    },
    /// NUMA-node placement was requested but OS NUMA discovery was unavailable.
    #[error("cannot resolve {requested:?}: NUMA discovery is unavailable")]
    NumaDiscoveryUnavailable {
        /// The placement requested by the caller.
        requested: CpuPlacement,
    },
    /// The requested OS NUMA node has no usable CPUs in this process.
    #[error("cannot resolve {requested:?}: NUMA node {node} is unavailable")]
    UnknownNumaNode {
        /// The placement requested by the caller.
        requested: CpuPlacement,
        /// The unknown or process-unavailable OS node ID.
        node: NumaNodeId,
    },
    /// A pinned engine could not be built for an otherwise valid placement.
    #[error("cannot resolve {requested:?}: engine construction failed: {source}")]
    EngineConstruction {
        /// The placement requested by the caller.
        requested: CpuPlacement,
        /// Typed worker-pool construction or affinity failure.
        #[source]
        source: CpuEngineConstructionError,
    },
    /// The placement state reached an impossible internal compatibility mode.
    #[error("cannot resolve {requested:?}: {message}")]
    InternalState {
        /// The placement requested by the caller.
        requested: CpuPlacement,
        /// Stable internal-state diagnostic.
        message: &'static str,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ResolvedCpuExecution {
    Compatibility,
    Managed(ResolvedCpuPlacement),
}

pub(crate) fn resolve_placement(
    requested: CpuPlacement,
    topology: &CpuTopology,
) -> Result<ResolvedCpuExecution, CpuPlacementError> {
    resolve_placement_with_affinity(
        requested,
        topology,
        cfg!(any(target_os = "linux", target_os = "android")),
    )
}

pub(crate) fn resolve_placement_with_affinity(
    requested: CpuPlacement,
    topology: &CpuTopology,
    managed_affinity_available: bool,
) -> Result<ResolvedCpuExecution, CpuPlacementError> {
    if !managed_affinity_available {
        return match requested {
            CpuPlacement::Auto => Ok(ResolvedCpuExecution::Compatibility),
            CpuPlacement::NumaNode(_) | CpuPlacement::AllAllowed => {
                Err(CpuPlacementError::ManagedAffinityUnavailable { requested })
            }
        };
    }

    let placement = match requested {
        CpuPlacement::Auto | CpuPlacement::AllAllowed => ResolvedCpuPlacement::AllAllowed {
            cpus: topology.allowed_cpus().clone(),
        },
        CpuPlacement::NumaNode(node) => {
            if !topology.has_numa_nodes() {
                return Err(CpuPlacementError::NumaDiscoveryUnavailable { requested });
            }
            let cpus = topology
                .node(node)
                .ok_or(CpuPlacementError::UnknownNumaNode { requested, node })?;
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
