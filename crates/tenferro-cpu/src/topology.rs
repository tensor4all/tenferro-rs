use std::fmt;

use thiserror::Error;

/// An operating-system logical CPU identifier.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::CpuId;
///
/// assert_eq!(CpuId::new(3).as_usize(), 3);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CpuId(usize);

impl CpuId {
    /// Create an identifier from the operating-system logical CPU number.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuId;
    ///
    /// assert_eq!(CpuId::new(5).as_usize(), 5);
    /// ```
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Return the operating-system logical CPU number.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::CpuId;
    ///
    /// assert_eq!(CpuId::new(7).as_usize(), 7);
    /// ```
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for CpuId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// An operating-system NUMA node identifier.
///
/// Node IDs are preserved as reported by the OS and may be sparse.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::NumaNodeId;
///
/// assert_eq!(NumaNodeId::new(4).as_usize(), 4);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NumaNodeId(usize);

impl NumaNodeId {
    /// Create an identifier from the operating-system NUMA node number.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::NumaNodeId;
    ///
    /// assert_eq!(NumaNodeId::new(2).as_usize(), 2);
    /// ```
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Return the operating-system NUMA node number.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::NumaNodeId;
    ///
    /// assert_eq!(NumaNodeId::new(9).as_usize(), 9);
    /// ```
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for NumaNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Failure to construct a non-empty CPU set.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuId, CpuSet, CpuSetError};
///
/// assert_eq!(CpuSet::new(Vec::<CpuId>::new()), Err(CpuSetError::Empty));
/// ```
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum CpuSetError {
    /// A CPU execution domain cannot be empty.
    #[error("CPU set is empty")]
    Empty,
}

/// A sorted, deduplicated, non-empty set of logical CPUs.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuId, CpuSet};
///
/// let cpus = CpuSet::new([CpuId::new(3), CpuId::new(1), CpuId::new(3)])?;
/// assert_eq!(cpus.as_usize_vec(), vec![1, 3]);
/// # Ok::<(), tenferro_cpu::CpuSetError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CpuSet {
    cpus: Vec<CpuId>,
}

impl CpuSet {
    /// Construct a sorted and deduplicated CPU set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet};
    ///
    /// let cpus = CpuSet::new([CpuId::new(2), CpuId::new(0)])?;
    /// assert_eq!(cpus.as_usize_vec(), vec![0, 2]);
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn new(cpus: impl IntoIterator<Item = CpuId>) -> Result<Self, CpuSetError> {
        let mut cpus: Vec<_> = cpus.into_iter().collect();
        cpus.sort_unstable();
        cpus.dedup();
        if cpus.is_empty() {
            return Err(CpuSetError::Empty);
        }
        Ok(Self { cpus })
    }

    /// Return the number of logical CPUs in this set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet};
    ///
    /// assert_eq!(CpuSet::new([CpuId::new(0), CpuId::new(1)])?.len(), 2);
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn len(&self) -> usize {
        self.cpus.len()
    }

    /// Report whether this set is empty.
    ///
    /// A successfully constructed `CpuSet` is never empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet};
    ///
    /// assert!(!CpuSet::new([CpuId::new(0)])?.is_empty());
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        self.cpus.is_empty()
    }

    /// Return the sorted logical CPU identifiers.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet};
    ///
    /// let cpus = CpuSet::new([CpuId::new(4), CpuId::new(2)])?;
    /// assert_eq!(cpus.as_slice(), &[CpuId::new(2), CpuId::new(4)]);
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn as_slice(&self) -> &[CpuId] {
        &self.cpus
    }

    /// Copy the sorted logical CPU numbers into a vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet};
    ///
    /// let cpus = CpuSet::new([CpuId::new(6), CpuId::new(5)])?;
    /// assert_eq!(cpus.as_usize_vec(), vec![5, 6]);
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn as_usize_vec(&self) -> Vec<usize> {
        self.cpus.iter().map(|cpu| cpu.as_usize()).collect()
    }

    /// Report whether the logical CPU belongs to this set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet};
    ///
    /// let cpus = CpuSet::new([CpuId::new(1), CpuId::new(3)])?;
    /// assert!(cpus.contains(CpuId::new(3)));
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn contains(&self, cpu: CpuId) -> bool {
        self.cpus.binary_search(&cpu).is_ok()
    }

    fn intersection(&self, other: &Self) -> Option<Self> {
        let mut intersection = Vec::with_capacity(self.len().min(other.len()));
        let (mut left, mut right) = (0, 0);
        while left < self.len() && right < other.len() {
            match self.cpus[left].cmp(&other.cpus[right]) {
                std::cmp::Ordering::Less => left += 1,
                std::cmp::Ordering::Greater => right += 1,
                std::cmp::Ordering::Equal => {
                    intersection.push(self.cpus[left]);
                    left += 1;
                    right += 1;
                }
            }
        }
        (!intersection.is_empty()).then_some(Self { cpus: intersection })
    }
}

/// One usable OS NUMA node and its process-allowed logical CPUs.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuId, CpuNode, CpuSet, NumaNodeId};
///
/// let node = CpuNode::new(NumaNodeId::new(2), CpuSet::new([CpuId::new(8)])?);
/// assert_eq!(node.id(), NumaNodeId::new(2));
/// # Ok::<(), tenferro_cpu::CpuSetError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuNode {
    id: NumaNodeId,
    cpus: CpuSet,
}

impl CpuNode {
    /// Construct a usable NUMA node description.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuNode, CpuSet, NumaNodeId};
    ///
    /// let node = CpuNode::new(NumaNodeId::new(7), CpuSet::new([CpuId::new(12)])?);
    /// assert_eq!(node.id(), NumaNodeId::new(7));
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn new(id: NumaNodeId, cpus: CpuSet) -> Self {
        Self { id, cpus }
    }

    /// Return the sparse operating-system NUMA node ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuNode, CpuSet, NumaNodeId};
    ///
    /// let node = CpuNode::new(NumaNodeId::new(3), CpuSet::new([CpuId::new(0)])?);
    /// assert_eq!(node.id().as_usize(), 3);
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn id(&self) -> NumaNodeId {
        self.id
    }

    /// Return this node's process-allowed logical CPUs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuNode, CpuSet, NumaNodeId};
    ///
    /// let node = CpuNode::new(NumaNodeId::new(0), CpuSet::new([CpuId::new(4)])?);
    /// assert!(node.cpus().contains(CpuId::new(4)));
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn cpus(&self) -> &CpuSet {
        &self.cpus
    }
}

/// Failure to canonicalize discovered NUMA topology.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuTopologyError, NumaNodeId};
///
/// let error = CpuTopologyError::DuplicateNode { node: NumaNodeId::new(2) };
/// assert!(error.to_string().contains("2"));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CpuTopologyError {
    /// The process affinity mask contained no logical CPU.
    #[error("the process-allowed CPU set is empty")]
    EmptyAllowedCpuSet,
    /// Discovery reported the same OS NUMA node more than once.
    #[error("NUMA node {node} was discovered more than once")]
    DuplicateNode {
        /// The duplicated OS NUMA node ID.
        node: NumaNodeId,
    },
    /// Two usable NUMA nodes contained at least one common logical CPU.
    #[error("NUMA nodes {first} and {second} overlap on CPUs {cpus:?}")]
    OverlappingNodes {
        /// The first overlapping OS node ID.
        first: NumaNodeId,
        /// The second overlapping OS node ID.
        second: NumaNodeId,
        /// Logical CPUs present in both usable node sets.
        cpus: CpuSet,
    },
}

/// Process-visible CPU topology used for execution placement.
///
/// Each usable NUMA node contains only CPUs also present in the process affinity
/// mask. Empty nodes are omitted and OS node IDs are not renumbered.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::{CpuId, CpuSet, CpuTopology};
///
/// let topology = CpuTopology::all_allowed(CpuSet::new([CpuId::new(0)])?);
/// assert_eq!(topology.allowed_cpus().len(), 1);
/// assert!(!topology.has_numa_nodes());
/// # Ok::<(), tenferro_cpu::CpuSetError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuTopology {
    allowed_cpus: CpuSet,
    nodes: Vec<CpuNode>,
}

impl CpuTopology {
    /// Construct a topology with only the all-allowed execution domain.
    ///
    /// This is the portable fallback when NUMA discovery is unavailable.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet, CpuTopology};
    ///
    /// let topology = CpuTopology::all_allowed(CpuSet::new([CpuId::new(2)])?);
    /// assert!(topology.nodes().is_empty());
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn all_allowed(allowed_cpus: CpuSet) -> Self {
        Self {
            allowed_cpus,
            nodes: Vec::new(),
        }
    }

    pub(crate) fn from_discovered(
        allowed_cpus: CpuSet,
        nodes: impl IntoIterator<Item = (NumaNodeId, CpuSet)>,
    ) -> Result<Self, CpuTopologyError> {
        if allowed_cpus.is_empty() {
            return Err(CpuTopologyError::EmptyAllowedCpuSet);
        }
        let mut usable = Vec::new();
        for (id, discovered_cpus) in nodes {
            if usable.iter().any(|node: &CpuNode| node.id == id) {
                return Err(CpuTopologyError::DuplicateNode { node: id });
            }
            if let Some(cpus) = discovered_cpus.intersection(&allowed_cpus) {
                usable.push(CpuNode::new(id, cpus));
            }
        }
        usable.sort_unstable_by_key(CpuNode::id);
        for (index, left) in usable.iter().enumerate() {
            for right in usable.iter().skip(index + 1) {
                if let Some(cpus) = left.cpus.intersection(&right.cpus) {
                    return Err(CpuTopologyError::OverlappingNodes {
                        first: left.id,
                        second: right.id,
                        cpus,
                    });
                }
            }
        }
        Ok(Self {
            allowed_cpus,
            nodes: usable,
        })
    }

    /// Return the complete process affinity CPU set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet, CpuTopology};
    ///
    /// let topology = CpuTopology::all_allowed(CpuSet::new([CpuId::new(1)])?);
    /// assert!(topology.allowed_cpus().contains(CpuId::new(1)));
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn allowed_cpus(&self) -> &CpuSet {
        &self.allowed_cpus
    }

    /// Return usable NUMA nodes ordered by their sparse OS node IDs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet, CpuTopology};
    ///
    /// let topology = CpuTopology::all_allowed(CpuSet::new([CpuId::new(0)])?);
    /// assert!(topology.nodes().is_empty());
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn nodes(&self) -> &[CpuNode] {
        &self.nodes
    }

    /// Copy the usable sparse operating-system NUMA node IDs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet, CpuTopology};
    ///
    /// let topology = CpuTopology::all_allowed(CpuSet::new([CpuId::new(0)])?);
    /// assert!(topology.node_ids().is_empty());
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn node_ids(&self) -> Vec<NumaNodeId> {
        self.nodes.iter().map(CpuNode::id).collect()
    }

    /// Look up a usable node by its original OS node ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet, CpuTopology, NumaNodeId};
    ///
    /// let topology = CpuTopology::all_allowed(CpuSet::new([CpuId::new(0)])?);
    /// assert!(topology.node(NumaNodeId::new(0)).is_none());
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn node(&self, id: NumaNodeId) -> Option<&CpuNode> {
        self.nodes
            .binary_search_by_key(&id, CpuNode::id)
            .ok()
            .map(|index| &self.nodes[index])
    }

    /// Report whether NUMA discovery produced any usable node domains.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_cpu::{CpuId, CpuSet, CpuTopology};
    ///
    /// let topology = CpuTopology::all_allowed(CpuSet::new([CpuId::new(0)])?);
    /// assert!(!topology.has_numa_nodes());
    /// # Ok::<(), tenferro_cpu::CpuSetError>(())
    /// ```
    pub fn has_numa_nodes(&self) -> bool {
        !self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests;
