use std::collections::BTreeMap;

use smallvec::SmallVec;
use tenferro_tensor::CpuDomainId;

const INLINE_DOMAIN_CAPACITY: usize = 8;

/// Policy used to select a CPU execution domain from input affinity metadata.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuAffinityPolicy;
///
/// let policy = CpuAffinityPolicy::DominantInputBytes;
/// assert_ne!(policy, CpuAffinityPolicy::RequireSingleDomain);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuAffinityPolicy {
    /// Select the domain with the largest total of positive logical input bytes.
    DominantInputBytes,
    /// Accept zero or one known input domain and reject mixed known domains.
    RequireSingleDomain,
}

/// CPU affinity metadata for one logical operation input.
///
/// The resolver reads this metadata only. It never changes, copies, or rehomes
/// tensor payloads.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuAffinityInput;
/// use tenferro_tensor::CpuDomainId;
///
/// let input = CpuAffinityInput {
///     domain: Some(CpuDomainId::new(3)),
///     logical_bytes: 64,
/// };
/// assert_eq!(input.logical_bytes, 64);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuAffinityInput {
    /// Known CPU execution domain, or `None` when affinity is unknown.
    pub domain: Option<CpuDomainId>,
    /// Logical input size used by [`CpuAffinityPolicy::DominantInputBytes`].
    pub logical_bytes: usize,
}

/// Why the CPU affinity resolver selected a domain.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuAffinitySelectionReason;
///
/// let reason = CpuAffinitySelectionReason::DefaultDomain;
/// assert_eq!(reason, CpuAffinitySelectionReason::DefaultDomain);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuAffinitySelectionReason {
    /// An operation-local explicit domain override took precedence.
    ExplicitOverride,
    /// Positive logical bytes made this domain dominant.
    DominantInputBytes,
    /// Strict policy observed exactly one known input domain.
    SingleInputDomain,
    /// No relevant input affinity was available.
    DefaultDomain,
}

/// Deterministic CPU affinity selection returned by the pure resolver.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuAffinitySelection, CpuAffinitySelectionReason};
/// use tenferro_tensor::CpuDomainId;
///
/// let selection = CpuAffinitySelection {
///     domain: CpuDomainId::new(2),
///     reason: CpuAffinitySelectionReason::DominantInputBytes,
/// };
/// assert_eq!(selection.domain.as_u64(), 2);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuAffinitySelection {
    /// Selected CPU execution domain.
    pub domain: CpuDomainId,
    /// Deterministic reason for the selection.
    pub reason: CpuAffinitySelectionReason,
}

/// Failure to resolve CPU affinity from input metadata.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuAffinityResolutionError;
/// use tenferro_tensor::CpuDomainId;
///
/// let error = CpuAffinityResolutionError::LogicalByteCountOverflow {
///     domain: CpuDomainId::new(4),
/// };
/// assert!(error.to_string().contains("4"));
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CpuAffinityResolutionError {
    /// Adding logical byte counts overflowed `usize` for one domain.
    #[error("logical input-byte total overflowed for CPU domain {domain:?}")]
    LogicalByteCountOverflow {
        /// Smallest CPU domain whose logical byte total overflowed.
        domain: CpuDomainId,
    },
    /// Strict policy observed at least two different known domains.
    #[error("CPU affinity policy requires one input domain, found {first:?} and {second:?}")]
    MultipleKnownDomains {
        /// Smallest known input domain.
        first: CpuDomainId,
        /// Second-smallest known input domain.
        second: CpuDomainId,
    },
}

/// Resolve a CPU execution domain from input affinity metadata.
///
/// Unknown affinities and zero-byte inputs do not contribute to dominant-byte
/// scoring. When no input contributes, `default_domain` is selected. Equal
/// positive totals are resolved in favor of the smallest [`CpuDomainId`]. The
/// input slice is only read; the resolver never retags or rehomes an input.
///
/// Use [`resolve_cpu_affinity_with_override`] when an operation-local explicit
/// placement has already been selected.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{resolve_cpu_affinity, CpuAffinityInput, CpuAffinityPolicy};
/// use tenferro_tensor::CpuDomainId;
///
/// let inputs = [
///     CpuAffinityInput { domain: Some(CpuDomainId::new(8)), logical_bytes: 6 },
///     CpuAffinityInput { domain: Some(CpuDomainId::new(3)), logical_bytes: 2 },
/// ];
/// let selected = resolve_cpu_affinity(
///     CpuAffinityPolicy::DominantInputBytes,
///     &inputs,
///     CpuDomainId::new(1),
/// )?;
/// assert_eq!(selected.domain, CpuDomainId::new(8));
/// # Ok::<(), tenferro_cpu::CpuAffinityResolutionError>(())
/// ```
///
/// # Errors
///
/// Returns [`CpuAffinityResolutionError::LogicalByteCountOverflow`] when one
/// domain's logical byte total cannot be represented by `usize`, or
/// [`CpuAffinityResolutionError::MultipleKnownDomains`] when strict policy sees
/// more than one known input domain.
pub fn resolve_cpu_affinity(
    policy: CpuAffinityPolicy,
    inputs: &[CpuAffinityInput],
    default_domain: CpuDomainId,
) -> Result<CpuAffinitySelection, CpuAffinityResolutionError> {
    resolve_cpu_affinity_with_override(policy, inputs, default_domain, None)
}

/// Resolve CPU affinity with an optional operation-local explicit override.
///
/// Explicit placement takes precedence before input-byte accounting or strict
/// mixed-domain validation. Passing `None` applies the same policy resolution
/// as [`resolve_cpu_affinity`].
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{
///     resolve_cpu_affinity_with_override, CpuAffinityInput, CpuAffinityPolicy,
///     CpuAffinitySelectionReason,
/// };
/// use tenferro_tensor::CpuDomainId;
///
/// let mixed = [
///     CpuAffinityInput { domain: Some(CpuDomainId::new(1)), logical_bytes: 1 },
///     CpuAffinityInput { domain: Some(CpuDomainId::new(2)), logical_bytes: 1 },
/// ];
/// let selected = resolve_cpu_affinity_with_override(
///     CpuAffinityPolicy::RequireSingleDomain,
///     &mixed,
///     CpuDomainId::new(1),
///     Some(CpuDomainId::new(9)),
/// )?;
/// assert_eq!(selected.domain, CpuDomainId::new(9));
/// assert_eq!(selected.reason, CpuAffinitySelectionReason::ExplicitOverride);
/// # Ok::<(), tenferro_cpu::CpuAffinityResolutionError>(())
/// ```
///
/// # Errors
///
/// When `explicit_domain` is `None`, returns
/// [`CpuAffinityResolutionError::LogicalByteCountOverflow`] for an unrepresentable
/// domain byte total or [`CpuAffinityResolutionError::MultipleKnownDomains`]
/// when strict policy sees more than one known input domain. A present explicit
/// override bypasses both policy errors.
pub fn resolve_cpu_affinity_with_override(
    policy: CpuAffinityPolicy,
    inputs: &[CpuAffinityInput],
    default_domain: CpuDomainId,
    explicit_domain: Option<CpuDomainId>,
) -> Result<CpuAffinitySelection, CpuAffinityResolutionError> {
    if let Some(domain) = explicit_domain {
        return Ok(CpuAffinitySelection {
            domain,
            reason: CpuAffinitySelectionReason::ExplicitOverride,
        });
    }
    match policy {
        CpuAffinityPolicy::DominantInputBytes => resolve_dominant(inputs, default_domain),
        CpuAffinityPolicy::RequireSingleDomain => resolve_single(inputs, default_domain),
    }
}

fn resolve_dominant(
    inputs: &[CpuAffinityInput],
    default_domain: CpuDomainId,
) -> Result<CpuAffinitySelection, CpuAffinityResolutionError> {
    let mut totals = DomainTotals::default();
    for input in inputs {
        let Some(domain) = input.domain else {
            continue;
        };
        if input.logical_bytes == 0 {
            continue;
        }
        totals.add(domain, input.logical_bytes);
    }

    if let Some(domain) = totals.smallest_overflowing_domain() {
        return Err(CpuAffinityResolutionError::LogicalByteCountOverflow { domain });
    }

    match totals.dominant_domain() {
        Some(domain) => Ok(CpuAffinitySelection {
            domain,
            reason: CpuAffinitySelectionReason::DominantInputBytes,
        }),
        None => Ok(default_selection(default_domain)),
    }
}

fn resolve_single(
    inputs: &[CpuAffinityInput],
    default_domain: CpuDomainId,
) -> Result<CpuAffinitySelection, CpuAffinityResolutionError> {
    let mut first = None;
    let mut second = None;
    for domain in inputs.iter().filter_map(|input| input.domain) {
        observe_smallest_two_distinct(domain, &mut first, &mut second);
    }

    if let (Some(first), Some(second)) = (first, second) {
        return Err(CpuAffinityResolutionError::MultipleKnownDomains { first, second });
    }

    Ok(match first {
        Some(domain) => CpuAffinitySelection {
            domain,
            reason: CpuAffinitySelectionReason::SingleInputDomain,
        },
        None => default_selection(default_domain),
    })
}

fn observe_smallest_two_distinct(
    domain: CpuDomainId,
    first: &mut Option<CpuDomainId>,
    second: &mut Option<CpuDomainId>,
) {
    if *first == Some(domain) || *second == Some(domain) {
        return;
    }
    match *first {
        None => *first = Some(domain),
        Some(current_first) if domain < current_first => {
            *second = *first;
            *first = Some(domain);
        }
        Some(_) if second.is_none_or(|current_second| domain < current_second) => {
            *second = Some(domain);
        }
        Some(_) => {}
    }
}

fn default_selection(domain: CpuDomainId) -> CpuAffinitySelection {
    CpuAffinitySelection {
        domain,
        reason: CpuAffinitySelectionReason::DefaultDomain,
    }
}

#[derive(Clone, Copy, Debug)]
struct DomainTotal {
    domain: CpuDomainId,
    logical_bytes: Option<usize>,
}

impl DomainTotal {
    fn new(domain: CpuDomainId, logical_bytes: usize) -> Self {
        Self {
            domain,
            logical_bytes: Some(logical_bytes),
        }
    }

    fn add(&mut self, logical_bytes: usize) {
        self.logical_bytes = self
            .logical_bytes
            .and_then(|total| total.checked_add(logical_bytes));
    }
}

enum DomainTotals {
    Inline(SmallVec<[DomainTotal; INLINE_DOMAIN_CAPACITY]>),
    Heap(BTreeMap<CpuDomainId, Option<usize>>),
}

impl Default for DomainTotals {
    fn default() -> Self {
        Self::Inline(SmallVec::new())
    }
}

impl DomainTotals {
    fn add(&mut self, domain: CpuDomainId, logical_bytes: usize) {
        let promoted = match self {
            Self::Inline(entries) => {
                // INVARIANT: the linear lookup is bounded by the inline capacity;
                // larger distinct-domain sets are promoted to `BTreeMap` below.
                if let Some(entry) = entries.iter_mut().find(|entry| entry.domain == domain) {
                    entry.add(logical_bytes);
                    return;
                }
                if entries.len() < INLINE_DOMAIN_CAPACITY {
                    entries.push(DomainTotal::new(domain, logical_bytes));
                    return;
                }
                let mut heap = BTreeMap::new();
                for entry in entries.drain(..) {
                    heap.insert(entry.domain, entry.logical_bytes);
                }
                heap.insert(domain, Some(logical_bytes));
                Some(heap)
            }
            Self::Heap(entries) => {
                let total = entries.entry(domain).or_insert(Some(0));
                *total = total.and_then(|current| current.checked_add(logical_bytes));
                None
            }
        };
        if let Some(heap) = promoted {
            *self = Self::Heap(heap);
        }
    }

    fn smallest_overflowing_domain(&self) -> Option<CpuDomainId> {
        match self {
            Self::Inline(entries) => entries
                .iter()
                .filter(|entry| entry.logical_bytes.is_none())
                .map(|entry| entry.domain)
                .min(),
            Self::Heap(entries) => entries
                .iter()
                .find_map(|(domain, total)| total.is_none().then_some(*domain)),
        }
    }

    fn dominant_domain(&self) -> Option<CpuDomainId> {
        let mut best = None;
        match self {
            Self::Inline(entries) => {
                for entry in entries {
                    if let Some(logical_bytes) = entry.logical_bytes {
                        consider_dominant(&mut best, entry.domain, logical_bytes);
                    }
                }
            }
            Self::Heap(entries) => {
                for (&domain, &logical_bytes) in entries {
                    if let Some(logical_bytes) = logical_bytes {
                        consider_dominant(&mut best, domain, logical_bytes);
                    }
                }
            }
        }
        best.map(|(domain, _)| domain)
    }
}

fn consider_dominant(
    best: &mut Option<(CpuDomainId, usize)>,
    domain: CpuDomainId,
    logical_bytes: usize,
) {
    let replace = match *best {
        None => true,
        Some((best_domain, best_bytes)) => {
            logical_bytes > best_bytes || (logical_bytes == best_bytes && domain < best_domain)
        }
    };
    if replace {
        *best = Some((domain, logical_bytes));
    }
}

#[cfg(test)]
mod tests;
