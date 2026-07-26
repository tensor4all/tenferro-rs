use std::error::Error;
use std::fmt;
use std::sync::Arc;

use super::identity::validate_identifier;
use super::{IdentityError, IdentityKind};

/// Validated runtime cache-owner identifier.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::runtime::CacheOwnerId;
///
/// assert_eq!(CacheOwnerId::new("tenferro.cache.owner")?.as_str(), "tenferro.cache.owner");
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CacheOwnerId(Arc<str>);

impl CacheOwnerId {
    /// Validate a namespaced ASCII cache-owner identifier.
    ///
    /// # Errors
    ///
    /// Returns [`IdentityError`] when `value` does not match the runtime
    /// identifier grammar.
    pub fn new(value: impl Into<Arc<str>>) -> Result<Self, IdentityError> {
        validate_identifier(value.into(), IdentityKind::CacheOwner).map(Self)
    }

    /// Borrow the validated identifier text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub(super) fn from_canonical_owner_id(value: Arc<str>) -> Self {
        Self(value)
    }
}

/// Aggregate cache statistics reported by one runtime cache owner.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::runtime::CacheStats;
///
/// assert_eq!(CacheStats::default().entries, 0);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheStats {
    /// Number of retained cache entries.
    pub entries: usize,
    /// Logical retained bytes.
    pub retained_bytes: usize,
    /// Cache hits.
    pub hits: u64,
    /// Cache misses.
    pub misses: u64,
    /// Cache evictions.
    pub evictions: u64,
    /// Explicit clears.
    pub clears: u64,
}

/// Runtime-owned cache participant.
pub trait RuntimeCacheOwner: fmt::Debug + Send + Sync + 'static {
    /// Return this owner's current cache statistics.
    ///
    /// # Errors
    ///
    /// Returns [`CacheOwnerError`] when the owner cannot report stats.
    fn cache_stats(&self) -> Result<CacheStats, CacheOwnerError>;

    /// Clear this owner's retained caches.
    ///
    /// # Errors
    ///
    /// Returns [`CacheOwnerError`] when the owner cannot clear its caches.
    fn clear_caches(&self) -> Result<(), CacheOwnerError>;
}

/// Cloneable typed cache-owner failure source.
#[derive(Clone)]
pub struct CacheOwnerError {
    source: Arc<dyn Error + Send + Sync>,
}

impl CacheOwnerError {
    /// Wrap a typed cache-owner failure source.
    pub fn new(source: Arc<dyn Error + Send + Sync>) -> Self {
        Self { source }
    }

    /// Return the original shared source.
    pub fn source_arc(&self) -> &Arc<dyn Error + Send + Sync> {
        &self.source
    }
}

impl fmt::Debug for CacheOwnerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CacheOwnerError")
            .field("source", &self.source.to_string())
            .finish()
    }
}

impl fmt::Display for CacheOwnerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.source)
    }
}

impl Error for CacheOwnerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.source.as_ref())
    }
}

/// Failure reported by one named runtime cache owner.
#[derive(Clone, Debug)]
pub struct CacheOwnerFailure {
    /// Cache owner that failed.
    pub owner: CacheOwnerId,
    /// Typed owner source.
    pub source: CacheOwnerError,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum FrozenCacheOwnerKind {
    Engine,
    Extension,
}

#[derive(Clone)]
pub(super) struct FrozenCacheOwner {
    pub(super) id: CacheOwnerId,
    pub(super) kind: FrozenCacheOwnerKind,
    pub(super) owner: Arc<dyn RuntimeCacheOwner>,
}

impl fmt::Debug for FrozenCacheOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrozenCacheOwner")
            .field("id", &self.id)
            .field("kind", &self.kind)
            .field("owner_strong_count", &Arc::strong_count(&self.owner))
            .finish()
    }
}

/// Runtime state failure shared by preparation and cache management.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeStateError {
    /// A synchronization primitive was poisoned by a panic in another thread.
    #[error("{lock} poisoned")]
    Poisoned {
        /// Static lock name.
        lock: &'static str,
    },
}

/// Aggregated runtime cache-management failure.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeCacheError {
    /// Runtime and/or registered cache owners failed.
    #[error("runtime cache operation failed")]
    Aggregate {
        /// Runtime state failure, if one occurred.
        runtime: Option<RuntimeStateError>,
        /// Owner failures in deterministic owner order.
        owners: Box<[CacheOwnerFailure]>,
    },
}
