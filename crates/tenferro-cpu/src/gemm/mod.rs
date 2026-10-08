//! Bounded, clearable numerical-plan caches for the CPU contraction route.
//!
//! Numerical plans, lane selection and scheduling belong to cpueinsum; tenferro
//! owns only where the prepared plans live, how many are retained, and the
//! accounting reported to the runtime cache owner.

use std::fmt;

use crate::CacheStats;

/// Default number of retained numerical plans per contraction form.
pub(crate) const DEFAULT_GEMM_ANALYSIS_CACHE_CAPACITY: usize = 1024;

/// One session's retained CPU contraction plans, split by contraction form.
///
/// The cache stores immutable prepared plans only. It never retains operands,
/// scratch buffers, or an execution lease; those belong to the execution
/// workspace.
#[doc(hidden)]
pub struct GemmAnalysisCache {
    max_slots: usize,
    clears: u64,
    pub(crate) binary: crate::contraction::BinaryCache,
    pub(crate) grouped: crate::contraction::grouped::GroupCache,
}

impl fmt::Debug for GemmAnalysisCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GemmAnalysisCache")
            .field("max_slots", &self.max_slots)
            .field("binary", &self.binary)
            .field("grouped", &self.grouped)
            .finish_non_exhaustive()
    }
}

impl GemmAnalysisCache {
    pub(crate) fn with_capacity(max_slots: usize) -> Self {
        Self {
            max_slots,
            clears: 0,
            binary: crate::contraction::BinaryCache::default(),
            grouped: crate::contraction::grouped::GroupCache::default(),
        }
    }

    /// The configured per-form plan capacity.
    #[doc(hidden)]
    pub fn capacity(&self) -> usize {
        self.max_slots
    }

    /// Reconfigure the per-form plan capacity, evicting plans when it shrinks.
    #[doc(hidden)]
    pub fn set_capacity(&mut self, max_slots: usize) {
        self.binary.set_capacity(max_slots);
        self.grouped.set_capacity(max_slots);
        self.max_slots = max_slots;
    }
}

impl Default for GemmAnalysisCache {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_GEMM_ANALYSIS_CACHE_CAPACITY)
    }
}

impl tenferro_tensor::RuntimeCacheControl for GemmAnalysisCache {
    fn clear(&mut self) {
        self.binary.clear();
        self.grouped.clear();
        self.clears = self.clears.saturating_add(1);
    }

    fn stats(&self) -> CacheStats {
        let binary = self.binary.stats();
        let grouped = self.grouped.stats();
        CacheStats {
            entries: binary.entries.saturating_add(grouped.entries),
            retained_bytes: binary.retained_bytes.saturating_add(grouped.retained_bytes),
            hits: binary.hits.saturating_add(grouped.hits),
            misses: binary.misses.saturating_add(grouped.misses),
            evictions: binary.evictions.saturating_add(grouped.evictions),
            clears: self.clears,
        }
    }
}
