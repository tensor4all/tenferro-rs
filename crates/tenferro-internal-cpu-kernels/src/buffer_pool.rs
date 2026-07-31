//! Typed host buffer pooling for reusable tensor allocations.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_internal_cpu_kernels::buffer_pool::{BufferPool, PoolScalar};
//!
//! let mut pool = BufferPool::new();
//! let mut buf = pool.acquire_zeroed::<f64>(4);
//! buf.fill(1.0);
//! <f64 as PoolScalar>::pool_release(&mut pool, buf);
//! assert_eq!(pool.len(), 1);
//! ```

use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fmt;
use std::mem::{size_of, ManuallyDrop, MaybeUninit};
use std::sync::OnceLock;

use num_complex::{Complex32, Complex64};

use crate::CacheStats;

/// Non-Copy proof of a tracked pool checkout.
#[derive(Debug)]
pub(crate) enum UninitCheckoutToken {
    /// The allocation was freshly allocated.
    Fresh { actual_capacity: usize },
    /// Retained storage was removed, with its actual capacity.
    Reused { actual_capacity: usize },
}

/// Environment variable overriding the CPU buffer-pool retention cap in bytes.
///
/// The value is parsed as an unsigned integer. Invalid values fall back to
/// [`DEFAULT_MAX_RETAINED_CAPACITY_BYTES`].
pub const BUFFER_POOL_MAX_RETAINED_BYTES_ENV: &str = "TENFERRO_BUFFER_POOL_MAX_RETAINED_BYTES";

/// Default retained CPU buffer capacity per backend.
///
/// The cap keeps long-running workloads from accumulating obsolete buffer
/// sizes as tensor shapes grow while still preserving reuse for hot working
/// sets.
pub const DEFAULT_MAX_RETAINED_CAPACITY_BYTES: usize = 100 * 1024 * 1024;

static DEFAULT_MAX_RETAINED_CAPACITY_FROM_ENV: OnceLock<usize> = OnceLock::new();

/// Snapshot of typed host buffers retained by a [`BufferPool`].
///
/// `buffers` counts retained `Vec` allocations, while `capacity_bytes` counts
/// their total element capacity in bytes. Allocators may keep freed memory in
/// process-local arenas after a pool is cleared, so this reports memory that is
/// still live in the pool rather than operating-system RSS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferPoolStats {
    /// Number of retained vector allocations.
    pub buffers: usize,
    /// Total retained vector capacity in bytes.
    pub capacity_bytes: usize,
}

/// Typed buffer pool keyed by element capacity and separated by scalar type.
///
/// Each supported dtype has an independent best-fit pool. Acquired buffers are
/// returned without zero-initialization so kernels can avoid redundant writes
/// when they fully overwrite the output. Use [`PoolScalar::pool_acquire_zeroed`]
/// when the caller may read the buffer before writing every element.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_cpu_kernels::buffer_pool::{BufferPool, PoolScalar};
///
/// let mut pool = BufferPool::new();
/// let buf = pool.acquire_zeroed::<f32>(8);
/// <f32 as PoolScalar>::pool_release(&mut pool, buf);
/// assert_eq!(pool.len(), 1);
/// ```
pub struct BufferPool {
    f64_pool: BTreeMap<usize, Vec<Vec<f64>>>,
    f32_pool: BTreeMap<usize, Vec<Vec<f32>>>,
    i32_pool: BTreeMap<usize, Vec<Vec<i32>>>,
    i64_pool: BTreeMap<usize, Vec<Vec<i64>>>,
    bool_pool: BTreeMap<usize, Vec<Vec<bool>>>,
    c64_pool: BTreeMap<usize, Vec<Vec<Complex64>>>,
    c32_pool: BTreeMap<usize, Vec<Vec<Complex32>>>,
    f64_in_flight: BTreeMap<usize, usize>,
    f32_in_flight: BTreeMap<usize, usize>,
    i32_in_flight: BTreeMap<usize, usize>,
    i64_in_flight: BTreeMap<usize, usize>,
    bool_in_flight: BTreeMap<usize, usize>,
    c64_in_flight: BTreeMap<usize, usize>,
    c32_in_flight: BTreeMap<usize, usize>,
    retained_capacity_bytes: usize,
    max_retained_capacity_bytes: usize,
}

impl fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferPool")
            .field("stats", &self.stats())
            .field(
                "max_retained_capacity_bytes",
                &self.max_retained_capacity_bytes,
            )
            .finish_non_exhaustive()
    }
}

/// Scalar types supported by [`BufferPool`].
///
/// The trait is sealed to the scalar dtypes that tenferro currently pools for
/// CPU execution.
///
/// # Examples
///
/// ```rust
/// use tenferro_internal_cpu_kernels::buffer_pool::{BufferPool, PoolScalar};
///
/// let mut pool = BufferPool::new();
/// let mut buf = pool.acquire_zeroed::<f64>(2);
/// buf.copy_from_slice(&[3.0, 4.0]);
/// <f64 as PoolScalar>::pool_release(&mut pool, buf);
/// ```
pub trait PoolScalar: Copy + Sized + Send + Sync + private::Sealed {
    /// Zero value used to initialize acquired buffers.
    fn pool_zero() -> Self;

    /// Acquire a buffer with length `len` and every element set to zero.
    ///
    /// This is the safe path for callers that may read the buffer before every
    /// element is overwritten. Full-overwrite kernels should use
    /// [`crate::PooledUninitOutput`] or an operation-specific uninitialized
    /// destination guard.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::new();
    /// let buf = <f64 as PoolScalar>::pool_acquire_zeroed(&mut pool, 2);
    /// assert_eq!(buf, vec![0.0, 0.0]);
    /// ```
    fn pool_acquire_zeroed(pool: &mut BufferPool, len: usize) -> Vec<Self>;

    /// Return a buffer to the typed pool for later reuse.
    ///
    /// Zero-capacity buffers are ignored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::new();
    /// let buf = vec![1.0_f32; 4];
    /// <f32 as PoolScalar>::pool_release(&mut pool, buf);
    /// assert_eq!(pool.len(), 1);
    /// ```
    fn pool_release(pool: &mut BufferPool, buf: Vec<Self>);
}

pub(crate) mod private {
    use std::mem::MaybeUninit;

    // INVARIANT: this sealed crate-private trait is the only implementation
    // boundary for tracked guard tokens; it is not part of the public API.
    #[allow(private_interfaces)]
    pub trait Sealed {
        /// Checks out uninitialized storage with an exact cleanup token.
        ///
        /// # Errors
        /// Returns `Error::Validation` if retained-capacity byte accounting
        /// overflows, or `Error::BackendSource` if fresh allocation fails.
        fn pool_acquire_uninit_tracked(
            pool: &mut super::BufferPool,
            len: usize,
        ) -> crate::Result<(Vec<MaybeUninit<Self>>, super::UninitCheckoutToken)>
        where
            Self: Sized;
        fn pool_discard_uninit(
            pool: &mut super::BufferPool,
            data: Vec<MaybeUninit<Self>>,
            checkout: super::UninitCheckoutToken,
        ) where
            Self: Sized;
    }
}

fn take_best_fit<T>(pool: &mut BTreeMap<usize, Vec<Vec<T>>>, len: usize) -> Option<Vec<T>> {
    let key = *pool.range(len..).next()?.0;
    let buf = {
        let vecs = pool.get_mut(&key)?;
        vecs.pop()
    };
    if pool.get(&key).is_some_and(Vec::is_empty) {
        pool.remove(&key);
    }
    buf
}

fn pool_len<T>(pool: &BTreeMap<usize, Vec<Vec<T>>>) -> usize {
    pool.values().map(Vec::len).sum()
}

fn evict_one_from_pool<T>(pool: &mut BTreeMap<usize, Vec<Vec<T>>>) -> Option<usize> {
    let key = *pool.keys().next()?;
    let vecs = pool.get_mut(&key)?;
    let _ = vecs.pop()?;
    if vecs.is_empty() {
        pool.remove(&key);
    }
    Some(key.saturating_mul(size_of::<T>()))
}

#[derive(Clone, Copy)]
enum TypedPoolKind {
    F64,
    F32,
    I32,
    I64,
    Bool,
    C64,
    C32,
}

fn smallest_pool_candidate<T>(
    pool: &BTreeMap<usize, Vec<Vec<T>>>,
    kind: TypedPoolKind,
) -> Option<(usize, TypedPoolKind)> {
    pool.keys()
        .next()
        .map(|&capacity| (capacity.saturating_mul(size_of::<T>()), kind))
}

fn increment_in_flight(in_flight: &mut BTreeMap<usize, usize>, cap: usize) {
    if cap > 0 {
        *in_flight.entry(cap).or_default() += 1;
    }
}

fn decrement_in_flight(in_flight: &mut BTreeMap<usize, usize>, cap: usize) {
    if cap == 0 {
        return;
    }
    let Some(count) = in_flight.get_mut(&cap) else {
        return;
    };
    *count -= 1;
    if *count == 0 {
        in_flight.remove(&cap);
    }
}

fn replenish_in_flight_for<T>(
    pool: &mut BTreeMap<usize, Vec<Vec<T>>>,
    in_flight: &mut BTreeMap<usize, usize>,
    retained_capacity_bytes: &mut usize,
) {
    for (&cap, &count) in in_flight.iter() {
        for _ in 0..count {
            let mut replacement = Vec::new();
            if replacement.try_reserve_exact(cap).is_err() {
                continue;
            }
            let actual_cap = replacement.capacity();
            *retained_capacity_bytes =
                retained_capacity_bytes.saturating_add(actual_cap.saturating_mul(size_of::<T>()));
            pool.entry(actual_cap).or_default().push(replacement);
        }
    }
    in_flight.clear();
}

macro_rules! impl_pool_scalar {
    ($ty:ty, $field:ident, $in_flight:ident, $zero:expr) => {
        // INVARIANT: this implementation is reachable only through the sealed
        // crate-private helper and cannot be named by sibling crates.
        #[allow(private_interfaces)]
        impl private::Sealed for $ty {
            fn pool_acquire_uninit_tracked(
                pool: &mut BufferPool,
                len: usize,
            ) -> crate::Result<(Vec<MaybeUninit<Self>>, UninitCheckoutToken)> {
                match take_best_fit(&mut pool.$field, len) {
                    Some(buf) => {
                        let cap = buf.capacity();
                        let bytes = cap.checked_mul(size_of::<Self>()).ok_or_else(|| {
                            crate::Error::invalid_argument(
                                "pooled_uninit_output",
                                "length",
                                "pool capacity byte length overflow",
                            )
                        })?;
                        pool.retained_capacity_bytes -= bytes;
                        increment_in_flight(&mut pool.$in_flight, cap);
                        let mut buf = ManuallyDrop::new(buf);
                        // SAFETY: MaybeUninit<Self> has the same layout as Self and len <= cap.
                        let buf = unsafe { Vec::from_raw_parts(buf.as_mut_ptr().cast(), len, cap) };
                        Ok((
                            buf,
                            UninitCheckoutToken::Reused {
                                actual_capacity: cap,
                            },
                        ))
                    }
                    None => {
                        let mut buf = Vec::new();
                        buf.try_reserve_exact(len).map_err(|err| {
                            crate::Error::backend_source("pooled_uninit_output", err)
                        })?;
                        // SAFETY: every bit pattern is valid for MaybeUninit.
                        unsafe { buf.set_len(len) };
                        let actual_capacity = buf.capacity();
                        Ok((buf, UninitCheckoutToken::Fresh { actual_capacity }))
                    }
                }
            }

            fn pool_discard_uninit(
                pool: &mut BufferPool,
                data: Vec<MaybeUninit<Self>>,
                checkout: UninitCheckoutToken,
            ) {
                drop(data);
                if let UninitCheckoutToken::Reused { actual_capacity } = checkout {
                    decrement_in_flight(&mut pool.$in_flight, actual_capacity);
                }
            }
        }

        impl PoolScalar for $ty {
            fn pool_zero() -> Self {
                $zero
            }

            fn pool_acquire_zeroed(pool: &mut BufferPool, len: usize) -> Vec<Self> {
                match take_best_fit(&mut pool.$field, len) {
                    Some(mut buf) => {
                        pool.retained_capacity_bytes = pool
                            .retained_capacity_bytes
                            .saturating_sub(buf.capacity().saturating_mul(size_of::<Self>()));
                        increment_in_flight(&mut pool.$in_flight, buf.capacity());
                        buf.resize(len, Self::pool_zero());
                        buf.fill(Self::pool_zero());
                        buf
                    }
                    None => vec![Self::pool_zero(); len],
                }
            }

            fn pool_release(pool: &mut BufferPool, buf: Vec<Self>) {
                let cap = buf.capacity();
                if cap > 0 {
                    decrement_in_flight(&mut pool.$in_flight, cap);
                    pool.retained_capacity_bytes = pool
                        .retained_capacity_bytes
                        .saturating_add(cap.saturating_mul(size_of::<Self>()));
                    pool.$field.entry(cap).or_default().push(buf);
                    pool.enforce_retention_limit();
                }
            }
        }
    };
}

impl_pool_scalar!(f64, f64_pool, f64_in_flight, 0.0);
impl_pool_scalar!(f32, f32_pool, f32_in_flight, 0.0);
impl_pool_scalar!(i32, i32_pool, i32_in_flight, 0);
impl_pool_scalar!(i64, i64_pool, i64_in_flight, 0);
impl_pool_scalar!(bool, bool_pool, bool_in_flight, false);
impl_pool_scalar!(Complex64, c64_pool, c64_in_flight, Complex64::new(0.0, 0.0));
impl_pool_scalar!(Complex32, c32_pool, c32_in_flight, Complex32::new(0.0, 0.0));

impl BufferPool {
    #[cfg(test)]
    pub(crate) fn in_flight_is_empty(&self) -> bool {
        self.f64_in_flight.is_empty()
            && self.f32_in_flight.is_empty()
            && self.i32_in_flight.is_empty()
            && self.i64_in_flight.is_empty()
            && self.bool_in_flight.is_empty()
            && self.c64_in_flight.is_empty()
            && self.c32_in_flight.is_empty()
    }
    /// Create an empty typed buffer pool.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::buffer_pool::BufferPool;
    ///
    /// let pool = BufferPool::new();
    /// assert!(pool.is_empty());
    /// ```
    pub fn new() -> Self {
        Self::with_max_retained_capacity_bytes(default_max_retained_capacity_bytes())
    }

    /// Create an empty typed buffer pool with a specific retention cap.
    ///
    /// A cap of zero disables retention. Use [`BufferPool::unbounded`] only for
    /// diagnostics or workloads that are externally memory-limited.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::buffer_pool::BufferPool;
    ///
    /// let pool = BufferPool::with_max_retained_capacity_bytes(1024);
    /// assert_eq!(pool.max_retained_capacity_bytes(), 1024);
    /// ```
    pub fn with_max_retained_capacity_bytes(max_retained_capacity_bytes: usize) -> Self {
        Self {
            f64_pool: BTreeMap::new(),
            f32_pool: BTreeMap::new(),
            i32_pool: BTreeMap::new(),
            i64_pool: BTreeMap::new(),
            bool_pool: BTreeMap::new(),
            c64_pool: BTreeMap::new(),
            c32_pool: BTreeMap::new(),
            f64_in_flight: BTreeMap::new(),
            f32_in_flight: BTreeMap::new(),
            i32_in_flight: BTreeMap::new(),
            i64_in_flight: BTreeMap::new(),
            bool_in_flight: BTreeMap::new(),
            c64_in_flight: BTreeMap::new(),
            c32_in_flight: BTreeMap::new(),
            retained_capacity_bytes: 0,
            max_retained_capacity_bytes,
        }
    }

    /// Create an empty typed buffer pool without a retention cap.
    ///
    /// This preserves the historical behavior and is mainly useful for
    /// diagnostics or controlled benchmarks.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::buffer_pool::BufferPool;
    ///
    /// let pool = BufferPool::unbounded();
    /// assert_eq!(pool.max_retained_capacity_bytes(), usize::MAX);
    /// ```
    pub fn unbounded() -> Self {
        Self::with_max_retained_capacity_bytes(usize::MAX)
    }

    /// Maximum retained typed host-buffer capacity in bytes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::buffer_pool::BufferPool;
    ///
    /// let pool = BufferPool::with_max_retained_capacity_bytes(4096);
    /// assert_eq!(pool.max_retained_capacity_bytes(), 4096);
    /// ```
    pub fn max_retained_capacity_bytes(&self) -> usize {
        self.max_retained_capacity_bytes
    }

    /// Update the maximum retained typed host-buffer capacity in bytes.
    ///
    /// Shrinking below the currently retained capacity immediately evicts
    /// retained buffers until the new cap is satisfied. A cap of zero disables
    /// retention.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_internal_cpu_kernels::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::with_max_retained_capacity_bytes(1024);
    /// <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(128));
    /// pool.set_max_retained_capacity_bytes(0);
    /// assert_eq!(pool.max_retained_capacity_bytes(), 0);
    /// assert!(pool.is_empty());
    /// ```
    pub fn set_max_retained_capacity_bytes(&mut self, max_retained_capacity_bytes: usize) {
        self.max_retained_capacity_bytes = max_retained_capacity_bytes;
        self.enforce_retention_limit();
    }

    /// Number of retained buffers across all typed pools.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::new();
    /// <f64 as PoolScalar>::pool_release(&mut pool, vec![0.0; 2]);
    /// assert_eq!(pool.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.stats().buffers
    }

    /// Total retained typed host-buffer capacity in bytes.
    ///
    /// This counts capacity that is still live in the pool. The operating
    /// system RSS may remain high after clearing the pool because the process
    /// allocator can keep freed pages for future allocations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::new();
    /// <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(2));
    /// assert_eq!(pool.retained_capacity_bytes(), 16);
    /// ```
    pub fn retained_capacity_bytes(&self) -> usize {
        self.stats().capacity_bytes
    }

    /// Snapshot retained-buffer count and capacity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::new();
    /// <f32 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(4));
    /// let stats = pool.stats();
    /// assert_eq!(stats.buffers, 1);
    /// assert_eq!(stats.capacity_bytes, 16);
    /// ```
    pub fn stats(&self) -> BufferPoolStats {
        BufferPoolStats {
            buffers: pool_len(&self.f64_pool)
                + pool_len(&self.f32_pool)
                + pool_len(&self.i32_pool)
                + pool_len(&self.i64_pool)
                + pool_len(&self.bool_pool)
                + pool_len(&self.c64_pool)
                + pool_len(&self.c32_pool),
            capacity_bytes: self.retained_capacity_bytes,
        }
    }

    /// Return cache-style stats for the buffers retained by this pool.
    ///
    /// `entries` is the number of retained buffers, and `retained_bytes` is the
    /// total retained vector capacity in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_internal_cpu_kernels::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::new();
    /// <f32 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(4));
    /// let stats = pool.cache_stats();
    /// assert_eq!(stats.entries, 1);
    /// assert_eq!(stats.retained_bytes, 16);
    /// ```
    pub fn cache_stats(&self) -> CacheStats {
        let stats = self.stats();
        CacheStats {
            entries: stats.buffers,
            retained_bytes: stats.capacity_bytes,
            hits: 0,
            misses: 0,
            evictions: 0,
            clears: 0,
        }
    }

    /// Acquire a typed vector with length 0 and at least `cap` capacity.
    ///
    /// Returned buffers come from the typed pool when possible and are ready
    /// for push-based population.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::buffer_pool::BufferPool;
    ///
    /// let mut pool = BufferPool::new();
    /// let mut buf = pool.acquire_empty_with_capacity::<f64>(4);
    /// buf.extend_from_slice(&[1.0, 2.0]);
    /// assert_eq!(buf.len(), 2);
    /// assert!(buf.capacity() >= 4);
    /// ```
    #[doc(hidden)]
    pub fn acquire_empty_with_capacity<T: PoolScalar>(&mut self, cap: usize) -> Vec<T> {
        if cap == 0 {
            return Vec::new();
        }

        let (data, _checkout) = <T as private::Sealed>::pool_acquire_uninit_tracked(self, cap)
            .expect("validated typed pool capacity must be acquirable");
        let mut data = ManuallyDrop::new(data);
        let ptr = data.as_mut_ptr().cast::<T>();
        let capacity = data.capacity();
        // SAFETY: `MaybeUninit<T>` and `T` have identical layouts, and the
        // returned vector has length zero, so no element is read before push.
        unsafe { Vec::from_raw_parts(ptr, 0, capacity) }
    }

    /// Acquire a typed vector with length `len` initialized to zero.
    ///
    /// Use this only when the caller may read elements before overwriting the
    /// entire buffer. Full-overwrite kernels should use
    /// [`crate::PooledUninitOutput`] to avoid the initialization cost.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::buffer_pool::BufferPool;
    ///
    /// let mut pool = BufferPool::new();
    /// let buf = pool.acquire_zeroed::<f32>(3);
    /// assert_eq!(buf, vec![0.0, 0.0, 0.0]);
    /// ```
    pub fn acquire_zeroed<T: PoolScalar>(&mut self, len: usize) -> Vec<T> {
        T::pool_acquire_zeroed(self, len)
    }

    /// Whether all typed pools are empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::buffer_pool::BufferPool;
    ///
    /// let pool = BufferPool::new();
    /// assert!(pool.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.f64_pool.is_empty()
            && self.f32_pool.is_empty()
            && self.i32_pool.is_empty()
            && self.i64_pool.is_empty()
            && self.bool_pool.is_empty()
            && self.c64_pool.is_empty()
            && self.c32_pool.is_empty()
    }

    /// Drop all retained buffers from the pool.
    ///
    /// This releases the vectors owned by the pool. The process allocator may
    /// still keep freed pages mapped for reuse, so operating-system RSS is not
    /// guaranteed to fall immediately.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_internal_cpu_kernels::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::new();
    /// <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(8));
    /// pool.clear();
    /// assert!(pool.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.f64_pool.clear();
        self.f32_pool.clear();
        self.i32_pool.clear();
        self.i64_pool.clear();
        self.bool_pool.clear();
        self.c64_pool.clear();
        self.c32_pool.clear();
        self.clear_in_flight_retained();
        self.retained_capacity_bytes = 0;
    }

    #[doc(hidden)]
    pub fn clear_in_flight_retained(&mut self) {
        self.f64_in_flight.clear();
        self.f32_in_flight.clear();
        self.i32_in_flight.clear();
        self.i64_in_flight.clear();
        self.bool_in_flight.clear();
        self.c64_in_flight.clear();
        self.c32_in_flight.clear();
    }

    #[doc(hidden)]
    pub fn replenish_in_flight_retained(&mut self) {
        replenish_in_flight_for(
            &mut self.f64_pool,
            &mut self.f64_in_flight,
            &mut self.retained_capacity_bytes,
        );
        replenish_in_flight_for(
            &mut self.f32_pool,
            &mut self.f32_in_flight,
            &mut self.retained_capacity_bytes,
        );
        replenish_in_flight_for(
            &mut self.i32_pool,
            &mut self.i32_in_flight,
            &mut self.retained_capacity_bytes,
        );
        replenish_in_flight_for(
            &mut self.i64_pool,
            &mut self.i64_in_flight,
            &mut self.retained_capacity_bytes,
        );
        replenish_in_flight_for(
            &mut self.bool_pool,
            &mut self.bool_in_flight,
            &mut self.retained_capacity_bytes,
        );
        replenish_in_flight_for(
            &mut self.c64_pool,
            &mut self.c64_in_flight,
            &mut self.retained_capacity_bytes,
        );
        replenish_in_flight_for(
            &mut self.c32_pool,
            &mut self.c32_in_flight,
            &mut self.retained_capacity_bytes,
        );
        self.enforce_retention_limit();
    }

    fn enforce_retention_limit(&mut self) {
        while self.retained_capacity_bytes > self.max_retained_capacity_bytes {
            let Some(evicted_bytes) = self.evict_smallest_retained_buffer() else {
                self.retained_capacity_bytes = 0;
                return;
            };
            if evicted_bytes == 0 {
                if self.is_empty() {
                    self.retained_capacity_bytes = 0;
                    return;
                }
                continue;
            }
            self.retained_capacity_bytes =
                self.retained_capacity_bytes.saturating_sub(evicted_bytes);
        }
    }

    fn evict_smallest_retained_buffer(&mut self) -> Option<usize> {
        let candidates = [
            smallest_pool_candidate(&self.f64_pool, TypedPoolKind::F64),
            smallest_pool_candidate(&self.f32_pool, TypedPoolKind::F32),
            smallest_pool_candidate(&self.i32_pool, TypedPoolKind::I32),
            smallest_pool_candidate(&self.i64_pool, TypedPoolKind::I64),
            smallest_pool_candidate(&self.bool_pool, TypedPoolKind::Bool),
            smallest_pool_candidate(&self.c64_pool, TypedPoolKind::C64),
            smallest_pool_candidate(&self.c32_pool, TypedPoolKind::C32),
        ];
        let (_, kind) = candidates
            .into_iter()
            .flatten()
            .min_by_key(|(bytes, _)| *bytes)?;
        match kind {
            TypedPoolKind::F64 => evict_one_from_pool(&mut self.f64_pool),
            TypedPoolKind::F32 => evict_one_from_pool(&mut self.f32_pool),
            TypedPoolKind::I32 => evict_one_from_pool(&mut self.i32_pool),
            TypedPoolKind::I64 => evict_one_from_pool(&mut self.i64_pool),
            TypedPoolKind::Bool => evict_one_from_pool(&mut self.bool_pool),
            TypedPoolKind::C64 => evict_one_from_pool(&mut self.c64_pool),
            TypedPoolKind::C32 => evict_one_from_pool(&mut self.c32_pool),
        }
    }
}

fn default_max_retained_capacity_bytes() -> usize {
    *DEFAULT_MAX_RETAINED_CAPACITY_FROM_ENV.get_or_init(|| {
        parse_default_max_retained_capacity_bytes(env::var_os(BUFFER_POOL_MAX_RETAINED_BYTES_ENV))
    })
}

fn parse_default_max_retained_capacity_bytes(value: Option<OsString>) -> usize {
    value
        .and_then(|value| value.into_string().ok())
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_MAX_RETAINED_CAPACITY_BYTES)
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
