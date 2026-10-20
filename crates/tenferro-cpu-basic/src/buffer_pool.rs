//! Typed host buffer pooling for reusable tensor allocations.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_cpu_basic::buffer_pool::{BufferPool, PoolScalar};
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
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use tenferro_tensor::HostBufferRecycler;

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
/// use tenferro_cpu_basic::buffer_pool::{BufferPool, PoolScalar};
///
/// let mut pool = BufferPool::new();
/// let buf = pool.acquire_zeroed::<f32>(8);
/// <f32 as PoolScalar>::pool_release(&mut pool, buf);
/// assert_eq!(pool.len(), 1);
/// ```
pub struct BufferPool {
    state: Arc<SharedPool>,
}

#[derive(Debug)]
struct SharedPool(Mutex<PoolState>);

#[derive(Debug)]
struct PoolState {
    f64_pool: BTreeMap<usize, Vec<Vec<f64>>>,
    f32_pool: BTreeMap<usize, Vec<Vec<f32>>>,
    i32_pool: BTreeMap<usize, Vec<Vec<i32>>>,
    i64_pool: BTreeMap<usize, Vec<Vec<i64>>>,
    bool_pool: BTreeMap<usize, Vec<Vec<bool>>>,
    c64_pool: BTreeMap<usize, Vec<Vec<Complex64>>>,
    c32_pool: BTreeMap<usize, Vec<Vec<Complex32>>>,
    retained_capacity_bytes: usize,
    max_retained_capacity_bytes: usize,
}

impl fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferPool")
            .field("stats", &self.stats())
            .field(
                "max_retained_capacity_bytes",
                &self.max_retained_capacity_bytes(),
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
/// use tenferro_cpu_basic::buffer_pool::{BufferPool, PoolScalar};
///
/// let mut pool = BufferPool::new();
/// let mut buf = pool.acquire_zeroed::<f64>(2);
/// buf.copy_from_slice(&[3.0, 4.0]);
/// <f64 as PoolScalar>::pool_release(&mut pool, buf);
/// ```
pub trait PoolScalar:
    Copy + Sized + Send + Sync + tenferro_tensor::TensorScalar + private::Sealed
{
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
    /// use tenferro_cpu_basic::buffer_pool::{BufferPool, PoolScalar};
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
    /// use tenferro_cpu_basic::buffer_pool::{BufferPool, PoolScalar};
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
        fn pool_finish_recycled(
            pool: &mut super::BufferPool,
            checkout: super::UninitCheckoutToken,
        ) -> std::sync::Weak<dyn tenferro_tensor::HostBufferRecycler<Self>>
        where
            Self: Sized;
    }
}

/// How many emptied bins one acquisition may walk before it starts dropping them.
///
/// A bin that empties while another capacity is taken is usually transient: a kernel
/// that cycles a handful of scratch sizes keeps them in flight in turn, and
/// re-inserting their map nodes is what the pool exists to avoid. `tenferro-linalg`'s
/// LAPACK allocation ceilings count exactly that and are the guard: they hold at this
/// budget of 8 or 16 and break at 4 (measured locally with
/// `--no-default-features --features blas`), so 16 leaves margin above the empties an
/// SVD/QR call crosses.
///
/// A varying-shape run instead leaves one emptied bin per capacity it has ever used,
/// and then every acquisition walks all of them (#2170: 41.8 us per acquire after
/// 10 000 distinct capacities against 21 ns with none). Past this budget the emptied
/// bins the walk passes are capacities the run has stopped using, so they go, and the
/// walk stays bounded by the budget instead of by the history.
const EMPTIED_BIN_SCAN_BUDGET: usize = 16;

fn take_best_fit<T>(pool: &mut BTreeMap<usize, Vec<Vec<T>>>, len: usize) -> Option<Vec<T>> {
    // Walk upward from `len` to the smallest retained capacity that can serve it,
    // counting the emptied bins passed and dropping them once past the budget. The
    // walk is one pass and allocates nothing until the budget is exceeded, so a hot
    // loop that finds its own bin immediately pays nothing for it.
    let mut cursor = len;
    let mut emptied = 0usize;
    let mut stale: Vec<usize> = Vec::new();
    let serving = loop {
        match pool
            .range(cursor..)
            .next()
            .map(|(&key, vecs)| (key, vecs.is_empty()))
        {
            None => break None,
            Some((key, false)) => break Some(key),
            Some((key, true)) => {
                emptied += 1;
                if emptied > EMPTIED_BIN_SCAN_BUDGET {
                    stale.push(key);
                }
                cursor = key.saturating_add(1);
            }
        }
    };
    for key in stale {
        pool.remove(&key);
    }
    let key = serving?;
    let vecs = pool.get_mut(&key)?;
    vecs.pop()
}

fn pool_len<T>(pool: &BTreeMap<usize, Vec<Vec<T>>>) -> usize {
    pool.values().map(Vec::len).sum()
}

#[cfg(test)]
fn empty_bins<T>(pool: &BTreeMap<usize, Vec<Vec<T>>>) -> usize {
    pool.values().filter(|vecs| vecs.is_empty()).count()
}

fn evict_one_from_pool<T>(pool: &mut BTreeMap<usize, Vec<Vec<T>>>) -> Option<usize> {
    let key = *pool.iter().find(|(_, vecs)| !vecs.is_empty())?.0;
    let vecs = pool.get_mut(&key)?;
    let _ = vecs.pop()?;
    if vecs.is_empty() {
        // `take_best_fit` drops an emptied bin too; eviction cannot leave one
        // behind either.
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
    // Bins are dropped when they empty (see `take_best_fit`), so the first
    // entry is the smallest retained capacity; the emptiness check stays as a
    // guard for a bin a future path could leave behind.
    pool.iter()
        .find(|(_, vecs)| !vecs.is_empty())
        .map(|(&capacity, _)| (capacity.saturating_mul(size_of::<T>()), kind))
}

macro_rules! impl_pool_scalar {
    ($ty:ty, $field:ident, $zero:expr) => {
        // INVARIANT: this implementation is reachable only through the sealed
        // crate-private helper and cannot be named by sibling crates.
        #[allow(private_interfaces)]
        impl private::Sealed for $ty {
            fn pool_acquire_uninit_tracked(
                pool: &mut BufferPool,
                len: usize,
            ) -> crate::Result<(Vec<MaybeUninit<Self>>, UninitCheckoutToken)> {
                let mut state = lock_pool(&pool.state);
                let pool = &mut *state;
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

            fn pool_finish_recycled(
                pool: &mut BufferPool,
                checkout: UninitCheckoutToken,
            ) -> Weak<dyn HostBufferRecycler<Self>> {
                let _ = checkout;
                let owner: Arc<dyn HostBufferRecycler<Self>> = pool.state.clone();
                Arc::downgrade(&owner)
            }
        }

        impl PoolScalar for $ty {
            fn pool_zero() -> Self {
                $zero
            }

            fn pool_acquire_zeroed(pool: &mut BufferPool, len: usize) -> Vec<Self> {
                let mut state = lock_pool(&pool.state);
                let pool = &mut *state;
                match take_best_fit(&mut pool.$field, len) {
                    Some(mut buf) => {
                        pool.retained_capacity_bytes = pool
                            .retained_capacity_bytes
                            .saturating_sub(buf.capacity().saturating_mul(size_of::<Self>()));
                        buf.resize(len, Self::pool_zero());
                        buf.fill(Self::pool_zero());
                        buf
                    }
                    None => vec![Self::pool_zero(); len],
                }
            }

            fn pool_release(pool: &mut BufferPool, buf: Vec<Self>) {
                pool.state.recycle(buf);
            }
        }

        impl HostBufferRecycler<$ty> for SharedPool {
            fn recycle(&self, buf: Vec<$ty>) {
                let mut state = lock_pool(self);
                let pool = &mut *state;
                let cap = buf.capacity();
                if cap > 0 {
                    pool.retained_capacity_bytes = pool
                        .retained_capacity_bytes
                        .saturating_add(cap.saturating_mul(size_of::<$ty>()));
                    pool.$field.entry(cap).or_default().push(buf);
                    pool.enforce_retention_limit();
                }
            }
        }
    };
}

impl_pool_scalar!(f64, f64_pool, 0.0);
impl_pool_scalar!(f32, f32_pool, 0.0);
impl_pool_scalar!(i32, i32_pool, 0);
impl_pool_scalar!(i64, i64_pool, 0);
impl_pool_scalar!(bool, bool_pool, false);
impl_pool_scalar!(Complex64, c64_pool, Complex64::new(0.0, 0.0));
impl_pool_scalar!(Complex32, c32_pool, Complex32::new(0.0, 0.0));

impl BufferPool {
    pub(crate) fn checkout_handle(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }

    /// Number of retained capacity bins, one per capacity the pool still holds.
    #[cfg(test)]
    pub(crate) fn bin_count(&self) -> usize {
        let state = lock_pool(&self.state);
        state.f64_pool.len()
            + state.f32_pool.len()
            + state.i32_pool.len()
            + state.i64_pool.len()
            + state.bool_pool.len()
            + state.c64_pool.len()
            + state.c32_pool.len()
    }

    /// Number of retained capacity bins that hold no buffer.
    #[cfg(test)]
    pub(crate) fn empty_bin_count(&self) -> usize {
        let state = lock_pool(&self.state);
        empty_bins(&state.f64_pool)
            + empty_bins(&state.f32_pool)
            + empty_bins(&state.i32_pool)
            + empty_bins(&state.i64_pool)
            + empty_bins(&state.bool_pool)
            + empty_bins(&state.c64_pool)
            + empty_bins(&state.c32_pool)
    }

    /// Create an empty typed buffer pool.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu_basic::buffer_pool::BufferPool;
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
    /// use tenferro_cpu_basic::buffer_pool::BufferPool;
    ///
    /// let pool = BufferPool::with_max_retained_capacity_bytes(1024);
    /// assert_eq!(pool.max_retained_capacity_bytes(), 1024);
    /// ```
    pub fn with_max_retained_capacity_bytes(max_retained_capacity_bytes: usize) -> Self {
        Self {
            state: Arc::new(SharedPool(Mutex::new(PoolState {
                f64_pool: BTreeMap::new(),
                f32_pool: BTreeMap::new(),
                i32_pool: BTreeMap::new(),
                i64_pool: BTreeMap::new(),
                bool_pool: BTreeMap::new(),
                c64_pool: BTreeMap::new(),
                c32_pool: BTreeMap::new(),
                retained_capacity_bytes: 0,
                max_retained_capacity_bytes,
            }))),
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
    /// use tenferro_cpu_basic::buffer_pool::BufferPool;
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
    /// use tenferro_cpu_basic::buffer_pool::BufferPool;
    ///
    /// let pool = BufferPool::with_max_retained_capacity_bytes(4096);
    /// assert_eq!(pool.max_retained_capacity_bytes(), 4096);
    /// ```
    pub fn max_retained_capacity_bytes(&self) -> usize {
        lock_pool(&self.state).max_retained_capacity_bytes
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
    /// use tenferro_cpu_basic::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::with_max_retained_capacity_bytes(1024);
    /// <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(128));
    /// pool.set_max_retained_capacity_bytes(0);
    /// assert_eq!(pool.max_retained_capacity_bytes(), 0);
    /// assert!(pool.is_empty());
    /// ```
    pub fn set_max_retained_capacity_bytes(&mut self, max_retained_capacity_bytes: usize) {
        let mut state = lock_pool(&self.state);
        state.max_retained_capacity_bytes = max_retained_capacity_bytes;
        state.enforce_retention_limit();
    }

    /// Number of retained buffers across all typed pools.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu_basic::buffer_pool::{BufferPool, PoolScalar};
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
    /// use tenferro_cpu_basic::buffer_pool::{BufferPool, PoolScalar};
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
    /// use tenferro_cpu_basic::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::new();
    /// <f32 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(4));
    /// let stats = pool.stats();
    /// assert_eq!(stats.buffers, 1);
    /// assert_eq!(stats.capacity_bytes, 16);
    /// ```
    pub fn stats(&self) -> BufferPoolStats {
        let state = lock_pool(&self.state);
        BufferPoolStats {
            buffers: pool_len(&state.f64_pool)
                + pool_len(&state.f32_pool)
                + pool_len(&state.i32_pool)
                + pool_len(&state.i64_pool)
                + pool_len(&state.bool_pool)
                + pool_len(&state.c64_pool)
                + pool_len(&state.c32_pool),
            capacity_bytes: state.retained_capacity_bytes,
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
    /// use tenferro_cpu_basic::buffer_pool::{BufferPool, PoolScalar};
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
    /// use tenferro_cpu_basic::buffer_pool::BufferPool;
    ///
    /// let mut pool = BufferPool::new();
    /// let mut buf = pool.acquire_with_capacity::<f64>(4);
    /// buf.extend_from_slice(&[1.0, 2.0]);
    /// assert_eq!(buf.len(), 2);
    /// assert!(buf.capacity() >= 4);
    /// ```
    pub fn acquire_with_capacity<T: PoolScalar>(&mut self, cap: usize) -> Vec<T> {
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
    /// use tenferro_cpu_basic::buffer_pool::BufferPool;
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
    /// use tenferro_cpu_basic::buffer_pool::BufferPool;
    ///
    /// let pool = BufferPool::new();
    /// assert!(pool.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    /// use tenferro_cpu_basic::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::new();
    /// <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(8));
    /// pool.clear();
    /// assert!(pool.is_empty());
    /// ```
    pub fn clear(&mut self) {
        let mut state = lock_pool(&self.state);
        state.clear();
    }
}

// INVARIANT: no user callback or tensor destructor runs under this lock; it
// protects only scalar Vec bins and their accounting, never execution admission.
fn lock_pool(state: &SharedPool) -> MutexGuard<'_, PoolState> {
    state
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl PoolState {
    fn clear(&mut self) {
        self.f64_pool.clear();
        self.f32_pool.clear();
        self.i32_pool.clear();
        self.i64_pool.clear();
        self.bool_pool.clear();
        self.c64_pool.clear();
        self.c32_pool.clear();
        self.retained_capacity_bytes = 0;
    }

    fn enforce_retention_limit(&mut self) {
        while self.retained_capacity_bytes > self.max_retained_capacity_bytes {
            let Some(evicted_bytes) = self.evict_smallest_retained_buffer() else {
                self.retained_capacity_bytes = 0;
                return;
            };
            if evicted_bytes == 0 {
                if self.retained_capacity_bytes == 0 {
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
