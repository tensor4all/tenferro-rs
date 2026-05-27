//! Typed host buffer pooling for reusable tensor allocations.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_tensor::buffer_pool::{BufferPool, PoolScalar};
//!
//! let mut pool = BufferPool::new();
//! let mut buf = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 4) };
//! buf.fill(1.0);
//! <f64 as PoolScalar>::pool_release(&mut pool, buf);
//! assert_eq!(pool.len(), 1);
//! ```

use std::collections::BTreeMap;
use std::env;
use std::mem::size_of;

use num_complex::{Complex32, Complex64};

use crate::CacheStats;

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
/// returned without zero-initialization so GEMM callers can avoid redundant
/// writes when they fully overwrite the output.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::buffer_pool::{BufferPool, PoolScalar};
///
/// let mut pool = BufferPool::new();
/// let buf = unsafe { <f32 as PoolScalar>::pool_acquire(&mut pool, 8) };
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
    retained_capacity_bytes: usize,
    max_retained_capacity_bytes: usize,
}

/// Scalar types supported by [`BufferPool`].
///
/// The trait is sealed to the scalar dtypes that tenferro currently pools for
/// CPU execution.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::buffer_pool::{BufferPool, PoolScalar};
///
/// let mut pool = BufferPool::new();
/// let mut buf = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 2) };
/// buf.copy_from_slice(&[3.0, 4.0]);
/// <f64 as PoolScalar>::pool_release(&mut pool, buf);
/// ```
pub trait PoolScalar: Copy + Sized + Send + private::Sealed {
    /// Acquire a buffer with length `len`.
    ///
    /// The vector length is set without initializing its contents. Callers must
    /// overwrite every element before any read.
    ///
    /// # Safety
    ///
    /// The returned vector may contain uninitialized elements. Reading any
    /// element before writing it is undefined behavior.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::new();
    /// let mut buf = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 2) };
    /// buf.copy_from_slice(&[1.0, 2.0]);
    /// assert_eq!(buf, vec![1.0, 2.0]);
    /// ```
    unsafe fn pool_acquire(pool: &mut BufferPool, len: usize) -> Vec<Self>;

    /// Return a buffer to the typed pool for later reuse.
    ///
    /// Zero-capacity buffers are ignored.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::new();
    /// let buf = vec![1.0_f32; 4];
    /// <f32 as PoolScalar>::pool_release(&mut pool, buf);
    /// assert_eq!(pool.len(), 1);
    /// ```
    fn pool_release(pool: &mut BufferPool, buf: Vec<Self>);
}

mod private {
    pub trait Sealed {}

    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for bool {}
    impl Sealed for num_complex::Complex64 {}
    impl Sealed for num_complex::Complex32 {}
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

macro_rules! impl_pool_scalar {
    ($ty:ty, $field:ident) => {
        impl PoolScalar for $ty {
            #[allow(clippy::uninit_vec)]
            unsafe fn pool_acquire(pool: &mut BufferPool, len: usize) -> Vec<Self> {
                match take_best_fit(&mut pool.$field, len) {
                    Some(mut buf) => {
                        pool.retained_capacity_bytes = pool
                            .retained_capacity_bytes
                            .saturating_sub(buf.capacity().saturating_mul(size_of::<Self>()));
                        // SAFETY: caller upholds that elements will be written
                        // before any read. len <= capacity by construction.
                        unsafe { buf.set_len(len) };
                        buf
                    }
                    None => {
                        let mut buf = Vec::with_capacity(len);
                        // SAFETY: caller upholds that elements will be written
                        // before any read. len == capacity here.
                        unsafe { buf.set_len(len) };
                        buf
                    }
                }
            }

            fn pool_release(pool: &mut BufferPool, buf: Vec<Self>) {
                let cap = buf.capacity();
                if cap > 0 {
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

impl_pool_scalar!(f64, f64_pool);
impl_pool_scalar!(f32, f32_pool);
impl_pool_scalar!(i32, i32_pool);
impl_pool_scalar!(i64, i64_pool);
impl_pool_scalar!(bool, bool_pool);
impl_pool_scalar!(Complex64, c64_pool);
impl_pool_scalar!(Complex32, c32_pool);

impl BufferPool {
    /// Create an empty typed buffer pool.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::buffer_pool::BufferPool;
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
    /// use tenferro_tensor::buffer_pool::BufferPool;
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
    /// use tenferro_tensor::buffer_pool::BufferPool;
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
    /// use tenferro_tensor::buffer_pool::BufferPool;
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
    /// use tenferro_tensor::buffer_pool::{BufferPool, PoolScalar};
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
    /// use tenferro_tensor::buffer_pool::{BufferPool, PoolScalar};
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
    /// use tenferro_tensor::buffer_pool::{BufferPool, PoolScalar};
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
    /// use tenferro_tensor::buffer_pool::{BufferPool, PoolScalar};
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
    /// use tenferro_tensor::buffer_pool::{BufferPool, PoolScalar};
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
    /// use tenferro_tensor::buffer_pool::BufferPool;
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

        let mut buf = unsafe { T::pool_acquire(self, cap) };
        // SAFETY: shrinking the length to zero does not read the buffer. The
        // pool only stores `PoolScalar` values, which are `Copy`.
        unsafe { buf.set_len(0) };
        buf
    }

    /// Whether all typed pools are empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::buffer_pool::BufferPool;
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
    /// use tenferro_tensor::buffer_pool::{BufferPool, PoolScalar};
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
        self.retained_capacity_bytes = 0;
    }

    fn enforce_retention_limit(&mut self) {
        while self.retained_capacity_bytes > self.max_retained_capacity_bytes {
            let Some(evicted_bytes) = self.evict_smallest_retained_buffer() else {
                self.retained_capacity_bytes = 0;
                return;
            };
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
    env::var(BUFFER_POOL_MAX_RETAINED_BYTES_ENV)
        .ok()
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
