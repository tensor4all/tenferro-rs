//! Typed host buffer pooling for reusable tensor allocations.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_tensor::buffer_pool::{BufferPool, PoolScalar};
//!
//! let mut pool = BufferPool::new();
//! let mut buf = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 4) };
//! buf.fill(1.0);
//! <f64 as PoolScalar>::pool_release(&mut pool, buf);
//! assert_eq!(pool.len(), 1);
//! ```

use std::collections::BTreeMap;

use num_complex::{Complex32, Complex64};

/// Typed buffer pool keyed by element capacity and separated by scalar type.
///
/// Each supported dtype has an independent best-fit pool. Acquired buffers are
/// returned without zero-initialization so GEMM callers can avoid redundant
/// writes when they fully overwrite the output.
///
/// # Examples
///
/// ```ignore
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
    c64_pool: BTreeMap<usize, Vec<Vec<Complex64>>>,
    c32_pool: BTreeMap<usize, Vec<Vec<Complex32>>>,
}

/// Scalar types supported by [`BufferPool`].
///
/// The trait is sealed to the scalar dtypes that tenferro currently pools for
/// CPU execution.
///
/// # Examples
///
/// ```ignore
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
    /// ```ignore
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
    /// ```ignore
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

macro_rules! impl_pool_scalar {
    ($ty:ty, $field:ident) => {
        impl PoolScalar for $ty {
            #[allow(clippy::uninit_vec)]
            unsafe fn pool_acquire(pool: &mut BufferPool, len: usize) -> Vec<Self> {
                match take_best_fit(&mut pool.$field, len) {
                    Some(mut buf) => {
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
                    pool.$field.entry(cap).or_default().push(buf);
                }
            }
        }
    };
}

impl_pool_scalar!(f64, f64_pool);
impl_pool_scalar!(f32, f32_pool);
impl_pool_scalar!(Complex64, c64_pool);
impl_pool_scalar!(Complex32, c32_pool);

impl BufferPool {
    /// Create an empty typed buffer pool.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::buffer_pool::BufferPool;
    ///
    /// let pool = BufferPool::new();
    /// assert!(pool.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            f64_pool: BTreeMap::new(),
            f32_pool: BTreeMap::new(),
            c64_pool: BTreeMap::new(),
            c32_pool: BTreeMap::new(),
        }
    }

    /// Number of retained buffers across all typed pools.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::buffer_pool::{BufferPool, PoolScalar};
    ///
    /// let mut pool = BufferPool::new();
    /// <f64 as PoolScalar>::pool_release(&mut pool, vec![0.0; 2]);
    /// assert_eq!(pool.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.f64_pool.values().map(Vec::len).sum::<usize>()
            + self.f32_pool.values().map(Vec::len).sum::<usize>()
            + self.c64_pool.values().map(Vec::len).sum::<usize>()
            + self.c32_pool.values().map(Vec::len).sum::<usize>()
    }

    /// Acquire a typed vector with length 0 and at least `cap` capacity.
    ///
    /// Returned buffers come from the typed pool when possible and are ready
    /// for push-based population.
    ///
    /// # Examples
    ///
    /// ```ignore
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
    /// ```ignore
    /// use tenferro_tensor::buffer_pool::BufferPool;
    ///
    /// let pool = BufferPool::new();
    /// assert!(pool.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.f64_pool.is_empty()
            && self.f32_pool.is_empty()
            && self.c64_pool.is_empty()
            && self.c32_pool.is_empty()
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferPool, PoolScalar};

    #[test]
    fn acquire_release_reuse() {
        let mut pool = BufferPool::new();

        let buf = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 64) };
        let ptr = buf.as_ptr();
        let cap = buf.capacity();
        <f64 as PoolScalar>::pool_release(&mut pool, buf);

        let reused = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 64) };
        assert_eq!(reused.as_ptr(), ptr);
        assert_eq!(reused.capacity(), cap);
        assert!(pool.is_empty());
    }

    #[test]
    fn best_fit() {
        let mut pool = BufferPool::new();
        <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(100));
        <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(200));
        <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(300));

        let reused = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 150) };
        assert_eq!(reused.capacity(), 200);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn type_separation() {
        let mut pool = BufferPool::new();
        <f64 as PoolScalar>::pool_release(&mut pool, Vec::with_capacity(16));
        assert_eq!(pool.len(), 1);

        let f32_buf = unsafe { <f32 as PoolScalar>::pool_acquire(&mut pool, 16) };
        assert_eq!(f32_buf.capacity(), 16);
        assert_eq!(pool.len(), 1);

        let f64_buf = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 16) };
        assert_eq!(f64_buf.capacity(), 16);
        assert!(pool.is_empty());
    }

    #[test]
    fn fresh_alloc_fallback() {
        let mut pool = BufferPool::new();
        let buf = unsafe { <f64 as PoolScalar>::pool_acquire(&mut pool, 32) };
        assert_eq!(buf.len(), 32);
        assert!(buf.capacity() >= 32);
        assert!(pool.is_empty());
    }

    #[test]
    fn zero_len_not_pooled() {
        let mut pool = BufferPool::new();
        <f64 as PoolScalar>::pool_release(&mut pool, Vec::new());
        assert!(pool.is_empty());
    }

    #[test]
    fn acquire_with_capacity_reuses_buffer_as_empty_vec() {
        let mut pool = BufferPool::new();

        let buf = vec![1.0_f64; 8];
        let ptr = buf.as_ptr();
        let cap = buf.capacity();
        <f64 as PoolScalar>::pool_release(&mut pool, buf);

        let reused = pool.acquire_with_capacity::<f64>(8);
        assert_eq!(reused.as_ptr(), ptr);
        assert_eq!(reused.len(), 0);
        assert_eq!(reused.capacity(), cap);
        assert!(pool.is_empty());
    }
}
