//! Simple type-erased buffer pool for reusing allocations.
//!
//! The pool stores freed `Vec<u8>` buffers keyed by byte capacity.
//! When a new allocation is requested, the pool returns the smallest
//! buffer that is large enough, avoiding repeated calls to the global
//! allocator.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro::buffer_pool::BufferPool;
//!
//! let mut pool = BufferPool::new();
//! let buf = pool.allocate(1024);
//! assert!(buf.len() >= 1024);
//! pool.return_buffer(buf);
//! let reused = pool.allocate(512);
//! // `reused` may be the same allocation as `buf`
//! ```

/// A simple buffer pool that recycles `Vec<u8>` allocations.
///
/// Buffers are matched by byte count: the smallest buffer whose capacity
/// is at least the requested size is returned. This avoids the overhead
/// of calling the global allocator for every temporary tensor.
///
/// # Examples
///
/// ```ignore
/// use tenferro::buffer_pool::BufferPool;
///
/// let mut pool = BufferPool::new();
/// let buf = pool.allocate(256);
/// pool.return_buffer(buf);
/// ```
pub struct BufferPool {
    pool: Vec<(usize, Vec<u8>)>,
}

impl BufferPool {
    /// Create an empty buffer pool.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::buffer_pool::BufferPool;
    /// let pool = BufferPool::new();
    /// ```
    pub fn new() -> Self {
        Self { pool: Vec::new() }
    }

    /// Allocate a buffer of at least `n_bytes` bytes.
    ///
    /// If the pool contains a buffer of matching or larger size, it is
    /// returned (resized to `n_bytes`). Otherwise a fresh allocation is made.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::buffer_pool::BufferPool;
    ///
    /// let mut pool = BufferPool::new();
    /// let buf = pool.allocate(64);
    /// assert_eq!(buf.len(), 64);
    /// ```
    pub fn allocate(&mut self, n_bytes: usize) -> Vec<u8> {
        // Find the smallest buffer that fits
        let best = self
            .pool
            .iter()
            .enumerate()
            .filter(|(_, (cap, _))| *cap >= n_bytes)
            .min_by_key(|(_, (cap, _))| *cap)
            .map(|(idx, _)| idx);

        if let Some(idx) = best {
            let (_, mut buf) = self.pool.swap_remove(idx);
            buf.resize(n_bytes, 0);
            buf
        } else {
            vec![0u8; n_bytes]
        }
    }

    /// Return a buffer to the pool for future reuse.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::buffer_pool::BufferPool;
    ///
    /// let mut pool = BufferPool::new();
    /// let buf = pool.allocate(128);
    /// pool.return_buffer(buf);
    /// ```
    pub fn return_buffer(&mut self, buf: Vec<u8>) {
        let cap = buf.capacity();
        if cap > 0 {
            self.pool.push((cap, buf));
        }
    }

    /// Number of buffers currently held in the pool.
    #[inline(never)]
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Whether the pool is empty.
    #[inline(never)]
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }
}

impl Default for BufferPool {
    #[inline(never)]
    fn default() -> Self {
        Self::new()
    }
}
