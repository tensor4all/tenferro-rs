use std::collections::BTreeMap;

use tenferro_algebra::Scalar;

const MAX_POOLED_BYTES: usize = 64 * 1024 * 1024; // 64 MB

/// Typed buffer pool using BTreeMap for O(log n) best-fit allocation.
/// Passed as argument to `execute_tree`, not thread-local.
pub(crate) struct BufferPool<T> {
    buffers: BTreeMap<usize, Vec<Vec<T>>>,
    total_bytes: usize,
}

impl<T: Scalar> BufferPool<T> {
    pub fn new() -> Self {
        Self {
            buffers: BTreeMap::new(),
            total_bytes: 0,
        }
    }

    /// Take a zero-initialized buffer of at least `len` elements from the pool.
    pub fn take(&mut self, len: usize) -> Vec<T> {
        // Find smallest buffer with capacity >= len
        let mut found_cap = None;
        if let Some((&cap, bufs)) = self.buffers.range_mut(len..).next() {
            if !bufs.is_empty() {
                found_cap = Some(cap);
            }
        }
        if let Some(cap) = found_cap {
            let bufs = self.buffers.get_mut(&cap).unwrap();
            let mut buf = bufs.pop().unwrap();
            if bufs.is_empty() {
                self.buffers.remove(&cap);
            }
            self.total_bytes -= cap * std::mem::size_of::<T>();
            buf.resize(len, T::zero());
            return buf;
        }
        vec![T::zero(); len]
    }

    /// Return a buffer to the pool for reuse.
    pub fn return_buf(&mut self, mut buf: Vec<T>) {
        let cap = buf.capacity();
        let bytes = cap * std::mem::size_of::<T>();
        if bytes == 0 || self.total_bytes + bytes > MAX_POOLED_BYTES {
            return; // drop
        }
        buf.clear();
        self.total_bytes += bytes;
        self.buffers.entry(cap).or_default().push(buf);
    }
}

impl<T: Scalar> Default for BufferPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
