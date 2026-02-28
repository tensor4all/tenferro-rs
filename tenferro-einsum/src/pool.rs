use std::collections::BTreeMap;

const MAX_POOLED_BYTES: usize = 64 * 1024 * 1024; // 64 MB

/// Typed buffer pool using BTreeMap for O(log n) best-fit allocation.
/// Passed as argument to `execute_tree`, not thread-local.
pub(crate) struct BufferPool<T> {
    buffers: BTreeMap<usize, Vec<Vec<T>>>,
    total_bytes: usize,
}

impl<T> BufferPool<T> {
    pub fn new() -> Self {
        Self {
            buffers: BTreeMap::new(),
            total_bytes: 0,
        }
    }

    /// Take a buffer of at least `len` elements from the pool.
    /// Returns an uninitialized buffer (caller must fill before reading).
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
            // Safety: capacity >= len; caller writes all elements before reading.
            unsafe { buf.set_len(len) };
            return buf;
        }
        let mut buf = Vec::with_capacity(len);
        unsafe { buf.set_len(len) };
        buf
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

impl<T> Default for BufferPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_returns_correct_length() {
        let mut pool = BufferPool::<f64>::new();
        let buf = pool.take(100);
        assert_eq!(buf.len(), 100);
        assert!(buf.capacity() >= 100);
    }

    #[test]
    fn return_and_reuse() {
        let mut pool = BufferPool::<f64>::new();
        let buf = pool.take(100);
        let ptr = buf.as_ptr();
        pool.return_buf(buf);
        let buf2 = pool.take(50);
        assert_eq!(buf2.as_ptr(), ptr);
        assert_eq!(buf2.len(), 50);
    }

    #[test]
    fn best_fit_selection() {
        let mut pool = BufferPool::<f64>::new();
        let small = Vec::<f64>::with_capacity(50);
        let large = Vec::<f64>::with_capacity(200);
        pool.return_buf(small);
        pool.return_buf(large);
        let buf = pool.take(60);
        assert!(buf.capacity() >= 60);
    }
}
