use std::collections::BTreeMap;

use tenferro_algebra::Scalar;
use tenferro_prims::TensorTempPoolContext;

#[cfg_attr(not(test), allow(dead_code))]
const MAX_POOLED_BYTES: usize = 64 * 1024 * 1024; // 64 MB

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) trait TensorBufferPool<T: Scalar> {
    fn take_with_ctx<Ctx: TensorTempPoolContext>(&mut self, ctx: &mut Ctx, len: usize) -> Vec<T>;
    fn return_buf(&mut self, buf: Vec<T>);
}

/// Typed local buffer pool using BTreeMap for O(log n) best-fit allocation.
///
/// The pool is call-local. When used from production entry points, misses can
/// fall back to the backend context temp pool and the local cache is flushed
/// back into the context at the end of the call.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct BufferPool<T> {
    buffers: BTreeMap<usize, Vec<Vec<T>>>,
    total_bytes: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
impl<T: Scalar> BufferPool<T> {
    pub fn new() -> Self {
        Self {
            buffers: BTreeMap::new(),
            total_bytes: 0,
        }
    }

    pub fn take(&mut self, len: usize) -> Vec<T> {
        self.take_local(len).unwrap_or_else(|| vec![T::zero(); len])
    }

    pub fn return_buf(&mut self, buf: Vec<T>) {
        self.return_local(buf);
    }

    pub fn flush_to_context<Ctx: TensorTempPoolContext>(&mut self, ctx: &mut Ctx)
    where
        T: Send + 'static,
    {
        for (_, mut bufs) in std::mem::take(&mut self.buffers) {
            for buf in bufs.drain(..) {
                ctx.put_temp_vec(buf);
            }
        }
        self.total_bytes = 0;
    }

    fn take_local(&mut self, len: usize) -> Option<Vec<T>> {
        let mut found_cap = None;
        if let Some((&cap, bufs)) = self.buffers.range_mut(len..).next() {
            if !bufs.is_empty() {
                found_cap = Some(cap);
            }
        }
        let cap = found_cap?;
        let bufs = self
            .buffers
            .get_mut(&cap)
            .expect("buffer bucket disappeared");
        let mut buf = bufs.pop().expect("buffer bucket was unexpectedly empty");
        if bufs.is_empty() {
            self.buffers.remove(&cap);
        }
        self.total_bytes -= cap * std::mem::size_of::<T>();
        buf.resize(len, T::zero());
        Some(buf)
    }

    fn return_local(&mut self, mut buf: Vec<T>) {
        let cap = buf.capacity();
        let bytes = cap * std::mem::size_of::<T>();
        if bytes == 0 || self.total_bytes + bytes > MAX_POOLED_BYTES {
            return;
        }
        buf.clear();
        self.total_bytes += bytes;
        self.buffers.entry(cap).or_default().push(buf);
    }
}

impl<T: Scalar> TensorBufferPool<T> for BufferPool<T>
where
    T: Send + 'static,
{
    fn take_with_ctx<Ctx: TensorTempPoolContext>(&mut self, ctx: &mut Ctx, len: usize) -> Vec<T> {
        if let Some(buf) = self.take_local(len) {
            return buf;
        }
        let mut buf = ctx.take_temp_vec::<T>(len);
        buf.resize(len, T::zero());
        buf
    }

    fn return_buf(&mut self, buf: Vec<T>) {
        self.return_local(buf);
    }
}

impl<T: Scalar> Default for BufferPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn with_context_buffer_pool<T, Ctx, R>(
    ctx: &mut Ctx,
    f: impl FnOnce(&mut Ctx, &mut BufferPool<T>) -> R,
) -> R
where
    T: Scalar + Send + 'static,
    Ctx: TensorTempPoolContext,
{
    let mut pool = BufferPool::new();
    let result = f(ctx, &mut pool);
    pool.flush_to_context(ctx);
    result
}

#[cfg(test)]
mod tests;
