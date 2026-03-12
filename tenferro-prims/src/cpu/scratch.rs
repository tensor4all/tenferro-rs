#[cfg(feature = "gemm-blas")]
use std::alloc::{self, Layout};
#[cfg(feature = "gemm-blas")]
use std::collections::BTreeMap;
#[cfg(feature = "gemm-blas")]
use std::marker::PhantomData;
#[cfg(feature = "gemm-blas")]
use std::ops::{Deref, DerefMut};
#[cfg(feature = "gemm-blas")]
use std::ptr::NonNull;
#[cfg(feature = "gemm-blas")]
use tenferro_device::{Error, Result};

/// Alignment for all scratch allocations (cache-line / AVX-512).
#[cfg(feature = "gemm-blas")]
const SCRATCH_ALIGN: usize = 64;

#[cfg(feature = "gemm-blas")]
const _: () = {
    assert!(SCRATCH_ALIGN.is_power_of_two());
    assert!(SCRATCH_ALIGN >= std::mem::align_of::<f64>());
};

#[cfg(feature = "gemm-blas")]
fn scratch_layout(cap_bytes: usize) -> Layout {
    // SAFETY: `SCRATCH_ALIGN` is a validated power-of-two constant.
    unsafe { Layout::from_size_align_unchecked(cap_bytes, SCRATCH_ALIGN) }
}

/// Raw byte buffer stored in the pool. Does NOT impl Drop — the pool
/// handles deallocation in its own Drop impl.
#[cfg(feature = "gemm-blas")]
struct RawBuf {
    ptr: NonNull<u8>,
    cap_bytes: usize,
}

#[cfg(feature = "gemm-blas")]
unsafe impl Send for RawBuf {}

/// Typed scratch buffer obtained from [`ScratchPool`].
///
/// Dereferences to `&[T]` / `&mut [T]`. On the normal path the caller
/// returns the buffer to the pool via [`ScratchPool::put`]; if dropped
/// without returning (e.g. during a panic), Drop deallocates the raw
/// memory so there is no leak.
#[cfg(feature = "gemm-blas")]
pub(super) struct ScratchBuf<T> {
    ptr: NonNull<u8>,
    pub(super) cap_bytes: usize,
    len: usize,
    _marker: PhantomData<T>,
}

#[cfg(feature = "gemm-blas")]
impl<T> ScratchBuf<T> {
    fn into_raw(self) -> RawBuf {
        let raw = RawBuf {
            ptr: self.ptr,
            cap_bytes: self.cap_bytes,
        };
        std::mem::forget(self);
        raw
    }
}

#[cfg(feature = "gemm-blas")]
impl<T> Deref for ScratchBuf<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const T, self.len) }
    }
}

#[cfg(feature = "gemm-blas")]
impl<T> DerefMut for ScratchBuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, self.len) }
    }
}

#[cfg(feature = "gemm-blas")]
impl<T> Drop for ScratchBuf<T> {
    fn drop(&mut self) {
        if self.cap_bytes > 0 {
            unsafe { alloc::dealloc(self.ptr.as_ptr(), scratch_layout(self.cap_bytes)) };
        }
    }
}

/// Type-independent byte-level scratch pool. Buffers are keyed by byte
/// capacity so an f64 allocation can be reused for f32 or vice-versa.
#[derive(Default)]
#[cfg(feature = "gemm-blas")]
pub(super) struct ScratchPool {
    pub(super) pool: BTreeMap<usize, Vec<RawBuf>>,
}

#[cfg(feature = "gemm-blas")]
impl ScratchPool {
    /// Obtain a scratch buffer holding at least `len` elements of `T`.
    /// Contents are **uninitialized**; callers must overwrite before reading.
    pub(super) fn take<T>(&mut self, len: usize) -> Result<ScratchBuf<T>> {
        debug_assert!(
            SCRATCH_ALIGN >= std::mem::align_of::<T>(),
            "SCRATCH_ALIGN ({SCRATCH_ALIGN}) < align_of::<T> ({})",
            std::mem::align_of::<T>(),
        );
        let needed = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            Error::InvalidArgument("scratch buffer size overflowed usize".to_string())
        })?;
        let raw = self
            .pool
            .range(needed..)
            .next()
            .map(|(&k, _)| k)
            .and_then(|k| {
                let bucket = self.pool.get_mut(&k)?;
                let buf = bucket.pop()?;
                if bucket.is_empty() {
                    self.pool.remove(&k);
                }
                Some(buf)
            });
        let (ptr, cap_bytes) = match raw {
            Some(buf) => (buf.ptr, buf.cap_bytes),
            None => {
                if needed == 0 {
                    return Ok(ScratchBuf {
                        ptr: NonNull::dangling(),
                        cap_bytes: 0,
                        len: 0,
                        _marker: PhantomData,
                    });
                }
                let layout = scratch_layout(needed);
                let ptr = unsafe { alloc::alloc(layout) };
                if ptr.is_null() {
                    alloc::handle_alloc_error(layout);
                }
                (unsafe { NonNull::new_unchecked(ptr) }, needed)
            }
        };
        Ok(ScratchBuf {
            ptr,
            cap_bytes,
            len,
            _marker: PhantomData,
        })
    }

    /// Return a scratch buffer to the pool for later reuse.
    pub(super) fn put<T>(&mut self, buf: ScratchBuf<T>) {
        let raw = buf.into_raw();
        if raw.cap_bytes == 0 {
            return;
        }
        self.pool.entry(raw.cap_bytes).or_default().push(raw);
    }
}

#[cfg(feature = "gemm-blas")]
impl Drop for ScratchPool {
    fn drop(&mut self) {
        for (_, bufs) in std::mem::take(&mut self.pool) {
            for buf in bufs {
                unsafe { alloc::dealloc(buf.ptr.as_ptr(), scratch_layout(buf.cap_bytes)) };
            }
        }
    }
}
