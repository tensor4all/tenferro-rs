use std::alloc::{self, Layout};
use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use tenferro_device::Result;

/// Alignment for reusable raw temporary buffers.
const TEMP_ALIGN: usize = 64;

const _: () = {
    assert!(TEMP_ALIGN.is_power_of_two());
    assert!(TEMP_ALIGN >= std::mem::align_of::<f64>());
};

fn temp_layout(cap_bytes: usize) -> Layout {
    // SAFETY: `TEMP_ALIGN` is a validated power-of-two constant.
    unsafe { Layout::from_size_align_unchecked(cap_bytes, TEMP_ALIGN) }
}

/// Raw byte buffer stored in the pool.
struct RawBuf {
    ptr: NonNull<u8>,
    cap_bytes: usize,
}

unsafe impl Send for RawBuf {}

#[allow(dead_code)]
/// Typed temporary buffer for future raw scratch use.
pub(crate) struct RawTempBuf {
    ptr: NonNull<u8>,
    len: usize,
    cap_bytes: usize,
    _marker: PhantomData<[u8]>,
}

#[allow(dead_code)]
impl RawTempBuf {
    fn into_raw(self) -> RawBuf {
        let raw = RawBuf {
            ptr: self.ptr,
            cap_bytes: self.cap_bytes,
        };
        std::mem::forget(self);
        raw
    }
}

impl Deref for RawTempBuf {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for RawTempBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for RawTempBuf {
    fn drop(&mut self) {
        if self.cap_bytes > 0 {
            unsafe { alloc::dealloc(self.ptr.as_ptr(), temp_layout(self.cap_bytes)) };
        }
    }
}

/// Reusable CPU-side temporary storage.
///
/// Typed vectors are bucketed by element type and capacity, while the raw byte
/// buffer scaffold is reserved for later faer scratch integration.
#[derive(Default)]
pub struct TempPool {
    typed_vecs: HashMap<TypeId, BTreeMap<usize, Vec<Box<dyn Any + Send>>>>,
    raw_bufs: BTreeMap<usize, Vec<RawBuf>>,
}

impl TempPool {
    /// Take a reusable temporary vector with at least `len` capacity.
    pub fn take_vec<T: Send + 'static>(&mut self, len: usize) -> Vec<T> {
        let type_id = TypeId::of::<T>();
        let mut taken = None;
        let mut remove_type_bucket = false;

        if let Some(bucket) = self.typed_vecs.get_mut(&type_id) {
            taken = take_typed_vec_from_bucket(bucket, len);
            remove_type_bucket = bucket.is_empty();
        }

        if remove_type_bucket {
            self.typed_vecs.remove(&type_id);
        }

        taken.unwrap_or_else(|| Vec::with_capacity(len))
    }

    /// Return a temporary vector to the pool for later reuse.
    pub fn put_vec<T: Send + 'static>(&mut self, mut vec: Vec<T>) {
        let cap = vec.capacity();
        if cap == 0 {
            return;
        }
        vec.clear();
        self.typed_vecs
            .entry(TypeId::of::<T>())
            .or_default()
            .entry(cap)
            .or_default()
            .push(Box::new(vec));
    }

    /// Take a raw byte buffer scaffold for future faer scratch use.
    #[allow(dead_code)]
    pub(crate) fn take_raw_bytes(&mut self, len: usize) -> Result<RawTempBuf> {
        if len == 0 {
            return Ok(RawTempBuf {
                ptr: NonNull::dangling(),
                len: 0,
                cap_bytes: 0,
                _marker: PhantomData,
            });
        }

        let mut taken = None;
        let mut remove_bucket = false;
        let cap = self.raw_bufs.range(len..).next().map(|(&cap, _)| cap);
        if let Some(cap) = cap {
            if let Some(bucket) = self.raw_bufs.get_mut(&cap) {
                taken = take_raw_buf_from_bucket(bucket);
                remove_bucket = bucket.is_empty();
            }
        }
        if remove_bucket {
            if let Some(cap) = cap {
                self.raw_bufs.remove(&cap);
            }
        }
        if let Some(raw) = taken {
            return Ok(RawTempBuf {
                ptr: raw.ptr,
                len,
                cap_bytes: raw.cap_bytes,
                _marker: PhantomData,
            });
        }

        let layout = temp_layout(len);
        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Ok(RawTempBuf {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            len,
            cap_bytes: len,
            _marker: PhantomData,
        })
    }

    /// Return a raw byte buffer to the pool for later reuse.
    #[allow(dead_code)]
    pub(crate) fn put_raw_bytes(&mut self, buf: RawTempBuf) {
        let raw = buf.into_raw();
        if raw.cap_bytes == 0 {
            return;
        }
        self.raw_bufs.entry(raw.cap_bytes).or_default().push(raw);
    }
}

fn take_typed_vec_from_bucket<T: Send + 'static>(
    bucket: &mut BTreeMap<usize, Vec<Box<dyn Any + Send>>>,
    min_capacity: usize,
) -> Option<Vec<T>> {
    let cap = bucket.range(min_capacity..).next().map(|(&cap, _)| cap)?;
    let boxed = {
        let entries = bucket.get_mut(&cap)?;
        let boxed = entries.pop()?;
        boxed
    };
    if bucket.get(&cap).is_some_and(|entries| entries.is_empty()) {
        bucket.remove(&cap);
    }
    Some(
        *boxed
            .downcast::<Vec<T>>()
            .expect("typed temp pool bucket had wrong type"),
    )
}

#[allow(dead_code)]
fn take_raw_buf_from_bucket(bucket: &mut Vec<RawBuf>) -> Option<RawBuf> {
    bucket.pop()
}

impl Drop for TempPool {
    fn drop(&mut self) {
        for (_, bufs) in std::mem::take(&mut self.raw_bufs) {
            for buf in bufs {
                unsafe { alloc::dealloc(buf.ptr.as_ptr(), temp_layout(buf.cap_bytes)) };
            }
        }
    }
}
