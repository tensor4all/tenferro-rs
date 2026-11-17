//! Metal storage: one shared-mode `MTLBuffer`, carried as the seam's buffer.
//!
//! The buffer is a `MTLResourceStorageModeShared` allocation, so its bytes are the same
//! bytes at the same address for the CPU and for the GPU. That is what makes the engine
//! transfer-free and what makes the admission below possible: nothing here maps or
//! unmaps, it only waits for the submissions that named the allocation and then lends the
//! elements.

use std::any::Any;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenferro_tensor::{AllocationDomainId, AllocationId, BackendStorage, HostAccessError};
use tenferro_tensor::{HostReadGuard, HostWriteGuard};

use super::context::RuntimeCell;
use super::pending::PendingTouches;
use super::MetalBackend;

/// One allocation in a Metal context's shared domain.
pub(crate) struct MetalStorage<T> {
    /// The seam buffer: domain identity, allocation identity and the runtime's handle.
    buffer: crate::gpu::Buffer<MetalBackend>,
    /// The newest submission that named this allocation, read or write.
    touches: Arc<PendingTouches>,
    /// The context's runtime, whose lock serialises host access against submission.
    cell: Arc<RuntimeCell>,
    /// Live `map_read`/`map_write` guards. A submission that names this allocation while
    /// the count is non-zero reports `MappedForHost` instead of committing.
    live_host_access: Arc<AtomicUsize>,
    /// Elements, for the borrow length.
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> MetalStorage<T> {
    /// Wrap one runtime allocation.
    pub(crate) fn new(
        buffer: crate::gpu::Buffer<MetalBackend>,
        touches: Arc<PendingTouches>,
        cell: Arc<RuntimeCell>,
        len: usize,
    ) -> Self {
        // The allocation becomes caller-visible only through this storage, so the live
        // byte count is opened here and closed by `Drop`, not when the raw buffer exists.
        cell.note_live_allocation(buffer.len());
        Self {
            buffer,
            touches,
            cell,
            live_host_access: Arc::new(AtomicUsize::new(0)),
            len,
            _marker: PhantomData,
        }
    }

    /// The runtime handle, for a launch argument.
    pub(crate) fn allocation(&self) -> &t4a_metal_runtime::Buffer {
        self.buffer.allocation()
    }

    /// Bytes in the allocation.
    pub(crate) fn bytes(&self) -> usize {
        self.buffer.len()
    }

    /// The pending-work bookkeeping the engine records a commit on.
    pub(crate) fn touches(&self) -> &Arc<PendingTouches> {
        &self.touches
    }

    /// The live host-access count, for the engine's admission check.
    ///
    /// The engine checks it while the context's runtime lock is held, before it encodes
    /// anything, so a guard cannot be created between the check and the commit.
    pub(crate) fn live_counter(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.live_host_access)
    }

    /// Wait for the submissions that named this allocation, then borrow its elements.
    pub(crate) fn host_slice(&self) -> Result<&[T], HostAccessError> {
        self.touches.wait()?;
        Ok(self.shared_borrow())
    }

    /// Wait, then overwrite the allocation from `values`.
    ///
    /// The live count is raised for the copy, so no submission can name this allocation
    /// while the host writes it, and a second host mapping is refused rather than allowed
    /// to overlap this one.
    pub(crate) fn write_from_slice(&self, values: &[T]) -> Result<(), HostAccessError> {
        if values.len() != self.len {
            return Err(HostAccessError::LengthMismatch {
                expected: self.len,
                actual: values.len(),
            });
        }
        // The count goes up under the runtime lock, so a submission either saw it (and was
        // refused) or recorded its completion before the wait below observes it. The wait
        // itself runs outside the lock: it may block for the runtime's timeout, and no
        // submission can name this allocation meanwhile.
        let live = {
            let _runtime = self.cell.lock();
            LiveHostAccess::try_new(&self.live_host_access)?
        };
        let _live = live;
        self.touches.wait()?;
        if self.len == 0 {
            return Ok(());
        }
        let ptr = self.buffer.allocation().host_ptr().cast::<T>();
        // SAFETY: the live count is raised, so no submission names this allocation, and the
        // wait covered every submission that named it before. The allocation holds `len`
        // elements by construction.
        let destination = unsafe { std::slice::from_raw_parts_mut(ptr, self.len) };
        destination.copy_from_slice(values);
        Ok(())
    }

    fn shared_borrow(&self) -> &[T] {
        // A zero-length allocation has no element to borrow and its host pointer may be
        // null; the empty slice is the correct borrow and dereferences nothing.
        if self.len == 0 {
            return &[];
        }
        let ptr = self.buffer.allocation().host_ptr().cast::<T>();
        // SAFETY: the buffer is shared mode, so this is the address the GPU addresses; the
        // caller waited for the last submission that named it, and the allocation holds at
        // least `len` elements (its byte length was computed from that count).
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }
}

impl<T> Drop for MetalStorage<T> {
    fn drop(&mut self) {
        self.cell.note_live_deallocation(self.buffer.len());
    }
}

impl<T: Copy> std::fmt::Debug for MetalStorage<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetalStorage")
            .field("domain", &self.buffer.identity().domain)
            .field("allocation", &self.buffer.identity().allocation)
            .field("bytes", &self.bytes())
            .field(
                "live_host_access",
                &self.live_host_access.load(Ordering::Relaxed),
            )
            .finish()
    }
}

/// One live host guard, counted while it exists.
struct LiveHostAccess {
    count: Arc<AtomicUsize>,
}

impl LiveHostAccess {
    /// Take the allocation's single host-access slot.
    ///
    /// # Errors
    ///
    /// [`HostAccessError::OverlappingHostMapping`] when a mapping of the same allocation is
    /// already live: two host accesses could otherwise overlap, and one of them may write.
    fn try_new(count: &Arc<AtomicUsize>) -> Result<Self, HostAccessError> {
        count
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| HostAccessError::OverlappingHostMapping)?;
        Ok(Self {
            count: Arc::clone(count),
        })
    }
}

impl Drop for LiveHostAccess {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A read guard's deref target: the shared elements, with the live count held.
struct LiveRead<'a, T> {
    values: &'a [T],
    _live: LiveHostAccess,
}

impl<T> Deref for LiveRead<'_, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.values
    }
}

impl<T: Copy + Send + Sync + 'static> BackendStorage<T> for MetalStorage<T> {
    fn backend_family(&self) -> &'static str {
        "metal"
    }

    fn len(&self) -> usize {
        self.len
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        Some(self.buffer.identity().domain)
    }

    fn allocation_id(&self) -> Option<AllocationId> {
        Some(self.buffer.identity().allocation)
    }

    fn map_read(&self) -> Result<HostReadGuard<'_, T>, HostAccessError> {
        // The runtime lock makes the count transition one step against submission: a
        // submission that slipped in before the guard is observed by the wait, and one that
        // starts after it is refused by the count. The wait runs outside the lock, because
        // it may block for the runtime's timeout and would otherwise stall every other
        // submission, allocation and statistic on the context.
        let live = {
            let _runtime = self.cell.lock();
            LiveHostAccess::try_new(&self.live_host_access)?
        };
        self.touches.wait()?;
        Ok(HostReadGuard::new(LiveRead {
            values: self.shared_borrow(),
            _live: live,
        }))
    }

    fn map_write(&mut self) -> Result<HostWriteGuard<'_, T>, HostAccessError> {
        // As in `map_read`: the count goes up under the lock, the wait runs outside it.
        let live = {
            let _runtime = self.cell.lock();
            LiveHostAccess::try_new(&self.live_host_access)?
        };
        self.touches.wait()?;
        let allocation = Arc::clone(self.buffer.allocation());
        let len = self.len;
        Ok(HostWriteGuard::new(len, move |source: &[T]| {
            // The closure owns the live count, so it is released when the guard is dropped.
            let _live = &live;
            if source.len() != len {
                return Err(HostAccessError::LengthMismatch {
                    expected: len,
                    actual: source.len(),
                });
            }
            if len == 0 {
                return Ok(());
            }
            let ptr = allocation.host_ptr().cast::<T>();
            // SAFETY: the guard holds the live count, so no submission names this
            // allocation while it exists, and the wait above covered every submission that
            // named it before. The allocation holds `len` elements.
            let destination = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            destination.copy_from_slice(source);
            Ok(())
        }))
    }

    fn shared_host_slice(&self) -> Result<&[T], HostAccessError> {
        self.host_slice()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
