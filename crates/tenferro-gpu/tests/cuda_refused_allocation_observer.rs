#![cfg(feature = "cuda")]

//! Host-reservation evidence for a refused device allocation (issue #2327).
//!
//! This target owns the process allocator so it can observe every Rust
//! global-allocator request the refused 200 GiB request performs. It does not
//! observe native CUDA-library allocations. An uninitialized vector can reserve
//! virtual address space without touching pages, so RSS is not evidence; the
//! requested sizes are. The request must fail on the device before
//! `CudaEngine::alloc_output` reserves anything close to its size on the host.

use std::alloc::{GlobalAlloc, Layout, System};
use std::error::Error as _;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tenferro_gpu::cuda::{gpu_available, CudaDeviceId, CudaEngine};
use tenferro_tensor::DType;

const REFUSED_BYTES: usize = 200 * 1024 * 1024 * 1024;
const REFUSED_ELEMENTS: usize = REFUSED_BYTES / std::mem::size_of::<f64>();
const HOST_RESERVATION_CEILING: usize = 1 << 30;

struct ObservingAllocator;

static ARMED: AtomicBool = AtomicBool::new(false);
static LARGEST_HOST_REQUEST: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the caller's original
// layout and pointer; the observer only records the requested size.
unsafe impl GlobalAlloc for ObservingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            LARGEST_HOST_REQUEST.fetch_max(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: `layout` is the caller's, forwarded unchanged.
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer`/`layout` are the caller's, forwarded unchanged.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            LARGEST_HOST_REQUEST.fetch_max(new_size, Ordering::Relaxed);
        }
        // SAFETY: all three arguments are the caller's, forwarded unchanged.
        System.realloc(pointer, layout, new_size)
    }
}

#[global_allocator]
static ALLOCATOR: ObservingAllocator = ObservingAllocator;

fn backend() -> CudaEngine {
    assert!(gpu_available(), "CUDA test requires an available device");
    CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine")
}

fn out_of_memory_bytes(error: &tenferro_tensor::Error) -> Option<usize> {
    let mut source = error.source()?;
    loop {
        if let Some(t4a_cuda_runtime::Error::OutOfMemory { bytes }) = source.downcast_ref() {
            return Some(*bytes);
        }
        source = source.source()?;
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_refused_device_allocation_requests_no_huge_host_reservation() {
    let engine = backend();

    LARGEST_HOST_REQUEST.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    let refused = engine.alloc_output(DType::F64, &[REFUSED_ELEMENTS]);
    ARMED.store(false, Ordering::Relaxed);

    let refused = refused.expect_err("the device cannot satisfy a 200 GiB allocation");
    assert_eq!(
        out_of_memory_bytes(&refused),
        Some(REFUSED_BYTES),
        "expected the runtime's typed out-of-memory cause, got {refused:?}"
    );
    let largest = LARGEST_HOST_REQUEST.load(Ordering::Relaxed);
    assert!(
        largest < HOST_RESERVATION_CEILING,
        "a refused 200 GiB device allocation requested a {largest}-byte host allocation"
    );
}
