use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tenferro_cpu::{CpuBackend, CpuBackendKind};

struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: this allocator forwards the unchanged layout to System.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr and layout came from the corresponding System allocation.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: ptr and layout came from System, and new_size is forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn count_allocations(op: impl FnOnce()) -> usize {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::SeqCst);
    op();
    COUNTING.store(false, Ordering::SeqCst);
    ALLOCATIONS.load(Ordering::Relaxed)
}

#[test]
#[cfg(feature = "cpu-faer")]
fn warm_empty_backend_install_has_no_mandatory_allocation() {
    for threads in [1, 2, 4] {
        let backend = CpuBackend::with_threads_and_kind(threads, CpuBackendKind::Faer).unwrap();
        for _ in 0..32 {
            backend.install(|| ());
        }

        let minimum = (0..64)
            .map(|_| count_allocations(|| backend.install(|| ())))
            .min()
            .unwrap();
        assert_eq!(
            minimum, 0,
            "every warm empty install allocated with {threads} workers"
        );
    }
}
