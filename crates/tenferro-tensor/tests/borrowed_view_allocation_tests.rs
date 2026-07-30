use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use tenferro_tensor::{TypedTensorView, TypedTensorViewMut};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        // SAFETY: this allocator forwards the unchanged layout to System.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr and layout came from the corresponding System allocation.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        // SAFETY: ptr and layout came from System, and new_size is forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn count_allocations(op: impl FnOnce()) -> usize {
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    op();
    COUNTING.set(false);
    ALLOCATIONS.get()
}

#[test]
fn small_dynamic_borrowed_view_metadata_stays_inline() {
    let data = [0_i32; 6];
    let read_allocations = count_allocations(|| {
        let view = TypedTensorView::from_slice([2, 3], [1, 2], 0, &data).unwrap();
        black_box(view);
    });

    let mut data = [0_i32; 6];
    let write_allocations = count_allocations(|| {
        let view = TypedTensorViewMut::from_slice([2, 3], [1, 2], 0, &mut data).unwrap();
        black_box(view);
    });

    assert_eq!(read_allocations, 0);
    assert_eq!(write_allocations, 0);
}
