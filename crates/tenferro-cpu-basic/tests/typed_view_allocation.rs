//! `typed_view` borrows the tensor's cached layout: a view of a host tensor
//! costs no allocation at all, which is what the per-operation helpers rely on.
//!
//! The counting allocator is the pattern used by
//! `crates/tenferro-tensor/tests/borrowed_view_allocation_tests.rs`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use tenferro_cpu_basic::typed_view;
use tenferro_tensor::TypedTensor;

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
fn typed_view_of_a_host_tensor_allocates_nothing() {
    let tensor = TypedTensor::<f64, tenferro_tensor::DynRank, tenferro_tensor::Host>::from_host_vec_col_major(
        vec![2, 3],
        vec![1.0; 6],
    )
    .unwrap()
    .into_dynamic();

    let allocations = count_allocations(|| {
        let view = typed_view("allocation", &tensor).unwrap();
        assert_eq!(view.dims(), &[2, 3]);
        assert_eq!(view.strides(), &[1, 2]);
        // The metadata is borrowed, not copied: the view points at the tensor's
        // own cached layout.
        assert_eq!(view.strides().as_ptr(), tensor.strides().as_ptr());
        black_box(view);
    });

    assert_eq!(allocations, 0);
}

#[test]
fn typed_view_of_a_dynamic_host_payload_allocates_nothing() {
    let tensor = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![2.0; 4]).unwrap();

    let allocations = count_allocations(|| {
        let view = typed_view("allocation", &tensor).unwrap();
        assert_eq!(view.dims(), &[4]);
        black_box(view);
    });

    assert_eq!(allocations, 0);
}

#[test]
fn typed_view_of_a_rank_above_the_inline_capacity_also_borrows() {
    // Rank 10 exceeds `StrideVec`'s inline capacity, so the tensor's layout is
    // heap-backed; the view must still borrow it.
    let shape = vec![2usize; 10];
    let tensor = TypedTensor::<f64>::from_vec_col_major(shape, vec![3.0; 1024]).unwrap();

    let allocations = count_allocations(|| {
        let view = typed_view("allocation", &tensor).unwrap();
        assert_eq!(view.dims().len(), 10);
        assert_eq!(view.strides().as_ptr(), tensor.strides().as_ptr());
        black_box(view);
    });

    assert_eq!(allocations, 0);
}
