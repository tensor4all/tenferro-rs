//! `typed_view` builds a strided view of a host tensor. Since strided-view
//! 0.6 the view owns its metadata inline up to `INLINE_RANK`, so the per-
//! operation helper allocates nothing at rank `<= INLINE_RANK` and spills only
//! the metadata at higher rank.
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
        // strided-view 0.6 owns view metadata inline: the view copies the
        // tensor's canonical layout rather than aliasing its cached vectors.
        assert_eq!(view.strides(), tensor.strides());
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
fn typed_view_of_a_rank_above_the_inline_capacity_exposes_the_layout() {
    // Rank 10 exceeds strided-view's `INLINE_RANK`, so the view's metadata is
    // heap-backed; the layout must still be the tensor's canonical one.
    let shape = vec![2usize; 10];
    let tensor = TypedTensor::<f64>::from_vec_col_major(shape, vec![3.0; 1024]).unwrap();

    let allocations = count_allocations(|| {
        let view = typed_view("allocation", &tensor).unwrap();
        assert_eq!(view.dims().len(), 10);
        assert_eq!(view.dims(), tensor.shape());
        assert_eq!(view.strides(), tensor.strides());
        black_box(view);
    });

    // Above `INLINE_RANK` the spill box plus its two metadata vectors are the
    // only allocations the view can make.
    assert!(
        allocations <= 3,
        "rank above INLINE_RANK spilled {allocations} allocations, expected at most 3"
    );
}
