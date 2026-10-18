//! Acceptance for #2143 item 6: the owned tensor keeps its canonical
//! column-major strides, and the per-operation view helper copies that layout
//! into a view whose metadata is inline up to strided-view's `INLINE_RANK`.
//!
//! The primary signal is the allocation count, not time: at rank
//! `<= INLINE_RANK` a per-operation view that allocates metadata once or twice
//! per eager operation is the defect this pins. Above `INLINE_RANK` the view
//! spills two metadata vectors and no zero-allocation claim is made.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use tenferro_cpu_basic::typed_view;
use tenferro_tensor::{Host, Rank, TypedTensor};

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
fn typed_view_allocates_no_metadata_at_inline_rank() {
    let dynamic = TypedTensor::<f64>::from_vec_col_major(vec![3, 4], vec![0.0; 12]).unwrap();
    let ranked =
        TypedTensor::<f64, Rank<2>, Host>::from_host_vec_col_major([3, 4], vec![0.0; 12]).unwrap();
    assert_eq!(ranked.strides(), &[1, 3]);

    // Rank 2 is within strided-view's `INLINE_RANK`, so the view's metadata
    // stays inline and the per-operation helper allocates nothing.
    let allocations = count_allocations(|| {
        let view = typed_view("typed_view_allocates_no_metadata", &dynamic).unwrap();
        black_box(&view);
    });
    assert_eq!(
        allocations,
        0,
        "typed_view allocated {allocations} metadata allocations for shape {:?}",
        dynamic.shape(),
    );

    // The view must expose the tensor's canonical column-major layout.
    let view = typed_view("typed_view_allocates_no_metadata", &dynamic).unwrap();
    assert_eq!(view.dims(), &[3, 4]);
    assert_eq!(view.strides(), &[1, 3]);
    assert_eq!(view.offset(), 0);
}

#[test]
fn typed_view_above_inline_rank_spills_only_metadata() {
    // Rank 9 exceeds `INLINE_RANK`; the view still serves the tensor's layout
    // and only its two metadata vectors may allocate.
    let high_rank = TypedTensor::<f64>::from_vec_col_major(vec![1; 9], vec![0.0]).unwrap();
    let allocations = count_allocations(|| {
        let view = typed_view("typed_view_above_inline_rank", &high_rank).unwrap();
        assert_eq!(view.dims(), high_rank.shape());
        assert_eq!(view.strides(), high_rank.strides());
        black_box(&view);
    });
    assert!(
        allocations <= 3,
        "rank above INLINE_RANK spilled {allocations} allocations, expected at most 3"
    );
}

#[test]
fn stored_strides_are_the_canonical_column_major_layout() {
    let dynamic = TypedTensor::<f64>::from_vec_col_major(vec![3, 4], vec![0.0; 12]).unwrap();
    assert_eq!(dynamic.strides(), &[1, 3]);

    // Row-major input is reordered into column-major storage, so the stored
    // strides describe the storage, not the caller's argument order.
    let row_major = TypedTensor::<f64>::from_vec_row_major(vec![3, 4], vec![0.0; 12]).unwrap();
    assert_eq!(row_major.strides(), &[1, 3]);

    // Rank conversion, host/dynamic conversion and duplication all preserve it.
    let host =
        TypedTensor::<f64, Rank<2>, Host>::from_host_vec_col_major([3, 4], vec![0.0; 12]).unwrap();
    assert_eq!(host.strides(), &[1, 3]);
    assert_eq!(host.clone().strides(), &[1, 3]);
    let converted = host.into_dynamic();
    assert_eq!(converted.strides(), &[1, 3]);
    let back = converted.into_host().unwrap();
    assert_eq!(back.strides(), &[1, 3]);

    // A derived view serves the same layout.
    assert_eq!(back.as_view().strides(), &[1, 3]);
    assert_eq!(back.layout().strides(), &[1, 3]);
}
