//! Measures what removing the seven `Tensor` variants would cost at the erased layer.
//!
//! The measurement lives in its own test binary because the harness installs a process-wide counting
//! allocator: a second test in the same binary would interleave its own allocations with these numbers.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use tenferro_tensor::Tensor;

/// Counts every allocation the test process makes.
struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: the layout is forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the pointer and layout came from the system allocator through `alloc`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn measure(mut body: impl FnMut()) -> (usize, usize) {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    body();
    (
        ALLOCATIONS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

fn prepared() -> Option<tenferro_tensor::TypedTensor<f64>> {
    let tensor =
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("shape matches data");
    Some(tensor.into_typed::<f64>().expect("an f64 tensor"))
}

/// Measures what a single erased payload would cost, which removing the seven variants would need.
///
/// The seven variants hold the typed tensor inline, so wrapping one is a move. A payload that holds the
/// typed tensor behind a pointer would instead allocate per tensor, outside the pool the erased layer
/// accounts for. This records both numbers rather than asserting the difference from the type alone.
#[test]
fn report_erased_payload_allocation_cost() {
    let mut typed_inline = prepared();
    let mut typed_boxed = prepared();

    let (inline_allocs, inline_bytes) = measure(|| {
        let erased = Tensor::from_typed(std::hint::black_box(
            typed_inline.take().expect("moved once"),
        ));
        std::hint::black_box(&erased);
    });
    let (boxed_allocs, boxed_bytes) = measure(|| {
        let boxed: Box<dyn std::any::Any + Send + Sync> = Box::new(std::hint::black_box(
            typed_boxed.take().expect("moved once"),
        ));
        std::hint::black_box(&boxed);
    });

    println!(
        "erased wrapper size: {} bytes",
        std::mem::size_of::<Tensor>()
    );
    println!("inline variant payload: {inline_allocs} allocations / {inline_bytes} bytes");
    println!("boxed single payload: {boxed_allocs} allocations / {boxed_bytes} bytes");
    assert_eq!(
        inline_allocs, 0,
        "the variant holds the typed tensor inline, so wrapping it is a move"
    );
    assert!(
        boxed_allocs >= 1,
        "a boxed payload allocates outside the pool the erased layer accounts for"
    );
}
