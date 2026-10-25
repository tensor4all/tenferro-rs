//! Steady-state Rust heap allocations for a two-operation fused region (#2178).
//!
//! `tenferro-cpu-fused` still builds several small metadata `Vec`s per region:
//! the validated input layouts, their erased descriptors, the destination
//! descriptors, and the output list. They are bounded by the region's input and
//! output counts rather than by the element count, so this target pins the
//! measured steady-state count instead of claiming the whole path is
//! allocation-free. Only Rust heap allocations are counted; provider C
//! allocations are out of scope.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use strided_basic::Exec;
use tenferro_cpu_basic::BufferPool;
use tenferro_cpu_fused::elementwise_fusion_with_pool;
use tenferro_tensor::backend::{ElementwiseFusionInst, ElementwiseFusionOp, ElementwiseFusionPlan};
use tenferro_tensor::{DType, Tensor, TypedTensor};

struct CountingAllocator;

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator unchanged; the counter only observes.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is the caller's, forwarded unchanged.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer`/`layout` are the caller's, forwarded unchanged.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: all three arguments are the caller's, forwarded unchanged.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() && ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        new_pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measure(operation: impl FnOnce()) -> usize {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    operation();
    ARMED.store(false, Ordering::Relaxed);
    ALLOCATIONS.load(Ordering::Relaxed)
}

/// A two-operation region above the fusion floor: `(lhs + rhs) * rhs`.
#[test]
fn two_operation_fused_region_has_a_pinned_steady_state_allocation_ceiling() {
    let len = 16 * 1024;
    let lhs = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![len], (0..len).map(|i| i as f64).collect()).unwrap(),
    );
    let rhs = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![len], (0..len).map(|i| i as f64 + 1.0).collect())
            .unwrap(),
    );
    let plan = ElementwiseFusionPlan::new(
        DType::F64,
        2,
        vec![3],
        vec![
            ElementwiseFusionInst::new(ElementwiseFusionOp::Add, vec![0, 1]),
            ElementwiseFusionInst::new(ElementwiseFusionOp::Multiply, vec![2, 1]),
        ],
    );
    let mut buffers = BufferPool::default();
    let run = |buffers: &mut BufferPool| {
        let outputs = elementwise_fusion_with_pool(buffers, &[&lhs, &rhs], &plan, Exec::serial())
            .expect("fusion");
        let outputs = outputs.expect("the two-operation region must fuse");
        assert_eq!(outputs[0].as_slice::<f64>().unwrap()[7], (7.0 + 8.0) * 8.0);
    };
    for _ in 0..8 {
        run(&mut buffers);
    }
    let allocations = measure(|| run(&mut buffers));
    eprintln!("two-operation fused region: {allocations} allocations");
    // Measured steady state: six bounded metadata Vecs, one each for the input
    // layouts, erased input descriptors, input pointers, pooled outputs,
    // destination descriptors and returned tensors. They are not `Default`
    // descriptor types, so an inline `TinyVec` would need a fake `Default`;
    // none of them scales with `len`.
    assert_eq!(
        allocations, 6,
        "two-operation fused region made {allocations} heap allocations"
    );
}
