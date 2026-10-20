//! The eager `dot_general` entry must reach the session's contraction-plan cache.
//!
//! #2139 item 1 and #2141 item 1: the eager entry passed no cache slot, so every
//! call rebuilt the whole numerical plan. Measured here with a counting global
//! allocator, because the plan build - not the kernel - is what a small `matmul`
//! pays for.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tenferro_cpu::CpuEngine;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct Allocator;

// SAFETY: every allocation is forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller's layout is forwarded verbatim.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` originate from `System`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if ENABLED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: the caller's pointer and requested size are forwarded verbatim.
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn allocations<T>(call: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATIONS.store(0, Ordering::SeqCst);
    ENABLED.store(true, Ordering::SeqCst);
    let out = call();
    ENABLED.store(false, Ordering::SeqCst);
    (out, ALLOCATIONS.load(Ordering::SeqCst))
}

#[test]
fn repeated_eager_matmul_does_not_rebuild_its_plan() {
    let lhs = TypedTensor::<f64>::from_vec_col_major(vec![64, 64], vec![1.0; 64 * 64]).unwrap();
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![64, 64], vec![2.0; 64 * 64]).unwrap();
    let mut engine = CpuEngine::serial();
    assert_eq!(engine.num_threads(), 1);
    let (out, first) = allocations(|| lhs.matmul(&rhs, &mut engine).unwrap());
    drop(out);
    // Warm the pooled output so only the plan build itself is measured.
    let (warm, _) = allocations(|| lhs.matmul(&rhs, &mut engine).unwrap());
    drop(warm);
    let (again, repeat) = allocations(|| lhs.matmul(&rhs, &mut engine).unwrap());
    drop(again);
    eprintln!("eager 64x64 matmul: first {first}, repeat {repeat} allocations");
    // Without a cache slot the eager path rebuilt the whole numerical plan on
    // every call: 84 allocations on this fixture, against 415 for the cold call
    // that also warms the provider. The repeat allocates nothing: the plan is
    // not rebuilt and the buffer pool keeps no per-operation bookkeeping.
    assert!(
        repeat == 0,
        "the repeated eager matmul still allocates: {first} then {repeat}"
    );
}
