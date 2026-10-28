//! The eager `dot_general` entry must reach the session's contraction-plan cache.
//!
//! #2139 item 1 and #2141 item 1: the eager entry passed no cache slot, so every
//! call rebuilt the whole numerical plan. Measured here with a counting global
//! allocator, because the plan build - not the kernel - is what a small `matmul`
//! pays for.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use tenferro_cpu::CpuEngine;
use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};

// Only the measuring thread is counted: another thread (a pool worker finishing its
// startup, the test harness) allocating during a measurement is not the op's cost, and
// counted globally it made these budgets flake (#2306). The engines measured here run the
// op on the calling thread.
thread_local! {
    static ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

struct Allocator;

// SAFETY: every allocation is forwarded unchanged to the system allocator.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.try_with(std::cell::Cell::get).unwrap_or(false) {
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
        if ENABLED.try_with(std::cell::Cell::get).unwrap_or(false) {
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
    ENABLED.with(|enabled| enabled.set(true));
    let out = call();
    ENABLED.with(|enabled| enabled.set(false));
    (out, ALLOCATIONS.load(Ordering::SeqCst))
}

/// The counting allocator is process-global, so measured regions run one at a
/// time; a panic in one region must not poison the other test.
static MEASURE_LOCK: Mutex<()> = Mutex::new(());

fn measure_lock() -> std::sync::MutexGuard<'static, ()> {
    MEASURE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
fn repeated_eager_matmul_does_not_rebuild_its_plan() {
    let _measure = measure_lock();
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

#[test]
fn repeated_tiny_typed_matmul_keeps_only_the_one_time_plan_build() {
    let _measure = measure_lock();
    let lhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0; 4]).unwrap();
    let rhs = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0; 4]).unwrap();
    let mut engine = CpuEngine::serial();

    // Warm the provider's one-time builder state under a different key, so the
    // measured first call isolates the 2x2 plan build instead of the cold
    // process's provider initialisation.
    let warmup = TypedTensor::<f64>::from_vec_col_major(vec![3, 3], vec![1.0; 9]).unwrap();
    let _ = allocations(|| warmup.matmul(&warmup, &mut engine).unwrap());

    let (out, first) = allocations(|| lhs.matmul(&rhs, &mut engine).unwrap());
    assert_eq!(out.as_slice().unwrap(), &[4.0; 4]);
    drop(out);
    // Warm the pooled output so only a plan build could show up.
    let (warm, warm_allocations) = allocations(|| lhs.matmul(&rhs, &mut engine).unwrap());
    drop(warm);
    let (again, repeat) = allocations(|| lhs.matmul(&rhs, &mut engine).unwrap());
    assert_eq!(again.as_slice().unwrap(), &[4.0; 4]);
    drop(again);
    eprintln!(
        "eager 2x2 matmul (warm provider): first {first}, warm {warm_allocations}, \
         repeat {repeat} allocations"
    );
    // #2139 item 1: the 88 allocations the umbrella inventory records for one
    // 2x2 typed matmul are the first call's one-time binary plan build, not a
    // per-call cost. Once that plan is cached, the warm and repeated calls
    // allocate nothing: no tenferro metadata `Vec` and no provider plan remain
    // on the per-call path.
    assert_eq!(
        warm_allocations, 0,
        "the warm 2x2 matmul still allocates: {first} then {warm_allocations}"
    );
    assert_eq!(
        repeat, 0,
        "the repeated 2x2 matmul still allocates: {first} then {repeat}"
    );
}
