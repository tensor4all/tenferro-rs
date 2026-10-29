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

// The warm-slot-miss gates below hold for the native provider only: a miss rebuilds
// the whole lower plan, and `cpueinsum-blas` planning allocates its own vendor
// metadata on every build, so under `blas` only the `Arc` (#2314) is saved.
#[cfg(feature = "native")]
/// A GEMM shape as `(m, k, n)`.
type Shape = (usize, usize, usize);

#[cfg(feature = "native")]
/// A left operand with no repeated value, so a wrong row or column shows up in
/// the value check instead of hiding behind a constant.
fn lhs(m: usize, k: usize) -> TypedTensor<f64> {
    let mut data = vec![0.0; m * k];
    for j in 0..k {
        for i in 0..m {
            data[i + m * j] = 1.0 + i as f64 - 0.5 * j as f64;
        }
    }
    TypedTensor::<f64>::from_vec_col_major(vec![m, k], data).unwrap()
}

#[cfg(feature = "native")]
fn rhs(k: usize, n: usize) -> TypedTensor<f64> {
    let mut data = vec![0.0; k * n];
    for l in 0..n {
        for j in 0..k {
            data[j + k * l] = 0.25 + 0.25 * j as f64 + l as f64;
        }
    }
    TypedTensor::<f64>::from_vec_col_major(vec![k, n], data).unwrap()
}

#[cfg(feature = "native")]
/// The column-major product of [`lhs`] and [`rhs`] for `shape`.
fn expected((m, k, n): Shape) -> Vec<f64> {
    let lhs = lhs(m, k);
    let rhs = rhs(k, n);
    let a = lhs.as_slice().unwrap();
    let b = rhs.as_slice().unwrap();
    let mut out = vec![0.0; m * n];
    for l in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for j in 0..k {
                sum += a[i + m * j] * b[j + k * l];
            }
            out[i + m * l] = sum;
        }
    }
    out
}

#[cfg(feature = "native")]
fn max_abs_error(out: &TypedTensor<f64>, expected: &[f64]) -> f64 {
    let actual = out.as_slice().unwrap();
    assert_eq!(actual.len(), expected.len(), "output length");
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            // `f64::max` drops NaN, so a non-finite output must fail here.
            assert!(actual.is_finite(), "non-finite output {actual}");
            (actual - expected).abs()
        })
        .fold(0.0, f64::max)
}

#[cfg(feature = "native")]
/// Send `shapes` through one cache slot, warm it with two rounds, then measure
/// one round: the round must allocate nothing while adding one miss per call.
fn assert_warm_round_is_allocation_free(label: &str, shapes: &[Shape], expected_misses: u64) {
    let _measure = measure_lock();
    let mut engine = CpuEngine::serial();
    // Every eager slot is a geometry hash modulo the capacity, so a capacity of
    // one forces every call into slot 0 and every call is a miss on a warm slot.
    engine.plan_cache().set_contraction_capacity(1);
    let operands: Vec<(TypedTensor<f64>, TypedTensor<f64>, Vec<f64>)> = shapes
        .iter()
        .map(|&(m, k, n)| (lhs(m, k), rhs(k, n), expected((m, k, n))))
        .collect();

    for _round in 0..2 {
        for (lhs, rhs, expected) in &operands {
            let out = lhs.matmul(rhs, &mut engine).unwrap();
            assert!(
                max_abs_error(&out, expected) < 1e-9,
                "{label}: warm value check failed"
            );
            drop(out);
        }
    }

    let before = engine.plan_cache().contraction_stats();
    let (worst, measured) = allocations(|| {
        let mut worst = 0.0f64;
        for (lhs, rhs, expected) in &operands {
            let out = lhs.matmul(rhs, &mut engine).unwrap();
            worst = worst.max(max_abs_error(&out, expected));
            // Drop inside the measured region: the pooled output buffer returns
            // to the engine's pool here, so the next call reuses it.
            drop(out);
        }
        worst
    });
    let after = engine.plan_cache().contraction_stats();

    assert!(
        worst < 1e-9,
        "{label}: measured value check failed: {worst}"
    );
    assert_eq!(
        after.misses - before.misses,
        expected_misses,
        "{label}: a capacity-1 slot misses on every call"
    );
    assert_eq!(
        after.hits - before.hits,
        0,
        "{label}: a capacity-1 slot never hits"
    );
    // #2314: the miss rebuilt the plan in the slot's own storage instead of
    // allocating a new `Arc` (one per call before the fix).
    assert_eq!(
        measured,
        0,
        "{label}: the warm round still allocates {measured} times over {} calls",
        shapes.len()
    );
}

#[cfg(feature = "native")]
#[test]
fn alternating_shapes_through_one_slot_reuse_their_plan_storage() {
    // Three alternating GEMM shapes: before #2314 this measured three
    // allocations per round, one new `Arc` per miss.
    assert_warm_round_is_allocation_free(
        "three alternating shapes",
        &[(4, 5, 6), (7, 3, 2), (6, 6, 6)],
        3,
    );
}

#[cfg(feature = "native")]
#[test]
fn a_su2_permute_block_sequence_reuses_its_plan_storage() {
    // The per-block recoupling GEMMs a symmetric-tensor permute issues: distinct
    // shapes, including a single contracted column and a non-square block, in a
    // fixed order through one slot. Before #2314 this measured eight allocations
    // per round.
    assert_warm_round_is_allocation_free(
        "eight permute blocks",
        &[
            (3, 1, 4),
            (5, 7, 2),
            (1, 1, 1),
            (12, 3, 9),
            (2, 4, 1),
            (6, 11, 5),
            (4, 2, 12),
            (9, 6, 3),
        ],
        8,
    );
}
