//! The contraction plan cache's memory against the allocator (#2742).
//!
//! A cached plan owns tcontract's scatter tables, hundreds of MiB per plan on a
//! large free group. These tests apply a two-qubit gate to an `n`-qubit state at
//! several positions (the issue's reproducer, at a size CI can afford) and compare
//! what the cache reports with what the allocator holds.
//!
//! A test binary of its own: the counting global allocator sees every allocation
//! of the process, so the tests run one at a time under `SERIAL`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use num_complex::Complex32;
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_einsum::{EinsumOps, EinsumSubscripts};
use tenferro_tensor::Tensor;

struct Counting;

/// Bytes allocated and not yet freed.
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static SERIAL: Mutex<()> = Mutex::new(());

// SAFETY: forwards every call to `System` unchanged and only counts sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            CURRENT.fetch_add(layout.size(), Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` with this `layout`.
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn held() -> usize {
    CURRENT.load(Ordering::SeqCst)
}

/// The relative error under which `retained_bytes` must match the allocator.
///
/// The count is tcontract's own report of what a plan owns; the allocator sees
/// that plus the `Arc` and slot bookkeeping, which is under 1 KiB per plan
/// against tens of MiB of tables.
const TOLERANCE: f64 = 0.02;

/// Allocator noise that is not plan memory: lazily built engine state.
const SLACK: usize = 1 << 20;

const QUBITS: usize = 22;
/// Positions with distinct contraction axes, so each builds its own plan.
const PAIRS: [(u32, u32); 4] = [(0, 11), (3, 15), (6, 19), (9, 21)];

struct Rig {
    state: Tensor,
    gate: Tensor,
    engine: CpuEngine,
    labels: Vec<u32>,
}

impl Rig {
    fn new() -> Self {
        let state = Tensor::from_vec_col_major(
            vec![2; QUBITS],
            (0..1usize << QUBITS)
                .map(|i| Complex32::new((i % 7) as f32, (i % 3) as f32))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let gate = Tensor::from_vec_col_major(
            vec![2; 4],
            (0..16)
                .map(|i| Complex32::new(i as f32, 0.0))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut engine = CpuEngine::new(Pool::builder().threads(2).build().unwrap());
        // Recycled outputs would otherwise count as held bytes.
        engine.set_buffer_pool_limit_bytes(0);
        Self {
            state,
            gate,
            engine,
            labels: (0..QUBITS as u32).collect(),
        }
    }

    /// out[.., p', .., q', ..] = sum_{p,q} G[p', q', p, q] state[.., p, .., q, ..]; dropped.
    fn apply(&mut self, (p, q): (u32, u32)) {
        let out: Vec<u32> = self
            .labels
            .iter()
            .map(|&l| match l {
                l if l == p => 1000,
                l if l == q => 1001,
                l => l,
            })
            .collect();
        let subs = EinsumSubscripts::new(&[&self.labels[..], &[1000, 1001, p, q][..]], &out);
        drop([&self.state, &self.gate].einsum(&subs, &mut self.engine).unwrap());
    }

    fn retained(&self) -> usize {
        self.engine.plan_cache().stats().retained_bytes
    }
}

fn within(reported: usize, measured: usize, what: &str) {
    let error = (reported as f64 - measured as f64).abs() / measured as f64;
    println!("{what}: reported {reported} B, allocator {measured} B, error {error:.5}");
    assert!(
        error <= TOLERANCE,
        "{what}: reported {reported} B vs allocator {measured} B (error {error:.4} > {TOLERANCE})"
    );
}

#[test]
fn retained_bytes_tracks_the_allocator_per_plan_and_after_clear() {
    let _serial = SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut rig = Rig::new();
    let base = held();
    let mut last_retained = rig.retained();
    let mut last_held = held();
    for pair in PAIRS {
        rig.apply(pair);
        let (retained, now) = (rig.retained(), held());
        // The tables of the new plan: a third of 3 * 2^(n-2) * 8 B at least.
        assert!(retained - last_retained > (3usize << (QUBITS - 2)) * 8 * 9 / 10);
        within(retained - last_retained, now - last_held, "new plan");
        (last_retained, last_held) = (retained, now);
    }
    within(rig.retained(), held() - base, "all plans");

    // A repeat adds nothing.
    rig.apply(PAIRS[0]);
    assert_eq!(rig.retained(), last_retained);

    let before_clear = held();
    let reported = rig.retained();
    rig.engine.plan_cache().clear();
    assert_eq!(rig.retained(), 0);
    within(reported, before_clear - held(), "freed by clear()");
    assert!(held() - base < SLACK, "held {} B above the base", held() - base);
}

#[test]
fn reset_buffer_pool_keeps_plans_and_the_report_says_so() {
    let _serial = SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut rig = Rig::new();
    rig.apply(PAIRS[0]);
    let retained = rig.retained();
    let before = held();
    rig.engine.reset_buffer_pool();
    assert_eq!(rig.retained(), retained);
    assert!(before - held() < SLACK, "reset freed {} B", before - held());
    assert!(rig.engine.runtime_cache_stats().retained_bytes >= retained);
    rig.engine.clear_runtime_caches();
    assert_eq!(rig.retained(), 0);
    assert!(held() < before - retained * 98 / 100);
}

#[test]
fn the_byte_bound_evicts_least_recently_used_and_releases_their_tables() {
    let _serial = SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut rig = Rig::new();
    let base = held();
    rig.apply(PAIRS[0]);
    let one = rig.retained();
    // Room for two plans and a half.
    let limit = one * 5 / 2;
    rig.engine.plan_cache().set_contraction_limit_bytes(limit);
    assert_eq!(rig.engine.plan_cache().contraction_limit_bytes(), limit);
    rig.apply(PAIRS[1]);
    rig.apply(PAIRS[0]); // PAIRS[1] is now the least recently used.
    assert_eq!(rig.engine.plan_cache().stats().evictions, 0);
    let two = held() - base;
    rig.apply(PAIRS[2]);
    let stats = rig.engine.plan_cache().stats();
    assert_eq!(stats.evictions, 1, "the third plan overflows the bound once");
    assert_eq!(stats.entries, 2);
    assert!(stats.retained_bytes <= limit);
    // The evicted plan's tables are gone from the heap, not only from the count.
    within(stats.retained_bytes, held() - base, "after eviction");
    assert!(held() - base < two + one / 4, "heap grew by a plan: {two} -> {}", held() - base);

    // Recency, not insertion order, chose the victim: PAIRS[0] is still cached
    // (a hit), PAIRS[1] is not (a miss).
    let hits = rig.engine.plan_cache().stats().hits;
    rig.apply(PAIRS[0]);
    assert_eq!(rig.engine.plan_cache().stats().hits, hits + 1);
    let misses = rig.engine.plan_cache().stats().misses;
    rig.apply(PAIRS[1]);
    assert_eq!(rig.engine.plan_cache().stats().misses, misses + 1);

    // Lowering the bound evicts down to it at once.
    rig.engine.plan_cache().set_contraction_limit_bytes(one * 3 / 2);
    let stats = rig.engine.plan_cache().stats();
    assert_eq!(stats.entries, 1);
    within(stats.retained_bytes, held() - base, "after lowering the bound");
}

#[test]
fn a_plan_over_the_bound_is_used_and_not_kept() {
    let _serial = SERIAL.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut rig = Rig::new();
    rig.engine.plan_cache().set_contraction_limit_bytes(1 << 20);
    let base = held();
    rig.apply(PAIRS[0]);
    let stats = rig.engine.plan_cache().stats();
    assert_eq!((stats.entries, stats.evictions), (0, 0));
    // The plan's tables (24 MiB) are freed with the call; what stays is lazily
    // built engine state, a few KiB.
    assert!(held() - base < SLACK, "held {} B above the base", held() - base);
    // Every call rebuilds: a miss each time.
    rig.apply(PAIRS[0]);
    assert_eq!(rig.engine.plan_cache().stats().hits, 0);
}
