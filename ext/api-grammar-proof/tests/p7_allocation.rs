//! Prototype 7, condition 2: steady-state allocation counts of the grammar on a borrowed engine
//! equal those of the `*_with_exec` entries.
//!
//! A separate test binary because it installs a global counting allocator (pattern copied from
//! `crates/tenferro-linalg/tests/cpu_linalg_allocation.rs`, `borrowed_entries_keep_their_steady_state_allocation_counts`).
//! The tests of this binary must not overlap: each holds [`SERIAL`].

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use num_complex::Complex64;
use tcontract_exec::Exec;
use tenferro_api_grammar_proof::p7_borrowed_engine::{
    BorrowedCpuEngine, BorrowedLinalg, CpuEngine, SvdOptions, SvdShape,
};
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_linalg::{
    eig_values_with_exec, eigh_values_with_exec, eigh_with_exec, qr_with_exec,
    svd_values_with_exec, svd_with_exec,
};
use tenferro_tensor::{Engine, TypedTensor};

struct CountingAllocator;
static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the caller's original layout and
// pointer; the counter only observes.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is the caller's.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ARMED.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer`/`layout` are the caller's.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: all three arguments are the caller's.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() && ARMED.load(Ordering::Relaxed) && new_size > layout.size() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        new_pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn measure(operation: impl FnOnce()) -> usize {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    operation();
    ARMED.store(false, Ordering::Relaxed);
    ALLOCATIONS.load(Ordering::Relaxed)
}

/// Fewest allocations over four rounds, so the pool is already warm.
fn steady(mut round: impl FnMut()) -> usize {
    (0..4).map(|_| measure(&mut round)).min().unwrap()
}

fn matrix(shape: &[usize]) -> TypedTensor<f64> {
    let len: usize = shape.iter().product();
    let data = (0..len)
        .map(|i| {
            let (r, c) = (i % shape[0], i / shape[0]);
            if r == c {
                shape[0] as f64 + 1.0
            } else {
                1.0 + r as f64 * 0.25 - c as f64 * 0.5
            }
        })
        .collect();
    TypedTensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn spd(n: usize) -> TypedTensor<f64> {
    let data = (0..n * n)
        .map(|i| {
            let (r, c) = (i % n, i / n);
            if r == c {
                n as f64 + 1.0
            } else {
                0.25 / (1.0 + r.abs_diff(c) as f64)
            }
        })
        .collect();
    TypedTensor::from_vec_col_major(vec![n, n], data).unwrap()
}

fn cmatrix(n: usize) -> TypedTensor<Complex64> {
    let data = (0..n * n)
        .map(|i| {
            Complex64::new(
                matrix(&[n, n]).host_data().unwrap()[i],
                0.25 - (i % 3) as f64 * 0.1,
            )
        })
        .collect();
    TypedTensor::from_vec_col_major(vec![n, n], data).unwrap()
}

#[test]
fn grammar_on_a_borrowed_engine_allocates_exactly_like_the_with_exec_entries() {
    let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let a = matrix(&[48, 48]);
    let h = spd(48);
    let c = cmatrix(32);
    let rank8 = TypedTensor::<f64>::from_vec_col_major(
        vec![4, 4, 1, 1, 1, 1, 1, 1],
        (0..16).map(|i| 1.0 + i as f64).collect(),
    )
    .unwrap();
    let (a, h, c, r8) = (a.as_view(), h.as_view(), c.as_view(), rank8.as_view());

    let mut host = CpuEngine::serial();
    let engine_pool_before = host.buffer_pool_stats();
    let mut free_pool = BufferPool::new();
    let mut grammar_pool = BufferPool::new();

    // (name, free-function round, grammar round)
    type Round<'a> = Box<dyn FnMut(&mut Exec<'_>, &mut BufferPool) + 'a>;
    let mut cases: Vec<(&str, Round, Round)> = Vec::new();
    macro_rules! case {
        ($name:expr, $free:expr, $grammar:expr) => {
            cases.push((
                $name,
                Box::new(|e: &mut Exec<'_>, b: &mut BufferPool| {
                    #[allow(clippy::redundant_closure_call)]
                    ($free)(e, b)
                }),
                Box::new(|e: &mut Exec<'_>, b: &mut BufferPool| {
                    let mut borrowed = BorrowedCpuEngine::new(e, b);
                    let engine: &mut dyn Engine = &mut borrowed;
                    #[allow(clippy::redundant_closure_call)]
                    ($grammar)(engine)
                }),
            ));
        };
    }
    case!(
        "svd/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(svd_with_exec(e, b, &a, false).unwrap()),
        |g: &mut dyn Engine| drop(a.svd(g).unwrap())
    );
    case!(
        "svd_full/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(svd_with_exec(e, b, &a, true).unwrap()),
        |g: &mut dyn Engine| drop(
            a.svd_with(&SvdOptions::default().shape(SvdShape::Full), g)
                .unwrap()
        )
    );
    case!(
        "svdvals/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(svd_values_with_exec(e, b, &a).unwrap()),
        |g: &mut dyn Engine| drop(a.svdvals(g).unwrap())
    );
    case!(
        "qr/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(qr_with_exec(e, b, &a).unwrap()),
        |g: &mut dyn Engine| drop(a.qr(g).unwrap())
    );
    case!(
        "eigh/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(eigh_with_exec(e, b, &h).unwrap()),
        |g: &mut dyn Engine| drop(h.eigh(g).unwrap())
    );
    case!(
        "eigvalsh/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(eigh_values_with_exec(e, b, &h).unwrap()),
        |g: &mut dyn Engine| drop(h.eigvalsh(g).unwrap())
    );
    case!(
        "eigvals/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(eig_values_with_exec(e, b, &a).unwrap()),
        |g: &mut dyn Engine| drop(a.eigvals(g).unwrap())
    );
    case!(
        "svd/complex64",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(svd_with_exec(e, b, &c, false).unwrap()),
        |g: &mut dyn Engine| drop(c.svd(g).unwrap())
    );
    case!(
        "qr/rank8",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(qr_with_exec(e, b, &r8).unwrap()),
        |g: &mut dyn Engine| drop(r8.qr(g).unwrap())
    );

    let mut report = Vec::new();
    for (name, mut free, mut grammar) in cases {
        let free_count = steady(|| {
            host.with_exec(|exec| free(exec, &mut free_pool)).unwrap();
        });
        let grammar_count = steady(|| {
            host.with_exec(|exec| grammar(exec, &mut grammar_pool))
                .unwrap();
        });
        println!(
            "p7 alloc {name}: with_exec {free_count}, grammar on borrowed engine {grammar_count}"
        );
        report.push((name, free_count, grammar_count));
    }
    for (name, free_count, grammar_count) in &report {
        assert_eq!(
            free_count, grammar_count,
            "{name}: grammar must allocate like with_exec"
        );
    }
    let rank8 = report.iter().find(|r| r.0 == "qr/rank8").unwrap();
    assert_eq!(rank8.2, 0, "a rank-8 operand must stay at zero allocations");
    assert!(report
        .iter()
        .filter(|r| r.0 != "svd/complex64")
        .all(|r| r.2 == 0 || r.1 == r.2));
    assert!(free_pool.stats().buffers > 0 && grammar_pool.stats().buffers > 0);
    assert_eq!(
        host.buffer_pool_stats(),
        engine_pool_before,
        "engine pool must not grow"
    );
}

/// Not asserted: the cost of the lane split itself (fork bookkeeping, result vector) on a serial
/// engine, to put a number on what `with_lanes` adds over a hand-built borrowed engine.
#[test]
fn with_lanes_overhead_is_reported() {
    let _guard = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let a = matrix(&[48, 48]);
    let a = a.as_view();
    let mut host = CpuEngine::serial();
    let mut pools = [BufferPool::new()];
    let lanes = steady(|| {
        let out = host
            .with_lanes(&[1], &mut pools, |_, e| drop(a.qr(e).unwrap()))
            .unwrap();
        drop(out);
    });
    let mut pool = BufferPool::new();
    let direct = steady(|| {
        host.with_exec(|exec| drop(a.qr(&mut BorrowedCpuEngine::new(exec, &mut pool)).unwrap()))
            .unwrap();
    });
    println!("p7 alloc qr/48 serial: with_lanes {lanes}, with_exec + BorrowedCpuEngine {direct}");
    assert!(lanes >= direct);
}
