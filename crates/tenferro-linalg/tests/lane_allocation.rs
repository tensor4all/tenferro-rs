//! Steady-state allocation counts of the factorization entries inside a lane of
//! `CpuEngine::with_lanes` (#2552 S2b).
//!
//! The contract: a lane draws every output, the provider workspace and a complex input's
//! transient spectrum from the lane's own pool, so after warm-up a call makes no host-side
//! allocation, exactly like the `*_with_exec` entries it replaces. The counter is armed inside
//! the lane closure, so the one-off bookkeeping of `with_lanes` itself is not part of a round.
//!
//! This target owns the process allocator, so it must stay a separate test binary. Pools are
//! reused across rounds: a cold first call legitimately allocates, and zero is a property of
//! reusing the same pool. A negative-stride operand is gathered into a pool buffer and is not
//! covered by the zero promise.

#![cfg(feature = "native")]
// The placement ledger allocates per token, so allocation counts are taken without it.
#![cfg(not(feature = "ledger"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use num_complex::Complex64;
use tcontract_exec::Exec;
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_cpu::CpuEngine;
use tenferro_linalg::{
    eig_values_with_exec, eigh_values_with_exec, eigh_with_exec, qr_with_exec,
    svd_values_with_exec, svd_with_exec, TensorReadLinalgExt,
};
use tenferro_tensor::{Engine, TensorRead, TensorView, TypedTensor};

/// Counts allocations while armed.
struct CountingAllocator;

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the caller's original layout and
// pointer; the counter only observes.
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
        if !new_pointer.is_null() && ARMED.load(Ordering::Relaxed) && new_size > layout.size() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        }
        new_pointer
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// The counting allocator sees every thread, so the tests of this binary must not overlap.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial_test() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn measure(operation: impl FnOnce()) -> usize {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    operation();
    ARMED.store(false, Ordering::Relaxed);
    ALLOCATIONS.load(Ordering::Relaxed)
}

/// Fewest allocations over four rounds, so the pool is already warm.
fn steady(mut round: impl FnMut()) -> usize {
    (0..4)
        .map(|_| measure(&mut round))
        .min()
        .unwrap_or(usize::MAX)
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
    TypedTensor::from_vec_col_major(shape.to_vec(), data).expect("matrix")
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
    TypedTensor::from_vec_col_major(vec![n, n], data).expect("spd")
}

fn cmatrix(n: usize) -> TypedTensor<Complex64> {
    let real = matrix(&[n, n]);
    let data = (0..n * n)
        .map(|i| {
            Complex64::new(
                real.host_data().expect("host data")[i],
                0.25 - (i % 3) as f64 * 0.1,
            )
        })
        .collect();
    TypedTensor::from_vec_col_major(vec![n, n], data).expect("cmatrix")
}

type FreeRound<'a> = Box<dyn Fn(&mut Exec<'_>, &mut BufferPool) + 'a>;
type LaneRound<'a> = Box<dyn Fn(&mut dyn Engine) + Sync + 'a>;

#[test]
fn lane_entries_allocate_exactly_like_the_with_exec_entries() {
    let _guard = serial_test();
    let a = matrix(&[48, 48]);
    let h = spd(48);
    let c = cmatrix(32);
    let rank8 = TypedTensor::<f64>::from_vec_col_major(
        vec![4, 4, 1, 1, 1, 1, 1, 1],
        (0..16).map(|i| 1.0 + i as f64).collect(),
    )
    .expect("rank 8");
    let (av, hv, cv, r8v) = (a.as_view(), h.as_view(), c.as_view(), rank8.as_view());
    let read_a = TensorRead::from_view(TensorView::F64(av.clone()));
    let read_h = TensorRead::from_view(TensorView::F64(hv.clone()));
    let read_c = TensorRead::from_view(TensorView::C64(cv.clone()));
    let read_r8 = TensorRead::from_view(TensorView::F64(r8v.clone()));
    let mut host = CpuEngine::serial();
    let engine_pool_before = host.buffer_pool_stats();
    let mut free_pool = BufferPool::new();
    let mut lane_pools = [BufferPool::new()];

    let mut cases: Vec<(&str, FreeRound, LaneRound)> = Vec::new();
    macro_rules! case {
        ($name:expr, $free:expr, $lane:expr) => {
            cases.push(($name, Box::new($free), Box::new($lane)));
        };
    }
    case!(
        "svd/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(svd_with_exec(e, b, &av, false).expect("svd")),
        |l: &mut dyn Engine| drop(read_a.clone().svd_read(l).expect("svd"))
    );
    case!(
        "svd_full/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(svd_with_exec(e, b, &av, true).expect("svd")),
        |l: &mut dyn Engine| drop(read_a.clone().svd_full_read(l).expect("svd_full"))
    );
    case!(
        "svdvals/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(
            svd_values_with_exec(e, b, &av).expect("svdvals")
        ),
        |l: &mut dyn Engine| drop(read_a.clone().svdvals_read(l).expect("svdvals"))
    );
    case!(
        "qr/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(qr_with_exec(e, b, &av).expect("qr")),
        |l: &mut dyn Engine| drop(read_a.clone().qr_read(l).expect("qr"))
    );
    case!(
        "eigh/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(eigh_with_exec(e, b, &hv).expect("eigh")),
        |l: &mut dyn Engine| drop(read_h.clone().eigh_read(l).expect("eigh"))
    );
    case!(
        "eigvalsh/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(
            eigh_values_with_exec(e, b, &hv).expect("eigvalsh")
        ),
        |l: &mut dyn Engine| drop(read_h.clone().eigvalsh_read(l).expect("eigvalsh"))
    );
    case!(
        "eigvals/48",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(
            eig_values_with_exec(e, b, &av).expect("eigvals")
        ),
        |l: &mut dyn Engine| drop(read_a.clone().eigvals_read(l).expect("eigvals"))
    );
    case!(
        "svd/complex64",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(svd_with_exec(e, b, &cv, false).expect("svd")),
        |l: &mut dyn Engine| drop(read_c.clone().svd_read(l).expect("svd"))
    );
    case!(
        "qr/rank8",
        |e: &mut Exec<'_>, b: &mut BufferPool| drop(qr_with_exec(e, b, &r8v).expect("qr")),
        |l: &mut dyn Engine| drop(read_r8.clone().qr_read(l).expect("qr"))
    );

    let mut report = Vec::new();
    for (name, free, lane) in &cases {
        let free_count = host
            .with_exec(|exec| steady(|| free(exec, &mut free_pool)))
            .expect("with_exec");
        let lane_count = host
            .with_lanes(&[1], &mut lane_pools, |_, engine| {
                steady(|| lane(&mut *engine))
            })
            .expect("with_lanes")[0];
        println!("lane alloc {name}: with_exec {free_count}, lane {lane_count}");
        report.push((*name, free_count, lane_count));
    }
    for (name, free_count, lane_count) in &report {
        assert_eq!(
            *lane_count, 0,
            "{name}: a lane call must make no host allocation in steady state"
        );
        assert_eq!(
            free_count, lane_count,
            "{name}: a lane must allocate like with_exec"
        );
    }
    assert!(free_pool.stats().buffers > 0 && lane_pools[0].stats().buffers > 0);
    assert_eq!(
        host.buffer_pool_stats(),
        engine_pool_before,
        "the engine's own pool must not grow"
    );
}

/// The rank boundary of the storage guarantee (#2505): a rank-8 operand keeps its metadata
/// inline and measures zero; a rank-9 operand spills dims and strides and does not.
#[test]
fn rank_nine_spills_metadata_in_a_lane_as_it_does_for_with_exec() {
    let _guard = serial_test();
    let block: Vec<f64> = (0..16).map(|index| 1.0 + index as f64).collect();
    let rank9 = TypedTensor::<f64>::from_vec_col_major(vec![4, 4, 1, 1, 1, 1, 1, 1, 1], block)
        .expect("rank 9");
    let read = TensorRead::from_view(TensorView::F64(rank9.as_view()));
    let mut host = CpuEngine::serial();
    let mut pools = [BufferPool::new()];
    let spill = host
        .with_lanes(&[1], &mut pools, |_, engine| {
            steady(|| drop(read.clone().qr_read(&mut *engine).expect("qr")))
        })
        .expect("with_lanes")[0];
    println!("lane alloc qr/rank9: {spill}");
    assert!(
        spill > 0,
        "a rank-9 operand spilled nothing: the documented metadata spill above INLINE_RANK \
         no longer happens"
    );
}
