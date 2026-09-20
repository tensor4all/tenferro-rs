//! Allocation accounting for the CPU LAPACK provider.
//!
//! LAPACK factors `A` in place and needs vendor scratch, so some copies and
//! buffers are unavoidable. What is avoidable is paying the *allocator* for
//! them on every call: the kernels already receive the session `BufferPool`,
//! so their scratch should be pooled and returned like every other CPU linalg
//! buffer.
//!
//! This target owns the process allocator, so it must stay a separate test
//! binary. It reports steady-state counts — the pool is primed by warm-up calls
//! first — because a cold first call legitimately allocates.

#![cfg(feature = "cpu-blas")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use num_complex::Complex64;
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_linalg::TensorLinalgExt;
use tenferro_tensor::{BackendSessionHost, Tensor, TypedTensor};

/// Counts allocations and live bytes while armed.
///
/// Only the measured window is armed, so harness and formatting allocations do
/// not pollute the numbers.
struct CountingAllocator;

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the caller's
// original layout and pointer; the counters only observe.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is the caller's, forwarded unchanged.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ARMED.load(Ordering::Relaxed) {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if ARMED.load(Ordering::Relaxed) {
            LIVE_BYTES.fetch_sub(
                layout.size().min(LIVE_BYTES.load(Ordering::Relaxed)),
                Ordering::Relaxed,
            );
        }
        // SAFETY: `pointer`/`layout` are the caller's, forwarded unchanged.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: all three arguments are the caller's, forwarded unchanged.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() && ARMED.load(Ordering::Relaxed) && new_size > layout.size() {
            record_allocation(new_size - layout.size());
        }
        new_pointer
    }
}

fn record_allocation(size: usize) {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    ALLOCATED_BYTES.fetch_add(size, Ordering::Relaxed);
    let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AllocationReport {
    allocations: usize,
    allocated_bytes: usize,
    peak_live_bytes: usize,
}

/// Measure one call with the counters armed.
///
/// Single-threaded by construction: the backends below are built with one
/// thread, and the counters are process-wide.
fn measure(operation: impl FnOnce()) -> AllocationReport {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    LIVE_BYTES.store(0, Ordering::Relaxed);
    PEAK_BYTES.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    operation();
    ARMED.store(false, Ordering::Relaxed);
    AllocationReport {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        peak_live_bytes: PEAK_BYTES.load(Ordering::Relaxed),
    }
}

fn blas_backend() -> CpuBackend {
    CpuBackend::with_threads_and_kind(1, CpuBackendKind::Blas).expect("BLAS CPU backend")
}

fn sample_real(m: usize, n: usize) -> Vec<f64> {
    (0..m * n)
        .map(|index| {
            let row = (index % m) as f64;
            let col = (index / m) as f64;
            1.0 + row * 0.5 - col * 0.25 + if row == col { m as f64 } else { 0.0 }
        })
        .collect()
}

fn f64_matrix(m: usize, n: usize) -> Tensor {
    Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![m, n], sample_real(m, n)).unwrap(),
    )
}

fn c64_matrix(n: usize) -> Tensor {
    let data = sample_real(n, n)
        .into_iter()
        .enumerate()
        .map(|(index, value)| Complex64::new(value, 0.125 * (index % 7) as f64))
        .collect();
    Tensor::from_typed::<Complex64>(TypedTensor::from_vec_col_major(vec![n, n], data).unwrap())
}

/// Run `operation` until the buffer pool is primed, then report the lightest of
/// several measured calls.
///
/// The counters are process-wide, so a BLAS worker thread allocating during a
/// measurement window would inflate that sample. Taking the minimum across
/// repeats removes that additive noise without weakening the ceiling: the
/// quiet sample is the one the kernels actually cost.
fn steady_state(
    host: &mut CpuBackend,
    operation: impl Fn(&mut dyn tenferro_tensor::BackendSession) + Send + Sync,
) -> AllocationReport {
    for _ in 0..4 {
        host.with_backend_session(|session| operation(session));
    }
    let mut best: Option<AllocationReport> = None;
    for _ in 0..5 {
        let report = host.with_backend_session(|session| measure(|| operation(session)));
        if best.is_none_or(|best| report.allocations < best.allocations) {
            best = Some(report);
        }
    }
    best.unwrap_or_default()
}

/// Steady-state allocation ceiling per measured case.
///
/// These are the counts this change achieves, not aspirations. They exist to
/// stop the LAPACK kernels from quietly reacquiring per-call scratch again.
/// Byte totals are printed rather than asserted: `lwork` is chosen by the
/// linked LAPACK, so the exact sizes are implementation-specific while the
/// *number* of allocations is structural.
const ALLOCATION_CEILINGS: &[(&str, usize)] = &[
    ("svd/square", 10),
    ("svdvals/square", 5),
    ("qr/square", 8),
    ("svd/tall", 10),
    ("svdvals/tall", 5),
    ("qr/tall", 8),
    ("svd/complex", 14),
    ("svdvals/complex", 6),
    ("qr/complex", 8),
    ("eigh/square", 9),
    ("lu/square", 12),
    ("eigh/complex", 11),
    ("lu/complex", 12),
];

fn ceiling(name: &str) -> usize {
    ALLOCATION_CEILINGS
        .iter()
        .find(|(case, _)| *case == name)
        .map(|(_, ceiling)| *ceiling)
        .unwrap_or_else(|| panic!("no recorded allocation ceiling for {name}"))
}

#[test]
fn lapack_kernels_take_their_scratch_from_the_session_buffer_pool() {
    let mut host = blas_backend();
    let square = f64_matrix(48, 48);
    let tall = f64_matrix(64, 24);
    let complex = c64_matrix(32);
    let mut failures = Vec::new();

    macro_rules! check {
        ($name:expr, $input:expr, $call:expr) => {{
            let name = $name;
            let input = $input;
            let report = steady_state(&mut host, |session| {
                #[allow(clippy::redundant_closure_call)]
                $call(input, session);
            });
            eprintln!(
                "{name}: {} allocations, {} bytes, peak {} live bytes",
                report.allocations, report.allocated_bytes, report.peak_live_bytes
            );
            let ceiling = ceiling(&name);
            if report.allocations > ceiling {
                failures.push(format!(
                    "{name}: {} allocations exceeds the recorded ceiling of {ceiling} \
                     ({} bytes, peak {} live bytes)",
                    report.allocations, report.allocated_bytes, report.peak_live_bytes
                ));
            }
        }};
    }

    for (shape, input) in [("square", &square), ("tall", &tall), ("complex", &complex)] {
        check!(
            format!("svd/{shape}"),
            input,
            |input: &Tensor, session: &mut dyn tenferro_tensor::BackendSession| {
                input.svd(session).unwrap();
            }
        );
        check!(
            format!("svdvals/{shape}"),
            input,
            |input: &Tensor, session: &mut dyn tenferro_tensor::BackendSession| {
                input.svdvals(session).unwrap();
            }
        );
        check!(
            format!("qr/{shape}"),
            input,
            |input: &Tensor, session: &mut dyn tenferro_tensor::BackendSession| {
                input.qr(session).unwrap();
            }
        );
    }
    // `eigh` needs a Hermitian input and `lu` a square one.
    for (shape, input) in [("square", &square), ("complex", &complex)] {
        check!(
            format!("eigh/{shape}"),
            input,
            |input: &Tensor, session: &mut dyn tenferro_tensor::BackendSession| {
                input.eigh(session).unwrap();
            }
        );
        check!(
            format!("lu/{shape}"),
            input,
            |input: &Tensor, session: &mut dyn tenferro_tensor::BackendSession| {
                input.lu(session).unwrap();
            }
        );
    }

    assert!(
        failures.is_empty(),
        "LAPACK kernels allocate more per call than the recorded ceilings:\n{}",
        failures.join("\n")
    );
}
