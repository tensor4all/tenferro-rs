//! Small batched QR / SVD / packed-LU provider-scratch ceilings (#2178 item 7).
//!
//! This target owns the process allocator, so it must stay a separate test
//! binary. Counts are steady state: the session buffer pool is primed by
//! warm-up calls first. Only Rust heap allocations are counted; the LAPACK/BLAS
//! C scratch (and any vendor-internal allocation) is outside this counter and
//! out of scope here.
//!
//! The provider hoists its workspace out of the batch loop, so the steady-state
//! count must not grow with the batch size. Each family therefore pins one
//! ceiling and asserts batch 1 and batch 64 use the same number of allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::{BackendSession, BackendSessionHost, Tensor, TypedTensor};

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

/// A column-major `n x n x batch` stack of diagonal-dominant matrices.
fn batched_matrix(n: usize, batch: usize) -> Tensor {
    let mut data = vec![0.0_f64; n * n * batch];
    for b in 0..batch {
        for col in 0..n {
            for row in 0..n {
                let index = row + col * n + b * n * n;
                data[index] = if row == col {
                    (n + 1) as f64
                } else {
                    0.25 / (1.0 + row.abs_diff(col) as f64)
                };
            }
        }
    }
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![n, n, batch], data).unwrap())
}

fn with_cpu<T>(
    session: &mut dyn BackendSession,
    operation: impl FnOnce(&mut tenferro_cpu::CpuExecSession<'_>) -> T,
) -> T {
    with_cpu_exec_session(session, operation).expect("host exposes a CPU exec session")
}

/// Steady-state count for one operation, pooling warmed by four rounds.
fn steady_state(
    backend: &mut CpuBackend,
    operation: impl Fn(&mut dyn BackendSession) + Send + Sync,
) -> usize {
    let mut best = usize::MAX;
    for _ in 0..8 {
        let count = backend
            .with_backend_session(|session| measure(|| operation(session)))
            .unwrap();
        best = best.min(count);
    }
    best
}

fn check_family(
    backend: &mut CpuBackend,
    name: &str,
    ceiling: usize,
    allowed_batch_growth: usize,
    operation: impl Fn(&mut dyn BackendSession, &Tensor) + Send + Sync,
) {
    let single = batched_matrix(4, 1);
    let batch = batched_matrix(4, 64);
    let single_count = steady_state(backend, |session| operation(session, &single));
    let batch_count = steady_state(backend, |session| operation(session, &batch));
    eprintln!("{name}: batch 1 = {single_count}, batch 64 = {batch_count} allocations");
    assert!(
        batch_count <= single_count + allowed_batch_growth,
        "{name} provider scratch grows with the batch size \
         (batch 1 = {single_count}, batch 64 = {batch_count})"
    );
    assert!(
        single_count <= ceiling,
        "{name} uses {single_count} allocations, above the recorded ceiling of {ceiling}"
    );
}

// Measured steady-state ceilings through the public linalg route. The count
// includes the pooled outputs plus the provider's residual plan/workspace
// scratch; the deferred `*_into`/`*_workspace` API is what would isolate the
// provider-only 4/3/5 scratch.
#[cfg(feature = "native")]
const QR_CEILING: usize = 12;
#[cfg(feature = "native")]
const SVD_CEILING: usize = 14;
#[cfg(feature = "native")]
const PACKED_LU_CEILING: usize = 13;
#[cfg(feature = "native")]
const PACKED_LU_BATCH_GROWTH: usize = 0;
#[cfg(feature = "blas")]
const QR_CEILING: usize = 9;
#[cfg(feature = "blas")]
const SVD_CEILING: usize = 13;
// Measured: batch 1 = 7, batch 64 = 8. The packed-LU route keeps one
// batch-sized scratch vector outside the hoisted workspace; recorded as a
// finding rather than hidden.
#[cfg(feature = "blas")]
const PACKED_LU_CEILING: usize = 8;
#[cfg(feature = "blas")]
const PACKED_LU_BATCH_GROWTH: usize = 1;

#[test]
fn small_batched_qr_svd_packed_lu_have_batch_independent_provider_scratch() {
    let mut backend = CpuBackend::with_threads(1).expect("CPU backend");
    check_family(&mut backend, "qr", QR_CEILING, 0, |session, input| {
        with_cpu(session, |cpu| {
            cpu.qr(input).unwrap();
        });
    });
    check_family(&mut backend, "svd", SVD_CEILING, 0, |session, input| {
        with_cpu(session, |cpu| {
            cpu.svd(input).unwrap();
        });
    });
    check_family(
        &mut backend,
        "packed_lu",
        PACKED_LU_CEILING,
        PACKED_LU_BATCH_GROWTH,
        |session, input| {
            with_cpu(session, |cpu| {
                cpu.lu_factor(input).unwrap();
            });
        },
    );
}
