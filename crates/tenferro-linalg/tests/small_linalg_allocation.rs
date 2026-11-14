//! Small batched QR / SVD / packed-LU provider-scratch ceilings (#2178 item 7).
//!
//! This target owns the process allocator, so it must stay a separate test
//! binary. Counts are steady state: the session buffer pool is primed by
//! warm-up calls first. Only Rust heap allocations are counted; any C-side
//! scratch inside the numeric library is outside this counter and out of scope.
//!
//! The provider hoists its workspace out of the batch loop, so the steady-state
//! count must not grow per item. Each family pins a single-item and a batched
//! ceiling and checks the batch-1 to batch-64 growth; on the native two-thread
//! lane the only growth is the second lane's hoisted scratch, recorded next to
//! the one-thread ceilings.

// The placement ledger allocates per token, so allocation counts are taken without it.
#![cfg(not(feature = "ledger"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tenferro_cpu::{with_cpu_engine, CpuEngine};
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::{Engine, Tensor, TypedTensor};

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
    session: &mut dyn Engine,
    operation: impl FnOnce(&mut tenferro_cpu::CpuEngine) -> T,
) -> T {
    with_cpu_engine(session, operation).expect("host exposes a CPU exec session")
}

/// Steady-state count for one operation, pooling warmed by four rounds.
fn steady_state(
    backend: &mut CpuEngine,
    operation: impl Fn(&mut dyn Engine) + Send + Sync,
) -> usize {
    let mut best = usize::MAX;
    for _ in 0..8 {
        let count = measure(|| operation(&mut *backend));
        best = best.min(count);
    }
    best
}

fn check_family(
    backend: &mut CpuEngine,
    name: &str,
    single_ceiling: usize,
    batch_ceiling: usize,
    allowed_batch_growth: usize,
    operation: impl Fn(&mut dyn Engine, &Tensor) + Send + Sync,
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
        single_count <= single_ceiling,
        "{name} uses {single_count} allocations, above the recorded single-item ceiling of \
         {single_ceiling}"
    );
    assert!(
        batch_count <= batch_ceiling,
        "{name} uses {batch_count} allocations at batch 64, above the recorded batched ceiling of \
         {batch_ceiling}"
    );
}

// Measured steady-state ceilings through the public linalg route, at
// `tlinalg 0.3.1` on the borrowed-workspace route. Every faer family now
// queries `*_workspace` and writes straight into pooled `*_into` destinations,
// so the provider scratch is one pooled region per call and the outputs return
// to the pool when the caller drops them.
//
// The residual 3 allocations per call on the one-thread lane are host/pool
// bookkeeping, not provider scratch:
//   * one `BufferPool` in-flight-accounting node insert: `BufferPoolLoan::drop`
//     clears the in-flight maps after every operation
//     (`tenferro-cpu/src/backend.rs`), so the next operation's first tracked
//     checkout re-inserts its key;
//   * the provider module's own result `Vec` (the `qr_impl`/`svd_impl` output
//     list or the `lu_factor` triple) and the CPU backend's dtype-erased
//     `Vec<Tensor>` mapping of it.
// The two-thread lane adds one allocation per fanned-out lane: rayon's
// `HeapJob` for the second lane's `tlinalg::batch::run` spawn.
const QR_CEILING_1T: (usize, usize) = (3, 3);
const QR_CEILING_2T: (usize, usize) = (3, 5);
const SVD_CEILING_1T: (usize, usize) = (3, 3);
const SVD_CEILING_2T: (usize, usize) = (3, 5);
const PACKED_LU_CEILING_1T: (usize, usize) = (3, 3);
const PACKED_LU_CEILING_2T: (usize, usize) = (3, 5);
const QR_BATCH_GROWTH_1T: usize = 0;
// Measured: batch 1 = 3, batch 64 = 5. The two extra allocations are the
// second lane's rayon `HeapJob`, created when Auto fans out from one lane to
// two; they do not grow per item beyond the budget.
const QR_BATCH_GROWTH_2T: usize = 2;
const SVD_BATCH_GROWTH_1T: usize = 0;
// Measured: batch 1 = 3, batch 64 = 5, the same second-lane spawn.
const SVD_BATCH_GROWTH_2T: usize = 2;
const PACKED_LU_BATCH_GROWTH_1T: usize = 0;
// Measured: batch 1 = 3, batch 64 = 5, the same second-lane spawn.
const PACKED_LU_BATCH_GROWTH_2T: usize = 2;

fn check_lane(
    threads: usize,
    qr_ceiling: (usize, usize),
    svd_ceiling: (usize, usize),
    packed_lu_ceiling: (usize, usize),
    qr_growth: usize,
    svd_growth: usize,
    packed_lu_growth: usize,
) {
    let mut backend = tenferro_cpu::CpuEngine::new(
        tenferro_cpu::Pool::builder()
            .threads(threads)
            .build()
            .unwrap(),
    );
    let suffix = format!("{threads}T");
    check_family(
        &mut backend,
        &format!("qr {suffix}"),
        qr_ceiling.0,
        qr_ceiling.1,
        qr_growth,
        |session, input| {
            with_cpu(session, |cpu| {
                cpu.qr(input).unwrap();
            });
        },
    );
    check_family(
        &mut backend,
        &format!("svd {suffix}"),
        svd_ceiling.0,
        svd_ceiling.1,
        svd_growth,
        |session, input| {
            with_cpu(session, |cpu| {
                cpu.svd(input).unwrap();
            });
        },
    );
    check_family(
        &mut backend,
        &format!("packed_lu {suffix}"),
        packed_lu_ceiling.0,
        packed_lu_ceiling.1,
        packed_lu_growth,
        |session, input| {
            with_cpu(session, |cpu| {
                cpu.lu_factor(input).unwrap();
            });
        },
    );
}

#[test]
fn small_batched_qr_svd_packed_lu_have_batch_independent_provider_scratch() {
    check_lane(
        1,
        QR_CEILING_1T,
        SVD_CEILING_1T,
        PACKED_LU_CEILING_1T,
        QR_BATCH_GROWTH_1T,
        SVD_BATCH_GROWTH_1T,
        PACKED_LU_BATCH_GROWTH_1T,
    );
    check_lane(
        2,
        QR_CEILING_2T,
        SVD_CEILING_2T,
        PACKED_LU_CEILING_2T,
        QR_BATCH_GROWTH_2T,
        SVD_BATCH_GROWTH_2T,
        PACKED_LU_BATCH_GROWTH_2T,
    );
}
