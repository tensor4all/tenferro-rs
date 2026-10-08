//! Allocation accounting for the CPU linalg routes the extraction is moving.
//!
//! Every family that moves out of `tenferro-linalg` needs a recorded steady-state allocation
//! baseline *before* it moves, or a move that quietly starts reacquiring scratch per call is
//! invisible. The packed-LU family (`lu_factor`, `lu_solve_prepared`, `lu_factor_solve`) landed
//! this way, then the SVD family; the remaining families are recorded below before they move.
//!
//! This target owns the process allocator, so it must stay a separate test binary. Counts are
//! steady-state: the pool is primed by warm-up calls first, because a cold first call legitimately
//! allocates.
//!
//! Byte totals are printed rather than asserted. The *number* of allocations is structural; the
//! sizes are not, because they depend on the shapes a caller chooses. `peak_live_bytes` is
//! indicative only: the balance also subtracts frees of memory that was allocated before the
//! window was armed, so it is not a peak-memory measurement.

#![cfg(feature = "native")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use num_complex::Complex64;
use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_linalg::{LinalgBackend, RankRevealingQrOptions, TensorLinalgExt};
use tenferro_tensor::{BackendSession, BackendSessionHost, Tensor, TypedTensor};

/// Counts allocations and live bytes while armed.
struct CountingAllocator;

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the caller's original layout and
// pointer; the counters only observe.
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
/// Single-threaded by construction: the backends below are built with one thread, and the counters
/// are process-wide.
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

fn faer_backend() -> CpuBackend {
    CpuBackend::with_threads(1).expect("faer CPU backend")
}

#[cfg(feature = "blas")]
fn blas_backend() -> CpuBackend {
    CpuBackend::with_threads(1).expect("BLAS CPU backend")
}

fn sample_real(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|index| {
            let row = (index % n) as f64;
            let col = (index / n) as f64;
            if row == col {
                n as f64 + 1.0
            } else {
                1.0 + row * 0.25 - col * 0.5
            }
        })
        .collect()
}

fn f64_matrix(n: usize) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![n, n], sample_real(n)).unwrap())
}

/// A tall column-major `m x n` matrix with `m > n`.
fn f64_tall(m: usize, n: usize) -> Tensor {
    let data: Vec<f64> = (0..m * n)
        .map(|index| {
            let row = index % m;
            let col = index / m;
            0.5 + (row as f64) * 0.25 - (col as f64) * 0.125
        })
        .collect();
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![m, n], data).unwrap())
}

/// A symmetric (Hermitian) positive-definite `n x n` matrix, for Cholesky and `eigh`.
fn f64_spd(n: usize) -> Tensor {
    let data = (0..n * n)
        .map(|index| {
            let (row, col) = (index % n, index / n);
            if row == col {
                n as f64 + 1.0
            } else {
                0.25 / (1.0 + row.abs_diff(col) as f64)
            }
        })
        .collect();
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![n, n], data).unwrap())
}

/// A Hermitian matrix with a dominant real diagonal.
fn c64_hermitian(n: usize) -> Tensor {
    let data = (0..n * n)
        .map(|index| {
            let (row, col) = (index % n, index / n);
            match row.cmp(&col) {
                std::cmp::Ordering::Equal => Complex64::new(n as f64 + 1.0, 0.0),
                std::cmp::Ordering::Less => Complex64::new(0.25, 0.125),
                std::cmp::Ordering::Greater => Complex64::new(0.25, -0.125),
            }
        })
        .collect();
    Tensor::from_typed::<Complex64>(TypedTensor::from_vec_col_major(vec![n, n], data).unwrap())
}

fn c64_matrix(n: usize) -> Tensor {
    let data = sample_real(n)
        .into_iter()
        .enumerate()
        .map(|(index, real)| {
            let imag = if index % (n + 1) == 0 { 0.5 } else { -0.25 };
            Complex64::new(real, imag)
        })
        .collect();
    Tensor::from_typed::<Complex64>(TypedTensor::from_vec_col_major(vec![n, n], data).unwrap())
}

/// Run `operation` until the report stops shrinking, so pooled scratch is already warm.
fn steady_state(
    host: &mut CpuBackend,
    operation: impl Fn(&mut dyn BackendSession) + Send + Sync,
) -> AllocationReport {
    let mut best: Option<AllocationReport> = None;
    for _ in 0..4 {
        let report = host
            .with_backend_session(|session| measure(|| operation(session)))
            .unwrap();
        if best.is_none_or(|best| report.allocations < best.allocations) {
            best = Some(report);
        }
    }
    best.unwrap_or_default()
}

/// Thin SVD through the public tensor route, so the measured call is the one users make.
fn svd_ext(input: &Tensor, session: &mut dyn BackendSession) {
    input.svd(session).unwrap();
}

/// Singular values only through the public tensor route.
fn svdvals_ext(input: &Tensor, session: &mut dyn BackendSession) {
    input.svdvals(session).unwrap();
}

fn with_session<T>(
    session: &mut dyn BackendSession,
    operation: impl FnOnce(&mut tenferro_cpu::CpuExecSession<'_>) -> T,
) -> T {
    with_cpu_exec_session(session, operation).expect("host exposes a CPU exec session")
}

/// Steady-state allocation ceiling per route and case.
///
/// These are the counts the current implementation achieves, not aspirations. They exist to stop a
/// move from quietly reacquiring per-call scratch. `lu_factor` returns `(packed_lu, pivots,
/// parity)`, so its wall is higher than a single-output primitive by construction.
const F64_LU_CEILINGS: &[(&str, usize)] = &[
    ("faer/lu_factor/48", 11),
    ("faer/lu_solve_prepared/48", 5),
    ("faer/lu_factor_solve/48", 13),
    ("blas/lu_factor/48", 6),
    ("blas/lu_solve_prepared/48", 5),
    ("blas/lu_factor_solve/48", 8),
    ("faer/lu_factor/complex32", 11),
    ("blas/lu_factor/complex32", 6),
    // The SVD family, faer route: the next slice's pre-move baseline.
    ("faer/svd/48", 12),
    ("faer/svdvals/48", 4),
    ("faer/svd/tall", 12),
    ("faer/svdvals/tall", 4),
    ("faer/svd/complex32", 14),
    ("faer/svdvals/complex32", 4),
];

/// Baselines for the families that followed SVD and packed LU out of `tenferro-linalg`, measured
/// through the public tensor routes on both CPU kinds. Recorded before the move and lowered where
/// the batched providers allocate less; no case is above its pre-move count.
const REMAINING_FAMILY_CEILINGS: &[(&str, usize)] = &[
    ("faer/cholesky/48", 4),
    ("faer/triangular_solve/48", 2),
    ("faer/lu/48", 13),
    ("faer/full_piv_lu/48", 17),
    ("faer/full_piv_lu_solve/48", 10),
    ("faer/solve/48", 8),
    ("faer/qr/48", 11),
    ("faer/qr/tall", 11),
    ("faer/householder_qr/tall", 73),
    ("faer/rank_revealing_qr/tall", 15),
    ("faer/eigh/48", 9),
    ("faer/eigvalsh/48", 4),
    ("faer/eig/48", 10),
    ("faer/eigvals/48", 6),
    ("faer/solve/complex32", 8),
    ("faer/qr/complex32", 11),
    ("faer/eigh/complex32", 11),
    ("faer/eig/complex32", 9),
    ("blas/cholesky/48", 3),
    ("blas/triangular_solve/48", 2),
    ("blas/lu/48", 12),
    ("blas/full_piv_lu/48", 15),
    ("blas/full_piv_lu_solve/48", 5),
    ("blas/solve/48", 5),
    ("blas/qr/48", 7),
    ("blas/qr/tall", 7),
    ("blas/householder_qr/tall", 4),
    ("blas/rank_revealing_qr/tall", 13),
    ("blas/eigh/48", 7),
    ("blas/eigvalsh/48", 4),
    ("blas/eig/48", 7),
    ("blas/eigvals/48", 4),
    ("blas/solve/complex32", 5),
    ("blas/qr/complex32", 7),
    ("blas/eigh/complex32", 8),
    ("blas/eig/complex32", 8),
];

fn ceiling(name: &str) -> usize {
    F64_LU_CEILINGS
        .iter()
        .chain(REMAINING_FAMILY_CEILINGS)
        .find(|(case, _)| *case == name)
        .map(|(_, ceiling)| *ceiling)
        .unwrap_or_else(|| panic!("no recorded allocation ceiling for {name}"))
}

fn check(
    host: &mut CpuBackend,
    failures: &mut Vec<String>,
    name: &str,
    operation: impl Fn(&mut dyn BackendSession) + Send + Sync,
) {
    let report = steady_state(host, operation);
    eprintln!(
        "{name}: {} allocations, {} bytes, peak {} live bytes",
        report.allocations, report.allocated_bytes, report.peak_live_bytes
    );
    let ceiling = ceiling(name);
    if report.allocations > ceiling {
        failures.push(format!(
            "{name}: {} allocations exceeds the recorded ceiling of {ceiling} \
             ({} bytes, peak {} live bytes)",
            report.allocations, report.allocated_bytes, report.peak_live_bytes
        ));
    }
}

#[test]
fn cpu_linalg_routes_take_their_scratch_from_the_session_buffer_pool() {
    let mut failures = Vec::new();

    let mut faer = faer_backend();
    let a = f64_matrix(48);
    let packed = faer
        .with_backend_session(|session| with_session(session, |cpu| cpu.lu_factor(&a).unwrap()))
        .unwrap();
    check(&mut faer, &mut failures, "faer/lu_factor/48", |session| {
        with_session(session, |cpu| {
            cpu.lu_factor(&a).unwrap();
        });
    });
    check(
        &mut faer,
        &mut failures,
        "faer/lu_solve_prepared/48",
        |session| {
            with_session(session, |cpu| {
                cpu.lu_solve_prepared(&a, &packed[0], &packed[1], &a, false, false)
                    .unwrap();
            });
        },
    );
    check(
        &mut faer,
        &mut failures,
        "faer/lu_factor_solve/48",
        |session| {
            with_session(session, |cpu| {
                cpu.lu_factor_solve(&a, &a).unwrap();
            });
        },
    );

    let c = c64_matrix(32);
    check(
        &mut faer,
        &mut failures,
        "faer/lu_factor/complex32",
        |session| {
            with_session(session, |cpu| {
                cpu.lu_factor(&c).unwrap();
            });
        },
    );

    #[cfg(feature = "blas")]
    {
        let mut blas = blas_backend();
        let packed = blas
            .with_backend_session(|session| with_session(session, |cpu| cpu.lu_factor(&a).unwrap()))
            .unwrap();
        check(&mut blas, &mut failures, "blas/lu_factor/48", |session| {
            with_session(session, |cpu| {
                cpu.lu_factor(&a).unwrap();
            });
        });
        check(
            &mut blas,
            &mut failures,
            "blas/lu_solve_prepared/48",
            |session| {
                with_session(session, |cpu| {
                    cpu.lu_solve_prepared(&a, &packed[0], &packed[1], &a, false, false)
                        .unwrap();
                });
            },
        );
        check(
            &mut blas,
            &mut failures,
            "blas/lu_factor_solve/48",
            |session| {
                with_session(session, |cpu| {
                    cpu.lu_factor_solve(&a, &a).unwrap();
                });
            },
        );
        check(
            &mut blas,
            &mut failures,
            "blas/lu_factor/complex32",
            |session| {
                with_session(session, |cpu| {
                    cpu.lu_factor(&c).unwrap();
                });
            },
        );
    }

    // The SVD family on the faer route: thin, values-only, square, tall and complex.
    let tall = f64_tall(64, 24);
    check(&mut faer, &mut failures, "faer/svd/48", |session| {
        svd_ext(&a, session);
    });
    check(&mut faer, &mut failures, "faer/svdvals/48", |session| {
        svdvals_ext(&a, session);
    });
    check(&mut faer, &mut failures, "faer/svd/tall", |session| {
        svd_ext(&tall, session);
    });
    check(&mut faer, &mut failures, "faer/svdvals/tall", |session| {
        svdvals_ext(&tall, session);
    });
    check(&mut faer, &mut failures, "faer/svd/complex32", |session| {
        svd_ext(&c, session);
    });
    check(
        &mut faer,
        &mut failures,
        "faer/svdvals/complex32",
        |session| {
            svdvals_ext(&c, session);
        },
    );

    assert!(
        failures.is_empty(),
        "CPU linalg routes allocate more per call than the recorded ceilings:\n{}",
        failures.join("\n")
    );
}

/// Every remaining family on one CPU kind, through the public tensor routes.
fn check_remaining_families(host: &mut CpuBackend, kind: &str, failures: &mut Vec<String>) {
    let a = f64_matrix(48);
    let spd = f64_spd(48);
    let tall = f64_tall(64, 24);
    let c = c64_matrix(32);
    let herm = c64_hermitian(32);
    let name = |case: &str| format!("{kind}/{case}");
    check(host, failures, &name("cholesky/48"), |session| {
        spd.cholesky(session).unwrap();
    });
    check(host, failures, &name("triangular_solve/48"), |session| {
        spd.triangular_solve(&a, true, true, false, false, session)
            .unwrap();
    });
    check(host, failures, &name("lu/48"), |session| {
        a.lu(session).unwrap();
    });
    check(host, failures, &name("full_piv_lu/48"), |session| {
        a.full_piv_lu(session).unwrap();
    });
    check(host, failures, &name("full_piv_lu_solve/48"), |session| {
        a.full_piv_lu_solve(&a, session).unwrap();
    });
    check(host, failures, &name("solve/48"), |session| {
        a.solve(&a, session).unwrap();
    });
    check(host, failures, &name("qr/48"), |session| {
        a.qr(session).unwrap();
    });
    check(host, failures, &name("qr/tall"), |session| {
        tall.qr(session).unwrap();
    });
    check(host, failures, &name("householder_qr/tall"), |session| {
        tall.householder_qr(session).unwrap();
    });
    check(host, failures, &name("rank_revealing_qr/tall"), |session| {
        tall.rank_revealing_qr(RankRevealingQrOptions::default(), session)
            .unwrap();
    });
    check(host, failures, &name("eigh/48"), |session| {
        spd.eigh(session).unwrap();
    });
    check(host, failures, &name("eigvalsh/48"), |session| {
        spd.eigvalsh(session).unwrap();
    });
    check(host, failures, &name("eig/48"), |session| {
        a.eig(session).unwrap();
    });
    check(host, failures, &name("eigvals/48"), |session| {
        a.eigvals(session).unwrap();
    });
    check(host, failures, &name("solve/complex32"), |session| {
        c.solve(&c, session).unwrap();
    });
    check(host, failures, &name("qr/complex32"), |session| {
        c.qr(session).unwrap();
    });
    check(host, failures, &name("eigh/complex32"), |session| {
        herm.eigh(session).unwrap();
    });
    check(host, failures, &name("eig/complex32"), |session| {
        c.eig(session).unwrap();
    });
}

#[test]
fn remaining_linalg_families_keep_their_steady_state_allocation_counts() {
    let mut failures = Vec::new();
    check_remaining_families(&mut faer_backend(), "faer", &mut failures);
    #[cfg(feature = "blas")]
    check_remaining_families(&mut blas_backend(), "blas", &mut failures);
    assert!(
        failures.is_empty(),
        "CPU linalg routes allocate more per call than the recorded ceilings:\n{}",
        failures.join("\n")
    );
}
