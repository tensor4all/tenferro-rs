#![cfg(feature = "fftw")]
//! FFTW's planner is process-global and not reentrant: two plans created at
//! once without a lock corrupt it (a SIGSEGV in `fftw_mkplan_d`). Loading a
//! provider calls `fftw_make_planner_thread_safe`, so FFTW's own lock
//! serializes all planning in the process; tenferro adds no planner hooks.
//! These tests plan concurrently from several tenferro threads, and from a
//! thread that uses the FFTW API directly as another library would.
//!
//! The tests share one FFTW instance and its process-wide planner thread
//! count, so each holds `SERIAL` for its whole body; the concurrency under test
//! is within each test.

#[path = "fftw_support/mod.rs"]
mod support;

use std::ffi::{c_int, c_uint, c_void};
use std::sync::{Mutex, PoisonError};

use libloading::Library;
use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftOptions, Fftw};
use tenferro_tensor::Tensor;

type PlanComplex =
    unsafe extern "C" fn(c_int, *mut c_void, *mut c_void, c_int, c_uint) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
type SetThreads = unsafe extern "C" fn(c_int);
type GetThreads = unsafe extern "C" fn() -> c_int;

static SERIAL: Mutex<()> = Mutex::new(());

const ESTIMATE: c_uint = 1 << 6;

#[test]
fn tenferro_threads_plan_concurrently() {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let directory = support::test_directory();
    for threads in [1, 2] {
        let Some(provider) = support::load(directory.as_deref(), threads) else {
            return;
        };
        support::plan_concurrently(&provider, 4, support::rounds());
    }
}

/// Another library planning through the FFTW API, with FFTW's default thread
/// count, while tenferro plans and destroys plans on other threads. Only FFTW's
/// lock serializes the two: tenferro's own lock does not cover the foreign
/// thread.
#[test]
fn foreign_planning_is_serialized_with_tenferro_planning() {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let directory = support::test_directory();
    let Some(provider) = support::load(directory.as_deref(), 2) else {
        return;
    };
    // The provider has loaded this library, so this handle refers to the same
    // FFTW instance and its global planner.
    // SAFETY: the library comes from the FFTW installation the provider loaded.
    let library =
        unsafe { Library::new(support::library_path(directory.as_deref(), "fftw3")) }.unwrap();
    // SAFETY: both symbols have these C signatures in every FFTW 3 release.
    let (plan, destroy) = unsafe {
        (
            *library.get::<PlanComplex>(b"fftw_plan_dft_1d").unwrap(),
            *library.get::<Destroy>(b"fftw_destroy_plan").unwrap(),
        )
    };
    let rounds = support::rounds();
    let start = std::sync::Arc::new(std::sync::Barrier::new(3));
    let foreign = std::thread::spawn({
        let start = start.clone();
        move || {
            start.wait();
            for step in 0..2 * rounds {
                let len = 7 + 3 * step;
                let mut buffer = vec![Complex64::new(0.0, 0.0); len];
                let p = buffer.as_mut_ptr().cast();
                // SAFETY: an in-place plan over a private buffer of `len` values.
                let handle = unsafe { plan(c_int::try_from(len).unwrap(), p, p, -1, ESTIMATE) };
                assert!(!handle.is_null(), "FFTW could not plan length {len}");
                // SAFETY: the plan was created above and is destroyed once.
                unsafe { destroy(handle) };
            }
        }
    });
    let tenferro: Vec<_> = (0..2)
        .map(|thread| {
            let (provider, start) = (provider.clone(), start.clone());
            std::thread::spawn(move || {
                start.wait();
                support::plan_many(&provider, thread, 2, rounds);
            })
        })
        .collect();
    foreign.join().unwrap();
    for thread in tenferro {
        thread.join().unwrap();
    }
}

/// One transform of a length no other test in this binary plans, so it creates
/// a new FFTW plan.
fn transform(provider: &Fftw, len: usize) {
    let mut engine = CpuEngine::serial();
    provider.install(&mut engine);
    let x = Tensor::from_vec_col_major([len], vec![Complex64::new(1.0, 0.0); len]).unwrap();
    let y = FftExecutor::default()
        .fft_with(&x, &FftOptions::default(), &mut engine)
        .unwrap();
    let dc = y.as_slice::<Complex64>().unwrap()[0];
    assert!((dc - Complex64::new(len as f64, 0.0)).norm() < 1e-9, "{dc}");
}

/// Other code sets the process-wide planner thread count, then a tenferro plan
/// is created: the plan sets the provider's own count, one for a
/// single-threaded provider, and leaves it there.
#[test]
fn tenferro_plans_set_their_own_thread_count() {
    let _serial = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    let directory = support::test_directory();
    let (Some(threaded), Some(single)) = (
        support::load(directory.as_deref(), 2),
        support::load(directory.as_deref(), 1),
    ) else {
        return;
    };
    // The provider has loaded these libraries, so the handles refer to the
    // same FFTW instance and its planner thread count.
    // SAFETY: the libraries come from the FFTW installation the provider loaded.
    let core =
        unsafe { Library::new(support::library_path(directory.as_deref(), "fftw3")) }.unwrap();
    // SAFETY: `fftw_planner_nthreads` has this C signature.
    let companion = if unsafe { core.get::<GetThreads>(b"fftw_planner_nthreads") }.is_ok() {
        None
    } else {
        let path = support::library_path(directory.as_deref(), "fftw3_threads");
        // SAFETY: as above, from the same FFTW installation.
        Some(unsafe { Library::new(path) }.unwrap())
    };
    let threads = companion.as_ref().unwrap_or(&core);
    // SAFETY: both symbols have these C signatures in FFTW 3.3.9 and later.
    let (set, get) = unsafe {
        (
            *threads
                .get::<SetThreads>(b"fftw_plan_with_nthreads")
                .unwrap(),
            *threads.get::<GetThreads>(b"fftw_planner_nthreads").unwrap(),
        )
    };
    // SAFETY: the threaded load initialized FFTW's threads, and `SERIAL` keeps
    // the other tests in this binary from planning meanwhile.
    unsafe { set(3) };
    transform(&single, 4099);
    // SAFETY: as above.
    assert_eq!(unsafe { get() }, 1);
    // SAFETY: as above.
    unsafe { set(3) };
    transform(&threaded, 4111);
    // SAFETY: as above.
    assert_eq!(unsafe { get() }, 2);
    // SAFETY: as above.
    unsafe { set(1) };
}
