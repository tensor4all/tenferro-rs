#![cfg(feature = "autodiff")]

use num_complex::Complex64;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuEngine;
use tenferro_fft::{EagerSessionFftExt, EagerTensorFftExt, FftNorm};
use tenferro_tensor::Tensor;

const LEN: usize = 4096;
const INPUT_BYTES: usize = LEN * size_of::<Complex64>();
// The allocation counters below are process-global, so the measurement cases
// must not run concurrently in one test binary.
static MEASUREMENT_LOCK: Mutex<()> = Mutex::new(());
static ENABLED: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static TOTAL_COUNT: AtomicUsize = AtomicUsize::new(0);
static TOTAL_BYTES: AtomicUsize = AtomicUsize::new(0);
struct Allocator;
fn record(size: usize) {
    if ENABLED.load(Ordering::Relaxed) {
        TOTAL_COUNT.fetch_add(1, Ordering::Relaxed);
        TOTAL_BYTES.fetch_add(size, Ordering::Relaxed);
        if size >= INPUT_BYTES {
            COUNT.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size, Ordering::Relaxed);
        }
    }
}
// SAFETY: original allocator arguments are forwarded unchanged to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: original allocation layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: allocation originated from System.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        // SAFETY: allocation and requested size are forwarded unchanged.
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn measured<T>(expected: (usize, usize), f: impl FnOnce() -> T) -> T {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ENABLED.store(false, Ordering::SeqCst);
        }
    }
    COUNT.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    TOTAL_COUNT.store(0, Ordering::Relaxed);
    TOTAL_BYTES.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::SeqCst);
    let reset = Reset;
    let output = f();
    drop(reset);
    assert_eq!(
        (COUNT.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed)),
        expected,
        "unexpected input/output-sized allocation count and bytes"
    );
    eprintln!(
        "input-sized count/bytes={expected:?}; total count/bytes={:?}",
        (
            TOTAL_COUNT.load(Ordering::Relaxed),
            TOTAL_BYTES.load(Ordering::Relaxed)
        )
    );
    output
}

#[test]
fn ordinary_eager_drop_and_consuming_fft_reuse_without_input_sized_allocations() {
    let _serial = MEASUREMENT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut engine = CpuEngine::serial();
    assert_eq!(engine.num_threads(), 1);
    let runtime = EagerRuntime::new().unwrap();
    let input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([64, 64], vec![Complex64::new(1., 0.); LEN]).unwrap(),
        runtime.clone(),
    )
    .unwrap();
    let transform = |engine: &mut CpuEngine| {
        runtime
            .with_eager_session(engine, |s| s.fft(&input, None, 1, FftNorm::Backward))
            .unwrap()
    };
    let first = measured((1, INPUT_BYTES), || transform(&mut engine));
    let second = measured((1, INPUT_BYTES), || transform(&mut engine));
    assert_ne!(
        first
            .value()
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap()
            .as_ptr(),
        second
            .value()
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap()
            .as_ptr()
    );
    drop(first);
    drop(second);
    let output = measured((0, 0), || transform(&mut engine));
    let pointer = output
        .value()
        .unwrap()
        .as_slice::<Complex64>()
        .unwrap()
        .as_ptr();
    drop(output);
    let output = measured((0, 0), || transform(&mut engine));
    assert_eq!(
        output
            .value()
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap()
            .as_ptr(),
        pointer
    );
    assert_eq!(
        input.value().unwrap().as_slice::<Complex64>().unwrap(),
        &[Complex64::new(1., 0.); LEN]
    );
    measured((0, 0), || {
        let owned = output.into_value().unwrap();
        {
            let s: &mut dyn tenferro_tensor::Engine = &mut engine;
            s.reclaim_buffer(owned)
        };
    });
    let output = measured((0, 0), || transform(&mut engine));
    assert_eq!(
        output
            .value()
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap()
            .as_ptr(),
        pointer
    );
    let output = measured((0, 0), || {
        output
            .fft_in_place(&mut engine, 1, FftNorm::Backward)
            .unwrap()
    });
    assert_eq!(
        output
            .value()
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap()
            .as_ptr(),
        pointer
    );
    drop(output);
    assert!(engine.buffer_pool_stats().buffers >= 1);
}

#[test]
fn whole_tensor_axis_fft_lane_comes_from_the_buffer_pool() {
    // The transformed axis is the whole tensor here, so the per-call lane and
    // the RustFFT in-place scratch are both full `LEN`-element complex buffers
    // and therefore visible to the input-sized counter (the short-axis case
    // above hides them). The r2c output is only half that size, so the count
    // observes the lane and the scratch alone. Before the lane was pooled, each
    // call added a fresh full-length lane allocation on top of the scratch;
    // the steady state below pins the reduced count.
    let _serial = MEASUREMENT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut engine = CpuEngine::serial();
    assert_eq!(engine.num_threads(), 1);
    let runtime = EagerRuntime::new().unwrap();
    let input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([LEN], vec![1.0f64; LEN]).unwrap(),
        runtime.clone(),
    )
    .unwrap();
    let transform = |engine: &mut CpuEngine| {
        runtime
            .with_eager_session(engine, |s| s.rfft(&input, None, 0, FftNorm::Backward))
            .unwrap()
    };
    // Warm the plan cache and the buffer pool; only a steady-state call shows
    // whether the lane is recycled.
    drop(transform(&mut engine));
    let output = measured((1, INPUT_BYTES), || transform(&mut engine));
    drop(output);
    let output = measured((1, INPUT_BYTES), || transform(&mut engine));
    drop(output);
}
