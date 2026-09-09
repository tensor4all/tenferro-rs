#![cfg(feature = "autodiff")]

use num_complex::Complex64;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_fft::{EagerTensorFftExt, FftNorm};
use tenferro_tensor::{BackendSessionHost, Tensor};

const LEN: usize = 4096;
const INPUT_BYTES: usize = LEN * size_of::<Complex64>();
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
    let mut backend = CpuBackend::with_threads(1).unwrap();
    assert_eq!(backend.num_threads(), 1);
    let runtime = EagerRuntime::with_cpu_backend(backend.clone()).unwrap();
    let input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([64, 64], vec![Complex64::new(1., 0.); LEN]).unwrap(),
        runtime,
    )
    .unwrap();
    let first = measured((1, INPUT_BYTES), || {
        input.fft(None, 1, FftNorm::Backward).unwrap()
    });
    let second = measured((1, INPUT_BYTES), || {
        input.fft(None, 1, FftNorm::Backward).unwrap()
    });
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
    let output = measured((0, 0), || input.fft(None, 1, FftNorm::Backward).unwrap());
    let pointer = output
        .value()
        .unwrap()
        .as_slice::<Complex64>()
        .unwrap()
        .as_ptr();
    drop(output);
    let output = measured((0, 0), || input.fft(None, 1, FftNorm::Backward).unwrap());
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
        backend.with_backend_session(|s| s.reclaim_buffer(owned));
    });
    let output = measured((0, 0), || input.fft(None, 1, FftNorm::Backward).unwrap());
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
        output.fft_in_place(1, FftNorm::Backward).unwrap()
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
    assert!(backend.buffer_pool_stats().unwrap().buffers >= 1);
}
