#![cfg(feature = "cuda")]

//! Refuse tenferro's typed host copy after a successful CUDA download (#2335).
//!
//! This binary owns its allocator. Only the payload-sized, f64-aligned reservation
//! is refused; the runtime's `DownloadSlot` uses a byte vector (alignment 1), and
//! raw downloads use native pinned staging. Neither staging allocation is refused.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::TryReserveError;
use std::error::Error as _;
use std::sync::atomic::{AtomicBool, Ordering};

use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::{Error, Tensor};

const ELEMENTS: usize = 4093;
const PAYLOAD_BYTES: usize = ELEMENTS * std::mem::size_of::<f64>();
const RAW_OP: &str = "raw download host reservation";

struct RefusingAllocator;

static ARMED: AtomicBool = AtomicBool::new(false);

fn refuse(size: usize, align: usize) -> bool {
    size == PAYLOAD_BYTES
        && align == std::mem::align_of::<f64>()
        && ARMED.swap(false, Ordering::Relaxed)
}

// SAFETY: the one selected request returns null, a permitted allocation failure;
// all other calls forward the caller's original layout and pointer to System.
unsafe impl GlobalAlloc for RefusingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if refuse(layout.size(), layout.align()) {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller's layout is forwarded unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller's pointer and layout are forwarded unchanged.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if refuse(new_size, layout.align()) {
            return std::ptr::null_mut();
        }
        // SAFETY: the caller's arguments are forwarded unchanged; a refused realloc
        // above leaves the original allocation untouched.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: RefusingAllocator = RefusingAllocator;

fn download(
    engine: &mut CudaEngine,
    device: &Tensor,
    raw: bool,
) -> tenferro_tensor::Result<Tensor> {
    if raw {
        engine.with_raw(RAW_OP, |session| {
            session
                .download_tensor(device.as_typed::<f64>().expect("f64 device tensor"), RAW_OP)
                .map(Tensor::from_typed)
        })
    } else {
        download_tensor(engine.runtime(), device)
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn post_download_host_reservation_failure_preserves_typed_source() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine");
    let values: Vec<f64> = (0..ELEMENTS).map(|i| i as f64 - 100.5).collect();
    let host = Tensor::from_vec_col_major(vec![ELEMENTS], values.clone()).expect("host tensor");
    let device = upload_tensor(engine.runtime(), &host).expect("upload");

    for raw in [false, true] {
        let op = if raw { RAW_OP } else { "cuda download" };
        // Warm each route and verify that the same input downloads successfully.
        let warm = download(&mut engine, &device, raw).expect("warm download");
        assert_eq!(warm.as_slice::<f64>().expect("f64 payload"), values);

        ARMED.store(true, Ordering::Relaxed);
        let result = download(&mut engine, &device, raw);
        let unconsumed = ARMED.swap(false, Ordering::Relaxed);
        assert!(!unconsumed, "{op}: the payload reservation was not refused");
        let error = result.expect_err("the typed host-copy reservation must fail fallibly");
        assert!(
            matches!(&error, Error::BackendSource { op: actual, .. } if *actual == op),
            "expected BackendSource for {op}, got {error:?}"
        );
        let mut source = error.source();
        let mut found = false;
        while let Some(cause) = source {
            if cause.downcast_ref::<TryReserveError>().is_some() {
                found = true;
                break;
            }
            source = cause.source();
        }
        assert!(found, "{op}: TryReserveError missing from {error:?}");

        let recovered = download(&mut engine, &device, raw).expect("download after refusal");
        assert_eq!(recovered.as_slice::<f64>().expect("f64 payload"), values);
    }
}
