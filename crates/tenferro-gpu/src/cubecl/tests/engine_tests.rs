use std::error::Error as _;
use std::panic::AssertUnwindSafe;

use crate::cubecl::{
    download_pending, download_tensor, upload_tensor, CudaDeviceId, CudaEngine, PinnedHostBuffer,
};
use crate::cuda::{CudaEngineError, PublicationError};
use crate::Tensor;
use tenferro_tensor::{TensorElementwise, TensorRead};

fn engine() -> CudaEngine {
    CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine")
}

fn vector(data: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(vec![data.len()], data.to_vec()).expect("vector")
}

fn source_error<T: std::error::Error + 'static>(error: &crate::Error) -> Option<&T> {
    let mut source = error.source();
    while let Some(current) = source {
        if let Some(found) = current.downcast_ref::<T>() {
            return Some(found);
        }
        source = current.source();
    }
    None
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires a CUDA driver to format synthetic driver errors"]
fn the_sticky_status_classification_covers_context_losing_driver_errors() {
    use cudarc::driver::sys::CUresult;

    let illegal = crate::Error::backend_source(
        "engine_test",
        cudarc::driver::DriverError(CUresult::CUDA_ERROR_ILLEGAL_ADDRESS),
    );
    assert!(crate::cubecl::engine::device_poison_reason(&illegal).is_some());

    let not_ready = crate::Error::backend_source(
        "engine_test",
        cudarc::driver::DriverError(CUresult::CUDA_ERROR_NOT_READY),
    );
    assert!(crate::cubecl::engine::device_poison_reason(&not_ready).is_none());
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_panic_marks_only_that_engine_broken_and_cleanup_still_runs() {
    let mut engine = engine();
    let panicked = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _: crate::Result<()> = engine.guarded("engine_test::panic", |_| panic!("injected"));
    }));
    assert!(panicked.is_err());
    assert!(engine.is_broken());

    // Retirement and synchronization are not operations: they still run on a broken engine.
    engine.synchronize().expect("cleanup on a broken engine");

    let error = engine
        .with_raw("engine_test::after_panic", |_| Ok(()))
        .expect_err("a broken engine refuses operations");
    assert!(matches!(
        source_error::<CudaEngineError>(&error),
        Some(CudaEngineError::Broken)
    ));
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_sticky_device_fault_poisons_every_engine_on_the_device() {
    let mut first = engine();
    let mut second = engine();
    // A synthetic device key keeps this test from poisoning the real GPU for the rest of the run.
    let test_device = [0x5a; 16];
    first.use_test_device_uuid(test_device);
    second.use_test_device_uuid(test_device);
    first.record_device_poison_for_test("injected illegal address");
    assert_eq!(
        first.device_poisoned(),
        Some("injected illegal address"),
        "the engine that observed the fault reports it"
    );

    let error = second
        .with_raw("engine_test::poisoned_device", |_| Ok(()))
        .expect_err("every engine of a poisoned device refuses operations");
    assert!(matches!(
        source_error::<CudaEngineError>(&error),
        Some(CudaEngineError::DevicePoisoned { reason }) if reason == "injected illegal address"
    ));
    assert_eq!(
        second.device_poisoned(),
        Some("injected illegal address"),
        "the refusing engine now reports the device fault too"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn an_oversized_transfer_is_refused_before_submission() {
    let mut engine = engine();
    let runtime = engine.runtime().clone();
    let host = vector(&[1.0; 32]);
    let seed = upload_tensor(&runtime, &host).expect("seed");
    let value = engine
        .add_read(
            TensorRead::from_tensor(&seed),
            TensorRead::from_tensor(&seed),
        )
        .expect("produce");
    let publication = engine.publish(value).expect("publish");

    // The copy would retain both its source and destination: 2 * 32 * 8 bytes.
    engine.set_unfinished_memory_limit(Some(1));
    let error = publication
        .copy_into(&mut engine)
        .expect_err("a transfer larger than the limit is refused");
    let PublicationError::CopyRejected { error, publication } = error else {
        panic!("expected a pre-submission refusal, got an unproven submission");
    };
    assert!(
        error.to_string().contains("unfinished-memory"),
        "the refusal names the limit it would exceed: {error}"
    );
    engine.set_unfinished_memory_limit(None);

    // The refused publication is still usable and the engine is unaffected.
    let copied = publication
        .copy_into(&mut engine)
        .expect("copy after raising limit");
    let received = copied.wait().expect("wait");
    let host = download_tensor(&runtime, &received).expect("read back");
    assert!(host
        .as_slice::<f64>()
        .expect("values")
        .iter()
        .all(|&x| x == 2.0));
}

/// A long run loop keeps the device memory retained by unfinished work within
/// the caller's limit. Each iteration detaches a pending copy without waiting;
/// the next admission either fits, drains completed work, or waits for the
/// stream, so the retained counter never passes the limit.
#[test]
#[ignore = "requires a CUDA device"]
fn unfinished_work_stays_within_its_limit_in_a_long_run_loop() {
    let mut engine = engine();
    let runtime = engine.runtime().clone();
    let host = vector(&[1.0; 64]);
    let seed = upload_tensor(&runtime, &host).expect("seed");
    let bytes = std::mem::size_of_val(host.as_slice::<f64>().expect("host values")) as u64;
    let limit = 4 * bytes;
    engine.set_unfinished_memory_limit(Some(limit));

    let mut observed_retained = false;
    for iteration in 0..64 {
        let value = engine
            .add_read(
                TensorRead::from_tensor(&seed),
                TensorRead::from_tensor(&seed),
            )
            .expect("produce a value");
        let publication = engine.publish(value).expect("publish");
        let pending = publication.copy_into(&mut engine).expect("copy");
        // Drop without waiting: the copy detaches its completion record, which
        // the next submission must account for.
        drop(pending);
        let retained = engine.unfinished_memory_bytes();
        observed_retained |= retained > 0;
        assert!(
            retained <= limit,
            "iteration {iteration}: retained {retained} bytes exceeds the limit {limit}"
        );
    }
    assert!(
        observed_retained,
        "the loop must actually retain unfinished work at least once"
    );

    engine.synchronize().expect("drain the run loop");
    engine.set_unfinished_memory_limit(None);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_live_pending_transfer_counts_against_the_unfinished_limit() {
    let mut engine = engine();
    let runtime = engine.runtime().clone();
    let host = vector(&[1.0; 32]);
    let bytes = std::mem::size_of_val(host.as_slice::<f64>().expect("host values")) as u64;
    let seed = upload_tensor(&runtime, &host).expect("seed");
    engine.set_unfinished_memory_limit(Some(bytes));

    let value = engine
        .add_read(
            TensorRead::from_tensor(&seed),
            TensorRead::from_tensor(&seed),
        )
        .expect("produce");
    let buffer = PinnedHostBuffer::new(&runtime, bytes as usize).expect("pinned buffer");
    let pending = download_pending(&runtime, value, buffer).expect("pending download");
    assert!(
        engine.unfinished_memory_bytes() >= bytes,
        "a live pending handle is unfinished work before it is detached"
    );

    // A second copy of the same size would exceed the limit while the first
    // reservation is live, even after the admission drains the stream.
    let second = engine
        .add_read(
            TensorRead::from_tensor(&seed),
            TensorRead::from_tensor(&seed),
        )
        .expect("produce");
    let buffer = PinnedHostBuffer::new(&runtime, bytes as usize).expect("pinned buffer");
    assert!(
        download_pending(&runtime, second, buffer).is_err(),
        "the live reservation is admitted before the second submission"
    );

    pending.wait().expect("complete the first copy");
    assert_eq!(
        engine.unfinished_memory_bytes(),
        0,
        "observing completion releases the live reservation"
    );
    engine.set_unfinished_memory_limit(None);
}
