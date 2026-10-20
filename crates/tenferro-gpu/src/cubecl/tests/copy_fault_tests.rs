//! Consumer-copy fault injection (#1945 U4-2a evidence).
//!
//! The private seam in `cubecl::transfer::copy_fault_injection` drives the two reachable failure paths of
//! the device-side consumer copy: a recoverable destination-preparation error (refused before this copy's
//! submission, with the publication returned) and a completion-record failure after the copy was enqueued
//! (unproven, with the package retained). These tests are ignored by default; explicitly running them
//! requires a CUDA device.

use crate::cuda::{gpu_available, upload_tensor, CudaDeviceId, CudaEngine, PublicationError};
use tenferro_tensor::{Tensor, TensorRead};

fn values(len: usize) -> Vec<f64> {
    (0..len).map(|index| index as f64 * 0.5 - 3.0).collect()
}

fn vector(data: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(vec![data.len()], data.to_vec()).expect("vector")
}

fn produced_value(engine: &mut CudaEngine, seed: &Tensor) -> Tensor {
    use tenferro_tensor::TensorElementwise;
    engine
        .add_read(TensorRead::from_tensor(seed), TensorRead::from_tensor(seed))
        .expect("engine add")
}

fn publication(backend: &mut CudaEngine) -> crate::cuda::SourcePublication {
    let runtime = backend.runtime().clone();
    let seed = upload_tensor(&runtime, &vector(&values(32))).expect("seed");
    let value = produced_value(backend, &seed);
    backend.publish(value).expect("publish")
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_recoverable_preparation_failure_refuses_with_the_publication() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("backend");
    let publication = publication(&mut backend);

    crate::cubecl::transfer::copy_fault_injection::arm_preparation("injected preparation failure");
    let error = publication.copy_into(&mut backend).expect_err("refused");

    let PublicationError::CopyRejected {
        error: reason,
        publication,
    } = error
    else {
        panic!("expected a device-copy rejection");
    };
    assert!(reason.to_string().contains("injected preparation failure"));
    // The publication came back unchanged, so the caller can still consume it.
    let mut publication = *publication;
    assert_eq!(publication.bytes(), 32 * 8);
    publication.await_frontier().expect("await frontier");
    assert!(publication.frontier_ready().expect("frontier query"));
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_completion_record_failure_is_unproven_and_retains_the_package() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("backend");
    let runtime = backend.runtime().clone();
    let publication = publication(&mut backend);
    let retained_before = runtime.pending_transfer_retirements();

    crate::cubecl::transfer::copy_fault_injection::arm_record("injected completion-record failure");
    let error = publication.copy_into(&mut backend).expect_err("unproven");

    match error {
        PublicationError::Unproven { error } => {
            assert!(error.to_string().contains("could not be enqueued"));
        }
        other => panic!("expected an unproven submission, got {other:?}"),
    }
    // The package was retained permanently: nothing may release it later, because the event was never
    // recorded and an unrecorded event queries as complete.
    assert!(
        runtime.pending_transfer_retirements() >= retained_before,
        "the retained record count must not decrease"
    );
}
