// INVARIANT: The parent `#[cfg(test)] mod tests;` declaration in
// `cuda/mod.rs` is the sole inclusion point for this test module.
#[path = "tests/descriptor.rs"]
mod descriptor;
#[path = "tests/ffi.rs"]
mod ffi;
#[path = "tests/plan.rs"]
mod plan;
#[path = "tests/source_contract.rs"]
mod source_contract;

#[test]
fn placement_error_explains_explicit_transfer_and_runtime_ownership() {
    let error = super::CudaFftPlacementError {
        source: tenferro_tensor::Error::runtime_state("cuda_fft", "residency mismatch"),
    };
    assert_eq!(
        error.to_string(),
        "CUDA FFT expected GPU tensor owned by this runtime; host tensors must use ".to_string()
            + "upload_tensor(), and foreign-runtime tensors must use their owning "
            + "runtime: cuda_fft: runtime state failure: residency mismatch"
    );
}
