//! Hardware-gated end-to-end run of the fixture kernel.

use cubecl_kernel_sample::run_scale_check;

#[test]
#[ignore] // requires CUDA hardware (see CI matrix: gpu tests run with --ignored)
fn scale_kernel_runs_end_to_end() {
    if !tenferro_gpu::cuda::gpu_available() {
        eprintln!("skipping: no CUDA hardware");
        return;
    }
    run_scale_check().expect("scale kernel fixture should run and verify");
}
