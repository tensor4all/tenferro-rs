//! Hardware-gated end-to-end runs of the fixture kernels.

use cubecl_kernel_sample::{run_kernel_cache_check, run_scale_check, run_typing_rules_check};

#[test]
#[ignore] // requires CUDA hardware (see CI matrix: gpu tests run with --ignored)
fn scale_kernel_runs_end_to_end() {
    if !tenferro_gpu::cuda::gpu_available() {
        eprintln!("skipping: no CUDA hardware");
        return;
    }
    run_scale_check().expect("scale kernel fixture should run and verify");
}

#[test]
#[ignore] // requires CUDA hardware (see CI matrix: gpu tests run with --ignored)
fn typing_rule_kernels_run_end_to_end() {
    if !tenferro_gpu::cuda::gpu_available() {
        eprintln!("skipping: no CUDA hardware");
        return;
    }
    run_typing_rules_check().expect("typing-rule kernels should run and verify");
}

#[test]
#[ignore] // requires CUDA hardware (see CI matrix: gpu tests run with --ignored)
fn compiled_kernel_is_reused_across_scopes_and_backends() {
    if !tenferro_gpu::cuda::gpu_available() {
        eprintln!("skipping: no CUDA hardware");
        return;
    }
    run_kernel_cache_check().expect("the same kernel should launch from every scope");
}
