#![cfg(feature = "cuda")]

//! A scheduled CUDA run owns its nodes' completion, and the engine owns its vendor handles.
//!
//! Two per-call costs a prepared linalg program used to pay on every run, both measured on the
//! A100 host against `gpu/linalg_jvp_vjp` (#2382, immutable AD of qr/lu/solve at n=256):
//!
//! * every launch drained the stream, so a 66-launch plan paid 66 host round trips and host and
//!   device never overlapped;
//! * every run rebuilt the cuSOLVER and cuBLAS handles, because a run borrows the engine's
//!   extension cache for its duration and puts it back afterwards, so a handle created during the
//!   run was dropped with it — 17 `cudaMalloc` and 3 stream creations per call.
//!
//! Run with: `cargo test -p tenferro-linalg --features cuda --test integration
//! deferred_completion -- --ignored --include-ignored`

use tenferro_gpu::cuda::{
    cuda_runtime_engine_registration, download_tensor, gpu_available, upload_tensor, CudaDeviceId,
    CudaEngine,
};
use tenferro_linalg::{Qr, TracedLinalgOps};
use tenferro_runtime::{
    DType, ExecutionEngine, GraphCompiler, Runtime, Tensor, TensorRead, TraceContext,
};
use tenferro_tensor::TypedTensor;

/// A diagonally dominant matrix: the QR stays well conditioned at a size that still launches many
/// kernels per run.
fn well_conditioned_matrix(n: usize) -> Tensor {
    let mut values = vec![0.0f64; n * n];
    for (index, value) in values.iter_mut().enumerate() {
        let (row, column) = (index % n, index / n);
        *value = if row == column {
            8.0
        } else {
            0.5 / (1 + row + column) as f64
        };
    }
    Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![n, n], values).expect("host matrix"),
    )
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_scheduled_run_defers_launch_completion_and_keeps_vendor_handles() {
    let trace = TraceContext::new();
    assert!(gpu_available(), "CUDA test requires an available device");
    let n = 32usize;

    let mut gpu = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("open CUDA engine");
    let engine_id = gpu.runtime_engine_id().clone();
    let mut builder = Runtime::builder();
    builder
        .register_engine(cuda_runtime_engine_registration(&gpu).expect("CUDA registration"))
        .expect("register CUDA engine");
    builder
        .install_extension_module(
            tenferro_linalg::extension_module::<CudaEngine>(engine_id).expect("linalg module"),
        )
        .expect("install linalg module");
    let runtime = builder.build().expect("CUDA runtime");

    // A QR (cuSOLVER) feeding an elementwise chain: several launches in one prepared program.
    let x = trace
        .input_symbolic_shape(DType::F64, 2)
        .expect("symbolic input");
    let Qr { q, r } = x.qr().expect("qr");
    let scaled = (&q * &r).expect("multiply");
    let reduced = scaled.reduce_sum(None).expect("sum");
    let program = GraphCompiler::new()
        .compile_with_input_specs(&reduced, &[(&x, DType::F64, &[n, n])])
        .expect("compile");

    let host = well_conditioned_matrix(n);
    let device = upload_tensor(gpu.runtime(), &host).expect("upload");
    let reads = [TensorRead::from_tensor(&device)];
    let prepared = runtime
        .prepare(&program, &reads, std::slice::from_ref(&gpu.target()))
        .expect("prepare");

    let run_once = |gpu: &mut CudaEngine| -> Vec<f64> {
        let outputs = {
            let mut engines: [&mut dyn ExecutionEngine; 1] = [gpu];
            runtime.run(&mut engines, &prepared, &reads).expect("run")
        };
        // The caller's own barrier: the one wait a caller pays for the run's work.
        gpu.runtime().synchronize().expect("synchronize");
        let value = outputs
            .into_iter()
            .next()
            .expect("one output")
            .into_tensor(gpu)
            .expect("materialize");
        let host = download_tensor(gpu.runtime(), &value).expect("download");
        host.as_slice::<f64>().expect("f64 output").to_vec()
    };

    let first = run_once(&mut gpu);
    let inits_after_first = gpu.vendor_resource_inits();
    assert!(
        inits_after_first > 0,
        "the QR has to create a vendor resource for this test to mean anything"
    );

    // The strong form: count real stream barriers, not the helper that is supposed to skip them,
    // so a reintroduced `runtime().synchronize()` inside an operation fails this test too.
    let waits_before = gpu.runtime().stream_waits();
    let syncs_before = gpu.launch_completion_syncs();
    let second = run_once(&mut gpu);

    assert_eq!(
        waits_before + 1,
        gpu.runtime().stream_waits(),
        "a scheduled run must add exactly the caller's barrier, not one drain per launch"
    );
    assert_eq!(
        syncs_before,
        gpu.launch_completion_syncs(),
        "a scheduled run owns its nodes' completion; it must not drain the stream per launch"
    );
    assert_eq!(
        inits_after_first,
        gpu.vendor_resource_inits(),
        "a vendor handle has to outlive the run that created it, not be rebuilt per run"
    );
    assert_eq!(first.len(), second.len(), "both runs must produce the sum");
    assert!(
        (first[0] - second[0]).abs() <= 1e-12 * first[0].abs().max(1.0),
        "two identical runs must agree: {} vs {}",
        first[0],
        second[0]
    );
}
