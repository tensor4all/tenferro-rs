//! `Runtime::warm_up` pays a prepared program's first execution before the caller measures.
//!
//! Run with: `cargo test -p tenferro-gpu --features cuda --test warm_up -- --ignored`

#![cfg(feature = "cuda")]

use tenferro_gpu::cuda::{
    cuda_runtime_engine_registration, gpu_available, upload_tensor, CudaDeviceId, CudaEngine,
};
use tenferro_runtime::{
    DType, ExecutionEngine, GraphCompiler, Runtime, Tensor, TensorRead, TracedTensor, TypedTensor,
};
use tenferro_tensor::TensorDeviceTransfer;

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn runtime_warm_up_runs_the_program_and_leaves_it_runnable() {
    assert!(gpu_available(), "CUDA test requires an available device");

    let mut engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("open CUDA engine");
    let mut builder = Runtime::builder();
    builder
        .register_engine(cuda_runtime_engine_registration(&engine).expect("register engine"))
        .expect("register engine");
    let runtime = builder.build().expect("build runtime");

    // An elementwise chain plus a reduction: two kernel families, both compiled on first launch.
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).expect("symbolic input");
    let squared = (&x * &x).expect("square");
    let shifted = (&squared + &x).expect("shift");
    let summed = shifted.reduce_sum(None).expect("sum all");
    let program = GraphCompiler::new()
        .compile_with_input_specs(&summed, &[(&x, DType::F64, &[64])])
        .expect("compile");

    let values: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let host = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![values.len()], values.clone()).unwrap(),
    );
    let device = upload_tensor(engine.runtime(), &host).expect("upload");
    let reads = [TensorRead::from_tensor(&device)];
    let prepared = runtime
        .prepare(&program, &reads, std::slice::from_ref(&engine.target()))
        .expect("prepare");

    // Warm-up runs the program once and waits; the following identical run must
    // produce the same value.
    runtime
        .warm_up(&mut [&mut engine], &prepared, &reads)
        .expect("warm up");
    let outputs = runtime
        .run(&mut [&mut engine], &prepared, &reads)
        .expect("run after warm-up");
    assert_eq!(outputs.len(), 1);
    let value = outputs
        .into_iter()
        .next()
        .unwrap()
        .into_tensor(&mut engine)
        .expect("materialize the output");
    let host = engine
        .download_to_host(TensorRead::from_tensor(&value))
        .expect("download");
    let got = host.as_slice::<f64>().expect("downloaded f64 scalar");
    let want: f64 = values.iter().map(|v| v * v + v).sum();
    assert_eq!(got.len(), 1);
    assert!(
        (got[0] - want).abs() <= 1e-9 * want.abs().max(1.0),
        "warm-up changed the result: got {}, want {want}",
        got[0]
    );
}
