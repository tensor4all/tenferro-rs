//! A single-domain scheduled run records no per-node completion (#2396).
//!
//! Run with: `cargo test -p tenferro-gpu --features cuda --test coalesced_run -- --ignored`

#![cfg(feature = "cuda")]

use tenferro_gpu::cuda::{
    cuda_runtime_engine_registration, gpu_available, CudaDeviceId, CudaEngine,
};
use tenferro_runtime::{DType, ExecutionEngine, GraphCompiler, Runtime, TensorRead, TracedTensor};

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn a_single_domain_run_keeps_no_per_node_completion() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("open CUDA engine");
    let mut builder = Runtime::builder();
    builder
        .register_engine(cuda_runtime_engine_registration(&engine).expect("register engine"))
        .expect("register engine");
    let runtime = builder.build().expect("build runtime");

    // Four elementwise nodes plus a reduction: several launches in one domain, no transfer.
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).expect("symbolic input");
    let squared = (&x * &x).expect("square");
    let shifted = (&squared + &x).expect("shift");
    let scaled = (&shifted * &shifted).expect("scale");
    let summed = scaled.reduce_sum(None).expect("sum");
    let program = GraphCompiler::new()
        .compile_with_input_specs(&summed, &[(&x, DType::F64, &[64])])
        .expect("compile");

    let values: Vec<f64> = (0..64).map(|i| i as f64 * 0.25).collect();
    let host = tenferro_runtime::Tensor::from_typed::<f64>(
        tenferro_runtime::TypedTensor::from_vec_col_major(vec![values.len()], values.clone())
            .expect("host vector"),
    );
    let device = tenferro_gpu::cuda::upload_tensor(engine.runtime(), &host).expect("upload");
    let reads = [TensorRead::from_tensor(&device)];
    let prepared = runtime
        .prepare(&program, &reads, std::slice::from_ref(&engine.target()))
        .expect("prepare");

    let run = |engine: &mut CudaEngine| {
        let outputs = {
            let mut engines: [&mut dyn ExecutionEngine; 1] = [engine];
            runtime.run(&mut engines, &prepared, &reads).expect("run")
        };
        engine.runtime().synchronize().expect("synchronize");
        outputs
            .into_iter()
            .next()
            .expect("one output")
            .into_tensor(engine)
            .expect("materialize")
    };

    let _warm = run(&mut engine);
    let before = engine.runtime().completion_checkpoints();
    let value = run(&mut engine);

    assert_eq!(
        before,
        engine.runtime().completion_checkpoints(),
        "a run whose nodes share one stream must not record a completion per node"
    );
    let host = tenferro_gpu::cuda::download_tensor(engine.runtime(), &value).expect("download");
    let got = host.as_slice::<f64>().expect("f64 output")[0];
    let want: f64 = values.iter().map(|v| (v * v + v).powi(2)).sum();
    assert!(
        (got - want).abs() <= 1e-9 * want.abs().max(1.0),
        "coalesced run changed the result: got {got}, want {want}"
    );
}
