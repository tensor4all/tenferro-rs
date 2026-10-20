#![cfg(all(feature = "autodiff", feature = "cuda"))]

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_einsum::EagerSessionEinsumExt;
use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_runtime::Tensor;

use tenferro_einsum::{EinsumNotation, EinsumSubscripts, TensorDotAxes};
// Test helpers: each call runs the borrowed-session contraction on the engine
// the caller holds.
fn einsum(
    engine: &mut dyn tenferro_runtime::ExecutionEngine,
    inputs: &[&EagerTensor],
    subscripts: &str,
) -> tenferro_einsum::Result<EagerTensor> {
    inputs[0]
        .runtime()
        .with_eager_session(engine, |s| s.einsum(inputs, subscripts))
}

#[allow(dead_code)]
fn einsum_notation(
    engine: &mut dyn tenferro_runtime::ExecutionEngine,
    inputs: &[&EagerTensor],
    notation: &EinsumNotation,
) -> tenferro_einsum::Result<EagerTensor> {
    inputs[0]
        .runtime()
        .with_eager_session(engine, |s| s.einsum_notation(inputs, notation))
}

#[allow(dead_code)]
fn einsum_subscripts(
    engine: &mut dyn tenferro_runtime::ExecutionEngine,
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> tenferro_einsum::Result<EagerTensor> {
    inputs[0]
        .runtime()
        .with_eager_session(engine, |s| s.einsum_subscripts(inputs, subscripts))
}

#[allow(dead_code)]
fn tensordot(
    engine: &mut dyn tenferro_runtime::ExecutionEngine,
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    axes: TensorDotAxes<'_>,
) -> tenferro_einsum::Result<EagerTensor> {
    lhs.runtime()
        .with_eager_session(engine, |s| s.tensordot(lhs, rhs, axes))
}

#[test]
#[ignore]
fn cuda_eager_three_operand_einsum_stays_resident() {
    if !gpu_available() {
        eprintln!("skipping CUDA eager einsum test: no CUDA device");
        return;
    }

    let mut engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
    let runtime = EagerRuntime::new().unwrap();
    let make_input = |engine: &mut CudaEngine, shape: Vec<usize>, values: Vec<f64>| {
        let host = Tensor::from_vec_col_major(shape, values).unwrap();
        let device = upload_tensor(engine.runtime(), &host).unwrap();
        runtime
            .with_eager_session(engine, |s| s.constant_from(device))
            .unwrap()
    };
    let lhs = make_input(&mut engine, vec![2, 2, 2], vec![1.0; 8]);
    let middle = make_input(&mut engine, vec![2, 2, 2], vec![1.0; 8]);
    let rhs = make_input(&mut engine, vec![2, 2, 2], vec![1.0; 8]);

    // Three operands force the general extension path rather than the direct
    // binary-dot path.
    let output = einsum(&mut engine, &[&lhs, &middle, &rhs], "bij,bjk,bkl->bil").unwrap();

    assert_eq!(output.runtime().id(), runtime.id());
    assert_eq!(output.shape(), &[2, 2, 2]);
    let resident = output.to_tensor(&mut engine).unwrap();
    assert!(resident.as_slice::<f64>().is_err());
    let host = download_tensor(engine.runtime(), &resident).unwrap();
    assert_eq!(host.as_slice::<f64>().unwrap(), &[4.0; 8]);
}
