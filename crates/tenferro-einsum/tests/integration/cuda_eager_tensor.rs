#![cfg(all(feature = "autodiff", feature = "cuda"))]

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_einsum::{EagerEinsumOps, TracedEinsumOps};
use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_runtime::program::ProgramInputSpec;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, Tensor, TraceContext};
use tenferro_tensor::{DType, TensorRead};

use tenferro_einsum::{EinsumNotation, EinsumSubscripts, TensorDotAxes};
// Test helpers: each call runs the borrowed-session contraction on the engine
// the caller holds.
fn einsum(
    engine: &mut dyn tenferro_runtime::ExecutionEngine,
    inputs: &[&EagerTensor],
    subscripts: &str,
) -> tenferro_ad::Result<EagerTensor> {
    inputs[0]
        .runtime()
        .with_eager_session(engine, |s| s.einsum(inputs, subscripts))
}

#[allow(dead_code)]
fn einsum_notation(
    engine: &mut dyn tenferro_runtime::ExecutionEngine,
    inputs: &[&EagerTensor],
    notation: &EinsumNotation,
) -> tenferro_ad::Result<EagerTensor> {
    inputs[0]
        .runtime()
        .with_eager_session(engine, |s| s.einsum(inputs, notation))
}

#[allow(dead_code)]
fn einsum_subscripts(
    engine: &mut dyn tenferro_runtime::ExecutionEngine,
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> tenferro_ad::Result<EagerTensor> {
    inputs[0]
        .runtime()
        .with_eager_session(engine, |s| s.einsum(inputs, subscripts))
}

#[allow(dead_code)]
fn tensordot(
    engine: &mut dyn tenferro_runtime::ExecutionEngine,
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    axes: TensorDotAxes<'_>,
) -> tenferro_ad::Result<EagerTensor> {
    lhs.runtime()
        .with_eager_session(engine, |s| s.tensordot(lhs, rhs, axes))
}

// Regression for tenferro-benchmark#291: ownership copies inside einsum must
// use the executing engine, not host-only Tensor::duplicate.
#[test]
#[ignore]
fn cuda_einsum_ownership_copies_stay_resident() {
    let trace = TraceContext::new();
    assert!(gpu_available(), "this test requires a CUDA device");
    let mut engine = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_gpu::cuda::cuda_runtime_engine_registration(&engine).unwrap())
        .unwrap();
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<CudaEngine>(engine.runtime_engine_id().clone())
                .unwrap(),
        )
        .unwrap();
    let runtime = builder.build().unwrap();
    let eager_runtime = EagerRuntime::new().unwrap();
    let matrix = || Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let cases = [
        ("ij->ij", vec![matrix()], vec![1., 2., 3., 4.]),
        ("ij->ji", vec![matrix()], vec![1., 3., 2., 4.]),
        ("ij->i", vec![matrix()], vec![4., 6.]),
        ("ii->i", vec![matrix()], vec![1., 4.]),
        (
            "i->ii",
            vec![Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap()],
            vec![1., 0., 0., 2.],
        ),
        (
            "abc,cde->adbe",
            vec![
                Tensor::from_vec_col_major(vec![2, 2, 2], (1..=8).map(|x| x as f32).collect())
                    .unwrap(),
                Tensor::from_vec_col_major(vec![2, 2, 2], (9..=16).map(|x| x as f32).collect())
                    .unwrap(),
            ],
            vec![
                59., 78., 71., 94., 97., 116., 117., 140., 83., 110., 95., 126., 137., 164., 157.,
                188.,
            ],
        ),
    ];
    for (expression, hosts, expected) in cases {
        let trace = TraceContext::new();
        let values = hosts
            .iter()
            .map(|host| {
                let value = trace
                    .input(ProgramInputSpec::new(
                        DType::F32,
                        host.shape().iter().copied().map(Into::into),
                    ))
                    .unwrap();
                trace.derive_tensor(value).unwrap()
            })
            .collect::<Vec<_>>();
        let refs = values.iter().collect::<Vec<_>>();
        let output = trace.einsum(&refs, expression).unwrap();
        let graph = trace.finish(&[output.trace_value()]).unwrap();
        let program = GraphCompiler::new().compile_traced_graph(&graph).unwrap();
        let host_refs = hosts.iter().collect::<Vec<_>>();
        let cpu = super::support::run_one(&program, &host_refs).unwrap();
        assert_eq!(cpu.as_slice::<f32>().unwrap(), expected, "CPU {expression}");
        let devices = hosts
            .iter()
            .map(|host| upload_tensor(engine.runtime(), host).unwrap())
            .collect::<Vec<_>>();
        let reads = devices
            .iter()
            .map(TensorRead::from_tensor)
            .collect::<Vec<_>>();
        let target = engine.target();
        let prepared = runtime.prepare(&program, &reads, &[target]).unwrap();
        let outputs = runtime
            .run(&mut [&mut engine], &prepared, &reads)
            .unwrap_or_else(|error| panic!("trace {expression}: {error}"));
        let traced = outputs
            .into_iter()
            .next()
            .unwrap()
            .into_tensor(&mut engine)
            .unwrap();
        assert_eq!(traced.shape(), cpu.shape(), "trace {expression}");
        assert!(
            traced.as_slice::<f32>().is_err(),
            "trace {expression} left CUDA"
        );
        let downloaded = download_tensor(engine.runtime(), &traced).unwrap();
        assert_eq!(
            downloaded.as_slice::<f32>().unwrap(),
            expected,
            "trace {expression}"
        );
        let eager_inputs = devices
            .into_iter()
            .map(|device| {
                eager_runtime
                    .with_eager_session(&mut engine, |s| s.constant_from(device))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let eager_refs = eager_inputs.iter().collect::<Vec<_>>();
        let eager = einsum(&mut engine, &eager_refs, expression).unwrap();
        let resident = eager
            .runtime()
            .with_eager_session(&mut engine, |s| s.duplicate_value(&eager))
            .unwrap();
        assert_eq!(resident.shape(), cpu.shape(), "eager {expression}");
        assert!(
            resident.as_slice::<f32>().is_err(),
            "eager {expression} left CUDA"
        );
        let downloaded = download_tensor(engine.runtime(), &resident).unwrap();
        assert_eq!(
            downloaded.as_slice::<f32>().unwrap(),
            expected,
            "eager {expression}"
        );
    }
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
    let resident = output
        .runtime()
        .with_eager_session(&mut engine, |s| s.duplicate_value(&output))
        .unwrap();
    assert!(resident.as_slice::<f64>().is_err());
    let host = download_tensor(engine.runtime(), &resident).unwrap();
    assert_eq!(host.as_slice::<f64>().unwrap(), &[4.0; 8]);
}
