#![cfg(feature = "cuda")]

//! An eager elementwise chain submits without waiting: no stream barrier per operation (#2666).
//!
//! The GPU benchmark page reported `elementwise_chain_f64_1m` at 3.2-3.4x PyTorch for the eager
//! arm against 1.03x for the traced one. Every eager operation drained the stream: a 24-launch
//! chain paid 24 `cuStreamSynchronize` calls plus the completion checkpoint each of them
//! records, and the traced path paid none because a prepared run is the barrier for its nodes
//! (#2382). This pins the counted boundary rather than the timing: a chain adds no stream
//! barrier, the caller's own barrier adds exactly one, and the chain is still numerically
//! identical to the same arithmetic on the CPU.
//!
//! Run with: `cargo test -p tenferro-ad --features cuda --test integration
//! gpu_eager_chain_no_stream_drain -- --ignored --include-ignored`

use tenferro_ad::EagerRuntime;
use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_runtime::{ExecutionEngine, Tensor, TypedTensor};

/// `tanh(t*x + b)` repeated, the shape of the benchmark's `elementwise_chain_f64_*`.
const STEPS: usize = 8;

fn f64_vector(len: usize, value: impl Fn(usize) -> f64) -> Tensor {
    Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![len], (0..len).map(value).collect())
            .expect("host vector"),
    )
}

#[test]
#[ignore = "requires a CUDA device"]
fn an_eager_chain_counts_no_stream_barrier_and_matches_the_cpu() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let (n, t, b) = (1usize << 20, 1.75f64, -0.4f64);
    let x = || f64_vector(n, |index| ((index % 97) as f64 - 48.0) / 12.0);

    let mut gpu = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("open CUDA engine");
    let ctx = EagerRuntime::new().expect("eager runtime");
    let device =
        |gpu: &mut CudaEngine, host: Tensor| upload_tensor(gpu.runtime(), &host).expect("upload");
    let x_device = device(&mut gpu, x());
    let t_device = device(&mut gpu, f64_vector(n, |_| t));
    let b_device = device(&mut gpu, f64_vector(n, |_| b));

    // The operands are registered outside the measured window: a constant is not a launch.
    let (x_value, t_value, b_value) = {
        let engine: &mut dyn ExecutionEngine = &mut gpu;
        let x_value = ctx
            .with_eager_session(&mut *engine, |session| session.constant_from(x_device))
            .expect("x");
        let t_value = ctx
            .with_eager_session(&mut *engine, |session| session.constant_from(t_device))
            .expect("t");
        let b_value = ctx
            .with_eager_session(&mut *engine, |session| session.constant_from(b_device))
            .expect("b");
        (x_value, t_value, b_value)
    };

    // One session, one chain: the operations submit and return, the host does not wait. The
    // launch count is asserted too, so a chain that stopped submitting work cannot pass this by
    // having nothing to wait for.
    let (launches_before, waits_before) = (
        gpu.runtime().stats().retirement.launches,
        gpu.runtime().stream_waits(),
    );
    let chained = ctx
        .with_eager_session(&mut gpu, |session| {
            let mut value = x_value.clone();
            for _ in 0..STEPS {
                let scaled = session.mul(&value, &t_value)?;
                let shifted = session.add(&scaled, &b_value)?;
                value = session.tanh(&shifted)?;
            }
            Ok::<_, tenferro_ad::Error>(value)
        })
        .expect("eager chain");
    assert!(
        gpu.runtime().stats().retirement.launches > launches_before,
        "the chain has to submit launches for this test to mean anything"
    );
    assert_eq!(
        gpu.runtime().stream_waits() - waits_before,
        0,
        "a {STEPS}-step eager chain must not drain the stream once per operation"
    );

    // The caller's own barrier is the one wait the chain pays; the value it then observes is the
    // chain's.
    let waits_before = gpu.runtime().stream_waits();
    gpu.runtime().synchronize().expect("synchronize");
    assert_eq!(
        gpu.runtime().stream_waits() - waits_before,
        1,
        "the caller's barrier is the only wait a chain adds"
    );

    let materialized = {
        let engine: &mut dyn ExecutionEngine = &mut gpu;
        chained
            .runtime()
            .with_eager_session(&mut *engine, |session| session.duplicate_value(&chained))
            .expect("materialize")
    };
    let observed = download_tensor(gpu.runtime(), &materialized).expect("download");
    let observed = observed.as_slice::<f64>().expect("f64 payload");

    let mut expected = x();
    {
        let values = expected.as_slice_mut::<f64>().expect("host payload");
        for _ in 0..STEPS {
            for value in values.iter_mut() {
                *value = (t * *value + b).tanh();
            }
        }
    }
    let expected = expected.as_slice::<f64>().expect("host payload");
    for (index, (gpu_value, cpu_value)) in observed.iter().zip(expected.iter()).enumerate() {
        assert!(
            (gpu_value - cpu_value).abs() <= 1e-9 * cpu_value.abs().max(1.0),
            "element {index}: CUDA {gpu_value} vs CPU {cpu_value}"
        );
    }
}
