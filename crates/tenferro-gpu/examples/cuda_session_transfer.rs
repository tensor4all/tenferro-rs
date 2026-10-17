//! Holding one CUDA session across a stage, and moving pinned data without a host wait.
//!
//! Run with `cargo run -p tenferro-gpu --features cuda --example cuda_session_transfer`.
//!
//! The first half keeps one session open for a chain of operations and pays the session entry once;
//! the second half moves host↔device data through caller-owned pinned buffers with a pending handle
//! whose completion the caller polls. Both are the CUDA counterparts of the CPU held session in
//! `tenferro-cpu`'s `held_session_driver` example; see `docs/guides/cuda-sessions-and-transfers.md`
//! for the rules each one keeps.

use tenferro_gpu::cuda::{
    cuda_devices, download_pending, upload_pending, upload_tensor, CudaBackend, PinnedHostBuffer,
};
use tenferro_tensor::{Tensor, TensorRead};

fn vector(values: &[f64]) -> Result<Tensor, Box<dyn std::error::Error>> {
    Ok(Tensor::from_vec_col_major(
        vec![values.len()],
        values.to_vec(),
    )?)
}

fn as_bytes(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn from_bytes(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_ne_bytes(*chunk))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(device) = cuda_devices()?.into_iter().next() else {
        println!("no CUDA device available");
        return Ok(());
    };
    let mut backend = CudaBackend::new(device.id())?;
    let runtime = backend.runtime().clone();

    // A stage: one session entry, several dependent operations, one explicit submit.
    let seed = upload_tensor(&runtime, &vector(&[1.0, 2.0, 3.0, 4.0])?)?;
    // snippet-start:cuda_held_session
    let mut session = backend.open_session()?;
    let mut state = session.with_session(|view| {
        view.add_read(
            TensorRead::from_tensor(&seed),
            TensorRead::from_tensor(&seed),
        )
    })??;
    for _ in 0..2 {
        state = session.with_session(|view| {
            view.add_read(
                TensorRead::from_tensor(&state),
                TensorRead::from_tensor(&state),
            )
        })??;
    }
    session.submit()?;
    let stats = session.close()?;
    // snippet-end:cuda_held_session
    assert_eq!(stats.held_operations, 3);
    println!(
        "held session: {} operations, {} submits, {} syncs",
        stats.held_operations, stats.close_submits, stats.explicit_synchronizes
    );

    // A prepared compiled graph runs on the same held session, so steady-state replay does not
    // re-open a scoped session per call.
    // snippet-start:cuda_prepared_in_held
    let mut prepared_builder = tenferro_runtime::Runtime::builder();
    prepared_builder.register_engine(tenferro_gpu::cuda::cuda_runtime_engine_registration(
        &backend,
        tenferro_runtime::EngineId::new("tenferro.example.held-prepared.v1")?,
    )?)?;
    let prepared_runtime = prepared_builder.build()?;
    let x = tenferro_runtime::TracedTensor::input_symbolic_shape(tenferro_runtime::DType::F64, 1)?;
    let y = (&x * &x)?;
    let program = tenferro_runtime::GraphCompiler::new()
        .compile_with_input_specs(&y, &[(&x, tenferro_runtime::DType::F64, &[4])])?;
    let prepared = prepared_runtime.prepare_compiled(&program, &[&seed])?;
    let prepared_inputs: [&Tensor; 1] = [&seed];
    let mut prepared_session = backend.open_session()?;
    let outputs = prepared_session.with_session(|view| {
        prepared_runtime.run_prepared_in(view, &prepared, &prepared_inputs)
    })??;
    prepared_session.close()?;
    assert_eq!(outputs[0].shape(), &[4]);
    // snippet-end:cuda_prepared_in_held

    // Owning transfers: the value and the buffer are moved in and come back from `wait`, so neither is
    // reachable while the copy is in flight.
    // snippet-start:cuda_pinned_transfer
    let host = vec![2.0, 4.0, 6.0, 8.0];
    let bytes = as_bytes(&host);
    let destination = upload_tensor(&runtime, &vector(&[0.0; 4])?).expect("device destination");
    let mut source = PinnedHostBuffer::new(&runtime, bytes.len())?;
    source.as_mut_slice().copy_from_slice(&bytes);
    let mut pending_upload = upload_pending(&runtime, source, destination)?;
    let mut observed_in_flight = false;
    for _ in 0..64 {
        if !pending_upload.is_ready()? {
            observed_in_flight = true;
            break;
        }
    }
    let (destination, reusable) = pending_upload.wait()?;
    // A 32-byte copy can finish before the first poll, so either answer is correct here; the
    // point is that `is_ready` never waits and the values are only handed over by `wait`.
    println!("pinned upload: observed in flight before wait = {observed_in_flight}");

    let buffer = PinnedHostBuffer::new(&runtime, bytes.len())?;
    let (returned, filled) = download_pending(&runtime, destination, buffer)?.wait()?;
    assert_eq!(from_bytes(filled.as_slice()), host);
    assert_eq!(reusable.as_slice(), filled.as_slice());
    // snippet-end:cuda_pinned_transfer
    assert_eq!(returned.shape(), &[4]);
    println!("pending download: {:?}", from_bytes(filled.as_slice()));
    Ok(())
}
