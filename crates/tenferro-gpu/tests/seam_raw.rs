#![cfg(feature = "cuda-next")]

//! The extension session: the addresses it hands out, the scratch it allocates, and the membership
//! check a vendor call relies on (issue #2208 U6).

use tenferro_gpu::cuda_next::{gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_tensor::Tensor;

fn engine() -> Option<CudaEngine> {
    if !gpu_available() {
        return None;
    }
    Some(CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine"))
}

fn device(engine: &CudaEngine, values: Vec<f32>) -> Tensor {
    let host = Tensor::from_vec_col_major(vec![values.len()], values).expect("host tensor");
    upload_tensor(engine.runtime(), &host).expect("upload")
}

#[test]
fn the_session_hands_out_scratch_and_tensor_addresses() {
    let Some(mut engine) = engine() else { return };
    let input = device(&engine, vec![1.0, 2.0, 3.0, 4.0]);
    let output = device(&engine, vec![0.0; 4]);

    let (scratch, read, write, retained) = engine
        .with_raw("seam_raw", |session| {
            let scratch = session.alloc_bytes(256, "seam_raw")?;
            assert_eq!(scratch.byte_len(), 256);
            let read = session.tensor(input.as_typed::<f32>().expect("f32 input"))?;
            let write = session.tensor_mut(output.as_typed::<f32>().expect("f32 output"))?;
            let retained =
                session.retain_tensor(input.as_typed::<f32>().expect("f32 input"), "seam_raw")?;
            Ok((
                scratch.with_ptr(|pointer| pointer as usize),
                read.with_ptr(|pointer| pointer as usize),
                write.with_ptr(|pointer| pointer as usize),
                retained.with_ptr(|pointer| pointer as usize),
            ))
        })
        .expect("raw session");

    assert_ne!(scratch, 0);
    assert_ne!(read, 0);
    assert_ne!(write, 0);
    // A retained tensor names the same allocation as its read view.
    assert_eq!(retained, read);
    assert_ne!(scratch, read, "scratch is its own allocation");
    assert_ne!(read, write, "the input and output are distinct allocations");
}

#[test]
fn a_tensor_of_another_engine_is_refused() {
    let Some(mut engine) = engine() else { return };
    let other = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("second engine");
    // Each engine owns its runtime, so each has its own allocation domain.
    let foreign = device(&other, vec![1.0f32, 2.0]);

    let refused = engine.with_raw("seam_raw", |session| {
        session.retain_tensor(foreign.as_typed::<f32>().expect("f32 input"), "seam_raw")
    });
    assert!(
        refused.is_err(),
        "a tensor of another runtime must be refused, not addressed"
    );
}

#[test]
fn the_runtime_runs_a_plan_setup_with_its_context_current() {
    let Some(engine) = engine() else { return };
    // A plan is created outside any scope: the context has to be current for the vendor call, and
    // the closure's failure has to come back unchanged.
    let value = engine
        .runtime()
        .with_current_context("seam_raw", || 7u32)
        .expect("context scope");
    assert_eq!(value, 7);
    // The closure's own error travels inside its value: the context wrapper only reports whether
    // the context could be activated.
    let refused = engine.runtime().with_current_context("seam_raw", || {
        Err::<u32, _>(tenferro_tensor::Error::unsupported(
            "seam_raw",
            "the vendor refused the plan",
        ))
    });
    assert!(
        matches!(refused, Ok(Err(_))),
        "the closure's failure is reported"
    );
}

#[test]
fn an_extension_reaches_the_concrete_engine_through_the_engine_trait() {
    use tenferro_gpu::cuda_next::raw::with_cuda_engine;

    let Some(mut engine) = engine() else { return };
    let domain = engine.runtime().allocation_domain_id();
    // As an extension does: it holds `&mut dyn Engine`, and reaches this engine's own surface.
    let visited = with_cuda_engine(&mut engine, |concrete| {
        concrete.runtime().allocation_domain_id()
    })
    .expect("the engine is this backend's");
    assert_eq!(visited, domain);
}
