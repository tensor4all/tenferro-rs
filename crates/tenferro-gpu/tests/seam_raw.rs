#![cfg(feature = "cuda")]

//! The extension session: the addresses it hands out, the scratch it allocates, and the membership
//! check a vendor call relies on (issue #2208 U6).

use tenferro_gpu::cuda::{gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
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
            let retained = unsafe {
                session.retain_tensor(input.as_typed::<f32>().expect("f32 input"), "seam_raw")?
            };
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

    let refused = engine.with_raw("seam_raw", |session| unsafe {
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
    use tenferro_gpu::cuda::raw::with_cuda_engine;

    let Some(mut engine) = engine() else { return };
    let domain = engine.runtime().allocation_domain_id();
    // As an extension does: it holds `&mut dyn Engine`, and reaches this engine's own surface.
    let visited = with_cuda_engine(&mut engine, |concrete| {
        concrete.runtime().allocation_domain_id()
    })
    .expect("the engine is this backend's");
    assert_eq!(visited, domain);
}

#[test]
fn a_nested_context_activation_does_not_block() {
    let Some(engine) = engine() else { return };
    // A cuFFT plan's rollback runs inside the context that created it, so activation nests. Without
    // re-entrancy tracking the inner call would wait on the submission lock the outer one holds.
    let value = engine
        .runtime()
        .with_current_context("seam_raw", || {
            engine
                .runtime()
                .with_current_context("seam_raw", || 41u32)
                .map(|inner| inner + 1)
        })
        .expect("outer context");
    assert!(matches!(value, Ok(42)), "the nested closure ran: {value:?}");
    // And a later, non-nested activation still enters the context normally.
    let later = engine.runtime().with_current_context("seam_raw", || 7u32);
    assert!(matches!(later, Ok(7)), "a later activation starts fresh");
}

#[test]
fn the_session_copies_uploads_and_downloads() {
    let Some(mut engine) = engine() else { return };
    let source = device(&engine, vec![3.0, 1.0, 2.0]);
    let destination = device(&engine, vec![0.0; 3]);

    // A device-to-device copy between two registered spans, then a download to host.
    let copied = engine
        .with_raw("seam_raw", |session| {
            let from = session.tensor(source.as_typed::<f32>().expect("f32 source"))?;
            let to = session.tensor_mut(destination.as_typed::<f32>().expect("f32 destination"))?;
            unsafe {
                session.copy_bytes(to.raw_ptr(), from.raw_ptr(), from.byte_len(), "seam_raw")?
            };
            session.download_tensor(
                destination.as_typed::<f32>().expect("f32 destination"),
                "seam_raw",
            )
        })
        .expect("raw session");
    assert_eq!(copied.as_slice().expect("host values"), &[3.0, 1.0, 2.0]);

    // An upload lands in fresh scratch the session owns.
    let uploaded = engine
        .with_raw("seam_raw", |session| {
            let scratch = session.upload_bytes(&7u64.to_ne_bytes(), "seam_raw")?;
            Ok(scratch.byte_len())
        })
        .expect("raw session");
    assert_eq!(uploaded, 8);
}

#[test]
fn a_vendor_resource_is_created_once_per_engine() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let Some(mut engine) = engine() else { return };
    let created = AtomicUsize::new(0);
    // Two sessions on the same engine share the handle: the initialiser runs once.
    for _ in 0..2 {
        let value = engine
            .with_raw("seam_raw", |session| {
                let handle = session.resource(|| {
                    created.fetch_add(1, Ordering::SeqCst);
                    Ok::<u64, tenferro_tensor::Error>(42)
                })?;
                Ok(*handle)
            })
            .expect("vendor resource");
        assert_eq!(value, 42);
    }
    assert_eq!(created.load(Ordering::SeqCst), 1);
}

#[test]
fn a_borrowed_view_canonicalizes_into_owned_storage() {
    use tenferro_gpu::cuda::download_tensor;
    let Some(mut engine) = engine() else { return };
    // A transposed view names the same allocation with swapped strides; canonicalizing it must
    // produce a compact tensor holding the transposed values.
    let host = Tensor::from_vec_col_major(vec![2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("host");
    let source = upload_tensor(engine.runtime(), &host).expect("upload");
    let transposed = {
        use tenferro_tensor::TensorStructural as _;
        engine
            .transpose_read(tenferro_tensor::TensorRead::from_tensor(&source), &[1, 0])
            .expect("transpose")
    };
    // A borrowed view of the transposed tensor, canonicalized through the entry an extension uses.
    let view = transposed.as_typed::<f32>().expect("f32 payload").as_view();
    let compact = engine.to_contiguous_view(&view).expect("canonicalize");
    let back = download_tensor(engine.runtime(), &Tensor::from_typed(compact)).expect("download");
    // Column-major `[3, 2]`: the source matrix is row0 = [1, 3, 5], row1 = [2, 4, 6].
    assert_eq!(
        back.as_slice::<f32>().expect("f32 payload"),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    );
}

#[test]
fn a_session_allocates_a_typed_output_the_extension_owns() {
    use tenferro_gpu::cuda::download_tensor;
    let Some(mut engine) = engine() else { return };
    let output = engine
        .with_raw("seam_raw", |session| session.alloc_output::<f32>(&[2, 3]))
        .expect("typed output");
    assert_eq!(output.shape(), &[2, 3]);
    let host = download_tensor(engine.runtime(), &Tensor::from_typed(output)).expect("download");
    assert_eq!(host.as_slice::<f32>().expect("f32 payload"), &[0.0; 6]);
}
