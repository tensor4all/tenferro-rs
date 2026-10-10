#![cfg(feature = "cuda")]

//! Held CUDA session contract (#1945 U3).
//!
//! These tests pin the lifecycle of [`CudaHeldSession`]: one reservation per backend state, a
//! binding the operations actually use, a lifetime-visible thread marker, unchanged extension
//! visitation, and numerics that agree with the CPU backend. They are ignored by default;
//! explicitly running them requires a CUDA device and fails if none is available.

use cubecl::stream_id::StreamId;

use tenferro_cpu::CpuBackend;
use tenferro_gpu::cuda::{
    download_tensor, gpu_available, upload_tensor, with_cuda_exec_session, CudaBackend,
    CudaDeviceId, GpuExtensionCapability,
};
use tenferro_tensor::{
    has_held_backend_session, BackendSessionHost, SessionEntryError, Tensor, TensorRead,
};

fn cuda() -> CudaBackend {
    assert!(gpu_available(), "CUDA test requires an available device");
    CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA backend")
}

fn vector(values: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(vec![values.len()], values.to_vec()).expect("vector")
}

fn payload(tensor: &Tensor) -> Vec<f64> {
    tensor.as_slice::<f64>().expect("f64 payload").to_vec()
}

#[test]
#[ignore = "requires a CUDA device"]
fn held_session_runs_a_chain_with_cpu_parity() {
    let mut backend = cuda();
    let host = vector(&[1.0, 2.0, 3.0, 4.0]);
    let a = upload_tensor(backend.runtime(), &host).expect("upload");
    let mut session = backend.open_session().expect("open");

    let doubled = session
        .with_session(|view| {
            view.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&a))
        })
        .expect("held callback")
        .expect("held add");
    let quadrupled = session
        .with_session(|view| {
            view.add_read(
                TensorRead::from_tensor(&doubled),
                TensorRead::from_tensor(&doubled),
            )
        })
        .expect("held callback")
        .expect("held add");

    let cpu = CpuBackend::default()
        .with_backend_session(|view| {
            let once = view
                .add_read(
                    TensorRead::from_tensor(&host),
                    TensorRead::from_tensor(&host),
                )
                .expect("cpu add");
            view.add_read(
                TensorRead::from_tensor(&once),
                TensorRead::from_tensor(&once),
            )
            .expect("cpu add")
        })
        .expect("cpu session");

    let stats = session.stats();
    assert_eq!(stats.held_operations, 2);
    assert_eq!(stats.explicit_submits, 0, "no submission before close");

    // A value produced inside the session stays valid after it closes, and close is the only
    // submission this chain performs.
    let closed = session.close().expect("close");
    let resident = download_tensor(backend.runtime(), &quadrupled).expect("download");
    assert_eq!(payload(&resident), vec![4.0, 8.0, 12.0, 16.0]);
    assert_eq!(payload(&resident), payload(&cpu), "CPU numerical parity");
    assert_eq!(closed.held_operations, 2);
    assert_eq!(closed.explicit_submits, 0);
    assert_eq!(closed.close_submits, 1);
}

#[test]
#[ignore = "requires a CUDA device"]
fn held_session_rejects_a_second_root_on_the_same_state() {
    let mut backend = cuda();
    let mut clone = backend.clone();
    let _session = backend.open_session().expect("first session");

    let same_handle = backend
        .open_session()
        .expect_err("second root on the same handle");
    assert!(matches!(same_handle, SessionEntryError::Reentered { .. }));

    let other_clone = clone
        .open_session()
        .expect_err("second root through a clone");
    assert!(
        matches!(other_clone, SessionEntryError::Reentered { .. }),
        "a clone shares the state, so the conflict is reentry on this thread: {other_clone:?}"
    );

    let scoped = backend
        .with_backend_session(|_| ())
        .expect_err("scoped root while held");
    assert!(matches!(scoped, SessionEntryError::Reentered { .. }));
}

#[test]
#[ignore = "requires a CUDA device"]
fn held_session_marks_the_thread_for_its_whole_lifetime() {
    let mut backend = cuda();
    assert!(!has_held_backend_session());

    let session = backend.open_session().expect("open");
    assert!(
        has_held_backend_session(),
        "the marker is visible between operations, not only inside a callback"
    );
    drop(session);
    assert!(!has_held_backend_session());

    let session = backend.open_session().expect("reopen after drop");
    assert!(has_held_backend_session());
    session.close().expect("close");
    assert!(!has_held_backend_session());
}

#[test]
#[ignore = "requires a CUDA device"]
fn held_session_releases_the_reservation_for_reuse() {
    let mut backend = cuda();
    let session = backend.open_session().expect("open");
    session.close().expect("close");
    let reopened = backend.open_session().expect("reopen after close");
    drop(reopened);
    let host = vector(&[2.0]);
    let a = upload_tensor(backend.runtime(), &host).expect("upload");
    backend
        .with_backend_session(|view| {
            view.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&a))
        })
        .expect("scoped entry after the held session ended")
        .expect("scoped add");
}

#[test]
#[ignore = "requires a CUDA device"]
fn held_session_keeps_extension_visitation_working() {
    let mut backend = cuda();
    let mut session = backend.open_session().expect("open");
    let visited = session
        .with_session(|view| {
            with_cuda_exec_session(view, |extension| {
                extension.supports(GpuExtensionCapability::CubeClKernel)
            })
        })
        .expect("held callback");
    assert!(
        visited.is_some(),
        "the held view must expose the CUDA native-session marker so extension dispatch works"
    );
    session.close().expect("close");
}

#[test]
#[ignore = "requires a CUDA device"]
fn held_session_uses_the_captured_stream_when_the_ambient_stream_changes() {
    let mut backend = cuda();
    let captured = StreamId::current();
    let mut session = backend.open_session().expect("open");

    let mut seen_inside = Vec::new();
    let mut first = Vec::new();
    session
        .with_session(|_view| first.push(StreamId::current()))
        .expect("held callback");
    let _ = &first;

    let elsewhere = StreamId {
        value: captured.value.wrapping_add(7),
    };
    elsewhere.executes(|| {
        session
            .with_session(|_view| seen_inside.push(StreamId::current()))
            .expect("held callback under a changed ambient stream");
    });
    session.close().expect("close");

    assert_eq!(
        first,
        vec![captured],
        "the session runs on its captured stream"
    );
    assert_eq!(
        seen_inside,
        vec![captured],
        "changing the ambient stream outside the session must not redirect session work"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_second_backend_state_cannot_be_opened_from_inside_a_held_callback() {
    let mut first = cuda();
    let mut second = cuda();
    let mut session = first.open_session().expect("first state");

    let nested = session
        .with_session(|_view| second.open_session().map(|_| ()))
        .expect("held callback");
    let nested = nested.expect_err("a held session already owns this thread");
    assert!(
        matches!(
            nested,
            SessionEntryError::Reentered {
                backend: "CudaBackend"
            }
        ),
        "the marker names the holding backend: {nested:?}"
    );

    session.close().expect("close");
    second
        .open_session()
        .expect("the failed attempt must not have kept the second state reserved")
        .close()
        .expect("close");
}
