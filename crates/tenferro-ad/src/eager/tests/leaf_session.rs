//! Eager leaf construction must not pay a backend session for host tensors
//! (#1704).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::eager_backend::EagerBackend;
use crate::{EagerRuntime, Error};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    BackendSessionHost, MemoryKind, Placement, Tensor, TensorRead, TensorView, TypedTensor,
    TypedTensorView,
};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

use super::super::EagerTensor;

#[test]
fn host_leaf_construction_does_not_enter_a_backend_session() -> Result<(), Error> {
    let materializations = Arc::new(AtomicUsize::new(0));
    let sessions = Arc::new(AtomicUsize::new(0));
    let ctx = Arc::new(EagerRuntime::from_backend(
        EagerBackend::recording_cpu_counting_sessions(
            Arc::clone(&materializations),
            Arc::clone(&sessions),
        ),
    )?);

    let native = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])
        .map_err(Error::from)?;
    let leaf = EagerTensor::from_tensor_in(native, Arc::clone(&ctx))?;

    assert_eq!(
        sessions.load(Ordering::Relaxed),
        0,
        "a host-placement leaf must not open a backend session"
    );
    assert_eq!(
        leaf.value()?.as_slice::<f64>().map_err(Error::from)?,
        &[1.0, 2.0, 3.0, 4.0],
        "the session-free path must materialize the same value"
    );

    // The counter observes real entries: an eager op still enters a session.
    let doubled = leaf.mul(&leaf)?;
    assert!(
        sessions.load(Ordering::Relaxed) > 0,
        "an eager operation must still enter a backend session"
    );
    assert_eq!(
        doubled.value()?.as_slice::<f64>().map_err(Error::from)?,
        &[1.0, 4.0, 9.0, 16.0]
    );
    Ok(())
}

/// The eager fast path must accept and decline exactly what the CPU backend's
/// session path does, so leaf construction cannot diverge from the backend.
#[test]
fn host_leaf_materialization_matches_the_cpu_backend_acceptance() -> Result<(), Error> {
    let backend = EagerBackend::cpu(CpuBackend::new());
    let host = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])
        .map_err(Error::from)?;

    // Accepted: an owned host tensor with a preset scalar, with the same bytes
    // the CPU backend's own entry produces.
    let fast = backend
        .to_contiguous_host_read(&TensorRead::from_tensor(&host))
        .expect("an owned host tensor has a session-free path")
        .map_err(Error::from)?;
    let mut cpu = CpuBackend::new();
    let session = cpu
        .with_backend_session(|__s| __s.to_contiguous_read(TensorRead::from_tensor(&host)))
        .map_err(Error::from)?;
    assert_eq!(
        fast.as_slice::<f64>().map_err(Error::from)?,
        session.as_slice::<f64>().map_err(Error::from)?
    );

    // Declined: a view read keeps the session path.
    let data = [1.0_f64, 2.0, 3.0, 4.0];
    let view = TensorView::F64(
        TypedTensorView::from_col_major(&[2, 2], &data)
            .map_err(Error::from)?
            .transpose_view([1, 0])
            .map_err(Error::from)?,
    );
    assert!(backend
        .to_contiguous_host_read(&TensorRead::from_view(view))
        .is_none());

    // Declined: a device placement is refused by the CPU backend, so the eager
    // path must decline and let the session report that typed error.
    let mut placed =
        TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).map_err(Error::from)?;
    placed.set_placement(Placement {
        memory_kind: MemoryKind::Device,
        device: None,
        cpu_affinity: None,
    });
    let placed = Tensor::from_typed(placed);
    assert!(backend
        .to_contiguous_host_read(&TensorRead::from_tensor(&placed))
        .is_none());
    assert!(CpuBackend::new()
        .with_backend_session(|__s| __s.to_contiguous_read(TensorRead::from_tensor(&placed)))
        .is_err());

    // Declined: a caller-owned external scalar keeps the session path.
    let payload =
        HostTensor::from_vec_col_major(vec![1], vec![7.0_f64]).expect("valid host tensor");
    let external = Tensor::external(ErasedHostTensor::new(payload));
    assert!(backend
        .to_contiguous_host_read(&TensorRead::from_tensor(&external))
        .is_none());
    Ok(())
}
