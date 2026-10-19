use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::{gpu_available, is_invalid_device_lookup, CudaRuntime, CudaRuntimeIdentity};
use crate::cuda::CudaEngine;
use cudarc::driver::{result::DriverError, sys::CUresult};

#[test]
fn selected_device_lookup_classifies_only_cuda_invalid_device() {
    assert!(is_invalid_device_lookup(DriverError(
        CUresult::CUDA_ERROR_INVALID_DEVICE
    )));
    assert!(!is_invalid_device_lookup(DriverError(
        CUresult::CUDA_ERROR_INVALID_VALUE
    )));
}

fn identity_hash(identity: &CudaRuntimeIdentity) -> u64 {
    let mut hasher = DefaultHasher::new();
    identity.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn cuda_runtime_identity_is_clone_stable_and_instance_scoped() {
    let first = CudaRuntimeIdentity::fresh();
    let first_key = identity_hash(&first);
    let clone = first.clone();
    let moved = clone;
    let independent = CudaRuntimeIdentity::fresh();

    assert_eq!(first, moved);
    assert_eq!(first_key, identity_hash(&moved));
    assert_ne!(first, independent);
}

#[test]
#[ignore = "requires CUDA"]
fn cuda_backend_identity_tracks_the_exact_runtime_when_hardware_is_available() {
    assert!(gpu_available(), "CUDA test requires an available device");

    let device = super::cuda_devices()
        .expect("CUDA device discovery should succeed")
        .into_iter()
        .next()
        .expect("CUDA device should be available")
        .id();
    let first = CudaEngine::open(device).expect("CUDA engine should initialize");
    let independent = CudaEngine::open(device).expect("second CUDA engine should initialize");

    // Every opened engine is its own runtime instance; its identity is stable.
    let first_identity = first.runtime_identity();
    let again = first.runtime_identity();
    let independent_identity = independent.runtime_identity();
    assert_eq!(first_identity, again);
    assert_eq!(identity_hash(&first_identity), identity_hash(&again));
    assert_ne!(first_identity, independent_identity);

    let runtime_clone = first.runtime().clone();
    let _: CudaRuntime = runtime_clone;
}

/// Build a runtime for the staging-lease tests.
fn staging_runtime() -> CudaRuntime {
    assert!(gpu_available(), "CUDA test requires an available device");
    let ordinal =
        cudarc::driver::result::device::get(0).expect("CUDA device lookup should succeed");
    let ordinal = u32::try_from(ordinal).expect("a CUDA device ordinal fits in u32");
    let device = crate::cuda::CudaDeviceId::from_ordinal(ordinal);
    CudaRuntime::new(device).expect("CUDA runtime should initialize")
}

/// An event that is never recorded: the case a failed record leaves behind, which must never be
/// destroyed or trusted to authorize a release.
fn unrecorded_event() -> cudarc::driver::sys::CUevent {
    cudarc::driver::result::event::create(
        cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
    )
    .expect("CUDA event creation should succeed")
}

#[test]
#[ignore = "requires CUDA"]
fn a_pinned_staging_lease_is_taken_out_of_the_slot_and_reused_once_proven() {
    let runtime = staging_runtime();

    // Two live leases never share one buffer: the lease takes it out of the slot.
    let mut first = runtime.acquire_pinned_staging().expect("first lease");
    let second = runtime.acquire_pinned_staging().expect("second lease");
    let first_address = first.buffer().address();
    assert_ne!(
        first_address,
        second.buffer().address(),
        "a lease must not hand out a buffer another lease already holds"
    );
    assert_eq!(first.len(), super::STAGING_CHUNK_BYTES);

    // A proven chunk returns the buffer, and the next lease reuses that same allocation.
    first.end_chunk();
    first.release();
    let reused = runtime.acquire_pinned_staging().expect("reused lease");
    assert_eq!(
        reused.buffer().address(),
        first_address,
        "a proven lease returns its buffer to the slot for the next transfer"
    );
    reused.release();
    second.release();
}

#[test]
#[ignore = "requires CUDA"]
fn an_unproven_pinned_staging_lease_is_retained_and_the_slot_stays_empty() {
    let runtime = staging_runtime();
    let before = runtime.pending_transfer_retirements();

    let mut lease = runtime.acquire_pinned_staging().expect("lease");
    let retired_address = lease.buffer().address();
    // A first chunk that completed, then a second that did not: the reused event's earlier success must
    // not be readable as the second chunk's completion, so the lease is retired, not released.
    lease.begin_chunk();
    lease.end_chunk();
    lease.begin_chunk();
    lease.retire(unrecorded_event(), None);

    assert_eq!(
        runtime.pending_transfer_retirements(),
        before + 1,
        "an unproven package is retained for soundness"
    );
    let replacement = runtime.acquire_pinned_staging().expect("replacement lease");
    assert_ne!(
        replacement.buffer().address(),
        retired_address,
        "a retired buffer must never be handed to another transfer"
    );
    replacement.release();
}

#[test]
#[ignore = "requires CUDA"]
fn dropping_an_unproven_pinned_staging_lease_leaks_it_in_place() {
    let runtime = staging_runtime();
    let before = runtime.pending_transfer_retirements();

    let mut lease = runtime.acquire_pinned_staging().expect("lease");
    let dropped_address = lease.buffer().address();
    // Unwinding between an enqueue and its proven wait: the buffer may still be part of a DMA.
    lease.begin_chunk();
    drop(lease);

    assert_eq!(
        runtime.pending_transfer_retirements(),
        before,
        "unwinding leaks the lease in place instead of retiring it"
    );
    let replacement = runtime.acquire_pinned_staging().expect("replacement lease");
    assert_ne!(
        replacement.buffer().address(),
        dropped_address,
        "a leaked buffer must never be handed to another transfer"
    );
    replacement.release();
}

/// Whether `address` is still a live pinned host allocation of this device.
fn is_pinned_host_allocation(address: usize) -> bool {
    let mut flags = 0u32;
    // SAFETY: the query only reads the allocation's flags.
    unsafe { cudarc::runtime::sys::cudaHostGetFlags(&mut flags, address as *mut _) }
        .result()
        .is_ok()
}

#[test]
#[ignore = "requires CUDA"]
fn runtime_teardown_releases_the_cached_staging_buffer() {
    let runtime = staging_runtime();
    let lease = runtime.acquire_pinned_staging().expect("lease");
    let address = lease.buffer().address();
    lease.release();
    assert!(
        is_pinned_host_allocation(address),
        "the slot holds the released buffer"
    );

    // The teardown path is the one a dropping runtime takes while its primary context is still
    // retained. Calling it directly is what makes the release observable: the buffer's own `Drop`
    // cannot upgrade a weak runtime reference during the state's own destruction, so a release that
    // goes through the context authority has to be visible here.
    runtime.inner.release_cuda_library_resources();
    assert!(
        !is_pinned_host_allocation(address),
        "teardown must free the cached staging buffer, not leak it"
    );
}
