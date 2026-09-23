use cubecl::stream_id::StreamId;
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;

use super::super::dispatch::{cubecl_buffer, launch_nullary_into};
use super::super::gemm::typed_device_ptr;
use super::super::interop::scale_typed_tensor;
use super::super::{cube_count_for_len, cube_dim_1d};
use crate::cuda::{download_tensor, gpu_available, upload_tensor, CudaBackend, CudaDeviceId};
use crate::kernels::structural;
use tenferro_tensor::Tensor;

/// Issue #1868: a queued CubeCL write must drop the memoized device address.
///
/// The memoized address lets a raw vendor call skip `get_resource`, which is
/// also the blocking server round trip that pushes queued kernels onto the
/// CUstream. Keeping the address across a queued write would let the vendor
/// call be issued ahead of a kernel that precedes it in program order.
#[test]
#[ignore = "requires CUDA"]
fn queued_cubecl_write_drops_the_memoized_device_address() {
    assert!(gpu_available(), "requires a CUDA device");
    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let rt = backend.runtime().clone();
    let host = Tensor::from_vec_col_major(vec![64_usize], vec![1.0_f64; 64]).unwrap();
    let x = upload_tensor(&rt, &host).unwrap();
    let typed = x.as_typed::<f64>().unwrap();

    // A raw-FFI access memoizes the address, which is what enables the fast
    // path for a later vendor call.
    typed_device_ptr(&rt, typed, "test").unwrap();
    let buffer = cubecl_buffer::<f64>(typed, "test").unwrap();
    assert_eq!(
        StreamId::current(),
        buffer.handle().stream,
        "same-stream is a precondition for the fast path"
    );
    assert!(
        buffer.cached_device_addr().is_some(),
        "a raw-FFI access should memoize the address"
    );

    // Queue a CubeCL kernel that writes the same buffer.
    launch_nullary_into(
        &rt,
        typed,
        "test",
        cube_count_for_len(typed.n_elements()).unwrap(),
        cube_dim_1d(),
        |client, count, dim, out| unsafe {
            structural::fill_zero_kernel::launch_unchecked::<f64, CubeclCudaRuntime>(
                client, count, dim, out,
            );
        },
    )
    .unwrap();

    assert!(
        cubecl_buffer::<f64>(typed, "test")
            .unwrap()
            .cached_device_addr()
            .is_none(),
        "the queued write must invalidate the memoized address so the next \
         raw-FFI access takes the `get_resource` round trip"
    );

    // Resolving again restores the fast path for subsequent read-only reuse.
    typed_device_ptr(&rt, typed, "test").unwrap();
    assert!(cubecl_buffer::<f64>(typed, "test")
        .unwrap()
        .cached_device_addr()
        .is_some());
}

/// Issue #1875: the in-place scale binding must drop the memoized device
/// address too.
///
/// `scale_typed_tensor_for_op` and `scale_typed_view` bind an existing
/// destination through `typed_view_mut_array_arg` and queue a scale kernel.
/// That is the same hazard as the audited `dispatch.rs` `launch_*` helpers, but
/// the older source-contract test scanned only helper names with a
/// `&TypedTensor` output, so it could never fail for this path.
#[test]
#[ignore = "requires CUDA"]
fn queued_scale_write_drops_the_memoized_device_address() {
    assert!(gpu_available(), "requires a CUDA device");
    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let rt = backend.runtime().clone();
    let host = Tensor::from_vec_col_major(vec![64_usize], vec![1.0_f64; 64]).unwrap();
    let mut x = upload_tensor(&rt, &host).unwrap();

    typed_device_ptr(&rt, x.as_typed::<f64>().unwrap(), "test").unwrap();
    assert!(
        cubecl_buffer::<f64>(x.as_typed::<f64>().unwrap(), "test")
            .unwrap()
            .cached_device_addr()
            .is_some(),
        "a raw-FFI access should memoize the address"
    );

    let typed = x.as_typed_mut::<f64>().unwrap();
    scale_typed_tensor(&rt, typed, 2.0_f64, |client, count, dim, out, factor| {
        // SAFETY: the scaling bridge validates residency, buffer length, and
        // the one-dimensional launch domain before this unchecked launch.
        unsafe {
            structural::scale_in_place_float_kernel::launch_unchecked::<f64, CubeclCudaRuntime>(
                client, count, dim, out, factor,
            );
        }
    })
    .unwrap();
    assert!(
        cubecl_buffer::<f64>(x.as_typed::<f64>().unwrap(), "test")
            .unwrap()
            .cached_device_addr()
            .is_none(),
        "the queued scale must invalidate the memoized address so the next \
         raw-FFI access takes the `get_resource` round trip"
    );

    let host = download_tensor(&rt, &x).unwrap();
    assert_eq!(
        host.as_slice::<f64>().unwrap(),
        vec![2.0_f64; 64].as_slice(),
        "the scale itself must still run"
    );
}

/// Issue #1868: every launch helper that writes an existing tensor must drop
/// that buffer's memoized device address.
///
/// The address is what lets a later raw vendor call skip `get_resource`, and
/// that round trip is also the barrier that pushes queued kernels onto the
/// CUstream. A new `*_into` helper added without the invalidation would
/// reopen the ordering hole silently, so pin the rule on the source rather
/// than on one behaviour.
#[test]
fn every_in_place_launch_helper_invalidates_the_memoized_address() {
    let source = include_str!("../dispatch.rs");
    let mut checked = 0usize;
    let mut offset = 0usize;
    while let Some(found) = source[offset..].find("pub(crate) fn launch_") {
        let start = offset + found;
        let end = source[start + 1..]
            .find("\npub(crate) fn ")
            .map(|idx| start + 1 + idx)
            .unwrap_or(source.len());
        let body = &source[start..end];
        let name = body["pub(crate) fn ".len()..]
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .next()
            .unwrap_or_default();
        let signature = body.split_once(") ->").map(|(sig, _)| sig).unwrap_or(body);
        // A shared reference to an output tensor means the caller already owns
        // the allocation, so a memoized address for it may exist.
        if signature.contains("output: &TypedTensor") {
            assert!(
                body.contains("invalidate_device_addr()"),
                "{name} writes an existing tensor but does not invalidate its \
                 memoized device address; see issue #1868"
            );
            checked += 1;
        }
        offset = end;
    }
    assert!(
        checked >= 4,
        "expected the in-place launch helpers to be found, saw {checked}"
    );
}

/// Issue #1875: the same rule must hold at the mutable CubeCL binding helpers.
///
/// A helper that converts a mutable tensor or view into an `ArrayArg` is about
/// to have a kernel write that buffer, whatever the caller's shape. The older
/// check keyed on `launch_*` names with a `&TypedTensor` output and therefore
/// never reached `typed_tensor_mut_array_arg` / `typed_view_mut_array_arg`,
/// which is where the in-place scale and fill paths bind their destination.
#[test]
fn every_write_binding_helper_invalidates_the_memoized_address() {
    let source = include_str!("../dispatch.rs");
    let mut checked = 0usize;
    let mut offset = 0usize;
    while let Some(found) = source[offset..].find("pub(crate) fn ") {
        let start = offset + found;
        let end = source[start + 1..]
            .find("\npub(crate) fn ")
            .map(|idx| start + 1 + idx)
            .unwrap_or(source.len());
        let body = &source[start..end];
        let name = body["pub(crate) fn ".len()..]
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .next()
            .unwrap_or_default();
        offset = end;
        // Only CubeCL array bindings can queue a scheduler-managed write.
        // `prepared_*_mut_access` returns a raw provider access instead and
        // keeps its own explicit enqueue ordering.
        if !body.contains("ArrayArg<CubeclCudaRuntime>") || !body.contains("prepare_device_write") {
            continue;
        }
        assert!(
            body.contains("invalidate_device_addr()"),
            "{name} binds a mutable CubeCL array but does not invalidate its \
             memoized device address; see issues #1868 and #1875"
        );
        checked += 1;
    }
    assert!(
        checked >= 2,
        "expected the mutable CubeCL binding helpers to be found, saw {checked}"
    );
}
