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

/// Issue #1949: an owned raw-vendor *destination* must not use the memoized
/// read fast path.
///
/// A queued CubeCL kernel drops a buffer's memoized address only when it
/// *writes* the buffer (#1868). A queued *read* leaves the memo valid, so a
/// vendor write resolved through the memo could be issued ahead of a read
/// that precedes it in program order. Every owned write site must therefore
/// resolve through `write_device_ptr`, which always takes the blocking
/// `get_resource` round trip, and `write_device_ptr` itself must never read
/// the memo.
#[test]
fn owned_raw_vendor_writes_use_the_round_trip_path() {
    let gemm = include_str!("../gemm.rs");

    let write_body = section_between(
        gemm,
        "pub(super) fn write_device_ptr<T: TensorScalar + 'static>(",
        "\n}\n",
    );
    assert!(
        write_body.contains(".get_resource("),
        "write_device_ptr must resolve through get_resource"
    );
    for memoized in [
        "cached_device_addr",
        "memoized_device_addr",
        "typed_device_ptr",
    ] {
        assert!(
            !write_body.contains(memoized),
            "write_device_ptr must not consult the memoized read helper: {memoized}"
        );
    }

    let resolve_write = section_between(
        gemm,
        "fn resolve_write_operand",
        "/// Resolve a strided view",
    );
    let (owned, _view) = resolve_write
        .split_once("WriteOperand::View")
        .expect("resolve_write_operand must handle views");
    assert!(
        owned.contains("write_device_ptr"),
        "a cuTENSOR owned destination must resolve through write_device_ptr"
    );
    assert!(
        !owned.contains("typed_device_ptr("),
        "a cuTENSOR owned destination must not use the memoized read helper"
    );

    let blas1 = include_str!("../blas1.rs");
    let device_ptr = section_between(
        blas1,
        "fn device_ptr(&mut self, rt: &CudaRuntime, op: &'static str)",
        "\n    }\n",
    );
    let (owned, _view) = device_ptr
        .split_once("Self::View")
        .expect("WriteRef::device_ptr must handle views");
    assert!(
        owned.contains("write_device_ptr"),
        "a cuBLAS owned destination must resolve through write_device_ptr"
    );
    assert!(
        !owned.contains("typed_device_ptr("),
        "a cuBLAS owned destination must not use the memoized read helper"
    );
}

fn section_between<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let offset = source
        .find(start)
        .unwrap_or_else(|| panic!("missing section start {start:?}"));
    let rest = &source[offset..];
    let length = rest
        .find(end)
        .unwrap_or_else(|| panic!("missing section end {end:?} after {start:?}"));
    &rest[..length]
}

/// Issue #1925: a borrowed read view shares its root buffer's memoized
/// address instead of a blocking `get_resource` per operand, and still sees a
/// CubeCL write queued before it.
#[test]
#[ignore = "requires CUDA"]
fn view_reads_share_the_root_memo_and_observe_queued_writes() {
    use tenferro_tensor::{BackendSessionHost, TensorRead, TensorView};

    assert!(gpu_available(), "requires a CUDA device");
    let mut backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let rt = backend.runtime().clone();
    let values: Vec<f64> = (0..32).map(|i| i as f64 * 0.25 - 3.0).collect();
    let mut x = upload_tensor(
        &rt,
        &Tensor::from_vec_col_major(vec![32_usize], values.clone()).unwrap(),
    )
    .unwrap();
    let rhs_values: Vec<f64> = (0..6).map(|i| 1.0 + i as f64).collect();
    let rhs = upload_tensor(
        &rt,
        &Tensor::from_vec_col_major(vec![3, 2], rhs_values.clone()).unwrap(),
    )
    .unwrap();
    let config = crate::DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };

    // Queue a CubeCL write of x (x *= 2) right before the view read.
    scale_typed_tensor(
        &rt,
        x.as_typed_mut::<f64>().unwrap(),
        2.0_f64,
        |client, count, dim, out, factor| {
            // SAFETY: the scaling bridge validates residency, buffer length, and
            // the one-dimensional launch domain before this unchecked launch.
            unsafe {
                structural::scale_in_place_float_kernel::launch_unchecked::<f64, CubeclCudaRuntime>(
                    client, count, dim, out, factor,
                );
            }
        },
    )
    .unwrap();

    // lhs is the [2, 3] region at offset 5 with leading dimension 4.
    let expected: Vec<f64> = (0..4)
        .map(|index| {
            let (i, j) = (index % 2, index / 2);
            (0..3)
                .map(|k| 2.0 * values[5 + i + 4 * k] * rhs_values[k + 3 * j])
                .sum()
        })
        .collect();
    for pass in 0..2 {
        let typed = x.as_typed::<f64>().unwrap();
        let lhs = typed
            .backend_region_view(vec![2, 3], vec![1, 4], 5)
            .unwrap();
        let product = backend
            .with_backend_session(|session| {
                session.dot_general_read(
                    TensorRead::from_view(TensorView::F64(lhs)),
                    TensorRead::from_tensor(&rhs),
                    &config,
                )
            })
            .unwrap()
            .unwrap();
        let product = download_tensor(&rt, &product).unwrap();
        assert_eq!(
            product.as_slice::<f64>().unwrap(),
            expected.as_slice(),
            "pass {pass}"
        );
        // The view read memoized the root address, so the next pass skips the
        // round trip.
        assert!(
            cubecl_buffer::<f64>(x.as_typed::<f64>().unwrap(), "test")
                .unwrap()
                .cached_device_addr()
                .is_some(),
            "pass {pass}: a view read must memoize its root buffer's address"
        );
    }
}
