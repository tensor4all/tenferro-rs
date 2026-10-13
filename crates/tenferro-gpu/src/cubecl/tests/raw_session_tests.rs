//! Raw-session contract tests (issue #1597).
//!
//! These tests require CUDA hardware and are therefore ignored by default,
//! matching the regular CUDA test convention in this crate.

use crate::cubecl::CudaRuntime;
use crate::cuda::{gpu_available, CudaBackend, GpuExtensionCapability};

use super::*;

fn first_cuda_backend() -> Option<CudaBackend> {
    let devices = crate::cuda::cuda_devices().ok()?;
    let device = devices.first()?;
    CudaBackend::new(device.id()).ok()
}

#[test]
#[ignore = "requires CUDA"]
fn raw_session_exposes_stream_and_runtime_identity() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        let identity = session.runtime_identity();
        session
            .with_raw("test.raw_identity", |raw| {
                assert_eq!(raw.runtime_identity(), identity);
                let _stream = raw.stream();
                Ok(())
            })
            .unwrap();
    });
    // Context/device should be best-effort restored after the session; a
    // subsequent backend operation that needs the primary context must still
    // work when restoration succeeds.
    assert!(session_after_raw_still_runs(&mut backend));
}

fn session_after_raw_still_runs(backend: &mut CudaBackend) -> bool {
    backend.runtime().synchronize().is_ok()
}

#[test]
#[ignore = "requires CUDA"]
fn raw_session_allocates_output_and_bytes() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_alloc", |raw| {
                let _output = raw.alloc_output::<f32>(&[4])?;
                let bytes = raw.alloc_bytes(1024, "test.raw_alloc")?;
                assert!(!bytes.is_empty());
                Ok(())
            })
            .unwrap();
    });
}

#[test]
#[ignore = "requires CUDA"]
fn raw_session_reports_capabilities() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        assert!(session.supports(GpuExtensionCapability::CubeClKernel));
        assert!(session.supports(GpuExtensionCapability::NativeModule));
        assert!(session.supports(GpuExtensionCapability::RuntimeCompilation));
        assert!(session.supports(GpuExtensionCapability::RawStream));
        assert!(session.supports(GpuExtensionCapability::SameDeviceAsyncCopy));
        // Peer copy is directional/hardware-dependent at the provider level.
        assert!(!session.supports(GpuExtensionCapability::PeerCopy));
    });
}

#[test]
#[ignore = "requires CUDA"]
fn raw_session_restores_context_on_error_path_observably() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        let result: tenferro_tensor::Result<()> = session.with_raw("test.raw_error", |_raw| {
            Err(tenferro_tensor::Error::runtime_state(
                "test.raw_error",
                "intentional failure",
            ))
        });
        assert!(result.is_err());
    });
    // The primary context must still be restorable afterwards.
    assert!(session_after_raw_still_runs(&mut backend));
}

#[test]
#[ignore = "requires CUDA"]
fn raw_tensor_ref_carries_validated_span() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    let host = tensor_f32(vec![8], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let gpu = upload(&backend, &host);
    let Some(gpu_typed) = gpu.as_typed::<f32>() else {
        unreachable!("f32 tensor")
    };
    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_tensor", |raw| {
                let reference = raw.tensor(gpu_typed)?;
                assert_eq!(reference.byte_len(), 8 * std::mem::size_of::<f32>());
                Ok(())
            })
            .unwrap();
    });
}

#[test]
#[ignore = "requires CUDA"]
fn raw_retain_tensor_pins_the_allocation_across_a_drop() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    // Clone the runtime handle so the inner scope can upload without
    // re-borrowing `backend` while the mutable session borrow is live.
    let rt = backend.runtime().clone();
    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_retain_tensor", |raw| {
                // Build the tensor inside a scope so it is dropped while the
                // retained guard below stays live.
                let (saved_ptr, retained) = {
                    let host = tensor_f32(vec![4], vec![1.0f32, 2.0, 3.0, 4.0]);
                    let gpu = upload_tensor(&rt, &host).unwrap();
                    let Some(gpu_typed) = gpu.as_typed::<f32>() else {
                        unreachable!("f32 tensor")
                    };
                    let reference = raw.tensor(gpu_typed)?;
                    // SAFETY: `reference` is a validated span for `gpu_typed`;
                    // copying the pointer value is read-only and the retained
                    // guard keeps the allocation alive past the drop below.
                    let saved_ptr = unsafe { reference.raw_ptr() };
                    let retained = raw.retain_tensor(gpu_typed, "test.raw_retain_tensor")?;
                    (saved_ptr, retained)
                    // `gpu` (and its owning handle refcount) is dropped here.
                };
                let mut retained_ptr = std::ptr::null_mut();
                retained.with_ptr(|ptr| retained_ptr = ptr);
                assert_eq!(
                    saved_ptr, retained_ptr,
                    "retention must keep the allocation alive across the drop"
                );
                assert!(!retained.is_empty());
                // Prove liveness, not just pointer identity: copy the retained
                // allocation into a freshly allocated probe and read it back
                // after the owning tensor is gone. If the guard had let the
                // allocation be reclaimed, the readback would not reproduce
                // the original payload.
                let probe = raw.alloc_output::<f32>(&[4])?;
                let probe_ref = raw.tensor(&probe)?;
                // SAFETY: `dst` is the freshly allocated, uniquely owned probe
                // span; `src` is the retained allocation still pinned by the
                // guard. The copy is stream-ordered, and `synchronize` below
                // completes it before the host readback.
                unsafe {
                    raw.copy_bytes(
                        probe_ref.raw_ptr(),
                        retained_ptr,
                        4 * std::mem::size_of::<f32>(),
                        "test.raw_retain_tensor",
                    )?;
                }
                raw.synchronize()?;
                let back = download_tensor(&rt, &Tensor::from_typed::<f32>(probe)).unwrap();
                assert_eq!(
                    back.as_slice::<f32>().unwrap(),
                    &[1.0f32, 2.0, 3.0, 4.0],
                    "retained allocation must still hold the original payload"
                );
                Ok(())
            })
            .unwrap();
    });
}

#[test]
#[ignore = "requires CUDA"]
fn raw_retain_tensor_rejects_tensor_from_another_runtime() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    // Upload on a separate runtime instance: its allocation domain differs
    // from the session runtime's, so retention must be rejected.
    let foreign_rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = tensor_f32(vec![4], vec![1.0f32, 2.0, 3.0, 4.0]);
    let gpu = upload_tensor(&foreign_rt, &host).unwrap();
    let Some(gpu_typed) = gpu.as_typed::<f32>() else {
        unreachable!("f32 tensor")
    };
    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_retain_foreign", |raw| {
                let err = raw
                    .retain_tensor(gpu_typed, "test.raw_retain_foreign")
                    .unwrap_err();
                assert!(
                    matches!(err, crate::Error::RuntimeState { .. }),
                    "foreign-runtime tensor must be rejected: {err}"
                );
                Ok(())
            })
            .unwrap();
    });
}

#[test]
#[ignore = "requires CUDA"]
fn raw_resource_guard_is_runtime_scoped_and_type_keyed() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_resource", |raw| {
                let guard = raw.resource(|| Ok(String::from("cached-value")))?;
                assert_eq!(&**guard, "cached-value");
                Ok(())
            })
            .unwrap();
    });
}

/// Entering a raw scope must not be a host barrier (tenferro-rs #2078).
///
/// A downstream single-scope forward interleaves fused raw kernels with eager
/// ops. If every scope blocked until queued device work finished, that pattern
/// could not be used at all: the host would pay the whole device time on entry.
#[test]
#[ignore = "requires CUDA"]
fn raw_session_entry_is_not_a_host_barrier() {
    use std::time::{Duration, Instant};

    use crate::DotGeneralConfig;
    use tenferro_tensor::TensorRead;

    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    let runtime = backend.runtime().clone();
    let n = 4096usize;
    let lhs = crate::cuda::upload_tensor(&runtime, &tensor_f32(vec![n, n], vec![1.0f32; n * n]))
        .expect("upload lhs");
    let rhs = crate::cuda::upload_tensor(&runtime, &tensor_f32(vec![n, n], vec![1.0f32; n * n]))
        .expect("upload rhs");
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    // Queue several tens of milliseconds of device work.
    let mut queued = Vec::new();
    for _ in 0..8 {
        let out = backend
            .with_backend_session(|session| {
                session.dot_general_read(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &config,
                )
            })
            .unwrap()
            .unwrap();
        queued.push(out);
    }
    // Two empty scopes: the first dispatches the queue, the second is the one a
    // fence would make wait for the queued GEMMs.
    let mut empty = Duration::ZERO;
    with_cuda_exec(&mut backend, |session| {
        let start = Instant::now();
        session
            .with_raw("test.raw_entry_no_barrier", |_raw| Ok(()))
            .unwrap();
        session
            .with_raw("test.raw_entry_no_barrier", |_raw| Ok(()))
            .unwrap();
        empty = start.elapsed();
    });
    let queued_device_time = Duration::from_millis(30);
    assert!(
        empty < Duration::from_millis(5),
        "two empty raw scopes behind queued device work took {empty:?}; entering a scope must \
         not wait for the device (queued work is well above {queued_device_time:?})"
    );
    // The queued work is still ordered on the same stream and still correct.
    runtime.synchronize().expect("synchronize");
    let back = crate::cuda::download_tensor(&runtime, &queued[0]).expect("download");
    assert_eq!(back.as_slice::<f32>().unwrap()[0], n as f32);
}

/// A raw session reuses the runtime's cuBLAS handle instead of creating one
/// per session, which is what tenferro-rs #2083 asks for: the documented raw
/// vendor pattern used to cost `cublasCreate` + `cublasDestroy` (0.45 ms and
/// 10 MiB per handle on the reporting card) on every call.
#[test]
#[ignore = "requires CUDA"]
fn raw_session_reuses_the_runtime_cublas_handle() {
    use cudarc::cublas::sys as cublas_sys;

    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    let mut handles = Vec::new();
    with_cuda_exec(&mut backend, |session| {
        for _ in 0..2 {
            let handle = session
                .with_raw("test.raw_cublas", |raw| {
                    raw.with_cublas_handle(
                        "test.raw_cublas",
                        cublas_sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
                        |handle| Ok(handle as usize),
                    )
                })
                .unwrap();
            assert_ne!(handle, 0, "the runtime must hand out a live handle");
            handles.push(handle);
        }
    });
    assert_eq!(
        handles[0], handles[1],
        "two raw scopes of one runtime must use the same cached cuBLAS handle"
    );

    // The handle really is the runtime's own: a vendor enqueue that does not go
    // through a raw session reaches the same cached handle.
    let runtime = backend.runtime().clone();
    let runtime_handle = runtime
        .with_current_context("test.raw_cublas_runtime", || {
            runtime.with_cublas_handle(
                "test.raw_cublas_runtime",
                cublas_sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
                Vec::new(),
                |handle| Ok(handle as usize),
            )
        })
        .expect("primary context")
        .expect("runtime cuBLAS handle");
    assert_eq!(
        runtime_handle, handles[0],
        "the raw accessor must return the runtime's cached handle"
    );
}
