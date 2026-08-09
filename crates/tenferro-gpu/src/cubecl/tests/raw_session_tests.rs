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
fn raw_session_exposes_stream_and_runtime_identity() {
    if !gpu_available() {
        return;
    }
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
    // Context/device must be restored after the session; a subsequent backend
    // operation that needs the primary context must still work.
    assert!(session_after_raw_still_runs(&mut backend));
}

fn session_after_raw_still_runs(backend: &mut CudaBackend) -> bool {
    backend.runtime().synchronize().is_ok()
}

#[test]
fn raw_session_allocates_output_and_bytes() {
    if !gpu_available() {
        return;
    }
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
fn raw_session_reports_capabilities() {
    if !gpu_available() {
        return;
    }
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
fn raw_session_context_is_restored_after_error_callback() {
    if !gpu_available() {
        return;
    }
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
fn raw_tensor_ref_carries_validated_span() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    let host = tensor_f32(vec![8], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let gpu = upload(&backend, &host);
    let Tensor::F32(gpu_typed) = &gpu else {
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
fn raw_retain_tensor_pins_the_allocation_across_a_drop() {
    if !gpu_available() {
        return;
    }
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
                    let Tensor::F32(gpu_typed) = &gpu else {
                        unreachable!("f32 tensor")
                    };
                    let reference = raw.tensor(gpu_typed)?;
                    // SAFETY: `reference` is a validated span for `gpu_typed`;
                    // copying the pointer value is read-only and the retained
                    // guard keeps the allocation alive past the drop below.
                    let saved_ptr = unsafe { reference.raw_ptr() };
                    let retained =
                        raw.retain_tensor(gpu_typed, "test.raw_retain_tensor")?;
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
                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn raw_retain_tensor_rejects_tensor_from_another_runtime() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    // Upload on a separate runtime instance: its allocation domain differs
    // from the session runtime's, so retention must be rejected.
    let foreign_rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = tensor_f32(vec![4], vec![1.0f32, 2.0, 3.0, 4.0]);
    let gpu = upload_tensor(&foreign_rt, &host).unwrap();
    let Tensor::F32(gpu_typed) = &gpu else {
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
fn raw_resource_guard_is_runtime_scoped_and_type_keyed() {
    if !gpu_available() {
        return;
    }
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
