//! CubeCL-session contract tests (issue #1597).
//!
//! These tests require CUDA hardware and are therefore ignored by default,
//! matching the regular CUDA test convention in this crate.

use crate::cuda::{gpu_available, CudaBackend};

use super::*;

fn first_cuda_backend() -> Option<CudaBackend> {
    let devices = crate::cuda::cuda_devices().ok()?;
    let device = devices.first()?;
    CudaBackend::new(device.id()).ok()
}

#[test]
fn cubecl_session_exposes_client_and_launch_helpers() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        session
            .with_cubecl("test.cubecl_helpers", |cubecl| {
                // A small one-dimensional problem gives the standard cube dim
                // and a bounded cube count.
                let count = cubecl.cube_count_1d(256)?;
                let cubecl::prelude::CubeCount::Static(x, _y, _z) = count else {
                    panic!("expected static cube count")
                };
                assert!(x >= 1);
                let _dim = cubecl.cube_dim_1d();
                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn cubecl_session_allocates_and_binds_output() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        session
            .with_cubecl("test.cubecl_alloc", |cubecl| {
                let output = cubecl.alloc_output::<f32>(&[8])?;
                // Binding a freshly allocated tensor must be valid.
                let _binding = cubecl.tensor_binding(&output, "test.cubecl_alloc")?;
                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn cubecl_session_allocates_zero_filled_output() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        let output = session
            .with_cubecl("test.cubecl_alloc_zero", |cubecl| {
                cubecl.alloc_zero_output::<f32>(&[16])
            })
            .unwrap();
        // The fill kernel must produce semantic zeros on the device.
        let result = session
            .with_raw("test.cubecl_alloc_zero_raw", |raw| {
                raw.download_tensor::<f32>(&output, "test.cubecl_alloc_zero_raw")
            })
            .unwrap();
        let values = result.host_data().unwrap();
        assert_eq!(values.len(), 16);
        assert!(values.iter().all(|&v| v == 0.0));
    });
}

#[test]
fn cubecl_session_scales_output_in_place() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        // Allocate and seed [1.0, 2.0, 3.0, 4.0] on device, then scale by 3
        // and read back, returning the typed tensor from the raw entrance.
        let output = session
            .with_raw("test.cubecl_scale_raw", |raw| {
                let mut output = raw.alloc_output::<f32>(&[4])?;
                let seed = [1.0f32, 2.0, 3.0, 4.0];
                let seed_bytes = unsafe {
                    std::slice::from_raw_parts(seed.as_ptr().cast::<u8>(), seed.len() * 4)
                };
                let uploaded = raw.upload_bytes(seed_bytes, "test.cubecl_scale_seed")?;
                let dst = raw.tensor_mut(&mut output)?;
                let dst_ptr = unsafe { dst.raw_ptr() };
                let mut copy_result = Ok(());
                // SAFETY: `uploaded` is an uploaded workspace of the same
                // byte size as `dst`, both on this runtime's stream; `dst`
                // uniquely owns the destination span.
                unsafe {
                    uploaded.with_ptr(|src_ptr| {
                        copy_result = raw.copy_bytes(
                            dst_ptr,
                            src_ptr,
                            seed_bytes.len(),
                            "test.cubecl_scale_copy",
                        );
                    });
                }
                copy_result?;
                Ok(output)
            })
            .unwrap();
        let mut output_enum = tenferro_tensor::Tensor::F32(output);
        session
            .with_cubecl("test.cubecl_scale", |cubecl| {
                cubecl.scale_tensor_write(
                    tenferro_tensor::TensorWrite::from_tensor(&mut output_enum),
                    3.0,
                )
            })
            .unwrap();
        let typed = match output_enum {
            tenferro_tensor::Tensor::F32(typed) => typed,
            _ => unreachable!(),
        };
        let bytes = session
            .with_raw("test.cubecl_scale_raw2", |raw| {
                raw.download_tensor::<f32>(&typed, "test.cubecl_scale_raw2")
            })
            .unwrap();
        let values = bytes.host_data().unwrap();
        assert_eq!(values[0], 3.0);
        assert_eq!(values[1], 6.0);
        assert_eq!(values[2], 9.0);
        assert_eq!(values[3], 12.0);
    });
}

#[test]
fn cubecl_session_flushes_on_exit_so_raw_sees_work() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        // Enqueue a trivial cubecl interaction, then immediately enter the raw
        // session. The cross-session flush must make the captured stream usable.
        session
            .with_cubecl("test.cubecl_then_raw", |cubecl| {
                let _ = cubecl.cube_count_1d(16)?;
                Ok(())
            })
            .unwrap();
        session
            .with_raw("test.cubecl_then_raw", |raw| {
                let _stream = raw.stream();
                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn cubecl_session_flushes_after_error_callback() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        let result: tenferro_tensor::Result<()> =
            session.with_cubecl("test.cubecl_error", |cubecl| {
                let _ = cubecl.cube_count_1d(4)?;
                Err(tenferro_tensor::Error::runtime_state(
                    "test.cubecl_error",
                    "intentional failure",
                ))
            });
        assert!(result.is_err());
        // A subsequent session still works after the error path flushed.
        session
            .with_cubecl("test.cubecl_after_error", |cubecl| {
                let _ = cubecl.cube_count_1d(8)?;
                Ok(())
            })
            .unwrap();
    });
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "nested backend session entry")]
fn cuda_with_backend_session_rejects_nested_entry_in_debug_builds() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    backend.with_backend_session(|_session| {
        // The CUDA override wraps its closure in the portable in-session
        // guard, so re-entering any session-entry point on this thread
        // (here the shared helper directly) trips the debug assert
        // (issue #1680 Phase 3).
        tenferro_tensor::with_session_entry_guard(|| ())
    });
}

#[cfg(debug_assertions)]
#[test]
fn cuda_with_backend_session_restores_the_in_session_flag_after_panic() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.with_backend_session(|_session| panic!("boom"))
    }));
    assert!(outcome.is_err());
    // The flag is usable again on the same thread.
    let value = backend.with_backend_session(|_session| 7usize);
    assert_eq!(value, 7);
}
