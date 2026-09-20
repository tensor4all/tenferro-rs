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
#[ignore = "requires CUDA"]
fn cubecl_session_exposes_client_and_launch_helpers() {
    assert!(gpu_available(), "CUDA test requires an available device");
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
#[ignore = "requires CUDA"]
fn cubecl_session_allocates_and_binds_output() {
    assert!(gpu_available(), "CUDA test requires an available device");
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
#[ignore = "requires CUDA"]
fn cubecl_session_allocates_zero_filled_output() {
    assert!(gpu_available(), "CUDA test requires an available device");
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
#[ignore = "requires CUDA"]
fn cubecl_session_fills_an_existing_output_with_zero() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        // Seed a destination the session owns with a value that `0 * y` would
        // preserve, then reset it through the public session entry.
        let output = session
            .with_raw("test.cubecl_fill_zero_raw", |raw| {
                let mut output = raw.alloc_output::<f32>(&[4])?;
                let seed = [f32::NAN, -0.0, f32::INFINITY, 2.5];
                let seed_bytes = unsafe {
                    std::slice::from_raw_parts(seed.as_ptr().cast::<u8>(), seed.len() * 4)
                };
                let uploaded = raw.upload_bytes(seed_bytes, "test.cubecl_fill_zero_seed")?;
                let dst = raw.tensor_mut(&mut output)?;
                let dst_ptr = unsafe { dst.raw_ptr() };
                let mut copy_result = Ok(());
                // SAFETY: `uploaded` is an uploaded workspace of the same byte
                // size as `dst`, both on this runtime's stream; `dst` uniquely
                // owns the destination span.
                unsafe {
                    uploaded.with_ptr(|src_ptr| {
                        copy_result = raw.copy_bytes(
                            dst_ptr,
                            src_ptr,
                            seed_bytes.len(),
                            "test.cubecl_fill_zero_copy",
                        );
                    });
                }
                copy_result?;
                Ok(output)
            })
            .unwrap();
        let mut output_enum = tenferro_tensor::Tensor::from_typed::<f32>(output);
        session
            .with_cubecl("test.cubecl_fill_zero", |cubecl| {
                cubecl.fill_zero_write(tenferro_tensor::TensorWrite::from_tensor(&mut output_enum))
            })
            .unwrap();
        let typed = output_enum
            .into_typed::<f32>()
            .expect("the fill keeps the destination dtype");
        let result = session
            .with_raw("test.cubecl_fill_zero_raw2", |raw| {
                raw.download_tensor::<f32>(&typed, "test.cubecl_fill_zero_raw2")
            })
            .unwrap();
        let values = result.host_data().unwrap();
        assert_eq!(values.len(), 4);
        for value in values {
            assert_eq!(value.to_bits(), 0.0_f32.to_bits(), "expected +0.0 bits");
        }
    });
}

#[test]
#[ignore = "requires CUDA"]
fn cubecl_session_scales_output_in_place() {
    assert!(gpu_available(), "CUDA test requires an available device");
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
        let mut output_enum = tenferro_tensor::Tensor::from_typed::<f32>(output);
        session
            .with_cubecl("test.cubecl_scale", |cubecl| {
                cubecl.scale_tensor_write(
                    tenferro_tensor::TensorWrite::from_tensor(&mut output_enum),
                    3.0,
                )
            })
            .unwrap();
        let typed = output_enum
            .into_typed::<f32>()
            .expect("the scale output keeps its dtype");
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
#[ignore = "requires CUDA"]
fn cubecl_session_flushes_on_exit_so_raw_sees_work() {
    assert!(gpu_available(), "CUDA test requires an available device");
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
#[ignore = "requires CUDA"]
fn cubecl_session_flushes_after_error_callback() {
    assert!(gpu_available(), "CUDA test requires an available device");
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
#[ignore = "requires CUDA"]
fn cuda_with_backend_session_rejects_nested_entry_in_debug_builds() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.with_backend_session(|_session| {
            // The CUDA override wraps its closure in the portable in-session
            // guard, so re-entering any session-entry point on this thread
            // (here the shared helper directly) trips the debug assert
            // (issue #1680 Phase 3).
            tenferro_tensor::with_session_entry_guard(|| ())
        })
    }));
    assert!(
        outcome.is_err(),
        "nested session entry must panic in debug builds"
    );
}

#[cfg(debug_assertions)]
#[test]
#[ignore = "requires CUDA"]
fn cuda_with_backend_session_restores_the_in_session_flag_after_panic() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.with_backend_session(|_session| panic!("boom"))
    }));
    assert!(outcome.is_err());
    // The flag is usable again on the same thread.
    let value = backend.with_backend_session(|_session| 7usize);
    assert_eq!(value, 7);
}
