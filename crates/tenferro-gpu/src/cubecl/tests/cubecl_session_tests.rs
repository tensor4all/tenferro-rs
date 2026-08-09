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
