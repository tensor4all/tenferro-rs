//! Raw PTX/CUBIN/NVRTC module loading and launch tests (issue #1597).
//!
//! These tests require CUDA hardware and are therefore ignored by default,
//! matching the regular CUDA test convention in this crate.

use super::*;
use crate::cubecl::tests::with_cuda_exec;
use crate::cuda::raw::{KernelArg, LaunchConfig, NvrtcOptions};
use crate::cuda::{gpu_available, CudaBackend};

fn first_cuda_backend() -> Option<CudaBackend> {
    let devices = crate::cuda::cuda_devices().ok()?;
    let device = devices.first()?;
    CudaBackend::new(device.id()).ok()
}

/// A one-dimensional add kernel compiled from CUDA source by NVRTC.
const ADD_KERNEL_SRC: &str = r#"
extern "C" __global__ void add_kernel(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
"#;

#[test]
fn raw_nvrtc_launch_roundtrip() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");

    // Upload inputs before entering the session (the session borrows backend).
    let a_host = tensor_f32(vec![8], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b_host = tensor_f32(
        vec![8],
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    );
    let a_gpu = upload(&backend, &a_host);
    let b_gpu = upload(&backend, &b_host);

    with_cuda_exec(&mut backend, |session| {
        let runtime = session.runtime().clone();
        session
            .with_raw("test.add", |sess| {
                let Tensor::F32(a_typed) = &a_gpu else {
                    unreachable!("f32")
                };
                let Tensor::F32(b_typed) = &b_gpu else {
                    unreachable!("f32")
                };
                let a_ref = sess.tensor(a_typed)?;
                let b_ref = sess.tensor(b_typed)?;

                let mut out = sess.alloc_output::<f32>(&[8])?;
                let out_ref = sess.tensor_mut(&mut out)?;

                let module = sess.compile_nvrtc(ADD_KERNEL_SRC, &NvrtcOptions::default())?;
                let function = module.function("add_kernel")?;

                let config = LaunchConfig::flat(8, 8, 0).expect("valid flat config");
                // SAFETY: ABI is (out*, a*, b*, int); spans are the validated
                // refs; the domain covers exactly n elements in-bounds; module
                // stays alive during the (unsynchronized) launch.
                unsafe {
                    sess.launch(
                        &function,
                        config,
                        &[
                            KernelArg::tensor_mut(&out_ref),
                            KernelArg::tensor(&a_ref),
                            KernelArg::tensor(&b_ref),
                            KernelArg::i32(8),
                        ],
                    )?;
                }

                sess.synchronize()?;

                let (shape, flat): (Vec<usize>, Vec<f32>) = download_tensor_typed(&runtime, out)?;
                assert_eq!(shape, vec![8]);
                for i in 0..8 {
                    let expected = (i as f32 + 1.0) + ((i as f32) + 1.0) * 10.0;
                    assert!((flat[i] - expected).abs() < 1e-4, "mismatch at {i}");
                }
                Ok(())
            })
            .expect("raw session should run");
    });
}

#[test]
fn raw_ptx_load_launch_roundtrip() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");

    let a_host = tensor_f32(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let b_host = tensor_f32(vec![4], vec![0.5, 0.5, 0.5, 0.5]);
    let a_gpu = upload(&backend, &a_host);
    let b_gpu = upload(&backend, &b_host);

    with_cuda_exec(&mut backend, |session| {
        let runtime = session.runtime().clone();
        session
            .with_raw("test.ptx", |sess| {
                // Compile to PTX text on the host, then load via load_ptx.
                let ptx = cudarc::nvrtc::compile_ptx(ADD_KERNEL_SRC).map_err(|err| {
                    tenferro_tensor::Error::backend_source("test.ptx.compile", err)
                })?;
                let ptx_text = ptx.to_src();
                let ptx_cstr = std::ffi::CString::new(ptx_text).expect("PTX has no NUL");
                let module = sess.load_ptx(&ptx_cstr)?;
                let function = module.function("add_kernel")?;

                let Tensor::F32(a_typed) = &a_gpu else {
                    unreachable!("f32")
                };
                let Tensor::F32(b_typed) = &b_gpu else {
                    unreachable!("f32")
                };
                let a_ref = sess.tensor(a_typed)?;
                let b_ref = sess.tensor(b_typed)?;
                let mut out = sess.alloc_output::<f32>(&[4])?;
                let out_ref = sess.tensor_mut(&mut out)?;

                let config = LaunchConfig::flat(4, 4, 0).expect("valid flat config");
                // SAFETY: same ABI guarantee as the NVRTC roundtrip.
                unsafe {
                    sess.launch(
                        &function,
                        config,
                        &[
                            KernelArg::tensor_mut(&out_ref),
                            KernelArg::tensor(&a_ref),
                            KernelArg::tensor(&b_ref),
                            KernelArg::i32(4),
                        ],
                    )?;
                }
                sess.synchronize()?;

                let (shape, flat): (Vec<usize>, Vec<f32>) = download_tensor_typed(&runtime, out)?;
                assert_eq!(shape, vec![4]);
                for i in 0..4 {
                    let expected = (i as f32 + 1.0) + 0.5;
                    assert!((flat[i] - expected).abs() < 1e-4, "mismatch at {i}");
                }
                Ok(())
            })
            .expect("raw session should run");
    });
}

#[test]
fn raw_launch_rejects_bad_geometry() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        let result: tenferro_tensor::Result<()> =
            session.with_raw("test.bad_geometry", |sess| unsafe {
                let module = sess.compile_nvrtc(ADD_KERNEL_SRC, &NvrtcOptions::default())?;
                let function = module.function("add_kernel")?;
                let mut out = sess.alloc_output::<f32>(&[1])?;
                let out_ref = sess.tensor_mut(&mut out)?;
                let bad = LaunchConfig {
                    grid: [1, 1, 1],
                    block: [0, 1, 1],
                    shared_mem_bytes: 0,
                };
                let result = sess.launch(&function, bad, &[KernelArg::tensor_mut(&out_ref)]);
                match result {
                    Err(_) => Ok(()), // geometry validation must fail first
                    Ok(()) => Err(tenferro_tensor::Error::runtime_state(
                        "test.bad_geometry",
                        "launch with zero block unexpectedly accepted",
                    )),
                }
            });
        assert!(result.is_ok(), "bad geometry must produce an error result");
    });
}

#[test]
fn raw_load_cubin_rejects_garbage() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        let result: tenferro_tensor::Result<()> = session.with_raw("test.bad_cubin", |sess| {
            let garbage = b"this is not a cubin image";
            match sess.load_cubin(garbage) {
                Ok(_) => Err(tenferro_tensor::Error::runtime_state(
                    "test.bad_cubin",
                    "garbage cubin unexpectedly loaded",
                )),
                Err(_) => Ok(()),
            }
        });
        assert!(result.is_ok());
    });
}

/// Download a typed f32 tensor via the public `download_tensor` seam.
fn download_tensor_typed(
    runtime: &crate::cubecl::CudaRuntime,
    tensor: crate::TypedTensor<f32>,
) -> crate::Result<(Vec<usize>, Vec<f32>)> {
    let wrapped = crate::Tensor::F32(tensor);
    let downloaded = crate::cuda::download_tensor(runtime, &wrapped)?;
    let crate::Tensor::F32(host) = downloaded else {
        unreachable!("f32 download")
    };
    host.into_vec_col_major()
}
