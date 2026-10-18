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
#[ignore = "requires CUDA"]
fn raw_nvrtc_launch_roundtrip() {
    assert!(gpu_available(), "CUDA test requires an available device");
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
                let Some(a_typed) = a_gpu.as_typed::<f32>() else {
                    unreachable!("f32")
                };
                let Some(b_typed) = b_gpu.as_typed::<f32>() else {
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
                for (i, &value) in flat.iter().enumerate() {
                    let expected = (i as f32 + 1.0) + ((i as f32) + 1.0) * 10.0;
                    assert!((value - expected).abs() < 1e-4, "mismatch at {i}");
                }
                Ok(())
            })
            .expect("raw session should run");
    });
}

#[test]
#[ignore = "requires CUDA"]
fn raw_ptx_load_launch_roundtrip() {
    assert!(gpu_available(), "CUDA test requires an available device");
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

                let Some(a_typed) = a_gpu.as_typed::<f32>() else {
                    unreachable!("f32")
                };
                let Some(b_typed) = b_gpu.as_typed::<f32>() else {
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
                for (i, &value) in flat.iter().enumerate() {
                    let expected = (i as f32 + 1.0) + 0.5;
                    assert!((value - expected).abs() < 1e-4, "mismatch at {i}");
                }
                Ok(())
            })
            .expect("raw session should run");
    });
}

#[test]
#[ignore = "requires CUDA"]
fn raw_launch_rejects_bad_geometry() {
    assert!(gpu_available(), "CUDA test requires an available device");
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
#[ignore = "requires CUDA"]
fn raw_load_cubin_rejects_garbage() {
    assert!(gpu_available(), "CUDA test requires an available device");
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

#[test]
fn raw_dynamic_shared_memory_opt_in_runs_once_per_function() {
    use crate::cubecl::raw::needs_shared_memory_opt_in;

    const DEFAULT_BYTES: u32 = 48 * 1024;
    const TILE_BYTES: u32 = 64 * 1024;

    // At or below the driver default: no attribute, no driver call.
    assert!(!needs_shared_memory_opt_in(0, 0));
    assert!(!needs_shared_memory_opt_in(0, DEFAULT_BYTES));
    // First request above the default raises the attribute once.
    assert!(needs_shared_memory_opt_in(0, TILE_BYTES));
    // Repeating the same request, or asking for less, reuses the attribute.
    assert!(!needs_shared_memory_opt_in(TILE_BYTES, TILE_BYTES));
    assert!(!needs_shared_memory_opt_in(TILE_BYTES, TILE_BYTES - 1024));
    assert!(!needs_shared_memory_opt_in(TILE_BYTES, DEFAULT_BYTES));
    // Only a larger request pays another driver call.
    assert!(needs_shared_memory_opt_in(TILE_BYTES, TILE_BYTES + 1024));
}

/// A kernel that fills a dynamic shared-memory tile and writes samples out.
///
/// The tile is `floats` f32 values of dynamic shared memory, so the launcher's
/// `shared_mem_bytes` decides the size the kernel can address.
const SHARED_MEMORY_KERNEL_SRC: &str = r#"
extern "C" __global__ void smem_kernel(float* out, int floats) {
    extern __shared__ float tile[];
    for (int i = threadIdx.x; i < floats; i += blockDim.x) tile[i] = (float)i;
    __syncthreads();
    if (threadIdx.x < 8) out[threadIdx.x] = tile[threadIdx.x * (floats / 8)];
    if (threadIdx.x == 0) out[8] = tile[floats - 1];
}
"#;

#[test]
#[ignore = "requires CUDA"]
fn raw_launch_uses_dynamic_shared_memory_above_48_kib() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");

    // 64 KiB of dynamic shared memory in 16384 f32 slots: above the 48 KiB
    // no-opt-in default, and below the A100/Turing opt-in per-block maximum.
    const SHARED_BYTES: u32 = 64 * 1024;
    const FLOATS: i32 = (SHARED_BYTES / 4) as i32;
    assert_eq!(FLOATS, 16384);

    with_cuda_exec(&mut backend, |session| {
        let runtime = session.runtime().clone();
        session
            .with_raw("test.dynamic_shared_memory", |sess| {
                let module =
                    sess.compile_nvrtc(SHARED_MEMORY_KERNEL_SRC, &NvrtcOptions::default())?;
                let function = module.function("smem_kernel")?;
                let mut out = sess.alloc_output::<f32>(&[9])?;

                // Launch twice with the same size: the attribute is set on the
                // first launch and reused on the second.
                for _ in 0..2 {
                    let out_ref = sess.tensor_mut(&mut out)?;
                    let config = LaunchConfig {
                        grid: [1, 1, 1],
                        block: [256, 1, 1],
                        shared_mem_bytes: SHARED_BYTES,
                    };
                    // SAFETY: ABI is (out*, int); `out` has 9 f32 slots for the
                    // 9 values the kernel writes; the tile indices stay inside
                    // the `shared_mem_bytes` budget; blockDim is 256; module,
                    // function, and tensor outlive the synchronization below.
                    unsafe {
                        sess.launch(
                            &function,
                            config,
                            &[KernelArg::tensor_mut(&out_ref), KernelArg::i32(FLOATS)],
                        )?;
                    }
                    sess.synchronize()?;
                }

                let (shape, flat): (Vec<usize>, Vec<f32>) = download_tensor_typed(&runtime, out)?;
                assert_eq!(shape, vec![9]);
                let floats = FLOATS as usize;
                for (sample, &value) in flat.iter().take(8).enumerate() {
                    let expected = (sample * (floats / 8)) as f32;
                    assert_eq!(value, expected, "unexpected tile value at sample {sample}");
                }
                assert_eq!(flat[8], (floats - 1) as f32, "unexpected last tile value");
                Ok(())
            })
            .expect("raw session should run");
    });
}

#[test]
#[ignore = "requires CUDA"]
fn raw_launch_rejects_dynamic_shared_memory_above_device_opt_in() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");

    with_cuda_exec(&mut backend, |session| {
        // The opt-in device cap is exposed by the CubeCL session; a raw
        // session deliberately adds no second accessor for it.
        let limit = session
            .with_cubecl("test.dynamic_shared_limit", |cubecl| {
                Ok(cubecl.client().properties().hardware.max_shared_memory_size)
            })
            .expect("CubeCL session should expose the device properties");
        assert!(
            limit > 48 * 1024,
            "device opt-in limit {limit} must exceed 48 KiB"
        );
        let over_limit = u32::try_from(limit).expect("limit fits u32") + 4096;

        let result: crate::Result<()> = session.with_raw("test.dynamic_shared_too_large", |sess| {
            let module = sess.compile_nvrtc(SHARED_MEMORY_KERNEL_SRC, &NvrtcOptions::default())?;
            let function = module.function("smem_kernel")?;
            let mut out = sess.alloc_output::<f32>(&[9])?;
            let out_ref = sess.tensor_mut(&mut out)?;
            let config = LaunchConfig {
                grid: [1, 1, 1],
                block: [256, 1, 1],
                shared_mem_bytes: over_limit,
            };
            // SAFETY: ABI and bounds as above, but the driver must reject
            // this launch before it runs: the opt-in exceeds the device cap.
            unsafe {
                sess.launch(
                    &function,
                    config,
                    &[KernelArg::tensor_mut(&out_ref), KernelArg::i32(1)],
                )
            }
        });

        match result {
            Err(crate::Error::BackendSource { op, .. }) => {
                assert_eq!(op, "raw.launch", "the attribute setter must report the typed driver error");
            }
            other => panic!(
                "requesting {over_limit} bytes of dynamic shared memory above the {limit}-byte device cap must fail with a typed driver error, got {other:?}"
            ),
        }
    });
}

/// Download a typed f32 tensor via the public `download_tensor` seam.
fn download_tensor_typed(
    runtime: &crate::cubecl::CudaRuntime,
    tensor: crate::TypedTensor<f32>,
) -> crate::Result<(Vec<usize>, Vec<f32>)> {
    let wrapped = crate::Tensor::from_typed::<f32>(tensor);
    let downloaded = crate::cuda::download_tensor(runtime, &wrapped)?;
    let Ok(host) = downloaded.into_typed::<f32>() else {
        unreachable!("f32 download")
    };
    host.into_vec_col_major()
        .map_err(|failure| failure.into_parts().1)
}
