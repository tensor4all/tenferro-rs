//! Externally owned CUDA memory interop tests (issue #2118).
//!
//! The validation-only test runs without a device. The allocation and launch
//! tests require CUDA hardware and are therefore ignored by default, matching
//! the regular CUDA test convention in this crate.

use std::ffi::c_void;
use std::sync::Arc;

use super::*;
use crate::cubecl::tests::with_cuda_exec;
use crate::cuda::raw::{self, KernelArg, LaunchConfig, NvrtcOptions};
use crate::cuda::{gpu_available, CudaBackend};
use cudarc::driver::{CudaContext, CudaStream, DevicePtr, DevicePtrMut};

fn first_cuda_backend() -> Option<CudaBackend> {
    let devices = crate::cuda::cuda_devices().ok()?;
    let device = devices.first()?;
    CudaBackend::new(device.id()).ok()
}

/// Wrap the session's captured stream as a borrowed cudarc stream.
///
/// The context retains the same primary context as the tenferro runtime, and the
/// wrapper does not own or destroy the CubeCL stream. The session stream is the
/// only stream the raw kernel is enqueued on, so cudarc work and the kernel are
/// ordered without a cross-stream event.
fn borrowed_session_stream(
    context: &Arc<CudaContext>,
    session: &raw::Session<'_>,
) -> Arc<CudaStream> {
    let cu_stream = unsafe { session.stream().raw_handle() } as cudarc::driver::sys::CUstream;
    // SAFETY: `cu_stream` is the live stream captured by the raw session; the
    // wrapped handle is borrowed and the runtime keeps the stream alive for the
    // whole callback.
    unsafe { context.wrap_borrowed_stream(cu_stream) }
}

/// Download a typed f32 tensor through the public transfer seam.
fn download_f32(
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

const WRITE_PATTERN_KERNEL_SRC: &str = r#"
extern "C" __global__ void write_pattern_kernel(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (float)(i * 3 + 1);
    }
}
"#;

const SCALE_INPUT_KERNEL_SRC: &str = r#"
extern "C" __global__ void scale_input_kernel(float* out, const float* in, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * alpha + 1.0f;
    }
}
"#;

#[test]
fn raw_external_span_validation_rejects_bad_spans() {
    let op = "test.external_span";

    let null =
        crate::cubecl::raw::validate_external_span::<f64>(std::ptr::null_mut(), 8, op).unwrap_err();
    assert_validation_kind(&null, op, ValidationKind::InvalidArgument);

    let empty =
        crate::cubecl::raw::validate_external_span::<f64>(0x1000usize as *mut c_void, 0, op)
            .unwrap_err();
    assert_validation_kind(&empty, op, ValidationKind::InvalidArgument);

    let misaligned =
        crate::cubecl::raw::validate_external_span::<f64>(0x1001usize as *mut c_void, 8, op)
            .unwrap_err();
    assert_validation_kind(&misaligned, op, ValidationKind::InvalidArgument);

    let partial_element =
        crate::cubecl::raw::validate_external_span::<f64>(0x1000usize as *mut c_void, 12, op)
            .unwrap_err();
    assert_validation_kind(&partial_element, op, ValidationKind::InvalidArgument);

    let near_max = (usize::MAX & !0x7) as *mut c_void;
    let overflow = crate::cubecl::raw::validate_external_span::<f64>(near_max, 16, op).unwrap_err();
    assert_validation_kind(&overflow, op, ValidationKind::InvalidArgument);
}

#[test]
#[ignore = "requires CUDA"]
fn raw_external_wrap_constructors_reject_bad_spans() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.external_bad_span", |sess| {
                let op = "test.external_bad_span";
                // SAFETY: none of these addresses is dereferenced; the
                // constructors must reject each span before it can be launched.
                let null = unsafe { sess.tensor_ref_from_raw::<f32>(std::ptr::null_mut(), 4, op) }
                    .unwrap_err();
                assert_validation_kind(&null, op, ValidationKind::InvalidArgument);

                let empty =
                    unsafe { sess.tensor_mut_from_raw::<f32>(0x1000usize as *mut c_void, 0, op) }
                        .unwrap_err();
                assert_validation_kind(&empty, op, ValidationKind::InvalidArgument);

                let misaligned =
                    unsafe { sess.tensor_ref_from_raw::<f32>(0x1001usize as *mut c_void, 4, op) }
                        .unwrap_err();
                assert_validation_kind(&misaligned, op, ValidationKind::InvalidArgument);
                Ok(())
            })
            .expect("raw session should run");
    });
}

#[test]
#[ignore = "requires CUDA"]
fn raw_external_cudarc_buffer_is_written_in_place() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        let ordinal = session.device_info().id().ordinal() as usize;
        session
            .with_raw("test.external_write", |sess| {
                const N: usize = 8;
                const BYTES: usize = N * std::mem::size_of::<f32>();
                let context = CudaContext::new(ordinal).expect("retain primary context");
                let stream = borrowed_session_stream(&context, sess);
                let mut external = stream.alloc_zeros::<f32>(N).expect("cudarc allocation");

                let module =
                    sess.compile_nvrtc(WRITE_PATTERN_KERNEL_SRC, &NvrtcOptions::default())?;
                let function = module.function("write_pattern_kernel")?;
                let config = LaunchConfig::flat(N as u32, 8, 0).expect("valid flat config");

                let (ptr, write_record) = external.device_ptr_mut(&stream);
                // SAFETY: `ptr` is the cudarc allocation on this session's
                // primary context and stream; it is exclusively borrowed for
                // the whole span; the module, function, and external slice stay
                // alive through the synchronization below.
                let out_ref = unsafe {
                    sess.tensor_mut_from_raw::<f32>(
                        ptr as *mut c_void,
                        BYTES,
                        "test.external_write",
                    )?
                };
                // SAFETY: ABI is (out*, int); the kernel writes exactly the N
                // in-bounds elements of the validated span.
                unsafe {
                    sess.launch(
                        &function,
                        config,
                        &[KernelArg::tensor_mut(&out_ref), KernelArg::i32(N as i32)],
                    )?;
                }
                // Record the write frontier after the launch, then wait for the
                // session stream before handing the buffer back to cudarc.
                drop(write_record);
                sess.synchronize()?;

                let host: Vec<f32> = stream.clone_dtoh(&external).expect("cudarc readback");
                let expected: Vec<f32> = (0..N).map(|i| (i * 3 + 1) as f32).collect();
                assert_eq!(
                    host, expected,
                    "kernel must have written the external span in place"
                );
                Ok(())
            })
            .expect("raw session should run");
    });
}

#[test]
#[ignore = "requires CUDA"]
fn raw_external_cudarc_buffer_is_read_by_a_kernel() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        let ordinal = session.device_info().id().ordinal() as usize;
        let runtime = session.runtime().clone();
        session
            .with_raw("test.external_read", |sess| {
                const N: usize = 8;
                const BYTES: usize = N * std::mem::size_of::<f32>();
                let context = CudaContext::new(ordinal).expect("retain primary context");
                let stream = borrowed_session_stream(&context, sess);
                let input_host: Vec<f32> = (0..N).map(|i| i as f32 + 1.0).collect();
                let external = stream
                    .clone_htod(&input_host)
                    .expect("cudarc upload on session stream");

                let module =
                    sess.compile_nvrtc(SCALE_INPUT_KERNEL_SRC, &NvrtcOptions::default())?;
                let function = module.function("scale_input_kernel")?;
                let config = LaunchConfig::flat(N as u32, 8, 0).expect("valid flat config");

                let (ptr, read_record) = external.device_ptr(&stream);
                // SAFETY: `ptr` is the cudarc allocation on this session's
                // primary context and stream; the upload was ordered on that
                // stream; nothing writes it during the launch; the allocation
                // stays alive through the synchronization below.
                let in_ref = unsafe {
                    sess.tensor_ref_from_raw::<f32>(
                        ptr as *mut c_void,
                        BYTES,
                        "test.external_read",
                    )?
                };
                drop(read_record);

                let mut out = sess.alloc_output::<f32>(&[N])?;
                let out_ref = sess.tensor_mut(&mut out)?;
                // SAFETY: ABI is (out*, in*, float, int); the kernel reads the
                // validated external input span and writes the N in-bounds
                // elements of the tenferro output.
                unsafe {
                    sess.launch(
                        &function,
                        config,
                        &[
                            KernelArg::tensor_mut(&out_ref),
                            KernelArg::tensor(&in_ref),
                            KernelArg::f32(2.0),
                            KernelArg::i32(N as i32),
                        ],
                    )?;
                }
                sess.synchronize()?;

                let (shape, flat): (Vec<usize>, Vec<f32>) = download_f32(&runtime, out)?;
                assert_eq!(shape, vec![N]);
                let expected: Vec<f32> = input_host.iter().map(|value| value * 2.0 + 1.0).collect();
                assert_eq!(
                    flat, expected,
                    "kernel must have read the external input span"
                );
                Ok(())
            })
            .expect("raw session should run");
    });
}
