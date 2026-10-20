#![cfg(feature = "cuda")]

//! #2084: a raw CUDA kernel reads an eager `constant_from_host` value in place.
//!
//! The eager runtime holds a constant as a view over its device allocation, so
//! `TensorRead::as_tensor()` is `None` and `raw::Session::tensor` (which takes an
//! owned `TypedTensor`) cannot bind it. Before `raw::Session::tensor_read` the
//! only way to obtain a device pointer was `duplicate_value`, an extra device
//! copy per weight. This test reproduces the downstream shape of the report:
//! bind the constant, launch a kernel that reads it, and compare the downloaded
//! values with the host data that was uploaded.

use tenferro_ad::{EagerRuntime, Tensor};
use tenferro_gpu::cuda::raw::{KernelArg, LaunchConfig, NvrtcOptions};
use tenferro_gpu::cuda::{gpu_available, with_cuda_engine, CudaDeviceId, CudaEngine};

const COPY_F64_KERNEL_SRC: &str = r#"
extern "C" __global__ void copy_f64(double* out, const double* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = src[i];
    }
}
"#;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(shape, data).expect("valid f64 tensor")
}

#[test]
#[ignore = "requires CUDA"]
fn raw_kernel_reads_a_constant_from_host_in_place() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let runtime = EagerRuntime::new().expect("eager runtime");
    let mut engine =
        CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA engine should initialize");

    let host_values: Vec<f64> = (1..=8).map(f64::from).collect();
    runtime
        .with_eager_session(&mut engine, |session| -> tenferro_ad::Result<()> {
            let weight = session.constant_from_host(f64_tensor(vec![8], host_values.clone()))?;
            let read = weight.tensor_read();
            assert!(
                read.as_tensor().is_none(),
                "an eager constant is view storage, which `Session::tensor` cannot bind (#2084)"
            );
            assert!(
                read.is_col_major_contiguous()?,
                "the eager constant's view must stay compact for the raw binding"
            );

            let copied = with_cuda_engine(session.engine(), |cuda| {
                cuda.with_raw(
                    "test.raw_constant_in_place",
                    |raw| -> tenferro_tensor::Result<(Vec<usize>, Vec<f64>)> {
                        let bound = raw.tensor_read::<f64>(&read)?;
                        assert_eq!(
                            bound.byte_len(),
                            host_values.len() * std::mem::size_of::<f64>(),
                            "the binding must cover the constant's own element span"
                        );
                        let repeated = raw.tensor_read::<f64>(&read)?;
                        // SAFETY: observes the validated address values only.
                        assert_eq!(
                            unsafe { repeated.raw_ptr() } as usize,
                            unsafe { bound.raw_ptr() } as usize,
                            "a repeated binding of the same constant must resolve the same device \
                             storage instead of duplicating it"
                        );

                        let mut out = raw.alloc_output::<f64>(&[8])?;
                        let out_ref = raw.tensor_mut(&mut out)?;
                        let module =
                            raw.compile_nvrtc(COPY_F64_KERNEL_SRC, &NvrtcOptions::default())?;
                        let function = module.function("copy_f64")?;
                        let config = LaunchConfig::flat(8, 8, 0)?;
                        // SAFETY: the kernel ABI is (double*, const double*, int);
                        // the span bound by `bound` covers the 8 constant values
                        // the launch reads, `out_ref` is an exclusively borrowed
                        // 8-element span that does not alias it, and the module,
                        // output, and eager value outlive the launch plus the
                        // synchronization below.
                        unsafe {
                            raw.launch(
                                &function,
                                config,
                                &[
                                    KernelArg::tensor_mut(&out_ref),
                                    KernelArg::tensor(&bound),
                                    KernelArg::i32(8),
                                ],
                            )?;
                        }
                        raw.synchronize()?;
                        let back = raw.download_tensor(&out, "test.raw_constant_in_place")?;
                        back.into_vec_col_major()
                            .map_err(|failure| failure.into_parts().1)
                    },
                )
            })
            .expect("CUDA backend session")?;

            let (shape, values) = copied;
            assert_eq!(shape, vec![8]);
            assert_eq!(
                values, host_values,
                "the raw kernel must read the eager constant's device storage in place"
            );
            Ok(())
        })
        .expect("eager CUDA session");
}
