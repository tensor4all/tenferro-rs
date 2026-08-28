// INVARIANT: this compile-only binary keeps independent downstream snippets
// type-checked without executing CUDA or nvcc on ordinary documentation CI.
#![allow(dead_code)]

mod nvcc_build_script {
    // snippet-start:custom_cuda_nvcc_build
    use std::{env, path::PathBuf, process::Command};

    fn run(args: &[&str]) {
        let status = Command::new("nvcc").args(args).status().expect("run nvcc");
        assert!(status.success(), "nvcc failed: {args:?}");
    }

    fn main() {
        let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        let ptx = out.join("add.ptx");
        let cubin = out.join("add.cubin");
        let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".into());

        run(&["--ptx", "kernel.cu", "-o", ptx.to_str().unwrap()]);
        run(&[
            "--cubin",
            "kernel.cu",
            "-arch",
            &arch,
            "-o",
            cubin.to_str().unwrap(),
        ]);
        println!("cargo:rerun-if-changed=kernel.cu");
    }
    // snippet-end:custom_cuda_nvcc_build
}

mod precompiled {
    // snippet-start:custom_cuda_precompiled
    use std::ffi::CString;
    use tenferro_gpu::cuda::raw;

    fn load_nvcc_module(
        session: &raw::Session<'_>,
        ptx_bytes: &[u8],
        cubin_bytes: &[u8],
        use_cubin: bool,
    ) -> tenferro_tensor::Result<raw::Module> {
        if use_cubin {
            session.load_cubin(cubin_bytes)
        } else {
            let ptx = CString::new(ptx_bytes).map_err(|_| {
                tenferro_tensor::Error::invalid_argument(
                    "custom_cuda.load",
                    "ptx",
                    "nvcc PTX contains an interior NUL",
                )
            })?;
            session.load_ptx(&ptx)
        }
    }
    // snippet-end:custom_cuda_precompiled
}

mod nvrtc {
    // snippet-start:custom_cuda_nvrtc
    use tenferro_gpu::cuda::raw::{self, NvrtcOptions};

    const CUDA_SOURCE: &str = r#"
extern "C" __global__ void add_kernel(
    float* out, const float* lhs, const float* rhs, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = lhs[i] + rhs[i];
}
"#;

    fn load_nvrtc_module(session: &raw::Session<'_>) -> tenferro_tensor::Result<raw::Module> {
        session.compile_nvrtc(CUDA_SOURCE, &NvrtcOptions::default())
    }
    // snippet-end:custom_cuda_nvrtc
}

mod launch {
    const CUDA_SOURCE: &str = r#"
extern "C" __global__ void add_kernel(
    float* out, const float* lhs, const float* rhs, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = lhs[i] + rhs[i];
}
"#;

    // snippet-start:custom_cuda_launch
    use tenferro_gpu::cuda::{raw, with_cuda_exec_session, CudaBackend};
    use tenferro_tensor::{BackendSessionHost, Error, TypedTensor};

    fn launch_add(
        backend: &mut CudaBackend,
        lhs: &TypedTensor<f32>,
        rhs: &TypedTensor<f32>,
    ) -> tenferro_tensor::Result<TypedTensor<f32>> {
        backend.with_backend_session(|backend_session| {
            with_cuda_exec_session(backend_session, |cuda| {
                cuda.with_raw("custom_cuda.add", |session| {
                    let lhs = session.tensor(lhs)?;
                    let rhs = session.tensor(rhs)?;
                    let mut output = session.alloc_output::<f32>(&[8])?;
                    let output_ref = session.tensor_mut(&mut output)?;

                    let module =
                        session.compile_nvrtc(CUDA_SOURCE, &raw::NvrtcOptions::default())?;
                    let function = module.function("add_kernel")?;
                    let config = raw::LaunchConfig::flat(8, 128, 0)?;

                    // SAFETY: the kernel ABI is (float*, const float*, const float*,
                    // int); all three tensors contain at least 8 f32 values; output
                    // is exclusively borrowed and does not alias either input; the
                    // kernel bounds-checks i < n; module and tensor borrows remain
                    // live until the explicit synchronization below.
                    unsafe {
                        session.launch(
                            &function,
                            config,
                            &[
                                raw::KernelArg::tensor_mut(&output_ref),
                                raw::KernelArg::tensor(&lhs),
                                raw::KernelArg::tensor(&rhs),
                                raw::KernelArg::i32(8),
                            ],
                        )?;
                    }

                    session.synchronize()?;
                    Ok(output)
                })
            })
            .ok_or_else(|| Error::runtime_state("custom_cuda.add", "backend session is not CUDA"))?
        })
    }
    // snippet-end:custom_cuda_launch
}

fn main() {}
