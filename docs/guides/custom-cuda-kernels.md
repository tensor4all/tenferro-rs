# Custom CUDA kernels

External crates can launch CUDA kernels against tenferro-owned tensors through
the public `tenferro_gpu::cuda::raw` API. The supported path is always:

```text
CudaBackend::with_backend_session
  -> with_cuda_exec_session
  -> CudaExecSession::with_raw
  -> load PTX/CUBIN or compile with NVRTC
  -> launch on the session stream
```

Do not extract pointers or streams from `CudaRuntime` directly. The raw session
is the scoped execution authority that keeps the tenferro CUDA context, stream,
and allocation ownership aligned.

## Dependencies

```toml
[dependencies]
tenferro-gpu = { version = "...", default-features = false, features = ["cuda"] }
tenferro-tensor = "..."
```

For a local checkout, replace the versions with matching `path` dependencies.

## Kernel

Both examples use the same kernel:

```cuda
// kernel.cu
extern "C" __global__ void add_kernel(
    float* out,
    const float* lhs,
    const float* rhs,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = lhs[i] + rhs[i];
    }
}
```

## Precompile with `nvcc`

PTX is portable across compatible CUDA devices; CUBIN is specific to the
selected architecture. A minimal `build.rs` can emit both:

<!-- snippet-source: docs/tutorial-code/src/bin/custom_cuda_kernels.rs#custom_cuda_nvcc_build -->
```rust
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
```
<!-- end-snippet-source -->

Set `CUDA_ARCH` to the deployment GPU, for example `sm_86` for an RTX 3060.
Embed or read the generated files and load exactly one image:

<!-- snippet-source: docs/tutorial-code/src/bin/custom_cuda_kernels.rs#custom_cuda_precompiled -->
```rust
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
```
<!-- end-snippet-source -->

Use PTX unless architecture-specific CUBIN startup or code generation has been
measured to matter.

## Compile at runtime with NVRTC

NVRTC needs no `build.rs`. Pass CUDA source directly to the raw session:

<!-- snippet-source: docs/tutorial-code/src/bin/custom_cuda_kernels.rs#custom_cuda_nvrtc -->
```rust
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
```
<!-- end-snippet-source -->

`compile_nvrtc` returns a typed backend error containing the compiler log when
compilation fails. Cache reusable modules with `raw::Session::resource` rather
than compiling on every call.

## Borrow tensors and launch

The launch code is identical for PTX, CUBIN, and NVRTC modules:

<!-- snippet-source: docs/tutorial-code/src/bin/custom_cuda_kernels.rs#custom_cuda_launch -->
```rust
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
```
<!-- end-snippet-source -->

Inputs must already be uploaded to the same `CudaBackend` runtime. The safe
`tensor` and `tensor_mut` methods validate residency and return bounded device
spans. `tensor_mut` requires an exclusive tensor borrow; a fresh
`alloc_output` is the simplest output path. Owned `TypedTensor` values are
compact column-major. Canonicalize a non-contiguous CUDA view through the
backend before entering this compact-only raw path; do not download it.

`session.stream()` exposes the same captured stream for a foreign CUDA library.
Its native handle and `TensorRef::raw_ptr` / `TensorMut::raw_ptr` are unsafe FFI
escapes: use them only inside the callback and never retain or destroy them.

## Ordering, synchronization, and safety

- Launch on the stream owned by the active raw session. Do not create a second
  stream behind tenferro's scheduler.
- Raw pointers, native stream handles, tensor spans, and workspace spans must
  not escape the session callback.
- A successful launch only enqueues work. Call `session.synchronize()` only
  when the host must wait; later work on the same stream is already ordered.
- Keep modules, inputs, outputs, and external-library resources alive until the
  last asynchronous use completes.
- Every `unsafe session.launch` must state the kernel ABI, argument order,
  element/span bounds, aliasing, initialization, and asynchronous lifetime
  proof.
- Mutable/output arguments must be exclusively borrowed and fully initialized
  before safe Rust reads them.
- There is no implicit CPU/GPU transfer or device-wide synchronization.

Compile-check the source-backed snippets without a GPU:

```bash
cargo check -p tenferro-tutorial-code --no-default-features \
  --features cuda,cpu-faer --bin custom_cuda_kernels
```

The CUDA tests `raw_ptx_load_launch_roundtrip` and
`raw_nvrtc_launch_roundtrip` compile these public loading and launch paths on
the non-GPU CUDA lane and execute them on CUDA hardware. CUBIN execution must
use a CI runner whose compute capability matches `CUDA_ARCH`.

## Device-side libraries

A downstream kernel may call device-side libraries such as cuSOLVERDx or
cuBLASDx:

```text
external CUDA kernel
  -> cuSOLVERDx/cuBLASDx device call
  -> nvcc or NVRTC
  -> raw Session module loading
  -> launch on the tenferro session stream
```

The library headers, link-time device code, architecture flags, and licenses
remain the downstream crate's responsibility. The tenferro ownership and
stream rules above do not change.
