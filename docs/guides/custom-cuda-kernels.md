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
        })?
    }
```
<!-- end-snippet-source -->

Inputs must already be uploaded to the same `CudaBackend` runtime. The safe
`tensor` and `tensor_mut` methods validate residency and return bounded device
spans. `tensor_mut` requires an exclusive tensor borrow; a fresh
`alloc_output` is the simplest output path. Owned `TypedTensor` values are
compact column-major. Canonicalize a non-contiguous CUDA view through the
backend before entering this compact-only raw path; do not download it.

Storage that only exists as a view — most visibly an eager value registered by
`constant_from_host`, whose `TensorRead::as_tensor()` is `None` — has no owned
`TypedTensor` to pass to `tensor`. Bind it with
`session.tensor_read::<f32>(&read)` instead: it resolves the view's base CubeCL
allocation, applies the view's logical offset, and returns only the view's own
span, without the `duplicate_value` device copy (#2084). That binding is also
compact column-major only, because a `TensorRef` is one base pointer plus one
byte length and carries no per-axis strides; a transposed or strided read is
rejected with a typed validation error instead of being materialized. Owned
reads passed to `tensor_read` keep using the `tensor` binding unchanged.

`session.stream()` exposes the same captured stream for a foreign CUDA library.
Its native handle and `TensorRef::raw_ptr` / `TensorMut::raw_ptr` are unsafe FFI
escapes: use them only inside the callback and never retain or destroy them.

## Large dynamic shared-memory tiles

Since Volta the driver caps a launch that requests more than 48 KiB of dynamic
shared memory unless the kernel opts in first. `raw::Session::launch` performs
that opt-in implicitly: the first launch whose `LaunchConfig::shared_mem_bytes`
exceeds 48 KiB raises
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` on the launched
`raw::Function`, and a later launch calls the driver again only when it asks for
more than the value already set. There is no separate opt-in flag and no
`Function` setter to call.

The device's opt-in ceiling is
`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`, which the CubeCL
session reports as `hardware.max_shared_memory_size` (166912 bytes on an A100,
101376 bytes on an RTX 3060):

<!-- snippet-source: docs/tutorial-code/src/bin/custom_cuda_kernels.rs#custom_cuda_smem_limit -->
```rust
    use tenferro_gpu::cuda::CudaExecSession;

    /// Dynamic shared memory a raw launch may opt into on this device, from
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`.
    fn max_shared_memory_bytes(
        session: &mut CudaExecSession<'_>,
    ) -> tenferro_tensor::Result<usize> {
        session.with_cubecl("custom_cuda.smem_limit", |cubecl| {
            Ok(cubecl.client().properties().hardware.max_shared_memory_size)
        })
    }
```
<!-- end-snippet-source -->

A request above that ceiling is not clamped: the driver rejects the attribute
and `launch` returns the typed `BackendSource` error
(`CUDA_ERROR_INVALID_VALUE`) before any kernel is enqueued. The CUDA tests
`raw_launch_uses_dynamic_shared_memory_above_48_kib` (a 64 KiB / 16384 f32 tile
launched on an A100, values checked after download) and
`raw_launch_rejects_dynamic_shared_memory_above_device_opt_in` (the typed
rejection above the ceiling) cover both paths (#2085).

## Ordering, synchronization, and safety

- Launch on the stream owned by the active raw session. Do not create a second
  stream behind tenferro's scheduler.
- Raw pointers, native stream handles, tensor spans, and workspace spans must
  not escape the session callback.
- A successful launch only enqueues work. Call `session.synchronize()` only
  when the host must wait; later work on the same stream is already ordered.
- A `#[cube]` launch issued *inside* a raw session is not ordered before the raw
  work that follows it; call `session.flush_cubecl()` after the launch, before
  the dependent raw work (#2092; see
  [Mixing `#[cube]` launches with raw work](#mixing-cube-launches-with-raw-work)).
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
  --features cuda,native --bin custom_cuda_kernels
```

The CUDA tests `raw_ptx_load_launch_roundtrip` and
`raw_nvrtc_launch_roundtrip` compile these public loading and launch paths on
the non-GPU CUDA lane and execute them on CUDA hardware. CUBIN execution must
use a CI runner whose compute capability matches `CUDA_ARCH`.

## Downstream `#[cube]` kernels

The same session seam also hosts kernels written with CubeCL's `#[cube]` macro
instead of PTX/CUBIN/NVRTC source. A downstream crate declares the framework
`t4a-cubecl` package (same version tenferro resolves, so there is one CubeCL in
the graph) and launches the kernel through the generated `launch_unchecked`
entry point. `samples/cubecl-kernel` is a runnable fixture: it defines this
page's kernels outside `tenferro-gpu`, and its tests launch all of them on CUDA
hardware.

```text
CudaBackend::with_backend_session
  -> with_cuda_exec_session
  -> CudaExecSession::with_cubecl
  -> cuda::cubecl::Session::client
  -> #[cube(launch_unchecked)] kernel launch
```

### Allocate, launch, download — the canonical example

The four things the CubeCL launch API leaves implicit are all named in the
fixture's `run_scale_in_session` (`samples/cubecl-kernel`, executed by
`tests/scale_kernel_test.rs`):

* **The output** comes from `Session::alloc_output::<T>(&[n])`. There is no need
  to upload a zero staging tensor for it.
* **The launch domain** comes from `Session::cube_count_1d(n)` — the
  `CubeCount::Static` of 256-unit cubes that covers `n` elements, one work unit
  per element — and `Session::cube_dim_1d()` (`CubeDim::new_1d(256)`). It rounds
  up: the domain is exactly `n` units only when `n` is a multiple of the cube
  dim, so any other `n` needs the kernel's own `ABSOLUTE_POS < n` guard. A
  single-cube kernel uses `CubeCount::new_single()` with
  `CubeDim::new_1d(units)` instead, which covers exactly `units` and is how the
  fixture's gather/reverse kernels are launched.
* **The arguments** are the `BufferArg` values `Session::array_arg` returns, in
  kernel-parameter order. `KernelArguments` is never built by hand: the
  generated `launch_unchecked` builds it from the typed parameter list. A
  `BufferArg` is not `Clone`, so a buffer passed twice needs two `array_arg`
  calls.
* **The compiled kernel value** never appears in caller code.
  `launch_unchecked` takes the `Client` from `Session::client()` plus the
  `CubeCount` and `CubeDim`; CubeCL compiles the kernel on first use and reuses
  the artifact (see
  [Caching the compiled kernel](#caching-the-compiled-kernel)).

`with_cubecl` flushes the launches it holds on exit (including on `Err` and
unwind) but does not wait for the device. A caller that needs the values calls
`CudaExecSession::synchronize` — the only host barrier — and then
`download_tensor`:

<!-- snippet-source: samples/cubecl-kernel/src/run.rs#downstream-cube-launch -->
```rust
/// Upload `input`, allocate an output on the session, launch [`scale`], and
/// download the result.
///
/// The three launch inputs the CubeCL API does not make obvious are all named
/// here: the output comes from `Session::alloc_output`, the launch domain from
/// `Session::cube_count_1d` plus `Session::cube_dim_1d`, and the kernel
/// arguments are the `BufferArg` values `Session::array_arg` hands back —
/// `KernelArguments` is never built by hand.
///
/// The runtime handle is cloned out of the session first so uploads do not
/// hold a borrow of `exec` while the mutable `with_cubecl`/`synchronize`
/// methods run.
///
/// `n` must be a multiple of the 256-unit cube dim: `cube_count_1d` rounds the
/// domain up, and `scale` has no `ABSOLUTE_POS < n` guard, so only an exact
/// domain keeps it in bounds.
fn run_scale_in_session(
    op: &'static str,
    exec: &mut CudaExecSession<'_>,
    host_input: TypedTensor<f32>,
    n: usize,
) -> tenferro_tensor::Result<TypedTensor<f32>> {
    let runtime = exec.runtime().clone();
    let input = upload_tensor(&runtime, &Tensor::from_typed::<f32>(host_input))?;
    let input = input.into_typed::<f32>().expect("f32 upload stays f32");

    let output = exec.with_cubecl(op, |cubecl| {
        // SAFETY: both bindings are created for tensors this call owns, and the
        // kernel writes only `output`.
        let input_arg: BufferArg = unsafe { cubecl.array_arg(&input, op)? };
        // Session-owned output allocation: no host staging tensor is uploaded.
        let output = cubecl.alloc_output::<f32>(&[n])?;
        // SAFETY: as above; `output` is exclusively owned here.
        let output_arg: BufferArg = unsafe { cubecl.array_arg(&output, op)? };

        let CubeCount::Static(x, y, z) = cubecl.cube_count_1d(n)? else {
            return Err(Error::runtime_state(op, "expected static cube count"));
        };
        // SAFETY: the kernel reads `input[i]` and writes `output[i]` only for
        // `i < n`; `n` is a multiple of the cube dim, so the rounded-up domain
        // above covers exactly `n` elements and never overruns either buffer.
        unsafe {
            scale::launch_unchecked(
                cubecl.client(),
                CubeCount::Static(x, y, z),
                cubecl.cube_dim_1d(),
                input_arg,
                output_arg,
                2.0f32,
            );
        }
        Ok(output)
    })?;

    // Explicit host barrier, then download.
    exec.synchronize()?;
    Ok(download_tensor(&runtime, &Tensor::from_typed(output))?
        .into_typed::<f32>()
        .expect("f32 download"))
}
```
<!-- end-snippet-source -->

### Mixing `#[cube]` launches with raw work

A `#[cube]` launch is handed to CubeCL's device thread, while raw kernels,
vendor calls, and transfers are enqueued from the caller's thread onto the same
stream. A raw enqueue that follows a launch issued inside the same raw session
can therefore reach the stream first and read the value the launch was about to
overwrite (tenferro-rs #2092). `cuda::raw::Session::flush_cubecl` hands the
queued launches to CubeCL's device thread and returns without reading a result
back, so it is not a host barrier for device completion:

<!-- snippet-source: samples/cubecl-kernel/src/run.rs#custom_cuda_cubecl_ordering -->
```rust
/// Launch [`scale`], then run raw work that depends on its output.
///
/// The fence is what orders the raw continuation after the launch; see the
/// external-kernel guide for the mechanism.
fn launch_then_raw(
    exec: &mut CudaExecSession<'_>,
    client: &Client,
    input: BufferArg,
    output: BufferArg,
) -> tenferro_tensor::Result<()> {
    exec.with_raw("downstream.hybrid", |raw| {
        // SAFETY: the caller owns both bindings, the launch domain matches their
        // element counts, and the kernel writes only `output`.
        unsafe {
            scale::launch_unchecked(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                input,
                output,
                2.0f32,
            );
        }
        // Dispatch the launch onto this session's stream before the raw work
        // that depends on it: without this fence the raw kernel below can reach
        // the stream first (#2092).
        raw.flush_cubecl()?;
        // The raw continuation reads `output` here: an NVRTC/PTX launch, a
        // vendor call, or a transfer.
        Ok(())
    })
}
```
<!-- end-snippet-source -->

Entering a raw session dispatches the launches queued before the scope, and
leaving a `with_cubecl` scope dispatches those issued inside it, so the fence is
needed only for a launch issued after the raw scope was entered. The fence
blocks the host while CubeCL's device thread takes the queue (and while CubeCL
retires dropped buffers), so keep it to the raw operations that depend on a
launch. It does not report a failed launch: that failure surfaces on the buffer
the work left unwritten. Call `session.synchronize()` when the host must wait
for device completion, and `download_tensor` when it needs the values.

### `#[cube]` typing rules on this pin

The sharp edge that costs downstream time is the index type. Buffer indexing is
defined only for `usize` and `usize` ranges: the pinned `t4a-cubecl 0.11.0`
implements `IndexExpand`/`IndexMutExpand` for `NativeExpand<usize>` (and the
`usize` range forms), but for no other integer type. Three rules decide whether
a kernel compiles; all three are exercised by the fixture kernels, which the
CUDA test lane launches:

* `usize::cast_from(value)` is the conversion, and it accepts **signed**
  sources: `usize::cast_from(indices[i])` on an `&[i32]` buffer compiles and
  runs, as the fixture's `gather` does. The `Cast` trait is generic over any
  `CubePrimitive` source on this pin, so the earlier #2119 report that
  `cast_from` is `From`-based and rejects `i32 -> usize` does not describe it.
* An `i32` index buffer is the only shape a downstream can bind: tenferro
  tensors carry `f32`, `f64`, `i32`, `i64`, `bool`, `C32` and `C64` payloads, so
  there is no `u32`/`u64` tensor to hand `Session::array_arg`. Keep index
  buffers `i32` (or `i64`) and cast inside the kernel.
* `ABSOLUTE_POS` is already `usize`, and an integer literal is inferred as
  `usize` where an index is expected: the fixture's `scale` indexes with
  `ABSOLUTE_POS` (`input[i]`), and `reverse_tile` uses an unsuffixed literal
  (`count - 1 - unit`). A literal written straight into an index, such as
  `output[0]` or a shared-memory `tile[0]`, compiles the same way. `UNIT_POS`,
  `CUBE_POS`, `CUBE_DIM` and the other topology constants are `u32` and need the
  same `usize::cast_from`.

<!-- snippet-source: samples/cubecl-kernel/src/kernel.rs#downstream-cube-index -->
```rust
/// Gather `input[usize::cast_from(indices[i])]` for each `i`.
///
/// `indices` is an `i32` buffer, so its elements are not `usize`; the explicit
/// `usize::cast_from` is what makes them usable as indices, and it accepts a
/// signed source on this pin. `ABSOLUTE_POS` is already `usize`, and an index
/// literal is inferred as `usize`.
///
/// An `i32` buffer is also the only shape a downstream can bind here: tenferro
/// tensors carry `f32`, `f64`, `i32`, `i64`, `bool`, `C32` and `C64` payloads,
/// so there is no `u32`/`u64` tensor to hand `Session::array_arg`.
///
/// Bounds contract: the launcher sizes the domain to `indices.len()`, `output`
/// must be at least `indices.len()` long, and every stored index must be in
/// `0..input.len()`: the cast does not bounds-check, so a negative value becomes
/// a huge unsigned offset rather than an error.
#[cube(launch_unchecked)]
pub fn gather(input: &[f32], indices: &[i32], output: &mut [f32]) {
    let i = ABSOLUTE_POS;
    let src = usize::cast_from(indices[i]);
    output[i] = input[src];
}
```
<!-- end-snippet-source -->

<!-- snippet-source: samples/cubecl-kernel/src/kernel.rs#downstream-cube-shared -->
```rust
/// Reverse one cube-sized tile of `input` through shared memory.
///
/// `UNIT_POS` is a `u32` (as are `CUBE_POS`, `CUBE_DIM` and the other topology
/// constants), so it cannot index a buffer directly: `usize::cast_from` is the
/// conversion. `ABSOLUTE_POS` is already `usize`, and an integer literal is
/// inferred as `usize` where an index is expected, so `count - 1 - unit` needs no
/// suffix and no cast.
///
/// Bounds contract: the launcher runs exactly one cube of `len` units with
/// `len <= 1024` (the shared-memory slice length), `input` holds at least `len`
/// elements, and `output` holds at least `len`.
#[cube(launch_unchecked)]
pub fn reverse_tile(input: &[f32], output: &mut [f32], len: u32) {
    let mut tile = Shared::<[f32]>::new_slice(1024usize);
    let unit = usize::cast_from(UNIT_POS);
    let count = usize::cast_from(len);
    if unit < count {
        tile[unit] = input[unit];
    }
    sync_cube();
    if unit < count {
        output[unit] = tile[count - 1 - unit];
    }
}
```
<!-- end-snippet-source -->

The cast does not bounds-check: a negative index becomes a huge unsigned offset,
not a typed error, so the launcher owns the `0..len` contract. The failure mode
for a typed non-`usize` index is a pair of errors, one with the offending
expression's span and one naming only the `#[cube]` attribute (trimmed from a
probe kernel `output[0] = input[offset]` with `offset: i32`):

```text
error[E0277]: the type `[f32]` cannot be indexed by `i32`
    |     output[0] = input[offset];
    |                       ^^^^^^ slice indices are of type `usize` or ranges of `usize`
    = help: the trait `SliceIndex<[f32]>` is not implemented for `i32`

error[E0277]: the trait bound `NativeExpand<[f32]>: t4a_cubecl::frontend::IndexExpand<NativeExpand<i32>>` is not satisfied
    | #[cube(launch_unchecked)]
    | ^^^^^^^^^^^^^^^^^^^^^^^^^ the trait `t4a_cubecl::frontend::IndexExpand<NativeExpand<i32>>` is not implemented for `NativeExpand<[f32]>`
```

`output[UNIT_POS]` and `input[UNIT_POS]` fail the same way with `u32` in place
of `i32`: a `[f32] cannot be indexed by u32` error on each expression, and the
`IndexExpand`/`IndexMutExpand` pair on the attribute (a read-only index produces
only `IndexExpand`). The `= help:` list of an `IndexExpand` error is the whole
rule in compiler form: it enumerates `NativeExpand<usize>` and the `usize`
range impls, and nothing else. When a kernel fails that way, scan every index
expression rather than the line the attribute points at. Inside the DSL
`UNIT_POS as usize` is rewritten by the macro to the same `Cast` call as
`usize::cast_from(UNIT_POS)` on this pin, so both work; `cast_from` is the
documented form and states the source type explicitly.

### Caching the compiled kernel

Nothing has to be cached downstream, and #2117's "cache the compiled kernel"
ask needs no helper: CubeCL keeps the compiled artifact in its per-device
server. The client `Session::client()` hands out was created as
`Client::load::<CudaServer>(device_id)` (`Runtime::client` for the CUDA
runtime), which reaches the process-global server for that device. On that
server the CUDA context owns the loader — `CudaContext::kernels:
KernelLoader<CudaModules>` — and the loader keys what it compiled by
`ArtifactId { kernel: KernelId, .. }`, where a `KernelId` is the kernel's type
identity plus its `CubeDim`, address type and execution mode. A repeated
identity is one lookup and a clone of the loaded kernel; nothing compiles
again. A kernel compiled inside one `with_cubecl` scope is therefore reused by
every later scope, and by a second `CudaBackend` constructed for the same
device, because both reach the same context and loader.

The fixture's `run_kernel_cache_check` measures exactly that: it launches the
same kernel from six successive `with_cubecl` scopes on one backend and then
from a freshly constructed backend. Measured on an A100 (CUDA 12.6, debug
profile, one test thread so the other fixture tests do not share the device):

| scope | host cost |
| --- | --- |
| backend 0, launch 0 (JIT compile) | 77.8 ms |
| backend 0, launches 1-5 | 499 µs, 239 µs, 215 µs, 214 µs, 216 µs |
| backend 1 (fresh `CudaBackend`, first launch) | 279 µs |

Only the first scope pays the ~78 ms compile (three consecutive runs: 77.8 ms,
77.5 ms, 78.9 ms). The second scope is the expensive repeat at ~500 µs
(499 µs, 495 µs, 554 µs over the same runs); from the third scope on the cost
settles at ~200–280 µs, and a fresh backend for the same device pays about the
same as a warm scope (268–298 µs over five runs), so the compiled artifact
outlives the backend object that first produced it.

Two consequences follow from the kernel identity. A kernel whose `CubeDim`
changes is a *different* kernel and compiles again, so fix one `CubeDim` per
kernel and size `CubeCount` instead (a 1-D domain of `cube_dim_1d()` is a good
default). And a cached scope is not free: the few hundred µs above is the
scope's own work — output allocation, launch hand-off, exit flush — not
compilation, which is why a latency-bound sampler batches many launches into
one scope rather than opening one scope per launch.

What is not needed, so do not build it: a downstream `HashMap` of compiled
kernels (the loader already is that map), a `Send + Sync` handle to the compiled
kernel (CubeCL never returns one to a caller), or a persistent `Session` — a
`Session` is `!Send + !Sync` and borrows the runtime for exactly one scope by
design. Enter `with_cubecl` per batch of work; the compile cost is paid once per
process and device. CubeCL can also keep the compiled PTX across processes in
its on-disk artifact store, but that is off unless `compilation.cache` is
enabled in the CubeCL runtime config, so do not plan on it.

### Running the fixture

The fixture's CUDA tests are the device gate for everything above: they are
`#[ignore]` so a GPU-less runner can still compile them, and the `extensions`
profile of `scripts/ci/run_profile.py` only compiles and runs the fixture's
non-ignored targets:

```bash
# CI parity (no device needed): compiles and tests the fixture.
cargo test --profile ci --manifest-path samples/cubecl-kernel/Cargo.toml --all-targets

# The device gate: the ignored tests launch every kernel above on CUDA.
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.6 \
LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test --manifest-path samples/cubecl-kernel/Cargo.toml --test scale_kernel_test \
  -- --ignored --nocapture --test-threads=1
```

`scale_kernel_runs_end_to_end` verifies the allocate/launch/download values,
`typing_rule_kernels_run_end_to_end` launches both typing-rule kernels, and
`compiled_kernel_is_reused_across_scopes_and_backends` prints the cache table
above. `--test-threads=1` keeps the timings meaningful; each test skips itself
when no CUDA device is reachable.

## CUDA toolkit minimum by path

tenferro compiles against the CUDA 12.8 `cudarc` bindings and resolves driver
and NVRTC entry points at runtime, so the floor depends on the path:

| Path | Supported minimum | Full-capability tier |
| --- | --- | --- |
| Custom `#[cube]` kernels (CubeCL JIT through NVRTC) | CUDA 12.6.2 | CUDA 12.8+ enables the complete CubeCL feature set |
| `raw` `load_ptx` / `load_cubin` | CUDA 12.6.2 | — |
| `raw` `compile_nvrtc` | CUDA 12.6.2 (NVRTC) | CUDA 12.8 NVRTC on a 12.8+ driver |
| cuTENSOR-backed operations | CUDA 12 cuTENSOR 2.x (2.1.x on compute capability 7.0) | — |

CUDA 12.6.2 is the baseline this project supports and tests. It is a library
floor, not a GPU one: CUDA 12.x supports the same GPU architectures throughout,
and 12.6.2 is what the CI-tested cuSOLVER provides (`cusolverDnXsyevBatched`
needs cuSOLVER >= 11.7.1, which is a requirement of the linalg provider rather
than of the kernel or `raw` paths in the table above). The 12.8-only CubeCL
features are advertised from the toolkit the crate is *compiled* against, not
from the loaded driver: the pinned `t4a-cubecl-cuda 0.11.0` gates them on
`cudarc::driver::sys::n` (`n >= 12080`), the `CUDA_VERSION` constant, so a build
against CUDA 12.4 through 12.7 does not offer them even on a 12.8 driver, and a
12.8-only tensor-map format that is reached anyway fails with an explicit reason
in `compute/server.rs` ("CUDA version 12.8 required for tensor map format
Im2ColWide"). See the
[Devices and GPU](devices-and-gpu.md#cuda-and-nvidia-library-compatibility)
compatibility matrix for the exact loader ABI and CI-tested packages.

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
