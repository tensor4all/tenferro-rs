# FFT (extension)

`tenferro-fft` is the FFT extension package for tenferro. It is an extension
crate imported directly alongside `tenferro-runtime` or `tenferro-tensor`.
Concrete non-AD execution uses `TensorFftExt` and `TensorReadFftExt`; eager
execution uses `EagerTensorFftExt` behind `autodiff`; traced graphs use
`TracedTensorFftExt`.

The current implementation provides one-dimensional CPU transforms backed by
RustFFT and an explicitly selected Apple Metal path backed by CubeK. The public API is ordinary
Rust extension-trait methods, so most users do not need to work with the
lower-level extension machinery directly.

## Setup

When working from a local checkout, use paths that match your project layout.
For a scratch crate created directly inside the `tenferro-rs` checkout, include
an empty `[workspace]` table:

```toml
[workspace]
```

Then add the dependencies:

```toml
[dependencies]
num-complex = "0.4"
tenferro-runtime = { path = "../crates/tenferro-runtime" }
tenferro-tensor = { path = "../crates/tenferro-tensor" }
tenferro-cpu = { path = "../crates/tenferro-cpu" }
tenferro-ad = { path = "../crates/tenferro-ad" }
tenferro-fft = { path = "../crates/tenferro-fft", features = ["autodiff"] }
```

The Apple shared path is not released yet. Until a later release task publishes
it, use the tested `8ffcc57b` revision and the checkout-relative path
dependencies below:

```bash
git clone https://github.com/tensor4all/tenferro-rs.git
cd tenferro-rs
git checkout 8ffcc57b
```

The tenferro workspace at that revision already pins the reviewed CubeCL and
CubeK Git revisions; applications do not need to declare CubeCL or CubeK
directly.

Concrete and graph-only users can omit `tenferro-ad` and the `autodiff`
feature. Enable `tenferro-fft`'s `autodiff` feature when registering FFT AD
rules. `rustfft` is pulled in automatically by `tenferro-fft`, and the first
local build can take a few minutes on a fresh machine.

For the Apple shared CPU/Metal path, also enable the WebGPU feature on both
operation and backend crates:

```toml
[dependencies]
tenferro-fft = { path = "../crates/tenferro-fft", features = ["webgpu"] }
tenferro-gpu = { path = "../crates/tenferro-gpu", default-features = false, features = ["webgpu"] }
tenferro-cpu = { path = "../crates/tenferro-cpu", default-features = false, features = ["cpu-faer"] }
tenferro-linalg = { path = "../crates/tenferro-linalg", default-features = false, features = ["cpu-faer"] }
tenferro-tensor = { path = "../crates/tenferro-tensor" }
```

`tenferro-linalg` and the CPU provider are needed by the Cholesky tutorial;
FFT-only applications may omit `tenferro-linalg`.

## Current API

The initial API mirrors the common PyTorch and JAX one-dimensional FFT
families:

| Operation family | Purpose |
| --- | --- |
| `fft`, `ifft` | complex-to-complex transforms; real input may be promoted to complex output |
| `rfft`, `irfft` | real-to-complex and complex-to-real one-dimensional transforms |

Each function accepts an optional transform length `n`, an `axis`, and an
`FftNorm` value. Negative axes are normalized relative to the input rank. The
normalization modes are:

| Mode | Behavior |
| --- | --- |
| `FftNorm::Backward` | forward unscaled, inverse scaled by `1 / n` |
| `FftNorm::Forward` | forward scaled by `1 / n`, inverse unscaled |
| `FftNorm::Ortho` | forward and inverse scaled by `1 / sqrt(n)` |

`Backward` is the default and matches NumPy, PyTorch, and JAX.

### Concrete Tensor And TensorRead

Use `TensorFftExt` when you have an owned compact `Tensor` and want immediate
non-AD execution on an explicit backend. Use `TensorReadFftExt` when the input
is a borrowed view or other read-oriented value. The `_read` suffix is reserved
for that `TensorRead` surface; compact `Tensor` inputs use unsuffixed method
names.

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#tenferro_fft_22 -->
```rust
use num_complex::Complex64;
use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_fft::{FftNorm, TensorFftExt, TensorReadFftExt};
use tenferro_runtime::BackendSessionHost;
use tenferro_tensor::{Tensor, TensorRead, TensorView, TypedTensorView};

let mut backend = CpuBackend::new();
backend.with_backend_session(|session| {
    with_cpu_exec_session(session, |backend| -> Result<(), tenferro_tensor::Error> {
        let x = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?;
        let full = x.fft(None, -1, FftNorm::Backward, backend)?;
        let one_sided = x.rfft(None, -1, FftNorm::Backward, backend)?;
        assert_eq!(full.as_slice::<Complex64>()?[0], Complex64::new(10.0, 0.0));
        assert_eq!(one_sided.shape(), &[3]);

        let data = [1.0_f64, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
        let view = TypedTensorView::from_slice([4], [2], 0, &data)?;
        let read = TensorRead::from_view(TensorView::F64(view));
        let read_full = read.fft_read(None, -1, FftNorm::Backward, backend)?;
        assert_eq!(
            read_full.as_slice::<Complex64>()?[0],
            Complex64::new(10.0, 0.0),
        );
        Ok(())
    })
    .ok_or_else(|| tenferro_tensor::Error::BackendFailure {
        op: "documentation",
        message: "CPU execution session is unavailable".to_owned(),
    })?
})?;
```
<!-- end-snippet-source -->

`TypedTensor<T>` wrappers are not part of the current API. FFT operations can
change dtype (`rfft` real to complex, `irfft` complex to real), so typed return
contracts need a separate design.

### Eager Tensors

Use `EagerTensorFftExt` for immediate execution in an `EagerRuntime`. The
methods have the same names and arguments as `TracedTensorFftExt`, register the
FFT execution runtime on demand, and record the existing extension operation
when gradients are enabled.

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#tenferro_fft_23 -->
```rust
use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_fft::{EagerTensorFftExt, FftNorm};

let x = EagerTensor::from_tensor_in(
    Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?,
    EagerRuntime::new()?,
)?;
let spectrum = x.rfft(None, -1, FftNorm::Backward)?;
let restored = spectrum.irfft(Some(4), -1, FftNorm::Backward)?;

assert_eq!(spectrum.shape(), &[3]);
assert_eq!(restored.to_tensor()?.as_slice::<f64>()?, &[1.0, 2.0, 3.0, 4.0]);
```
<!-- end-snippet-source -->

### Apple shared CPU and Metal execution

`AppleContext` makes allocation ownership shared, not backend selection
implicit. Clone its mutable backend handles and pass the desired backend to
each FFT call:

<!-- snippet-source: docs/tutorial-code/src/bin/math_snippets.rs#tenferro_fft_24 -->
```rust
use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_fft::{FftNorm, TensorFftExt};
use tenferro_gpu::{
    apple::AppleContext,
    webgpu::with_webgpu_exec_session,
};
use tenferro_runtime::BackendSessionHost;
use tenferro_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = AppleContext::new()?;
    let host = Tensor::from_vec_col_major([8], vec![1.0_f32; 8])?;
    let managed = context.upload_tensor(&host)?;
    let after_creation = context.transfer_stats();

    let mut cpu = context.cpu_backend().clone();
    let cpu_spectrum = cpu
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |exec_session| {
                managed.rfft(None, 0, FftNorm::Backward, exec_session)
            })
            .expect("CpuBackend must expose a CPU execution session")
        })?;

    let mut metal = context.metal_backend().clone();
    let metal_spectrum = metal
        .with_backend_session(|session| {
            with_webgpu_exec_session(session, |exec_session| {
                managed.rfft(None, 0, FftNorm::Backward, exec_session)
            })
            .expect("WebGpuBackend must expose a WebGPU execution session")
        })?;
    metal.synchronize()?;

    assert_eq!(cpu_spectrum.shape(), metal_spectrum.shape());
    assert_eq!(context.transfer_stats(), after_creation);
    Ok(())
}
```
<!-- end-snippet-source -->

RustFFT supports managed `F32`, `F64`, `C32`, and `C64` tensors. The initial
CubeK Metal implementation supports C32 CFFT/IFFT, F32 one-sided RFFT, and C32
IRFFT for power-of-two transform lengths of at least 2. RFFT/IRFFT may pad or
truncate to a supported requested length; CFFT cannot change the input-axis
length. Full-spectrum real FFT, `F64`/`C64`, non-power-of-two lengths, foreign
domains, and ordinary device-local WebGPU buffers return typed errors. There
is no automatic RustFFT fallback.

The complete runnable tutorial additionally proves stable input allocation
identity across CPU/Metal/CPU use, equal CPU/Metal results, unchanged
post-creation transfer counters, C64 CPU success, and typed C64 Metal failure:

```bash
cargo test -p tenferro-tutorial-code --no-default-features \
  --features cpu-faer,apple-shared,doc-snippets --test tutorial_binaries
```

Source: `docs/tutorial-code/src/bin/apple_shared_fft.rs`.

### Traced Graphs

<!-- snippet-source: crates/tenferro-fft/examples/traced_fft.rs -->
```rust
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_fft::{FftNorm, TracedTensorFftExt};
use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};

fn cpu_runtime_with_fft() -> Result<Runtime, Box<dyn std::error::Error>> {
    let backend = CpuBackend::new();
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend)?)?;
    builder.install_extension_module(tenferro_fft::extension_module::<CpuBackend>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    Ok(builder.build()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = TracedTensor::from_vec_col_major(
        vec![4],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let y = x.fft(None, -1, FftNorm::Backward)?;

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y)?;
    let runtime = cpu_runtime_with_fft()?;
    let mut outputs = runtime.run_compiled(&program, &[])?;
    assert_eq!(outputs.len(), 1);
    let out = outputs.remove(0);
    assert_eq!(out.shape(), &[4]);
    assert_eq!(
        out.as_slice::<Complex64>().unwrap(),
        &[
            Complex64::new(10.0, 0.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-2.0, -2.0),
        ],
    );

    Ok(())
}
```
<!-- end-snippet-source -->

For real-input transforms, the transformed axis follows the standard
half-spectrum shape rule: input length `n` produces `n / 2 + 1` complex values
using integer division. When `irfft` receives `n = None`, it infers the output
length as `2 * (input_len - 1)`. That matches even-length round trips; for odd
original lengths it silently returns one element too short, so pass
`Some(original_len)`.

## Planned Extensions

The remaining FFT families are planned but not part of the initial API:

| Operation family | Purpose |
| --- | --- |
| `fftn`, `ifftn` | multidimensional complex transforms |
| `rfftn`, `irfftn` | multidimensional real/half-spectrum transforms |
| `fft2`, `ifft2`, `rfft2`, `irfft2` | two-dimensional convenience wrappers |

## Compatibility Target

The compatibility target is the behavior users expect from:

- `torch.fft.fft`, `torch.fft.ifft`, `torch.fft.rfft`, `torch.fft.irfft`,
  and their `n`/`2` variants,
- `jax.numpy.fft.fft`, `jax.numpy.fft.ifft`, `jax.numpy.fft.rfft`,
  `jax.numpy.fft.irfft`, and their `n`/`2` variants.

The extension should normalize axes and lengths before execution, then
return results in the same logical axis order as the input. Backend-specific
layout or transposition needed to call an FFT implementation should stay inside
the extension.

## Automatic Differentiation

FFT is linear, so the extension can support AD through registered extension
rules. The current package registers JVP/VJP rules for complex-to-complex
`fft` and `ifft`: the tangent or cotangent is transformed with the same
extension op and normalization.

Use `AdContext` for explicit extension-rule ownership, or import
`tenferro_ad::TracedTensorAdExt` for the compact traced AD method syntax.
For eager AD, construct the runtime with
`EagerRuntime::with_cpu_backend_and_ad_context` using the same `AdContext`.

Real-to-complex and complex-to-real AD are not enabled yet. They require the
usual Hermitian symmetry handling so cotangents match the half-spectrum
convention; until those rules are implemented and tested, AD through `rfft` and
`irfft` reports an unsupported operation instead of returning an incorrect
gradient.

## Status

`tenferro-fft` currently lives in the top-level `tenferro-fft` crate. It
supports 1D `fft`, `ifft`, `rfft`, and `irfft` through CPU RustFFT on host or
matching Apple managed tensors, plus the narrower CubeK Metal matrix described
above. CUDA/cuFFT and multidimensional transforms remain future work.

For the general extension mechanism, see
[Custom Tensor Operations](custom-operations.md).
