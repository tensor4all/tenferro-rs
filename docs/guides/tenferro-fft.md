# FFT (extension)

`tenferro-fft` is the FFT extension package for tenferro. It is an extension
crate imported directly alongside `tenferro-runtime` or `tenferro-tensor`.
Concrete non-AD execution uses `TensorFftExt` and `TensorReadFftExt`; traced
graphs use `TracedTensorFftExt`.

The current implementation provides one-dimensional CPU-host transforms backed
by `rustfft` through tenferro extension operations. The public API is ordinary
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

For published crates, use the same crate set with version requirements:

```toml
[dependencies]
num-complex = "0.4"
tenferro-runtime = "..."
tenferro-tensor = "..."
tenferro-cpu = "..."
tenferro-ad = "..."
tenferro-fft = { version = "...", features = ["autodiff"] }
```

Concrete and graph-only users can omit `tenferro-ad` and the `autodiff`
feature. Enable `tenferro-fft`'s `autodiff` feature when registering FFT AD
rules. `rustfft` is pulled in automatically by `tenferro-fft`, and the first
local build can take a few minutes on a fresh machine.

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

```rust
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_fft::{FftNorm, TensorFftExt, TensorReadFftExt};
use tenferro_tensor::{Tensor, TensorRead, TensorView, TypedTensorView};

let mut backend = CpuBackend::new();
let x = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])?;
let full = x.fft(None, -1, FftNorm::Backward, &mut backend)?;
let one_sided = x.rfft(None, -1, FftNorm::Backward, &mut backend)?;
assert_eq!(full.as_slice::<Complex64>()?[0], Complex64::new(10.0, 0.0));
assert_eq!(one_sided.shape(), &[3]);

let data = [1.0_f64, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
let view = TypedTensorView::from_slice([4], [2], 0, &data)?;
let read = TensorRead::from_view(TensorView::F64(view));
let read_full = read.fft_read(None, -1, FftNorm::Backward, &mut backend)?;
assert_eq!(read_full.as_slice::<Complex64>()?[0], Complex64::new(10.0, 0.0));
# Ok::<(), tenferro_tensor::Error>(())
```

`TypedTensor<T>` wrappers are not part of the current API. FFT operations can
change dtype (`rfft` real to complex, `irfft` complex to real), so typed return
contracts need a separate design.

### Traced Graphs

<!-- snippet-source: crates/tenferro-fft/examples/traced_fft.rs -->
```rust
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_fft::{FftNorm, TracedTensorFftExt};
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};

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
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor.register_extension(tenferro_fft::register_runtime)?;
    let out = executor.run(&program)?;
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

Real-to-complex and complex-to-real AD are not enabled yet. They require the
usual Hermitian symmetry handling so cotangents match the half-spectrum
convention; until those rules are implemented and tested, AD through `rfft` and
`irfft` reports an unsupported operation instead of returning an incorrect
gradient.

## Status

`tenferro-fft` currently lives in the top-level `tenferro-fft` crate. It
supports 1D `fft`, `ifft`, `rfft`, and `irfft` on host tensors. CUDA/cuFFT and
multidimensional transforms remain future work.

For the general extension mechanism, see
[Custom Tensor Operations](custom-operations.md).
