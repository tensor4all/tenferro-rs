# FFT (extension)

`tenferro-fft` is the FFT extension package for tenferro. It is an extension
crate imported directly alongside `tenferro-runtime`: users add the package,
import its `traced_tensor` functions, and use them with `TracedTensor` graphs.

The current implementation provides one-dimensional CPU-host transforms backed
by `rustfft` through tenferro extension operations. The public functions are
ordinary Rust wrappers, so most users do not need to work with the lower-level
extension machinery directly.

## Setup

```toml
[dependencies]
num-complex = "0.4"
tenferro-runtime = { path = "../crates/tenferro-runtime" }
tenferro-cpu = { path = "../crates/tenferro-cpu" }
tenferro-fft = { path = "../crates/tenferro-fft" }
```

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

<!-- snippet-source: crates/tenferro-fft/examples/traced_fft.rs -->
```rust
use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_fft::{traced_tensor, FftNorm};
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
    );
    let y = traced_tensor::fft(&x, None, -1, FftNorm::Backward)?;

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

AD examples require `tenferro-ad` and the `tenferro-fft` `autodiff` feature.
Add `tenferro-ad` and replace the basic `tenferro-fft` line with:

```toml
tenferro-ad = { path = "../crates/tenferro-ad" }
tenferro-fft = { path = "../crates/tenferro-fft", features = ["autodiff"] }
```

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
