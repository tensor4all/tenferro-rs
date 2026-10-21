# CPU activations use the vectorized composite, not the scalar fused hook (#2139)

## Decisions

- The CPU `Engine::fused_activation_read` hook implemented `sigmoid`, `silu`,
  `softplus`, `gelu` and `gelu_tanh` as a scalar per-element closure
  (`VectorKernel::None`). The shared composite evaluates the same formulas on
  the SIMD elementwise kernels, and the #2139 acceptance campaign measured the
  scalar traversal 2-6x slower at 1024x64 f32.
- The CPU backend now declines the fused hook. The trait default returns
  `Ok(None)`, so the eager and typed surfaces fall back to the vectorized
  composite; the `activation` module and its success test are removed. The
  decline test stays: the hook is still exercised for dtypes without a fused
  form, and a future SIMD fused kernel can re-enable the CPU implementation
  with a measurement behind it.
- The change is a policy change, not a semantic one: the composite and the
  fused kernel implement the same formulas (the test that was removed checked
  that equality). No public API changes.

## Measurement

Host: AMD Ryzen AI 9 HX 470 (12 physical cores / 24 threads), 1T measured on
core 0, release, provider `native` (faer). Harness: tenferro-benchmark
`cpu/perf_issues`; the eager single-call arms run the `cpu-followup-mwe`
reproducer. Reference: PyTorch 2.12.0+cpu.

| activation (1024x64 f32) | before | after | vs PyTorch before -> after |
|---|---:|---:|---:|
| sigmoid | 363.7 us | 166.8 us | 3.95x -> 1.81x |
| silu | 345.2 us | 182.9 us | 3.76x -> 1.98x |
| softplus | 984.5 us | 190.5 us | 4.86x -> 0.95x |
| gelu (exact) | 2110.1 us | 332.8 us | 27.16x -> 3.45x |
| gelu_tanh | 1509.2 us | 307.8 us | 6.28x -> 1.27x |
| erf | 287.4 us | 286.8 us | 1.46x -> 1.74x |

`erf` is unchanged because it already takes the primitive (vectorized) route.
The remaining `gelu` gap is the exact `erf` composite itself, not the fused
hook.

## Follow-ups (campaign gaps that this does not close)

- **softmax** is an eight-primitive composite with no fused kernel. It stays at
  1T 3.12x / 1.64x and 4T 5.64x / 3.23x of PyTorch for len64 / len512 and does
  not scale 1T -> 4T for len64. A fused CPU softmax (or a parallel reduction
  over the batch axis) is the next step.
- **decode_proj** at len 8/64 is 1.8-4.2x direct faer. An untracked eager
  `dot_general` fast path was prototyped and measured: 4.35x vs 4.17x before,
  so the eager recording is not the bottleneck; the gap is in the engine's
  `dot_general` dispatch/kernel choice for the small-len shape.
