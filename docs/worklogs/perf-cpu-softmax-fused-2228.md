# CPU softmax fused kernel and shared parallel chunking (#2228)

## Summary

Port the essential shape of PyTorch's `_vec_softmax_lastdim` to the CPU backend:
one kernel over a group of contiguous first-axis slices, vectorized AVX2 max
and scale, the SIMD `exp` over a cache-sized group buffer, and no full-size
intermediate. The composite's `-inf`/NaN clamps are preserved with a division
rather than a reciprocal multiply.

The cache-sized group and the small-problem worker cap live in the new
`tenferro-cpu/src/parallel.rs` (`group_len`, `task_len`) so other kernels can use
the same policy. `Engine::softmax_read` is a new optional hook; `Tensor::softmax`,
the typed tensor, and the eager session route through it and keep the composite
as the fallback.

## Measurements

1024x64 f32 attention softmax, 1T core 0 / 4T cores 0-3, same host:

| row | eager | session | PyTorch | eager ratio |
|---|---:|---:|---:|---:|
| 1T len64 | 64.1 us | 63.7 us | 68.0 us | 0.94x |
| 1T len512 | 2571.6 us | 2435.9 us | 2184.9 us | 1.18x |
| 4T len64 | 47.5 us | 45.5 us | 18.7 us | 2.54x |
| 4T len512 | 910.3 us | 905.3 us | 813.2 us | 1.12x |

The 1T acceptance passes for both sizes and the 4T len512 case passes; the 4T
len64 case (32k elements) remains launch/wrapper-bound at 2.54x. Capping the
workers at two for small tensors regressed it (58.6 us) and was rejected.

## Not included

- A fused SIMD activation hook: measured slower than the vectorized composite
  (sigmoid 237.8 vs 77.9 us, gelu 484.0 vs 332.8 us) and was reverted.
- rfft/irfft via `realfft`: still to do; the current full-length complex FFT
  path in `tenferro-fft/src/cpu/lanes.rs` is unchanged.
