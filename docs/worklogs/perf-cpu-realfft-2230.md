# realfft-based rfft/irfft and softmax coverage (#2230, #2228)

## realfft rfft/irfft

`tenferro-fft` used a full-length complex FFT with a conjugate mirror for the
real transforms. This adds `realfft = "3"` and runs the half-length real
transforms directly:

- `cache.rs`: a `RealPlan<T>` (forward `RealToComplex` / inverse
  `ComplexToReal`), cached under a `FftPlanKey` that now carries a `real` flag.
- `cpu/lanes.rs::execute_real`: gathers the real (R2C) or one-sided (C2R) lanes,
  runs the real plan, and scatters with the same normalization fold. `R2cFull`
  fills the conjugate half; the C2R drops the imaginary parts of the DC and
  Nyquist bins, which `irfft` ignores and `realfft` rejects.
- `cpu.rs` dispatches `R2cFull | R2cOnesided | C2r` to `execute_real`.

Measurements (1T, 524289-point fixtures, MKL DFTI reference):

| op | before | now | MKL | ratio |
|---|---:|---:|---:|---:|
| rfft | 2.32x | 8.08 ms | 4.34 ms | **1.86x** |
| irfft | 2.53x | 7.14 ms | 3.24 ms | **2.20x** |
| fft | — | 14.53 ms | 15.14 ms | 0.96x |
| ifft | — | 8.28 ms | 6.87 ms | 1.21x |

The residual is the MKL's specialized real FFT being faster than the
half-length complex FFT that `realfft` builds on; the <=1.3x acceptance in
#2230 is not met.

## softmax coverage

`softmax.rs` sat at 67.2% changed-line coverage. The new in-module tests cover
the scalar f32/f64 group fallbacks, the AVX2/scalar equivalence and the
`softmax_tensor` path directly, so the fused kernel keeps the file above the
80% gate.

## Tests

- `cargo test -p tenferro-fft` (21 + 1 + 21 pass)
- `cargo test -p tenferro-cpu --lib softmax` (oracle, masked/NaN, decline,
  scalar groups, AVX2 equivalence, `softmax_tensor`)
