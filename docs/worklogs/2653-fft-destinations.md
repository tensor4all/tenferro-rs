# FFT caller-owned destinations

## Decisions

- Add overwrite destinations to the existing concrete FFT family, retaining the
  result dtype and explicit placement rules. Keep eager/traced allocation
  behavior and the consuming eager exception unchanged.
- Retain prepared requests in the bounded caller-owned cache and reuse the CPU
  engine's pool, rather than creating a second scratch owner. The durable
  [destination contract](../design/cpu-fft-output-reuse.md#caller-owned-fft-destinations-2653)
  defines the warm-up and retention conditions.
- Reuse the provider's lane traversal for higher column-major axes. Preserve
  the explicit full-input gather boundary for noncompact views.

## Verification conclusions and constraints

- Global allocation counting records zero Rust heap allocations after warm-up
  for all eight FFT forms, both scalar widths, serial and four-worker CPU
  engines, and owned rank-nine tensors, in debug and release builds. The
  compact higher-axis case avoids a full gather; a transposed negative control
  observes the documented full-input gather.
- The complete FFT feature suite and doctests pass, as do the local PR gate,
  registry conformance, generated API snapshots, strict FFT rustdoc and docs-site
  inputs. Independent source review found no actionable defects. CUDA execution
  remains a hardware validation obligation.
- This macOS host has no NVIDIA CUDA device. CUDA device execution and the A100
  gate require a configured GPU host; host checks cannot establish that result.
