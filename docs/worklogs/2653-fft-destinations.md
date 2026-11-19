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
- After macOS CI observed two allocations, test setup now uses threadteam's
  documented pre-reservation parking warm-up to keep Rayon's first-sleep
  initialization outside measurement. This is a plausible source of the CI
  noise, not a traced attribution; process-wide counting and zero-allocation
  assertions remain unchanged. The repaired fixture passes CI-profile, debug
  and release checks locally with effective CPU widths one and four.
- Lane boundary regressions now check invalid transform lengths, mismatched
  real buffers and empty batches for unchanged output and zero plan/scratch
  work. Their compiled tests are classified as host tests in the CUDA inventory.
- A test-only mismatched real FFT plan exercises serial/parallel provider
  rejection before scatter, verifying both scratch slabs are returned and can
  be checked out again with unchanged pool accounting. It tests error cleanup,
  not general rollback of partially completed transforms.
- Rust 1.100 FFT coverage with autodiff passes all 160 tests and measures
  82.5% line coverage for the lane implementation, above its existing 78%
  threshold. The complete workspace coverage build exceeded local free disk;
  its hosted counterpart remains required.
