# Task 1 report: Apple shared allocation domain

## Implemented

- Pinned CubeCL to `11b52669f13e27bbe188f988fd696df6d989a562`
  and CubeK to `fb5dbc8e994bb3023bfefe38abe0140f29cbb15e`.
- Added process-unique allocation domain IDs, backend allocation IDs, typed
  guarded reads, write-only guarded writes, and typed host-access failures.
- Added the object-safe `SharedTensorAllocationDomain` allocation owner and
  retained it in `CpuBackend`.
- Added `AppleContext`, paired CPU/Metal backends, fresh host-visible Metal
  clients, managed placement preservation, and explicit transfer counters.
- Retained resolved CubeCL managed resources in Apple WebGPU buffers.
- Added public rustdoc examples and a reviewer-facing worklog.

## Tests added

- backend-neutral guard identity/writeback and default unsupported mapping
- CPU allocation-owner retention across clones
- two-context domain isolation and foreign-domain rejection
- one-upload accounting, write-only mapping, overlap rejection, stable physical
  allocation identity, and explicit download accounting
- CPU-owned matching-domain allocation with no transfer accounting
- map/Metal dot/map identity, managed output domain, and unchanged counters

## Verification

- `cargo test -p tenferro-tensor`
- `cargo test -p tenferro-cpu --lib`
- `cargo test -p tenferro-gpu --features webgpu --test integration`
- `cargo test -p tenferro-gpu --features webgpu --doc`
- `cargo clippy -p tenferro-tensor -p tenferro-cpu -p tenferro-gpu --features tenferro-gpu/webgpu --all-targets -- -D warnings`
- `python3 scripts/check-public-error-docs.py --changed-from 1780eb62`
- `cargo fmt --all -- --check`

All listed checks passed on local Apple Metal hardware.
