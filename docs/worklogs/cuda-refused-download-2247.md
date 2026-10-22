# Refused CUDA readback allocation

## Decisions

- Prepare and bind the device source before reserving host payload memory.
  A private prepared readback owns the resolved allocation and validated byte
  length through the existing direct copy and stream barrier. This preserves
  stream ordering and avoids a separate per-allocation device check.
- Use fallible host reservation in fresh and reused destinations. Reject
  preparation/reservation failures before clearing a reused buffer. Preserve
  the original typed error and keep the existing destination/source retention
  on an unproven copy or barrier failure.
- Keep the direct-to-owned-vector path. CubeCL fallback reads already check
  failed buffers before preparing their staging copies; no additional payload
  staging or zero initialization is introduced.

## Verification conclusions and constraints

- Host tests cover deterministic reservation failure with its typed source,
  unchanged buffer contents/allocation, capacity reuse, and growth. A portable
  source contract guards preparation before reservation in both direct routes.
- Debug validation passes 122 host unit tests and 76 integration tests; the two
  reservation regressions also pass in release. The existing compile-fail UI
  test has four Rust 1.99.1 diagnostic snapshot mismatches in the initial full
  integration run. Focused validation excludes that unrelated UI snapshot
  test; no fixtures or CI exclusions are changed.
- The existing 200 GiB device-refusal regression now checks raw, ordinary, and
  destination-reuse downloads, including later successful device operations.
- NVIDIA hardware is unavailable on this macOS host. The ignored refusal and
  normal transfer regressions require hosted CUDA execution.
