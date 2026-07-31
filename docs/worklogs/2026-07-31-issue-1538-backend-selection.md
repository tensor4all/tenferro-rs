# Worklog: #1538 Backend Selection And Capability Documentation

This worklog records the documentation and runtime-registration changes for
[tenferro-rs issue #1538](https://github.com/tensor4all/tenferro-rs/issues/1538).

## Scope

- Added `docs/guides/choosing-a-backend.md` as the owner for execution-backend
  and CPU-provider selection, capability ownership, nested parallelism,
  placement, and unsupported-operation policy.
- Added the required SVD/QR batch-versus-inner parallelism section and PJRT
  threading/synchronization section to the existing guides.
- Replaced the duplicated provider-selection block in
  `parallelism-and-caching.md` with a link to the choosing guide.
- Added a documentation-consistency check mapping every item in the design
  document's Documentation Requirements section to a named guide heading.

## Runtime API decision

`runtime_engine_registration` keeps the canonical
`tenferro-cpu.default.v1` identity. The new
`runtime_engine_registration_with_id` helper accepts a validated caller-chosen
`EngineId`, so distinct CPU backends can be registered in one `Runtime`
without colliding. `CpuBackendKind::Faer` and `CpuBackendKind::Blas` remain
per-backend choices when both base features are compiled. Concrete external
BLAS implementations and their worker environment remain build/provider
process-wide; the guide documents that limitation rather than implying
runtime selection between OpenBLAS and MKL.

## Verification

- `cargo fmt --all`
- `cargo test -p tenferro-cpu --lib runtime_adapter::tests::public_cpu_runtime_registration`
- `python3 scripts/test-doc-consistency.py`
- `git diff --check`
- The conditional faer-plus-BLAS provider test was compiled with
  `--features cpu-faer,cpu-blas`; linking could not complete in this
  environment because no CBLAS symbols (`cblas_*gemm`) were installed. The
  normal CI BLAS profile remains the environment for the linked-provider run.
