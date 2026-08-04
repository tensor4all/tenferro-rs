# Provider owner domain requirement

## Scope

This is one bounded P7/P8 migration step for Issue #1555. CUDA's
`CubeclBuffer` and WebGPU's `WebGpuBuffer` are physical allocation owners, so
their allocation domain is a required identity field. The step does not
change the public `BackendStorage::allocation_domain()` return type and does
not introduce a compatibility adapter.

## Design

- Store `AllocationDomainId`, rather than `Option<AllocationDomainId>`, in
  both provider owner structs.
- Require the domain in `CubeclBuffer::new`; runtime allocation sites pass the
  runtime domain directly.
- Keep `WebGpuBuffer::domain` optional because it denotes an Apple managed
  resource endpoint, not the allocation identity. `new_for_runtime` always
  stores the runtime's mandatory allocation domain.
- Simplify CUDA/WebGPU residency checks to compare the stored domain directly.
- Test-only synthetic allocations receive an explicit fresh domain. They do
  not model a valid provider owner with missing identity.

## Safety and non-goals

The domain is diagnostic/coherence identity, not write authority. Rust owner
borrowing and provider prepared access remain the access authority. This step
adds no retry, recovery, quarantine, COW, hidden transfer, repeated
validation, raw-handle API, or provider bridge.

## Verification

Run formatting, workspace check, GPU feature no-run compilation, the GPU
public-surface contract, and the existing storage/provider tests. Record the
results in the P7/P8 worklog. The parent issue remains open until the complete
provider root/prepared-access migration and P13-B audit pass.
