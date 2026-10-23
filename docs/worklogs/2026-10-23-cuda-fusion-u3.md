# U3: fused regions, artifact preparation and the cold/warm counters

Date: 2026-10-23. Issue: [tenferro-rs#2208](https://github.com/tensor4all/tenferro-rs/issues/2208)
unit U3.

## What landed

- `cuda_next::PreparedFusion`: takes a `FusionPlan`, emits it with the kernel crate
  (`emit_fusion`, whose source carries the ABI header and the compiled operation chain),
  compiles it through `Runtime::compile` only if this runtime does not already hold the
  artifact, binds `t4a_fused_elementwise`, and launches it with the plan's own geometry.
  `PreparedFusion::key()` is the kernel crate's canonical `ArtifactKey` (with the target
  architecture and compiler folded in), and `counters()` reports the runtime's NVRTC
  attempts, artifact-cache hits and module loads.
- `CudaRuntime::module_loads` exposes the runtime's module-load counter.
- The prepare path selects what it needs: `KernelModules::for_entries` compiles only the
  modules the requested entries export, and a fusion plan compiles only its own emitted
  source.

## Evidence

`tests/seam_fusion.rs` on the local A100 (driver 580.126.09, CUDA 12.6):

- `a_fused_region_matches_the_host_and_prepares_once` — the region `a * b + a` over 1024
  f64 values matches the host result; a cold prepare performs **exactly one** NVRTC
  compilation and one module load; a second prepare of the same plan performs **zero**
  compilations (the artifact cache served it, `artifact_hits >= 1`), loads no second
  module, and produces the same `ArtifactKey`; a warm run also compiles nothing.
- `a_region_with_the_wrong_operand_count_is_refused` — the plan needs one output and two
  inputs, and a short operand list is a typed refusal rather than a launch.
- `an_ineligible_plan_is_refused_before_any_compile` — a Bool plan is refused by the
  kernel crate and NVRTC is never reached (`compile_attempts` unchanged).

## Observations

- The emitted source is flat (one thread per element, operations compiled into the
  chain), so the launch is the ordinary flat geometry and the region needs no by-value
  parameter block. The runtime's by-value support (U0a) stays available for the
  structured entries and for the fused parameter block if the emitter grows one.
- The artifact key folds the target architecture and the compiler name into the kernel
  crate's material; the runtime's own cache additionally keys the full compile request,
  so a target or option change cannot reuse a stale module.

## Not done (later units)

- The consumer's region *planning* (which operations a region contains) stays with
  tenferro's eager/traced layer (U5); this unit prepares and launches a plan it is given.
- Vendor operations (cuBLAS/cuTENSOR/cuSOLVER/cuFFT) are U4.
