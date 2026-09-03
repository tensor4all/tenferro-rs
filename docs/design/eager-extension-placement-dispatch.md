# Placement-aware eager extension dispatch

**Status:** Reviewed and approved for tenferro-rs #1752, a dependency of tensor4all-rs #720

## Problem

The eager einsum and linalg convenience APIs validate that all operands belong
to one `EagerRuntime`, but then install an extension module specialized for
`tenferro_cpu::CpuBackend` regardless of that runtime's actual backend.
Consequently:

- a non-direct CUDA eager einsum can enter a CPU extension session with device
  operands and fail at the CPU placement boundary;
- CUDA eager QR and SVD fail preparation with no eligible extension engine;
- enabling CUDA features or proving direct CUDA-session kernels does not make
  the eager extension surfaces CUDA-capable.

On an NVIDIA A100, direct CUDA-session QR and SVD tests pass while the
corresponding eager calls fail. This isolates the defect to eager extension
module selection rather than the CUDA kernels.

`tenferro-fft` already uses the intended seam:
`apply_eager_with_targeted_extension_session` obtains the owning runtime's
validated `EagerExtensionTarget`, and a module factory selects the matching
backend module.

## Scope

Migrate eager extension dispatch in:

- `tenferro-einsum` for the extension-backed N-ary/general einsum path;
- `tenferro-linalg` for eager decomposition and solve operations.

The change owns no tensor transfer and changes no numerical kernel. Inputs must
already belong to one runtime and satisfy that runtime's placement contract.
Outputs remain in the same runtime and placement.

## Dispatch contract

Each eager operation delegates to
`apply_eager_with_targeted_extension_session`. The module factory matches the
validated target:

- CPU target -> `extension_module::<CpuBackend>(engine_id)`;
- CUDA target, under `cuda` -> `extension_module::<CudaBackend>(engine_id)`;
- WebGPU target, under `webgpu`, only for an operation family with an actual
  WebGPU session implementation -> `extension_module::<WebGpuBackend>(engine_id)`;
- a compiled target without an implementation returns a typed unsupported
  error before execution.

Backend modules may be retained in one process-wide immutable `OnceLock` per
backend family because they contain extension registration metadata, not a
backend, device allocation, execution context, stream, or mutable cache. The
runtime remains the owner of backend state and extension caches. The cached
module must be keyed strongly enough that adding nonzero-device engine IDs in
the future cannot reuse incompatible registration metadata; if engine IDs vary
by device, cache per engine ID or avoid the global cache.

The selected module must match the runtime target. There is no CPU fallback,
implicit upload/download, host materialization, or fresh backend construction.
Mixed eager contexts continue to fail in the existing input validation before
module selection.

## Feature behavior

CPU-only builds keep their current behavior and do not compile GPU types.
CUDA and WebGPU match arms remain feature-gated. Enabling multiple backend
features is supported; runtime identity, not Cargo feature priority, selects
the module.

`tenferro-linalg` must not claim WebGPU operations that its session capability
contract rejects. Such calls return the existing typed unsupported/runtime
error. `tenferro-einsum` preserves its supported WebGPU eager path.

## Tests

### Hardware-independent

- Existing CPU eager einsum and linalg suites remain green.
- CPU-only, CUDA, WebGPU, and combined feature builds compile.
- A source/behavior regression proves eager module selection is target-based,
  not CPU-hardcoded.
- Mixed eager contexts fail before extension installation.
- Repeated calls reuse the runtime registration/cache rather than replacing a
  module on every call.
- Unsupported target/operation combinations preserve typed sources and never
  execute on CPU.

### CUDA hardware

Using one CUDA `EagerRuntime`:

- a three-operand einsum that cannot take the direct binary-dot path executes
  and returns a CUDA-resident output;
- eager QR reconstructs the input and returns CUDA-resident factors;
- eager SVD reconstructs the input and returns CUDA-resident factors/values;
- F32, F64, C32, and C64 coverage follows the existing direct CUDA-session
  capability matrix;
- host extraction fails before explicit download;
- downloaded results match the direct-session/CPU reference within existing
  tolerances;
- no transfer or CPU backend call occurs inside the eager operation.

## Performance gate

Measure one warm-up followed by repeated small N-ary einsum, QR, and SVD calls
on CPU and CUDA. The change must not add repeated extension-module replacement
or backend construction to the steady-state path. Report setup, explicit
transfer, warm-up/JIT, synchronized steady state, and explicit download
separately. Numerical and residency assertions remain outside timed loops.

## Non-goals

- Changing the public eager operation vocabulary.
- Adding CUDA kernels or silently emulating unsupported operations.
- Changing tensor4all execution contexts or SRC.
- Multi-GPU context construction, device selection, sharding, or transfer.
- Making CUDA a default feature.
- Changing AD rules or decomposition gauge/tolerance semantics.

## Review gate

- Reviewer: `reviewer-flash` (DeepSeek family, read-only)
- Review round: pre-implementation design review
- Verdict: **Correct-to-implement**
- Evidence: the reviewer confirmed the CPU hard-coding in eager einsum/linalg,
  the existing FFT targeted-dispatch seam, feature-gated backend matching, and
  found no blocking issue in the dispatch, cache, no-fallback, or test design.

## Downstream handoff

After this change is merged, tensor4all-rs can update all tenferro pins together
and implement its caller-owned, context-scoped intermediate construction seam.
Tensor4all SRC must still explicitly construct probes/caps in the supplied
context and validate all input/result residency; this tenferro change alone
does not make SRC placement-aware.
