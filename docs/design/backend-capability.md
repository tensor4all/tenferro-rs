# Backend Capability Descriptor

This note records the backend capability descriptor for tenferro's concrete
tensor layer: what it declares, who reads it, and what it does not do.

## Status

The descriptor is a **declaration**. Each backend states, in a hand-written
static table, which operation/dtype pairs it serves and how. Code that chooses a
runtime path may consult it. Submission does not enforce it: an engine call is
not checked against the table before the kernel runs, and a backend can serve or
refuse a pair independently of what its entry says. Where this document or the
guides read as if the table gates execution, it does not.

Enforcement is proposed separately in
[#2477](https://github.com/tensor4all/tenferro-rs/issues/2477). Any gate that
rejects an operation because of the descriptor would come from that work, not
from the current code.

## Problem

`TypedTensor<T>` and `Tensor` are two faces of the same concrete tensor value:
the typed face carries dtype knowledge in Rust trait bounds, while the erased
face discovers dtype support at runtime. Before this descriptor, backend
support was encoded only as scattered dispatch arms and string-based
`BackendFailure` errors, with no single written statement of what each backend
serves. That made CUDA documentation drift likely.

## Schema

The descriptor types live in `tenferro-tensor` (`OperationCapability`,
`CapabilityQuery`, `CapabilityAxis`, `SupportLevel`). Each backend exposes its
static table through `TensorBackendCapability`: `tenferro_cpu::cpu_capabilities()`
for `CpuEngine` and `tenferro_gpu::cuda::cuda_capabilities()` for `CudaEngine`.

Each entry is keyed by:

- backend id,
- `tenferro-core-ops::PrimitiveOpKind`,
- representative input dtype.

Each entry records:

- output dtype,
- owned-result support (`result`),
- borrowed-input support (`read_inputs`),
- caller-provided output support (`write_output`),
- strided-output support (`strided_output`),
- accumulation/read-modify-write support (`accumulation`).

Support is three-valued. The meanings are what the backend author declares,
not what submission checks:

| Level | Declared meaning |
| --- | --- |
| `Unsupported` | The backend does not serve this op/dtype/axis. |
| `FallbackCopy` | The backend serves it through a generic copy/materialization path. |
| `Native` | The backend has a direct implementation for that axis. |

The core-op catalog remains the semantic source for dtype policy and output
dtype mapping (`capability_output_dtype`). Capability entries add backend
implementation status; they must not duplicate or redefine the catalog's op
vocabulary.

## Current Scope

The table covers core elementwise, analytic, reduction, and `dot_general`
entries for CPU and CUDA. Linalg, FFT, indexing, structural ops, and provider
installation details are outside the table.

The CUDA guide's core primitive table
([`docs/guides/devices-and-gpu.md`](../guides/devices-and-gpu.md#coverage)) is
a hand-maintained copy of `cuda_capabilities()`. No generator or test keeps the
two in step; a change to the CUDA table must update the guide in the same
change. Manual rows below it cover families outside the descriptor.

## Consumers

Typed code uses compile-time trait bounds and does not read the descriptor.

`tenferro_tensor::capability_entry` is the one lookup rule: it matches the
backend id, operation and dtype in a table and, when an axis is given, keeps
the entry only if that axis reports support. `TensorBackendCapability::capability`
and `::require_capability` are thin wrappers over it, and so is every kernel
that reads a table. `require_capability` has no production caller; it is
exercised by its doctest and the CPU capability unit test.

The only production read of a descriptor today is the CPU analytic unary
dispatch (`tenferro-cpu/src/analytic.rs`), which asks `capability_entry` for the
`read_inputs` axis of `cpu_capabilities()` before dispatching on dtype and
returns `UnsupportedDType` when the entry is missing or unsupported. No CUDA
dispatch path and no other CPU path reads the table.

The runtime planner does not read this descriptor. Preparation selects a
provider per operation family (`tenferro_runtime::CoreCapabilityKind`:
elementwise, reduction, indexing, dot-general, layout) from the engine's
registered provider slots; that is a different, family-level capability set,
and it carries no per-dtype information.

A future consumer that keeps a descriptor check at the dtype-stripping seam
should do it once per operation: one small table lookup added to the existing
per-operation dispatch cost, never inside element loops or kernel-launch inner
loops, and without tensor materialization.

## Verification

`crates/tenferro-gpu/tests/seam_capability.rs` (CUDA, `#[ignore]`, run by the
GPU lane) holds part of the CUDA table to observed behaviour. For seven
operations (`add`, `mul`, `abs`, `exp`, `compare`, `reduce_sum`,
`dot_general`) across the seven public dtypes, it checks the `result` axis
only: an entry declared `Native` must succeed, any other entry, or a pair with
no entry, must return an error. It compares success against failure; it does
not compare numerical results with the CPU backend, and it does not check the
other axes or the remaining operations in the table.

`crates/tenferro-cpu/src/tests/cpu_tests/capability.rs` asserts selected CPU
entries and one `require_capability` acceptance.

An unsupported pair should stay as an explicit `Unsupported` entry rather than
be dropped, so that adding support later changes the descriptor and the guide
table in the same change.
