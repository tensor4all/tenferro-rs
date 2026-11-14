# Backend Capability Descriptor

This note records the backend capability descriptor for tenferro's concrete
tensor layer: what it declares, who reads it, and what it does not do.

## Status

The descriptor is a **declaration**, not a gate. Each backend states, in a
hand-written static table, which operation/dtype pairs it serves and how. Code
that chooses a runtime path may consult it, and tests hold it to what the
engines actually do (see [Verification](#verification)). Submission does not
enforce it: an engine call is not checked against the table before the kernel
runs, and a backend can serve or refuse a pair independently of what its entry
says.

Option C — rejecting an operation because of the descriptor, at plan or dispatch
time — was considered for
[#2477](https://github.com/tensor4all/tenferro-rs/issues/2477) and rejected:

- Every backend's dispatch already refuses an unsupported dtype on its own, each
  in its own layer's spelling (`Error::UnsupportedDType`, `ErrorKind::Unsupported`,
  on CPU; the kernel's refusal wrapped as `RuntimeStateSource` on CUDA). A table
  gate in front of it would be a second copy of the same decision, kept by hand;
  the `dot_general` row that drifted in #2478 shows what hand-kept tables do.
- Most axes (`read_inputs`, `strided_output`) declare whether an internal copy
  happens rather than whether the operation works. Refusing on them is the
  per-call policy that #2552 withdrew.

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
The CUDA table's data, and the renderer of the guide table below, live in
`tenferro-gpu/src/capability.rs`, which compiles without the `cuda` feature so
the documentation check runs in a default build; the engine-facing
`TensorBackendCapability` impl and the public `cuda_capabilities()` stay behind
the feature.

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
vocabulary, so the tables name no pair that `capability_output_dtype` does not
admit.

## Current Scope

The table covers core elementwise, analytic, reduction, and `dot_general`
entries for CPU and CUDA. Linalg, FFT, indexing, structural ops, and provider
installation details are outside the table.

The CUDA guide's core primitive table
([`docs/guides/devices-and-gpu.md`](../guides/devices-and-gpu.md#coverage)) is
**generated** from `cuda_capabilities()` by
`tenferro-gpu/src/capability.rs::render_capability_markdown`, and a
default-features test checks the block between the
`cuda-core-capability:start`/`:end` markers against the rendered text; a
mismatch fails the test with the expected block, and
`TENFERRO_WRITE_CAPABILITY_DOCS=1` rewrites the block instead. Manual rows below
that table cover families outside the descriptor.

## Consumers

Typed code uses compile-time trait bounds and does not read the descriptor.

`tenferro_tensor::capability_entry` is the one lookup rule: it matches the
backend id, operation and dtype in a table and, when an axis is given, keeps
the entry only if that axis reports support.
`TensorBackendCapability::capability` is a thin wrapper over it, and so is
every kernel that reads a table. `capability_output_dtype` answers the catalog's
output-dtype question on its own, without a backend.

The only production read of a descriptor today is the CPU analytic unary
dispatch (`tenferro-cpu/src/analytic.rs`), which asks `capability_entry` for the
`read_inputs` axis of `cpu_capabilities()` before dispatching on dtype and
returns `UnsupportedDType` when the entry is missing or unsupported. No CUDA
dispatch path and no other CPU path reads the table.

There is no "require this" entry point, and none is planned: a caller that needs
support asks `capability()` for the declaration, and the engine refuses on its
own when it cannot serve the pair, with the error the backend's layer produces
(`UnsupportedDType` on CPU, a wrapped kernel refusal on CUDA). #2477 removed the
unused `TensorBackendCapability::require_capability` for that reason.

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

The `result` axis is tested against both engines, for every entry in the tables
and for every public dtype, from one shared dispatcher
(`tenferro-cpu/tests/support/capability_cases.rs`, included by path by both
lanes) whose fixtures also cover the extra arguments of `clamp`, `select`,
`pow`, `compare`, the reductions and `dot_general`:

- **CPU** (`tenferro-cpu/tests/integration/capability_conformance.rs`, normal
  CI): every `cpu_capabilities()` operation × the seven public dtypes. A
  supported entry must run and return its declared output dtype; an
  `Unsupported` entry, or a pair the table does not name, must not be served.
  How a backend spells that refusal is its own business — `Unsupported`,
  `Validation(DTypeMismatch)`, or a kernel refusal wrapped in a runtime-state
  error — so the rule stops at "not served". The operations come from the table,
  and the dispatcher panics on an operation it has no case for, so a new entry
  cannot pass unexamined.
- **CUDA** (`tenferro-gpu/tests/seam_capability.rs`, GPU lane, `#[ignore]`):
  the same rule over the union of both tables' operations, plus numerical
  parity — every served pair is downloaded and compared with the CPU engine's
  result on the same fixture, exactly for integers and `Bool`, and to a relative
  tolerance of 8 ulps for arithmetic and 64 ulps for the transcendental kernels,
  whose host and device implementations differ in the last bits.
- **Guide table** (`tenferro-gpu/src/capability/tests.rs`, normal CI): the block
  rendered from `cuda_capabilities()` equals the block in the guide, and the CPU
  and CUDA tables cover the same operations.

The other four axes (`read_inputs`, `write_output`, `strided_output`,
`accumulation`) are **declared only**. No test holds them to an engine, because
the observable difference is whether an internal copy happened, which the
operation's return value does not show.

A pair outside the catalog's dtype policy is outside the descriptor: the tables
name none, which both conformance tests assert. An engine that serves such a
pair is an engine gap against the catalog, not a missing table row.

An unsupported pair should stay as an explicit `Unsupported` entry rather than
be dropped, so that adding support later changes the descriptor and the guide
table in the same change.
