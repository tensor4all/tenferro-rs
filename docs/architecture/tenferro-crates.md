# tenferro-rs Crate Architecture

**Repo:** tenferro-rs
**Parent:** `../index.md`
**Related:** `computegraph.md`, `chainrules.md`, `tidu.md`,
`../spec/backend-contract.md`, `../spec/primitive-catalog.md`,
`../spec/extension-op.md`

---

## I. Purpose

This document records the current workspace crate split and dependency
boundaries. The workspace intentionally has no root `tenferro` facade crate:
users import the crates that own the layer or operation family they need.

The design goal is to keep these concerns separate:

- host tensor data model
- concrete runtime tensors and backend traits
- CPU/GPU backend implementations
- traced graph construction and execution
- eager and traced AD APIs
- first-class operation families such as einsum, linalg, and FFT
- internal graph operation vocabulary and primitive metadata

## II. Current Workspace Crates

| Crate | Role |
|---|---|
| `tenferro-tensor-core` | Host-only tensor data model, dtype tags, scalar trait, and metadata-only views |
| `tenferro-tensor` | Dense runtime tensors, backend traits, CPU backend, and core execution kernels |
| `tenferro-gpu` | CubeCL/CUDA backend and GPU transfer helpers |
| `tenferro-runtime` | Concrete tensor helpers, traced tensors, graph compilation/execution, extension runtime registration, and extension cache storage |
| `tenferro-ad` | Eager runtime, eager tensors, and traced AD extension traits |
| `tenferro-einsum` | Subscripts, contraction planning, traced/eager einsum APIs, extension runtime, and AD rule |
| `tenferro-linalg` | Linear algebra traced APIs, eager helpers, extension runtime, and optional linalg AD rules |
| `tenferro-fft` | FFT extension runtime and public FFT APIs |
| `tenferro-core-ops` | Internal core primitive operation catalog used by graph, runtime, and backend dispatch |
| `tenferro-internal-ops` | Graph op vocabulary and AD rule implementations |
| `tenferro-internal-device` | Shared device and error infrastructure |
| `tenferro-internal-extension-macros` | Procedural macros for extension-op registration |

The public user-facing crates are `tenferro-tensor-core`, `tenferro-tensor`,
`tenferro-gpu`, `tenferro-runtime`, `tenferro-ad`, `tenferro-einsum`,
`tenferro-linalg`, and `tenferro-fft`. Crates with `internal` in their name are
implementation crates and should not be presented as user-facing API surfaces.

## III. Layering

```text
Layer 4: tenferro-ad
         eager runtime, eager tensors, traced AD extension traits

Layer 3: tenferro-runtime
         concrete tensor helpers, traced tensors, graph compilation/execution,
         extension runtime registration, extension cache storage

         tenferro-einsum / tenferro-linalg / tenferro-fft
         first-class operation-family crates

Layer 2: tenferro-tensor
         dense runtime tensors, backend traits, CPU backend,
         core execution kernels

         tenferro-gpu
         CubeCL/CUDA backend and GPU transfer helpers

Layer 1: tenferro-tensor-core
         host-only tensor data model, dtype tags, scalar trait,
         metadata-only views

Internal: tenferro-core-ops
          core primitive operation catalog

          tenferro-internal-ops
          graph op vocabulary and AD rule implementations

          tenferro-internal-device
          shared device and error infrastructure

          tenferro-internal-extension-macros
          extension-op registration macros
```

## IV. Dependency Direction

The dependency direction is deliberately one-way:

```text
tenferro-tensor-core
    |
    v
tenferro-tensor <---------------- tenferro-gpu
    |                                  ^
    |                                  |
    v                                  |
tenferro-internal-ops <-------- tenferro-einsum
    |                         \-- tenferro-linalg
    |                         \-- tenferro-fft
    v
tenferro-runtime
    |
    v
tenferro-ad
```

Additional internal dependencies:

- `tenferro-runtime`, `tenferro-tensor`, `tenferro-gpu`, and
  `tenferro-internal-ops` use `tenferro-core-ops` for primitive metadata.
- `tenferro-einsum`, `tenferro-linalg`, and `tenferro-fft` depend on
  `tenferro-runtime` for extension application and runtime registration.
- Operation-family AD support depends on `tenferro-ad` only when the crate's
  `autodiff` feature is enabled.
- CUDA support flows through explicit `cuda` features and `tenferro-gpu`.

Rules:

- `tenferro-tensor-core` must remain backend-independent and must not depend on
  GPU, BLAS/LAPACK provider crates, backend buffers, runtime caches, or AD.
- `tenferro-tensor` owns concrete runtime tensor values and CPU execution, but
  not CubeCL/CUDA implementation details.
- `tenferro-gpu` owns GPU backend implementation and transfer helpers.
- `tenferro-runtime` owns graph construction, compilation, execution, extension
  runtime registration, and extension cache ownership.
- `tenferro-ad` owns eager AD surfaces and traced AD helper APIs. Primitive AD
  rule implementations remain in `tenferro-internal-ops/src/ad/`.
- Standard operation families stay in direct crates. Do not add a root
  `tenferro` facade path for them.

## V. Extension Boundary

Operation families are first-class crates, not modules hidden behind a facade.

`tenferro-einsum`, `tenferro-linalg`, and `tenferro-fft` own their public
operation APIs, extension payloads, shape/dtype inference, runtime
registration, and operation-specific tests. When they support AD, they register
AD rules for their extension families instead of adding those rules to a
monolithic tensor type.

Custom external crates follow the same pattern:

1. Implement an extension operation payload.
2. Expose a small Rust API around it.
3. Register an extension runtime with `tenferro-runtime` when execution needs
   family-specific runtime state or caches.
4. Register AD rules through `tenferro-ad` when gradients are supported.

See [`../guides/custom-operations.md`](../guides/custom-operations.md) and
[`../spec/extension-op.md`](../spec/extension-op.md).

## VI. AD Boundary

`chainrules-rs` owns the generic `PrimitiveOp` contract. `tidu-rs` owns the
generic graph transforms such as `differentiate` and `transpose`. Neither crate
is tied to tenferro's concrete tensor types.

tenferro supplies one concrete graph vocabulary, `StdTensorOp`, plus
operation-family extension carriers. Core primitive AD rule implementations
live in `tenferro-internal-ops/src/ad/`. Extension-family AD rules live with
the operation family that owns the semantics, for example `tenferro-einsum` or
`tenferro-linalg`.

This is the current split:

```text
chainrules-rs          PrimitiveOp contract
tidu-rs                generic AD graph transforms
tenferro-internal-ops  StdTensorOp AD rules
tenferro-ad            public eager/traced AD APIs
operation crates       optional extension-family AD rules
```

## VII. Documentation Notes

Historical files under `docs/plans/` may describe older crate names and should
not be treated as current architecture. This file, `REPOSITORY_RULES.md`,
`AGENTS.md`, and the crate `Cargo.toml` files should agree on the live crate
split.
