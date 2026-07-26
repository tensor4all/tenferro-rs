# tenferro-rs Crate Architecture

**Repo:** tenferro-rs
**Parent:** `../index.md`
**Related:** `computegraph.md`, `primitive-ad.md`, `semantic-ad.md`,
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
- experimental XLA/PJRT lowering for static traced programs
- first-class operation families such as einsum, linalg, and FFT
- internal graph operation vocabulary and primitive metadata

![tenferro-rs architecture overview](../assets/tenferro-architecture.svg)

The overview diagram focuses on the native CPU/GPU runtime and extension
stack. `tenferro-xla` is a peer executor over compiled static programs, not a
`TensorBackend` inside that stack.

## II. Current Workspace Crates

| Crate | Role |
|---|---|
| `tenferro-tensor-core` | Rank/layout metadata, dtype tags, scalar trait, and host-only tensor adapters |
| `tenferro-tensor` | Runtime `TypedTensor<T, R>`/`Tensor` values, typed views, backend traits, and backend-independent contracts |
| `tenferro-cpu` | CPU backend, CPU execution sessions, CPU kernels, buffer pools, CPU provider selection, and the CPU runtime-registration preparation/execution adapter |
| `tenferro-gpu` | CubeCL/CUDA backend and GPU transfer helpers |
| `tenferro-runtime` | Concrete tensor helpers, traced tensors, graph compilation/execution, immutable runtime snapshots, preparation SPI/cache substrate, runtime-owned compiled-graph execution, scheduled-graph staging, extension runtime registration, and extension cache storage |
| `tenferro-xla` | Experimental StableHLO lowering and runtime-loaded PJRT plugin support for static-shaped traced programs |
| `tenferro-ad` | Eager runtime, eager tensors, and traced AD extension traits |
| `tenferro-einsum` | Subscripts, contraction planning, concrete/traced/eager einsum APIs, extension runtime, and AD rule |
| `tenferro-linalg` | Linear algebra traced APIs, eager helpers, extension runtime, and optional linalg AD rules |
| `tenferro-fft` | FFT extension runtime and public concrete/traced FFT APIs |
| `tenferro-core-ops` | Internal core primitive operation catalog used by graph, runtime, and backend dispatch |
| `tenferro-internal-ops` | Graph op vocabulary and AD rule implementations |
| `tenferro-internal-extension-macros` | Procedural macros for extension-op registration |

The public user-facing crates are `tenferro-tensor-core`, `tenferro-tensor`,
`tenferro-cpu`, `tenferro-gpu`, `tenferro-runtime`, `tenferro-ad`,
`tenferro-xla`, `tenferro-einsum`, `tenferro-linalg`, and `tenferro-fft`. Crates with
`internal` in their name are
implementation crates and should not be presented as user-facing API surfaces.

### Non-workspace source-provider packages

`third_party/t4a-tblis-src` is stored in this repository but is not a tenferro
workspace crate. It is a self-contained, independently versioned source-build
and native-link provider prepared for a separate crates.io release. The root
workspace excludes it and consumes it through the neutral `tblis-src`
dependency alias with a path-plus-exact-version requirement.

`tenferro-cpu` always uses `tblis-ffi` for Rust declarations. Its
`cpu-tblis-runtime` route enables the FFI crate's dynamic loader and does not
depend on the source provider. Its `cpu-tblis-linked` route instead anchors
`t4a-tblis-src` in the Rust crate graph with `build_from_source` and `static`,
while leaving `tblis-ffi/dynamic_loading` disabled. This package is deployment
glue, not a tenferro user-facing API surface or a new runtime layer.

## III. Layering

```text
Layer 4: tenferro-ad
         eager runtime, eager tensors, traced AD extension traits

Layer 3: tenferro-runtime
         concrete tensor helpers, traced tensors, graph compilation/execution,
         extension runtime registration, extension cache storage

         tenferro-xla
         experimental StableHLO/PJRT peer executor for static programs

         tenferro-einsum / tenferro-linalg / tenferro-fft
         first-class operation-family crates

Layer 2: tenferro-tensor
         runtime TypedTensor<T, R>/Tensor values, typed views,
         backend traits, backend-independent contracts

         tenferro-cpu
         CPU backend, CPU execution sessions, CPU kernels, buffer pools

         tenferro-gpu
         CubeCL/CUDA backend and GPU transfer helpers

Layer 1: tenferro-tensor-core
         rank/layout metadata, dtype tags, scalar trait,
         host-only tensor adapters

Internal: tenferro-core-ops
          core primitive operation catalog

          tenferro-internal-ops
          graph op vocabulary and AD rule implementations

          tenferro-internal-extension-macros
          extension-op registration macros
```

## IV. Dependency Direction

The dependency direction is deliberately one-way. Arrows below mean
"depends on". The CPU-to-runtime line records the implemented runtime
registration adapters: Phase 4 preparation metadata and the Phase 5
runtime-owned execution bridge. It is not a runtime-to-CPU dependency:

```text
tenferro-tensor           -> tenferro-tensor-core, tenferro-core-ops
tenferro-cpu              -> tenferro-tensor
tenferro-cpu              -> tenferro-runtime
tenferro-gpu              -> tenferro-tensor, tenferro-core-ops
tenferro-internal-ops     -> tenferro-tensor, tenferro-core-ops,
                              tenferro-internal-extension-macros
tenferro-runtime          -> tenferro-tensor, tenferro-core-ops,
                              tenferro-internal-ops
tenferro-xla              -> tenferro-runtime, tenferro-internal-ops,
                              tenferro-tensor
tenferro-ad               -> tenferro-runtime, tenferro-internal-ops,
                              tenferro-tensor, tenferro-cpu

tenferro-einsum           -> tenferro-runtime, tenferro-internal-ops,
                              tenferro-tensor, tenferro-cpu
tenferro-linalg           -> tenferro-runtime, tenferro-internal-ops,
                              tenferro-tensor, tenferro-cpu
tenferro-fft              -> tenferro-runtime, tenferro-internal-ops,
                              tenferro-tensor
```

Additional internal dependencies:

- `tenferro-runtime`, `tenferro-tensor`, `tenferro-gpu`, and
  `tenferro-internal-ops` use `tenferro-core-ops` for primitive metadata.
- The production `tenferro-cpu` to `tenferro-runtime` edge is approved solely
  for engine-registration adapters: direct core preparation capability
  registration, runtime cache-owner registration, and the erased tensor backend
  execution bridge. The preparation/execution substrate is implemented in
  `tenferro-runtime`; any CPU dependency in the opposite direction remains
  dev/test-only and is not a production edge.
- `tenferro-einsum`, `tenferro-linalg`, and `tenferro-fft` depend on
  `tenferro-runtime` for extension application and runtime registration.
- `tenferro-xla` depends on `tenferro-runtime` to read compiled programs and
  owns its own runtime-loaded PJRT boundary. It must not make XLA/PJRT a
  compile-time dependency of `tenferro-runtime`.
- Operation-family AD support depends on `tenferro-ad` only when the crate's
  `autodiff` feature is enabled.
- CUDA support flows through explicit `cuda` features and `tenferro-gpu`.

Rules:

- `tenferro-tensor-core` must remain backend-independent and must not depend on
  GPU, BLAS/LAPACK provider crates, backend buffers, runtime caches, or AD.
- `tenferro-tensor-core` must not expose public `TypedTensor` aliases.
  Backend-capable typed tensors are owned by `tenferro-tensor`.
- `tenferro-tensor` owns concrete runtime tensor values, arbitrary-stride typed
  views, backend traits, and backend-independent contracts.
- `tenferro-cpu` owns `CpuBackend`, `CpuContext`, CPU execution sessions,
  CPU kernels, buffer pools, CPU provider selection, and its narrow
  runtime-registration adapters for preparation metadata and runtime-owned
  execution.
- `tenferro-gpu` owns GPU backend implementation and transfer helpers.
- `tenferro-runtime` owns graph construction, compilation, execution,
  immutable runtime snapshots/reconfiguration, preparation SPI/cache substrate,
  scheduled-graph staging, runtime-owned compiled-graph execution, extension
  runtime registration, and extension cache ownership.
- `tenferro-xla` owns StableHLO lowering, explicit host-layout conversion at
  the XLA boundary, and runtime PJRT plugin loading.
- `tenferro-ad` owns eager AD surfaces and traced AD helper APIs. Primitive AD
  rule implementations remain in `tenferro-internal-ops/src/ad/`.
- Device, placement, and error concepts are owned by the crate that uses them:
  `tenferro-tensor` owns backend-independent tensor contracts,
  `tenferro-gpu` owns GPU backend details, and each operation-family crate owns
  operation-specific parse/planning errors.
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

`tenferro-ad` owns the current `SemanticProgram -> SemanticProgram` AD
transforms. `tenferro-internal-ops` still owns the primitive rule vocabulary
used to implement those transforms, while operation-family semantic rules live
with the crate that owns the operation semantics.

This is the current split:

```text
tenferro-internal-ops  primitive rule vocabulary
tenferro-ad            semantic transforms and public eager/traced AD APIs
operation crates       optional extension-family semantic rules
```

## VII. Documentation Notes

Historical files under `docs/plans/` may describe older crate names and should
not be treated as current architecture. This file, `REPOSITORY_RULES.md`,
`AGENTS.md`, and the crate `Cargo.toml` files should agree on the live crate
split.
