# No-Facade Crate Boundary Redesign

## Goal

Redesign the long-term crate layout so tenferro has no umbrella `tenferro`
facade crate. Users and downstream libraries should depend directly on the
crates that own the APIs they use: tensor values, runtime execution, AD,
standard operation families, and GPU backends.

This design updates the direction from the earlier extension-runtime
restructure notes. The earlier direction kept `tenferro` as a small
runtime/foundation facade. This design removes that facade layer entirely.

## Motivation

Issue #897 identifies a structural weakness in relying only on Cargo features
for heavy optional capabilities such as automatic differentiation and GPU
support. Cargo features are additive and feature-unified across the dependency
graph. If one downstream dependency enables `autodiff` or `cuda` on a shared
crate, every user of that crate in the final build sees the unified feature
set.

Crate-level opt-in is a better long-term boundary for tenferro because it makes
heavy capabilities visible in `Cargo.toml`, avoids accidental dependency
activation in CPU-only or no-AD builds, and gives each subsystem a smaller
public contract.

Removing the `tenferro` facade crate strengthens that direction. A facade tends
to accumulate re-exports and convenience features, which reintroduces the same
feature-unification and public-dependency pressure that crate splitting is
meant to avoid.

## Non-Goals

- Do not preserve the current `tenferro` crate as an umbrella facade.
- Do not add `tenferro::linalg`, `tenferro::einsum`, `tenferro::gpu`, or
  similar facade modules.
- Do not create a `tenferro-full` convenience crate in the first migration.
- Do not split CUDA and ROCm into separate public crates yet.
- Do not rely on linker-section auto-registration or implicit plugin discovery.
- Do not add compatibility shims for the old facade layout.

## Design Principles

### Crate Names Should Match Ownership

Every public crate name should describe the API contract it owns. A user should
be able to look at their dependency list and tell whether they opted into graph
execution, AD rules, linalg, einsum, FFT, or GPU support.

### No Umbrella Public Dependency

There should be no public crate whose main role is to re-export the whole
workspace. Re-export facades obscure public dependencies and make it easy for
one downstream crate to opt an application into heavy features unintentionally.

### Explicit Registration

Extension runtimes, AD rules, and backends should be registered explicitly on a
runtime, compiler, AD context, or builder object. Registration should not happen
because a crate was linked into the process.

### Registry Ownership

Long-lived registries should be owned by explicit top-level runtime objects,
not by process-global state. This matches the repository cache ownership rules
and makes tests, multiple runtimes, and plugin versioning easier to reason
about.

### GPU Is One Public Crate For Now

The first public GPU boundary should be `tenferro-gpu`, not
`tenferro-gpu-cuda` and `tenferro-gpu-rocm`. CubeCL is the current shared
implementation path, and the first design objective is to remove GPU
dependencies from the core tensor/runtime crates. Vendor-specific splitting
should wait until CUDA and ROCm need genuinely separate public dependency
graphs or incompatible build surfaces.

## Target Crate Shape

```text
tenferro-tensor
  Tensor, TypedTensor, DType, TensorBackend, CpuBackend, layout metadata,
  CPU kernels, backend cache traits, vendor-neutral device identifiers,
  placement metadata, and shared tensor/backend error types.

tenferro-ops
  Standard graph operation vocabulary, extension op payload traits,
  symbolic dimensions, shape/dtype inference helpers.

tenferro-ops-derive
  Proc macros for stable extension-family identifiers and other extension
  authoring helpers. Re-exported from tenferro-ops only if accepted as a
  stable public authoring API.

tenferro-runtime
  TracedTensor, graph construction, GraphCompiler, GraphExecutor,
  GraphProgram, extension runtime executor, runtime-owned caches. This absorbs
  the current tenferro-internal-runtime crate.

tenferro-ad
  AD transforms and user APIs: grad, vjp, jvp, hvp, backward support,
  AD registry, AD context, chainrules/tidu integration.

tenferro-linalg
  Primal linalg extension APIs, op payloads, shape/dtype inference,
  runtime registration, eager/traced glue where it is primal-only.

tenferro-linalg-ad
  AD rules for tenferro-linalg operations. Depends on tenferro-linalg
  and tenferro-ad. Registers rules explicitly.

tenferro-einsum
  Primal einsum APIs, contraction planning, runtime registration,
  tensordot sugar.

tenferro-einsum-ad
  Optional later split if einsum AD grows heavy enough to warrant a
  separate crate. It is not required for the first migration.

tenferro-fft
  Primal FFT extension APIs and runtime registration.

tenferro-gpu
  CubeCL-backed GPU backend, device/runtime initialization, upload/download,
  GPU TensorBackend implementation, GPU backend registration helpers.
```

The old `tenferro` crate name should either be retired or left unpublished
during the migration. It should not become a new all-in-one facade.

## Package Rename Matrix

Cargo package names use hyphens and Rust import names use underscores. The
migration should keep one package for each Rust import path; do not create a
new package while leaving an old internal package with the same `lib.name`.

```text
Target package        Rust crate name       Current source
--------------------  --------------------  ---------------------------------
tenferro-tensor       tenferro_tensor       tenferro-internal-tensor
tenferro-ops          tenferro_ops          tenferro-internal-ops
tenferro-runtime      tenferro_runtime      tenferro-internal-runtime + tenferro
tenferro-ad           tenferro_ad           tenferro autodiff modules + ops AD
tenferro-linalg       tenferro_linalg       existing tenferro-linalg primal API
tenferro-linalg-ad    tenferro_linalg_ad    linalg AD rules
tenferro-einsum       tenferro_einsum       existing tenferro-einsum
tenferro-fft          tenferro_fft          existing tenferro-fft
tenferro-gpu          tenferro_gpu          tensor cubecl + device cuda + gpubackend
tenferro-ops-derive   tenferro_ops_derive   tenferro-internal-extension-macros
```

The current `tenferro-internal-runtime` package already has
`lib.name = "tenferro_runtime"`, so it should be renamed in place. The same
rule applies to `tenferro-internal-tensor` and `tenferro-internal-ops`. This
keeps downstream import paths stable for the new direct crates while removing
the internal package names.

## Disposition Of Current Internal Crates

Several current crates already have public-looking Rust import names even
though their package names are internal. The migration must account for them
explicitly rather than creating duplicate packages.

### `tenferro-internal-runtime`

The current `tenferro-internal-runtime` package already exposes the Rust crate
name `tenferro_runtime`. The target `tenferro-runtime` package is not a second
crate with the same import path. It is the public successor that absorbs:

- extension runtime dispatch and cache infrastructure from
  `tenferro-internal-runtime`,
- traced graph construction, compiler, executor, programs, and runtime-owned
  cache APIs from the current `tenferro` crate.

The old `tenferro-internal-runtime` package should be renamed to
`tenferro-runtime`, then the current `tenferro` runtime code should move into
it. The current `tenferro` package should then be removed or left unpublished
as an empty transitional package only if a separate migration plan approves
that.

### `tenferro-internal-device`

The current `tenferro-internal-device` package should not become a public
`tenferro-device` crate in the first no-facade design. The durable public
contract is tensor placement and backend dispatch, so the vendor-neutral parts
should be absorbed into `tenferro-tensor`.

Split the current package by responsibility:

- Move vendor-neutral placement and device identifiers into `tenferro-tensor`.
  This includes the equivalent of `LogicalMemorySpace`, tensor placement
  metadata, shared tensor/backend `Error` and `Result` types, and small
  backend-neutral helpers such as batch indexing if tensor code owns their use.
- Move CUDA runtime modules, CUDA availability checks, CUDA generator state,
  driver handles, and cudarc dependencies into `tenferro-gpu`.
- Do not keep `preferred_compute_devices` as a tensor-core global policy. It
  performs backend selection and can pull GPU availability concerns into core.
  Device selection should come from explicit runtime/backend construction or a
  capability query owned by `tenferro-runtime` or `tenferro-gpu`.
- Do not leave duplicate device models in both `tenferro-tensor` and
  `tenferro-gpu`. `tenferro-gpu` may expose constructors and backend-specific
  helpers, but it should use the `DeviceId`, `DeviceKind`, `MemoryKind`, and
  `Placement` types from `tenferro-tensor`.

The `tenferro-internal-device` package should be deleted after its pieces move
to their owners. A later `tenferro-device` crate should be introduced only if
multiple non-tensor public crates need a shared device contract without
depending on `tenferro-tensor`; that is not the current architecture.

### `tenferro-internal-extension-macros`

The current proc-macro package should either become `tenferro-ops-derive` or be
absorbed as a private implementation detail of `tenferro-ops`. If third-party
extension authors are expected to derive extension identifiers, the derive
crate is a public authoring dependency and should be named/stabilized
accordingly.

If extension derive support is not part of the stable public API, the design
must say so and require extension crates to implement the relevant traits
manually.

### Current `tenferro` Package

The current `tenferro` package is an umbrella plus implementation crate. Its
modules should be split by ownership rather than moved wholesale:

```text
Current module or surface                 Target owner
----------------------------------------  ----------------------------------
graph/{compiler,executor,program,cache}   tenferro-runtime
compiler                                  tenferro-runtime
traced, traced_tensor                     tenferro-runtime
exec, segment, eager_exec                 tenferro-runtime
shape_infer, sym_dim, shape_packing       tenferro-ops or tenferro-runtime
extension runtime re-exports             tenferro-runtime
extension AD rule traits and globals      tenferro-ad
metadata global registry                  replace with runtime/trace context
checkpoint                                tenferro-ad
tensor, typed_tensor eager free funcs     tenferro-tensor
eager, eager_backend, eager_tensor        tenferro-ad
public re-exports from tenferro_tensor    remove with facade
cuda facade module                        remove; use tenferro-gpu
```

`shape_infer` and `sym_dim` should move to `tenferro-ops` when they describe
operation semantics independent of graph execution. Runtime-only graph
bookkeeping should stay in `tenferro-runtime`.

The current concrete eager free-function API (`tensor::add`,
`typed_tensor::where_`, and similar) should become a `tenferro-tensor` API
because it operates directly on concrete tensors and a `TensorBackend`. Traced
free functions should become `tenferro-runtime::traced_tensor::*`.

The current autodiff-gated eager APIs should not stay in `tenferro-runtime`.
They should be renamed around AD ownership, for example `AdContext`,
`AdTensor`, or `TapeRuntime`, rather than keeping a generic `EagerRuntime`
name that sounds like a primal execution API.

`checkpoint` should move to `tenferro-ad` because its surviving behavior is
used by AD graph replay and alias collection. If a future non-AD checkpoint
feature is needed, it should be designed separately as a runtime feature.

## Naming Rationale

### `tenferro-runtime` Instead Of `tenferro-graph`

The current `tenferro` crate is not only graph data structures. It owns traced
tensor construction, compilation, execution, extension runtime state, and
execution caches. The name `tenferro-runtime` better captures that broader
responsibility.

`graph` should remain a module name inside `tenferro-runtime` for graph IR,
programs, compiler implementation, and executor implementation details.

Recommended user-facing imports:

```rust
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_tensor::{CpuBackend, Tensor};
use tenferro_linalg::traced_tensor::solve;
```

### `tenferro-gpu` Instead Of Vendor-Specific Crates

The public decision users need to make first is "do I want GPU support?" not
"which vendor crate owns the backend abstraction?" A single `tenferro-gpu`
crate keeps the first split simple while preserving room for vendor-specific
features or modules:

```rust
use tenferro_gpu::cuda::CudaBackend;

let backend = CudaBackend::cuda(0)?;
let mut executor = tenferro_runtime::GraphExecutor::new(backend);
```

If ROCm later requires a different dependency graph, `tenferro-gpu` can become
a common GPU facade over smaller vendor crates, or CUDA/ROCm crates can be
introduced then. That split should be driven by real build and maintenance
pressure, not by speculation.

### Singular And Plural Names

Domain crates use singular names when they own one subsystem contract
(`tenferro-tensor`, `tenferro-runtime`, `tenferro-ad`, `tenferro-gpu`).
`tenferro-ops` remains plural because it owns an operation vocabulary, not a
single operation subsystem.

## Concrete Device Model In `tenferro-tensor`

`tenferro-tensor` should own one vendor-neutral device and placement model. The
model should be typed enough to avoid stringly API contracts while staying
small enough that CPU-only users do not compile GPU dependencies.

Recommended public types:

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Gpu(GpuBackendKind),
    Other(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GpuBackendKind {
    Cuda,
    Rocm,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId {
    pub kind: DeviceKind,
    pub ordinal: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    UnpinnedHost,
    PinnedHost,
    Device,
    Managed,
    Other(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Placement {
    pub memory_kind: MemoryKind,
    pub device: Option<DeviceId>,
}
```

The exact names can change during implementation, but these semantics should
hold:

- `Tensor` and `TypedTensor` report `Placement` without requiring CubeCL,
  cudarc, CUDA, or ROCm types.
- CPU host tensors use `MemoryKind::UnpinnedHost` with no accelerator device.
- Pinned host, device, and managed memory may carry an associated `DeviceId`
  when an accelerator backend owns or optimizes that allocation.
- `DeviceKind::Gpu(GpuBackendKind::Cuda)` and
  `DeviceKind::Gpu(GpuBackendKind::Rocm)` identify placement and dispatch
  intent only. They do not initialize a driver or prove availability.
- Backend constructors in `tenferro-gpu` are responsible for validating actual
  device availability and for mapping a `DeviceId` to CubeCL/vendor handles.
- Shared tensor/backend errors live with this model in `tenferro-tensor`.
  GPU-specific failure detail may wrap these errors from `tenferro-gpu`, but
  tensor core should not expose vendor-driver error types.

This replaces both current stringly `ComputeDevice { kind, ordinal }`-style
metadata and the separate `tenferro-internal-device` package boundary. The
resulting public API has one placement vocabulary and one dependency owner.

### Current Device API Mapping

The migration from the current `tenferro-internal-device` package should use
this ownership table:

```text
Current item or module                         Target owner
---------------------------------------------  -----------------------------
LogicalMemorySpace                            tenferro-tensor::MemoryKind
ComputeDevice                                 tenferro-tensor::DeviceId
Error, Result                                 tenferro-tensor
batch_index.rs helpers                        tenferro-tensor::shape
Generator::cpu and CPU default generator      tenferro-tensor::random
Generator::cuda and CUDA default generators   tenferro-gpu
cuda/** runtime modules                       tenferro-gpu::cuda
cuda_device_count and availability checks     tenferro-gpu
OpKind                                        tenferro-ops capability model
preferred_compute_devices                     remove
```

`LogicalMemorySpace` should not survive as a second public enum. Its
semantics map directly to `MemoryKind` plus optional `DeviceId`:

```text
LogicalMemorySpace::MainMemory              -> MemoryKind::UnpinnedHost
LogicalMemorySpace::PinnedMemory            -> MemoryKind::PinnedHost
LogicalMemorySpace::ManagedMemory           -> MemoryKind::Managed
LogicalMemorySpace::GpuMemory { device_id } -> MemoryKind::Device
                                              + DeviceId { kind: Gpu(..), ordinal: device_id }
```

The CPU generator can stay in `tenferro-tensor` only as a CPU tensor
construction utility. CUDA generator state should move to `tenferro-gpu`
because it depends on backend runtime state and may require vendor kernels.
There should be no public `Generator::cuda` constructor in `tenferro-tensor`.

The current tensor-side `Buffer::Cubecl(CubeclBuffer<T>)` variant should also
move out of `tenferro-tensor`. Tensor core should keep host storage plus an
opaque backend-owned buffer handle. `tenferro-gpu` can define its concrete
CubeCL buffer representation and store or resolve it through the backend handle
contract. This keeps CubeCL types out of the tensor public API and rustdoc.

### Opaque Backend Buffer Contract

The opaque backend buffer is load-bearing for GPU isolation and should be a
typed trait object owned by `tenferro-tensor`, not a bare numeric ID. A bare
`u64` handle would require a process-global allocation table or backend-global
lookup, both of which recreate hidden ownership.

Recommended shape:

```rust
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

pub trait BackendBuffer<T>: Debug + Send + Sync + 'static {
    fn backend_family(&self) -> &'static str;
    fn device(&self) -> DeviceId;
    fn len(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
}

#[derive(Clone, Debug)]
pub enum Buffer<T> {
    Host(Vec<T>),
    Backend(Arc<dyn BackendBuffer<T>>),
}
```

`tenferro-gpu` can then define `CubeclBuffer<T>` locally and implement
`tenferro_tensor::BackendBuffer<T>` for it. GPU execution validates placement,
backend family, device, and length, then downcasts through `as_any()` to the
local concrete buffer. This is orphan-rule safe because the implemented type is
local to `tenferro-gpu`.

This design gives tensor core:

- cloneable tensor storage through `Arc`,
- deterministic ownership and drop behavior without global buffer tables,
- no dependency on CubeCL or vendor handle types,
- enough metadata to produce useful "wrong backend/device" diagnostics.

`BackendBuffer<T>` should not expose raw pointers or vendor handles. Backend
crates may provide backend-specific accessors on their local buffer type.
Tensor core should treat backend buffers as opaque storage that can only be
interpreted by the backend that created them.

The current `OpKind` should not remain a device-selection enum. If backend
capability queries need a coarse operation classifier, define it in
`tenferro-ops` as part of a backend capability model, for example
`BackendCapabilityQuery`. Capability queries should be called on an explicit
backend or runtime owner, not through a free function that probes global device
availability.

### Cargo Boundary For Device And GPU

`tenferro-tensor` should not have `gpu`, `cuda`, or `rocm` features after this
migration. Its feature set should remain about CPU implementation choices such
as faer, BLAS providers, and source-built BLAS/LAPACK. These dependencies are
acceptable in tensor core because `CpuBackend` lives there.

Move these dependencies out of `tenferro-tensor`:

```text
cudarc
cubecl
cubecl-cuda
cubecl-runtime
tenferro-internal-gpubackend
```

`tenferro-gpu` should own those dependencies and expose vendor features inside
that isolated crate:

```toml
[features]
default = []
cuda = ["dep:cubecl", "dep:cubecl-cuda", "dep:cubecl-runtime", "dep:cudarc"]
rocm = ["dep:cubecl", "dep:cubecl-runtime"]
```

The exact dependency list can change with CubeCL, but the rule is stable:
enabling a GPU feature on `tenferro-gpu` must not change the compile-time
dependency set of `tenferro-tensor`, `tenferro-ops`, or `tenferro-runtime`.

### Feature Policy

Crate splitting should replace heavy default features. The target default
features should be conservative:

```text
Crate                  Default features
---------------------  -----------------------------------------------
tenferro-tensor        CPU implementation default only, such as cpu-faer
tenferro-runtime       no AD, no GPU
tenferro-ops           no AD
tenferro-linalg        primal linalg only, no AD
tenferro-einsum        primal einsum only, no AD
tenferro-fft           primal FFT only, no AD
tenferro-ad            AD enabled because the crate is the opt-in boundary
tenferro-linalg-ad     AD enabled because the crate is the opt-in boundary
tenferro-gpu           no default vendor backend unless release policy chooses one
```

The current `default = ["autodiff", "cpu-faer"]` pattern in `tenferro`,
`tenferro-internal-ops`, `tenferro-linalg`, `tenferro-einsum`, and
`tenferro-fft` contradicts crate-level opt-in. After the split, operation
crates should not have an `autodiff` feature that forwards into `tenferro-ad`.
Users opt into AD by depending on AD crates directly.

Extension crate feature forwarding should also stop referencing facade paths
such as `tenferro/autodiff`, `tenferro/cuda`, or
`tenferro-internal-tensor/cuda`. The new pattern is:

```toml
[features]
default = ["cpu-faer"]
cpu-faer = ["tenferro-tensor/cpu-faer"]
cpu-blas = ["tenferro-tensor/cpu-blas"]
cuda = []
rocm = []
```

Primal extension crates may expose empty concrete backend features such as `cuda`
or `rocm`
only if they gate their own GPU-specific primal code. They must not forward
those features into `tenferro-tensor` or `tenferro-runtime`. GPU execution is
selected by using `tenferro-gpu` as the backend crate.

## Dependency Direction

The intended dependency graph is one-way:

```text
tenferro-tensor
tenferro-ops       -> tenferro-tensor
tenferro-runtime   -> tenferro-ops -> tenferro-tensor
tenferro-ad        -> tenferro-runtime -> tenferro-ops -> tenferro-tensor
tenferro-linalg    -> tenferro-runtime -> tenferro-ops -> tenferro-tensor
tenferro-linalg-ad -> tenferro-linalg + tenferro-ad
tenferro-einsum    -> tenferro-runtime -> tenferro-ops -> tenferro-tensor
tenferro-fft       -> tenferro-runtime -> tenferro-ops -> tenferro-tensor
tenferro-gpu       -> tenferro-tensor
tenferro-ops-derive -> proc-macro dependencies only
```

`tenferro-runtime` must not depend on standard operation crates such as
`tenferro-linalg`, `tenferro-einsum`, or `tenferro-fft`.

`tenferro-tensor` must not depend on `tenferro-gpu`. This is the important GPU
boundary: CPU-only tensor users should not compile CubeCL, cudarc, CUDA, or
ROCm dependencies.

## Runtime And Trace Context API

`tenferro-runtime` should own tracing, graph compilation, program execution,
runtime extension dispatch, runtime caches, and graph metadata. It should not
own concrete tensor kernels, GPU driver initialization, or AD transforms.

Recommended top-level modules:

```text
tenferro_runtime::graph          GraphProgram, GraphCompiler, GraphExecutor
tenferro_runtime::trace          TraceContext, TracedTensor, traced_tensor ops
tenferro_runtime::extension      ExtensionRuntime, ExtensionRegistry
tenferro_runtime::exec           ExecOp evaluation and segmented dispatch
tenferro_runtime::metadata       context-owned tensor metadata
tenferro_runtime::cache          graph and runtime cache stats/limits
```

The long-term trace API should make graph metadata ownership explicit. A
`TraceContext` or equivalent builder should own traced value identifiers,
symbolic dimensions, and tensor metadata for one graph-building session:

```rust
use tenferro_runtime::{GraphCompiler, TraceContext};
use tenferro_tensor::DType;

let mut trace = TraceContext::new();
let x = trace.input("x", DType::F64, [2, 2]);
let y = tenferro_runtime::traced_tensor::add(&x, &x);
let program = GraphCompiler::new().compile(&trace, &[y])?;
```

The exact API can preserve ergonomic constructors, but metadata should not be
stored in process-global maps. If a convenience constructor creates a traced
constant without an explicit context, the resulting `TracedTensor` must carry
or reference its own context-owned metadata so that composing graphs does not
require a global lookup.

`GraphExecutor<B>` should remain backend-parametric and should own:

- backend value `B`,
- runtime extension registry for `B`,
- extension runtime caches,
- graph execution caches,
- optional execution options such as cache limits and workspace policy.

Runtime extension registration should remain explicit and attached to the
executor or builder:

```rust
let executor = GraphExecutor::builder(CpuBackend::new())
    .with_extension(tenferro_linalg::runtime())
    .with_extension(tenferro_einsum::runtime())
    .build()?;
```

The existing `executor.register_extension(tenferro_linalg::register_runtime)`
shape is also acceptable, but the final API should avoid process-global
runtime registration and should make missing extension runtimes a local
executor configuration error.

### `computegraph` Boundary

The split must choose whether `computegraph` is a public authoring dependency
or hidden behind `tenferro-runtime`. Because current extension crates use
`computegraph::FragmentBuilder`, `ValRef`, `OpMode`, `GlobalValKey`, and
`OpEmitter` directly, this is a semver-relevant boundary.

Recommended long-term decision: `tenferro-runtime` should expose a small
stable graph-authoring API and should not require normal extension authors to
import `computegraph` directly. `computegraph` can remain a workspace
foundation dependency, but `tenferro-runtime` should wrap or re-export only
the types that are part of tenferro's public extension contract:

```rust
pub mod graph_api {
    pub use computegraph::fragment::FragmentBuilder;
    pub use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
    pub use computegraph::OpEmitter;
}
```

If raw re-exports are chosen, `computegraph` version changes become part of
`tenferro-runtime` semver. If wrappers are chosen, the migration has more code
churn but a cleaner long-term boundary. The implementation should prefer
wrappers for user-facing extension APIs and allow internal runtime modules to
use raw `computegraph` types.

Design rule:

- `tenferro-ops` may depend on `computegraph` internally for operation
  vocabulary integration.
- `tenferro-runtime` owns the public graph-authoring surface used by extension
  crates.
- `tenferro-linalg`, `tenferro-einsum`, `tenferro-fft`, and AD rule crates
  should import graph-authoring types from `tenferro_runtime::graph_api` or
  an equivalent tenferro-owned module, not directly from `computegraph`.

## AD Registry Design

The current extension AD rule mechanism uses process-global registration. That
is acceptable as a short-term bridge but should not be the long-term design.

Long term, AD rules should be registered on an explicit owner:

```rust
let mut ad = tenferro_ad::AdRegistry::new();
tenferro_linalg_ad::register(&mut ad)?;
```

or on an AD-side builder that wraps runtime construction:

```rust
let runtime = tenferro_ad::RuntimeBuilder::new(tenferro_runtime::RuntimeBuilder::new())
    .with_ad(|ad| tenferro_linalg_ad::register(ad))
    .build()?;
```

The exact builder API can be deferred. The hard requirement is that AD rule
registration is explicit and not process-global.

`tenferro-runtime` should not depend on `tenferro-ad`. Any builder that owns AD
configuration should live in `tenferro-ad`, wrap a runtime builder, or use a
runtime-defined generic hook that does not mention AD types. The example above
uses an AD-side wrapper specifically to avoid a dependency cycle from runtime
back to AD.

`tenferro-ad` is the single public tenferro AD crate for tensor AD users.
`tidu`, `chainrules-core`, and `chainrules` may remain separate foundation
crates, but normal tensor AD users should not need to depend on them directly.
If AD rule authors need ChainRules-style error/result types, `tenferro-ad`
should either provide stable wrapper types or deliberately document those
foundation crates as public dependencies of the AD-rule authoring API.

Primitive AD rules for core `StdTensorOp` values should move out of
`tenferro-ops` and into `tenferro-ad`. `tenferro-ops` owns the primal operation
vocabulary and shape/dtype semantics; `tenferro-ad` owns linearization,
transpose, and AD metadata behavior. This avoids making `tenferro-ops` depend
on AD-only crates. The orphan-rule-safe pattern is for `tenferro-ad` to define
the AD dispatcher/registry traits or functions locally and pattern-match on
`tenferro-ops` operation values, rather than implementing a foreign trait for a
foreign type.

Recommended AD-side modules:

```text
tenferro_ad::context       AdContext, AdRegistry, rule registration
tenferro_ad::transforms    grad, value_and_grad, vjp, jvp, hvp
tenferro_ad::rules         primitive StdTensorOp AD rules
tenferro_ad::extension     ExtensionAdRule, ExtensionChainRule
tenferro_ad::metadata      AD-only linearization metadata
```

Recommended user-facing shape:

```rust
use tenferro_ad::AdContext;

let mut ad = AdContext::builder()
    .with_rule_set(tenferro_ad::core_rules())
    .with_extension_rules(tenferro_linalg_ad::rules())
    .build()?;
```

AD transforms should accept explicit runtime and AD owners instead of looking
up process-global rule state:

```rust
let grad_program = ad.grad(&program, &[input_id])?;
let out = executor.run(&grad_program)?;
```

This keeps runtime execution and AD transformation independently testable.
`tenferro-runtime` can compile and execute primal programs without linking
`tenferro-ad`, while `tenferro-ad` can depend on runtime graph types to build
transformed programs.

## Extension Runtime Registry

The runtime extension registry is already closer to the target shape: graph and
eager execution own extension runtime state and caches. The long-term
`tenferro-runtime` crate should keep that ownership model.

Operation crates should provide registration functions:

```rust
let mut executor = GraphExecutor::new(CpuBackend::new());
executor.register_extension(tenferro_linalg::register_runtime)?;
executor.register_extension(tenferro_einsum::register_runtime)?;
```

This pattern is more maintainable than implicit auto-discovery. It makes error
paths explicit: missing linalg runtime registration is a configuration error on
the executor that tried to run a linalg op.

Standard operation crates should expose three separate surfaces when needed:

```text
Surface                         Owner crate
------------------------------  ---------------------------------------------
Primal eager concrete tensors   tenferro-linalg, tenferro-einsum, tenferro-fft
Primal traced graph builders    same primal operation crate
Runtime registration            same primal operation crate
AD rule registration            tenferro-linalg-ad, optional tenferro-einsum-ad
```

For example, `tenferro-linalg` should not expose AD registration. It should
own linalg payloads, shape/dtype inference, eager linalg helpers, traced linalg
builders, and runtime execution registration:

```rust
let y = tenferro_linalg::traced_tensor::solve(&a, &b);
executor.register_extension(tenferro_linalg::register_runtime)?;
```

`tenferro-linalg-ad` should depend on `tenferro-linalg` and `tenferro-ad` and
should provide only AD rule registration:

```rust
let mut ad = tenferro_ad::AdRegistry::new();
tenferro_linalg_ad::register(&mut ad)?;
```

`tenferro-einsum` should keep `tensordot` as primal sugar in the einsum crate.
It should not require a separate crate unless its AD rules become heavy enough
to justify `tenferro-einsum-ad`.

## GPU Boundary

`tenferro-gpu` should own:

- CubeCL backend type and runtime initialization,
- CUDA/ROCm-specific environment and driver checks,
- upload/download helpers,
- GPU buffer representation if that representation requires CubeCL or vendor
  dependencies,
- implementation of tensor backend traits for the GPU backend.

`tenferro-tensor` should own only the backend traits and device-neutral tensor
contracts needed by all backends. It should not expose `CubeclBuffer` as part
of its public core API. If an opaque backend buffer abstraction is needed, it
should be vendor-neutral and should not require CubeCL dependencies.

The orphan-rule-safe pattern for GPU is:

- `TensorBackend` and backend-facing traits live in `tenferro-tensor`.
- `CudaBackend` and any GPU buffer types that require CubeCL/vendor
  dependencies live in `tenferro-gpu`.
- `tenferro-gpu` implements `tenferro_tensor::TensorBackend` for its local
  `CudaBackend` type. This is legal because the implementing type is local to
  `tenferro-gpu`.
- Third-party backend crates follow the same pattern: define their backend type
  locally and implement the public traits from `tenferro-tensor`.

Extension op authoring follows the corresponding operation pattern:

- `ExtensionOp` and primal extension payload traits live in `tenferro-ops`.
- Runtime execution traits and registries live in `tenferro-runtime`.
- AD rule traits and registries live in `tenferro-ad`.
- Extension crates implement the public traits for their local payload/rule
  types.

The current `tenferro-internal-gpubackend` name should not remain the public
opt-in boundary. It can be absorbed into `tenferro-gpu` or kept as an
unpublished/private implementation crate. If it remains a visible crate, it
should be treated as implementation detail rather than user-facing API.

Recommended decision: absorb `tenferro-internal-gpubackend` into
`tenferro-gpu` as private modules. Keeping a second unpublished GPU crate is
only worthwhile if compile times or backend implementation ownership require
it. The public user-facing boundary should still be only `tenferro-gpu`.

Recommended `tenferro-gpu` modules:

```text
tenferro_gpu::backend       CudaBackend and TensorBackend impl
tenferro_gpu::memory        upload_tensor, download_tensor, device buffer store
tenferro_gpu::device        availability and DeviceId validation
tenferro_gpu::cuda          CUDA-specific runtime and FFI, behind feature
tenferro_gpu::rocm          ROCm-specific runtime, behind feature when real
tenferro_gpu::linalg        cuSOLVER/cuBLAS/CubeCL linalg implementation
tenferro_gpu::kernels       GPU kernels and fusion codegen
```

Recommended public API:

```rust
use tenferro_gpu::{download_tensor, upload_tensor, CudaBackend};
use tenferro_tensor::{DeviceId, DeviceKind, GpuBackendKind};

let device = DeviceId {
    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
    ordinal: 0,
};
let mut backend = CudaBackend::new(device)?;
let gpu_x = upload_tensor(&mut backend, &cpu_x)?;
let cpu_x = download_tensor(&mut backend, &gpu_x)?;
```

`CudaBackend::cuda(0)` can exist as a convenience constructor, but the
canonical constructor should accept the vendor-neutral `DeviceId`. That keeps
runtime construction aligned with tensor placement metadata while still
isolating driver validation in `tenferro-gpu`.

GPU linalg gaps should continue to be reported by backend capability errors,
not by hiding fallback transfers inside tensor core. The no-implicit-transfer
policy remains unchanged: users or the runtime execution pipeline choose when
to upload or download tensors.

## Public Usage Model

Application and library code should import exactly what it uses:

```rust
use tenferro_tensor::{CpuBackend, Tensor};
use tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_linalg::traced_tensor::solve;

let a = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
let b = TracedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]);
let x = solve(&a, &b);

let mut compiler = GraphCompiler::new();
let program = compiler.compile(&x)?;

let mut executor = GraphExecutor::new(CpuBackend::new());
executor.register_extension(tenferro_linalg::register_runtime)?;
let out = executor.run(&program)?;
```

GPU usage is also explicit:

```rust
use tenferro_gpu::{download_tensor, upload_tensor, CudaBackend};
use tenferro_runtime::GraphExecutor;

let backend = CudaBackend::cuda(0)?;
let mut executor = GraphExecutor::new(backend);
```

AD usage is explicit:

```rust
use tenferro_ad::AdRegistry;

let mut ad = AdRegistry::new();
tenferro_linalg_ad::register(&mut ad)?;
```

The examples are intentionally not facade-based. Discoverability should come
from crate docs, examples, and an operation catalog rather than from one
umbrella import path.

## Benefits

- CPU-only builds do not accidentally compile AD or GPU dependency trees.
- Library authors express their public dependencies directly.
- Each crate has a smaller semver surface and clearer ownership.
- GPU environment failures occur in `tenferro-gpu`, not in tensor core.
- AD rules can evolve independently from primal operation crates.
- Rust feature unification becomes less dangerous because heavy capabilities
  move to separate crates instead of shared facade features.
- AI agents and maintainers can navigate ownership by crate name rather than
  following re-export chains.

## Costs And Risks

- Users import more crates explicitly.
- Documentation must teach crate composition rather than one facade path.
- Crate count and CI matrix grow.
- Moving `TensorBackend` implementations out of the tensor crate requires
  careful trait/type ownership to avoid Rust orphan-rule problems.
- `LinalgOp` and similar extension payload types may need a deliberately
  public contract so external AD crates can downcast and inspect them.
- Removing process-global AD registration is a larger design change than simply
  moving files between crates.

These costs are acceptable if the goal is long-term dependency hygiene and
clear subsystem ownership.

## Migration Sketch

1. Rename internal package shells in place:
   `tenferro-internal-tensor` to `tenferro-tensor`,
   `tenferro-internal-ops` to `tenferro-ops`, and
   `tenferro-internal-runtime` to `tenferro-runtime`.
2. Establish `tenferro-tensor` as the concrete tensor crate:
   absorb vendor-neutral device/error types, move concrete tensor free
   functions from the current `tenferro::{tensor, typed_tensor}` modules, and
   remove CubeCL/cudarc dependencies and `cuda`/`rocm` features.
3. Create `tenferro-gpu`:
   move the current `tenferro-internal-tensor/src/cubecl/**`,
   `tenferro-internal-device/src/cuda/**`, CUDA generator state, and
   `tenferro-internal-gpubackend` public backend surface into it.
4. Create the final `tenferro-runtime`:
   move current graph, traced, compiler, executor, segmented execution,
   extension runtime, and runtime cache modules into the renamed runtime
   package.
5. Replace process-global graph metadata with trace/runtime-owned metadata.
   This may be done in the same runtime phase or immediately before it if it
   simplifies moving `TracedTensor`.
6. Create `tenferro-ad`:
   move current autodiff-gated eager/tape APIs, primitive `StdTensorOp` AD
   rules, extension AD rule traits, and AD registries into it.
7. Split linalg AD:
   keep primal linalg APIs and runtime registration in `tenferro-linalg`, then
   move linalg AD rules and registration into `tenferro-linalg-ad`.
8. Update einsum and FFT to depend directly on `tenferro-runtime`,
   `tenferro-ops`, and `tenferro-tensor`; keep `tensordot` in
   `tenferro-einsum` as primal sugar.
9. Remove the current `tenferro` package from the workspace or mark it
   unpublished and empty only if a short mechanical transition needs it.
10. Remove `tenferro` facade paths from README, rustdoc, examples, tests,
    docs, and repository rules.
11. Add direct-crate examples for concrete tensor eager ops, CPU traced
    runtime, extension runtime registration, AD registration, and GPU runtime
    construction.

The migration should be done as one integrated breaking change or as a small
number of tightly staged PRs. Long-lived compatibility layers should be
avoided.

Recommended PR sequencing if the change is staged:

```text
PR 1: tensor/device cleanup, no GPU deps in tenferro-tensor
PR 2: tenferro-gpu extraction and GPU examples
PR 3: tenferro-runtime extraction and trace metadata ownership
PR 4: tenferro-ad extraction and linalg-ad split
PR 5: docs, AGENTS.md, REPOSITORY_RULES.md, README, examples, final cleanup
```

Each staged PR should leave the workspace building without compatibility
facades. If a staging crate is temporarily needed for mechanical moves, it
should be unpublished and deleted before the final PR.

## Repository Documentation And Agent Rules

The crate-boundary migration must update repository guidance in the same change
set as the code it describes. Otherwise AI agents and maintainers will continue
to recreate the old facade shape.

### `AGENTS.md`

`AGENTS.md` should stop describing `tenferro` as the public traced frontend or
facade crate. It should describe the no-facade crate model instead:

```text
tenferro-tensor    - Tensor values, dense layout, CPU backend, backend traits,
                    vendor-neutral device identifiers, placement, shared errors
tenferro-ops       - Standard op vocabulary, extension op carrier, symbolic dims
tenferro-ops-derive - Extension authoring proc macros, if public
tenferro-runtime   - Traced tensors, compiler, executor, extension runtime
tenferro-ad        - AD transforms, AD registry, chainrules/tidu integration
tenferro-linalg    - Primal linalg extension
tenferro-linalg-ad - Linalg AD rules
tenferro-einsum    - Einsum extension
tenferro-fft       - FFT extension
tenferro-gpu       - CubeCL-backed GPU backend
```

The "Standard Extension Boundary" section should also change from
"`tenferro` must not expose facade paths" to "`tenferro` is not the facade
crate; standard crates are imported directly." The rule should explicitly say
not to add a new umbrella crate unless a later design approves it.

The GPU status section should name `tenferro-gpu` as the public opt-in GPU
crate and treat any lower-level GPU kernel crate as implementation detail.

### `REPOSITORY_RULES.md`

`REPOSITORY_RULES.md` should gain a durable "No Facade Crate" rule:

- no public umbrella `tenferro` crate,
- no facade modules such as `tenferro::linalg`, `tenferro::einsum`, or
  `tenferro::gpu`,
- operation families and heavy capabilities are imported as direct crates,
- public crate names must match subsystem ownership,
- explicit registration is required for extension runtimes, AD rules, and GPU
  backends.

The "Standard Extension Boundary" rule should be updated to refer to
`tenferro-runtime` and direct operation crates instead of the old `tenferro`
facade.

The cache and registry ownership rules should mention AD registries explicitly:
long-lived AD registries should be owned by `tenferro-ad` contexts/builders, not
hidden in process-global state.

The "Documentation Policy" section should stop saying that online docs teach
the `tenferro` facade crate and stop requiring `use tenferro::{...}` imports.
The replacement policy should be:

- user-facing docs import each item from the public crate that owns it,
- docs may mention implementation crates only when those crates are themselves
  stable public crates,
- examples should show direct crate composition, such as
  `tenferro_tensor::{Tensor, CpuBackend}` plus
  `tenferro_runtime::{GraphCompiler, GraphExecutor, TracedTensor}`,
- no user-facing docs should import from retired `tenferro-internal-*`
  packages.

### User-Facing Docs

The following docs must be updated as part of the migration:

- `README.md`: replace all `tenferro::{...}` facade imports with direct crate
  imports.
- `tenferro/README.md`: remove or move with the crate rename; do not keep
  facade-oriented examples.
- crate-level rustdocs: every new public crate should explain its ownership and
  show direct-crate examples.
- `docs/design/index.md`: add the new crate-boundary document or replace the
  older extension-runtime restructure entry.
- `docs/design/extension-runtime-restructure.md`: either supersede it with the
  no-facade model or add a clear note that it is historical and no longer the
  target architecture.
- `docs/design/gpu-backend-design.md`: rewrite the public GPU API section in
  terms of direct `tenferro_gpu` imports, not `tenferro::cuda` facade paths,
  and update the owner of public GPU backend APIs from the tensor crate to
  `tenferro-gpu`.
- `docs/design/supported-ops.md`: update crate-by-crate operation ownership and
  distinguish primal operation support from AD-rule crate support.
- `docs/api_index.md`, if present in the branch at migration time: update
  import paths and crate ownership.
- examples under `README.md`, `docs/getting-started/`, and crate rustdocs:
  compile-check direct crate imports.

### Historical Plans

Existing files under `docs/plans/` should not be rewritten to match the new
architecture. They are historical records. New planning docs may reference that
older plans are superseded, but should not edit them retroactively.

## Verification Requirements

Each migration phase should verify the dependency boundaries it claims:

```bash
cargo check -p tenferro-tensor --no-default-features
cargo check -p tenferro-ops --no-default-features
cargo check -p tenferro-runtime --no-default-features
cargo check -p tenferro-linalg --no-default-features
cargo check -p tenferro-linalg-ad
cargo check -p tenferro-einsum --no-default-features
cargo check -p tenferro-fft --no-default-features
cargo check -p tenferro-gpu
```

Additional checks should confirm that:

- `tenferro-tensor` does not depend on CubeCL, cudarc, chainrules, or tidu.
- `tenferro-tensor` is the only owner of public device identifier, memory kind,
  placement, and shared tensor/backend error types.
- `tenferro-tensor` contains no CUDA runtime, CubeCL runtime, cudarc, or
  vendor-driver API.
- no crate defaults to AD except AD-specific crates such as `tenferro-ad` and
  `tenferro-linalg-ad`.
- no extension crate feature forwards through retired paths such as
  `tenferro/autodiff`, `tenferro/cuda`, or `tenferro-internal-tensor/cuda`.
- `tenferro-runtime` does not depend on standard operation crates.
- extension crates use the tenferro-owned graph authoring API rather than
  importing raw `computegraph` types directly, unless that dependency is
  explicitly accepted as public API.
- `tenferro-linalg` builds without AD dependencies.
- `tenferro-linalg-ad` restores the same differentiated linalg behavior when
  explicitly registered.
- `tenferro-gpu` is the only public crate that pulls GPU implementation
  dependencies and the only owner of CUDA/ROCm availability checks and
  CubeCL/vendor runtime handles.
- retired package names such as `tenferro`, `tenferro-internal-runtime`,
  `tenferro-internal-device`, and `tenferro-internal-extension-macros` are
  either removed, renamed, or left unpublished with explicit migration notes.

Before a PR, run the repository's normal formatting, test, coverage, rustdoc,
and docs-site checks.
