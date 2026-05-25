# Extension Runtime Restructure

This note sketches a post-`origin/main` restructure for making tenferro a lean
tensor graph runtime while moving domain-heavy operation families into
extension crates. The immediate motivation is einsum, but the design must also
cover linalg, FFT, future random/sparse/special packages, and non-AD builds.

The key decision is to treat crate boundaries and feature boundaries
separately:

- crate boundaries split operation domains,
- feature boundaries split optional capabilities such as automatic
  differentiation and CUDA.

## Goals

- Keep `tenferro` small enough to be the runtime/foundation crate.
- Keep `tenferro-internal-ops` focused on the minimal IR, primitive ops, and extension
  carrier.
- Move domain-specific APIs and implementation ownership into extension crates.
- Make extension crates usable directly, Rust-style, without requiring
  `tenferro` to re-export every domain.
- Let automatic differentiation be controlled by an `autodiff` feature rather
  than by separate AD crates.
- Make primal-only builds possible without `tidu`, `chainrules-core`, or
  `chainrules` dependencies, per issue #861.
- Support multi-output, mixed-dtype, backend-aware extension execution well
  enough for linalg.

## Non-Goals

- Do not make `tenferro` an ndarray replacement.
- Do not require Python/NumPy-style discoverability through one flat facade.
- Do not split AD into separate crates unless feature gates prove
  unmanageable.
- Do not optimize partial-output execution in the first migration. Correctness
  and a stable contract come first.

## Target Crate Shape

```text
tenferro
  Runtime/foundation crate:
  - TracedTensor / EagerTensor
  - GraphCompiler / GraphExecutor
  - generic ExtensionOp carrier and execution runtime
  - minimal primitive ops and wrappers
  - backend dispatch infrastructure

tenferro-internal-ops
  Minimal operation IR:
  - primitive StdTensorOp variants
  - StdTensorOp::Extension
  - primal ExtensionOp trait
  - autodiff-gated AD traits/registry/builders

tenferro-einsum
  Directly used extension crate:
  - tenferro_einsum::einsum / einsum_with / integer-label APIs
  - EinsumExtensionOp
  - parser, optimizer, lowering, eager/runtime implementation
  - extension-owned caches
  - autodiff-gated einsum AD rule

tenferro-linalg
  Future extension crate:
  - svd, qr, eig, eigh, cholesky, lu, solve, triangular_solve, ...
  - linalg extension ops and backend dispatch
  - linalg-specific AD rules under autodiff

tenferro-fft
  Existing external extension crate shape:
  - fft, ifft, rfft, irfft
  - should move AD registration behind autodiff

future:
  tenferro-random, tenferro-sparse, tenferro-special, tenferro-signal
```

Usage should be explicit:

```rust
use tenferro::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
use tenferro_einsum::einsum;
use tenferro_fft::{fft, FftNorm};
use tenferro_linalg::svd;
```

`tenferro` does not need to provide `tenferro::fft` or `tenferro::linalg`.
Discoverability should come from crate names, docs, examples, and catalog pages
rather than from a flat facade.

## Dependency Direction

Domain extension crates should be ordinary downstream crates of `tenferro`.
That keeps extension APIs close to the user-facing tensor types and matches
`tenferro-fft`:

```text
tenferro-einsum -> tenferro -> tenferro-internal-ops -> tenferro-internal-tensor
tenferro-fft    -> tenferro -> tenferro-internal-ops -> tenferro-internal-tensor
tenferro-linalg -> tenferro -> tenferro-internal-ops -> tenferro-internal-tensor
```

`tenferro` must not depend on `tenferro-einsum`, `tenferro-fft`, or
`tenferro-linalg`, otherwise extension crates cannot own their implementation
without creating facade cycles. If a batteries-included experience is useful
later, add a separate `tenferro-full` or `tenferro-prelude` crate rather than
making the foundation crate re-export every domain.

## Feature Naming

Public feature names should describe capabilities, not implementation details:

```toml
[features]
default = ["cpu-faer", "autodiff"]
autodiff = ["dep:tidu", "dep:chainrules-core", "dep:chainrules"]
cpu-faer = [...]
cpu-blas = [...]
gpu = [...]
cuda = ["gpu", ...]
rocm = ["gpu", ...] # planned
```

If an `ad` feature exists during migration, rename it to `autodiff` rather than
keeping both names long term. A temporary `ad = ["autodiff"]` alias is acceptable
inside the workspace during transition, but it should not be documented as a
stable public feature.

`gpu` should be the public capability feature for GPU infrastructure. Concrete
vendor backends should use vendor feature names: `cuda` now, `rocm` later.
`cubecl` may remain an internal module or private implementation detail, but
users should not need to select it directly. Because the current workspace
contains a visible `tenferro-cubecl` crate, a feature rename alone does not make
CubeCL invisible to workspace users. Rename the implementation crate to the
technology-neutral `tenferro-internal-gpubackend` as part of this restructure. That crate
can use CubeCL internally today and still leave room for CUDA, ROCm, or a
non-CubeCL implementation later. Step 1 of the migration should introduce `gpu`
as the shared capability feature, `cuda` as the first concrete vendor backend,
and update workspace/package paths from `tenferro-cubecl` to
`tenferro-internal-gpubackend`.

Feature semantics:

```text
gpu
  Enables GPU-facing abstractions and extension plumbing. It does not by itself
  promise a usable vendor backend.

cuda
  Enables the CUDA backend and implies gpu.

rocm
  Planned. Enables the ROCm backend and implies gpu.
```

## Extension Primal Contract

The primal extension contract must remain available without `autodiff`.

Required surface:

- stable `family_id`,
- deterministic `payload_hash`,
- structural `payload_eq`,
- fixed `n_inputs` and `n_outputs`,
- per-output shape and dtype inference,
- forward execution through a registered family executor.

Forward execution should receive an execution context. A context-free
`eager_execute(&[&Tensor])` method should not be part of the long-term primary
contract. Even host implementations may need context for planner caches,
workspace, profiling, determinism settings, or CPU buffer pools.

## Backend-Aware Extension Execution

Use one context-aware execution contract for eager and compiled execution. Do
not put a generic method such as `fn execute<B: TensorBackend>(...)` directly
on `dyn ExtensionOp`; generic trait methods are not object-safe.

The selected design is a backend-typed, object-safe executor registry:

```rust
pub trait ExtensionExecutor<B: TensorBackend>: Send + Sync + 'static {
    fn family_id(&self) -> ExtensionFamilyId;
    fn accepts_schema_version(&self, version: ExtensionSchemaVersion) -> bool;

    fn execute(
        &self,
        ctx: &mut ExtensionExecutionContext<'_, B>,
        op: &dyn ExtensionOp,
        inputs: &[&Tensor],
    ) -> tenferro_tensor::Result<Vec<Tensor>>;
}

pub struct ExtensionRegistry<B: TensorBackend> {
    executors: HashMap<ExtensionFamilyId, Arc<dyn ExtensionExecutor<B>>>,
}
```

`dyn ExtensionExecutor<B>` is object-safe because `B` is fixed at the registry
type. This avoids an erased-context downcast protocol. A `GraphExecutor<B>` or
`EagerRuntime<B>` executes extensions through the registry for its own backend
type. Extension crates may provide one executor that is generic over many
backends, or separate CPU/CUDA executors when backend-specific kernels or
planner caches are required.

The concrete shape should be:

```text
ExtensionOp
  object-safe payload and metadata:
  - family_id
  - schema_version
  - payload_hash / payload_eq
  - n_inputs / n_outputs
  - infer_output_meta

ExtensionExecutor
  object-safe family executor registered by family_id for one backend type B:
  - accepts_schema_version(schema_version)
  - execute(ctx, op, inputs) -> Vec<Tensor>
  - may downcast op payload after checking family_id and schema_version

ExtensionRegistry<B>
  runtime-owned mapping:
  - family_id -> ExtensionExecutor<B>
  - registration rejects duplicate family IDs for the same backend registry
  - execution rejects ops whose schema_version is not accepted by the executor
```

The executor context should provide enough information for:

- backend-specific execution,
- backend runtime caches,
- CPU buffer pools,
- GPU streams/sessions,
- dispatch mode,
- extension-owned runtime caches,
- temporary workspace management.

Conceptually, this is the typed shape each concrete backend implementation may
use internally:

```rust
pub struct ExtensionExecutionContext<'a, B: TensorBackend> {
    pub backend: &'a mut B,
    pub backend_runtime: &'a mut BackendRuntimeState<B>,
    pub extension_caches: &'a mut dyn ExtensionCacheStore,
    pub dispatch_mode: DispatchMode,
}
```

`BackendRuntimeState<B>` is owned by the runtime/session, not borrowed out of
`B`. This avoids aliasing `&mut B` and `&mut B::RuntimeCache` when a backend
implementation stores cache-like state internally. Backend implementations may
still delegate from `BackendRuntimeState<B>` into `B` if that is their local
implementation strategy, but the extension runtime contract treats backend
runtime state as an external execution resource.

The object-safe call exposed by the runtime is typed by the surrounding backend
registry:

```text
GraphExecutor<B>
  -> ExtensionRegistry<B>
  -> dyn ExtensionExecutor<B>
  -> ExtensionExecutionContext<'_, B>
```

Simple host-only extensions can ignore unused context fields; they should not
need a separate no-context method. The existing
`eager_execute(&[&Tensor]) -> Vec<Tensor>` method is a transitional API only.
After the registry lands, remove it from the primary trait and route eager and
compiled execution through the same context-aware executor surface.

Registration should be explicit and non-global. A runtime/session builder owns
the registry and hands immutable executor registrations plus mutable cache
storage to eager and compiled execution:

```rust
let extensions = ExtensionRegistry::<CpuBackend>::new()
    .with(tenferro_einsum::executor::<CpuBackend>())
    .with(tenferro_fft::executor::<CpuBackend>());

let executor = GraphExecutor::builder(cpu_backend)
    .extensions(extensions)
    .build();
```

Registries are per backend type and per runtime/session. A process using CPU
and CUDA/ROCm in the same run constructs one `ExtensionRegistry<CpuBackend>` and
one registry for each GPU backend type. There is no global cross-backend
registry in the first design; a future convenience layer may build multiple
typed registries from one extension catalog.

Convenience constructors may install standard extension crates in examples or a
future `tenferro-full` crate, but `tenferro` itself should not globally register
or depend on domain executors.

The public runtime surface needed by external extension crates is:

- construct a traced/eager extension application from an `Arc<dyn ExtensionOp>`,
- register an `ExtensionExecutor<B>` with an `ExtensionRegistry<B>`,
- access typed extension runtime caches by family-owned cache IDs,
- report per-output metadata and validation errors,
- return multiple output tensors from one extension execution,
- attach `autodiff` rules when the feature is enabled.

Einsum should not need private `tenferro` helpers once these APIs exist.

## Multi-Output Semantics

Multi-output extensions are first-class.

Rules:

- `n_outputs()` declares the number of output slots.
- `infer_output_meta()` returns exactly one `(DType, Vec<SymDim>)` per output.
- mixed output dtypes are legal.
- each output slot has independent metadata.
- the graph compiler and executor allocate, track, and reclaim output slots
  independently.
- output metadata is part of graph validation and compile-cache safety.

Linalg needs this immediately:

```text
svd(C64[m,n])  -> U: C64[m,k], S: F64[k], Vh: C64[k,n]
eigh(C64[n,n]) -> W: F64[n], V: C64[n,n]
eig(F64[n,n])  -> W: C64[n], V: C64[n,n]
lu(A)          -> multiple factor/pivot outputs
```

Before moving linalg, add tests for a synthetic extension returning mixed-dtype
multi-output values, for example `(C64 matrix, F64 vector)`.

## Partial Output Use

A graph may consume only some outputs of a multi-output extension:

```rust
let (_u, s, _vh) = svd(&x);
loss(&s)
```

Initial rule:

```text
The executor may materialize all outputs declared by an extension op.
Unused downstream values may be removed by graph DCE, but the runtime is not
required to split one extension execution into independently computed outputs.
Per-output lazy execution is future work.
```

This is conservative and fits linalg, where backend kernels often compute all
outputs together or expose separate modes only for some operations. It also
keeps the first migration focused on correctness.

Future extension metadata may describe output computability:

```text
AllOrNothing
IndependentlyComputable
SubsetModes([...])
```

but that should not block the first extension-runtime contract.

## Autodiff Feature Boundary

Use `autodiff`, not `ad`, as the public feature name.

Primal pieces available without `autodiff`:

- `ExtensionOp`,
- `StdTensorOp::Extension`,
- graph compilation and execution,
- eager forward execution,
- domain extension forward APIs.

Autodiff-gated pieces:

- `ExtensionAdRule`,
- `ExtensionChainRule`,
- `AdValue`,
- `FruleBuilder`,
- `RRuleBuilder`,
- extension AD registry,
- `PrimitiveOp` impls that depend on `chainrules-core`,
- `TensorInputKey` AD keying,
- `TracedTensor::{grad, jvp, vjp, ...}`,
- `EagerTensor::{requires_grad_in, grad, backward, ...}`,
- `tidu` integration,
- Cargo dependencies on `tidu`, `chainrules-core`, and `chainrules`.

Concrete current items to gate or move during implementation:

```text
tenferro-internal-ops/src/std_tensor_op.rs
  impl PrimitiveOp for StdTensorOp

tenferro-internal-ops/src/input_key.rs
  impl ADKey for TensorInputKey

tenferro-internal-ops/src/ext_op.rs
  ExtensionAdRule, ExtensionChainRule, AdValue,
  FruleBuilder, RRuleBuilder, AD support reporting

tenferro/src/error.rs
  Error variants that directly expose AD rule failures

tenferro/src/traced.rs and traced tensor API modules
  grad, try_grad, jvp, try_jvp, vjp, try_vjp, hvp-style helpers

tenferro/src/eager.rs and eager tensor API modules
  requires_grad_in, grad, clear_grad, tracks_grad, backward,
  eager tape/tidu integration
```

The feature introduction step is only naming. Issue #861 is not satisfied until
these impls and APIs are actually `#[cfg(feature = "autodiff")]` and
`cargo check --no-default-features` no longer builds `tidu`,
`chainrules-core`, or `chainrules`. `chainrules-rs` is the repository name;
`chainrules-core` and `chainrules` are the Cargo dependency names that need to
disappear from primal-only builds.

The current `tenferro-internal-ops/src/ext_op.rs` mixes primal extension and extension
AD. It should be split into an always-available primal module and an
`autodiff`-gated module, for example:

```text
tenferro_ops::ext_op      // primal ExtensionOp
tenferro_ops::ext_ad      // ExtensionAdRule, builders, registry
tenferro::extension       // primal facade
tenferro::extension::ad   // autodiff facade
```

This split is a prerequisite for no-AD builds that still support extension
forward execution.

## Multi-Output VJP Contract

The transpose rule for a multi-output extension receives one optional cotangent
per output:

```text
cotangent_out: Vec<Option<Cotangent>>
```

Semantics:

- `None` means that output is inactive in the differentiated scalar.
- `None` is not the same as a materialized zero tensor unless the rule chooses
  to treat it that way.
- a rule may support only some active-output subsets and return
  `Unsupported` for others.
- support should eventually be finer than one boolean per op.

Examples:

```text
svd:
  only S active      -> relatively common and useful
  U/S/Vh active      -> full rule, more complex
  only U or only Vh  -> may be unsupported initially

eigh:
  only eigenvalues active  -> simpler
  eigenvectors active      -> more complex, degeneracy-sensitive

eig:
  eigenvalues active       -> complex-valued even for real inputs
  eigenvectors active      -> more complex, degeneracy-sensitive

qr/lu:
  selected outputs may be active depending on downstream use
```

AD APIs should also allow callers or wrapper APIs to declare which outputs are
intended to participate in autodiff. This is separate from ordinary graph DCE:
an output may be computed for primal use while explicitly excluded from
autodiff.

The contract should distinguish three concepts:

```text
produced outputs
  All outputs the primal extension op returns.

used outputs
  Outputs consumed by the primal graph after ordinary DCE.

autodiff-active outputs
  Outputs whose cotangents are allowed to flow into the extension's VJP rule.
```

For linalg this matters because partial rules are common and useful. A wrapper
can expose an explicit active-output policy:

```rust
let (u, s, vh) = svd(&x);
let s = s.autodiff_active();
let u = u.stop_gradient();
let vh = vh.stop_gradient();
```

or, more ergonomically, operation-specific wrappers can encode the intended
policy:

```rust
let s = svd_values(&x);        // only singular values are autodiff-active
let (w, v) = eigh(&x);         // wrapper may mark both active
let w = eigvalsh(&x);          // only eigenvalues are autodiff-active
```

The lower-level extension rule should receive an active-output mask in addition
to the optional cotangents:

```rust
pub struct ExtensionVjpRequest<'a> {
    pub op: &'a dyn ExtensionOp,
    pub inputs: &'a [AdValue],
    pub outputs: &'a [AdValue],
    pub active_outputs: &'a [bool],
    pub cotangent_out: &'a [Option<Cotangent>],
}

pub trait ExtensionAdRule {
    fn vjp(&self, request: ExtensionVjpRequest<'_>) -> Result<Vec<Option<Cotangent>>>;
    fn jvp(&self, request: ExtensionJvpRequest<'_>) -> Result<Vec<Option<Tangent>>>;
}
```

These two fields are intentionally not the same. `cotangent_out[i]` describes
what the transformed graph currently needs. `active_outputs[i]` describes the
AD contract selected by the API, wrapper, or explicit stop-gradient boundary.
`active_outputs[i] == false` means output `i` is outside the requested autodiff
contract even if the primal value is produced and used. If a cotangent is
nevertheless present for an inactive output, the AD transform must treat the
inactive marker as a stop-gradient boundary and pass `None` to the extension
rule for that slot. It should not reject the graph merely because a primal user
kept the value alive; wrappers such as `svd_values` should compose predictably
with ordinary graph use.

This allows a rule to be precise:

```text
svd full rule:
  supports active outputs {S}
  may later support {U, S, Vh}
  rejects {U}, {Vh}, {U, Vh}, etc. until implemented

eigh:
  supports {W}
  may later support {W, V}
```

For the first implementation, the active-output mask can be derived from
ordinary graph activity plus explicit `stop_gradient` boundaries. Later, public
operation wrappers can expose more intentional APIs for common subsets.

The wrapper or API records the intended active-output policy; the AD transform
combines that policy with graph activity and `stop_gradient` boundaries before
invoking `ExtensionAdRule::vjp`.

The AD rule API already passes output cotangent options in spirit. The
restructure should preserve that, make the active-output meaning explicit, and
add tests for partial-output VJP behavior.

## Extension Cache Model

Cache identity should be typed, not string-mutation based.

Rules:

- extension families own cache semantics and cache IDs,
- `tenferro` owns generic storage/control/reporting infrastructure,
- public cache mutation uses `ExtensionCacheId`,
- no `*_by_name` or family-string mutation APIs,
- compiler-side and executor-side caches remain distinct,
- family IDs are globally namespaced strings, for example
  `tenferro.einsum`, `tenferro.fft`, and `tenferro.linalg`,
- schema versions are explicit and participate in executor/cache
  compatibility checks.

`family_id` identifies the operation family, not a single cache. Versioning
should be monotonic within a family. A new schema version may reuse old cache
entries only if the family executor declares compatibility; otherwise the
runtime treats caches as separate. Two crates must not register different
executors for the same `(family_id, schema_version)` unless they are the same
implementation version.
For the first implementation, `ExtensionExecutor::accepts_schema_version`
governs both op payload compatibility and cache-entry compatibility. If a
future family needs finer-grained cache compatibility, add a separate
`accepts_cache_from_schema_version` hook.

The selected ownership model is:

```text
GraphCompiler
  owns compile-time extension caches:
  - parser caches
  - symbolic/static planning caches
  - metadata inference caches

GraphExecutor<B> / EagerRuntime<B>
  owns backend runtime extension caches:
  - runtime contraction plans
  - FFT plans
  - backend workspace pools
  - backend-specific analysis caches
```

The storage object is a type map keyed by a typed cache key:

```rust
pub struct ExtensionCacheKey {
    pub family_id: ExtensionFamilyId,
    pub schema_version: ExtensionSchemaVersion,
    pub cache_id: ExtensionCacheId,
    pub backend: Option<BackendCacheKey>,
}

pub enum BackendCacheKey {
    BackendType(&'static str),
    Device { backend: &'static str, device_id: String },
    RuntimeInstance(u64),
}

pub enum ExtensionCacheSelector {
    All,
    Family(ExtensionFamilyId),
    Cache(ExtensionCacheKey),
}

pub enum ExtensionCacheLimit {
    Entries(usize),
    Bytes(usize),
    Unlimited,
}

pub trait ExtensionCacheStore {
    fn get_or_insert_any(
        &mut self,
        key: ExtensionCacheKey,
        init: &mut dyn FnMut() -> Box<dyn Any + Send + Sync>,
    ) -> &mut dyn Any;

    fn clear(&mut self, selector: ExtensionCacheSelector);
    fn stats(&self, selector: ExtensionCacheSelector) -> ExtensionCacheStats;
    fn set_limit(&mut self, key: ExtensionCacheKey, limit: ExtensionCacheLimit);
}

pub trait ExtensionCacheStoreExt {
    fn get_or_insert<T: Any + Send + Sync>(
        &mut self,
        key: ExtensionCacheKey,
        init: impl FnOnce() -> T,
    ) -> tenferro_tensor::Result<&mut T>;
}
```

`ExtensionCacheStore` is object-safe for runtime contexts. The generic
`ExtensionCacheStoreExt::get_or_insert<T>` helper can live as an extension trait
implemented for `dyn ExtensionCacheStore`, performing downcast checks and
returning a typed cache entry to extension crates. Public controls use selectors
such as `Family(family_id)`, `Cache(key)`, or `All`; they do not expose string
mutation APIs.

Use two distinct stores, not one store discriminated only by key: one
compile-time store owned by `GraphCompiler`, and one runtime store owned by each
`GraphExecutor<B>` or `EagerRuntime<B>`. Runtime stores are per runtime/session
instance and externally serialized through `&mut self` execution APIs in the
first design. Sharing one cache store across concurrent executors requires an
explicit synchronization wrapper and is future work.

`BackendCacheKey` should be omitted for compile-time caches and present for
backend runtime caches. Prefer `Device` for GPU planner caches that depend on a
specific device, `BackendType` for backend-wide CPU planner state, and
`RuntimeInstance` only for state that cannot be shared safely across runtime
instances.

Einsum likely needs:

```text
parse cache                 compiler or extension build side
static contraction plans     compiler/build side
runtime contraction plans    executor/runtime side
```

FFT may need planner caches later.

Linalg may need backend analysis or factorization plan caches later.

The important design point is that `GraphCompiler` and `GraphExecutor` should
not contain hard-coded `einsum_*` fields once `tenferro-einsum` is a true
extension crate.

## Domain Migration Notes

### Einsum

Einsum should be the first migrated domain because it is mostly single-output
and the current experimental branch already exposes the right failure modes.

Move into `tenferro-einsum`:

- `EinsumSubscripts` or an equivalent integer-label API,
- `EinsumExtensionOp`,
- `tenferro.einsum` family identity and schema versions,
- cache IDs and cache semantics,
- traced API `tenferro_einsum::einsum`,
- eager/runtime execution,
- `autodiff`-gated AD rule.

`tenferro-einsum` should own all implementation details: parsing, integer-label
normalization, path planning, static contraction tree construction, lowering,
runtime execution, and cache policy. `tenferro` should provide only the generic
extension application API, executor registry, runtime context, and cache storage.
Any helper currently required by the prototype but private to `tenferro` should
be promoted to a domain-neutral public API or removed from the extension path.
`tenferro-einsum` is an implementation crate, not a facade-only crate, so unit
tests may live there.

Remove from `tenferro-internal-ops`:

- `StdTensorOp::NaryEinsum`,
- einsum-specific AD rules,
- einsum-specific support manifest entries.

Remove from `tenferro`:

- dependency on `tenferro-einsum`,
- `tenferro::einsum` facade unless a separate batteries-included crate is
  created,
- hard-coded einsum compiler/executor cache fields,
- all einsum-named implementation files and tests.

Concrete files from the current branch/prototype to move or delete include:

```text
tenferro/src/einsum.rs
tenferro/src/eager_einsum.rs
tenferro/src/einsum_subscripts.rs
tenferro/src/einsum_extension.rs
tenferro/src/tensor.rs einsum facade functions
tenferro/src/typed_tensor.rs einsum facade functions
tenferro/src/eager_tensor.rs einsum facade functions
tenferro/tests/graph_einsum.rs
tenferro/tests/tensor_einsum.rs
tenferro/tests/einsum_ad.rs
tenferro/tests/einsum_extension_cache.rs
tenferro/tests/einsum_extension_symbolic.rs
tenferro/tests/nary_einsum_cache.rs
tenferro/tests/nary_einsum_symbolic.rs
```

Post-migration acceptance criterion:

- `tenferro/Cargo.toml` has no normal dependency on `tenferro-einsum`,
- `tenferro/src/**/*.rs` contains no public item named `einsum*`,
- `tenferro` exports nothing named `einsum*`,
- `tenferro` has no hard-coded einsum cache fields or cache APIs,
- no `pub` item in `tenferro` exists solely to support `tenferro-einsum`,
- implementation and AD tests for einsum live in `tenferro-einsum/tests`.

A `dev-dependency` from `tenferro` tests to `tenferro-einsum` is allowed only
for runtime/extension integration coverage. It must not leak into normal
dependencies or public API.

### Linalg

Linalg should not be the first full migration. It is the stress test for the
extension runtime.

Before migrating linalg, solve:

- backend-aware extension execution,
- mixed-dtype multi-output metadata,
- partial output use,
- multi-output VJP support,
- extension AD behind `autodiff`,
- public graph-building helpers needed by an external crate.

Linalg candidates for `tenferro-linalg`:

- `svd`, `qr`, `eig`, `eigh`,
- `cholesky`, `lu`, `full_piv_lu`,
- `solve`, `full_piv_lu_solve`, `triangular_solve`,
- derived wrappers such as `det`, `inv`, `pinv`, `norm`.

Core may keep `dot_general` and minimal primitives used by linalg lowering.

Concrete phase-1 boundary:

- keep existing linalg backend kernels and low-level backend traits in
  `tenferro-internal-tensor` until extension backend dispatch is proven,
- keep direct `StdTensorOp` linalg variants in `tenferro-internal-ops` until the
  multi-output extension executor supports mixed dtypes and AD-active output
  masks,
- move only public wrappers that can be expressed through the generic extension
  surface after that surface exists,
- remove direct linalg variants only after parity tests cover eager, compiled,
  shape inference, AD support reporting, and GPU/backend paths.

The variants to account for are:

```text
Cholesky, Lu, FullPivLu, Solve, FullPivLuSolve,
Svd, Qr, Eigh, Eig, TriangularSolve, ValidateNonsingular
```

### FFT

`tenferro-fft` is already close to the desired external extension-crate shape.

Needed adjustments:

- put AD registration behind `autodiff`,
- consider backend-aware execution before adding GPU/cuFFT support,
- add planner caches through the generic extension cache model when needed.

### Random

Random is a good future extension because it differentiates tenferro from
plain ndarray-style helpers if it is graph/backend aware:

- explicit keys/seeds,
- CPU/GPU reproducibility policy,
- graph random ops,
- non-differentiability or reparameterization rules under `autodiff`.

## Alternatives Considered

### One Domain Crate With Features

An alternative is a single `tenferro-domains` crate:

```toml
tenferro-domains = { features = ["einsum", "fft", "linalg", "random"] }
```

This improves discoverability and reduces dependency-list noise, but it makes
independent release cadence, compile parallelism, optional heavy dependencies,
and domain ownership worse. It also recreates a broad facade under a different
name.

Prefer separate crates for now:

```text
tenferro-einsum
tenferro-fft
tenferro-linalg
tenferro-random
```

A later `tenferro-full` can re-export common crates if users want a bundled
experience.

## Migration Plan

Start from `origin/main`, not from the current einsum experiment branch.

1. Stabilize names and direction.
   - introduce `autodiff` as the stable public AD feature,
   - remove or temporarily alias any existing `ad` feature to `autodiff`,
   - expose `gpu` as the shared public GPU capability feature,
   - expose `cuda` as the first concrete vendor backend feature and make it
     imply `gpu`,
   - reserve `rocm` as the future ROCm backend feature, also implying `gpu`,
   - stop requiring users to select implementation-facing `cubecl`,
   - rename the implementation crate from `tenferro-cubecl` to
     `tenferro-internal-gpubackend` in this restructure,
   - document that domain crates are imported directly rather than re-exported
     from `tenferro`,
   - keep existing defaults initially unless publish constraints require a
     lean default.

2. Redesign primal extension execution.
   - introduce `ExtensionRegistry<B>` and object-safe `ExtensionExecutor<B>`,
   - add `accepts_schema_version` to the executor contract,
   - make context-aware execution the primary forward contract,
   - remove context-free `eager_execute` from the long-term trait,
   - ensure eager and compiled execution use the same semantic surface,
   - let simple host-only extensions ignore context fields,
   - make registration explicit through runtime/session builders, not global
     initialization,
   - expose the public runtime helpers external extension crates need,
   - add object-safety tests for `dyn ExtensionExecutor<B>`.

3. Split primal extension from extension AD.
   - keep `ExtensionOp` always available,
   - gate AD registry/builders/rules behind `autodiff`,
   - gate `PrimitiveOp`/`ADKey` impls and tensor AD APIs behind `autodiff`,
   - add no-AD compile checks,
   - add CI for `cargo check --no-default-features`,
   - add CI for `cargo check --no-default-features --features cpu-faer`,
   - add a dependency assertion such as
     `cargo tree --no-default-features -e normal -p tenferro` and equivalent
     per-crate checks that fail if `tidu`, `chainrules-core`, or `chainrules`
     appear.

4. Lift extension graph/runtime support to multi-output.
   - ensure graph nodes can represent N extension outputs,
   - ensure per-output dtype and shape metadata flows through shape inference,
     compiler validation, segmenting, and execution,
   - ensure `GraphExecutor<B>` returns and stores all extension outputs without
     assuming a single output tensor,
   - ensure mixed-dtype outputs are accepted before linalg migration.

5. Add multi-output and mixed-dtype extension tests.
   - synthetic extension returning multiple outputs,
   - partial output use,
   - explicit autodiff-active output masks,
   - inactive outputs behave as stop-gradient when cotangents would otherwise
     flow,
   - compile cache correctness,
   - eager and compiled execution.

6. Move einsum to `tenferro-einsum`.
   - remove direct `NaryEinsum` from `tenferro-internal-ops`,
   - expose `tenferro_einsum::einsum`,
   - move parser/subscripts/path planning/lowering/runtime/cache ownership out
     of `tenferro`,
   - remove all public `tenferro::einsum` and
     `tenferro::{tensor,typed_tensor,eager_tensor}` einsum facade functions
     unless a compatibility crate is created,
   - remove all hard-coded einsum cache APIs from `GraphCompiler` and
     `GraphExecutor`,
   - move tests into `tenferro-einsum/tests`,
   - keep AD under `autodiff`,
   - verify the structured acceptance criteria from the Einsum section,
     including no normal `tenferro -> tenferro-einsum` dependency and no
     `tenferro` public `einsum*` exports.

7. Revisit linalg.
   - first make linalg work as extensions while backend kernels may still live
     in `tenferro-internal-tensor`,
   - later decide whether kernel ownership should also move.

8. Update docs and examples.
   - document extension crate catalog,
   - stop implying all domains live under `tenferro::*`,
   - explain `autodiff`, `gpu`, `cuda`, and planned `rocm` features.

## Open Questions

- Should `tenferro` default to `autodiff`, or should the crates.io foundation
  pass default to primal-only?
- Should a future `tenferro-full` or `tenferro-prelude` crate re-export common
  domain crates, or should all domain crates remain explicit?
- How much of linalg backend kernel ownership belongs in `tenferro-internal-tensor`
  versus `tenferro-linalg`?
- How fine-grained should AD support reporting become for partial multi-output
  rules?
