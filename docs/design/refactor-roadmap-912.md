# Issue 912 Refactor Roadmap

Status: approved design.

Issue #912 completes the workspace-wide cleanup tracked by issues #903 through
#911. The target is a cleaner end state, not a compatibility-preserving
migration. Old APIs, aliases, duplicated dispatch paths, and ad hoc matches
should be removed when the new owner is in place.

## Goals

- Move linalg ownership fully into `tenferro-linalg`.
- Replace scattered primitive-op metadata with a single core-op catalog.
- Collapse runtime and GPU dispatch duplication into descriptor-driven tables.
- Remove `TensorExec` and make backend sessions the operation execution surface.
- Make `try_*`, traced extension `Result`, crate-local `Result` aliases, and
  backend error construction consistent.
- Keep file and test cleanup driven by ownership boundaries, not line count
  alone.

## Core Op Boundary

Create `tenferro-core-ops` with crate name `tenferro_core_ops`.

This crate owns core primitive operation vocabulary and metadata:

- `PrimitiveOpKind`
- primitive descriptors
- arity/category/effect metadata
- dtype policies used by runtime and backend dispatch
- catalog-generation macros

Use `macro_rules!` for the catalog source of truth. Do not use a proc macro or
`build.rs`.

The crate does not contain linalg, FFT, or einsum. Those remain standard
extension operation families owned by their crates.

`tenferro-internal-ops` remains the graph-facing integration layer:

- `StdTensorOp` carrier and payloads
- conversion from `StdTensorOp` to `PrimitiveOpKind` where applicable
- extension op carrier/glue
- graph-level primitive AD rule emission

The catalog should remove repeated match groups for metadata, support
classification, dtype policy, and dispatch lookup. It should not force payload
data out of `StdTensorOp`; payload-bearing operations can still store their
configuration in `StdTensorOp` while using `PrimitiveOpKind` for table lookup.

## Primitive AD Registry

Primitive AD rules use a static function table keyed by `PrimitiveOpKind`.

The registry shape is conceptually:

```text
PrimitiveAdRule {
    kind: PrimitiveOpKind,
    linearize: fn(&StdTensorOp, graph_context) -> Result<outputs>,
    transpose: fn(&StdTensorOp, graph_context) -> Result<outputs>,
}
```

Rule functions may inspect the full `StdTensorOp` payload. The key point is
that rule lookup is table-driven, while semantic graph emission remains in
`tenferro-internal-ops`.

The existing rule source-of-truth rule still applies: supported AD rules need
corresponding oracle or finite-difference coverage before being treated as
mainline support.

## Runtime And Extension Dispatch

Runtime dispatch should be table-driven at the single point where an op is
selected. It should not become a fully dynamic execution system.

Core primitive dispatch:

```text
StdTensorOp -> PrimitiveOpKind -> runtime descriptor -> backend/session call
```

Extension dispatch:

```text
StdTensorOp::Extension -> family id -> registered extension runtime
```

Standard extensions use family descriptors/registries, not the core primitive
catalog. This keeps standard extension families first-class crates rather than
turning them into hidden core ops.

`tenferro-internal-extension-macros` should provide a function-like macro:

```rust
define_extension_runtime! {
    runtime,
    family_id,
    op_type,
    execute,
    register_fn
}
```

The macro generates the extension runtime implementation and registration
function. Related macro helpers may handle idempotent AD rule registration, but
process-global AD registration should not be the primary new design.

## GPU Dispatch

GPU cleanup is part of #912 completion.

The final GPU dispatch shape is:

```text
PrimitiveOpKind -> GpuOpDescriptor -> dtype policy -> typed launch
```

`tenferro-gpu` owns CubeCL/CUDA backend resources, kernel launch helpers, FFI
handles, buffer pools, and backend-native execution. It should not own linalg as
a public operation family.

The cleanup target is:

- no op-local hand-written dtype match groups in the final steady state
- dtype dispatch isolated at typed launch boundaries
- reusable generic helpers or declarative macros for unavoidable scalar
  instantiation
- no hidden CPU/GPU transfers
- no single-thread GPU fallbacks for tensor-sized work
- CUDA context, FFI handles, caches, and buffer pools stay on the backend owner

The core catalog supplies operation identity and dtype policy. GPU descriptors
map those policies to backend launches.

## Backend Session Model

Remove `TensorExec`.

`TensorBackend` becomes the backend owner and session factory. It owns device
state, resource pools, caches, contexts, and backend identity.

`BackendSession<'a>` is a lightweight borrowing wrapper that implements the
operation category traits. Direct/eager calls may create short-lived sessions;
graph execution should create and reuse a session while evaluating a program.

Short-lived sessions are intentionally cheap. They must not own CUDA contexts,
FFI handles, buffer pools, or caches.

## Linalg Ownership

`tenferro-linalg` fully owns linalg.

Remove direct tensor linalg methods such as:

- `Tensor::svd`
- `Tensor::qr`
- `Tensor::cholesky`
- `Tensor::eigh`
- `Tensor::eig`
- `Tensor::triangular_solve`

Do not add compatibility aliases or deprecated shims.

`tenferro-tensor` and `tenferro-gpu` should be linalg-free at the public
trait/core surface. `tenferro-tensor` keeps core tensor execution and GEMM.
Linalg-only CPU dependencies and implementations move to `tenferro-linalg`.
If a lower-level GEMM dependency remains necessary for tensor core operations,
only that GEMM portion stays in `tenferro-tensor`.

`tenferro-linalg` exposes:

- eager tensor APIs under `tenferro_linalg::eager_tensor::*`
- traced tensor APIs under `tenferro_linalg::traced_tensor::*`
- direct tensor helpers if needed, owned by the linalg crate

`tenferro-linalg/cuda` enables `tenferro-gpu/cuda`; GPU linalg implementation
lives under `tenferro-linalg/src/gpu/`. `tenferro-gpu` remains the backend,
runtime, and buffer owner.

## Public API Consistency

Use `try_*` only for Result-returning checked variants. Rename `try_grad` to
`grad_optional`. Do not keep aliases.

Traced extension APIs that can fail return `Result`. Convert panic/assert
validation in public or traced extension paths to typed errors.

Each crate owns its own public error type and crate-local:

```rust
pub type Result<T> = std::result::Result<T, Error>;
```

Do not collapse all crate `Result<T>` aliases into one lower-layer canonical
alias. Remove only redundant aliases within the same crate.

Production code should not construct `Error::BackendFailure` directly. Use:

```rust
Error::backend_failure(op, message)
```

Tests may pattern-match on the enum variant. Test expected values should use
the helper. Rustdoc enum variant examples may remain where useful.

## Documentation

User-facing docs should describe public crates and APIs, not internal
source-code details.

Update README, rustdoc, and guide examples so direct tensor execution only
shows core tensor operations. Linalg examples move to the standard operation
crate sections:

- eager linalg uses `tenferro_linalg::eager_tensor::*`
- traced linalg uses `tenferro_linalg::traced_tensor::*` and `?`
- examples of removed direct methods such as `Tensor::svd` are deleted

Keep docs aligned with the no-facade-crate architecture: users import
`tenferro-linalg`, `tenferro-einsum`, and `tenferro-fft` directly.

## File And Test Cleanup

Line count is a review trigger, not the reason to split.

Inline `#[cfg(test)] mod tests { ... }` blocks should be essentially absent
from normal modules. Leave only tiny leaf-module tests inline when they are
trivially small. Prefer module-local `src/<module>/tests/*.rs` and keep only
`#[cfg(test)] mod tests;` in production modules.

Audit files above roughly 1000 lines. Split only when there is a clear
behavior, abstraction, feature, ownership, public/private, dispatch,
validation, cache, or backend-glue boundary. Do not split into arbitrary
`part1`/`part2` files. If a large file remains one coherent concern, document
why rather than splitting mechanically.

## Verification

Use targeted checks while implementing:

- affected crate tests after local changes
- `cargo check --workspace` after catalog, backend trait, and linalg milestones
- CUDA targeted ignored tests after GPU dispatch changes
- doctests and docs-site checks after API/doc changes

Final gate:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Run applicable CUDA tests with the repository CUDA environment when GPU code is
changed.

## Residual Redundancy Policy

After #912, remaining redundancy should be intentional and localized:

- FFI ABI glue may repeat externally required signatures.
- Macro expansion may instantiate per-dtype code, but the source should remain
  single-entry.
- Public docs may summarize behavior, but source-level descriptors remain the
  implementation source of truth.
- Core primitive op metadata, dtype policy, AD rule lookup, runtime dispatch,
  and GPU dispatch should not each maintain independent hand-written match
  groups for the same operation set.
