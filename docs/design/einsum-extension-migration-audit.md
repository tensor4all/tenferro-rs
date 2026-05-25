# Einsum Extension Migration Audit

This is the working audit table for moving the current einsum experiment into a
real external extension crate. Implementation should start from `origin/main`,
but the current branch is the source inventory. The goal is to reuse the
existing code deliberately, not to regenerate an einsum implementation from
scratch.

## Rules

- Treat this document as the migration checklist.
- Move or adapt the listed source items before inventing replacement code.
- If an item is intentionally not moved, record the reason in the implementation
  PR.
- The owning agent may implement, but this thread must verify each step against
  this table before moving on.
- `tenferro-einsum` owns einsum implementation. `tenferro` owns only generic
  extension runtime surface.

Status values:

```text
pending       not started
ported        code moved/adapted
verified      verification command passed
not-needed    intentionally replaced or removed with reason
```

## Pre-Migration Setup

| Source item | Current path | Target | Verification | Status |
|-------------|--------------|--------|--------------|--------|
| Clean implementation base | git branch/worktree | Start implementation from `origin/main`, using this experimental branch only as source inventory | `git merge-base --is-ancestor origin/main HEAD` on the implementation branch, or record the worktree base explicitly | pending |
| GPU feature naming | `Cargo.toml`, crate manifests | Use `gpu` as shared capability feature, `cuda` as first vendor backend, and reserve `rocm` | `rg -n "gpu|cuda|rocm|cubecl" Cargo.toml tenferro*/Cargo.toml` | pending |
| GPU implementation crate rename | `tenferro-cubecl` | Rename to `tenferro-internal-gpubackend` in this restructure | `rg -n "tenferro-cubecl|tenferro-internal-gpubackend" Cargo.toml tenferro*/Cargo.toml` | pending |
| AD feature naming | crate manifests | Use `autodiff`; do not expose stable `ad` feature | `rg -n "\\bad\\b|autodiff|tidu|chainrules" Cargo.toml tenferro*/Cargo.toml` | pending |

## Existing Assets To Reuse

| Source item | Current path | Target | Verification | Status |
|-------------|--------------|--------|--------------|--------|
| String/integer subscript parser core | `tenferro-einsum/src/syntax/subscripts.rs`, `syntax/notation.rs` | Keep in `tenferro-einsum`; do not duplicate parser in `tenferro` | `cargo test -p tenferro-einsum subscripts` | pending |
| Nested contraction syntax | `tenferro-einsum/src/syntax/nested.rs` | Keep in `tenferro-einsum`; reuse for `EinsumOptimize::Nested` | `cargo test -p tenferro-einsum nested` | pending |
| Contraction planning and optimizer options | `tenferro-einsum/src/planning/*` | Keep in `tenferro-einsum`; expose through traced API as needed | `cargo test -p tenferro-einsum planning` | pending |
| Fragment lowering | `tenferro-einsum/src/builder.rs` | Reuse from extension executor/runtime path | `rg -n "build_einsum_fragment" tenferro-einsum/src` | pending |
| Standalone eager tensor execution | `tenferro-einsum/src/eager.rs` | Keep as direct `tenferro_einsum::eager_*` APIs | `cargo test -p tenferro-einsum eager` | pending |
| Typed eager facade | `tenferro-einsum/src/typed_eager.rs` | Keep in `tenferro-einsum`; remove `tenferro::typed_tensor::einsum*` facade | `cargo test -p tenferro-einsum typed_eager` | pending |
| Eager runtime plan cache | `tenferro/src/eager.rs`, `tenferro-internal-runtime/src/extension_runtime.rs` | Owned by `EagerRuntime` through `ExtensionExecutor<EagerBackend>`; direct `tenferro-einsum` eager APIs do not retain a hidden cache | `cargo test -p tenferro-einsum --test traced_graph_cache eager_einsum` | done |

## Move From `tenferro` To `tenferro-einsum`

| Source item | Current path | Target | Verification | Status |
|-------------|--------------|--------|--------------|--------|
| Integer-label public API wrapper `EinsumSubscripts` | `tenferro/src/einsum_subscripts.rs` | `tenferro-einsum/src/api/subscripts.rs` or folded into `Subscripts` API | `rg -n "EinsumSubscripts|parse_einsum_subscripts" tenferro-einsum/src tenferro-einsum/tests` | pending |
| Traced public API `einsum`, `einsum_with`, `einsum_subscripts`, `einsum_subscripts_with` | `tenferro/src/einsum.rs` | `tenferro-einsum/src/traced.rs`; exported as `tenferro_einsum::traced_tensor::einsum*` with root compatibility re-exports | `rg -n "pub fn einsum" tenferro-einsum/src && ! rg -n "pub fn einsum" tenferro/src` | pending |
| `EinsumOptimize` and strategy conversion | `tenferro/src/einsum.rs` | `tenferro-einsum/src/traced.rs` or `src/optimize.rs` | `rg -n "EinsumOptimize|resolve_strategy|nested_to_v1_pairs" tenferro-einsum/src` | pending |
| Symbolic optimization validation | `tenferro/src/einsum.rs` | `tenferro-einsum` traced API, preserving current symbolic restrictions | `rg -n "symbolic einsum supports only default automatic optimization" tenferro-einsum` | pending |
| Traced extension payload construction | `tenferro/src/einsum.rs`, `build_einsum_extension_tensor` | `tenferro-einsum` traced API using generic `tenferro::extension::apply` | `rg -n "build_einsum_extension_tensor|ExtensionOp" tenferro-einsum/src` | pending |
| Eager AD-facing API for `EagerTensor` | `tenferro/src/eager_einsum.rs` | `tenferro-einsum/src/eager_tensor.rs`, under `autodiff` where needed | `rg -n "EagerTensor|requires_grad|StdTensorOp::Extension" tenferro-einsum/src` | pending |
| Tensor facade eager APIs | `tenferro/src/tensor.rs` | Remove from `tenferro`; direct users call `tenferro_einsum::eager_*` | `! rg -n "einsum" tenferro/src/tensor.rs` | pending |
| TypedTensor facade eager APIs | `tenferro/src/typed_tensor.rs` | Remove from `tenferro`; direct users call `tenferro_einsum::typed_eager_*` | `! rg -n "einsum" tenferro/src/typed_tensor.rs` | pending |
| EagerTensor facade APIs | `tenferro/src/eager_tensor.rs` | Remove from `tenferro`; direct users call `tenferro_einsum` | `! rg -n "einsum" tenferro/src/eager_tensor.rs` | pending |

## Move Extension Payload, Execution, And Caches

| Source item | Current path | Target | Verification | Status |
|-------------|--------------|--------|--------------|--------|
| Family identity | `tenferro/src/einsum_extension.rs`, `EINSUM_EXTENSION_FAMILY_ID` | `tenferro-einsum/src/extension.rs`, family `tenferro.einsum` plus schema version | `rg -n "tenferro\\.einsum|schema" tenferro-einsum/src` | pending |
| Cache IDs: static plans, parse, runtime plans | `tenferro/src/einsum_extension.rs`, `EINSUM_*_CACHE_ID` | `tenferro-einsum` cache module with typed IDs | `rg -n "STATIC.*PLAN|PARSE|RUNTIME.*PLAN|ExtensionCache" tenferro-einsum/src` | pending |
| Default cache capacity | `tenferro/src/einsum_extension.rs`, `DEFAULT_EINSUM_CACHE_CAPACITY` | `tenferro-einsum` cache module | `rg -n "DEFAULT_EINSUM_CACHE_CAPACITY|DEFAULT_.*CACHE" tenferro-einsum/src` | pending |
| Parsed-subscript compile cache | `GraphCompilerExtensionCaches::cached_subscripts` | Extension compile cache through generic `ExtensionCacheStore` | `rg -n "cached_subscripts|ParsedEinsum" tenferro-einsum/src tenferro/src` | pending |
| Static contraction-tree compile cache | `GraphCompilerExtensionCaches::cached_static_einsum_tree` | Extension compile cache through generic `ExtensionCacheStore` | `rg -n "cached_static_einsum_tree|static.*einsum" tenferro-einsum/src tenferro/src` | pending |
| Runtime contraction-plan cache | `GraphExecutorExtensionCaches::runtime_einsum_plans` | Extension runtime cache through generic `ExtensionCacheStore` | `rg -n "runtime_einsum_plans|runtime.*plan" tenferro-einsum/src tenferro/src` | pending |
| Cache stats retained-byte helpers | `einsum_subscripts_retained_bytes`, `einsum_plan_cache_stats`, `einsum_parse_cache_stats` | `tenferro-einsum` cache module | `rg -n "retained_bytes|cache_stats" tenferro-einsum/src` | pending |
| Extension payload `EinsumExtensionOp` | `tenferro/src/einsum_extension.rs` | `tenferro-einsum/src/extension.rs` | `rg -n "struct EinsumExtensionOp|impl ExtensionOp for EinsumExtensionOp" tenferro-einsum/src` | pending |
| Payload hashing/equality/clone | `impl ExtensionOp for EinsumExtensionOp` | Preserve in `tenferro-einsum` with structural hash; avoid pointer identity for cached tree | `rg -n "payload_hash|payload_eq|clone_arc" tenferro-einsum/src` | pending |
| Shape and dtype inference | `EinsumExtensionOp::infer_output_meta` | Preserve in `tenferro-einsum`; tests move with symbolic cases | `rg -n "infer_output_meta" tenferro-einsum/src && cargo test -p tenferro-einsum symbolic` | pending |
| Eager compatibility fallback | `EinsumExtensionOp::eager_execute` | Keep only as context-free compatibility fallback; `EagerRuntime` execution routes through `ExtensionExecutor` | `rg -n "exec_op_on_tensors_with_extension_executor|register_extension" tenferro/src tenferro-einsum/src` | done |
| Context-aware execution | `execute_einsum_extension`, `execute_einsum_extension_op`, `execute_einsum_tree` | `tenferro-einsum` runtime implementing `ExtensionRuntime<B>` | `rg -n "impl .*ExtensionRuntime|execute_einsum" tenferro-einsum/src` | done |
| Runtime builder execution bridge | `execute_einsum_tree` using `build_einsum_fragment` and exec IR | Reuse in `tenferro-einsum`; expose any missing generic runtime helpers from `tenferro` | `rg -n "build_einsum_fragment|eval_exec" tenferro-einsum/src tenferro/src` | pending |

## Generic Runtime Work Remaining In `tenferro`

| Source item | Current path | Target | Verification | Status |
|-------------|--------------|--------|--------------|--------|
| Primal extension carrier | `tenferro-internal-ops/src/ext_op.rs` | Keep in `tenferro-internal-ops`; add schema version and remove long-term `eager_execute` dependency | `cargo test -p tenferro-internal-ops ext_op` | pending |
| AD split for extension rules | `tenferro-internal-ops/src/ext_op.rs` | Split primal `ext_op` from `autodiff`-gated `ext_ad` | `cargo check -p tenferro-internal-ops --no-default-features` | pending |
| Backend-typed executor registry | new generic runtime surface | `tenferro` or `tenferro-internal-ops` depending on dependency needs; no einsum names | `rg -n "ExtensionRegistry|ExtensionExecutor" tenferro tenferro-internal-ops` | pending |
| Runtime execution context | new generic runtime surface | `tenferro`, with `BackendRuntimeState<B>` external to backend `B` | `rg -n "ExtensionExecutionContext|BackendRuntimeState" tenferro` | pending |
| Generic cache store | current `tenferro/src/graph/cache.rs` plus experiment cache API | `tenferro` generic cache module; no hard-coded einsum fields | `rg -n "ExtensionCacheKey|ExtensionCacheStore|ExtensionCacheSelector" tenferro/src` | pending |
| Graph compiler cache stats | `GraphCompilerCacheStats` currently has `static_einsum_plans`, `einsum_parse` | Replace hard-coded fields with generic extension cache stats | `! rg -n "static_einsum_plans|einsum_parse" tenferro/src` | pending |
| Graph executor cache stats | `GraphExecutorCacheStats` currently has `runtime_einsum_plans` | Replace hard-coded field with generic extension cache stats | `! rg -n "runtime_einsum_plans" tenferro/src` | pending |
| Compiler cache key extension fingerprint | `tenferro/src/graph/cache.rs` | Keep generic extension payload fingerprinting | `cargo test -p tenferro graph_compile` | pending |
| `extension::apply` traced helper | `tenferro/src/extension.rs` | Keep generic application API; no einsum knowledge | `cargo test -p tenferro extension_op` | pending |
| `extension::apply_eager` | `tenferro/src/extension.rs` | Route through `EagerRuntime`'s registry/cache context | `rg -n "ctx\\.exec_outputs" tenferro/src/extension.rs` | done |
| Multi-output extension graph support | `shape_infer`, `compiler/mod.rs`, `exec.rs`, `segment.rs` | Make N-output and mixed-dtype extension execution generic | `cargo test -p tenferro extension_op shape_inference exec_dispatch` | pending |

## Remove From `tenferro-internal-ops`

| Source item | Current path | Target | Verification | Status |
|-------------|--------------|--------|--------------|--------|
| Built-in `StdTensorOp::NaryEinsum` variant | `tenferro-internal-ops/src/std_tensor_op.rs` | Remove; use `StdTensorOp::Extension(EinsumExtensionOp)` from `tenferro-einsum` | `! rg -n "NaryEinsum" tenferro-internal-ops/src` | pending |
| N-ary einsum AD registration | `tenferro-internal-ops/src/ad/mod.rs`, `ad/contraction.rs` | Move/replace with `tenferro-einsum` extension AD rule under `autodiff` | `! rg -n "einsum|NaryEinsum" tenferro-internal-ops/src/ad` | pending |
| Einsum AD support manifest | `tenferro-internal-ops/src/ad/support.rs` | Remove from core support manifest; extension crate owns support reporting | `! rg -n "einsum|NaryEinsum" tenferro-internal-ops/src/ad/support.rs` | pending |
| Std op tests for `NaryEinsum` | `tenferro-internal-ops/src/tests/std_tensor_op_tests.rs` | Move relevant behavior to `tenferro-einsum` tests or extension tests | `! rg -n "NaryEinsum|einsum" tenferro-internal-ops/src/tests` | pending |

## AD Migration

| Source item | Current path | Target | Verification | Status |
|-------------|--------------|--------|--------------|--------|
| Einsum extension AD rule type | `EinsumExtensionAdRule` in `tenferro/src/einsum_extension.rs` | `tenferro-einsum/src/ad.rs`, gated by `autodiff` | `rg -n "EinsumExtensionAdRule|ExtensionAdRule" tenferro-einsum/src` | pending |
| Rule registration | `ensure_einsum_extension_rule_registered` | `tenferro-einsum` registration helper, gated by `autodiff` | `rg -n "ensure_.*rule_registered|register_extension" tenferro-einsum/src` | pending |
| Linearize rule | `linearize_einsum_extension` | `tenferro-einsum/src/ad.rs`; preserve output-shape hints | `rg -n "linearize_einsum_extension|linearize" tenferro-einsum/src` | pending |
| Transpose rule | `transpose_einsum_extension` | `tenferro-einsum/src/ad.rs`; preserve repeated/new label behavior | `rg -n "transpose_einsum_extension|transpose" tenferro-einsum/src` | pending |
| AD tests for traced einsum | `tenferro/tests/einsum_ad.rs` | `tenferro-einsum/tests/ad.rs`, gated by `autodiff` | `cargo test -p tenferro-einsum --features autodiff --test ad` | pending |
| No-AD build boundary | `tenferro`, `tenferro-internal-ops`, `tenferro-einsum` Cargo features | Gate `tidu`, `chainrules-core`, `chainrules` behind `autodiff` | `cargo check --no-default-features && cargo tree --no-default-features -e normal -p tenferro` | pending |

## Runtime Wiring To Preserve Or Generalize

| Source item | Current path | Target | Verification | Status |
|-------------|--------------|--------|--------------|--------|
| Compiler symbolic/static path | `tenferro/src/einsum.rs`, `GraphCompiler::cached_*` | `tenferro-einsum` uses generic compiler extension cache | `cargo test -p tenferro-einsum symbolic` | pending |
| Compiled extension dispatch special case | `tenferro/src/exec.rs` downcasts `EinsumExtensionOp` | Replace with generic `ExtensionExecutor<B>` dispatch; no einsum downcast in `tenferro` | `! rg -n "EinsumExtensionOp|execute_einsum" tenferro/src/exec.rs tenferro/src/eager_exec.rs` | pending |
| Eager extension dispatch special case | `tenferro/src/eager_exec.rs` downcasts `EinsumExtensionOp` | Replace with generic context-aware extension execution | `! rg -n "EinsumExtensionOp|execute_einsum" tenferro/src/eager_exec.rs` | pending |
| Segment helper cache threading | `tenferro/src/segment.rs` uses `EinsumPlanCache` | Replace with generic extension cache/context threading | `! rg -n "EinsumPlanCache|einsum" tenferro/src/segment.rs` | pending |
| Compiler lowering from std op to exec op | `tenferro/src/compiler/mod.rs` | Keep generic `StdTensorOp::Extension` lowering | `cargo test -p tenferro compiler_wiring` | pending |
| Shape inference for extensions | `tenferro/src/shape_infer.rs` | Keep generic `infer_output_meta`; ensure multi-output metadata is used | `cargo test -p tenferro shape_inference` | pending |
| `TracedTensor` symbolic shape comments and behavior | `tenferro/src/traced.rs` | Remove einsum-specific comments from `tenferro`; keep generic symbolic extension behavior | `! rg -n "einsum" tenferro/src/traced.rs` | pending |

## Tests To Move Or Rewrite

| Source test | Current path | Target | Verification | Status |
|-------------|--------------|--------|--------------|--------|
| Graph einsum static/runtime cache tests | `tenferro/tests/graph_einsum.rs` | `tenferro-einsum/tests/graph.rs` or `tests/cache.rs` | `cargo test -p tenferro-einsum --test graph` | pending |
| Extension cache management tests | `tenferro/tests/einsum_extension_cache.rs`, `tenferro/tests/cache_management.rs` | Split generic cache tests in `tenferro`, einsum cache behavior in `tenferro-einsum` | `cargo test -p tenferro cache_management && cargo test -p tenferro-einsum cache` | pending |
| Symbolic einsum tests | `tenferro/tests/einsum_extension_symbolic.rs`, deleted `nary_einsum_symbolic.rs` | `tenferro-einsum/tests/symbolic.rs` | `cargo test -p tenferro-einsum symbolic` | pending |
| Tensor eager facade tests | `tenferro/tests/tensor_einsum.rs`, `eager_tensor_einsum.rs` | Move to `tenferro-einsum/tests/eager_tensor.rs` or delete if facade removed | `cargo test -p tenferro-einsum eager_tensor` | pending |
| Legacy einsum tests | `tenferro/tests/einsum.rs` | Move remaining API behavior to `tenferro-einsum/tests/traced.rs` | `cargo test -p tenferro-einsum traced` | pending |
| CPU backend einsum reuse tests | `tenferro/tests/cpu_backend.rs` | Keep only generic backend/buffer-pool tests in `tenferro`; move einsum-specific cases | `! rg -n "einsum" tenferro/tests/cpu_backend.rs` | pending |
| Compiler/executor cache capacity tests | `tenferro/tests/graph_compile.rs`, `graph_executor.rs` | Replace einsum-specific capacity assertions with generic extension cache tests | `! rg -n "einsum_cache|runtime_einsum" tenferro/tests/graph_compile.rs tenferro/tests/graph_executor.rs` | pending |
| Extension op smoke tests | `tenferro/tests/extension_op.rs`, `tenferro-internal-ops/src/tests/ext_op_tests.rs` | Keep and update for new executor registry/context | `cargo test -p tenferro extension_op && cargo test -p tenferro-internal-ops ext_op` | pending |

## Cleanup Acceptance Checks

Run these before declaring the migration complete:

```sh
# tenferro must not own or export einsum implementation.
! rg -n "pub .*einsum|EinsumSubscripts|EinsumOptimize|EinsumExtensionOp|execute_einsum" tenferro/src

# tenferro must not normally depend on tenferro-einsum.
! rg -n 'tenferro-einsum' tenferro/Cargo.toml

# tenferro-internal-ops must not contain the old built-in op or AD rule.
! rg -n "NaryEinsum|einsum" tenferro-internal-ops/src/ad tenferro-internal-ops/src/std_tensor_op.rs

# Generic runtime may mention extension caches, but not hard-coded einsum cache fields.
! rg -n "static_einsum_plans|einsum_parse|runtime_einsum_plans|einsum_cache" tenferro/src

# The implementation must live in tenferro-einsum.
rg -n "EinsumExtensionOp|ExtensionExecutor|tenferro\\.einsum|einsum" tenferro-einsum/src tenferro-einsum/tests
```

Run these build checks:

```sh
cargo check --workspace
cargo test -p tenferro-einsum
cargo test -p tenferro
cargo test -p tenferro-internal-ops
cargo check --no-default-features
cargo check --no-default-features --features cpu-faer
cargo tree --no-default-features -e normal -p tenferro
```

The `cargo tree` output must not contain `tidu`, `chainrules-core`, or
`chainrules` for primal-only builds.

## Step-By-Step Review Gates

1. Inventory gate: every source item above is marked `ported` or `not-needed`
   with a reason.
2. Runtime gate: `tenferro` has generic extension registry/cache APIs and no
   einsum downcasts.
3. Crate gate: `tenferro-einsum` owns traced, eager, AD, extension payload,
   execution, and cache behavior.
4. Cache gate: parse, static-plan, and runtime-plan caches exist in explicit
   compiler/runtime owners; context-free direct eager caches are removed.
5. AD gate: einsum AD tests pass with `autodiff`; primal-only builds exclude AD
   dependencies.
6. Cleanup gate: all acceptance commands above pass.
