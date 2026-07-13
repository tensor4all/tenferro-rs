# Extension shape equality constraints

## Session summary

Issue [#1370](https://github.com/tensor4all/tenferro-rs/issues/1370)
adds declarative equality relations to extension shape inference. The accepted
design is
[`../superpowers/specs/2026-07-13-extension-shape-equality-constraints-design.md`](../superpowers/specs/2026-07-13-extension-shape-equality-constraints-design.md).

The implementation makes a clean break to `ExtensionShapeContext`, preserves
declared relations in graph-owned scopes, discharges them with a deliberately
small deterministic solver, and retains undecidable relations as executor
guards. The contract survives core-op fast paths, graph optimization,
checkpointing, JVP/VJP construction, AD transform-cache reuse, and compile-cache
reuse.

Two focused stage records retain the detailed RED/GREEN and review history
without duplicating it here:

- [`2026-07-13-issue-1370-graph-constraint-scopes.md`](2026-07-13-issue-1370-graph-constraint-scopes.md)
- [`2026-07-13-issue-1370-shape-guard-executor.md`](2026-07-13-issue-1370-shape-guard-executor.md)

The graph-scope record's residual AD-transfer risk described that intermediate
stage. It was subsequently resolved by the persistent transfer boundary
described below.

## Context read

- Current shared tensor4all repository, Rust, performance, documentation,
  testing, and numerical rules; `AGENTS.md`; and `REPOSITORY_RULES.md`.
- The approved design, implementation plan Task 9, the two stage work logs,
  the normative extension/backend specs, and the dynamic symbolic shape design.
- CodeGraph source and call paths for extension inference, graph metadata
  analysis, traced scope composition, compiler lowering, cache identity,
  executor preflight, checkpointing, and AD transform construction.
- The complete issue-branch commit sequence and aggregate diff from the design
  base through `75f3b7bc`, with focused commit diffs for constraint adopters,
  expanded fast paths, host-boundary defense, and persistent AD transfer.

## Reference concepts considered

JAX export shape polymorphism supplied two useful concepts: interacting
dimension expressions share a symbolic scope, and concrete calls check shape
assertions derived from symbolic specifications. JAX also documents that some
nonlinear forms cannot be solved. See the official
[JAX shape polymorphism documentation](https://docs.jax.dev/en/latest/export/shape_poly.html).

PyTorch's compiler uses one per-compilation `ShapeEnv` to track symbolic sizes
and accumulated guards, then installs guards with compiled code. That separation
between symbolic reasoning and guarded reuse informed the scope/guard split,
without adopting PyTorch's full SymPy-backed solver. See the official
[dynamic-shape core concepts](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/dynamic_shapes_core_concepts.html)
and [`ShapeEnv` API](https://docs.pytorch.org/docs/main/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html).

These were design references, not compatibility targets. Tenferro retains its
existing `SymDim`/`DimExpr` vocabulary and typed Rust error ownership.

## Decisions and implementation

### One declarative extension boundary

`ExtensionOp::infer_output_meta` now receives
`&mut ExtensionShapeContext<'_>`. The context owns bounds-checked dtype/shape/
axis accessors and `require_equal`, `require_axes_equal`, and
`require_same_shape`. Every implementation migrated in one change; the former
separate-slice signature and compatibility adapter were removed.

The fundamental relation accepts expressions. An extension may record
`a == 2 * b` without requiring the callback to solve it. A call-local symbolic
namespace is substituted back to the complete original `DimExpr`, so reordered,
nested, and globally indexed input expressions do not collide.

### Graph ownership and bounded collection

Inference produces op-local relations. Graph analysis records them with all
output origins and ordered graph input keys in a private representation behind
`ShapeConstraintScope`. Metadata and local constraints are collected in one
root-local walk. Registered external metadata is read directly; an unregistered
key triggers one on-demand parent lookup using a lazily built owner index.

Traced tensors carry persistent `Arc`-backed scope chains. Empty scopes are
skipped, shared parents remain shared, and one compiler materialization walk
deduplicates chain nodes and scopes by pointer identity. This avoids cloning or
rescanning full histories for each graph operation.

Ordinary extension nodes and equivalent expanded core graphs use the same
inferred contract. In particular, the direct and expanded einsum paths attach
repeated-label equality even when optimization leaves no extension instruction.
Graph-scoped live constraints use pre-optimizer origins and survive eliminated
identity layout operations. Compiler-inferred constraints from dead extension
instructions are pruned; graph-scoped constraints are pruned only when every
origin is dead. A missing live key, slot, or axis is a typed evaluation error.

The symbolic analysis table remains separate from executable concrete metadata.
`Reshape` and `BroadcastInDim` use best-effort symbolic resolution: an
unresolved namespace expression remains symbolic there, while the concrete
execution path stays authoritative and typed.

### Small equality engine and runtime enforcement

The solver performs checked constant folding, semantics-preserving structural
normalization, deterministic commutative ordering, and union/binding for bare
axis symbols. Union closure retains a deterministic spanning set of guards, so
reasoning does not erase the runtime obligations that justified it.

The solver deliberately does not rearrange expressions, invert multiplication,
prove inequalities, or provide a general symbolic algebra system. Symbolic
`a == 2 * b` remains a guard. It is proved or disproved only after substitution
or concrete evaluation makes that possible. Rewrites such as `x * 0 -> 0` are
not applied when they could hide an invalid reference or arithmetic error.

Compiler-specialized graph descriptors usually make relations concrete.
Low-level compilation still accepts symbolic shapes and retains normalized
`ExecProgram` guards. All executor entry points validate those guards before
uploads, zero synthesis, backend session/workspace creation, or extension
dispatch.

Guard relation and normalized operands participate in program and compile-cache
identity. Provenance does not. Cache hits reuse executable structure but replace
the cached guards with the current graph's guard vector, preserving current
family/instruction diagnostics.

### Checkpoint and AD persistence

Checkpointing preserves the existing constraint chain while replacing the
materialized leaf and adding its metadata scopes. The runtime/AD boundary uses
opaque `ConstraintScopeTransfer` values rather than exposing constraint
encoding. Transfers are constant-time clones of persistent histories; JVP and
VJP layer newly analyzed linear, residual, and transposed scopes over primal,
tangent, and cotangent histories.

Both first construction and AD transform-cache hits retain the contract. Tests
compare cold and hot JVP/VJP failures field-for-field, including provenance, and
also cover direct primal-VJP construction.

### Real adopters and host defense

- Ordinary einsum declares equality for repeated labels on direct and expanded
  paths.
- Sparse matmul declares payload NNZ equalities plus primal/tangent/cotangent
  exact-shape requirements.
- Tropical einsum declares repeated-label equality plus JVP/VJP exact-shape
  requirements.

Sparse and tropical host references separately validate concrete count, dtype,
rank, and exact shape before indexing. Compiled graph guards are defense in
depth, not a substitute for direct host-boundary validation.

## Alternatives rejected or deferred

- Keeping the old inference signature through an adapter would split extension
  authors between two contracts and was rejected in favor of a clean break.
- Embedding constraints in extension payload identity would mix graph facts
  with semantic operation state and was rejected in favor of graph-owned
  scopes.
- A general algebra dependency, inverse solving, inequalities, divisibility,
  and broadcasting relations are deferred. The current public declaration
  boundary can accommodate stronger internal reasoning later.
- Dropping unresolved relations because graph compilation currently specializes
  shapes was rejected. Low-level symbolic compilation and future polymorphic
  compilation require executable guards now.
- Flattening scope histories at every AD or traced boundary was replaced by an
  opaque persistent chain to keep composition work bounded.

## Verification

Focused stages used strict RED/GREEN tests for the context, solver, compiler,
cache, executor, graph scopes, real extension families, and JVP/VJP cold/hot
paths. The completed implementation stages also passed:

```bash
cargo fmt --all --check
cargo test --workspace --all-targets --release
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

The documentation head was then checked with:

```bash
rg -n "infer_output_meta\([^)]*input_dtypes|infer_output_meta\([^)]*input_shapes" README.md docs crates ext --glob '*.md' --glob '*.rs'
cargo fmt --all --check
cargo test --workspace --doc --release
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

The stale-signature search returned no matches; all four validation commands
passed.

The final PR gate MUST rerun the coverage pair from the final committed head;
an earlier stage result is not a substitute for Task 10's fresh coverage
artifact. The final gate also runs the repository-rules review against
`origin/main`.

This change is backend-independent and contains no GPU kernel or CUDA-specific
path changes. Verification was CPU-only; hardware-gated CUDA tests were not run.
No CUDA capability claim is made by this work.

## Remaining scope

The intentional remaining limitation is solver strength, not contract
preservation. General algebra, inequalities, and new relation families require
separate accepted designs. Runtime-scalar exact extents for operations such as
`DynamicTruncate` remain the separate deferred work described in
[`../design/dynamic-symbolic-shapes.md`](../design/dynamic-symbolic-shapes.md).
