# AD Architecture

**Repos:** computegraph-rs, tenferro-rs
**Parent:** `../index.md`
**Related:** `computegraph.md`, `primitive-ad.md`, `semantic-ad.md`,
`../spec/ad-contract.md`, `../spec/backend-contract.md`,
`../spec/primitive-catalog.md`

---

## Purpose

This document describes the current end-to-end AD pipeline after U8. The active
implementation is tenferro-owned: `tenferro-ad` transforms semantic programs,
`tenferro-internal-ops` owns core primitive rules, and operation-family crates
register semantic extension rules.

The design model remains linearize-then-transpose:

- `linearize` is the derivative-producing step.
- `transpose_rule` reverses already-linear flow.
- compilation and execution happen after the derivative program has been built
  and validated.

## Current Traced Pipeline

Traced tensor workflows build a backend-neutral program first:

```text
TracedTensor operations
  -> TraceContext
  -> GraphCompiler
  -> CompiledGraph / FrozenProgram
```

Functional AD transforms then operate on that frozen semantic program:

```text
JVP:
  FrozenProgram
    -> semantic_jvp_with_cache
    -> SemanticAdProgram
    -> GraphCompiler / Runtime
    -> Tensor outputs

VJP:
  FrozenProgram
    -> semantic_vjp_with_cache
    -> SemanticAdProgram
    -> GraphCompiler / Runtime
    -> cotangent Tensor outputs
```

The semantic transform preserves validation and source metadata. It does not
execute tensor buffers and does not choose backend kernels. Backend execution
starts only when the resulting program is compiled and handed to `Runtime`.

## Current Eager Pipeline

Eager operations execute immediately, but eager AD still records enough
semantic information to reuse the same rule model:

```text
EagerTensor operation
  -> execute concrete operation
  -> record operation metadata for AD
  -> use AdContext / EagerRuntime transform cache when a functional AD
     transform or backward pass needs derivative programs
```

Stateful eager reverse mode (`backward` and `backward_with`) additionally
accumulates gradients into tracked leaf slots. Functional eager transforms
(`grad`, `vjp`, and `jvp`) return tensors and do not mutate gradient slots.

## Rule Dispatch

Core primitive rules are registered by `PrimitiveOpKind` in
`tenferro-internal-ops/src/ad/registry.rs`. The core rule entry points are:

```text
PrimitiveAdRule::linearize
PrimitiveAdRule::transpose_rule
```

Extension rules are registered by operation-family crates in a
`SemanticExtensionRuleSet`. `tenferro-ad` owns lookup and dispatch for semantic
extension rules. Missing rules must report unsupported AD; they must not drop
tangent or cotangent flow.

Direct primal VJP rules are allowed only as explicit escape hatches. The
default route for reverse-mode support is:

```text
linearize -> transpose_rule -> optimize -> compile -> execute
```

## Shape And Output Activity

Before a transform emits derivative operations, tenferro computes which primal
outputs and inputs are active. That information is carried through
`ShapeGuardContext` and semantic metadata so multi-output rules can avoid
building unused derivative branches.

For example, an SVD whose caller only consumes singular values should be able
to emit the singular-value derivative path without also building the vector
F-matrix chain. `materialize_merge`, semantic validation, and compiler passes
still deduplicate later, but rule emission should avoid known-dead work early.

Rules must distinguish:

- rank only;
- exact static extents;
- conservative or unknown extents;
- runtime shape sources.

Do not require exact extents when rank or runtime shape references are enough.

## Symbolic Zero And Constants

`None` in a tangent or cotangent slot means symbolic zero flow. Rules keep that
state symbolic until another primitive requires a real tensor value. At that
boundary, rules instantiate zeros through dtype-aware semantic helpers.

The same discipline applies to constants, one-like tensors, and identity
matrices. AD rules should use helpers such as `constant_scalar`, `zero_like`,
`one_like`, and `identity_matrix`, not analytic operations such as `exp`,
`log`, or `sin`, to manufacture constants.

## Cache Ownership

AD transform caching is owner-scoped:

- `AdContext` owns extension rules and a bounded transform cache.
- `EagerRuntime` instances created with an `AdContext` share that cache handle.
- directly constructed eager runtimes own private caches.
- direct traced helper methods remain stateless unless the caller supplies an
  `AdContext`.

Cached AD entries store graph/program artifacts only. They must not retain
tensor buffers, backend allocations, or concrete execution outputs. Cache keys
must be deterministic and metadata-only: semantic fingerprints, input metadata,
active masks, requested output slots, and alias information.

## Scalar Example

For `f(x) = exp(a * x)`, the traced program first records the primal:

```text
p0 = Input(x)
p1 = Input(a)
p2 = Mul(p0, p1)
p3 = Exp(p2)
```

JVP with respect to `x` emits linear flow equivalent to:

```text
t0 = Input(dx)
t1 = Mul(a, t0)
t2 = Mul(exp(a*x), t1)
```

VJP transposes that linear flow:

```text
c0 = Input(dy)
c1 = Mul(exp(a*x), c0)
c2 = Mul(a, c1)
```

The generated program is then compiled and executed forward like any other
semantic program.

## Reduction Example

For `f(x) = sum(x)`, linearization emits the same reduction on tangent flow:

```text
dy = sum(dx)
```

Reverse mode transposes that linear map by broadcasting the scalar cotangent to
the input shape:

```text
dx_bar = broadcast_in_dim(dy_bar, shape(x))
```

This example is why transpose rules must keep shape metadata available without
forcing tensor execution during rule emission.

## Complex Example

For a holomorphic elementwise function `y = f(z)`, forward mode treats the map
as real-linear:

```text
dy = f'(z) * dz
```

Reverse mode transposes under the real inner product:

```text
dz_bar = y_bar * conj(f'(z))
```

The normative complex convention lives in
[`../spec/ad-contract.md`](../spec/ad-contract.md).

## Golden Invariants

AD implementation and review should preserve these invariants:

- derivative programs are validation-preserving `Program -> Program`
  transforms;
- rule emission is deterministic for payload, metadata, active masks, context,
  and extension rule set;
- active-output pruning happens before large multi-output derivative branches
  are emitted;
- explicit cotangent accumulation remains visible as graph/program operations;
- cache entries are bounded, clearable, introspectable, and owner-scoped;
- numerical oracle or finite-difference coverage is the correctness gate for
  supported AD rules.

Historical design notes and migration worklogs may still mention older engine
names. Active architecture and spec pages should use the current tenferro-owned
interfaces documented here.
