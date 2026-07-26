# AD Contract

**Date:** 2026-07-26
**Parent:** [`../index.md`](../index.md)
**Related:** [`primitive-catalog.md`](primitive-catalog.md),
[`../architecture/primitive-ad.md`](../architecture/primitive-ad.md),
[`../architecture/semantic-ad.md`](../architecture/semantic-ad.md)

---

## Purpose

This document is the normative specification for tenferro AD rule emission.
Concrete primitive rules must satisfy this contract whether they are core
`StdTensorOp` rules or semantic extension rules owned by operation-family
crates.

For the AD pipeline architecture, see
[`../architecture/ad-pipeline.md`](../architecture/ad-pipeline.md). For the
rule-structure rationale, see
[`../architecture/primitive-ad.md`](../architecture/primitive-ad.md).

## Core Primitive Rule Contract

Core rules are defined in `tenferro-internal-ops/src/ad/` and registered by
`PrimitiveOpKind`. The canonical internal rule trait is:

```rust
pub(crate) trait PrimitiveAdRule: Send + Sync {
    fn kind(&self) -> PrimitiveOpKind;

    fn linearize(
        &self,
        op: &StdTensorOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        primal_in: &[ValueKey<StdTensorOp>],
        primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;

    fn transpose_rule(
        &self,
        op: &StdTensorOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[TransposeInputRef<'_>],
        mode: &OperationRole,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>;
}
```

`ADRuleResult<T>` is tenferro's rule-emission result type. Rule failures must
be reported through typed `ADRuleError` values instead of panicking or silently
dropping derivative flow.

`PrimitiveRuleBuilder` is the only mutation surface available to core rules.
Rules emit graph operations; they do not execute tensors, read runtime caches,
or inspect backend state.

## Transpose Inputs

Transpose rules receive typed `TransposeInputRef` values derived from
`PrimitiveTransposeInput<StdTensorOp>` so they can distinguish fixed operands
from active linear flow.

Residual or metadata inputs are independent of the differentiated tangent flow
and may be used as ordinary rule operands. Linear inputs belong to tangent
flow; their primal counterpart may be used only for metadata, runtime shape
sources, or fixed coefficients that are independent of tangent flow. A linear
input without a valid fixed/primal source must be rejected at that use site
rather than smuggled into the residual graph.

## Rules

1. **Closure**: `linearize` and `transpose_rule` must add only operations that
   remain in the supported tenferro semantic vocabulary. tenferro owns this
   closure requirement for core rules; operation-family crates own it for
   extension rules.

2. **Cotangent accumulation**: fan-out cotangents are accumulated through
   explicit `Add` graph nodes. `Add` must remain AD-supported, and its
   transpose behavior must pass cotangents to both active inputs.

3. **Linear ops**: an op whose `linearize` is the identity tangent map only
   needs a `transpose_rule`. Examples include metadata-only reshape,
   transpose, and broadcast paths when their payloads preserve linear flow.

4. **Primal reuse**: rule emission may reference primal values as fixed
   operands through the graph/semantic builder when the emitted derivative
   operation is independent of tangent flow. Shared primal computations should
   be reused rather than duplicated.

5. **Shape-source discipline**: AD graph-emission rules must distinguish rank,
   exact extents, conservative extents, and runtime shape sources. Exact-shape
   requirements are appropriate only when constructing a concrete payload that
   cannot represent runtime dimensions.

6. **Extension AD boundary**: `StdTensorOp::Extension` may participate in AD
   only when its operation family registers semantic extension AD rules in the
   active `SemanticExtensionRuleSet`. Missing extension rules must report
   unsupported AD.

## Mode Interpreters And Cacheability

The AD contract has one derivative rule model. `linearize` is the
primitive-local derivative producer. `transpose_rule` applies only to linear
flow that was already produced by linearization; it is not a separate primal
reverse-mode derivative rule. Eager and traced execution choose different
interpreters for the same rule model:

- **Traced transforms** operate on frozen semantic programs, then compile and
  execute the materialized derivative program.
- **Eager transforms** record eager operations as small graphs/programs and
  reuse the same context-owned transform machinery.
- **Stateful eager reverse mode** (`backward()` and `backward_with(seed)`)
  accumulates reachable tracked leaves into gradient slots.
- **Functional eager transforms** (`grad`, `vjp`, `jvp`) return ordinary eager
  tensors and do not mutate gradient slots.

Rule emission must be deterministic for a fixed primitive payload, input and
output metadata, active masks, requested output slots, AD context, and
extension rule set. Rules must not read hidden runtime state or environment
state to decide graph structure.

Direct primal VJP rules are optional escape hatches for cases where the generic
`linearize -> transpose_rule -> optimize` path is incomplete or too slow; they
are not the default obligation for making a primitive reverse-differentiable.

Symbolic zero propagation should remain symbolic until a rule must pass a real
zero value to another primitive. At that forced-instantiation boundary,
tenferro rules carry dtype, rank, and an anchor value as `SymbolicZero`, then
instantiate it as a dtype-aware scalar zero plus shape-restoring broadcast when
needed. Do not synthesize zeros through analytic operations or tensor buffers.

The same rule applies to AD-emitted scalar constants, one-like tensors, and
identity matrices. Rule implementations must use semantic graph-emission
helpers such as `tenferro_ops::ad::support::{constant_scalar, zero_like,
one_like, identity_matrix}` rather than analytic identities.

`AdContext` is the explicit owner for shared AD transform memoization. It owns
the extension AD rules and a bounded AD transform cache used by context-driven
traced transforms. Eager runtimes created with
`EagerRuntime::with_*_and_ad_context` share that same cache handle; eager
runtimes created directly own a private cache. Direct `TracedTensorAdExt`
methods remain stateless.

The AD transform cache stores graph/program artifacts only: eager recorded
graph transforms, traced JVP transformed programs, and traced VJP residual or
transposed program artifacts. It must not keep tensor buffers, backend
allocations, concrete execution outputs, or dead linear sweeps alive after the
optimized derivative program has been built.

The default retention policy is bounded by both entry count and logical
retained bytes. Owners expose limits, stats, and clear APIs through `AdContext`
and `EagerRuntime`; retained-byte stats are logical estimates and do not report
process RSS.

Cache keys must be deterministic, structural, and metadata-only. Eager keys
cover the recorded graph fingerprint and requested output slots. Traced keys
cover semantic program fingerprints, input metadata, active inputs, active
outputs, and aliases. Rules whose emitted program depends on additional
metadata must make that metadata part of the cache key or bypass caching for
the affected transform.

The AD graph optimizer remains per-invocation apart from storing its final
program inside an owner-scoped transform-cache entry. Reachability, rewrite
facts, and multi-output live masks are scratch data. Partial output pruning is
legal only when the operation family explicitly opts in, currently through
`ExtensionOp::prune_outputs`.

## Complex AD Convention

tenferro follows the JAX-style complex AD convention.

Forward mode treats complex primitives as real-linear maps. For a holomorphic
elementwise map `f`, the JVP multiplies the tangent by the local derivative
coefficient `f'(z)` without conjugating that coefficient.

Reverse mode transposes real-linear maps under the real inner product
`<a, b> = Re(conj(a) * b)`. Therefore the VJP for a holomorphic elementwise map
uses the conjugated local derivative coefficient:

```text
primal: y = f(z)
JVP:    dy = f'(z) * dz
VJP:    dz_bar = y_bar * conj(f'(z))
```

The same rule applies to fixed derivative coefficients emitted by composite
transpose rules. For example, if a binary holomorphic op emits a coefficient
`c(x, y)` for one input in forward linearization, its transpose rule must
multiply the output cotangent by `conj(c(x, y))` when the corresponding
real-linear map is complex-valued. Do not conjugate those coefficients in JVP
rules.

This convention is the normative source for tenferro complex VJP behavior.
Oracle comparisons and finite-difference tests must be interpreted under this
real-inner-product convention.

### Complex `Abs` and `Sign`

tenferro follows JAX's real-output convention for complex absolute value:

```text
primal: C32 abs -> F32
primal: C64 abs -> F64
JVP:    d abs(z) = Re(conj(sign(z)) * dz)
VJP:    z_bar = abs_bar * sign(z)
```

The `abs` cotangent is real because the primal output is real. The VJP maps
that real cotangent back into the complex input tangent space by multiplying by
`sign(z)`.

`Sign` has zero AD for both real and complex inputs. Treat this as the
operation contract, not as a holomorphic derivative.

## Boundary And Nondifferentiable Elementwise Rules

When a primitive has a nondifferentiable boundary and JAX has a clear rule,
tenferro follows JAX unless a later design document explicitly says otherwise.

`Convert` follows JAX's `convert_element_type` AD convention. Casts between
floating-point and complex dtypes are differentiated by casting the tangent or
cotangent to the corresponding tangent dtype, including lossy casts such as
`F64 -> F32`. Casts whose input or output dtype is `I32`, `I64`, or `Bool` are
inactive for AD. JAX represents those integer/bool tangent spaces with
`float0`; tenferro has no public `float0` dtype, so traced AD represents the
same contract as `None` from the `*_optional` AD APIs.

`Maximum` and `Minimum` split tangent and cotangent contributions equally among
inputs that are equal to the primal output. For a two-input tie, each active
side receives half of the tangent/cotangent. Away from ties, the winning side
receives the full contribution and the losing side receives zero.

`Clamp(input, lower, upper)` uses strict JAX boundary masks:

```text
input tangent/cotangent active iff input > lower && input < upper
lower tangent/cotangent active iff lower > input && lower < upper
upper tangent/cotangent active iff upper < input
```

At exact lower or upper boundaries, the corresponding derivative contribution
is zero. Do not review clamp AD against inclusive `<=` / `>=` masks.

## Indexing Bounds Contract

Indexing AD follows the JAX/StableHLO-style `promise_in_bounds` contract:
gradients are guaranteed only for in-bounds starts and indices. Runtime primal
behavior may clamp dynamic slices or drop out-of-range scatter windows, but
that boundary behavior is not an AD correctness promise.

Reviews and oracle tests for `Gather`, `Scatter`, `DynamicSlice`,
`DynamicUpdateSlice`, and dynamic-slice-size gather AD must use in-bounds
indices unless a future design changes the contract. Out-of-bounds primal
compatibility tests are valid, but they must not be interpreted as finite
difference requirements for AD at those discontinuous boundaries.

## Owned By This Document

- Core `PrimitiveAdRule` rule-emission contract
- Extension semantic AD boundary
- Closure rule
- Cotangent accumulation rule
- Linear op rule
- Primal reuse rule
- Eager/traced interpreter split
- Rule-emission cacheability contract
- Complex AD convention
- Convert dtype-boundary AD convention
- Elementwise nondifferentiable boundary AD convention
- Indexing AD bounds contract

Other documents link here for the AD contract; they do not re-state these
definitions.
