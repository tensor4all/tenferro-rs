# AD Contract

**Date:** 2026-06-10
**Parent:** [`../index.md`](../index.md)
**Related:** [`primitive-catalog.md`](primitive-catalog.md), [`../architecture/primitive-ad.md`](../architecture/primitive-ad.md), [`../architecture/tidu.md`](../architecture/tidu.md)

---

## Purpose

This document is the normative specification for the AD trait contract that
concrete primitives must satisfy. It owns the `Primitive` trait signature
and the rules that `linearize` and `transpose_rule` must follow.

For the AD pipeline architecture (linearize, linear_transpose, higher-order AD),
see [`../architecture/ad-pipeline.md`](../architecture/ad-pipeline.md).

For the AD trait design rationale, see
[`../architecture/primitive-ad.md`](../architecture/primitive-ad.md).

---

## Primitive trait (canonical signature)

Defined in `tidu-rs/src/rules/primitive_op.rs`. Extends `GraphOperation` with
the constraint `Self::InputKey: ADKey`.

```rust
pub trait Primitive: GraphOperation
where
    Self::InputKey: ADKey,
{
    type ADContext: Default;

    /// Returns the addition operation used for cotangent accumulation.
    /// tidu's `linear_transpose` emits `Op::add()` nodes when multiple cotangents
    /// flow to the same value.
    fn add() -> Self where Self: Sized;

    /// Emit the JVP rule for this primitive.
    fn jvp_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        primal_inputs: &[ValueKey<Self>],
        primal_outputs: &[ValueKey<Self>],
        tangent_inputs: &[Option<LocalValueId>],
        ctx: &mut Self::ADContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>
    where
        Self: Sized;

    /// Emit the transpose rule for this linear primitive.
    fn transpose_rule(
        &self,
        builder: &mut impl PrimitiveBuilder<Self>,
        cotangent_outputs: &[Option<LocalValueId>],
        inputs: &[PrimitiveValue<Self>],
        role: &OperationRole,
        ctx: &mut Self::ADContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>>
    where
        Self: Sized;
}
```

`ADRuleResult<T>` is tidu's rule-emission result type. Rule failures must be
reported through it instead of panicking or silently dropping derivative flow.

## ADKey trait (canonical signature)

Defined in `tidu-rs/src/rules/ad_key.rs`. Required bound on
`Primitive::InputKey`.

```rust
pub trait ADKey: Clone + Debug + Hash + Eq + Send + Sync + 'static {
    /// Create a tangent input key derived from this key.
    /// `pass` is a unique identifier for the `linearize` call.
    fn tangent_of(&self, pass: DiffPassId) -> Self;
}
```

`DiffPassId` is `u64`.

## LinearizedGraph (canonical definition)

Defined in `tidu-rs/src/linearized_graph.rs`. Returned by
`tidu::linearize` (which internally calls `Primitive::jvp_rule`
per operation node — note that `jvp_rule` itself returns
`ADRuleResult<Vec<Option<LocalValueId>>>`, not `LinearizedGraph`; the graph is
assembled by `linearize`).

```rust
pub struct LinearizedGraph<Op: GraphOperation> {
    graph: Graph<Op>,
    tangent_inputs: Vec<(Op::InputKey, LocalValueId)>,
    tangent_outputs: Vec<Option<LocalValueId>>,
}

impl<Op: GraphOperation> LinearizedGraph<Op> {
    pub fn as_graph(&self) -> &Graph<Op>;
    pub fn tangent_inputs(&self) -> &[(Op::InputKey, LocalValueId)];
    pub fn tangent_outputs(&self) -> &[Option<LocalValueId>];
}
```

## Rules

1. **Closure**: `linearize` and `transpose_rule` must add only ops that
   themselves implement `Primitive`. This is the sole closure requirement.
   tenferro-rs is responsible for satisfying it.

2. **Cotangent accumulation**: when a value fans out to multiple consumers,
   tidu's `linear_transpose` accumulates cotangents via `Op::add()`. This means
   `Add` must implement `Primitive` and its linear_transpose rule must be the
   identity (cotangent passes through to both inputs).

3. **Linear ops**: an op whose `linearize` returns itself (identity tangent
   map) only needs a `transpose_rule`. Examples: `Transpose`, `Reshape`,
   `BroadcastInDim`.

4. **Primal reuse**: `linearize` may reference primal values via
   `External(ValueKey)` in the graph builder. These are resolved
   during `materialize_merge` so that shared primal computations are not
   duplicated.

5. **Extension AD boundary**: built-in AD is defined for `StdTensorOp`.
   `StdTensorOp::Extension` may participate in AD only when its operation
   family registers an extension AD rule. Missing extension rules must report
   unsupported AD; they must not silently drop or zero gradients.

## Mode Interpreters And Cacheability

The AD trait contract has one derivative rule set. `jvp_rule` is the only
primitive-local derivative producer. `transpose_rule` applies only to linear
flow that was already produced by linearization; it is not a separate primal
reverse-mode derivative rule. Eager and traced execution choose different
interpreters for the same rule set:

- **Traced transforms** operate on recorded graphs, then compile and execute a
  materialized graph.
- **Eager transforms** record each eager op as a small `RecordedGraph`.
  tidu owns the eager forward and backward graph traversal. tenferro owns
  concrete graph execution, public eager APIs, runtime state, extension rules,
  and cache storage.
- **Stateful eager reverse mode** (`backward()` and `backward_with(seed)`)
  accumulates reachable tracked leaves into gradient slots.
- **Functional eager transforms** (`grad`, `vjp`, `jvp`) return ordinary eager
  tensors and do not mutate gradient slots. When their derivative computation
  depends on tracked eager values, the returned tensor remains traceable, so
  transforms such as `jvp(grad(f))` can be expressed by composing eager calls.

Rule emission must be deterministic for a fixed primitive payload, input and
output metadata, active mask, requested output slots, AD context, and extension
rule set. Rules must not read hidden runtime state or environment state to
decide graph structure. That purity lets runtime owners safely memoize
recorded-graph linearization through tidu's eager executor hook.

tenferro's eager runtime owns a bounded Tier-1 AD transform cache keyed by the
recorded graph structural fingerprint and requested output slots. This cache
reuses linearization for repeated transforms on the same eager tape node. It is
not a whole-program backward cache and it does not generalize across
structurally similar but distinct tapes.

## Complex AD convention

tenferro follows the tidu/JAX-style complex AD convention.

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

## Owned by this document

- `Primitive` trait signature
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

Other documents link here for the AD contract; they do not re-state
these definitions.
