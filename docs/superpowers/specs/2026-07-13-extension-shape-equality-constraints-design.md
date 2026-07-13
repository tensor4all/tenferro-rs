# Extension Shape Equality Constraints

**Issue:** #1370
**Date:** 2026-07-13
**Status:** Approved design

## Summary

Extension operations need to state shape relations between independent inputs,
not only compute output metadata. The first supported relation is equality of
symbolic dimension expressions. This covers exact-shape contracts such as a
tangent matching its primal and also represents relations such as
`a == 2 * b` without committing the public API to a particular solver.

The implementation will make a clean break in `ExtensionOp::infer_output_meta`.
Every extension receives an `ExtensionShapeContext`, reads input metadata
through it, and records constraints through builder methods. Constraints live
in graph-owned scopes, survive graph composition and AD transformations, and
are discharged by the compiler when concrete shapes are available. Any
constraint that cannot yet be decided becomes a runtime guard in the compiled
program.

Phase one deliberately implements a small equality engine rather than a
general symbolic algebra solver. The public API and internal pipeline leave
room for stronger reasoning later.

## Goals

1. Allow an extension to require equality between axes or arbitrary `SymDim`
   expressions from its inputs.
2. Accept equal concrete dimensions from independent placeholders.
3. Reject contradictory concrete dimensions before backend execution.
4. Preserve unresolved constraints through graph composition, compilation,
   cache reuse, checkpoint/replay paths, and AD graph transformations.
5. Report violations and evaluation failures with typed, actionable errors.
6. Replace the existing extension metadata inference API in one migration,
   without a compatibility adapter.

## Non-Goals

Phase one does not add:

- inequalities or range reasoning;
- divisibility, modular, or broadcasting constraints;
- a general symbolic algebra solver;
- inverse algebraic solving such as deriving `b` from `a == 2 * b`;
- a new solver dependency, feature flag, or backend-specific constraint path.

## Public Extension API

`ExtensionOp::infer_output_meta` changes from separate dtype and shape slices
to a mutable context:

```rust
fn infer_output_meta(
    &self,
    ctx: &mut ExtensionShapeContext<'_>,
) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>>;
```

The context is the public surface for both input inspection and shape
requirements. Its initial builder methods are:

```rust
impl ExtensionShapeContext<'_> {
    pub fn input_dtype(&self, input: usize) -> Result<DType>;
    pub fn input_shape(&self, input: usize) -> Result<&[SymDim]>;
    pub fn input_axis(&self, input: usize, axis: usize) -> Result<SymDim>;

    pub fn require_equal(
        &mut self,
        lhs: SymDim,
        rhs: SymDim,
    ) -> Result<()>;

    pub fn require_axes_equal(
        &mut self,
        lhs: (usize, usize),
        rhs: (usize, usize),
    ) -> Result<()>;

    pub fn require_same_shape(
        &mut self,
        lhs_input: usize,
        rhs_input: usize,
    ) -> Result<()>;
}
```

`require_axes_equal` and `require_same_shape` are convenience methods over
`require_equal`. `require_same_shape` first checks rank equality, then records
one equality per axis. Input and axis bounds errors use the existing extension
metadata inference error path.

The fundamental method accepts expressions rather than only axis pairs. For
example, an extension can express:

```rust
let a = ctx.input_axis(0, 0)?;
let b = ctx.input_axis(1, 0)?;
ctx.require_equal(a, 2 * b)?;
```

Recording a valid equality does not require the relation to be solvable at
trace time. The inference driver subsequently normalizes and classifies the
recorded relation. This separation keeps extension implementations declarative
and allows the equality engine to evolve independently.

All existing implementations of `ExtensionOp` migrate to this API in the same
change. The old signature and an old-to-new adapter are not retained.

## Internal Constraint Model

The public context does not expose the graph constraint representation. Its
initial internal relation is conceptually:

```rust
enum ShapeConstraint {
    Equal {
        lhs: DimExpr,
        rhs: DimExpr,
        source: ConstraintSource,
    },
}
```

The enum intentionally leaves space for future relation kinds, but phase one
constructs only `Equal`. Each expression uses the existing symbolic dimension
operations: constants, input axes, addition, subtraction, multiplication,
floor division, minimum, and maximum.

`ConstraintSource` identifies the extension family and the originating graph
node/input relation well enough to produce an actionable diagnostic. Its
structural origin is lowered to stable compiled node and input positions.
Human-readable labels derived from that origin are diagnostic metadata; those
labels must not affect equality, hashing, compilation, or cache identity. A
cached program therefore retains stable structural provenance and renders its
diagnostic labels from that provenance rather than storing graph-instance
identifiers.

During extension inference, expressions refer to the extension's input slots.
When the extension graph node is built, the inference result is translated to
an op-local constraint plus its ordered graph input values. The constraint is
registered in a graph-level constraint scope associated with the originating
node. It is not embedded in the extension payload and is not part of the
extension author's internal state.

## Ownership and Data Flow

Constraint storage parallels metadata scopes:

1. `ExtensionShapeContext` collects constraints while output metadata is
   inferred.
2. Extension graph construction maps each constraint's input slots to the
   actual graph values and adds it to a local constraint scope.
3. `TracedTensor` output construction merges the local scope with parent
   constraint scopes in the same way metadata scopes are composed.
4. Graph transformations retain the constraint scope for every reachable
   originating node. Newly created extension nodes run metadata inference and
   register their own constraints normally.
5. The compiler gathers constraints reachable from the requested outputs,
   substitutes known input descriptors, and asks the equality engine to prove
   or disprove each relation.
6. Proven constraints are removed. Disproven constraints return a typed error
   before backend execution. Unknown constraints become normalized runtime
   guards in `GraphProgram`.
7. `GraphExecutor` evaluates guards against concrete binding shapes before it
   executes any backend instruction.

Constraint reachability follows graph-node reachability. A constraint is
pruned only when its originating node is dead with respect to all requested
outputs. A live constraint that references a missing value is an internal
constraint evaluation error, not an invitation to silently drop the
constraint.

Although the current compiler normally specializes program inputs using
concrete descriptors, runtime guards remain part of the design. This makes the
contract correct for future polymorphic compilation and prevents a later
compiler change from weakening extension validation.

## Equality Engine

The equality engine returns one of three outcomes:

```rust
enum Proof {
    Proven,
    Disproven { lhs: usize, rhs: usize },
    Unknown,
}
```

Phase one supports:

- checked constant folding;
- proof of structurally identical normalized expressions;
- union-find over bare axis symbols, including transitive equalities such as
  `a == b` and `b == c`;
- binding a bare symbol to a constant and detecting contradictory bindings;
- evaluation of all existing dimension expressions once their referenced
  input shapes are concrete.

The engine operates in order-independent stages. It first collects all bare
symbol equalities and constant bindings, computes their union-find closure,
and reports contradictory bindings. It then substitutes representatives and
bindings into every relation before classifying the remaining expressions.
Consequently, the result does not depend on the order in which extensions
record constraints.

Normalization performs only deterministic, semantics-preserving rewrites:

- constant folding;
- `a + 0 -> a`, `a - 0 -> a`, and `a * 1 -> a`;
- deterministic operand ordering for commutative operations;
- replacement of bare symbols with their union-find representative.

Rewrites must not elide evaluation of a non-constant subexpression. In
particular, phase one does not rewrite `a * 0` to zero because that could hide
an overflow, invalid reference, or division-by-zero error inside `a`.

Complex relations such as `a == 2 * b` are fully representable. If substitution
and constant evaluation do not prove or disprove them, they remain `Unknown`
and are retained as guards. Phase one does not rearrange the equation or solve
for a symbol.

Dimension arithmetic is checked. Overflow, underflow, division by zero, a
missing input, or an invalid axis produces an evaluation error rather than
being treated as `Unknown`. Future range or linear reasoning can be inserted
between normalization and guard emission without changing the public
extension API.

## Compilation, Programs, and Cache Identity

The compiler processes the complete reachable constraint set before lowering
backend work:

- constraints made concrete by input descriptors are evaluated immediately;
- contradictions abort compilation before an executable backend program is
  returned;
- unresolved constraints are converted to normalized program guards;
- guards use the compiled program's input numbering rather than transient
  graph identities.

Normalized guards are part of `GraphProgram` identity and the compilation
cache key. Two programs with identical instructions but different shape
contracts must not share a cache entry. Equivalent constraints normalize to a
deterministic ordering so graph construction order does not cause unnecessary
cache misses.

The executor validates explicit input bindings and program input shapes as it
does today, then evaluates all program guards, then begins backend execution.
This ordering ensures that a guard violation cannot cause partial execution or
backend-dependent behavior.

## AD and Graph Transformation Semantics

An equality is a graph contract, not merely a trace-time assertion. Therefore:

- graph composition merges constraint scopes;
- checkpoint and replay paths carry the same scopes as metadata;
- copied reachable nodes retain their originating constraints;
- extension nodes created by JVP or VJP rules infer metadata through the new
  context and register their own constraints;
- pruning removes constraints only together with their dead originating node.

This is required for the primary motivating contract: an extension can call
`require_same_shape(primal, tangent)`, and the exact-shape requirement remains
effective after AD graph transformation and compilation.

## Errors and Diagnostics

Constraint failures are typed errors. The exact ownership within the existing
error hierarchy will follow current crate boundaries, but the semantic payload
is:

```rust
Error::ShapeConstraintViolation {
    family,
    relation: ShapeRelation::Equal,
    lhs_expr,
    rhs_expr,
    lhs_value,
    rhs_value,
    source,
}

Error::ShapeConstraintEvaluation {
    family,
    expression,
    cause: ShapeConstraintEvalError,
}
```

`ShapeConstraintEvalError` distinguishes at least:

- `MissingInput`;
- `AxisOutOfBounds`;
- `Overflow`;
- `Underflow`;
- `DivisionByZero`.

Expression and source strings exist to help the user locate the failed
contract. They are not used for structural equality or cache identity. Known
constant contradictions found during inference or compilation use the same
violation type as a runtime guard failure, so the diagnostic remains stable
regardless of when a relation becomes concrete.

## Testing Strategy

### Equality engine unit tests

- equal and unequal constants;
- structural equality after normalization;
- transitive symbol equality;
- symbol-to-constant binding and contradictory bindings;
- deterministic union representatives and normalized ordering;
- `a == 2 * b` remaining unknown when symbolic;
- concrete pass and fail cases for `a == 2 * b`;
- overflow, underflow, division by zero, missing input, and invalid axis;
- deterministic equality and hashing for normalized guards.

### Extension integration tests

A small test extension will require equality between axes of independent
inputs. Tests will verify:

- equal concrete independent placeholders are accepted;
- contradictory concrete placeholders fail before execution;
- unresolved relations survive graph composition and compilation;
- a retained runtime guard passes and fails with the expected typed payload;
- `require_same_shape` checks rank and every axis;
- invalid input and axis indices return metadata inference errors.

### Transformation and cache tests

- primal/tangent exact-shape constraints survive JVP and VJP graph
  transformations;
- checkpoint/replay and metadata reconstruction preserve reachable
  constraints;
- dead originating nodes allow their constraints to be pruned;
- live constraints with broken references fail loudly;
- identical normalized guards reuse a cache entry;
- semantically different guards cannot reuse the same entry.

### Migration and documentation verification

- all existing extension implementations and their tests use
  `ExtensionShapeContext`;
- every new public type and method has a compiling, runnable doc example;
- modified and new source files meet the repository's 90% line coverage
  target;
- the workspace formatting, release tests, coverage, documentation, clippy,
  and repository-rule review commands pass before the implementation PR.

## Documentation

The implementation PR will update the durable graph/extension design
documentation and add a work log describing the clean-break migration, solver
boundary, rejected alternatives, and residual risks. Public rustdoc will show
both the common axis-equality case and expression equality such as
`a == 2 * b`.

## Rejected Alternatives

### Store constraints only inside each extension node

This localizes storage but makes composition, AD propagation, centralized
diagnostics, and cache identity inconsistent across extension families.
Graph-level ownership provides one lifecycle and one enforcement point.

### Canonicalize `SymDim` values and infer equality implicitly

Canonicalization can prove that two expressions have the same origin, but it
cannot express the intended equality of independent inputs. Extensions need an
explicit declarative contract.

### Add a full symbolic algebra solver now

A general solver would increase implementation and maintenance cost before the
required relation set is known. Equality recording plus a small proof engine
and runtime guards meets issue #1370 while preserving an extension point for
stronger reasoning.

## Completion Criteria

The feature is complete when extension authors can declare equality between
input-derived dimension expressions; all existing extensions use the new
context API; concrete contradictions are rejected before backend execution;
unresolved constraints survive composition, AD, compilation, and caching as
runtime guards; and the test and documentation requirements above pass.
