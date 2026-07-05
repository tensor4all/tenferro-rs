# AD Architecture

**Repos:** computegraph-rs, tidu-rs, tenferro-rs
**Parent:** `../index.md`
**Related:** `computegraph.md`, `primitive-ad.md`, `tidu.md`, `../spec/backend-contract.md`, `../spec/primitive-catalog.md`

---

## I. Vision

Build a differentiable programming stack in Rust where:

- `linearize` is the only derivative-producing transform. It consumes a
  resolved logical view of computation and produces a new linear graph.
- `linear_transpose` is not differentiation. It reverses active linear flow in a
  linear graph and reuses the same derivative information.
- AD transforms operate on **graphs**, not on a single eagerly merged
  graph.
- What higher-order AD needs is **resolve**, not physical merge. External
  references must be traceable; they do not need to be copied into one graph
  at every stage.
- Physical flattening happens only in `materialize_merge`, typically just
  before `compile`.
- Evaluation is always forward on a **materialized graph** after flattening,
  CSE, and slot assignment.

This is the intended operation set:

```text
build             user constructs a primal graph
resolve           create a logical DAG view over one or more graphs
linearize         resolved view -> new linear graph (JVP)
linear_transpose  linear graph -> new linear graph (reverse linear flow)
materialize_merge resolved view -> MaterializedGraph (flatten + CSE)
compile           MaterializedGraph -> CompiledProgram
eval              CompiledProgram + input values -> output values
```

The key pipeline distinction is:

```text
linearize -> resolve -> linearize -> resolve -> ...
```

not

```text
linearize -> physical merge -> linearize -> physical merge -> ...
```

`materialize_merge` is still required, but only when a backend, serializer, or
debugger needs one concrete graph.

Typical pipelines:

```text
JVP:
  build -> resolve -> linearize -> materialize_merge -> compile -> eval

VJP:
  build -> resolve -> linearize -> linear_transpose -> materialize_merge -> compile -> eval

2nd directional derivative:
  build -> resolve -> linearize -> resolve -> linearize -> materialize_merge -> compile -> eval

n-th derivative:
  build -> (resolve -> linearize) x n -> [linear_transpose] -> materialize_merge -> compile -> eval
```

Current tenferro traced VJP follows the generic VJP pipeline first:

```text
build -> resolve -> linearize active outputs -> linear_transpose -> materialize_merge -> compile -> eval
```

Direct primary-graph transpose is an escape hatch. It is used only when the
generic `linearize + linear_transpose` path cannot build a usable reverse graph.
This keeps forward and reverse AD behavior anchored on the same linearization
rules while preserving support for extension rules that still need primal
outputs directly.

Before `linearize`, traced AD computes the set of primal `ValueKey`s reachable
from the requested output. That active-output set is attached to the
`ShapeGuardContext` so multi-output rules can avoid emitting tangent branches
for unused outputs. For example, an SVD whose caller only consumes singular
values can emit just the `dS = diag(U^H dA V)` path and skip the vector
F-matrix chain before `materialize_merge` ever runs. `materialize_merge` and
the compiler still perform their own deduplication and backend-oriented
optimization, but AD rules should not rely on late flattening to remove large,
known-inactive derivative subgraphs.

Current tenferro eager AD uses the same primitive rule set through a different
interpreter:

```text
forward eager op:
  execute concrete op -> record a small RecordedGraph node

stateful eager backward:
  tidu backward walker -> per-node linearize -> linear_transpose -> execute cotangents

functional eager jvp:
  tidu forward walker -> per-node linearize -> execute tangents

functional eager grad/vjp:
  tidu backward walker -> execute emitted derivative primitives through eager recording
```

tidu owns the eager trace traversal and determines which recorded graph nodes
are visited. tenferro owns concrete tensor execution, public eager APIs,
gradient slots, extension-rule context, and runtime caches. Functional eager
transforms return ordinary eager tensors rather than writing into gradient
slots, so their derivative computations can be traced by later eager
transforms.

Traced AD transform outputs are not cached in a new persistent AD graph cache.
The caller-owned traced result keeps the transformed graph and any required
extra roots alive, while compile-time and runtime caches remain owned by the
existing compiler, executor, and extension runtime contexts. Eager runtimes do
own a bounded Tier-1 AD transform cache for `RecordedGraph` linearization,
keyed by recorded graph structural fingerprint plus requested output slots.
That cache is a same-tape per-node memoization layer; a future whole-program AD
optimizer cache would need keys that include the resolved output keys, `wrt`
inputs, extension rule set identity, shape/dtype guard inputs, and optimizer
configuration version. Such keys must not live inside semantic op payloads.

Four crates, strictly layered:

```text
computegraph   GraphOperation + Operand traits, Graph, resolve,
               materialize_merge, compile (SSA), eval,
               compilation cache
    ↓
tidu           Primitive: GraphOperation (adds add + JVP + transpose_rule),
               linearize, linear_transpose, eager forward/backward walkers;
               no graph infrastructure of its own
    ↓
tenferro       Concrete tensor primitives + execution lowering
```

`computegraph` provides the general-purpose computation graph engine. It is
usable without AD (e.g. multi-tensor einsum as a graph of binary
contractions). `tidu` is a thin layer that adds AD-specific graph transforms
(`linearize`, `linear_transpose`), fully generic over `Op: Primitive`.
Neither `computegraph` nor `tidu` references specific primitives. The
responsibility for ensuring that `linearize` and `transpose_rule` produce
valid, closed graphs belongs entirely to the downstream primitive
implementor (tenferro).

---

## II. Core Model

### Graph vs MaterializedGraph

A **Graph** is the unit produced by `build`, `linearize`, and
`linear_transpose`.

A graph:

- owns only its **local** nodes and ops
- may reference values defined elsewhere through external references
- is valid as long as those external references are **resolvable**

A **MaterializedGraph** is different. It is the fully flattened graph produced
by `materialize_merge`:

- all reachable definitions are collected
- same-key nodes are unified
- a concrete DAG exists for compile, serialization, and debug printing

So the intended mental model is:

```text
Graph          = transform-time object
ResolvedView      = logical traversal object
MaterializedGraph = compile-time object
```

### Local ids are local only

Local ids are graph-scoped. They must not be used as cross-graph
identity.

```rust
type LocalValueId = usize;
type LocalOperationId = usize;

enum ValueRef<Op: GraphOperation> {
    Local(LocalValueId),
    External(ValueKey<Op>),
}

enum ValueKey<Op: GraphOperation> {
    Input(Op::InputKey),
    Derived {
        operation: Arc<OperationKey<Op>>,
        output_slot: u8,
    },
}

struct OperationKey<Op> {
    operation: Op,
    inputs: Vec<ValueKey<Op>>,
    role: OperationRole,
}

enum OperationRole {
    Primary,
    Linearized { active_mask: Vec<bool> },
}
```

`ValueKey` is the identity that matters across graphs. It is
structural:

- inputs are keyed by `InputKey`
- derived values are keyed by primitive, global input keys, output slot, and
  linear metadata

This is what makes the following possible:

- external reference resolution
- cross-graph CSE
- higher-order tracing through earlier graphs
- linear_transpose accumulation bucketed by global identity

### Active mask is part of identity

Linear nodes use the same primitive set as primal nodes, but the linear mode is
not optional metadata. It changes the meaning of linear_transpose and therefore must
participate in identity.

Examples:

```text
Mul(a, b)   role=Primary
Mul(a, dx)  role=Linearized { active_mask=[fixed, active] }
Add(dx, dy) role=Linearized { active_mask=[active, active] }
```

The first and second node both evaluate as multiplication, but they are not the
same graph object. They linear_transpose differently, so they must not alias.

### Graph data structure

Conceptually:

```rust
struct ValueNode<Op> {
    key: ValueKey<Op>,
    producer: Option<(LocalOperationId, usize)>, // None for graph inputs
}

struct OperationNode<Op> {
    operation: Op,
    inputs: Vec<ValueRef<Op>>,
    outputs: Vec<LocalValueId>,
    role: OperationRole,
}

struct Graph<Op: GraphOperation> {
    values: Vec<ValueNode<Op>>,
    operations: Vec<OperationNode<Op>>,
    inputs: Vec<LocalValueId>,
    outputs: Vec<LocalValueId>,
    parents: Vec<Arc<Graph<Op>>>,
}
```

`parents` are not eager ownership. They are the lookup base used by `resolve`.

### Resolver and ResolvedView

`resolve` does not copy nodes into one graph. It builds a logical lookup view
over graphs.

```rust
enum ValueDef<Op> {
    Input {
        key: InputKey,
    },
    Produced {
        operation: Op,
        inputs: Vec<ValueRef<Op>>,
        role: OperationRole,
        output_slot: usize,
    },
}

trait Resolver<Op> {
    fn resolve_value(&self, key: &ValueKey<Op>) -> Option<ValueDef<Op>>;
}

struct ResolvedView<Op> {
    roots: Vec<Arc<Graph<Op>>>,
    resolver: Arc<dyn Resolver<Op>>,
}
```

The intended implementation is a resolver assembled from parent graphs, not a
mandatory central registry.

`resolve` therefore means:

- an external value is not dangling
- its definition can be found
- its dependencies can be followed recursively

It does **not** mean:

- nodes are copied into one physical graph
- CSE has already run
- slot assignment has already happened

### Logical traversal

All transform-time walkers operate on the same logical rule:

```text
Local(LocalValueId)       -> follow the local producer
External(ValueKey)  -> ask the resolver for the defining op
```

This must work recursively through any number of graph boundaries.

Topological traversal at transform time is therefore **logical**, not physical:

- visitation is keyed by `ValueKey`
- the ordering is computed on the resolved logical DAG
- local ids only matter inside the graph currently being built

### Materialized graph

Compile does not consume a graph. It consumes the result of
`materialize_merge`.

```rust
struct MaterializedGraph<Op> {
    values: Vec<MaterializedVal<Op>>,
    operations: Vec<MaterializedOp<Op>>,
    inputs: Vec<ValueKey<Op>>,
    outputs: Vec<ValueKey<Op>>,
}
```

`materialize_merge` is the stage that:

- collects the reachable subgraph
- deduplicates by global identity
- computes one concrete topological order
- produces compile-ready graph state

---

## III. Transformations

### `build`

`build` creates a primal graph.

- all nodes are `OperationRole::Primary`
- graph inputs use `ValueKey::Input`
- no eager merge is implied

### `resolve`

Conceptually:

```rust
fn resolve<Op: GraphOperation>(roots: Vec<Arc<Graph<Op>>>) -> ResolvedView<Op>;
```

`resolve` is cheap and logical. It prepares a traversal view over graph
parents and external references.

This is the correct replacement for the old statement:

```text
"merge must precede next linearize"
```

The precise rule is:

```text
resolve must precede any transform that needs to trace through external refs
```

In practice:

- higher-order `linearize` requires `resolve`
- dependency analysis requires `resolve`
- `linear_transpose` usually needs only the linear graph itself plus active masks

### `linearize`

`linearize` consumes a resolved view and returns a new linear graph.

```rust
struct LinearizedGraph<Op> {
    graph: Graph<Op>,
    tangent_inputs: Vec<(InputKey, LocalValueId)>,
    tangent_outputs: Vec<Option<LocalValueId>>,
}

fn linearize<Op: Primitive>(
    view: &ResolvedView<Op>,
    outputs: &[ValueKey<Op>],
    wrt: &[InputKey],
) -> LinearizedGraph<Op>;
```

Important consequences:

- callers specify **which primal inputs** they linearize with respect to
- tangent inputs are created **inside the returned graph**
- those tangent inputs receive fresh `InputKey`s and are returned to the caller
- primal values are referenced by `External(ValueKey)`, not copied

Algorithm sketch:

```text
1. Traverse the reachable logical DAG in topological order.
2. Seed tangent inputs for the requested primal InputKeys.
3. For each reachable primitive, call its linearization rule.
4. Emit new local linear nodes into the new graph.
5. Reference primal values through External(ValueKey).
6. Skip unreachable tangent flow with zero propagation.
```

There is no physical merge in this step.

### Linear nodes use the primal primitive set

The linear graph uses the same primitive set as the primal graph. There is no
dedicated `Scale` primitive.

Examples:

```text
Mul(a, dx)   role=Linearized { active_mask=[fixed, active] }
Add(dx, dy)  role=Linearized { active_mask=[active, active] }
Exp(x)       role=Primary
```

That last line matters: `Exp` itself is not a linear node. The linearization of
`Exp(x)` emits linear nodes such as:

```text
Mul(External(exp(x)), dx) role=Linearized { active_mask=[fixed, active] }
```

The design rule is:

- linearization may reference primal inputs or outputs as fixed operands
- linearization must stay linear in tangent inputs
- active-vs-fixed information is recorded in `OperationRole::Linearized`

### `linear_transpose`

`linear_transpose` consumes a linear graph and produces another linear graph with
active inputs and outputs reversed.

```rust
fn linear_transpose<Op: Primitive>(
    linear: &LinearizedGraph<Op>,
) -> LinearizedGraph<Op>;
```

It does not linearize again. It reuses the same local linear rules with
direction reversed.

`tidu::linear_transpose` is generic. It traverses the linear graph in reverse
topological order and, for each op node, calls `Op::transpose_rule` to obtain
the local transposed contribution. `tidu` does not know which primitives
exist; it only requires that every op in the linear graph implements
`Primitive::transpose_rule`.

Transpose accumulation must use global identity. When multiple reverse
contributions flow back to the same original tangent node, bucket by the
**global key of that tangent value**, not by a graph-local id.

### `materialize_merge`

`materialize_merge` is the physical graph-building step.

```rust
fn materialize_merge<Op: GraphOperation>(
    view: &ResolvedView<Op>,
    outputs: &[ValueKey<Op>],
) -> MaterializedGraph<Op>;
```

This step:

- walks the resolved logical DAG from the requested outputs
- collects reachable definitions
- deduplicates by `ValueKey` / `OperationKey`
- computes one physical DAG
- prepares the input to `compile`

This is where "merge" actually belongs.

The terminology should therefore be:

```text
resolve            = make external references traceable
materialize_merge  = flatten graphs into one concrete graph
```

### `compile` and `eval`

`compile` consumes a `MaterializedGraph`, not a graph.

```rust
let view = resolve(vec![graph_a, graph_b, graph_c]);
let graph = materialize_merge(&view, &requested_outputs);
let prog = compile(&graph);
let values = prog.run(&runtime_inputs);
```

This separation is deliberate:

- transforms stay graph-based
- compile stays backend-oriented
- flattening and CSE happen once, late

---

## IV. Scalar Example: `f(x) = exp(a * x)`

### Step 1: build the primal graph `F0`

```text
p0 = Input(x)
p1 = Input(a)
p2 = Mul(p0, p1)
p3 = Exp(p2)
```

Key identities:

```text
key(p0) = Input(x)
key(p1) = Input(a)
key(p2) = Derived { op=Mul(Input(x), Input(a)), output_slot=0 }
key(p3) = Derived { op=Exp(key(p2)), output_slot=0 }
```

### Step 2: linearize the resolved primal view

```text
L1 = linearize(resolve([F0]), outputs=[key(p3)], wrt=[x])
```

One possible linear graph:

```text
t0 = Input(t_x)                                         // new tangent input key
t1 = Mul(External(key(p1)), Local(t0))                  role=Linearized { active_mask=[fixed, active] }
t2 = Mul(External(key(p3)), Local(t1))                  role=Linearized { active_mask=[fixed, active] }
```

Important facts:

- `t0`, `t1`, and `t2` are local to `L1`
- `key(p1)` and `key(p3)` are external references into `F0`
- no physical merge has happened

### Step 3: resolve for higher-order tracing

If we want a second derivative, we resolve the combined logical view:

```text
R1 = resolve([F0, L1])
```

Now `linearize` can trace the output of `L1` through `key(p3)` and then
further through the primal chain back to `x`.

This is the critical distinction:

```text
R1 is enough for higher-order AD.
No physical merge is required yet.
```

### Step 4: linear_transpose the linear graph

```text
T1 = linear_transpose(L1)
```

One possible transposed graph:

```text
c0 = Input(ct_y)
c1 = Mul(External(key(p3)), Local(c0))                  role=Linearized { active_mask=[fixed, active] }
c2 = Mul(External(key(p1)), Local(c1))                  role=Linearized { active_mask=[fixed, active] }
```

This computes the cotangent with respect to `x`.

### Step 5: materialize only when compiling

```text
view  = resolve([F0, T1])
graph = materialize_merge(view, [key(p3), key(c2)])
prog  = compile(graph)
```

Conceptually, the materialized graph contains:

```text
Input(x)
Input(a)
Mul(x, a)
Exp(a*x)
Input(ct_y)
Mul(exp(a*x), ct_y)
Mul(a, exp(a*x) * ct_y)
```

### Resulting formulas

```text
y    = exp(a*x)
dy   = exp(a*x) * a * dx
ct_x = a * exp(a*x) * ct_y
```

### Higher order

Second directional derivative uses `resolve`, then `linearize` again:

```text
L2 = linearize(resolve([F0, L1]), outputs=[key(t2)], wrt=[x])
```

Again, no physical merge is required before this step.

---

## V. Vector Examples

The vector examples remain mathematically identical to the earlier version.
What changes is only the graph interpretation: graphs stay separate until
`materialize_merge`.

For readability, `Sum` below is shorthand for `ReduceSum` over all axes.

### Vector example 1: elementwise `y = exp(a * x)` with `x, a in R^2`

Primal graph:

```text
u0 = Input(x:[2])
u1 = Input(a:[2])
u2 = Mul(u0, u1)
u3 = Exp(u2)
```

Linear graph from `linearize(resolve([F0]), outputs=[key(u3)], wrt=[x])`:

```text
u4 = Input(t_x:[2])
u5 = Mul(External(key(u1)), Local(u4))                 role=Linearized { active_mask=[fixed, active] }
u6 = Mul(External(key(u3)), Local(u5))                 role=Linearized { active_mask=[fixed, active] }
```

Transposed graph:

```text
u7 = Input(ct_y:[2])
u8 = Mul(External(key(u3)), Local(u7))                 role=Linearized { active_mask=[fixed, active] }
u9 = Mul(External(key(u1)), Local(u8))                 role=Linearized { active_mask=[fixed, active] }
```

Resulting formulas:

```text
y    = [exp(a0*x0), exp(a1*x1)]
dy   = [exp(a0*x0) * a0 * dx0,
        exp(a1*x1) * a1 * dx1]
ct_x = [a0 * exp(a0*x0) * ct_y0,
        a1 * exp(a1*x1) * ct_y1]
```

This stays purely elementwise. The JVP matches finite differences and the
linear_transpose satisfies `<ct_y, dy> = <ct_x, t_x>`.

### Vector example 2: reduction `y = Sum(exp(a * x))` with `x, a in R^2`

Primal graph:

```text
r0 = Input(x:[2])
r1 = Input(a:[2])
r2 = Mul(r0, r1)
r3 = Exp(r2)
r4 = Sum(r3)
```

Linear graph:

```text
r5 = Input(t_x:[2])
r6 = Mul(External(key(r1)), Local(r5))                 role=Linearized { active_mask=[fixed, active] }
r7 = Mul(External(key(r3)), Local(r6))                 role=Linearized { active_mask=[fixed, active] }
r8 = Sum(Local(r7))                                    role=Linearized { active_mask=[active] }
```

Transposed graph:

```text
r9  = Input(ct_y:[])
r10 = BroadcastInDim(Local(r9), shape=[2], dims=[])    role=Linearized { active_mask=[active] }
r11 = Mul(External(key(r3)), Local(r10))               role=Linearized { active_mask=[fixed, active] }
r12 = Mul(External(key(r1)), Local(r11))               role=Linearized { active_mask=[fixed, active] }
```

Resulting formulas:

```text
y    = exp(a0*x0) + exp(a1*x1)
dy   = exp(a0*x0) * a0 * dx0 + exp(a1*x1) * a1 * dx1
ct_x = [a0 * exp(a0*x0) * ct_y,
        a1 * exp(a1*x1) * ct_y]
```

This is the smallest vector example that makes reduction linear_transpose explicit
without requiring eager merge.

A reproducible checker for these two examples is in
`scripts/vector_ad_examples_check.py`.

---

## VI. Primitive Set and Traits

### Operand

`Operand` is the runtime value type (tensor-like; scalars are rank-0 tensors).
Canonical signature in [`spec/primitive-catalog.md`](../spec/primitive-catalog.md).

### Primitive

`Primitive` extends `GraphOperation` with `add()` (cotangent accumulation),
`linearize`, and `transpose_rule`. tidu is fully generic over this trait.
Canonical signature in [`spec/ad-contract.md`](../spec/ad-contract.md).

### Linearization and linear_transpose rules

A primitive's `linearize` must be linear in tangent inputs. It may:

- reference primal inputs or outputs through `External(ValueKey)`
- add primal primitives in `OperationRole::Linearized`
- add `Conj` when required by linear_transpose semantics

It must not introduce nonlinear dependence on tangent inputs.

Fan-out accumulation (when multiple cotangents flow to the same tangent
node during linear_transpose) is handled internally by `tidu::linear_transpose`, not
by an explicit `Dup` primitive. `tidu` buckets reverse contributions by
`ValueKey` and accumulates them by emitting `Op::add()` nodes.

A primitive's `transpose_rule` receives cotangent outputs and must produce
cotangent inputs. It must only add primitives that themselves implement
`Primitive`. The downstream implementor is responsible for ensuring that
the set of primitives reachable through `linearize` and `transpose_rule`
is closed.

### Closure responsibility

`tidu` does not define or constrain the primitive set. It is fully generic
over `Op: Primitive`. The only rule is:

> `linearize` and `transpose_rule` must add only ops that themselves
> implement `Primitive`.

This ensures that `tidu` can apply `linearize` and `linear_transpose` to any
graph without knowledge of the specific primitives involved. The concrete
primitive set and its closure guarantees are entirely the downstream
implementor's responsibility (e.g. tenferro).

There is no dedicated `Scale` primitive in this design.

---

## VII. Compilation and Execution

### Pipeline

```text
Graphs (primal / linear / transposed)
    |
    | resolve                          ← computegraph
    v
Resolved logical DAG
    |
    | materialize_merge                ← computegraph
    v
MaterializedGraph
    |
    | compile (SSA)                    ← computegraph
    v
CompiledProgram
    |
    | eval                             ← computegraph
    | or lower to StableHLO backends   ← tenferro
    v
Runtime values
```

### CompiledProgram

`CompiledProgram` is an SSA-form instruction sequence produced by
`computegraph::compile`. Each slot is written exactly once.

```rust
struct CompiledProgram<Op: GraphOperation> {
    instructions: Vec<Instruction<Op>>,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    n_slots: usize,
}

struct Instruction<Op> {
    operation: Op,
    inputs: Vec<usize>,
    outputs: Vec<usize>,
}
```

`compile` runs only after `materialize_merge`:

- one physical DAG exists
- same-key nodes are already unified
- slot assignment can be computed once

Compile once, eval many times. `computegraph` caches compiled programs keyed
by graph structure to avoid recompilation.

### Backend boundary

This document stops at `MaterializedGraph -> CompiledProgram`. Backend-specific
details (execution lowering, GPU dispatch) remain in `../spec/backend-contract.md`.

The important contract is:

- AD transforms (tidu) are graph-based and resolver-backed
- eager AD traversal (tidu) is separate from concrete execution and cache
  ownership (tenferro)
- graph infrastructure (computegraph) is AD-agnostic
- backends only see the materialized or compiled result

---

## VIII. Advantages of the Graph + Resolver Model

### No eager merge between transforms

Higher-order AD requires traceability, not a physically merged graph. Delaying
physical merge avoids repeated flattening and repeated global CSE.

### Better fit for partial transforms

Cross-country evaluation and partial linear_transpose are more natural when transforms
operate on graphs rather than on one giant graph.

### Global identity is explicit

`ValueKey` gives one identity mechanism for:

- external refs
- accumulation buckets
- CSE
- logical reachability

### Compile-time work is isolated

Only `materialize_merge` and `compile` need one concrete graph. Transform-time
code can stay light and local.

### Higher-order AD stays clean

The rule for higher order is simple:

```text
resolve before the next linearize
materialize_merge before compile
```

---

## IX. Golden Tests

Minimal tests that validate the graph-based transform procedure:

| # | Function | What it checks |
|---|----------|----------------|
| 1 | `x + x` | linear_transpose accumulation buckets by global identity |
| 2 | `x * y` | binary linearization with distinct reverse sinks |
| 3 | `c * z` (complex) | `Conj` appears only in linear_transpose |
| 4 | `x^2` | higher-order AD without eager physical merge |
| 5 | `exp(a*x)` | external refs, resolve-before-linearize, linear_transpose correctness |
| 6 | `Sum(exp(a*x))` | reduction linear_transpose via `BroadcastInDim` |
| 7 | `exp(a*x)` 3rd order | repeated higher-order closure over graphs |

Expected second-order result for `x^2` with unit seeds:

| Mode | Output |
|------|--------|
| FoF | 2 |
| FoR | 2 |
| RoF | 2 |
| RoR | 2 |

Expected second-order result for `exp(a*x)` with unit seeds:

| Mode | Output |
|------|--------|
| FoF | `a^2 exp(ax)` |
| FoR | `a^2 exp(ax)` |
| RoF | `a^2 exp(ax)` |
| RoR | `a^2 exp(ax)` |

---

## X. Implementation Status

Phases 1–3 (scalar graph AD, tensor primitives, backend compilation) are
implemented and tested. Eager reverse mode, eager JVP, functional eager
`grad`/`vjp`/`jvp`, and per-recorded-graph eager linearization caching are
implemented on the same primitive rule set. Current work focuses on:

- Logical-DAG-aware checkpoint scheduling
- Partial linear_transpose / cross-country mode
- Late materialization heuristics
- Operator fusion in compiled IR

---

## XI. Superseded Issues

This document unifies and supersedes:

- tidu-rs#12: tape AD design
- tidu-rs#13: graph-based AD design
- tidu-rs#7: trait unification
- tidu-rs#8: DifferentiableOp trait
- tenferro-rs#616: Traced Tensor + StableHLO IR
- tenferro-rs#618: tenferro roadmap (AD portions)
