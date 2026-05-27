# AD Architecture

**Repos:** chainrules-rs, tidu-rs, tenferro-rs
**Parent:** `../index.md`
**Related:** `computegraph.md`, `chainrules.md`, `tidu.md`, `../spec/backend-contract.md`, `../spec/primitive-catalog.md`

---

## I. Vision

Build a differentiable programming stack in Rust where:

- `differentiate` is the only derivative-producing transform. It consumes a
  resolved logical view of computation and produces a new linear fragment.
- `transpose` is not differentiation. It reverses active linear flow in a
  linear fragment and reuses the same derivative information.
- AD transforms operate on **fragments**, not on a single eagerly merged
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
build             user constructs a primal fragment
resolve           create a logical DAG view over one or more fragments
differentiate     resolved view -> new linear fragment (JVP)
transpose         linear fragment -> new linear fragment (reverse linear flow)
materialize_merge resolved view -> MaterializedGraph (flatten + CSE)
compile           MaterializedGraph -> CompiledProgram
eval              CompiledProgram + input values -> output values
```

The key pipeline distinction is:

```text
differentiate -> resolve -> differentiate -> resolve -> ...
```

not

```text
differentiate -> physical merge -> differentiate -> physical merge -> ...
```

`materialize_merge` is still required, but only when a backend, serializer, or
debugger needs one concrete graph.

Typical pipelines:

```text
JVP:
  build -> resolve -> differentiate -> materialize_merge -> compile -> eval

VJP:
  build -> resolve -> differentiate -> transpose -> materialize_merge -> compile -> eval

2nd directional derivative:
  build -> resolve -> differentiate -> resolve -> differentiate -> materialize_merge -> compile -> eval

n-th derivative:
  build -> (resolve -> differentiate) x n -> [transpose] -> materialize_merge -> compile -> eval
```

Four crates, strictly layered:

```text
computegraph   GraphOp + Operand traits, Fragment, resolve,
               materialize_merge, compile (SSA), eval,
               compilation cache
    ↓
chainrules     PrimitiveOp: GraphOp (adds add + linearize + transpose_rule)
    ↓
tidu           differentiate, transpose — generic AD transforms
               over PrimitiveOp; no graph infrastructure of its own
    ↓
tenferro       Concrete tensor primitives + execution lowering
```

`computegraph` provides the general-purpose computation graph engine. It is
usable without AD (e.g. multi-tensor einsum as a graph of binary
contractions). `tidu` is a thin layer that adds AD-specific graph transforms
(`differentiate`, `transpose`), fully generic over `Op: PrimitiveOp`.
Neither `computegraph` nor `tidu` references specific primitives. The
responsibility for ensuring that `linearize` and `transpose_rule` produce
valid, closed fragments belongs entirely to the downstream primitive
implementor (tenferro).

---

## II. Core Model

### Fragment vs MaterializedGraph

A **Fragment** is the unit produced by `build`, `differentiate`, and
`transpose`.

A fragment:

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
Fragment          = transform-time object
ResolvedView      = logical traversal object
MaterializedGraph = compile-time object
```

### Local ids are local only

Local ids are fragment-scoped. They must not be used as cross-fragment
identity.

```rust
type LocalValId = usize;
type LocalOpId = usize;

enum ValRef<Op: GraphOp> {
    Local(LocalValId),
    External(GlobalValKey<Op>),
}

enum GlobalValKey<Op: GraphOp> {
    Input(Op::InputKey),
    Derived {
        op: GlobalOpKey<Op>,
        output_slot: u8,
    },
}

struct GlobalOpKey<Op> {
    primitive: Op,
    inputs: Vec<GlobalValKey<Op>>,
    mode: OpMode,
}

enum OpMode {
    Primal,
    Linear { active_mask: Vec<bool> },
}
```

`GlobalValKey` is the identity that matters across fragments. It is
structural:

- inputs are keyed by `InputKey`
- derived values are keyed by primitive, global input keys, output slot, and
  linear metadata

This is what makes the following possible:

- external reference resolution
- cross-fragment CSE
- higher-order tracing through earlier fragments
- transpose accumulation bucketed by global identity

### Active mask is part of identity

Linear nodes use the same primitive set as primal nodes, but the linear mode is
not optional metadata. It changes the meaning of transpose and therefore must
participate in identity.

Examples:

```text
Mul(a, b)   mode=Primal
Mul(a, dx)  mode=Linear { active_mask=[fixed, active] }
Add(dx, dy) mode=Linear { active_mask=[active, active] }
```

The first and second node both evaluate as multiplication, but they are not the
same graph object. They transpose differently, so they must not alias.

### Fragment data structure

Conceptually:

```rust
struct ValNode<Op> {
    key: GlobalValKey<Op>,
    producer: Option<(LocalOpId, usize)>, // None for fragment inputs
}

struct OpNode<Op> {
    op: Op,
    inputs: Vec<ValRef<Op>>,
    outputs: Vec<LocalValId>,
    mode: OpMode,
}

struct Fragment<Op: GraphOp> {
    vals: Vec<ValNode<Op>>,
    ops: Vec<OpNode<Op>>,
    inputs: Vec<LocalValId>,
    outputs: Vec<LocalValId>,
    parents: Vec<Arc<Fragment<Op>>>,
}
```

`parents` are not eager ownership. They are the lookup base used by `resolve`.

### Resolver and ResolvedView

`resolve` does not copy nodes into one graph. It builds a logical lookup view
over fragments.

```rust
enum ValDef<Op> {
    Input {
        key: InputKey,
    },
    Produced {
        op: Op,
        inputs: Vec<ValRef<Op>>,
        mode: OpMode,
        output_slot: usize,
    },
}

trait Resolver<Op> {
    fn resolve_val(&self, key: &GlobalValKey<Op>) -> Option<ValDef<Op>>;
}

struct ResolvedView<Op> {
    roots: Vec<Arc<Fragment<Op>>>,
    resolver: Arc<dyn Resolver<Op>>,
}
```

The intended implementation is a resolver assembled from parent fragments, not a
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
Local(LocalValId)       -> follow the local producer
External(GlobalValKey)  -> ask the resolver for the defining op
```

This must work recursively through any number of fragment boundaries.

Topological traversal at transform time is therefore **logical**, not physical:

- visitation is keyed by `GlobalValKey`
- the ordering is computed on the resolved logical DAG
- local ids only matter inside the fragment currently being built

### Materialized graph

Compile does not consume a fragment. It consumes the result of
`materialize_merge`.

```rust
struct MaterializedGraph<Op> {
    vals: Vec<MaterializedVal<Op>>,
    ops: Vec<MaterializedOp<Op>>,
    inputs: Vec<GlobalValKey<Op>>,
    outputs: Vec<GlobalValKey<Op>>,
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

`build` creates a primal fragment.

- all nodes are `OpMode::Primal`
- fragment inputs use `GlobalValKey::Input`
- no eager merge is implied

### `resolve`

Conceptually:

```rust
fn resolve<Op: GraphOp>(roots: Vec<Arc<Fragment<Op>>>) -> ResolvedView<Op>;
```

`resolve` is cheap and logical. It prepares a traversal view over fragment
parents and external references.

This is the correct replacement for the old statement:

```text
"merge must precede next differentiate"
```

The precise rule is:

```text
resolve must precede any transform that needs to trace through external refs
```

In practice:

- higher-order `differentiate` requires `resolve`
- dependency analysis requires `resolve`
- `transpose` usually needs only the linear fragment itself plus active masks

### `differentiate`

`differentiate` consumes a resolved view and returns a new linear fragment.

```rust
struct LinearFragment<Op> {
    fragment: Fragment<Op>,
    tangent_inputs: Vec<(InputKey, LocalValId)>,
    tangent_outputs: Vec<Option<LocalValId>>,
}

fn differentiate<Op: PrimitiveOp>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
    wrt: &[InputKey],
) -> LinearFragment<Op>;
```

Important consequences:

- callers specify **which primal inputs** they differentiate with respect to
- tangent inputs are created **inside the returned fragment**
- those tangent inputs receive fresh `InputKey`s and are returned to the caller
- primal values are referenced by `External(GlobalValKey)`, not copied

Algorithm sketch:

```text
1. Traverse the reachable logical DAG in topological order.
2. Seed tangent inputs for the requested primal InputKeys.
3. For each reachable primitive, call its linearization rule.
4. Emit new local linear nodes into the new fragment.
5. Reference primal values through External(GlobalValKey).
6. Skip unreachable tangent flow with zero propagation.
```

There is no physical merge in this step.

### Linear nodes use the primal primitive set

The linear graph uses the same primitive set as the primal graph. There is no
dedicated `Scale` primitive.

Examples:

```text
Mul(a, dx)   mode=Linear { active_mask=[fixed, active] }
Add(dx, dy)  mode=Linear { active_mask=[active, active] }
Exp(x)       mode=Primal
```

That last line matters: `Exp` itself is not a linear node. The linearization of
`Exp(x)` emits linear nodes such as:

```text
Mul(External(exp(x)), dx) mode=Linear { active_mask=[fixed, active] }
```

The design rule is:

- linearization may reference primal inputs or outputs as fixed operands
- linearization must stay linear in tangent inputs
- active-vs-fixed information is recorded in `OpMode::Linear`

### `transpose`

`transpose` consumes a linear fragment and produces another linear fragment with
active inputs and outputs reversed.

```rust
fn transpose<Op: PrimitiveOp>(
    linear: &LinearFragment<Op>,
) -> LinearFragment<Op>;
```

It does not differentiate again. It reuses the same local linear rules with
direction reversed.

`tidu::transpose` is generic. It traverses the linear fragment in reverse
topological order and, for each op node, calls `Op::transpose_rule` to obtain
the local transposed contribution. `tidu` does not know which primitives
exist; it only requires that every op in the linear fragment implements
`PrimitiveOp::transpose_rule`.

Transpose accumulation must use global identity. When multiple reverse
contributions flow back to the same original tangent node, bucket by the
**global key of that tangent value**, not by a fragment-local id.

### `materialize_merge`

`materialize_merge` is the physical graph-building step.

```rust
fn materialize_merge<Op: GraphOp>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
) -> MaterializedGraph<Op>;
```

This step:

- walks the resolved logical DAG from the requested outputs
- collects reachable definitions
- deduplicates by `GlobalValKey` / `GlobalOpKey`
- computes one physical DAG
- prepares the input to `compile`

This is where "merge" actually belongs.

The terminology should therefore be:

```text
resolve            = make external references traceable
materialize_merge  = flatten fragments into one concrete graph
```

### `compile` and `eval`

`compile` consumes a `MaterializedGraph`, not a fragment.

```rust
let view = resolve(vec![fragment_a, fragment_b, fragment_c]);
let graph = materialize_merge(&view, &requested_outputs);
let prog = compile(&graph);
let values = prog.run(&runtime_inputs);
```

This separation is deliberate:

- transforms stay fragment-based
- compile stays backend-oriented
- flattening and CSE happen once, late

---

## IV. Scalar Example: `f(x) = exp(a * x)`

### Step 1: build the primal fragment `F0`

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

### Step 2: differentiate the resolved primal view

```text
L1 = differentiate(resolve([F0]), outputs=[key(p3)], wrt=[x])
```

One possible linear fragment:

```text
t0 = Input(t_x)                                         // new tangent input key
t1 = Mul(External(key(p1)), Local(t0))                  mode=Linear { active_mask=[fixed, active] }
t2 = Mul(External(key(p3)), Local(t1))                  mode=Linear { active_mask=[fixed, active] }
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

Now `differentiate` can trace the output of `L1` through `key(p3)` and then
further through the primal chain back to `x`.

This is the critical distinction:

```text
R1 is enough for higher-order AD.
No physical merge is required yet.
```

### Step 4: transpose the linear fragment

```text
T1 = transpose(L1)
```

One possible transposed fragment:

```text
c0 = Input(ct_y)
c1 = Mul(External(key(p3)), Local(c0))                  mode=Linear { active_mask=[fixed, active] }
c2 = Mul(External(key(p1)), Local(c1))                  mode=Linear { active_mask=[fixed, active] }
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

Second directional derivative uses `resolve`, then `differentiate` again:

```text
L2 = differentiate(resolve([F0, L1]), outputs=[key(t2)], wrt=[x])
```

Again, no physical merge is required before this step.

---

## V. Vector Examples

The vector examples remain mathematically identical to the earlier version.
What changes is only the graph interpretation: fragments stay separate until
`materialize_merge`.

For readability, `Sum` below is shorthand for `ReduceSum` over all axes.

### Vector example 1: elementwise `y = exp(a * x)` with `x, a in R^2`

Primal fragment:

```text
u0 = Input(x:[2])
u1 = Input(a:[2])
u2 = Mul(u0, u1)
u3 = Exp(u2)
```

Linear fragment from `differentiate(resolve([F0]), outputs=[key(u3)], wrt=[x])`:

```text
u4 = Input(t_x:[2])
u5 = Mul(External(key(u1)), Local(u4))                 mode=Linear { active_mask=[fixed, active] }
u6 = Mul(External(key(u3)), Local(u5))                 mode=Linear { active_mask=[fixed, active] }
```

Transposed fragment:

```text
u7 = Input(ct_y:[2])
u8 = Mul(External(key(u3)), Local(u7))                 mode=Linear { active_mask=[fixed, active] }
u9 = Mul(External(key(u1)), Local(u8))                 mode=Linear { active_mask=[fixed, active] }
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
transpose satisfies `<ct_y, dy> = <ct_x, t_x>`.

### Vector example 2: reduction `y = Sum(exp(a * x))` with `x, a in R^2`

Primal fragment:

```text
r0 = Input(x:[2])
r1 = Input(a:[2])
r2 = Mul(r0, r1)
r3 = Exp(r2)
r4 = Sum(r3)
```

Linear fragment:

```text
r5 = Input(t_x:[2])
r6 = Mul(External(key(r1)), Local(r5))                 mode=Linear { active_mask=[fixed, active] }
r7 = Mul(External(key(r3)), Local(r6))                 mode=Linear { active_mask=[fixed, active] }
r8 = Sum(Local(r7))                                    mode=Linear { active_mask=[active] }
```

Transposed fragment:

```text
r9  = Input(ct_y:[])
r10 = BroadcastInDim(Local(r9), shape=[2], dims=[])    mode=Linear { active_mask=[active] }
r11 = Mul(External(key(r3)), Local(r10))               mode=Linear { active_mask=[fixed, active] }
r12 = Mul(External(key(r1)), Local(r11))               mode=Linear { active_mask=[fixed, active] }
```

Resulting formulas:

```text
y    = exp(a0*x0) + exp(a1*x1)
dy   = exp(a0*x0) * a0 * dx0 + exp(a1*x1) * a1 * dx1
ct_x = [a0 * exp(a0*x0) * ct_y,
        a1 * exp(a1*x1) * ct_y]
```

This is the smallest vector example that makes reduction transpose explicit
without requiring eager merge.

A reproducible checker for these two examples is in
`scripts/vector_ad_examples_check.py`.

---

## VI. Primitive Set and Traits

### Operand

`Operand` is the runtime value type (tensor-like; scalars are rank-0 tensors).
Canonical signature in [`spec/primitive-catalog.md`](../spec/primitive-catalog.md).

### PrimitiveOp

`PrimitiveOp` extends `GraphOp` with `add()` (cotangent accumulation),
`linearize`, and `transpose_rule`. tidu is fully generic over this trait.
Canonical signature in [`spec/ad-contract.md`](../spec/ad-contract.md).

### Linearization and transpose rules

A primitive's `linearize` must be linear in tangent inputs. It may:

- reference primal inputs or outputs through `External(GlobalValKey)`
- emit primal primitives in `OpMode::Linear`
- emit `Conj` when required by transpose semantics

It must not introduce nonlinear dependence on tangent inputs.

Fan-out accumulation (when multiple cotangents flow to the same tangent
node during transpose) is handled internally by `tidu::transpose`, not
by an explicit `Dup` primitive. `tidu` buckets reverse contributions by
`GlobalValKey` and accumulates them by emitting `Op::add()` nodes.

A primitive's `transpose_rule` receives cotangent outputs and must produce
cotangent inputs. It must only emit primitives that themselves implement
`PrimitiveOp`. The downstream implementor is responsible for ensuring that
the set of primitives reachable through `linearize` and `transpose_rule`
is closed.

### Closure responsibility

`tidu` does not define or constrain the primitive set. It is fully generic
over `Op: PrimitiveOp`. The only rule is:

> `linearize` and `transpose_rule` must emit only ops that themselves
> implement `PrimitiveOp`.

This ensures that `tidu` can apply `differentiate` and `transpose` to any
fragment without knowledge of the specific primitives involved. The concrete
primitive set and its closure guarantees are entirely the downstream
implementor's responsibility (e.g. tenferro).

There is no dedicated `Scale` primitive in this design.

---

## VII. Compilation and Execution

### Pipeline

```text
Fragments (primal / linear / transposed)
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
struct CompiledProgram<Op: GraphOp> {
    instructions: Vec<Instruction<Op>>,
    input_slots: Vec<usize>,
    output_slots: Vec<usize>,
    n_slots: usize,
}

struct Instruction<Op> {
    op: Op,
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

- AD transforms (tidu) are fragment-based and resolver-backed
- graph infrastructure (computegraph) is AD-agnostic
- backends only see the materialized or compiled result

---

## VIII. Advantages of the Fragment + Resolver Model

### No eager merge between transforms

Higher-order AD requires traceability, not a physically merged graph. Delaying
physical merge avoids repeated flattening and repeated global CSE.

### Better fit for partial transforms

Cross-country evaluation and partial transpose are more natural when transforms
operate on fragments rather than on one giant graph.

### Global identity is explicit

`GlobalValKey` gives one identity mechanism for:

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
resolve before the next differentiate
materialize_merge before compile
```

---

## IX. Golden Tests

Minimal tests that validate the fragment-based transform procedure:

| # | Function | What it checks |
|---|----------|----------------|
| 1 | `x + x` | transpose accumulation buckets by global identity |
| 2 | `x * y` | binary linearization with distinct reverse sinks |
| 3 | `c * z` (complex) | `Conj` appears only in transpose |
| 4 | `x^2` | higher-order AD without eager physical merge |
| 5 | `exp(a*x)` | external refs, resolve-before-differentiate, transpose correctness |
| 6 | `Sum(exp(a*x))` | reduction transpose via `BroadcastInDim` |
| 7 | `exp(a*x)` 3rd order | repeated higher-order closure over fragments |

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

Phases 1–3 (scalar fragment AD, tensor primitives, backend compilation) are
implemented and tested. Current work focuses on:

- Logical-DAG-aware checkpoint scheduling
- Partial transpose / cross-country mode
- Late materialization heuristics
- Operator fusion in compiled IR

---

## XI. Superseded Issues

This document unifies and supersedes:

- tidu-rs#12: tape AD design
- tidu-rs#13: graph-based AD design
- chainrules-rs#7: trait unification
- chainrules-rs#8: DifferentiableOp trait
- tenferro-rs#616: Traced Tensor + StableHLO IR
- tenferro-rs#618: tenferro roadmap (AD portions)
