# computegraph-rs Design

**Repo:** computegraph-rs (new)
**Parent:** `../index.md`

---

## I. Purpose

`computegraph-rs` is a general-purpose computation graph engine in Rust. It
provides the data structures and transforms for building, resolving, flattening,
compiling, and evaluating computation graphs.

It is **AD-agnostic**. Automatic differentiation is not part of this crate.
The graph infrastructure is equally usable for:

- differentiable programming (via tidu-rs)
- multi-tensor einsum (graph of binary contractions)
- any DAG-structured computation

---

## II. Core Traits

### Operand (removed)

The `Operand` trait has been removed. The associated type `GraphOp::Operand`
is now bounded by `Clone + Send + Sync + 'static` only — no trait of its own.
Downstream consumers supply whatever runtime value type they need; computegraph
imposes no algebraic or structural interface on it.

### GraphOp

`GraphOp` is the operation node trait for **metadata only**: input/output
counts and associated types. `computegraph` is fully generic over this trait
and never references specific primitives. Associated types: `Operand` (runtime
value, no trait bound beyond `Clone + Send + Sync + 'static`), `Context`
(backend execution state), `InputKey` (downstream-chosen key representation).

Evaluation is **not** part of `GraphOp`. See `EvalGraphOp` below.

```rust
pub trait GraphOp: Clone + Debug + Hash + Eq + Send + Sync + 'static {
    type Operand: Clone + Send + Sync + 'static;
    type Context;
    type InputKey: Clone + Debug + Hash + Eq + Send + Sync + 'static;

    fn n_inputs(&self) -> usize;
    fn n_outputs(&self) -> usize;
}
```

### EvalGraphOp

`EvalGraphOp` is an **opt-in** extension trait that adds evaluation capability
to a `GraphOp`. Not every graph operation needs to be directly evaluable —
some operations exist only as logical nodes for analysis or transformation.
When evaluation is required, implement this trait.

```rust
pub trait EvalGraphOp: GraphOp {
    fn eval(&self, ctx: &mut Self::Context, inputs: &[&Self::Operand]) -> Vec<Self::Operand>;
}
```

---

## III. Data Structures

### Fragment

A **Fragment** is the unit of graph construction. It owns only local nodes and
may reference values in other fragments through external references.

```rust
type LocalValId = usize;
type LocalOpId = usize;

enum ValRef<Op: GraphOp> {
    Local(LocalValId),
    External(GlobalValKey<Op>),
}

struct ValNode<Op> {
    key: GlobalValKey<Op>,
    producer: Option<(LocalOpId, usize)>,
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

`parents` are the lookup base used by `resolve`, not eager ownership.

### Global Identity

`GlobalValKey` is the cross-fragment identity mechanism. It is structural:

```rust
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

`GlobalValKey` enables:

- external reference resolution
- cross-fragment CSE
- logical reachability analysis

### Key Interning

`GlobalValKey` is recursive and grows with graph depth. To keep equality
comparison O(1) and avoid storing duplicate substructure, all keys are
interned in a global `KeyInterner`:

```rust
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct ValKeyId(u32);

struct KeyInterner<Op: GraphOp> {
    map: HashMap<GlobalValKey<Op>, ValKeyId>,
    keys: Vec<GlobalValKey<Op>>,  // id → full key (reverse lookup)
}
```

Fragment construction registers every new `GlobalValKey` with the interner
and stores the resulting `ValKeyId`. Cross-fragment equality, CSE, and
compilation cache lookups all use `ValKeyId` comparison (O(1)).

The interner is global (shared across all fragments). This is appropriate
because the primary use case is cross-fragment identity. Thread-safe access
can be added later (`DashMap` or `RwLock<HashMap>`) if parallel fragment
construction becomes necessary.

`parents` in `Fragment` form an acyclic structure: a fragment can only
reference fragments that existed at its construction time, so cycles are
structurally impossible.

### ResolvedView

`resolve` builds a logical lookup view over fragments without copying nodes.

```rust
trait Resolver<Op> {
    fn resolve_val(&self, key: &GlobalValKey<Op>) -> Option<ValDef<Op>>;
}

struct ResolvedView<Op> {
    roots: Vec<Arc<Fragment<Op>>>,
    resolver: Arc<dyn Resolver<Op>>,
}
```

### MaterializedGraph

The fully flattened graph produced by `materialize_merge`:

```rust
struct MaterializedGraph<Op> {
    vals: Vec<MaterializedVal<Op>>,
    ops: Vec<MaterializedOp<Op>>,
    inputs: Vec<GlobalValKey<Op>>,
    outputs: Vec<GlobalValKey<Op>>,
}
```

---

## IV. Transforms

### `resolve`

```rust
fn resolve<Op: GraphOp>(roots: Vec<Arc<Fragment<Op>>>) -> ResolvedView<Op>;
```

Cheap and logical. Prepares a traversal view over fragment parents and
external references. Does not copy nodes or run CSE.

### `materialize_merge`

```rust
fn materialize_merge<Op: GraphOp>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
) -> MaterializedGraph<Op>;
```

Walks the resolved logical DAG, collects reachable definitions, deduplicates
by `GlobalValKey`/`GlobalOpKey`, and computes one concrete topological order.

### `compile`

```rust
fn compile<Op: GraphOp>(graph: &MaterializedGraph<Op>) -> CompiledProgram<Op>;
```

Produces an SSA-form instruction sequence. Each slot is written exactly once.

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

### `eval`

```rust
impl<Op: EvalGraphOp> CompiledProgram<Op> {
    fn eval(&self, ctx: &mut Op::Context, inputs: &[&Op::Operand]) -> Vec<Op::Operand>;
}
```

Compile once, eval many times. Evaluation is **opt-in**: only operations that
implement `EvalGraphOp` can be evaluated. `CompiledProgram` itself is generic
over `GraphOp`; the `eval` method is available only when `Op: EvalGraphOp`.

---

## V. Compilation Cache

`computegraph` caches compiled programs keyed by graph structure
(`GlobalValKey`-based identity) to avoid recompilation when the same graph
is submitted multiple times with different input values.

Note: `GlobalValKey` contains concrete `InputKey` and (via downstream usage)
`DiffPassId` values. Two graphs that differ only in these identity tokens but
have identical topology are **not** automatic cache hits at the computegraph
level. Normalization of `InputKey`/`DiffPassId` to achieve cache hits across
higher-order AD passes is the responsibility of the downstream consumer (e.g.
tenferro `GraphCompiler`), not computegraph itself. computegraph provides the
raw structural identity; the compiler maps that to a normalized cache key.

---

## VI. Logical Traversal

All walkers operate on the same rule:

```text
Local(LocalValId)       -> follow the local producer
External(GlobalValKey)  -> ask the resolver for the defining op
```

This works recursively through any number of fragment boundaries.
Topological traversal is logical, not physical:

- visitation is keyed by `GlobalValKey`
- ordering is computed on the resolved logical DAG
- local ids only matter inside the fragment currently being built

---

## VII. Design Boundaries

```text
computegraph owns:
  - GraphOp, EvalGraphOp traits
  - Fragment, ResolvedView, MaterializedGraph
  - resolve, materialize_merge, compile, eval (opt-in via EvalGraphOp)
  - compilation cache

computegraph does NOT own:
  - AD transforms (differentiate, transpose) → tidu-rs
  - AD traits (linearize, transpose_rule) → chainrules-rs
  - concrete primitives → downstream (tenferro-rs)
  - backend-specific lowering (StableHLO) → downstream
```
