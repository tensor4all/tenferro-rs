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

- differentiable programming as the graph substrate used by tenferro AD
- multi-tensor einsum (graph of binary contractions)
- any DAG-structured computation

---

## II. Core Traits

### Operand (removed)

The `Operand` trait has been removed. The associated type `GraphOperation::Operand`
is now bounded by `Clone + Send + Sync + 'static` only — no trait of its own.
Downstream consumers supply whatever runtime value type they need; computegraph
imposes no algebraic or structural interface on it.

### GraphOperation

`GraphOperation` is the operation node trait for **metadata only**: input/output
counts and associated types. `computegraph` is fully generic over this trait
and never references specific primitives. Associated types: `Operand` (runtime
value, no trait bound beyond `Clone + Send + Sync + 'static`), `Context`
(backend execution state), `InputKey` (downstream-chosen key representation).

Evaluation is **not** part of `GraphOperation`. See `EvaluableGraphOperation` below.

```rust
pub trait GraphOperation: Clone + Debug + Hash + Eq + Send + Sync + 'static {
    type Operand: Clone + Send + Sync + 'static;
    type Context;
    type InputKey: Clone + Debug + Hash + Eq + Send + Sync + 'static;

    fn input_count(&self) -> usize;
    fn output_count(&self) -> usize;
}
```

### EvaluableGraphOperation

`EvaluableGraphOperation` is an **opt-in** extension trait that adds evaluation capability
to a `GraphOperation`. Not every graph operation needs to be directly evaluable —
some operations exist only as logical nodes for analysis or transformation.
When evaluation is required, implement this trait.

```rust
pub trait EvaluableGraphOperation: GraphOperation {
    fn eval(&self, ctx: &mut Self::Context, inputs: &[&Self::Operand]) -> Vec<Self::Operand>;
}
```

---

## III. Data Structures

### Graph

A **Graph** is the unit of graph construction. It owns only local nodes and
may reference values in other graphs through external references.

```rust
type LocalValueId = usize;
type LocalOperationId = usize;

enum ValueRef<Op: GraphOperation> {
    Local(LocalValueId),
    External(ValueKey<Op>),
}

struct ValueNode<Op> {
    key: ValueKey<Op>,
    producer: Option<(LocalOperationId, usize)>,
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

`parents` are the lookup base used by `resolve`, not eager ownership.

### Global Identity

`ValueKey` is the cross-graph identity mechanism. It is structural:

```rust
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

`ValueKey` enables:

- external reference resolution
- cross-graph CSE
- logical reachability analysis

### Key Interning

`ValueKey` is recursive and grows with graph depth. To keep equality
comparison O(1) and avoid storing duplicate substructure, all keys are
interned in a global `ValueKeyInterner`:

```rust
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct ValueKeyId(u32);

struct ValueKeyInterner<Op: GraphOperation> {
    map: HashMap<ValueKey<Op>, ValueKeyId>,
    keys: Vec<ValueKey<Op>>,  // id → full key (reverse lookup)
}
```

Graph construction registers every new `ValueKey` with the interner
and stores the resulting `ValueKeyId`. Cross-graph equality, CSE, and
compilation cache lookups all use `ValueKeyId` comparison (O(1)).

The interner is global (shared across all graphs). This is appropriate
because the primary use case is cross-graph identity. Thread-safe access
can be added later (`DashMap` or `RwLock<HashMap>`) if parallel graph
construction becomes necessary.

`parents` in `Graph` form an acyclic structure: a graph can only
reference graphs that existed at its construction time, so cycles are
structurally impossible.

### ResolvedView

`resolve` builds a logical lookup view over graphs without copying nodes.

```rust
trait Resolver<Op> {
    fn resolve_value(&self, key: &ValueKey<Op>) -> Option<ValueDef<Op>>;
}

struct ResolvedView<Op> {
    roots: Vec<Arc<Graph<Op>>>,
    resolver: Arc<dyn Resolver<Op>>,
}
```

### MaterializedGraph

The fully flattened graph produced by `materialize_merge`:

```rust
struct MaterializedGraph<Op> {
    values: Vec<MaterializedVal<Op>>,
    operations: Vec<MaterializedOp<Op>>,
    inputs: Vec<ValueKey<Op>>,
    outputs: Vec<ValueKey<Op>>,
}
```

---

## IV. Transforms

### `resolve`

```rust
fn resolve<Op: GraphOperation>(roots: Vec<Arc<Graph<Op>>>) -> ResolvedView<Op>;
```

Cheap and logical. Prepares a traversal view over graph parents and
external references. Does not copy nodes or run CSE.

### `materialize_merge`

```rust
fn materialize_merge<Op: GraphOperation>(
    view: &ResolvedView<Op>,
    outputs: &[ValueKey<Op>],
) -> MaterializedGraph<Op>;
```

Walks the resolved logical DAG, collects reachable definitions, deduplicates
by `ValueKey`/`OperationKey`, and computes one concrete topological order.

### `compile`

```rust
fn compile<Op: GraphOperation>(graph: &MaterializedGraph<Op>) -> CompiledProgram<Op>;
```

Produces an SSA-form instruction sequence. Each slot is written exactly once.

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

### `eval`

```rust
impl<Op: EvaluableGraphOperation> CompiledProgram<Op> {
    fn eval(&self, ctx: &mut Op::Context, inputs: &[&Op::Operand]) -> Vec<Op::Operand>;
}
```

Compile once, eval many times. Evaluation is **opt-in**: only operations that
implement `EvaluableGraphOperation` can be evaluated. `CompiledProgram` itself is generic
over `GraphOperation`; the `eval` method is available only when `Op: EvaluableGraphOperation`.

---

## V. Compilation Cache

`computegraph` caches compiled programs keyed by graph structure
(`ValueKey`-based identity) to avoid recompilation when the same graph
is submitted multiple times with different input values.

Note: `ValueKey` contains concrete `InputKey` and (via downstream usage)
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
Local(LocalValueId)       -> follow the local producer
External(ValueKey)  -> ask the resolver for the defining op
```

This works recursively through any number of graph boundaries.
Topological traversal is logical, not physical:

- visitation is keyed by `ValueKey`
- ordering is computed on the resolved logical DAG
- local ids only matter inside the graph currently being built

---

## VII. Design Boundaries

```text
computegraph owns:
  - GraphOperation, EvaluableGraphOperation traits
  - Graph, ResolvedView, MaterializedGraph
  - resolve, materialize_merge, compile, eval (opt-in via EvaluableGraphOperation)
  - compilation cache

computegraph does NOT own:
  - AD rule registries or semantic AD transforms -> tenferro crates
  - AD rule traits (linearize, transpose_rule) -> tenferro-internal-ops
  - concrete primitives → downstream (tenferro-rs)
  - backend-specific lowering (StableHLO) → downstream
```
