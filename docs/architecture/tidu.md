# tidu-rs Design

**Repo:** tidu-rs
**Parent:** `../index.md`
**Depends on:** `computegraph-rs`

---

## I. Purpose

`tidu-rs` provides AD-specific graph transforms (`linearize`, `linear_transpose`)
that are fully generic over `Op: Primitive`. It owns no graph infrastructure
(that belongs to `computegraph-rs`) and references no specific primitives.

Among the JAX concepts, `linearize` is the closest analogue to
`jax.linearize`: it traverses a primal computation and builds a new linear
computation by calling each primitive's local linearization rule. The output is
not StableHLO and not a backend kernel plan; it is another graph composed of
the same downstream primitive vocabulary.

---

## II. Transforms

### `linearize`

Consumes a resolved view and returns a new linearized graph (JVP). The returned
value separates the strictly-linear tangent sweep from residual values
referenced by that sweep.

```rust
use computegraph::{ResolvedView, ValueKey, LocalValueId, Graph};
use std::sync::Arc;
use tidu::Primitive;

struct LinearizedGraph<Op> {
    linear: Graph<Op>,
    residual: Arc<Graph<Op>>,
    tangent_inputs: Vec<(Op::InputKey, LocalValueId)>,
    tangent_outputs: Vec<Option<LocalValueId>>,
    linear_primals: Vec<Option<ValueKey<Op>>>,
}

fn linearize<Op: Primitive>(
    view: &ResolvedView<Op>,
    outputs: &[ValueKey<Op>],
    wrt: &[Op::InputKey],
) -> LinearizedGraph<Op>;
```

Each call to `linearize` receives a unique `DiffPassId` (monotonically
increasing counter). Tangent input keys are generated via
`wrt_key.tangent_of(pass_id)` (see `ADKey` in the primitive AD contract).

Algorithm:

1. Traverse the reachable logical DAG in topological order.
2. Seed tangent inputs for the requested primal `InputKey`s
   (keys generated via `ADKey::tangent_of`).
3. For each reachable primitive, call `Op::linearize`.
4. Emit tangent-flow nodes into the strictly-linear graph.
5. Put residual values into the residual graph and reference them through
   `External(ValueKey)`.
6. Skip unreachable tangent flow with zero propagation.

This is the graph-level analogue of JAX building a jaxpr whose linearized
body is itself a composition of primitives.

### `linear_transpose`

Consumes a linearized graph and produces another with active inputs and
outputs reversed.

```rust
fn linear_transpose<Op: Primitive>(
    linear: &LinearizedGraph<Op>,
) -> LinearizedGraph<Op>;
```

Traverses the strictly-linear graph in reverse topological order and, for each
op node, calls `Op::transpose_rule` to obtain the local transposed
contribution.

`transpose_rule` receives typed inputs:

```rust
enum PrimitiveTransposeInput<Op> {
    Residual(ValueKey<Op>),
    Linear { key: ValueKey<Op>, primal: Option<ValueKey<Op>> },
}
```

Residual inputs are ordinary fixed operands. Linear inputs belong to tangent
flow and must not be retained as tensor operands in the transposed graph. When
`primal` is available, downstream primitive sets may use it for metadata,
runtime shape sources, or fixed coefficients that are independent of the
tangent flow. A `Linear { primal: None, .. }` input must not be collapsed into
a residual `ValueRef`.

Transpose accumulation must use global identity: when multiple reverse
contributions flow back to the same original tangent node, bucket by the
**global key of that tangent value**, not by a graph-local id.

Fan-out accumulation is handled internally by `linear_transpose`, not by an
explicit `Dup` primitive. When multiple cotangents flow to the same
`ValueKey`, `linear_transpose` accumulates them by emitting `Op::add()` nodes.
This follows
the JAX approach where `add_jaxvals` is built into the linear_transpose pass
rather than expressed as a separate primitive in the graph. Downstream
primitive implementors do not need to implement `Dup`.

### Transpose algorithm

```rust
fn linear_transpose<Op: Primitive>(linear: &LinearizedGraph<Op>) -> LinearizedGraph<Op> {
    let mut builder = SplitGraphBuilder::new();
    let mut ct_env: HashMap<ValueKey<Op>, LocalValueId> = HashMap::new();

    // 1. Seed cotangent outputs
    for (out_key, ct_input_id) in cotangent_seeds {
        ct_env.insert(out_key, ct_input_id);
    }

    // 2. Reverse topological traversal
    for op_node in linear.as_graph().operations().iter().rev() {
        // Look up cotangent for this op's outputs
        let ct_outs: Vec<Option<LocalValueId>> = op_node.outputs.iter()
            .map(|out_id| ct_env.get(&global_key(out_id)).copied())
            .collect();

        let rule_inputs = op_node.inputs.iter()
            .map(|input| classify_transpose_input(linear, input))
            .collect::<Vec<_>>();

        // Delegate to per-op linear_transpose rule
        let ct_ins = op_node.operation.transpose_rule(
            &mut builder, &ct_outs, &rule_inputs, &op_node.role,
        );

        // 3. Accumulate cotangents by ValueKey
        for (input, ct_in) in rule_inputs.iter().zip(ct_ins) {
            if let Some(ct) = ct_in {
                let key = input.key();
                match ct_env.entry(key) {
                    Vacant(e)  => { e.insert(ct); }
                    Occupied(e) => {
                        // Fan-out: add Add node for accumulation
                        let existing = *e.get();
                        let sum = builder.add_operation(
                            Op::add(),
                            vec![ValueRef::Local(existing), ValueRef::Local(ct)],
                            OperationRole::Linearized { active_mask: vec![true, true] },
                        );
                        *e.into_mut() = sum[0];
                    }
                }
            }
        }
    }
    // Build transposed LinearizedGraph from split builder + ct_env
}
```

The accumulation `Add` nodes emitted during linear_transpose are **normal graph
nodes** in the transposed graph. They carry
`OperationRole::Linearized { active_mask: [active, active] }` and participate in
subsequent AD transforms like any other node. This is why `Primitive`
includes `add()`: `tidu` needs one generic way to construct those
accumulation nodes.

### Worked example: linear_transpose of `f(x) = (x+x)*x`

Primal graph F0:

```text
p0 = Input(x)
p1 = Add(p0, p0)          // 2x
p2 = Mul(p1, p0)          // 2x²
```

Linearize wrt x → L1:

```text
t0 = Input(dx)
t1 = Add(t0, t0)                          Linear{[active, active]}   // 2·dx
t2 = Mul(External(p1), Local(t0))          Linear{[fixed, active]}    // 2x·dx
t3 = Mul(Local(t1), External(p0))          Linear{[active, fixed]}    // 2·dx·x
t4 = Add(Local(t2), Local(t3))             Linear{[active, active]}   // 4x·dx
```

Transpose L1, seed ct_y. `ct_env` state after each step:

```text
seed:  ct_env = { t4.key → c0 }              c0 = Input(ct_y)

Reverse t4 = Add(t2, t3):
  Add linear_transpose → ct_t2 = c0, ct_t3 = c0
  ct_env = { t4.key → c0, t2.key → c0, t3.key → c0 }

Reverse t3 = Mul(t1, p0) [active, fixed]:
  Mul linear_transpose wrt active → ct_t1 = Mul(p0, c0)
  c1 = Mul(External(p0), Local(c0))
  ct_env = { ..., t1.key → c1 }

Reverse t2 = Mul(p1, t0) [fixed, active]:
  Mul linear_transpose wrt active → ct_t0 = Mul(p1, c0)
  c2 = Mul(External(p1), Local(c0))
  ct_env = { ..., t0.key → c2 }                         ← 1st entry for t0

Reverse t1 = Add(t0, t0) [active, active]:
  Add linear_transpose → both inputs get ct_t1 = c1
  Left input t0:  ct_env[t0.key] = c2 (existing) → add Add
                   c3 = Add(c2, c1)                      ← accumulation #1
                   ct_env[t0.key] = c3
  Right input t0: ct_env[t0.key] = c3 (existing) → add Add
                   c4 = Add(c3, c1)                      ← accumulation #2
                   ct_env[t0.key] = c4
```

Transposed graph T1:

```text
c0 = Input(ct_y)
c1 = Mul(External(p0), Local(c0))          // x · ct_y
c2 = Mul(External(p1), Local(c0))          // 2x · ct_y
c3 = Add(Local(c2), Local(c1))             // accumulation Add #1
c4 = Add(Local(c3), Local(c1))             // accumulation Add #2
output: c4 = 2x·ct_y + x·ct_y + x·ct_y = 4x·ct_y  ✓  (f'=4x)
```

Note: c1 is referenced by both c3 and c4 — fan-out in the transposed
graph itself. This is handled correctly by subsequent transforms
(see next section).

---

## III. Higher-Order AD and Accumulation Correctness

### FoR: linearize the transposed graph

The transposed graph T1 computes `ct_x = 4x · ct_y` as a function
of `(x, ct_y)`. To get the second derivative (FoR), linearize T1
wrt x via `resolve([F0, T1])`.

Primal tangents:

```text
dp0 = dx2
dp1 = d(Add(p0, p0)) = Add(dx2, dx2) = 2·dx2
```

Tangent of each T1 node (dc0 = None because ct_y does not depend on x):

```text
dc1 = d(Mul(p0, c0)):  dp0 = dx2, dc0 = None
    → Mul(dx2, c0) = dx2 · ct_y

dc2 = d(Mul(p1, c0)):  dp1 = 2·dx2, dc0 = None
    → Mul(Add(dx2, dx2), c0) = 2·dx2 · ct_y

dc3 = d(Add(c2, c1)):  ← accumulation Add, linearized normally
    → Add(dc2, dc1) = 2·dx2·ct_y + dx2·ct_y = 3·dx2·ct_y

dc4 = d(Add(c3, c1)):  ← accumulation Add, linearized normally
    → Add(dc3, dc1) = 3·dx2·ct_y + dx2·ct_y = 4·dx2·ct_y
```

Result: `dc4 = 4·dx2·ct_y` → f'' = 4 ✓ (f=2x², f'=4x, f''=4)

### Why this is self-consistent

1. **Accumulation produces normal graph nodes.** The `Add` nodes emitted
   during linear_transpose carry `role=Linearized{[active, active]}`. They have the
   same `linearize` and `transpose_rule` as any other `Add` node.

2. **Fan-out in transposed graphs is safe.** c1 is used by both c3
   and c4. In the forward direction (FoR), `dc1` feeds into both
   `dc3` and `dc4`'s linearize — this is just multiple references to
   the same tangent value, which is always correct in forward mode.

3. **Further linear_transpose (RoR) also works.** If we linear_transpose the FoR
   graph, `dc1` being used twice would cause two cotangents to flow
   to `dc1`'s key. The same `HashMap` accumulation mechanism handles
   this recursively.

4. **No special-casing at any level.** The `linearize` and `linear_transpose`
   algorithms are uniform: `linearize` calls `Op::linearize` for each
   node, `linear_transpose` calls `Op::transpose_rule` and accumulates. The
   accumulation `Add` is indistinguishable from any other `Add` in the
   graph.

---

## IV. Typical Pipelines

```text
JVP:
  build -> resolve -> linearize -> materialize_merge -> compile -> eval

VJP (grad):
  build -> resolve -> linearize -> linear_transpose -> materialize_merge -> compile -> eval

2nd directional derivative (FoF):
  build -> resolve -> linearize -> resolve -> linearize
       -> materialize_merge -> compile -> eval

HVP (FoR = jvp(vjp(f))):
  build -> resolve -> linearize -> linear_transpose -> resolve -> linearize
       -> materialize_merge -> compile -> eval

n-th derivative:
  build -> (resolve -> linearize) x n -> [linear_transpose] -> materialize_merge -> compile -> eval
```

`resolve`, `materialize_merge`, `compile`, `eval` are provided by
`computegraph-rs`. `tidu-rs` only adds `linearize` and `linear_transpose`.

---

## V. Linear Nodes

The linear graph uses the same primitive set as the primal graph. There is no
dedicated `Scale` primitive.

```text
Mul(a, dx)   role=Linearized { active_mask=[fixed, active] }
Add(dx, dy)  role=Linearized { active_mask=[active, active] }
Exp(x)       role=Primary
```

The linearization of `Exp(x)` emits:

```text
Mul(External(exp(x)), dx) role=Linearized { active_mask=[fixed, active] }
```

Active mask is part of identity (`OperationRole::Linearized` vs `Primary`). Nodes that
evaluate the same way but linear_transpose differently must not alias.

---

## VI. Design Boundaries

```text
tidu-rs owns:
  - Primitive trait
  - linearize (JVP transform)
  - linear_transpose (reverse linear flow)
  - LinearizedGraph split between strictly-linear and residual graphs

tidu-rs does NOT own:
  - graph infrastructure → computegraph-rs
  - concrete primitives → downstream (tenferro-rs)
```
