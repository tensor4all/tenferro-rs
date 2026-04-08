# v2 tidu-rs Design

**Date:** 2026-04-03
**Status:** Draft
**Repo:** tidu-rs
**Parent:** `../index.md`
**Depends on:** `computegraph-rs`, `chainrules-rs`

---

## I. Purpose

`tidu-rs` provides AD-specific graph transforms (`differentiate`, `transpose`)
that are fully generic over `Op: PrimitiveOp`. It owns no graph infrastructure
(that belongs to `computegraph-rs`) and references no specific primitives.

Among the JAX concepts, `differentiate` is the closest analogue to
`jax.linearize`: it traverses a primal computation and builds a new linear
computation by calling each primitive's local linearization rule. The output is
not StableHLO and not a backend kernel plan; it is another fragment composed of
the same downstream primitive vocabulary.

---

## II. Transforms

### `differentiate`

Consumes a resolved view and returns a new linear fragment (JVP).

```rust
use computegraph::{ResolvedView, GlobalValKey, LocalValId, Fragment};
use chainrules::PrimitiveOp;

struct LinearFragment<Op> {
    fragment: Fragment<Op>,
    tangent_inputs: Vec<(Op::InputKey, LocalValId)>,
    tangent_outputs: Vec<Option<LocalValId>>,
}

fn differentiate<Op: PrimitiveOp>(
    view: &ResolvedView<Op>,
    outputs: &[GlobalValKey<Op>],
    wrt: &[Op::InputKey],
) -> LinearFragment<Op>;
```

Each call to `differentiate` receives a unique `DiffPassId` (monotonically
increasing counter). Tangent input keys are generated via
`wrt_key.tangent_of(pass_id)` (see `ADKey` trait in `chainrules.md`).

Algorithm:

1. Traverse the reachable logical DAG in topological order.
2. Seed tangent inputs for the requested primal `InputKey`s
   (keys generated via `ADKey::tangent_of`).
3. For each reachable primitive, call `Op::linearize`.
4. Emit new local linear nodes into the new fragment.
5. Reference primal values through `External(GlobalValKey)`.
6. Skip unreachable tangent flow with zero propagation.

This is the fragment-level analogue of JAX building a jaxpr whose linearized
body is itself a composition of primitives.

### `transpose`

Consumes a linear fragment and produces another with active inputs and outputs
reversed.

```rust
fn transpose<Op: PrimitiveOp>(
    linear: &LinearFragment<Op>,
) -> LinearFragment<Op>;
```

Traverses the linear fragment in reverse topological order and, for each op
node, calls `Op::transpose_rule` to obtain the local transposed contribution.

Transpose accumulation must use global identity: when multiple reverse
contributions flow back to the same original tangent node, bucket by the
**global key of that tangent value**, not by a fragment-local id.

Fan-out accumulation is handled internally by `transpose`, not by an
explicit `Dup` primitive. When multiple cotangents flow to the same
`GlobalValKey`, `transpose` accumulates them by emitting `Op::add()` nodes.
This follows
the JAX approach where `add_jaxvals` is built into the transpose pass
rather than expressed as a separate primitive in the graph. Downstream
primitive implementors do not need to implement `Dup`.

### Transpose algorithm

```rust
fn transpose<Op: PrimitiveOp>(linear: &LinearFragment<Op>) -> LinearFragment<Op> {
    let mut builder = FragmentBuilder::new();
    let mut ct_env: HashMap<GlobalValKey<Op>, LocalValId> = HashMap::new();

    // 1. Seed cotangent outputs
    for (out_key, ct_input_id) in cotangent_seeds {
        ct_env.insert(out_key, ct_input_id);
    }

    // 2. Reverse topological traversal
    for op_node in linear.fragment.ops().iter().rev() {
        // Look up cotangent for this op's outputs
        let ct_outs: Vec<Option<LocalValId>> = op_node.outputs.iter()
            .map(|out_id| ct_env.get(&global_key(out_id)).copied())
            .collect();

        // Delegate to per-op transpose rule
        let ct_ins = op_node.op.transpose_rule(
            &mut builder, &ct_outs, &op_node.inputs, &op_node.mode,
        );

        // 3. Accumulate cotangents by GlobalValKey
        for (input, ct_in) in op_node.inputs.iter().zip(ct_ins) {
            if let Some(ct) = ct_in {
                let key = global_key_of(input);
                match ct_env.entry(key) {
                    Vacant(e)  => { e.insert(ct); }
                    Occupied(e) => {
                        // Fan-out: emit Add node for accumulation
                        let existing = *e.get();
                        let sum = builder.add_op(
                            Op::add(),
                            vec![ValRef::Local(existing), ValRef::Local(ct)],
                            OpMode::Linear { active_mask: vec![true, true] },
                        );
                        *e.into_mut() = sum[0];
                    }
                }
            }
        }
    }
    // Build transposed LinearFragment from builder + ct_env
}
```

The accumulation `Add` nodes emitted during transpose are **normal graph
nodes** in the transposed fragment. They carry
`OpMode::Linear { active_mask: [active, active] }` and participate in
subsequent AD transforms like any other node. This is why `PrimitiveOp`
includes `add()`: `tidu` needs one generic way to construct those
accumulation nodes.

### Worked example: transpose of `f(x) = (x+x)*x`

Primal fragment F0:

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
  Add transpose → ct_t2 = c0, ct_t3 = c0
  ct_env = { t4.key → c0, t2.key → c0, t3.key → c0 }

Reverse t3 = Mul(t1, p0) [active, fixed]:
  Mul transpose wrt active → ct_t1 = Mul(p0, c0)
  c1 = Mul(External(p0), Local(c0))
  ct_env = { ..., t1.key → c1 }

Reverse t2 = Mul(p1, t0) [fixed, active]:
  Mul transpose wrt active → ct_t0 = Mul(p1, c0)
  c2 = Mul(External(p1), Local(c0))
  ct_env = { ..., t0.key → c2 }                         ← 1st entry for t0

Reverse t1 = Add(t0, t0) [active, active]:
  Add transpose → both inputs get ct_t1 = c1
  Left input t0:  ct_env[t0.key] = c2 (existing) → emit Add
                   c3 = Add(c2, c1)                      ← accumulation #1
                   ct_env[t0.key] = c3
  Right input t0: ct_env[t0.key] = c3 (existing) → emit Add
                   c4 = Add(c3, c1)                      ← accumulation #2
                   ct_env[t0.key] = c4
```

Transposed fragment T1:

```text
c0 = Input(ct_y)
c1 = Mul(External(p0), Local(c0))          // x · ct_y
c2 = Mul(External(p1), Local(c0))          // 2x · ct_y
c3 = Add(Local(c2), Local(c1))             // accumulation Add #1
c4 = Add(Local(c3), Local(c1))             // accumulation Add #2
output: c4 = 2x·ct_y + x·ct_y + x·ct_y = 4x·ct_y  ✓  (f'=4x)
```

Note: c1 is referenced by both c3 and c4 — fan-out in the transposed
fragment itself. This is handled correctly by subsequent transforms
(see next section).

---

## III. Higher-Order AD and Accumulation Correctness

### FoR: differentiate the transposed fragment

The transposed fragment T1 computes `ct_x = 4x · ct_y` as a function
of `(x, ct_y)`. To get the second derivative (FoR), differentiate T1
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
   during transpose carry `mode=Linear{[active, active]}`. They have the
   same `linearize` and `transpose_rule` as any other `Add` node.

2. **Fan-out in transposed fragments is safe.** c1 is used by both c3
   and c4. In the forward direction (FoR), `dc1` feeds into both
   `dc3` and `dc4`'s linearize — this is just multiple references to
   the same tangent value, which is always correct in forward mode.

3. **Further transpose (RoR) also works.** If we transpose the FoR
   fragment, `dc1` being used twice would cause two cotangents to flow
   to `dc1`'s key. The same `HashMap` accumulation mechanism handles
   this recursively.

4. **No special-casing at any level.** The `differentiate` and `transpose`
   algorithms are uniform: `differentiate` calls `Op::linearize` for each
   node, `transpose` calls `Op::transpose_rule` and accumulates. The
   accumulation `Add` is indistinguishable from any other `Add` in the
   graph.

---

## IV. Typical Pipelines

```text
JVP:
  build -> resolve -> differentiate -> materialize_merge -> compile -> eval

VJP (grad):
  build -> resolve -> differentiate -> transpose -> materialize_merge -> compile -> eval

2nd directional derivative (FoF):
  build -> resolve -> differentiate -> resolve -> differentiate
       -> materialize_merge -> compile -> eval

HVP (FoR = jvp(vjp(f))):
  build -> resolve -> differentiate -> transpose -> resolve -> differentiate
       -> materialize_merge -> compile -> eval

n-th derivative:
  build -> (resolve -> differentiate) x n -> [transpose] -> materialize_merge -> compile -> eval
```

`resolve`, `materialize_merge`, `compile`, `eval` are provided by
`computegraph-rs`. `tidu-rs` only adds `differentiate` and `transpose`.

---

## V. Linear Nodes

The linear graph uses the same primitive set as the primal graph. There is no
dedicated `Scale` primitive.

```text
Mul(a, dx)   mode=Linear { active_mask=[fixed, active] }
Add(dx, dy)  mode=Linear { active_mask=[active, active] }
Exp(x)       mode=Primal
```

The linearization of `Exp(x)` emits:

```text
Mul(External(exp(x)), dx) mode=Linear { active_mask=[fixed, active] }
```

Active mask is part of identity (`OpMode::Linear` vs `Primal`). Nodes that
evaluate the same way but transpose differently must not alias.

---

## VI. Design Boundaries

```text
tidu-rs owns:
  - differentiate (JVP transform)
  - transpose (reverse linear flow)
  - LinearFragment data structure

tidu-rs does NOT own:
  - graph infrastructure → computegraph-rs
  - PrimitiveOp trait → chainrules-rs
  - concrete primitives → downstream (tenferro-rs)
```
