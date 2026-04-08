# chainrules-rs Design

**Repo:** chainrules-rs
**Parent:** `../index.md`
**Depends on:** `computegraph-rs`

---

## I. Purpose

`chainrules-rs` defines the AD trait (`PrimitiveOp`) that extends
`computegraph::GraphOp` with a cotangent-accumulation constructor plus
linearization and transpose rules. It contains no graph infrastructure and no
concrete primitives.

This is the counterpart of the AD behavior that JAX stores in
`primitive_jvps` and `primitive_transposes`. The information is the same kind
of information, but the representation is different:

- JAX: global registries keyed by primitive object
- tenferro: methods on the concrete primitive type itself

---

## II. PrimitiveOp Trait

`PrimitiveOp` extends `GraphOp` with `add()` (cotangent accumulation
constructor), `linearize` (JVP rule), and `transpose_rule` (VJP rule).

Canonical trait signature: [`../spec/ad-contract.md`](../spec/ad-contract.md).

`add()` returns the primitive used by `tidu::transpose` when multiple
cotangent contributions flow to the same `GlobalValKey`. This keeps fan-out
accumulation inside the generic transpose pass without requiring a separate
built-in `Dup` or `Add` primitive in `tidu`.

---

## III. Linearization Rules

A primitive's `linearize` must be linear in tangent inputs. It may:

- reference primal inputs or outputs through `External(GlobalValKey)`
- emit primitives in `OpMode::Linear`
- emit `Conj` when required by transpose semantics

It must not introduce nonlinear dependence on tangent inputs.

The intended mental model is close to JAX `linearize`:

- JAX `linearize` applies a primitive's JVP rule and emits a new composition of
  JAX primitives in a jaxpr
- `PrimitiveOp::linearize` emits a new composition of downstream concrete
  primitives into a fragment

---

## IV. Transpose Rules

A primitive's `transpose_rule` receives cotangent outputs and produces
cotangent inputs. It must only emit primitives that themselves implement
`PrimitiveOp`.

When transpose encounters fan-out, `tidu` accumulates multiple reverse
contributions by emitting `PrimitiveOp::add()` nodes. So every downstream
primitive set used with `tidu` must provide an addition primitive suitable for
cotangent accumulation.

---

## V. ADKey Trait

`chainrules-rs` defines the `ADKey` trait that constrains `InputKey` for AD
use. `tidu-rs` uses this trait to generate tangent input keys during
`differentiate`.

```rust
pub type DiffPassId = u64;

pub trait ADKey: Clone + Debug + Hash + Eq + Send + Sync + 'static {
    /// Create a tangent input key derived from this key.
    /// `pass` is a unique identifier for the `differentiate` call.
    fn tangent_of(&self, pass: DiffPassId) -> Self;
}
```

`PrimitiveOp` requires `Self::InputKey: ADKey`
(see [`../spec/ad-contract.md`](../spec/ad-contract.md) for the canonical
`PrimitiveOp` trait signature).

The concrete implementation of `ADKey` is the downstream implementor's
choice. A typical pattern is a recursive enum:

```rust
// tenferro-rs
enum TensorInputKey {
    User(String),
    Tangent { of: Box<TensorInputKey>, pass: DiffPassId },
}
```

This gives debuggable keys like
`Tangent { of: Tangent { of: User("x"), pass: 1 }, pass: 3 }` for
higher-order AD.

---

## VI. Closure Responsibility

`chainrules-rs` defines the contract but does not enforce closure. The
downstream implementor (e.g. tenferro-rs) is responsible for ensuring that
the set of primitives reachable through `linearize` and `transpose_rule`
is closed — i.e., every emitted op also implements `PrimitiveOp`.

---

## VII. Design Boundaries

```text
chainrules-rs owns:
  - PrimitiveOp trait (`add` + `linearize` + `transpose_rule`)

chainrules-rs does NOT own:
  - graph infrastructure → computegraph-rs
  - AD transforms (differentiate, transpose) → tidu-rs
  - concrete primitives → downstream (tenferro-rs)
```
