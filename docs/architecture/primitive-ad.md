# Primitive AD Contract

**Repo:** tidu-rs
**Parent:** `../index.md`
**Depends on:** `computegraph-rs`

---

## I. Purpose

`tidu-rs` defines the AD trait (`Primitive`) that extends
`computegraph::GraphOperation` with a cotangent-accumulation constructor plus
JVP and transpose rules. It contains no concrete primitives.

This is the counterpart of the AD behavior that JAX stores in
`primitive_jvps` and `primitive_transposes`. The information is the same kind
of information, but the representation is different:

- JAX: global registries keyed by primitive object
- tenferro: methods on the concrete primitive type itself

---

## II. Primitive Trait

`Primitive` extends `GraphOperation` with `add()` (cotangent accumulation
constructor), `linearize` (JVP rule), and `transpose_rule` (VJP rule).

Canonical trait signature: [`../spec/ad-contract.md`](../spec/ad-contract.md).

`add()` returns the primitive used by `tidu::linear_transpose` when multiple
cotangent contributions flow to the same `ValueKey`. This keeps fan-out
accumulation inside the generic linear_transpose pass without requiring a separate
built-in `Dup` or `Add` primitive in `tidu`.

---

## III. Linearization Rules

A primitive's `linearize` must be linear in tangent inputs. It may:

- reference primal inputs or outputs through `External(ValueKey)`
- add primitives in `OperationRole::Linearized`
- add `Conj` when required by linear_transpose semantics

It must not introduce nonlinear dependence on tangent inputs.

The intended mental model is close to JAX `linearize`:

- JAX `linearize` applies a primitive's JVP rule and emits a new composition of
  JAX primitives in a jaxpr
- `Primitive::jvp_rule` emits a new composition of downstream concrete
  primitives into a graph

---

## IV. Transpose Rules

A primitive's `transpose_rule` receives cotangent outputs and produces
cotangent inputs. It must only add primitives that themselves implement
`Primitive`.

When linear_transpose encounters fan-out, `tidu` accumulates multiple reverse
contributions by emitting `Primitive::add()` nodes. So every downstream
primitive set used with `tidu` must provide an addition primitive suitable for
cotangent accumulation.

---

## V. ADKey Trait

`tidu-rs` defines the `ADKey` trait that constrains `InputKey` for AD
use. `tidu-rs` uses this trait to generate tangent input keys during
`linearize`.

```rust
pub type DiffPassId = u64;

pub trait ADKey: Clone + Debug + Hash + Eq + Send + Sync + 'static {
    /// Create a tangent input key derived from this key.
    /// `pass` is a unique identifier for the `linearize` call.
    fn tangent_of(&self, pass: DiffPassId) -> Self;
}
```

`Primitive` requires `Self::InputKey: ADKey`
(see [`../spec/ad-contract.md`](../spec/ad-contract.md) for the canonical
`Primitive` trait signature).

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

`tidu-rs` defines the contract but does not enforce closure. The
downstream implementor (e.g. tenferro-rs) is responsible for ensuring that
the set of primitives reachable through `linearize` and `transpose_rule`
is closed — i.e., every emitted op also implements `Primitive`.

---

## VII. Design Boundaries

```text
tidu-rs owns:
  - Primitive trait (`add` + `jvp_rule` + `transpose_rule`)
  - AD transforms (`linearize`, `linear_transpose`)

tidu-rs does NOT own:
  - graph infrastructure → computegraph-rs
  - concrete primitives → downstream (tenferro-rs)
```
