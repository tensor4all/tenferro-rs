# Automatic Differentiation (Next Implementation)

This document defines the next AD architecture discussed for tenferro-rs.

It is a target design. Current behavior is documented in
[autodiff-current.md](./autodiff-current.md).

For math derivations, see [AD Formula Notes](../AD/index.md).

## Design Goals

1. Keep `tenferro-einsum` explicit AD interfaces as-is.
2. Provide torch-like UX in higher layers (`dyadtensor` and similar wrappers).
3. Keep `chainrules-core` minimal and stable for external custom types.
4. Support both homogeneous and heterogeneous AD graphs:
   - `Tape<V>` for monomorphic, type-safe fast path
   - `DynTape` for mixed custom-value graphs
5. Expose explicit `backward` options:
   - `retain_graph`
   - `create_graph`

## Fixed Decisions

- `chainrules-core` remains math contracts only:
  `Differentiable`, `ReverseRule<V>`, `ForwardRule<V>`, errors.
- Torch-like API lives in `chainrules` (not in `chainrules-core`).
- `Tape<V>` and `DynTape` co-exist and are both public.
- `tenferro-einsum` keeps explicit interfaces (`tracked_einsum`, `dual_einsum`,
  `einsum_rrule`, `einsum_frule`, `einsum_hvp`).
- Autograd context is shared by `Arc<Mutex<_>>`.
- `backward` and `hvp` accumulate into stored gradients; `zero_grad` clears.
- Unsupported modes return typed errors (`ModeNotSupported`).

## Public API Sketch

```rust
pub struct BackwardOptions<V: Differentiable> {
    pub retain_graph: bool,
    pub create_graph: bool,
    pub grad: Option<V::Tangent>,
}

impl<V: Differentiable> Default for BackwardOptions<V> {
    fn default() -> Self {
        Self {
            retain_graph: false,
            create_graph: false,
            grad: None,
        }
    }
}

pub struct Variable<V: Differentiable> {
    // value, node id, and shared context (internal)
}

impl<V: Differentiable> Variable<V> {
    pub fn requires_grad(&self) -> bool;
    pub fn requires_grad_(self, enabled: bool) -> AdResult<Self>;
    pub fn with_tangent_(self, tangent: V::Tangent) -> AdResult<Self>;
    pub fn tangent(&self) -> Option<&V::Tangent>;
    pub fn detach(&self) -> Self;

    pub fn backward(&self, options: BackwardOptions<V>) -> AdResult<()>;
    pub fn backward_hvp(&self, options: BackwardOptions<V>) -> AdResult<()>;
    pub fn grad(&self) -> Option<V::Tangent>;
    pub fn hvp(&self) -> Option<V::Tangent>;
    pub fn zero_grad(&self) -> AdResult<()>;
}

pub mod autograd {
    pub fn grad<V: Differentiable>(
        outputs: &[&Variable<V>],
        inputs: &[&Variable<V>],
        options: BackwardOptions<V>,
    ) -> AdResult<Vec<Variable<V>>>;
}
```

## Review Guide

Use this order when reviewing:

1. Read **Terminology Bridge** if you are new to PyTorch-style AD terms.
2. Read **Usage Examples 1-4** to understand same-type flows (`Variable<V>`).
3. Read **Usage Example 7** only if you need heterogeneous custom types.
4. Read **Error Contract** and **Context Merge Rule** as behavioral constraints.

## Terminology Bridge

| Concept | This design | PyTorch term | Meaning |
|---------|-------------|--------------|---------|
| Differentiable value | `Variable<V>` | `Tensor` (autograd-enabled) | Value carrying graph linkage and AD metadata |
| Leaf variable | `requires_grad_(true)` at creation edge | Leaf tensor | Gradient storage endpoint |
| Backward seed | `BackwardOptions { grad: Some(...) }` | `gradient` argument to `backward` | Cotangent seed for non-scalar outputs |
| Reuse graph | `retain_graph = true` | `retain_graph=True` | Keep graph after backward |
| Build grad graph | `create_graph = true` | `create_graph=True` | Make first-order gradients differentiable |
| Graph break | `detach()` | `detach()` | Stop tracking reverse and forward paths |
| JVP direction | `with_tangent_(v)` | forward AD tangent | Seed forward-mode direction |
| HVP | `backward_hvp(...)` | composed `jvp(grad(f))` pattern | Hessian-vector product |

## Usage Examples

In examples below, `AdError` is a short alias for `chainrules_core::AutodiffError`.

### 1. Same Type for Primal / Forward / Backward / HVP

```rust
type ADMyScalar = Variable<MyScalar>;

// primal only
let x0 = ADMyScalar::new(MyScalar(3.0));
let y0 = square(&x0)?;
assert_eq!(y0.value().0, 9.0);

// forward (JVP)
let x1 = ADMyScalar::new(MyScalar(3.0)).with_tangent_(MyScalar(1.0))?;
let y1 = square(&x1)?;
assert_eq!(y1.value().0, 9.0);
assert_eq!(y1.tangent().unwrap().0, 6.0);

// backward
let x2 = ADMyScalar::new(MyScalar(3.0)).requires_grad_(true)?;
let y2 = square(&x2)?;
y2.backward(BackwardOptions::default())?;
assert_eq!(x2.grad().unwrap().0, 6.0);

// hvp
let x3 = ADMyScalar::new(MyScalar(3.0))
    .requires_grad_(true)?
    .with_tangent_(MyScalar(5.0))?;
let y3 = square(&x3)?;
y3.backward_hvp(BackwardOptions::default())?;
assert_eq!(x3.hvp().unwrap().0, 10.0);
```

### 2. `retain_graph = true`

```rust
let x = ADMyScalar::new(MyScalar(2.0)).requires_grad_(true)?;
let loss = square(&x)?; // f(x) = x^2

loss.backward(BackwardOptions {
    retain_graph: true,
    ..Default::default()
})?;

// Same graph reused without re-running forward construction.
loss.backward(BackwardOptions {
    retain_graph: true,
    ..Default::default()
})?;

// grad is accumulated by default
let gx = x.grad().unwrap();
assert_eq!(gx.0, 8.0); // 2*2 + 2*2
```

### 3. `detach` (PyTorch-compatible graph break)

```rust
let x = ADMyScalar::new(MyScalar(3.0))
    .requires_grad_(true)?
    .with_tangent_(MyScalar(1.0))?;

let y = square(&x)?;
let y_detached = y.detach();

assert!(!y_detached.requires_grad());
assert!(y_detached.tangent().is_none()); // forward tangent is also cleared
```

### 4. `create_graph = true` (higher-order differentiation)

```rust
let x = ADMyScalar::new(MyScalar(3.0)).requires_grad_(true)?;
let loss = cube(&x)?; // f(x) = x^3

loss.backward(BackwardOptions {
    retain_graph: true,
    create_graph: true,
    ..Default::default()
})?;

// gx = df/dx = 3x^2 and remains graph-connected
let gx = autograd::grad(
    &[&loss],
    &[&x],
    BackwardOptions {
        retain_graph: true,
        create_graph: true,
        ..Default::default()
    },
)?[0].clone();

// grad(gx, x) = d2f/dx2 at x=3
let gxx = autograd::grad(&[&gx], &[&x], BackwardOptions::default())?[0].clone();
assert_eq!(gxx.value().0, 18.0);
```

If `gx` is non-scalar, supply a seed gradient via `BackwardOptions { grad: Some(...) }`
in the second `autograd::grad` call.

```rust
let gx = vector_grad(&x)?; // non-scalar gx
let seed = gx.ones_like();
let gxx = autograd::grad(
    &[&gx],
    &[&x],
    BackwardOptions {
        grad: Some(seed),
        ..Default::default()
    },
)?[0].clone();
# let _ = gxx;
```

### 5. Minimal End-to-End Flow (non-PyTorch readers)

```rust
type ADMyScalar = Variable<MyScalar>;

// Step 1: create value
let x = ADMyScalar::new(MyScalar(3.0)).requires_grad_(true)?;

// Step 2: run forward computation
let loss = cube(&x)?; // scalar output

// Step 3: run backward
loss.backward(BackwardOptions::default())?;

// Step 4: read first-order gradient
let gx = x.grad().unwrap();
assert_eq!(gx.0, 27.0); // d(x^3)/dx at x=3

// Step 5: clear accumulation before next step
x.zero_grad()?;
```

### 6. Non-Scalar Output with Seed Gradient

```rust
let y = vector_output(&x)?; // y is non-scalar
let seed = y.ones_like();

y.backward(BackwardOptions {
    grad: Some(seed),
    ..Default::default()
})?;
```

### 7. Heterogeneous Custom Types with `DynTape`

```rust
let tape = DynTape::new();
let s = tape.leaf(MyScalar(2.0));
let v = tape.leaf(MyVec2 { x: 3.0, y: -1.0 });

let y = mul_scalar_vec(&tape, &s, &v)?;  // MyScalar x MyVec2 -> MyVec2
let loss = squared_norm(&tape, &y)?;     // MyVec2 -> MyScalar

let result = tape.hvp(&loss)?;
let grad_s = result.gradients.get::<MyScalar>(s.node_id()).unwrap();
let hvp_s = result.hvp.get::<MyScalar>(s.node_id()).unwrap();
let _ = (grad_s, hvp_s);
```

## Common Failure Cases

### Non-scalar backward without seed gradient

```rust
let y = vector_output(&x)?; // non-scalar
let err = y.backward(BackwardOptions::default()).unwrap_err();
assert!(matches!(err, AdError::InvalidArgument(_)));
```

### Mixed contexts in one operation

```rust
type ADMyScalar = Variable<MyScalar>;
let a = ADMyScalar::new(MyScalar(1.0)).requires_grad_(true)?;
let b = ADMyScalar::new(MyScalar(2.0)).requires_grad_(true)?; // different context
let err = add(&a, &b).unwrap_err();
assert!(matches!(err, AdError::InvalidArgument(_)));
```

### HVP on rule without tangent support

```rust
let x = ADMyScalar::new(MyScalar(3.0))
    .requires_grad_(true)?
    .with_tangent_(MyScalar(1.0))?;
let y = custom_op_without_hvp_rule(&x)?;
let err = y.backward_hvp(BackwardOptions::default()).unwrap_err();
assert!(matches!(err, AdError::ModeNotSupported { .. }));
```

## Context Merge Rule

For multi-input operations, context selection is deterministic:

1. All inputs `ctx=None` -> output `ctx=None`.
2. Exactly one shared context -> use it.
3. Multiple contexts but all same `Arc` -> use it.
4. Different contexts mixed -> return `InvalidArgument`.

## Error Contract

- Unsupported JVP path -> `ModeNotSupported { mode: "frule", ... }`
- Unsupported HVP path -> `ModeNotSupported { mode: "hvp", ... }`
- Non-scalar backward without `grad` seed -> `InvalidArgument`
- Context mismatch across operands -> `InvalidArgument`

## Compatibility With Current APIs

- `tenferro-einsum` explicit AD functions remain stable and unchanged.
- `Tape<V>` remains public and first-class.
- `DynTape` is public for heterogeneous user-defined type graphs.
- Torch-like wrappers are additive and do not remove existing explicit APIs.
