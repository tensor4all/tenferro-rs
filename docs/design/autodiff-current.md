# Automatic Differentiation (Current Implementation)

This document describes the AD architecture that is currently implemented in the
workspace (`extern/chainrules-core`, `extern/chainrules`, `tenferro-einsum`,
`tenferro-linalg`).

For detailed math derivations, see [AD Formula Notes](../AD/index.md).

## Crate Split

The current AD stack is split into two crates:

- **`chainrules-core`**: trait and error definitions only
  - `Differentiable`
  - `ReverseRule<V>`, `ForwardRule<V>`
  - `AutodiffError`, `AdResult`, `NodeId`, `SavePolicy`
- **`chainrules`**: execution engine
  - `Tape<V>`, `TrackedTensor<V>`, `DualTensor<V>`
  - `Tape::pullback`, `Tape::hvp`
  - `Gradients<V>`, `PullbackPlan<V>`, `HvpResult<V>`

Neither crate depends on tenferro tensor/einsum/linalg crates.

```
chainrules-core          ← AD traits and error contract
    ↑
chainrules               ← Explicit tape engine
    ↑
tenferro-tensor          ← Differentiable impl for Tensor<T>
    ↑
tenferro-einsum          ← tracked_einsum / dual_einsum / rrule / frule / hvp
tenferro-linalg          ← Stateless rrule / frule (depends on chainrules-core only)
```

## Operation Ownership

Operation-specific AD rules remain with operation crates:

- `tenferro-einsum`: `tracked_einsum`, `dual_einsum`, `einsum_rrule`,
  `einsum_frule`, `einsum_hvp`
- `tenferro-linalg`: stateless `*_rrule` / `*_frule` functions
- `chainrules-scalarops`: scalar elementary primal/frule/rrule helpers

This keeps `chainrules` generic and avoids circular dependencies.

## Current Usage Examples

### Reverse Mode (`Tape::pullback`)

```rust
use chainrules::Tape;
use tenferro_einsum::tracked_einsum;
use tenferro_tensor::{MemoryOrder, Tensor};
use tenferro_device::LogicalMemorySpace;

let tape = Tape::<Tensor<f64>>::new();
let a = tape.leaf(Tensor::ones(
    &[2, 3],
    LogicalMemorySpace::MainMemory,
    MemoryOrder::ColumnMajor,
));
let b = tape.leaf(Tensor::ones(
    &[3, 4],
    LogicalMemorySpace::MainMemory,
    MemoryOrder::ColumnMajor,
));

let c = tracked_einsum("ij,jk->ik", &[&a, &b])?;
let loss = tracked_einsum("ij,ij->", &[&c, &c])?;

let grads = tape.pullback(&loss)?;
let ga = grads.get(a.node_id().unwrap()).unwrap();
let gb = grads.get(b.node_id().unwrap()).unwrap();
# let _ = (ga, gb);
# Ok::<(), chainrules::AutodiffError>(())
```

### Forward Mode (`DualTensor` + frule-backed ops)

```rust
use chainrules::DualTensor;
use tenferro_einsum::dual_einsum;
use tenferro_tensor::{MemoryOrder, Tensor};
use tenferro_device::LogicalMemorySpace;

let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)?;
let da = Tensor::<f64>::ones(
    &[2, 2],
    LogicalMemorySpace::MainMemory,
    MemoryOrder::ColumnMajor,
);
let b = Tensor::<f64>::ones(
    &[2, 2],
    LogicalMemorySpace::MainMemory,
    MemoryOrder::ColumnMajor,
);

let a_dual = DualTensor::with_tangent(a, da)?;
let b_dual = DualTensor::new(b);
let c_dual = dual_einsum("ij,jk->ik", &[&a_dual, &b_dual])?;

let primal = c_dual.primal();
let tangent = c_dual.tangent().unwrap();
# let _ = (primal, tangent);
# Ok::<(), Box<dyn std::error::Error>>(())
```

### HVP (`Tape::hvp`)

```rust
use chainrules::Tape;
use tenferro_einsum::tracked_einsum;
use tenferro_tensor::{MemoryOrder, Tensor};
use tenferro_device::LogicalMemorySpace;

let tape = Tape::<Tensor<f64>>::new();
let x = tape.leaf_with_tangent(
    Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
    Tensor::ones(&[3], LogicalMemorySpace::MainMemory, MemoryOrder::ColumnMajor),
)?;

let loss = tracked_einsum("i,i->", &[&x, &x])?;
let result = tape.hvp(&loss)?;

let grad_x = result.gradients.get(x.node_id().unwrap()).unwrap();
let hvp_x = result.hvp.get(x.node_id().unwrap()).unwrap();
# let _ = (grad_x, hvp_x);
# Ok::<(), chainrules::AutodiffError>(())
```

### Repeated Pullback via `PullbackPlan`

The current engine does not provide torch-like `backward(options)` arguments.
For repeated reverse execution from the same loss node, use `PullbackPlan`.

```rust
use chainrules::{PullbackPlan, Tape};

let tape = Tape::<f64>::new();
let x = tape.leaf(2.0);
let plan = PullbackPlan::build(&x)?;

let g1 = plan.execute(&x)?;
let g2 = plan.execute(&x)?;

assert_eq!(*g1.get(x.node_id().unwrap()).unwrap(), 1.0);
assert_eq!(*g2.get(x.node_id().unwrap()).unwrap(), 1.0);
# Ok::<(), chainrules::AutodiffError>(())
```

## Current Error Contract

Unsupported AD modes must return typed errors:

```rust
use chainrules_core::AutodiffError;

let err = AutodiffError::ModeNotSupported {
    mode: "frule".into(),
    reason: "tropical einsum supports rrule only (max is not smooth)".into(),
};
# let _ = err;
```

`ModeNotSupported` is preferred over string-only errors for FFI and host-side
branching.

## Notes on Scope

- Current `chainrules` is explicit-tape oriented.
- `retain_graph` / `create_graph` options are not first-class public API in the
  current engine.
- High-level torch-like ergonomics are tracked in the next design document:
  [autodiff-next.md](./autodiff-next.md).
