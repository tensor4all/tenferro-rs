# DimExpr Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace concrete `Vec<usize>` shapes in StdTensorOp with `DimExpr` expressions, enabling graph reuse across varying tensor sizes.

**Architecture:** Define `DimExpr` (evaluated at execution time from actual input shapes) in tenferro-ops, update op variants to use it, update AD rules to construct DimExpr references to primal inputs, and update the StableHLO lowering to evaluate DimExpr. TracedTensor changes from storing `shape: Vec<usize>` to `rank: usize`.

**Tech Stack:** Rust, tenferro-ops, tenferro, computegraph

**Spec:** `docs/superpowers/specs/2026-04-07-shape-agnostic-graph-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `tenferro-ops/src/dim_expr.rs` | Create | DimExpr enum, eval, Hash/Eq, max_input_idx |
| `tenferro-ops/src/lib.rs` | Modify | Add `pub mod dim_expr` |
| `tenferro-ops/src/std_tensor_op.rs` | Modify | Update Reshape, BroadcastInDim, ReduceSum/Prod/Max/Min |
| `tenferro-ops/src/semiring_ops.rs` | Modify | Update trait signatures |
| `tenferro-ops/src/ad/structural.rs` | Modify | Update Reshape + BroadcastInDim AD rules |
| `tenferro-ops/src/ad/contraction.rs` | Modify | Update ReduceSum/Prod/Max/Min AD rules |
| `tenferro-ops/src/ad/mod.rs` | Modify | Update AD dispatch |
| `tenferro-ops/src/tests/std_tensor_op_tests.rs` | Modify | Fix tests |
| `tenferro/src/traced.rs` | Modify | shape->rank, sym_size, update op construction |
| `tenferro/src/sym_dim.rs` | Create | SymDim type with operator overloading |
| `tenferro/src/lib.rs` | Modify | Add `pub mod sym_dim` |
| `tenferro/src/compiler.rs` | Modify | Evaluate DimExpr during lowering |
| `tenferro/src/linalg_api.rs` | Modify | Update reduce_sum/reshape/broadcast calls |
| `tenferro-einsum/src/builder.rs` | Modify | Update SemiringOps calls to use DimExpr |

---

### Task 1: Define DimExpr Type

**Files:**
- Create: `tenferro-ops/src/dim_expr.rs`
- Modify: `tenferro-ops/src/lib.rs`
- Test: `tenferro-ops/src/dim_expr.rs` (inline tests, small leaf module)

- [ ] **Step 1: Create dim_expr.rs with DimExpr enum + eval**

```rust
// tenferro-ops/src/dim_expr.rs
use std::hash::{Hash, Hasher};

/// Arithmetic expression over tensor dimension sizes.
///
/// Evaluated at execution time from actual input tensor shapes.
/// `InputDim { input_idx, axis }` references the axis size of
/// the op's `input_idx`-th input tensor.
///
/// # Examples
///
/// ```ignore
/// use tenferro_ops::dim_expr::DimExpr;
///
/// let expr = DimExpr::mul(
///     DimExpr::InputDim { input_idx: 0, axis: 0 },
///     DimExpr::InputDim { input_idx: 0, axis: 1 },
/// );
/// let result = expr.eval(&[&[3, 4]]);
/// assert_eq!(result, 12);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DimExpr {
    Const(usize),
    InputDim { input_idx: usize, axis: usize },
    Add(Box<DimExpr>, Box<DimExpr>),
    Sub(Box<DimExpr>, Box<DimExpr>),
    Mul(Box<DimExpr>, Box<DimExpr>),
    FloorDiv(Box<DimExpr>, Box<DimExpr>),
    Min(Box<DimExpr>, Box<DimExpr>),
    Max(Box<DimExpr>, Box<DimExpr>),
}

impl DimExpr {
    /// Evaluate the expression using actual input tensor shapes.
    pub fn eval(&self, input_shapes: &[&[usize]]) -> usize {
        match self {
            Self::Const(v) => *v,
            Self::InputDim { input_idx, axis } => input_shapes[*input_idx][*axis],
            Self::Add(a, b) => a.eval(input_shapes) + b.eval(input_shapes),
            Self::Sub(a, b) => a.eval(input_shapes) - b.eval(input_shapes),
            Self::Mul(a, b) => a.eval(input_shapes) * b.eval(input_shapes),
            Self::FloorDiv(a, b) => a.eval(input_shapes) / b.eval(input_shapes),
            Self::Min(a, b) => a.eval(input_shapes).min(b.eval(input_shapes)),
            Self::Max(a, b) => a.eval(input_shapes).max(b.eval(input_shapes)),
        }
    }

    /// Return the maximum `input_idx` referenced, or `None` if all Const.
    pub fn max_input_idx(&self) -> Option<usize> {
        match self {
            Self::Const(_) => None,
            Self::InputDim { input_idx, .. } => Some(*input_idx),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::FloorDiv(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => match (a.max_input_idx(), b.max_input_idx()) {
                (Some(x), Some(y)) => Some(x.max(y)),
                (Some(x), None) | (None, Some(x)) => Some(x),
                (None, None) => None,
            },
        }
    }

    /// Convenience: create `Const(v)`.
    pub fn constant(v: usize) -> Self {
        Self::Const(v)
    }

    /// Convenience: create `Add(a, b)`.
    pub fn add(a: Self, b: Self) -> Self {
        Self::Add(Box::new(a), Box::new(b))
    }

    /// Convenience: create `Sub(a, b)`.
    pub fn sub(a: Self, b: Self) -> Self {
        Self::Sub(Box::new(a), Box::new(b))
    }

    /// Convenience: create `Mul(a, b)`.
    pub fn mul(a: Self, b: Self) -> Self {
        Self::Mul(Box::new(a), Box::new(b))
    }

    /// Convenience: create `FloorDiv(a, b)`.
    pub fn floor_div(a: Self, b: Self) -> Self {
        Self::FloorDiv(Box::new(a), Box::new(b))
    }

    /// Convenience: create `Min(a, b)`.
    pub fn min(a: Self, b: Self) -> Self {
        Self::Min(Box::new(a), Box::new(b))
    }

    /// Convenience: create `Max(a, b)`.
    pub fn max(a: Self, b: Self) -> Self {
        Self::Max(Box::new(a), Box::new(b))
    }

    /// Whether this expression is a plain constant.
    pub fn is_const(&self) -> bool {
        matches!(self, Self::Const(_))
    }

    /// Convert a `Vec<usize>` to `Vec<DimExpr::Const>`.
    pub fn from_concrete(shape: &[usize]) -> Vec<Self> {
        shape.iter().map(|&v| Self::Const(v)).collect()
    }

    /// Evaluate a slice of DimExpr to concrete sizes.
    pub fn eval_all(exprs: &[Self], input_shapes: &[&[usize]]) -> Vec<usize> {
        exprs.iter().map(|e| e.eval(input_shapes)).collect()
    }
}

impl From<usize> for DimExpr {
    fn from(v: usize) -> Self {
        Self::Const(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_eval() {
        let e = DimExpr::Const(42);
        assert_eq!(e.eval(&[]), 42);
    }

    #[test]
    fn test_input_dim_eval() {
        let e = DimExpr::InputDim { input_idx: 0, axis: 1 };
        assert_eq!(e.eval(&[&[3, 7, 5]]), 7);
    }

    #[test]
    fn test_arithmetic() {
        // inputs[0].shape = [3, 4], inputs[1].shape = [5]
        let shapes: &[&[usize]] = &[&[3, 4], &[5]];
        let e = DimExpr::mul(
            DimExpr::InputDim { input_idx: 0, axis: 0 },
            DimExpr::InputDim { input_idx: 0, axis: 1 },
        );
        assert_eq!(e.eval(shapes), 12);

        let e2 = DimExpr::add(e.clone(), DimExpr::Const(3));
        assert_eq!(e2.eval(shapes), 15);

        let e3 = DimExpr::floor_div(e, DimExpr::Const(4));
        assert_eq!(e3.eval(shapes), 3);
    }

    #[test]
    fn test_min_max() {
        let shapes: &[&[usize]] = &[&[3, 7]];
        let e = DimExpr::min(
            DimExpr::InputDim { input_idx: 0, axis: 0 },
            DimExpr::InputDim { input_idx: 0, axis: 1 },
        );
        assert_eq!(e.eval(shapes), 3);

        let e2 = DimExpr::max(
            DimExpr::InputDim { input_idx: 0, axis: 0 },
            DimExpr::InputDim { input_idx: 0, axis: 1 },
        );
        assert_eq!(e2.eval(shapes), 7);
    }

    #[test]
    fn test_max_input_idx() {
        assert_eq!(DimExpr::Const(5).max_input_idx(), None);
        assert_eq!(
            DimExpr::InputDim { input_idx: 2, axis: 0 }.max_input_idx(),
            Some(2)
        );
        let e = DimExpr::add(
            DimExpr::InputDim { input_idx: 0, axis: 0 },
            DimExpr::InputDim { input_idx: 3, axis: 1 },
        );
        assert_eq!(e.max_input_idx(), Some(3));
    }

    #[test]
    fn test_from_concrete() {
        let exprs = DimExpr::from_concrete(&[3, 4, 5]);
        assert_eq!(exprs.len(), 3);
        assert_eq!(exprs[0].eval(&[]), 3);
        assert_eq!(exprs[1].eval(&[]), 4);
        assert_eq!(exprs[2].eval(&[]), 5);
    }

    #[test]
    fn test_hash_eq() {
        use std::collections::HashSet;
        let a = DimExpr::mul(DimExpr::Const(2), DimExpr::Const(3));
        let b = DimExpr::mul(DimExpr::Const(2), DimExpr::Const(3));
        let c = DimExpr::mul(DimExpr::Const(3), DimExpr::Const(2));
        assert_eq!(a, b);
        assert_ne!(a, c); // structural, not algebraic

        let mut set = HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }
}
```

- [ ] **Step 2: Register module in lib.rs**

Add to `tenferro-ops/src/lib.rs`:
```rust
pub mod dim_expr;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p tenferro-ops dim_expr`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tenferro-ops/src/dim_expr.rs tenferro-ops/src/lib.rs
git commit -m "feat: add DimExpr type for symbolic dimension expressions (#651)"
```

---

### Task 2: Update StdTensorOp Variants + SemiringOps

This task updates the op enum fields and trait signatures. All downstream
consumers will break until subsequent tasks fix them.

**Files:**
- Modify: `tenferro-ops/src/std_tensor_op.rs`
- Modify: `tenferro-ops/src/semiring_ops.rs`

- [ ] **Step 1: Update StdTensorOp variants**

In `tenferro-ops/src/std_tensor_op.rs`, add the import and change these variants:

```rust
use crate::dim_expr::DimExpr;

// Change Reshape (line 26-29):
Reshape {
    to_shape: Vec<DimExpr>,
    input_rank: usize,
},

// Change BroadcastInDim (line 30-33):
BroadcastInDim {
    shape: Vec<DimExpr>,
    dims: Vec<usize>,
},

// Change ReduceSum (line 42-45):
ReduceSum {
    axes: Vec<usize>,
    input_rank: usize,
},

// Change ReduceProd (line 101-104):
ReduceProd {
    axes: Vec<usize>,
    input_rank: usize,
},

// Change ReduceMax (line 105-108):
ReduceMax {
    axes: Vec<usize>,
    input_rank: usize,
},

// Change ReduceMin (line 109-112):
ReduceMin {
    axes: Vec<usize>,
    input_rank: usize,
},
```

- [ ] **Step 2: Update Hash impl**

In the Hash impl (lines 222-328), update the affected match arms:

```rust
Self::Reshape { to_shape, input_rank } => {
    to_shape.hash(state);
    input_rank.hash(state);
}
Self::BroadcastInDim { shape, dims } => {
    shape.hash(state);
    dims.hash(state);
}
Self::ReduceSum { axes, input_rank }
| Self::ReduceProd { axes, input_rank }
| Self::ReduceMax { axes, input_rank }
| Self::ReduceMin { axes, input_rank } => {
    axes.hash(state);
    input_rank.hash(state);
}
```

- [ ] **Step 3: Update n_inputs to be dynamic for Reshape/BroadcastInDim**

In the `GraphOp::n_inputs()` impl (line 340), update:

```rust
Self::Reshape { to_shape, .. } => {
    let max_idx = to_shape.iter()
        .filter_map(|d| d.max_input_idx())
        .max()
        .map_or(0, |m| m + 1);
    max_idx.max(1)
}
Self::BroadcastInDim { shape, .. } => {
    let max_idx = shape.iter()
        .filter_map(|d| d.max_input_idx())
        .max()
        .map_or(0, |m| m + 1);
    max_idx.max(1)
}
```

Keep ReduceSum/Prod/Max/Min at `1` (unchanged).

- [ ] **Step 4: Update SemiringOps trait**

In `tenferro-ops/src/semiring_ops.rs`:

```rust
use crate::dim_expr::DimExpr;

pub trait SemiringOps: GraphOp {
    fn add_op() -> Self;
    fn mul_op() -> Self;
    fn dot_general(config: DotGeneralConfig) -> Self;
    fn reduce_sum(axes: Vec<usize>, input_rank: usize) -> Self;
    fn transpose_op(perm: Vec<usize>) -> Self;
    fn reshape(to_shape: Vec<DimExpr>, input_rank: usize) -> Self;
    fn broadcast_in_dim(shape: Vec<DimExpr>, dims: Vec<usize>) -> Self;
    fn extract_diag(axis_a: usize, axis_b: usize) -> Self;
    fn embed_diag(axis_a: usize, axis_b: usize) -> Self;
}
```

- [ ] **Step 5: Update SemiringOps impl for StdTensorOp**

In `tenferro-ops/src/std_tensor_op.rs` (lines 476-515):

```rust
impl SemiringOps for StdTensorOp {
    // ... unchanged methods ...

    fn reduce_sum(axes: Vec<usize>, input_rank: usize) -> Self {
        StdTensorOp::ReduceSum { axes, input_rank }
    }

    fn reshape(to_shape: Vec<DimExpr>, input_rank: usize) -> Self {
        StdTensorOp::Reshape { to_shape, input_rank }
    }

    fn broadcast_in_dim(shape: Vec<DimExpr>, dims: Vec<usize>) -> Self {
        StdTensorOp::BroadcastInDim { shape, dims }
    }
}
```

- [ ] **Step 6: Verify it compiles (expect downstream errors)**

Run: `cargo check -p tenferro-ops 2>&1 | head -50`
Expected: tenferro-ops compiles (with warnings). Downstream crates (tenferro, tenferro-einsum) will have errors — fixed in later tasks.

- [ ] **Step 7: Commit**

```bash
git add tenferro-ops/src/std_tensor_op.rs tenferro-ops/src/semiring_ops.rs
git commit -m "refactor: update StdTensorOp variants and SemiringOps for DimExpr (#651)"
```

---

### Task 3: Update AD Rules (structural.rs)

**Files:**
- Modify: `tenferro-ops/src/ad/structural.rs`

- [ ] **Step 1: Update linearize_reshape**

In `structural.rs` (lines 29-58), the linearize rule applies the same
Reshape to the tangent. Update to match new field names:

```rust
pub fn linearize_reshape(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    op: &StdTensorOp,
) -> Vec<Option<LocalValId>> {
    let StdTensorOp::Reshape { to_shape, input_rank } = op else {
        unreachable!("linearize_reshape expects Reshape");
    };

    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Reshape {
                    to_shape: to_shape.clone(),
                    input_rank: *input_rank,
                },
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
```

- [ ] **Step 2: Update transpose_reshape**

In `structural.rs` (lines 190-219). The backward Reshape needs to reshape
cotangent back to the primal input's shape. Add the primal input as input 1
and use `DimExpr::InputDim` to reference its axes:

```rust
pub fn transpose_reshape(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    op: &StdTensorOp,
    inputs: &[ValRef<StdTensorOp>],
) -> Vec<Option<LocalValId>> {
    let StdTensorOp::Reshape { to_shape, input_rank } = op else {
        unreachable!("transpose_reshape expects Reshape");
    };

    match cotangent_out[0] {
        Some(ct) => {
            // Backward: reshape cotangent to primal input's shape.
            // inputs[0] is the primal data input; add it as input 1
            // to the backward reshape so DimExpr can reference its shape.
            let backward_to_shape: Vec<DimExpr> = (0..*input_rank)
                .map(|a| DimExpr::InputDim { input_idx: 1, axis: a })
                .collect();
            let out = builder.add_op(
                StdTensorOp::Reshape {
                    to_shape: backward_to_shape,
                    input_rank: to_shape.len(),
                },
                vec![ValRef::Local(ct), inputs[0].clone()],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
```

- [ ] **Step 3: Update linearize_broadcast_in_dim**

In `structural.rs` (lines 60-82). Same Reshape/BroadcastInDim applies
to the tangent:

```rust
pub fn linearize_broadcast_in_dim(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    shape: &[DimExpr],
    dims: &[usize],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::BroadcastInDim {
                    shape: shape.to_vec(),
                    dims: dims.to_vec(),
                },
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
```

- [ ] **Step 4: Update transpose_broadcast_in_dim**

In `structural.rs` (lines 221-246). The backward creates a ReduceSum to
undo the broadcast. `broadcast_axes` is computed from `dims` and
`shape.len()` (output rank) — both are known statically:

```rust
pub fn transpose_broadcast_in_dim(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    shape: &[DimExpr],
    dims: &[usize],
) -> Vec<Option<LocalValId>> {
    let output_rank = shape.len();
    let broadcast_axes: Vec<usize> = (0..output_rank)
        .filter(|dim| !dims.contains(dim))
        .collect();

    match cotangent_out[0] {
        Some(ct) if broadcast_axes.is_empty() => vec![Some(ct)],
        Some(ct) => {
            let out = builder.add_op(
                StdTensorOp::ReduceSum {
                    axes: broadcast_axes,
                    input_rank: output_rank,
                },
                vec![ValRef::Local(ct)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
```

- [ ] **Step 5: Update normalize_scalar_cotangent and normalize_reduction_cotangent in contraction.rs**

These helper functions in `contraction.rs` create `Reshape { from_shape: vec![1], to_shape: vec![] }`. Update them:

```rust
// normalize_reduction_cotangent (line 420-440):
fn normalize_reduction_cotangent(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent: LocalValId,
    kept_dims: &[usize],
) -> ValRef<StdTensorOp> {
    if kept_dims.is_empty() {
        let scalar = builder.add_op(
            StdTensorOp::Reshape {
                to_shape: vec![],
                input_rank: 1,
            },
            vec![ValRef::Local(cotangent)],
            OpMode::Linear {
                active_mask: vec![true],
            },
        );
        ValRef::Local(scalar[0])
    } else {
        ValRef::Local(cotangent)
    }
}

// normalize_scalar_cotangent (line 486-506):
fn normalize_scalar_cotangent(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent: LocalValId,
    output_rank: usize,
) -> ValRef<StdTensorOp> {
    if output_rank == 0 {
        let scalar = builder.add_op(
            StdTensorOp::Reshape {
                to_shape: vec![],
                input_rank: 1,
            },
            vec![ValRef::Local(cotangent)],
            OpMode::Linear {
                active_mask: vec![true],
            },
        );
        ValRef::Local(scalar[0])
    } else {
        ValRef::Local(cotangent)
    }
}
```

- [ ] **Step 6: Commit**

```bash
git add tenferro-ops/src/ad/structural.rs tenferro-ops/src/ad/contraction.rs
git commit -m "refactor: update structural AD rules for DimExpr (#651)"
```

---

### Task 4: Update AD Rules (contraction.rs) — Reduce Ops

**Files:**
- Modify: `tenferro-ops/src/ad/contraction.rs`

- [ ] **Step 1: Update broadcast_reduction_output_fixed**

This helper broadcasts a reduced output back to input shape.
Change it to use `DimExpr::InputDim` referencing a shape-source tensor:

```rust
/// Broadcast a reduced output back to input shape using DimExpr.
/// `shape_source` is passed as an extra input so DimExpr can reference it.
fn broadcast_reduction_output(
    builder: &mut FragmentBuilder<StdTensorOp>,
    output: ValRef<StdTensorOp>,
    shape_source: ValRef<StdTensorOp>,
    input_rank: usize,
    kept_dims: &[usize],
) -> LocalValId {
    let shape: Vec<DimExpr> = (0..input_rank)
        .map(|a| DimExpr::InputDim { input_idx: 1, axis: a })
        .collect();
    builder.add_op(
        StdTensorOp::BroadcastInDim {
            shape,
            dims: kept_dims.to_vec(),
        },
        vec![output, shape_source],
        OpMode::Primal,
    )[0]
}
```

- [ ] **Step 2: Update transpose_reduce_sum**

```rust
pub fn transpose_reduce_sum(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    op: &StdTensorOp,
    inputs: &[ValRef<StdTensorOp>],
) -> Vec<Option<LocalValId>> {
    let StdTensorOp::ReduceSum { axes, input_rank } = op else {
        unreachable!("transpose_reduce_sum expects ReduceSum");
    };

    match cotangent_out[0] {
        Some(ct) => {
            let kept_dims = kept_dims(*input_rank, axes);
            let cotangent = normalize_reduction_cotangent(builder, ct, &kept_dims);
            // Build BroadcastInDim with DimExpr referencing primal input (inputs[0])
            let shape: Vec<DimExpr> = (0..*input_rank)
                .map(|a| DimExpr::InputDim { input_idx: 1, axis: a })
                .collect();
            let out = builder.add_op(
                StdTensorOp::BroadcastInDim {
                    shape,
                    dims: kept_dims,
                },
                vec![cotangent, inputs[0].clone()],
                OpMode::Linear {
                    active_mask: vec![true, false],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
```

- [ ] **Step 3: Update linearize_reduce_sum**

```rust
pub fn linearize_reduce_sum(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    op: &StdTensorOp,
    _axes: &[usize],
) -> Vec<Option<LocalValId>> {
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                op.clone(),
                vec![ValRef::Local(dx)],
                OpMode::Linear {
                    active_mask: vec![true],
                },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
```

(No change needed — it clones the op, which now has `input_rank`.)

- [ ] **Step 4: Update linearize_reduce_prod**

Replace `input_shape: &[usize]` parameter with `input_rank: usize`.
Update `broadcast_reduction_output_fixed` calls to `broadcast_reduction_output`:

```rust
pub fn linearize_reduce_prod(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    axes: &[usize],
    input_rank: usize,
) -> Vec<Option<LocalValId>> {
    let Some(dx) = tangent_in[0] else {
        return vec![None];
    };

    let kept_dims = kept_dims(input_rank, axes);
    let prod_broadcast = broadcast_reduction_output(
        builder,
        ValRef::External(primal_out[0].clone()),
        ValRef::External(primal_in[0].clone()),
        input_rank,
        &kept_dims,
    );
    let coeff = builder.add_op(
        StdTensorOp::Div,
        vec![
            ValRef::Local(prod_broadcast),
            ValRef::External(primal_in[0].clone()),
        ],
        OpMode::Primal,
    )[0];
    let scaled_tangent = builder.add_op(
        StdTensorOp::Mul,
        vec![ValRef::Local(coeff), ValRef::Local(dx)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    let out = builder.add_op(
        StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
            input_rank,
        },
        vec![ValRef::Local(scaled_tangent)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0];
    vec![Some(out)]
}
```

- [ ] **Step 5: Update linearize_reduce_chooser**

Same pattern: replace `input_shape: &[usize]` with `input_rank: usize`.

```rust
pub fn linearize_reduce_chooser(
    builder: &mut FragmentBuilder<StdTensorOp>,
    primal_in: &[GlobalValKey<StdTensorOp>],
    primal_out: &[GlobalValKey<StdTensorOp>],
    tangent_in: &[Option<LocalValId>],
    axes: &[usize],
    input_rank: usize,
) -> Vec<Option<LocalValId>> {
    let Some(dx) = tangent_in[0] else {
        return vec![None];
    };

    let kept_dims = kept_dims(input_rank, axes);
    let answer_broadcast = broadcast_reduction_output(
        builder,
        ValRef::External(primal_out[0].clone()),
        ValRef::External(primal_in[0].clone()),
        input_rank,
        &kept_dims,
    );
    let indicators = reduction_location_indicators(
        builder,
        ValRef::External(primal_in[0].clone()),
        ValRef::Local(answer_broadcast),
    );
    let weighted_tangent = builder.add_op(
        StdTensorOp::Mul,
        vec![ValRef::Local(indicators), ValRef::Local(dx)],
        OpMode::Linear {
            active_mask: vec![false, true],
        },
    )[0];
    let tangent_sum = builder.add_op(
        StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
            input_rank,
        },
        vec![ValRef::Local(weighted_tangent)],
        OpMode::Linear {
            active_mask: vec![true],
        },
    )[0];
    let counts = reduction_location_counts(builder, indicators, axes, input_rank);
    let out = builder.add_op(
        StdTensorOp::Div,
        vec![ValRef::Local(tangent_sum), ValRef::Local(counts)],
        OpMode::Linear {
            active_mask: vec![true, false],
        },
    )[0];
    vec![Some(out)]
}
```

- [ ] **Step 6: Update reduction_location_counts**

```rust
fn reduction_location_counts(
    builder: &mut FragmentBuilder<StdTensorOp>,
    indicators: LocalValId,
    axes: &[usize],
    input_rank: usize,
) -> LocalValId {
    builder.add_op(
        StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
            input_rank,
        },
        vec![ValRef::Local(indicators)],
        OpMode::Primal,
    )[0]
}
```

- [ ] **Step 7: Update transpose_reduce_prod and transpose_reduce_chooser**

These functions (lines 282-402) follow the same pattern as
`transpose_reduce_sum`. Replace `input_shape` usage with `input_rank`
and `DimExpr::InputDim` references. The key change in each:

```rust
// Extract input_rank instead of input_shape from op:
let StdTensorOp::ReduceProd { axes, input_rank } = op else { ... };
// (or ReduceMax / ReduceMin)

// Use broadcast_reduction_output instead of broadcast_reduction_output_fixed:
let broadcast = broadcast_reduction_output(
    builder, output_ref, inputs[0].clone(), *input_rank, &kept_dims,
);
```

Read the current `transpose_reduce_prod` and `transpose_reduce_chooser`
functions fully before modifying them. Apply the same `input_shape` ->
`input_rank` + DimExpr pattern used in `transpose_reduce_sum`.

- [ ] **Step 8: Commit**

```bash
git add tenferro-ops/src/ad/contraction.rs
git commit -m "refactor: update contraction AD rules for DimExpr (#651)"
```

---

### Task 5: Update AD Dispatch (mod.rs)

**Files:**
- Modify: `tenferro-ops/src/ad/mod.rs`

- [ ] **Step 1: Update linearize dispatch**

In `linearize_non_semiring` (lines 14-121), update the dispatch calls
to match new function signatures. Key changes:

```rust
// ReduceSum: no change needed (linearize_reduce_sum takes op reference)
StdTensorOp::ReduceSum { axes, .. } => {
    contraction::linearize_reduce_sum(builder, tangent_in, op, axes)
}

// ReduceProd: pass input_rank instead of input_shape
StdTensorOp::ReduceProd { axes, input_rank } => {
    contraction::linearize_reduce_prod(
        builder, primal_in, primal_out, tangent_in, axes, *input_rank,
    )
}

// ReduceMax / ReduceMin: pass input_rank instead of input_shape
StdTensorOp::ReduceMax { axes, input_rank }
| StdTensorOp::ReduceMin { axes, input_rank } => {
    contraction::linearize_reduce_chooser(
        builder, primal_in, primal_out, tangent_in, axes, *input_rank,
    )
}

// BroadcastInDim: pass &[DimExpr] instead of &[usize]
StdTensorOp::BroadcastInDim { shape, dims } => {
    structural::linearize_broadcast_in_dim(builder, tangent_in, shape, dims)
}
```

- [ ] **Step 2: Update transpose dispatch**

In `transpose_non_semiring` (lines 138-212), update the dispatch calls.
Key changes — transpose_reshape and transpose_reduce_sum now receive `inputs`:

```rust
// Reshape: pass inputs so backward can reference primal input shape
StdTensorOp::Reshape { .. } => {
    structural::transpose_reshape(builder, cotangent_out, op, inputs)
}

// BroadcastInDim: pass DimExpr shape
StdTensorOp::BroadcastInDim { shape, dims } => {
    structural::transpose_broadcast_in_dim(builder, cotangent_out, shape, dims)
}

// ReduceSum: pass inputs
StdTensorOp::ReduceSum { .. } => {
    contraction::transpose_reduce_sum(builder, cotangent_out, op, inputs)
}

// ReduceProd: pass inputs
StdTensorOp::ReduceProd { .. } => {
    contraction::transpose_reduce_prod(builder, cotangent_out, inputs, op)
}

// ReduceMax / ReduceMin: pass inputs
StdTensorOp::ReduceMax { .. } | StdTensorOp::ReduceMin { .. } => {
    contraction::transpose_reduce_chooser(builder, cotangent_out, inputs, op)
}
```

Read the current `transpose_reduce_prod` and `transpose_reduce_chooser`
signatures to verify they accept `inputs` (they may already — check first).

- [ ] **Step 3: Verify tenferro-ops compiles**

Run: `cargo check -p tenferro-ops`
Expected: Compiles. Downstream crates still broken.

- [ ] **Step 4: Commit**

```bash
git add tenferro-ops/src/ad/mod.rs
git commit -m "refactor: update AD dispatch for DimExpr (#651)"
```

---

### Task 6: Update tenferro-einsum Builder

**Files:**
- Modify: `tenferro-einsum/src/builder.rs`

- [ ] **Step 1: Find all SemiringOps calls in builder.rs**

Search for `Op::reshape(`, `Op::broadcast_in_dim(`, `Op::reduce_sum(`
in `tenferro-einsum/src/builder.rs`. Each call currently passes
`Vec<usize>` — update to wrap in `DimExpr::from_concrete()`.

The einsum builder operates with concrete shapes at trace time, so all
DimExpr values will be `Const`.

For each call site, apply these transformations:

```rust
// Before:
Op::reshape(from_shape, to_shape)
// After:
Op::reshape(DimExpr::from_concrete(&to_shape), from_shape.len())

// Before:
Op::broadcast_in_dim(shape, dims)
// After:
Op::broadcast_in_dim(DimExpr::from_concrete(&shape), dims)

// Before:
Op::reduce_sum(axes, input_shape)
// After:
Op::reduce_sum(axes, input_shape.len())
```

Add the import at the top of the file:
```rust
use tenferro_ops::dim_expr::DimExpr;
```

- [ ] **Step 2: Verify tenferro-einsum compiles**

Run: `cargo check -p tenferro-einsum`

- [ ] **Step 3: Commit**

```bash
git add tenferro-einsum/src/builder.rs
git commit -m "refactor: update einsum builder for DimExpr (#651)"
```

---

### Task 7: Update Compiler (StableHLO Lowering)

**Files:**
- Modify: `tenferro/src/compiler.rs`

- [ ] **Step 1: Add DimExpr evaluation in lowering**

The `lower_to_stablehlo` function translates `StdTensorOp` -> `StableHloOp`.
Currently it passes shapes through directly. After the change, DimExpr
fields must be evaluated using actual input tensor shapes.

The `CompiledProgram` has `instructions` with `input_slots` and
`output_slots`. At lowering time, we do NOT have actual tensor data —
lowering happens before execution. So we need to defer DimExpr evaluation
to execution time.

**Key insight**: The current lowering is a 1-to-1 op translation with no
runtime data. DimExpr evaluation requires runtime shapes. So the approach
is: **keep DimExpr in StableHloOp** (or evaluate during execution, not
during lowering).

Check `tenferro/src/stablehlo.rs` to see if `StableHloOp::Reshape` and
`StableHloOp::BroadcastInDim` already use `Vec<usize>`. If so, evaluation
must happen at the `StableHloOp -> ExecOp` step or during `ExecOp`
execution.

Read `tenferro/src/exec.rs` to understand where actual tensor data is
available and evaluate DimExpr there.

The simplest approach: change `StableHloOp::Reshape { shape: Vec<usize> }`
to `StableHloOp::Reshape { shape: Vec<DimExpr> }`, then evaluate in
`eval_exec_ir` when actual tensors are available.

Alternatively, evaluate during `lower_to_stablehlo` by passing input
shapes through. This depends on whether input shapes are available at
that point — check the call site in `traced.rs`.

**Read these files before making changes:**
- `tenferro/src/stablehlo.rs` (StableHloOp definition)
- `tenferro/src/exec.rs` (ExecOp, eval_exec_ir)
- `tenferro/src/traced.rs` (where lower_to_stablehlo is called)

The spec says: "All DimExpr resolution happens during the
StdTensorOp-to-StableHloOp lowering step." But if actual shapes are not
available at lowering time, resolution must be deferred. Adjust the
approach based on what you find.

- [ ] **Step 2: Implement DimExpr evaluation at the appropriate layer**

Apply the evaluation. The exact change depends on findings from Step 1.
If input shapes are available at lowering time:

```rust
StdTensorOp::Reshape { to_shape, .. } => {
    // input_shapes must be available here
    let concrete: Vec<usize> = DimExpr::eval_all(&to_shape, &input_shapes);
    StableHloOp::Reshape { shape: concrete }
}
```

If not, propagate DimExpr through to the execution layer and evaluate there.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p tenferro`

- [ ] **Step 4: Commit**

```bash
git add tenferro/src/compiler.rs tenferro/src/stablehlo.rs tenferro/src/exec.rs
git commit -m "refactor: evaluate DimExpr during lowering/execution (#651)"
```

---

### Task 8: Update TracedTensor (shape -> rank)

**Files:**
- Modify: `tenferro/src/traced.rs`
- Modify: `tenferro/src/linalg_api.rs`

- [ ] **Step 1: Change TracedTensor struct**

```rust
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TRACED_ID: AtomicU64 = AtomicU64::new(0);

/// Unique identifier for a TracedTensor, used by SymDim references.
pub type TracedTensorId = u64;

fn next_traced_id() -> TracedTensorId {
    NEXT_TRACED_ID.fetch_add(1, Ordering::Relaxed)
}

pub struct TracedTensor {
    pub id: TracedTensorId,
    pub rank: usize,
    pub dtype: DType,
    pub fragment: Arc<Fragment<StdTensorOp>>,
    pub val: LocalValId,
    pub data: Option<Tensor>,
    pub(crate) inputs_map: Arc<HashMap<TensorInputKey, Tensor>>,
    pub(crate) extra_roots: Vec<Arc<Fragment<StdTensorOp>>>,
}
```

- [ ] **Step 2: Update apply_unary, apply_binary, apply_nullary**

Change `out_shape: Vec<usize>` to `out_rank: usize` in all apply_ helpers:

```rust
pub(crate) fn apply_unary(
    op: StdTensorOp,
    input: &TracedTensor,
    out_rank: usize,
) -> TracedTensor {
    apply_unary_with_dtype(op, input, out_rank, input.dtype)
}

pub(crate) fn apply_unary_with_dtype(
    op: StdTensorOp,
    input: &TracedTensor,
    out_rank: usize,
    out_dtype: DType,
) -> TracedTensor {
    let mut builder = FragmentBuilder::new();
    builder.add_parent(input.fragment.clone());
    let input_ref = ValRef::External(input.fragment.vals()[input.val].key.clone());
    let outputs = builder.add_op(op, vec![input_ref], OpMode::Primal);
    builder.set_outputs(outputs.clone());
    let fragment = Arc::new(builder.build());

    TracedTensor {
        id: next_traced_id(),
        rank: out_rank,
        dtype: out_dtype,
        fragment,
        val: outputs[0],
        data: None,
        inputs_map: input.inputs_map.clone(),
        extra_roots: input.extra_roots.clone(),
    }
}
```

Apply the same pattern to `apply_binary` and `apply_nullary`.

- [ ] **Step 3: Update reshape, reduce_sum, broadcast_in_dim methods**

```rust
pub fn reshape(&self, shape: &[usize]) -> TracedTensor {
    apply_unary(
        StdTensorOp::Reshape {
            to_shape: DimExpr::from_concrete(shape),
            input_rank: self.rank,
        },
        self,
        shape.len(),
    )
}

pub fn reduce_sum(&self, axes: &[usize]) -> TracedTensor {
    let out_rank = self.rank - axes.len();
    apply_unary(
        StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
            input_rank: self.rank,
        },
        self,
        out_rank,
    )
}

pub fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> TracedTensor {
    apply_unary(
        StdTensorOp::BroadcastInDim {
            shape: DimExpr::from_concrete(shape),
            dims: dims.to_vec(),
        },
        self,
        shape.len(),
    )
}
```

- [ ] **Step 4: Update all other methods that use self.shape**

Search `self.shape` in traced.rs and update every occurrence:
- `transpose`: `out_rank = self.rank` (rank unchanged by transpose)
- `dot_general`: compute output rank from config
- `extract_diag`: `out_rank = self.rank - 1`
- `embed_diag`: `out_rank = self.rank + 1`
- `broadcast_shape`: needs rewrite — compute output rank from input ranks
- Other element-wise ops: `out_rank = self.rank` (or `lhs.rank`)
- `from_tensor`: `rank: tensor.shape().len()`

Also update `linalg_api.rs` — every function that constructs TracedTensors
with `shape: Vec<usize>` must change to `rank: usize`.

- [ ] **Step 5: Update linalg_api.rs**

Search for `.shape` and `out_shape` in `tenferro/src/linalg_api.rs` and
update. For reduce_prod, reduce_max, reduce_min, update the StdTensorOp
construction to use `input_rank` instead of `input_shape`. Example:

```rust
// reduce_prod:
StdTensorOp::ReduceProd {
    axes: axes.to_vec(),
    input_rank: input.rank,
}
```

For SVD, QR, etc. output rank computation:
```rust
// SVD: u_rank = input.rank, s_rank = input.rank - 1, vt_rank = input.rank
// QR: q_rank = input.rank, r_rank = input.rank
```

Read the current linalg_api.rs fully before modifying.

- [ ] **Step 6: Verify compilation**

Run: `cargo check -p tenferro`

- [ ] **Step 7: Commit**

```bash
git add tenferro/src/traced.rs tenferro/src/linalg_api.rs
git commit -m "refactor: TracedTensor shape to rank (#651)"
```

---

### Task 9: Fix Tests

**Files:**
- Modify: `tenferro-ops/src/tests/std_tensor_op_tests.rs`
- Modify: `tenferro/tests/ad.rs`
- Modify: `tenferro/tests/primitive_ops.rs`
- Modify: other test files as needed

- [ ] **Step 1: Fix tenferro-ops unit tests**

Search for `from_shape`, `input_shape`, `to_shape:` in test files and
update to match new field names. Tests that construct StdTensorOp
directly need updating:

```rust
// Before:
StdTensorOp::Reshape { from_shape: vec![6], to_shape: vec![2, 3] }
// After:
StdTensorOp::Reshape { to_shape: DimExpr::from_concrete(&[2, 3]), input_rank: 1 }

// Before:
StdTensorOp::ReduceSum { axes: vec![0], input_shape: vec![3, 4] }
// After:
StdTensorOp::ReduceSum { axes: vec![0], input_rank: 2 }

// Before:
StdTensorOp::BroadcastInDim { shape: vec![3, 4], dims: vec![1] }
// After:
StdTensorOp::BroadcastInDim { shape: DimExpr::from_concrete(&[3, 4]), dims: vec![1] }
```

- [ ] **Step 2: Fix integration tests**

Update `tenferro/tests/*.rs` — these tests use TracedTensor API methods.
Replace `.shape` accesses with `.rank` assertions where applicable.
The traced tensor methods (reshape, reduce_sum, etc.) still accept
`&[usize]`, so most test code should work as-is.

Tests that assert on output shapes (e.g., `assert_eq!(result.shape, vec![2, 3])`)
need to change to rank assertions or execute the graph and check the
result tensor's shape:

```rust
// Before:
assert_eq!(y.shape, vec![2, 3]);
// After:
assert_eq!(y.rank, 2);
```

- [ ] **Step 3: Run all tests**

Run: `cargo test --workspace --release`
Fix any remaining failures.

- [ ] **Step 4: Run coverage check**

Run: `cargo llvm-cov --workspace --json --output-path coverage.json && python3 scripts/check-coverage.py coverage.json`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: fix tests for DimExpr migration (#651)"
```

---

### Task 10: SymDim API + sym_size

**Files:**
- Create: `tenferro/src/sym_dim.rs`
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/src/traced.rs`
- Test: `tenferro/tests/sym_dim.rs`

- [ ] **Step 1: Create sym_dim.rs**

```rust
// tenferro/src/sym_dim.rs
use std::ops;
use tenferro_ops::dim_expr::DimExpr;

/// User-facing symbolic dimension value.
///
/// Created via `TracedTensor::sym_size(axis)`. Supports arithmetic
/// via operator overloading to build `DimExpr` expression trees.
///
/// # Examples
///
/// ```ignore
/// let merged = x.sym_size(0) * x.sym_size(1);
/// let y = x.reshape_sym(&[merged, 4.into()]);
/// ```
#[derive(Clone, Debug)]
pub struct SymDim(pub(crate) RawSymDim);

/// Unique ID for a TracedTensor, used to resolve SymDim -> DimExpr.
pub type TracedTensorId = u64;

#[derive(Clone, Debug)]
pub(crate) enum RawSymDim {
    Const(usize),
    TensorAxis { tensor_id: TracedTensorId, axis: usize },
    Add(Box<RawSymDim>, Box<RawSymDim>),
    Sub(Box<RawSymDim>, Box<RawSymDim>),
    Mul(Box<RawSymDim>, Box<RawSymDim>),
    FloorDiv(Box<RawSymDim>, Box<RawSymDim>),
    Min(Box<RawSymDim>, Box<RawSymDim>),
    Max(Box<RawSymDim>, Box<RawSymDim>),
}

impl SymDim {
    /// Resolve this SymDim to a DimExpr using a tensor_id -> input_idx map.
    pub(crate) fn to_dim_expr(
        &self,
        tensor_map: &[(TracedTensorId, usize)],
    ) -> Result<DimExpr, String> {
        self.0.to_dim_expr(tensor_map)
    }
}

impl RawSymDim {
    fn to_dim_expr(
        &self,
        tensor_map: &[(TracedTensorId, usize)],
    ) -> Result<DimExpr, String> {
        match self {
            Self::Const(v) => Ok(DimExpr::Const(*v)),
            Self::TensorAxis { tensor_id, axis } => {
                let input_idx = tensor_map
                    .iter()
                    .find(|(id, _)| id == tensor_id)
                    .map(|(_, idx)| *idx)
                    .ok_or_else(|| {
                        format!("SymDim references tensor_id {} which is not an input", tensor_id)
                    })?;
                Ok(DimExpr::InputDim { input_idx, axis: *axis })
            }
            Self::Add(a, b) => Ok(DimExpr::add(
                a.to_dim_expr(tensor_map)?,
                b.to_dim_expr(tensor_map)?,
            )),
            Self::Sub(a, b) => Ok(DimExpr::sub(
                a.to_dim_expr(tensor_map)?,
                b.to_dim_expr(tensor_map)?,
            )),
            Self::Mul(a, b) => Ok(DimExpr::mul(
                a.to_dim_expr(tensor_map)?,
                b.to_dim_expr(tensor_map)?,
            )),
            Self::FloorDiv(a, b) => Ok(DimExpr::floor_div(
                a.to_dim_expr(tensor_map)?,
                b.to_dim_expr(tensor_map)?,
            )),
            Self::Min(a, b) => Ok(DimExpr::min(
                a.to_dim_expr(tensor_map)?,
                b.to_dim_expr(tensor_map)?,
            )),
            Self::Max(a, b) => Ok(DimExpr::max(
                a.to_dim_expr(tensor_map)?,
                b.to_dim_expr(tensor_map)?,
            )),
        }
    }
}

impl From<usize> for SymDim {
    fn from(v: usize) -> Self {
        SymDim(RawSymDim::Const(v))
    }
}

// SymDim op SymDim
impl ops::Add for SymDim {
    type Output = SymDim;
    fn add(self, rhs: SymDim) -> SymDim {
        SymDim(RawSymDim::Add(Box::new(self.0), Box::new(rhs.0)))
    }
}

impl ops::Sub for SymDim {
    type Output = SymDim;
    fn sub(self, rhs: SymDim) -> SymDim {
        SymDim(RawSymDim::Sub(Box::new(self.0), Box::new(rhs.0)))
    }
}

impl ops::Mul for SymDim {
    type Output = SymDim;
    fn mul(self, rhs: SymDim) -> SymDim {
        SymDim(RawSymDim::Mul(Box::new(self.0), Box::new(rhs.0)))
    }
}

impl ops::Div for SymDim {
    type Output = SymDim;
    fn div(self, rhs: SymDim) -> SymDim {
        SymDim(RawSymDim::FloorDiv(Box::new(self.0), Box::new(rhs.0)))
    }
}

// SymDim op usize
impl ops::Add<usize> for SymDim {
    type Output = SymDim;
    fn add(self, rhs: usize) -> SymDim {
        self + SymDim::from(rhs)
    }
}

impl ops::Sub<usize> for SymDim {
    type Output = SymDim;
    fn sub(self, rhs: usize) -> SymDim {
        self - SymDim::from(rhs)
    }
}

impl ops::Mul<usize> for SymDim {
    type Output = SymDim;
    fn mul(self, rhs: usize) -> SymDim {
        self * SymDim::from(rhs)
    }
}

impl ops::Div<usize> for SymDim {
    type Output = SymDim;
    fn div(self, rhs: usize) -> SymDim {
        self / SymDim::from(rhs)
    }
}

// usize op SymDim
impl ops::Add<SymDim> for usize {
    type Output = SymDim;
    fn add(self, rhs: SymDim) -> SymDim {
        SymDim::from(self) + rhs
    }
}

impl ops::Mul<SymDim> for usize {
    type Output = SymDim;
    fn mul(self, rhs: SymDim) -> SymDim {
        SymDim::from(self) * rhs
    }
}
```

- [ ] **Step 2: Register module and add sym_size to TracedTensor**

In `tenferro/src/lib.rs`, add:
```rust
pub mod sym_dim;
```

In `tenferro/src/traced.rs`, add `sym_size` and `reshape_sym` methods:

```rust
use crate::sym_dim::{SymDim, TracedTensorId};

impl TracedTensor {
    /// Return a symbolic reference to this tensor's axis size.
    pub fn sym_size(&self, axis: usize) -> SymDim {
        SymDim(crate::sym_dim::RawSymDim::TensorAxis {
            tensor_id: self.id,
            axis,
        })
    }

    /// Reshape with symbolic dimensions.
    pub fn reshape_sym(&self, shape: &[SymDim]) -> Result<TracedTensor> {
        let tensor_map = [(self.id, 0usize)];
        let to_shape: Vec<DimExpr> = shape
            .iter()
            .map(|s| s.to_dim_expr(&tensor_map).map_err(|e| Error::Other(e)))
            .collect::<Result<_>>()?;
        Ok(apply_unary(
            StdTensorOp::Reshape {
                to_shape,
                input_rank: self.rank,
            },
            self,
            shape.len(),
        ))
    }
}
```

- [ ] **Step 3: Write integration test**

Create `tenferro/tests/sym_dim.rs`:

```rust
use tenferro::engine::Engine;
use tenferro::traced::TracedTensor;
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{DType, TypedTensor};

#[test]
fn test_sym_reshape_flatten() {
    let mut engine = Engine::new(CpuBackend::new());

    // Create a 2x3 tensor
    let data = TypedTensor::<f64>::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let x = TracedTensor::from_tensor(data.into());
    assert_eq!(x.rank, 2);

    // Reshape using symbolic sizes: flatten to [x.shape[0] * x.shape[1]]
    let merged = x.sym_size(0) * x.sym_size(1);
    let y = x.reshape_sym(&[merged]).unwrap();
    assert_eq!(y.rank, 1);

    // Execute and verify
    let results = engine.eval(&[&y]).unwrap();
    assert_eq!(results[0].shape(), &[6]);
}

#[test]
fn test_sym_reshape_with_constant() {
    let mut engine = Engine::new(CpuBackend::new());

    let data = TypedTensor::<f64>::from_shape_vec(vec![6], (1..=6).map(|x| x as f64).collect());
    let x = TracedTensor::from_tensor(data.into());

    // Reshape [6] -> [2, 3] using sym: x.sym_size(0) / 2, 2
    let half = x.sym_size(0) / 2;
    let y = x.reshape_sym(&[2.into(), half]).unwrap();
    assert_eq!(y.rank, 2);

    let results = engine.eval(&[&y]).unwrap();
    assert_eq!(results[0].shape(), &[2, 3]);
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --workspace --release`

- [ ] **Step 5: Commit**

```bash
git add tenferro/src/sym_dim.rs tenferro/src/lib.rs tenferro/src/traced.rs tenferro/tests/sym_dim.rs
git commit -m "feat: add SymDim API with sym_size and reshape_sym (#651)"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Full test suite**

Run: `cargo test --workspace --release`

- [ ] **Step 2: Formatting**

Run: `cargo fmt --all --check`
If fails: `cargo fmt --all`

- [ ] **Step 3: Coverage**

Run: `cargo llvm-cov --workspace --json --output-path coverage.json && python3 scripts/check-coverage.py coverage.json`

- [ ] **Step 4: Docs**

Run: `cargo doc --workspace --no-deps && python3 scripts/check-docs-site.py`

- [ ] **Step 5: Final commit if any fixes**

```bash
git add -A
git commit -m "chore: final cleanup for DimExpr phase 1 (#651)"
```
