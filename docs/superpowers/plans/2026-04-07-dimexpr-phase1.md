# DimExpr Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Uniformly convert all `Vec<usize>` shape fields in StdTensorOp to `Vec<DimExpr>`, enabling graph reuse across varying tensor sizes.

**Architecture:** Every shape field (`input_shape`, `from_shape`, `to_shape`, `lhs_shape`, `rhs_shape`) becomes `Vec<DimExpr>`. At trace time, these store `InputDim` references to the op's own inputs. AD rules read DimExpr from forward ops and remap input indices for backward ops via `remap_input_idx`. DimExpr is evaluated to concrete sizes during StableHLO lowering.

**Tech Stack:** Rust, tenferro-ops, tenferro, tenferro-einsum, computegraph

**Spec:** `docs/superpowers/specs/2026-04-07-shape-agnostic-graph-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `tenferro-ops/src/dim_expr.rs` | Create | DimExpr enum, eval, Hash/Eq, remap, helpers |
| `tenferro-ops/src/lib.rs` | Modify | Add `pub mod dim_expr` |
| `tenferro-ops/src/std_tensor_op.rs` | Modify | All shape fields → `Vec<DimExpr>` |
| `tenferro-ops/src/semiring_ops.rs` | Modify | Update trait signatures |
| `tenferro-ops/src/ad/structural.rs` | Modify | Reshape + BroadcastInDim AD rules |
| `tenferro-ops/src/ad/contraction.rs` | Modify | ReduceSum/Prod/Max/Min AD rules |
| `tenferro-ops/src/ad/linalg.rs` | Modify | Linalg AD rules |
| `tenferro-ops/src/ad/mod.rs` | Modify | AD dispatch |
| `tenferro-ops/src/tests/std_tensor_op_tests.rs` | Modify | Fix tests |
| `tenferro/src/traced.rs` | Modify | shape→rank, op construction |
| `tenferro/src/sym_dim.rs` | Create | SymDim type with operator overloading |
| `tenferro/src/lib.rs` | Modify | Add `pub mod sym_dim` |
| `tenferro/src/compiler.rs` | Modify | Evaluate DimExpr during lowering |
| `tenferro/src/linalg_api.rs` | Modify | Update op construction |
| `tenferro-einsum/src/builder.rs` | Modify | Wrap concrete shapes in DimExpr::Const |

---

### Task 1: Define DimExpr Type

**Files:**
- Create: `tenferro-ops/src/dim_expr.rs`
- Modify: `tenferro-ops/src/lib.rs`

- [ ] **Step 1: Create dim_expr.rs**

```rust
// tenferro-ops/src/dim_expr.rs

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
/// assert_eq!(expr.eval(&[&[3, 4]]), 12);
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
    /// Evaluate using actual input tensor shapes.
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

    /// Maximum `input_idx` referenced, or `None` if all Const.
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

    /// Remap `InputDim { input_idx: from, .. }` to `InputDim { input_idx: to, .. }`.
    ///
    /// Used by AD rules to adjust input references when constructing
    /// backward ops with different input ordering than the forward op.
    pub fn remap(&self, from: usize, to: usize) -> Self {
        match self {
            Self::Const(v) => Self::Const(*v),
            Self::InputDim { input_idx, axis } => {
                let new_idx = if *input_idx == from { to } else { *input_idx };
                Self::InputDim { input_idx: new_idx, axis: *axis }
            }
            Self::Add(a, b) => Self::add(a.remap(from, to), b.remap(from, to)),
            Self::Sub(a, b) => Self::sub(a.remap(from, to), b.remap(from, to)),
            Self::Mul(a, b) => Self::mul(a.remap(from, to), b.remap(from, to)),
            Self::FloorDiv(a, b) => Self::floor_div(a.remap(from, to), b.remap(from, to)),
            Self::Min(a, b) => Self::min(a.remap(from, to), b.remap(from, to)),
            Self::Max(a, b) => Self::max(a.remap(from, to), b.remap(from, to)),
        }
    }

    // --- Convenience constructors ---

    pub fn constant(v: usize) -> Self { Self::Const(v) }

    pub fn add(a: Self, b: Self) -> Self { Self::Add(Box::new(a), Box::new(b)) }
    pub fn sub(a: Self, b: Self) -> Self { Self::Sub(Box::new(a), Box::new(b)) }
    pub fn mul(a: Self, b: Self) -> Self { Self::Mul(Box::new(a), Box::new(b)) }
    pub fn floor_div(a: Self, b: Self) -> Self { Self::FloorDiv(Box::new(a), Box::new(b)) }
    pub fn min(a: Self, b: Self) -> Self { Self::Min(Box::new(a), Box::new(b)) }
    pub fn max(a: Self, b: Self) -> Self { Self::Max(Box::new(a), Box::new(b)) }

    pub fn is_const(&self) -> bool { matches!(self, Self::Const(_)) }

    /// Convert `Vec<usize>` to `Vec<DimExpr::Const>`.
    pub fn from_concrete(shape: &[usize]) -> Vec<Self> {
        shape.iter().map(|&v| Self::Const(v)).collect()
    }

    /// Build `[InputDim(input_idx, 0), ..., InputDim(input_idx, rank-1)]`.
    pub fn input_shape(input_idx: usize, rank: usize) -> Vec<Self> {
        (0..rank).map(|axis| Self::InputDim { input_idx, axis }).collect()
    }

    /// Evaluate a slice of DimExpr to concrete sizes.
    pub fn eval_all(exprs: &[Self], input_shapes: &[&[usize]]) -> Vec<usize> {
        exprs.iter().map(|e| e.eval(input_shapes)).collect()
    }

    /// Remap all InputDim references in a slice.
    pub fn remap_all(exprs: &[Self], from: usize, to: usize) -> Vec<Self> {
        exprs.iter().map(|e| e.remap(from, to)).collect()
    }

    /// Compute max_input_idx across a slice of DimExpr.
    pub fn max_input_idx_all(exprs: &[Self]) -> Option<usize> {
        exprs.iter().filter_map(|d| d.max_input_idx()).max()
    }
}

impl From<usize> for DimExpr {
    fn from(v: usize) -> Self { Self::Const(v) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_eval() {
        assert_eq!(DimExpr::Const(42).eval(&[]), 42);
    }

    #[test]
    fn test_input_dim_eval() {
        let e = DimExpr::InputDim { input_idx: 0, axis: 1 };
        assert_eq!(e.eval(&[&[3, 7, 5]]), 7);
    }

    #[test]
    fn test_arithmetic() {
        let shapes: &[&[usize]] = &[&[3, 4], &[5]];
        let e = DimExpr::mul(
            DimExpr::InputDim { input_idx: 0, axis: 0 },
            DimExpr::InputDim { input_idx: 0, axis: 1 },
        );
        assert_eq!(e.eval(shapes), 12);
        assert_eq!(DimExpr::add(e.clone(), DimExpr::Const(3)).eval(shapes), 15);
        assert_eq!(DimExpr::floor_div(e, DimExpr::Const(4)).eval(shapes), 3);
    }

    #[test]
    fn test_min_max() {
        let shapes: &[&[usize]] = &[&[3, 7]];
        let e_min = DimExpr::min(
            DimExpr::InputDim { input_idx: 0, axis: 0 },
            DimExpr::InputDim { input_idx: 0, axis: 1 },
        );
        assert_eq!(e_min.eval(shapes), 3);
        let e_max = DimExpr::max(
            DimExpr::InputDim { input_idx: 0, axis: 0 },
            DimExpr::InputDim { input_idx: 0, axis: 1 },
        );
        assert_eq!(e_max.eval(shapes), 7);
    }

    #[test]
    fn test_max_input_idx() {
        assert_eq!(DimExpr::Const(5).max_input_idx(), None);
        assert_eq!(DimExpr::InputDim { input_idx: 2, axis: 0 }.max_input_idx(), Some(2));
        let e = DimExpr::add(
            DimExpr::InputDim { input_idx: 0, axis: 0 },
            DimExpr::InputDim { input_idx: 3, axis: 1 },
        );
        assert_eq!(e.max_input_idx(), Some(3));
    }

    #[test]
    fn test_remap() {
        let e = DimExpr::mul(
            DimExpr::InputDim { input_idx: 0, axis: 0 },
            DimExpr::InputDim { input_idx: 0, axis: 1 },
        );
        let remapped = e.remap(0, 1);
        assert_eq!(
            remapped,
            DimExpr::mul(
                DimExpr::InputDim { input_idx: 1, axis: 0 },
                DimExpr::InputDim { input_idx: 1, axis: 1 },
            )
        );
    }

    #[test]
    fn test_remap_selective() {
        // Only remap input_idx 0, leave input_idx 2 unchanged
        let e = DimExpr::add(
            DimExpr::InputDim { input_idx: 0, axis: 0 },
            DimExpr::InputDim { input_idx: 2, axis: 1 },
        );
        let remapped = e.remap(0, 1);
        assert_eq!(
            remapped,
            DimExpr::add(
                DimExpr::InputDim { input_idx: 1, axis: 0 },
                DimExpr::InputDim { input_idx: 2, axis: 1 },
            )
        );
    }

    #[test]
    fn test_input_shape() {
        let exprs = DimExpr::input_shape(0, 3);
        assert_eq!(exprs.len(), 3);
        assert_eq!(exprs[0], DimExpr::InputDim { input_idx: 0, axis: 0 });
        assert_eq!(exprs[2], DimExpr::InputDim { input_idx: 0, axis: 2 });
    }

    #[test]
    fn test_from_concrete() {
        let exprs = DimExpr::from_concrete(&[3, 4, 5]);
        assert_eq!(exprs, vec![DimExpr::Const(3), DimExpr::Const(4), DimExpr::Const(5)]);
    }

    #[test]
    fn test_hash_eq_structural() {
        use std::collections::HashSet;
        let a = DimExpr::mul(DimExpr::Const(2), DimExpr::Const(3));
        let b = DimExpr::mul(DimExpr::Const(2), DimExpr::Const(3));
        let c = DimExpr::mul(DimExpr::Const(3), DimExpr::Const(2)); // commuted
        assert_eq!(a, b);
        assert_ne!(a, c);
        let mut set = HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }
}
```

- [ ] **Step 2: Register module**

Add `pub mod dim_expr;` to `tenferro-ops/src/lib.rs`.

- [ ] **Step 3: Run tests**

Run: `cargo test -p tenferro-ops dim_expr`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add tenferro-ops/src/dim_expr.rs tenferro-ops/src/lib.rs
git commit -m "feat: add DimExpr type with eval, remap, and helpers (#651)"
```

---

### Task 2: Update StdTensorOp + SemiringOps

Change all `Vec<usize>` shape fields to `Vec<DimExpr>`. This breaks
downstream crates until later tasks fix them.

**Files:**
- Modify: `tenferro-ops/src/std_tensor_op.rs`
- Modify: `tenferro-ops/src/semiring_ops.rs`

- [ ] **Step 1: Update StdTensorOp variants**

Add `use crate::dim_expr::DimExpr;` at the top.

Change every shape field from `Vec<usize>` to `Vec<DimExpr>`:

```rust
Reshape {
    from_shape: Vec<DimExpr>,  // was Vec<usize>
    to_shape: Vec<DimExpr>,    // was Vec<usize>
},
BroadcastInDim {
    shape: Vec<DimExpr>,       // was Vec<usize>
    dims: Vec<usize>,          // unchanged
},
ReduceSum {
    axes: Vec<usize>,          // unchanged
    input_shape: Vec<DimExpr>, // was Vec<usize>
},
ReduceProd {
    axes: Vec<usize>,
    input_shape: Vec<DimExpr>,
},
ReduceMax {
    axes: Vec<usize>,
    input_shape: Vec<DimExpr>,
},
ReduceMin {
    axes: Vec<usize>,
    input_shape: Vec<DimExpr>,
},
Cholesky {
    input_shape: Vec<DimExpr>,
},
Lu {
    input_shape: Vec<DimExpr>,
},
Svd {
    eps: f64,
    input_shape: Vec<DimExpr>,
},
Qr {
    input_shape: Vec<DimExpr>,
},
Eigh {
    eps: f64,
    input_shape: Vec<DimExpr>,
},
Eig {
    input_dtype: DType,
    input_shape: Vec<DimExpr>,
},
TriangularSolve {
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
    lhs_shape: Vec<DimExpr>,
    rhs_shape: Vec<DimExpr>,
},
```

- [ ] **Step 2: Update Hash impl**

The Hash impl already hashes each field. Since `DimExpr` derives `Hash`,
no logic changes are needed — only field type changes. Verify all match
arms still compile.

- [ ] **Step 3: Update n_inputs to be dynamic**

For ops whose DimExpr may reference additional inputs (added by AD rules),
compute n_inputs from DimExpr fields. Add a helper:

```rust
/// Compute n_inputs from DimExpr fields: max referenced input_idx + 1,
/// but at least `min_inputs` (the number of data inputs).
fn n_inputs_from_dim_exprs(min_inputs: usize, exprs: &[&[DimExpr]]) -> usize {
    let max_idx = exprs.iter()
        .flat_map(|e| e.iter())
        .filter_map(|d| d.max_input_idx())
        .max()
        .map_or(0, |m| m + 1);
    max_idx.max(min_inputs)
}
```

Update `n_inputs()` for affected ops:

```rust
Self::Reshape { from_shape, to_shape } => {
    n_inputs_from_dim_exprs(1, &[from_shape, to_shape])
}
Self::BroadcastInDim { shape, .. } => {
    n_inputs_from_dim_exprs(1, &[shape])
}
Self::ReduceSum { input_shape, .. }
| Self::ReduceProd { input_shape, .. }
| Self::ReduceMax { input_shape, .. }
| Self::ReduceMin { input_shape, .. } => {
    n_inputs_from_dim_exprs(1, &[input_shape])
}
Self::Cholesky { input_shape }
| Self::Lu { input_shape }
| Self::Svd { input_shape, .. }
| Self::Qr { input_shape }
| Self::Eigh { input_shape, .. }
| Self::Eig { input_shape, .. } => {
    n_inputs_from_dim_exprs(1, &[input_shape])
}
Self::TriangularSolve { lhs_shape, rhs_shape, .. } => {
    n_inputs_from_dim_exprs(2, &[lhs_shape, rhs_shape])
}
```

- [ ] **Step 4: Update SemiringOps trait**

```rust
use crate::dim_expr::DimExpr;

pub trait SemiringOps: GraphOp {
    fn add_op() -> Self;
    fn mul_op() -> Self;
    fn dot_general(config: DotGeneralConfig) -> Self;
    fn reduce_sum(axes: Vec<usize>, input_shape: Vec<DimExpr>) -> Self;
    fn transpose_op(perm: Vec<usize>) -> Self;
    fn reshape(from_shape: Vec<DimExpr>, to_shape: Vec<DimExpr>) -> Self;
    fn broadcast_in_dim(shape: Vec<DimExpr>, dims: Vec<usize>) -> Self;
    fn extract_diag(axis_a: usize, axis_b: usize) -> Self;
    fn embed_diag(axis_a: usize, axis_b: usize) -> Self;
}
```

- [ ] **Step 5: Update SemiringOps impl**

```rust
fn reduce_sum(axes: Vec<usize>, input_shape: Vec<DimExpr>) -> Self {
    StdTensorOp::ReduceSum { axes, input_shape }
}
fn reshape(from_shape: Vec<DimExpr>, to_shape: Vec<DimExpr>) -> Self {
    StdTensorOp::Reshape { from_shape, to_shape }
}
fn broadcast_in_dim(shape: Vec<DimExpr>, dims: Vec<usize>) -> Self {
    StdTensorOp::BroadcastInDim { shape, dims }
}
```

- [ ] **Step 6: Verify tenferro-ops compiles**

Run: `cargo check -p tenferro-ops`
Expected: Compiles (downstream crates will have errors).

- [ ] **Step 7: Commit**

```bash
git add tenferro-ops/src/std_tensor_op.rs tenferro-ops/src/semiring_ops.rs
git commit -m "refactor: convert all shape fields to Vec<DimExpr> (#651)"
```

---

### Task 3: Update AD Rules — Structural

**Files:**
- Modify: `tenferro-ops/src/ad/structural.rs`

- [ ] **Step 1: Update linearize_reshape**

No logic change — just match the new field names (both `from_shape` and
`to_shape` are now `Vec<DimExpr>`). The linearize rule clones the same
op for the tangent:

```rust
pub fn linearize_reshape(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    op: &StdTensorOp,
) -> Vec<Option<LocalValId>> {
    let StdTensorOp::Reshape { from_shape, to_shape } = op else {
        unreachable!("linearize_reshape expects Reshape");
    };
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::Reshape {
                    from_shape: from_shape.clone(),
                    to_shape: to_shape.clone(),
                },
                vec![ValRef::Local(dx)],
                OpMode::Linear { active_mask: vec![true] },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
```

- [ ] **Step 2: Update transpose_reshape**

The backward reshape swaps from/to and remaps InputDim(0,..) → InputDim(1,..)
because the backward op has inputs `[cotangent, primal_input]`:

```rust
pub fn transpose_reshape(
    builder: &mut FragmentBuilder<StdTensorOp>,
    cotangent_out: &[Option<LocalValId>],
    op: &StdTensorOp,
    inputs: &[ValRef<StdTensorOp>],
) -> Vec<Option<LocalValId>> {
    let StdTensorOp::Reshape { from_shape, to_shape } = op else {
        unreachable!("transpose_reshape expects Reshape");
    };
    match cotangent_out[0] {
        Some(ct) => {
            // Backward: swap from/to, remap InputDim(0,..) -> InputDim(1,..)
            let out = builder.add_op(
                StdTensorOp::Reshape {
                    from_shape: DimExpr::remap_all(to_shape, 0, 1),
                    to_shape: DimExpr::remap_all(from_shape, 0, 1),
                },
                vec![ValRef::Local(ct), inputs[0].clone()],
                OpMode::Linear { active_mask: vec![true, false] },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
```

Note: `transpose_reshape` now takes `inputs` parameter. Update the
call site in `ad/mod.rs` (Task 5).

- [ ] **Step 3: Update linearize_broadcast_in_dim**

Change parameter type from `&[usize]` to `&[DimExpr]`:

```rust
pub fn linearize_broadcast_in_dim(
    builder: &mut FragmentBuilder<StdTensorOp>,
    tangent_in: &[Option<LocalValId>],
    shape: &[DimExpr],
    dims: &[usize],
) -> Vec<Option<LocalValId>> {
    // Same logic, shape is now Vec<DimExpr>
    match tangent_in[0] {
        Some(dx) => {
            let out = builder.add_op(
                StdTensorOp::BroadcastInDim {
                    shape: shape.to_vec(),
                    dims: dims.to_vec(),
                },
                vec![ValRef::Local(dx)],
                OpMode::Linear { active_mask: vec![true] },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
```

- [ ] **Step 4: Update transpose_broadcast_in_dim**

The backward creates ReduceSum. `input_shape` for that ReduceSum is the
BroadcastInDim's output shape (which is `shape`), remapped:

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
            // ReduceSum's input_shape = the cotangent's shape = broadcast output shape
            // Remap InputDim(0,..) -> InputDim(0,..) (no remap needed here;
            // the cotangent IS input 0 of the ReduceSum)
            let out = builder.add_op(
                StdTensorOp::ReduceSum {
                    axes: broadcast_axes,
                    input_shape: shape.to_vec(),
                },
                vec![ValRef::Local(ct)],
                OpMode::Linear { active_mask: vec![true] },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
```

- [ ] **Step 5: Commit**

```bash
git add tenferro-ops/src/ad/structural.rs
git commit -m "refactor: update structural AD rules for DimExpr (#651)"
```

---

### Task 4: Update AD Rules — Contraction

**Files:**
- Modify: `tenferro-ops/src/ad/contraction.rs`

- [ ] **Step 1: Update helper functions**

Read `contraction.rs` fully first. Update these helpers:

`normalize_reduction_cotangent` and `normalize_scalar_cotangent` create
`Reshape { from_shape: vec![1], to_shape: vec![] }`. Change to DimExpr:

```rust
StdTensorOp::Reshape {
    from_shape: DimExpr::from_concrete(&[1]),
    to_shape: vec![],
}
```

`broadcast_reduction_output_fixed` creates BroadcastInDim with
`input_shape`. Update to accept `&[DimExpr]` and remap:

```rust
fn broadcast_reduction_output(
    builder: &mut FragmentBuilder<StdTensorOp>,
    output: ValRef<StdTensorOp>,
    shape_source: ValRef<StdTensorOp>,
    input_shape: &[DimExpr],
    kept_dims: &[usize],
) -> LocalValId {
    // Remap InputDim(0,..) -> InputDim(1,..) because inputs are
    // [output, shape_source]
    let shape = DimExpr::remap_all(input_shape, 0, 1);
    builder.add_op(
        StdTensorOp::BroadcastInDim { shape, dims: kept_dims.to_vec() },
        vec![output, shape_source],
        OpMode::Primal,
    )[0]
}
```

`reduction_location_counts` creates ReduceSum with `input_shape`:

```rust
fn reduction_location_counts(
    builder: &mut FragmentBuilder<StdTensorOp>,
    indicators: LocalValId,
    axes: &[usize],
    input_shape: &[DimExpr],
) -> LocalValId {
    builder.add_op(
        StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
            input_shape: input_shape.to_vec(),
        },
        vec![ValRef::Local(indicators)],
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
    let StdTensorOp::ReduceSum { axes, input_shape } = op else {
        unreachable!();
    };
    match cotangent_out[0] {
        Some(ct) => {
            let kept_dims = kept_dims(input_shape.len(), axes);
            let cotangent = normalize_reduction_cotangent(builder, ct, &kept_dims);
            // Remap: forward's InputDim(0,..) -> backward's InputDim(1,..)
            let shape = DimExpr::remap_all(input_shape, 0, 1);
            let out = builder.add_op(
                StdTensorOp::BroadcastInDim { shape, dims: kept_dims },
                vec![cotangent, inputs[0].clone()],
                OpMode::Linear { active_mask: vec![true, false] },
            );
            vec![Some(out[0])]
        }
        None => vec![None],
    }
}
```

- [ ] **Step 3: Update linearize_reduce_prod and linearize_reduce_chooser**

These functions take `input_shape: &[usize]` parameters. Change to
`&[DimExpr]`. Update internal ReduceSum/BroadcastInDim constructions
similarly. Read each function fully, apply the same pattern:
- `input_shape.len()` for rank (unchanged — DimExpr slice has same len)
- `input_shape.to_vec()` for ReduceSum's `input_shape` field
- Use `broadcast_reduction_output` helper (updated in Step 1)

- [ ] **Step 4: Update transpose_reduce_prod and transpose_reduce_chooser**

Same pattern as `transpose_reduce_sum`: extract `input_shape: &[DimExpr]`
from op, use `DimExpr::remap_all` for backward ops, pass `inputs[0]` as
shape source.

- [ ] **Step 5: Commit**

```bash
git add tenferro-ops/src/ad/contraction.rs
git commit -m "refactor: update contraction AD rules for DimExpr (#651)"
```

---

### Task 5: Update AD Dispatch + Linalg AD

**Files:**
- Modify: `tenferro-ops/src/ad/mod.rs`
- Modify: `tenferro-ops/src/ad/linalg.rs`

- [ ] **Step 1: Update AD dispatch (mod.rs)**

Read `mod.rs` fully. Update dispatch calls to pass new parameter types.
Key changes:

- `transpose_reshape` now takes `inputs` — add it to the call
- `linearize_reduce_prod`/`linearize_reduce_chooser` take `&[DimExpr]`
  instead of `&[usize]` — the match arm already extracts `input_shape`
- `transpose_reduce_sum`/`prod`/`chooser` now take `inputs` — add it

- [ ] **Step 2: Update linalg AD rules**

Read `linalg.rs` fully. Every function that takes `input_shape: &[usize]`
changes to `&[DimExpr]`. Internal usage patterns:

```rust
// Before:
let (m, n, batch_shape) = (input_shape[0], input_shape[1], &input_shape[2..]);

// After:
let (m, n, batch_shape) = (&input_shape[0], &input_shape[1], &input_shape[2..]);
// m, n are now &DimExpr, batch_shape is &[DimExpr]
```

Where linalg AD builds Reshape or BroadcastInDim ops, it already passes
shape vectors — these are now `Vec<DimExpr>` instead of `Vec<usize>`.
The `matrix_shape` and `vector_shape` helpers that build shape vectors
must return `Vec<DimExpr>`.

For `min(m, n)` (used in SVD/QR), use `DimExpr::min(m.clone(), n.clone())`.

This is a large file. Read it carefully and update systematically. Every
`Vec<usize>` shape construction becomes `Vec<DimExpr>`.

- [ ] **Step 3: Verify tenferro-ops compiles**

Run: `cargo check -p tenferro-ops`

- [ ] **Step 4: Commit**

```bash
git add tenferro-ops/src/ad/mod.rs tenferro-ops/src/ad/linalg.rs
git commit -m "refactor: update AD dispatch and linalg rules for DimExpr (#651)"
```

---

### Task 6: Update Einsum Builder

**Files:**
- Modify: `tenferro-einsum/src/builder.rs`

- [ ] **Step 1: Update SemiringOps calls**

Add `use tenferro_ops::dim_expr::DimExpr;` at the top.

Find all calls to `Op::reshape(`, `Op::broadcast_in_dim(`, `Op::reduce_sum(`
and wrap concrete shapes:

```rust
// Before:
Op::reshape(from_shape, to_shape)
// After:
Op::reshape(DimExpr::from_concrete(&from_shape), DimExpr::from_concrete(&to_shape))

// Before:
Op::broadcast_in_dim(shape, dims)
// After:
Op::broadcast_in_dim(DimExpr::from_concrete(&shape), dims)

// Before:
Op::reduce_sum(axes, input_shape)
// After:
Op::reduce_sum(axes, DimExpr::from_concrete(&input_shape))
```

- [ ] **Step 2: Verify compilation**

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
- Possibly: `tenferro/src/stablehlo.rs`, `tenferro/src/exec.rs`

- [ ] **Step 1: Understand the execution pipeline**

Read these files to understand where actual tensor shapes are available:
- `tenferro/src/compiler.rs` — `lower_to_stablehlo`
- `tenferro/src/stablehlo.rs` — `StableHloOp` definition
- `tenferro/src/exec.rs` — `eval_exec_ir`, `ExecOp`

The spec says DimExpr evaluation happens during StdTensorOp→StableHloOp
lowering. Check if input tensor shapes are available at that point. If
not, the evaluation must happen later (during execution).

- [ ] **Step 2: Implement DimExpr evaluation**

The approach depends on findings from Step 1. Two options:

**Option A: Evaluate during lowering** (if shapes available):
```rust
StdTensorOp::Reshape { to_shape, .. } => {
    let concrete = DimExpr::eval_all(&to_shape, &input_shapes_for_this_op);
    StableHloOp::Reshape { shape: concrete }
}
```

**Option B: Propagate DimExpr to StableHloOp** (if shapes not available
at lowering):
Change StableHloOp's shape fields to `Vec<DimExpr>` and evaluate in
`eval_exec_ir` where actual tensors are available.

Read the code to determine which option is needed. The current lowering
in `compiler.rs` already ignores `from_shape` and `input_shape` fields:
```rust
StdTensorOp::Reshape { to_shape, .. } => StableHloOp::Reshape { shape: to_shape.clone() }
StdTensorOp::ReduceSum { axes, .. } => StableHloOp::ReduceSum { axes: axes.clone() }
```

So `to_shape` must be evaluable at lowering time OR deferred.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p tenferro`

- [ ] **Step 4: Commit**

```bash
git add tenferro/src/compiler.rs
# Include stablehlo.rs and exec.rs if modified
git commit -m "refactor: evaluate DimExpr during lowering/execution (#651)"
```

---

### Task 8: Update TracedTensor (shape → rank)

**Files:**
- Modify: `tenferro/src/traced.rs`
- Modify: `tenferro/src/linalg_api.rs`

- [ ] **Step 1: Change TracedTensor struct**

```rust
static NEXT_TRACED_ID: AtomicU64 = AtomicU64::new(0);
pub type TracedTensorId = u64;

pub struct TracedTensor {
    pub id: TracedTensorId,
    pub rank: usize,              // was: shape: Vec<usize>
    pub dtype: DType,
    pub fragment: Arc<Fragment<StdTensorOp>>,
    pub val: LocalValId,
    pub data: Option<Tensor>,
    pub(crate) inputs_map: Arc<HashMap<TensorInputKey, Tensor>>,
    pub(crate) extra_roots: Vec<Arc<Fragment<StdTensorOp>>>,
}
```

- [ ] **Step 2: Update apply_unary, apply_binary, apply_nullary**

Change `out_shape: Vec<usize>` → `out_rank: usize`. In the TracedTensor
construction, set `id: NEXT_TRACED_ID.fetch_add(1, Ordering::Relaxed)`
and `rank: out_rank`.

- [ ] **Step 3: Update op construction methods**

```rust
pub fn reshape(&self, shape: &[usize]) -> TracedTensor {
    apply_unary(
        StdTensorOp::Reshape {
            from_shape: DimExpr::input_shape(0, self.rank),
            to_shape: DimExpr::from_concrete(shape),
        },
        self,
        shape.len(),
    )
}

pub fn reduce_sum(&self, axes: &[usize]) -> TracedTensor {
    apply_unary(
        StdTensorOp::ReduceSum {
            axes: axes.to_vec(),
            input_shape: DimExpr::input_shape(0, self.rank),
        },
        self,
        self.rank - axes.len(),
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

- [ ] **Step 4: Update all other methods using self.shape**

Search `self.shape` in traced.rs. For each occurrence:
- Replace `self.shape.len()` with `self.rank`
- Replace `self.shape[i]` — this is no longer available. For output rank
  computation, use rank inference from the op parameters.
- `transpose`: `out_rank = self.rank`
- `dot_general`: compute from config dimensions
- `extract_diag`: `out_rank = self.rank - 1`
- `embed_diag`: `out_rank = self.rank + 1`
- `broadcast_shape`: compute output rank from input ranks
- `from_tensor`: `rank: tensor.shape().len()`
- Element-wise ops: `out_rank = self.rank`

For `broadcast_shape` helper, it currently computes the output shape
element-by-element. Since we no longer have concrete sizes, replace it
with rank-only computation: `output_rank = max(a.rank, b.rank)`.

- [ ] **Step 5: Update linalg_api.rs**

Search `self.shape`, `.shape`, `out_shape` in `linalg_api.rs`. Update
all op constructions to use `DimExpr::input_shape(0, self.rank)` for
input_shape fields. Replace output shape computations with rank-only.

For SVD output ranks: `u_rank = input.rank, s_rank = input.rank - 1,
vt_rank = input.rank`.

For reduce_prod/max/min: same pattern as reduce_sum.

- [ ] **Step 6: Verify compilation**

Run: `cargo check -p tenferro`

- [ ] **Step 7: Commit**

```bash
git add tenferro/src/traced.rs tenferro/src/linalg_api.rs
git commit -m "refactor: TracedTensor shape to rank, DimExpr in op construction (#651)"
```

---

### Task 9: Fix Tests

**Files:**
- Modify: `tenferro-ops/src/tests/std_tensor_op_tests.rs`
- Modify: `tenferro/tests/*.rs`

- [ ] **Step 1: Fix tenferro-ops tests**

Update all StdTensorOp constructions in test files:

```rust
// Before:
StdTensorOp::Reshape { from_shape: vec![6], to_shape: vec![2, 3] }
// After:
StdTensorOp::Reshape {
    from_shape: DimExpr::from_concrete(&[6]),
    to_shape: DimExpr::from_concrete(&[2, 3]),
}

// Before:
StdTensorOp::ReduceSum { axes: vec![0], input_shape: vec![3, 4] }
// After:
StdTensorOp::ReduceSum {
    axes: vec![0],
    input_shape: DimExpr::from_concrete(&[3, 4]),
}
```

- [ ] **Step 2: Fix integration tests**

Update `tenferro/tests/*.rs`:
- Replace `.shape` with `.rank` in assertions
- `assert_eq!(y.shape, vec![2, 3])` → `assert_eq!(y.rank, 2)`

- [ ] **Step 3: Run all tests**

Run: `cargo test --workspace --release`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: fix tests for DimExpr migration (#651)"
```

---

### Task 10: SymDim API

**Files:**
- Create: `tenferro/src/sym_dim.rs`
- Modify: `tenferro/src/lib.rs`
- Modify: `tenferro/src/traced.rs`
- Test: `tenferro/tests/sym_dim.rs`

- [ ] **Step 1: Create sym_dim.rs**

See Task 10 in the previous plan version for the full SymDim
implementation. Key types:

```rust
pub struct SymDim(pub(crate) RawSymDim);
pub type TracedTensorId = u64;

enum RawSymDim {
    Const(usize),
    TensorAxis { tensor_id: TracedTensorId, axis: usize },
    Add(Box<RawSymDim>, Box<RawSymDim>),
    // ... same variants as DimExpr
}
```

Implement `From<usize>`, operator overloading (`Add`, `Mul`, `Sub`, `Div`
for `SymDim×SymDim`, `SymDim×usize`, `usize×SymDim`), and
`to_dim_expr(&[(TracedTensorId, usize)]) -> Result<DimExpr>`.

- [ ] **Step 2: Add sym_size and reshape_sym to TracedTensor**

```rust
impl TracedTensor {
    pub fn sym_size(&self, axis: usize) -> SymDim {
        SymDim(RawSymDim::TensorAxis { tensor_id: self.id, axis })
    }

    pub fn reshape_sym(&self, shape: &[SymDim]) -> Result<TracedTensor> {
        let tensor_map = [(self.id, 0usize)];
        let to_shape: Vec<DimExpr> = shape.iter()
            .map(|s| s.to_dim_expr(&tensor_map).map_err(|e| Error::Other(e)))
            .collect::<Result<_>>()?;
        Ok(apply_unary(
            StdTensorOp::Reshape {
                from_shape: DimExpr::input_shape(0, self.rank),
                to_shape,
            },
            self,
            shape.len(),
        ))
    }
}
```

- [ ] **Step 3: Write integration test**

Create `tenferro/tests/sym_dim.rs` with tests for `sym_size`,
`reshape_sym`, and mixed `usize`/`SymDim` usage. Test that executing the
graph produces correct results.

- [ ] **Step 4: Run tests and commit**

```bash
cargo test --workspace --release
git add tenferro/src/sym_dim.rs tenferro/src/lib.rs tenferro/src/traced.rs tenferro/tests/sym_dim.rs
git commit -m "feat: add SymDim API with sym_size and reshape_sym (#651)"
```

---

### Task 11: Final Verification

- [ ] **Step 1:** `cargo fmt --all --check` (fix with `cargo fmt --all` if needed)
- [ ] **Step 2:** `cargo test --workspace --release`
- [ ] **Step 3:** `cargo llvm-cov --workspace --json --output-path coverage.json && python3 scripts/check-coverage.py coverage.json`
- [ ] **Step 4:** `cargo doc --workspace --no-deps && python3 scripts/check-docs-site.py`
- [ ] **Step 5:** Commit any fixes

```bash
git add -A
git commit -m "chore: final cleanup for DimExpr migration (#651)"
```
