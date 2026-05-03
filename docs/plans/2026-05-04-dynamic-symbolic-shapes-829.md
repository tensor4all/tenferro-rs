# Dynamic Symbolic Shapes #829 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the first #829 slice: explicit exact/upper-bound shape metadata, stop treating `DynamicTruncate`'s runtime-sized axis as exact, and remove the `transpose_scatter` symbolic-slice panic.

**Architecture:** Add a value-side shape extent guarantee type in `tenferro-ops`, then thread it through `TensorMeta` and shape inference without breaking existing exact-shape callers. Keep backend configs concrete; introduce a graph/execution gather form whose `slice_sizes` can be resolved from runtime input shapes immediately before backend dispatch. Defer exact `RuntimeScalar` shape expressions for `DynamicTruncate` until the compiler and execution layers can resolve them safely.

**Tech Stack:** Rust, `computegraph`, `tenferro-ops`, `tenferro`, `tenferro-tensor`, existing `StdTensorOp` / `ExecOp` compiler and AD pipeline.

---

## Scope

This plan targets the first implementation change set for #829. It should
cover:

- `DynamicTruncate` metadata no longer claiming the truncated axis is exact.
- Consumers that require exact dimensions can ask for exactness explicitly.
- `transpose_scatter` no longer calls `constant_value().unwrap_or_else(panic)` for symbolic update-window dimensions.
- Backend-facing `tenferro_tensor::GatherConfig` remains concrete.

PR policy: make local commits at the task boundaries below, but do not open
intermediate PRs. Open one PR only after every task in this plan is complete
and the final verification passes.

This plan intentionally does not implement exact runtime-scalar dimensions such
as `RuntimeScalar { input_idx: 1, semantics: DynamicTruncateSize }`. That is a
follow-up after exactness checks are in place.

Before touching AD rules, re-read `REPOSITORY_RULES.md`.

---

### Task 1: Add Shape Extent Types

**Files:**
- Create: `tenferro-ops/src/shape_extent.rs`
- Modify: `tenferro-ops/src/lib.rs`
- Test: `tenferro-ops/src/tests/shape_extent_tests.rs`
- Modify: `tenferro-ops/src/tests/mod.rs`

**Step 1: Write the failing tests**

Add `mod shape_extent_tests;` in `tenferro-ops/src/tests/mod.rs`.

Create `tenferro-ops/src/tests/shape_extent_tests.rs`:

```rust
use tenferro_ops::shape_extent::{ShapeExtent, ShapeMeta};
use tenferro_ops::SymDim;

#[test]
fn exact_extent_exposes_exact_dim() {
    let extent = ShapeExtent::exact(SymDim::from(4usize));
    assert_eq!(extent.as_exact(), Some(&SymDim::from(4usize)));
    assert!(extent.is_exact());
}

#[test]
fn upper_bound_is_not_exact() {
    let extent = ShapeExtent::upper_bound(SymDim::from(4usize));
    assert_eq!(extent.as_exact(), None);
    assert_eq!(extent.bound_expr(), Some(&SymDim::from(4usize)));
    assert!(!extent.is_exact());
}

#[test]
fn shape_meta_reports_rank_and_exact_shape() {
    let meta = ShapeMeta::exact(vec![SymDim::from(2usize), SymDim::from(3usize)]);
    assert_eq!(meta.rank(), 2);
    assert_eq!(
        meta.exact_shape(),
        Some(vec![SymDim::from(2usize), SymDim::from(3usize)])
    );
}

#[test]
fn shape_meta_exact_shape_rejects_upper_bound() {
    let meta = ShapeMeta::new(vec![
        ShapeExtent::exact(SymDim::from(2usize)),
        ShapeExtent::upper_bound(SymDim::from(3usize)),
    ]);
    assert_eq!(meta.rank(), 2);
    assert_eq!(meta.exact_shape(), None);
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro-ops shape_extent --release
```

Expected: compile failure because `shape_extent` does not exist.

**Step 3: Add the implementation**

Create `tenferro-ops/src/shape_extent.rs`:

```rust
//! Shape extent metadata with exactness guarantees.

/// A dimension expression plus the guarantee it provides.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShapeExtent<D> {
    Exact(D),
    UpperBound(D),
    Unknown,
}

impl<D> ShapeExtent<D> {
    pub fn exact(dim: D) -> Self {
        Self::Exact(dim)
    }

    pub fn upper_bound(dim: D) -> Self {
        Self::UpperBound(dim)
    }

    pub fn unknown() -> Self {
        Self::Unknown
    }

    pub fn is_exact(&self) -> bool {
        matches!(self, Self::Exact(_))
    }

    pub fn as_exact(&self) -> Option<&D> {
        match self {
            Self::Exact(dim) => Some(dim),
            Self::UpperBound(_) | Self::Unknown => None,
        }
    }

    pub fn bound_expr(&self) -> Option<&D> {
        match self {
            Self::Exact(dim) | Self::UpperBound(dim) => Some(dim),
            Self::Unknown => None,
        }
    }

    pub fn map<E>(self, f: impl FnOnce(D) -> E) -> ShapeExtent<E> {
        match self {
            Self::Exact(dim) => ShapeExtent::Exact(f(dim)),
            Self::UpperBound(dim) => ShapeExtent::UpperBound(f(dim)),
            Self::Unknown => ShapeExtent::Unknown,
        }
    }
}

/// Rank-exact shape metadata.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShapeMeta<D> {
    extents: Vec<ShapeExtent<D>>,
}

impl<D> ShapeMeta<D> {
    pub fn new(extents: Vec<ShapeExtent<D>>) -> Self {
        Self { extents }
    }

    pub fn exact(shape: Vec<D>) -> Self {
        Self::new(shape.into_iter().map(ShapeExtent::Exact).collect())
    }

    pub fn rank(&self) -> usize {
        self.extents.len()
    }

    pub fn extents(&self) -> &[ShapeExtent<D>] {
        &self.extents
    }
}

impl<D: Clone> ShapeMeta<D> {
    pub fn exact_shape(&self) -> Option<Vec<D>> {
        self.extents.iter().map(|extent| extent.as_exact().cloned()).collect()
    }

    pub fn bound_shape(&self) -> Option<Vec<D>> {
        self.extents.iter().map(|extent| extent.bound_expr().cloned()).collect()
    }
}
```

Update `tenferro-ops/src/lib.rs`:

```rust
pub mod shape_extent;
pub use shape_extent::{ShapeExtent, ShapeMeta};
```

**Step 4: Run test to verify it passes**

Run:

```bash
cargo test -p tenferro-ops shape_extent --release
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-ops/src/shape_extent.rs tenferro-ops/src/lib.rs tenferro-ops/src/tests/mod.rs tenferro-ops/src/tests/shape_extent_tests.rs
git commit -m "feat: add shape extent metadata types"
```

---

### Task 2: Thread Shape Extents Through Value Metadata

**Files:**
- Modify: `tenferro-ops/src/ad/context.rs`
- Modify: `tenferro/src/metadata.rs`
- Test: `tenferro-ops/src/ad/tests/mod.rs`

**Step 1: Write the failing tests**

Add tests near the existing metadata tests in `tenferro-ops/src/ad/tests/mod.rs`:

```rust
use tenferro_ops::shape_extent::ShapeExtent;

#[test]
fn metadata_exposes_exact_extents() {
    let key = input_key(700);
    let mut ctx = ShapeGuardContext::default();
    let meta = TensorMeta::exact(DType::F64, vec![SymDim::from(2usize), SymDim::from(3usize)]);
    ctx.insert_metadata(key.clone(), meta.clone());

    let extents = ctx.extents_of(&ValRef::External(key));
    assert_eq!(extents, meta.extents());
    assert_eq!(ctx.exact_shape_of(&ValRef::External(key)), Some(meta.shape.clone()));
}

#[test]
fn metadata_exact_shape_rejects_upper_bound() {
    let key = input_key(701);
    let mut ctx = ShapeGuardContext::default();
    let meta = TensorMeta {
        dtype: DType::F64,
        shape: vec![SymDim::from(4usize)],
        extents: vec![ShapeExtent::upper_bound(SymDim::from(4usize))],
    };
    ctx.insert_metadata(key.clone(), meta);
    assert_eq!(ctx.exact_shape_of(&ValRef::External(key)), None);
}
```

Use the local helper names already present in that test module. If `input_key`
or `insert_metadata` names differ, adapt to the existing helpers instead of
adding new infrastructure.

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro-ops metadata_exposes_exact_extents --release
```

Expected: compile failure because `TensorMeta::exact`, `extents`, and
`ShapeGuardContext::extents_of` do not exist.

**Step 3: Update `TensorMeta`**

In `tenferro-ops/src/ad/context.rs`, extend `TensorMeta`:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMeta {
    pub dtype: DType,
    pub shape: Vec<SymDim>,
    pub extents: Vec<ShapeExtent<SymDim>>,
}

impl TensorMeta {
    pub fn exact(dtype: DType, shape: Vec<SymDim>) -> Self {
        let extents = shape.iter().cloned().map(ShapeExtent::exact).collect();
        Self { dtype, shape, extents }
    }

    pub fn with_extents(dtype: DType, extents: Vec<ShapeExtent<SymDim>>) -> Self {
        let shape = extents
            .iter()
            .map(|extent| extent.bound_expr().cloned().unwrap_or_else(|| SymDim::from(0usize)))
            .collect();
        Self { dtype, shape, extents }
    }

    pub fn extents(&self) -> &[ShapeExtent<SymDim>] {
        &self.extents
    }

    pub fn exact_shape(&self) -> Option<Vec<SymDim>> {
        self.extents.iter().map(|extent| extent.as_exact().cloned()).collect()
    }
}
```

The `shape` field stays temporarily for existing callers. Treat it as a
compatibility bound shape, not proof of exactness.

Add `ShapeGuardContext` accessors:

```rust
pub fn extents_of(&mut self, val: &ValRef<StdTensorOp>) -> Vec<ShapeExtent<SymDim>> {
    self.metadata_of(val).extents.clone()
}

pub fn exact_shape_of(&mut self, val: &ValRef<StdTensorOp>) -> Option<Vec<SymDim>> {
    self.metadata_of(val).exact_shape()
}
```

**Step 4: Update tenferro metadata constructors**

In `tenferro/src/metadata.rs`, change constructors to keep exact metadata:

```rust
pub(crate) fn tensor_meta(dtype: DType, shape: Vec<SymDim>) -> TensorMeta {
    TensorMeta::exact(dtype, shape)
}
```

Later tasks will use `TensorMeta::with_extents` when shape inference reports
upper bounds.

**Step 5: Run tests**

Run:

```bash
cargo test -p tenferro-ops metadata_exposes_exact_extents --release
cargo test -p tenferro symbolic_grad --release
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro-ops/src/ad/context.rs tenferro/src/metadata.rs tenferro-ops/src/ad/tests/mod.rs
git commit -m "feat: carry shape extents in tensor metadata"
```

---

### Task 3: Infer Shape Extents And Mark DynamicTruncate As Upper Bound

**Files:**
- Modify: `tenferro/src/shape_infer.rs`
- Modify: `tenferro/src/metadata.rs`
- Test: `tenferro/tests/shape_inference.rs`

**Step 1: Write the failing test**

In `tenferro/tests/shape_inference.rs`, add:

```rust
use tenferro::shape_infer::infer_output_extents;
use tenferro_ops::ShapeExtent;

#[test]
fn dynamic_truncate_extent_is_upper_bound_on_truncated_axis() {
    let input = vec![cst(4), cst(7)];
    let scalar = vec![];
    let op = StdTensorOp::DynamicTruncate { axis: 1 };

    let extents = infer_output_extents(&op, &[&input, &scalar]);

    assert_eq!(extents.len(), 1);
    assert_eq!(extents[0][0], ShapeExtent::exact(cst(4)));
    assert_eq!(extents[0][1], ShapeExtent::upper_bound(cst(7)));
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro dynamic_truncate_extent_is_upper_bound_on_truncated_axis --release
```

Expected: compile failure because `infer_output_extents` does not exist.

**Step 3: Implement `infer_output_extents`**

In `tenferro/src/shape_infer.rs`, import `ShapeExtent` and add:

```rust
pub fn infer_output_extents(
    op: &StdTensorOp,
    input_shapes: &[&[DimExpr]],
) -> Vec<Vec<ShapeExtent<DimExpr>>> {
    match op {
        StdTensorOp::DynamicTruncate { axis } => {
            let shape = require_input(op, input_shapes, 0);
            assert!(
                *axis < shape.len(),
                "DynamicTruncate axis {axis} out of bounds for rank {}",
                shape.len()
            );
            let mut extents: Vec<_> = shape.iter().cloned().map(ShapeExtent::exact).collect();
            extents[*axis] = ShapeExtent::upper_bound(shape[*axis].clone());
            vec![extents]
        }
        _ => infer_output_shapes(op, input_shapes)
            .into_iter()
            .map(|shape| shape.into_iter().map(ShapeExtent::exact).collect())
            .collect(),
    }
}
```

**Step 4: Use extent inference in metadata registration**

In `tenferro/src/metadata.rs`, import `infer_output_extents`. In
`infer_output_metas`, replace the non-extension path's shape-only inference
with extent inference:

```rust
let output_dtype = infer_output_dtype(op, &input_dtypes);
infer_output_extents(op, &input_shape_refs)
    .into_iter()
    .map(|extents| {
        let resolved_inputs: Vec<&[SymDim]> =
            input_metas.iter().map(|meta| meta.shape.as_slice()).collect();
        let resolved_extents = extents
            .into_iter()
            .map(|extent| extent.map(|dim| SymDim::from_dim_expr(&dim, &resolved_inputs)))
            .collect();
        TensorMeta::with_extents(output_dtype, resolved_extents)
    })
    .collect()
```

Keep `infer_output_shapes` unchanged for compatibility until Task 4 updates
compiler consumers.

**Step 5: Run tests**

Run:

```bash
cargo test -p tenferro dynamic_truncate_extent_is_upper_bound_on_truncated_axis --release
cargo test -p tenferro symbolic_grad --release
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro/src/shape_infer.rs tenferro/src/metadata.rs tenferro/tests/shape_inference.rs
git commit -m "fix: mark dynamic truncate extent as upper bound"
```

---

### Task 4: Add Exactness Helpers For Compiler Consumers

**Files:**
- Modify: `tenferro/src/compiler/mod.rs`
- Modify: `tenferro/src/exec.rs`
- Test: `tenferro/tests/compiler_wiring.rs`

**Step 1: Write the failing test**

Add a test in `tenferro/tests/compiler_wiring.rs` that builds:

```text
input [InputDim(0,0)] -> DynamicTruncate(axis=0) -> Reshape(to_shape=[InputDim(0,0)])
```

The test should verify that the compiler does not silently use the
`DynamicTruncate` upper bound as an exact reshape target. Expected behavior for
this change set: optional passes skip the rewrite and leave the original
`DynamicTruncate` executable when no exact dimension is required. Do not make
`compile_std_to_exec` fallible in this task unless an existing caller already
has a `Result` path that can be reused cleanly.

Use the smallest test that fails on the current false-exact behavior. If the
current compiler API is infallible for this path, write this test as a TODO
guard in the same file and complete it when the helper is wired.

**Step 2: Add exactness helper**

Add helper functions near compiler pass helpers:

```rust
fn require_exact_extent<'a>(
    extent: &'a ShapeExtent<DimExpr>,
    op: &'static str,
    field: &'static str,
) -> crate::error::Result<&'a DimExpr> {
    extent.as_exact().ok_or_else(|| Error::UnsupportedDynamicShape {
        op,
        field,
        reason: format!("required exact extent, got {extent:?}"),
    })
}
```

This requires adding an error variant in `tenferro/src/error.rs`:

```rust
#[error("unsupported dynamic shape in {op}.{field}: {reason}")]
UnsupportedDynamicShape {
    op: &'static str,
    field: &'static str,
    reason: String,
},
```

**Step 3: Make compiler shape-extents available**

Add `output_extents` to `ExecInstruction` in `tenferro/src/exec.rs`:

```rust
pub output_extents: Vec<Vec<ShapeExtent<DimExpr>>>,
```

In `compile_std_to_exec`, compute both `output_shapes` and `output_extents`.
For existing exact ops, extents are `Exact`. For `DynamicTruncate`, use
`infer_output_extents`.

Update manual `ExecInstruction` construction in tests by using:

```rust
output_extents: output_shapes
    .iter()
    .cloned()
    .map(|shape| shape.into_iter().map(ShapeExtent::exact).collect())
    .collect(),
```

**Step 4: Update pass consumers**

Search:

```bash
rg -n "output_shapes" tenferro/src/compiler/mod.rs tenferro/src/segment.rs
```

For each compiler pass that builds `Reshape` or `BroadcastInDim` shapes from
another instruction's metadata, use `output_extents` and `require_exact_extent`.

Rules:

- If the pass is optional and the original op remains correct, skip the rewrite.
- If the pass must create a concrete runtime shape parameter and cannot skip,
  keep the existing infallible API behavior for this change set and surface a
  clear internal error at the narrowest existing boundary. A follow-up can make
  compiler APIs fallible end-to-end.
- Do not reinterpret `UpperBound` as exact.

**Step 5: Run tests**

Run:

```bash
cargo test -p tenferro compiler_wiring --release
cargo test -p tenferro compiler_passes --release
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro/src/compiler/mod.rs tenferro/src/exec.rs tenferro/src/error.rs tenferro/tests/compiler_wiring.rs tenferro/tests/compiler_passes.rs tenferro/tests/segment_tests.rs tenferro/tests/exec_dispatch.rs
git commit -m "fix: require exact extents in compiler shape consumers"
```

---

### Task 5: Add Graph Gather With Dynamic Slice Sizes

**Files:**
- Modify: `tenferro-ops/src/std_tensor_op.rs`
- Modify: `tenferro/src/shape_infer.rs`
- Modify: `tenferro/src/compiler/mod.rs`
- Modify: `tenferro/src/exec.rs`
- Modify: `tenferro/src/eager_exec.rs`
- Test: `tenferro/tests/shape_inference.rs`
- Test: `tenferro/tests/exec_dispatch.rs`

**Step 1: Write failing shape inference test**

In `tenferro/tests/shape_inference.rs`, add a test for a new graph op:

```rust
#[test]
fn gather_dynamic_slice_sizes_uses_shape_source_input() {
    let operand = vec![cst(4), cst(5)];
    let indices = vec![cst(1), cst(1)];
    let updates = vec![cst(1), cst(2)];
    let op = StdTensorOp::GatherDynamicSliceSizes {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![cst(1), DimExpr::InputDim { input_idx: 2, axis: 1 }],
    };

    assert_eq!(
        infer_output_shapes(&op, &[&operand, &indices, &updates]),
        vec![vec![cst(1), cst(2)]]
    );
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro gather_dynamic_slice_sizes_uses_shape_source_input --release
```

Expected: compile failure because the op variant does not exist.

**Step 3: Add `StdTensorOp` variant**

Add a new variant in `tenferro-ops/src/std_tensor_op.rs` while leaving
`StdTensorOp::Gather(GatherConfig)` unchanged:

```rust
GatherDynamicSliceSizes {
    offset_dims: Vec<usize>,
    collapsed_slice_dims: Vec<usize>,
    start_index_map: Vec<usize>,
    index_vector_dim: usize,
    slice_sizes: Vec<DimExpr>,
},
```

Set `n_inputs()` for this variant to:

```rust
2 + max_input_idx(slice_sizes).saturating_sub(1)
```

More explicitly: `slice_sizes` may reference `input_idx >= 2` as shape-source
inputs after `(operand, start_indices)`. Validate in constructors/tests that it
does not reference `input_idx` outside the actual op inputs.

**Step 4: Add lowering and execution**

In `tenferro/src/exec.rs`, add:

```rust
GatherDynamicSliceSizes {
    offset_dims: Vec<usize>,
    collapsed_slice_dims: Vec<usize>,
    start_index_map: Vec<usize>,
    index_vector_dim: usize,
    slice_sizes: Vec<DimExpr>,
},
```

When executing:

```rust
let input_shapes: Vec<&[usize]> = inst
    .input_slots
    .iter()
    .map(|slot| slots[*slot].as_ref().expect("slot").shape())
    .collect();
let resolved_slice_sizes = DimExpr::eval_all(&slice_sizes, &input_shapes);
let config = GatherConfig {
    offset_dims: offset_dims.clone(),
    collapsed_slice_dims: collapsed_slice_dims.clone(),
    start_index_map: start_index_map.clone(),
    index_vector_dim: *index_vector_dim,
    slice_sizes: resolved_slice_sizes,
};
let operand = get(slots, &inst.input_slots, 0)?;
let indices = get(slots, &inst.input_slots, 1)?;
slots[inst.output_slots[0]] = Some(backend.gather(operand, indices, &config)?);
```

Shape-source inputs are real instruction inputs so liveness/last-use remains
correct, but backend gather only receives the first two tensors.

Mirror the same resolution in `tenferro/src/eager_exec.rs`.

**Step 5: Run tests**

Run:

```bash
cargo test -p tenferro gather_dynamic_slice_sizes_uses_shape_source_input --release
cargo test -p tenferro exec_dispatch --release
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro-ops/src/std_tensor_op.rs tenferro/src/shape_infer.rs tenferro/src/compiler/mod.rs tenferro/src/exec.rs tenferro/src/eager_exec.rs tenferro/tests/shape_inference.rs tenferro/tests/exec_dispatch.rs
git commit -m "feat: add gather with dynamic slice sizes"
```

---

### Task 6: Replace transpose_scatter Symbolic Panic

**Files:**
- Modify: `tenferro-ops/src/ad/indexing.rs`
- Test: `tenferro-ops/src/ad/tests/indexing_tests.rs`

**Step 1: Write the failing test**

In `tenferro-ops/src/ad/tests/indexing_tests.rs`, add a symbolic updates-shape
case beside `transpose_scatter_window_dims_derive_slice_sizes_from_updates_shape`:

```rust
#[test]
fn transpose_scatter_symbolic_window_dim_emits_dynamic_gather() {
    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut ctx = ShapeGuardContext::default();
    let cot = builder.add_input(tensor_input(900));
    let operand_key = input_key(901);
    let indices_key = input_key(902);
    let updates_key = input_key(903);

    ctx.insert_metadata(
        operand_key.clone(),
        TensorMeta::exact(DType::F64, vec![SymDim::from(4usize), SymDim::from(2usize)]),
    );
    ctx.insert_metadata(
        indices_key.clone(),
        TensorMeta::exact(DType::I64, vec![SymDim::from(1usize), SymDim::from(1usize)]),
    );
    ctx.insert_metadata(
        updates_key.clone(),
        TensorMeta::exact(DType::F64, vec![SymDim::from(1usize), SymDim::tensor_axis(903, 1)]),
    );

    let config = ScatterConfig {
        update_window_dims: vec![1],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };

    let op = StdTensorOp::Scatter(config);
    let inputs = vec![
        ValRef::External(operand_key),
        ValRef::External(indices_key.clone()),
        ValRef::External(updates_key.clone()),
    ];
    let result = op.transpose_rule(
        &mut builder,
        &[Some(cot)],
        &inputs,
        &OpMode::Linear { active_mask: vec![false, false, true] },
        &mut ctx,
    );

    assert!(result[2].is_some());
    let graph = builder.build();
    let gather = graph.ops().last().expect("expected gather op");
    match &gather.op {
        StdTensorOp::GatherDynamicSliceSizes { slice_sizes, .. } => {
            assert_eq!(slice_sizes[1], DimExpr::InputDim { input_idx: 2, axis: 1 });
            assert_eq!(gather.inputs[2], ValRef::External(updates_key));
        }
        other => panic!("expected GatherDynamicSliceSizes, got {other:?}"),
    }
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test -p tenferro-ops transpose_scatter_symbolic_window_dim_emits_dynamic_gather --release
```

Expected: current code panics with
`transpose_scatter: symbolic updates dim ... cannot be used as a slice size`.

**Step 3: Implement symbolic path**

Replace `compute_inverse_slice_sizes` with a helper that returns either a
concrete `GatherConfig` or a dynamic gather descriptor:

```rust
enum InverseGather {
    Concrete(GatherConfig),
    Dynamic {
        offset_dims: Vec<usize>,
        collapsed_slice_dims: Vec<usize>,
        start_index_map: Vec<usize>,
        index_vector_dim: usize,
        slice_sizes: Vec<DimExpr>,
        shape_sources: Vec<ValRef<StdTensorOp>>,
    },
}
```

For symbolic update-window sizes, generate:

```rust
DimExpr::InputDim { input_idx: 2, axis: update_axis }
```

and include `inputs[2].clone()` as the third op input to
`GatherDynamicSliceSizes`.

Concrete update-window sizes should keep emitting `StdTensorOp::Gather` so the
existing tests continue to assert the old fast path.

**Step 4: Run tests**

Run:

```bash
cargo test -p tenferro-ops transpose_scatter --release
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-ops/src/ad/indexing.rs tenferro-ops/src/ad/tests/indexing_tests.rs
git commit -m "fix: support symbolic scatter transpose slice sizes"
```

---

### Task 7: End-To-End Regression Tests

**Files:**
- Test: `tenferro/tests/dynamic_symbolic_shapes.rs`

**Step 1: Write integration tests**

Create `tenferro/tests/dynamic_symbolic_shapes.rs`:

```rust
use tenferro::{CpuBackend, Engine, Tensor, TracedTensor};
use tenferro_tensor::DType;

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().expect("expected f64")
}

#[test]
fn dynamic_truncate_eval_uses_runtime_size_not_metadata_bound() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let size = TracedTensor::from_vec(vec![], vec![2.0_f64]);
    let mut y = x.dynamic_truncate(&size, 0);

    let binding = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut engine = Engine::new(CpuBackend::new());
    let out = y.eval_with_inputs(&mut engine, &[(&x, &binding)]).unwrap();

    assert_eq!(out.shape(), &[2]);
    assert_eq!(f64_data(out), &[1.0, 2.0]);
}

#[test]
fn scatter_grad_with_symbolic_update_window_does_not_panic() {
    // Build a minimal graph that reaches transpose_scatter with symbolic
    // updates metadata. Use the public traced scatter helper if one exists;
    // otherwise build the fragment directly in tenferro-ops tests and keep
    // this integration test focused on DynamicTruncate.
}
```

If no public scatter helper exists, do not invent one for this task. Keep the
scatter regression in `tenferro-ops/src/ad/tests/indexing_tests.rs`.

**Step 2: Run tests**

Run:

```bash
cargo test -p tenferro dynamic_symbolic_shapes --release
cargo test -p tenferro-ops transpose_scatter --release
```

Expected: PASS.

**Step 3: Commit**

```bash
git add tenferro/tests/dynamic_symbolic_shapes.rs
git commit -m "test: add dynamic symbolic shape regressions"
```

---

### Task 8: Documentation And Verification

**Files:**
- Modify: `docs/design/dynamic-symbolic-shapes.md`
- Modify if needed: `docs/spec/primitive-catalog.md`
- Modify if needed: `docs/spec/backend-contract.md`

**Step 1: Update docs**

Update `docs/design/dynamic-symbolic-shapes.md` to mark the first
implementation slice as implemented:

- `ShapeExtent` internal metadata exists.
- `DynamicTruncate` truncated axis is an upper bound.
- `transpose_scatter` symbolic update-window dimensions use dynamic gather
  slice sizes.
- Runtime-scalar exact extents remain deferred.

If `GatherDynamicSliceSizes` is public in `StdTensorOp`, add it to
`docs/spec/primitive-catalog.md`. If it is internal-only and not exposed as a
user primitive, document it only in `docs/design/dynamic-symbolic-shapes.md`.

**Step 2: Run focused checks**

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-ops --release
cargo test -p tenferro --release dynamic_symbolic_shapes
cargo test -p tenferro --release shape_inference
cargo test -p tenferro --release compiler_wiring
python3 scripts/check-docs-site.py --quiet
```

Expected: all PASS.

**Step 3: Run broader pre-PR checks if time allows**

Run:

```bash
cargo test --workspace --release
cargo doc --workspace --no-deps
```

Expected: all PASS.

**Step 4: Commit**

```bash
git add docs/design/dynamic-symbolic-shapes.md docs/spec/primitive-catalog.md docs/spec/backend-contract.md
git commit -m "docs: record dynamic symbolic shape implementation status"
```

---

## Follow-Up Work

These are intentionally not part of the first implementation change set:

- Add exact runtime-scalar extent expressions for `DynamicTruncate`.
- Make all AD construction paths return structured `Result` values instead of
  relying on infallible `PrimitiveOp` dispatch.
- Make compiler APIs fallible end-to-end for unsupported dynamic-shape rewrites
  that cannot be skipped.
- Audit remaining `constant_value().unwrap_or_else(panic)` sites in AD rules,
  especially `tenferro-ops/src/ad/structural.rs`.
- Extend symbolic config support to `DynamicSlice` if an issue demonstrates the
  same pattern there.
- Add coverage thresholds for the new files once line coverage JSON is
  generated.
