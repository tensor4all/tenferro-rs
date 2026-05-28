# Typed Tensor Rank, View, And Stride Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement issue #934 by making `TypedTensor<T, R>` the canonical rank-generic execution tensor while moving rank, layout, stride, and view metadata validation into `tenferro-tensor-core`.

**Architecture:** `tenferro-tensor-core` owns sealed rank markers and storage-neutral `TensorLayout<R>` validation. `tenferro-tensor` owns host/backend buffers, placement, typed tensors, typed views, same-placement canonicalization, and backend dispatch. Existing public `TypedTensor*` names in core are retired or renamed to host-specific names so `TypedTensor` consistently means GPU-capable execution tensor.

**Tech Stack:** Rust workspace crates `tenferro-tensor-core`, `tenferro-tensor`, `tenferro-runtime`, `tenferro-linalg`, `tenferro-einsum`, `tenferro-fft`, and `tenferro-gpu`; `smallvec`, `num-complex`, `thiserror`, `strided-kernel`, CubeCL CUDA backend.

---

## Task 1: Add Core Rank And Layout Test Targets

**Files:**
- Modify: `tenferro-tensor-core/tests/core.rs`
- Modify: `tenferro-tensor-core/src/lib.rs`
- Create: `tenferro-tensor-core/src/rank.rs`
- Create: `tenferro-tensor-core/src/layout.rs`

**Step 1: Write failing rank and compact-layout tests**

Add tests to `tenferro-tensor-core/tests/core.rs`:

```rust
use tenferro_tensor_core::{DynRank, Rank, TensorLayout, TensorRank};

#[test]
fn dynamic_rank_shape_roundtrips_vec() {
    let shape = <DynRank as TensorRank>::shape_from_vec(vec![2, 3].into()).unwrap();
    assert_eq!(shape.as_ref(), &[2, 3]);
    assert_eq!(
        <DynRank as TensorRank>::shape_into_vec(shape).as_slice(),
        &[2, 3]
    );
}

#[test]
fn static_rank_rejects_wrong_shape_length() {
    let err = <Rank<2> as TensorRank>::shape_from_vec(vec![2, 3, 4].into()).unwrap_err();
    assert!(err.to_string().contains("rank"));
}

#[test]
fn compact_layout_for_static_rank_has_column_major_strides() {
    let layout = TensorLayout::<Rank<2>>::compact([2, 3]).unwrap();
    assert_eq!(layout.shape(), &[2, 3]);
    assert_eq!(layout.strides(), &[1, 2]);
    assert_eq!(layout.offset(), 0);
    assert!(layout.is_compact_col_major());
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor-core --test core
```

Expected: FAIL because `DynRank`, `Rank`, `TensorRank`, and `TensorLayout` are not defined.

**Step 3: Implement the minimal rank and layout modules**

Add `rank.rs` with sealed `TensorRank`, `DynRank`, and `Rank<N>`. Use existing `ShapeVec`, `StrideVec`, `Error`, and `Result` from `lib.rs`.

```rust
use crate::{Error, Result, ShapeVec, StrideVec};
use std::fmt::Debug;

pub trait TensorRank: private::Sealed + Clone + Copy + Debug + Eq + Send + Sync + 'static {
    const RANK: Option<usize>;
    type Shape: Clone + Debug + PartialEq + Eq + AsRef<[usize]>;
    type Strides: Clone + Debug + PartialEq + Eq + AsRef<[isize]>;

    fn shape_from_vec(shape: ShapeVec) -> Result<Self::Shape>;
    fn shape_into_vec(shape: Self::Shape) -> ShapeVec;
    fn strides_from_vec(strides: StrideVec) -> Result<Self::Strides>;
    fn strides_into_vec(strides: Self::Strides) -> StrideVec;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DynRank;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rank<const N: usize>;

mod private {
    pub trait Sealed {}
    impl Sealed for super::DynRank {}
    impl<const N: usize> Sealed for super::Rank<N> {}
}
```

Fill the impls so `DynRank` returns vectors and `Rank<N>` converts with `try_into()` and returns `Error::RankMismatch` on length mismatch.

Add `layout.rs` with:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorLayout<R: TensorRank = DynRank> {
    shape: R::Shape,
    strides: R::Strides,
    offset: isize,
}
```

Implement `compact`, `from_parts`, `shape`, `strides`, `offset`, and `is_compact_col_major`.

Update `lib.rs` to `mod rank; mod layout;` and re-export:

```rust
pub use layout::TensorLayout;
pub use rank::{DynRank, Rank, TensorRank};
```

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor-core --test core
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor-core/src/lib.rs tenferro-tensor-core/src/rank.rs tenferro-tensor-core/src/layout.rs tenferro-tensor-core/tests/core.rs
git commit -m "feat: add tensor rank and layout metadata"
```

## Task 2: Move Core View Bounds To Reachable-Range Validation

**Files:**
- Modify: `tenferro-tensor-core/src/layout.rs`
- Modify: `tenferro-tensor-core/src/lib.rs`
- Modify: `tenferro-tensor-core/tests/core.rs`

**Step 1: Write failing negative-stride and bounds tests**

Add tests:

```rust
use tenferro_tensor_core::{DynRank, TensorLayout};

#[test]
fn layout_accepts_negative_stride_when_reachable_range_is_in_bounds() {
    let layout = TensorLayout::<DynRank>::from_parts(vec![3].into(), vec![-1].into(), 2, 3).unwrap();
    assert_eq!(layout.shape(), &[3]);
    assert_eq!(layout.strides(), &[-1]);
    assert_eq!(layout.offset(), 2);
}

#[test]
fn layout_rejects_negative_stride_when_reachable_range_is_out_of_bounds() {
    assert!(TensorLayout::<DynRank>::from_parts(vec![3].into(), vec![-1].into(), 1, 3).is_err());
}

#[test]
fn layout_rejects_positive_stride_when_max_offset_exceeds_buffer_len() {
    assert!(TensorLayout::<DynRank>::from_parts(vec![3].into(), vec![2].into(), 0, 3).is_err());
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor-core --test core layout_
```

Expected: FAIL because current validation rejects negative strides or does not exist in `TensorLayout`.

**Step 3: Implement reachable-range validation**

In `layout.rs`, implement a helper that computes reachable min and max physical offsets:

```rust
fn reachable_offset_range(shape: &[usize], strides: &[isize], offset: isize) -> Result<Option<(isize, isize)>> {
    if shape.iter().any(|&extent| extent == 0) {
        return Ok(None);
    }
    let mut min = offset;
    let mut max = offset;
    for (&extent, &stride) in shape.iter().zip(strides) {
        let last = isize::try_from(extent.saturating_sub(1)).map_err(|_| Error::IntegerOverflow)?;
        let delta = last.checked_mul(stride).ok_or(Error::IntegerOverflow)?;
        if delta < 0 {
            min = min.checked_add(delta).ok_or(Error::IntegerOverflow)?;
        } else {
            max = max.checked_add(delta).ok_or(Error::IntegerOverflow)?;
        }
    }
    Ok(Some((min, max)))
}
```

Use it from `TensorLayout::from_parts(shape, strides, offset, buffer_len)`. Empty layouts may accept offsets in `0..=buffer_len`; non-empty layouts require `min >= 0` and `max < buffer_len`.

Update existing `validate_view_bounds` in `lib.rs` to call the new helper or to mirror the same logic until old core host names are removed.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor-core --test core layout_
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor-core/src/layout.rs tenferro-tensor-core/src/lib.rs tenferro-tensor-core/tests/core.rs
git commit -m "feat: validate tensor layout reachable ranges"
```

## Task 3: Implement Core Metadata-Only View Transforms

**Files:**
- Modify: `tenferro-tensor-core/src/layout.rs`
- Modify: `tenferro-tensor-core/tests/core.rs`

**Step 1: Write failing transform tests**

Add tests covering transpose, slice, reshape, and broadcast:

```rust
use tenferro_tensor_core::{Rank, SliceSpec, TensorLayout};

#[test]
fn transpose_view_permutes_layout_metadata() {
    let layout = TensorLayout::<Rank<2>>::compact([2, 3]).unwrap();
    let transposed = layout.transpose_view([1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.strides(), &[2, 1]);
    assert_eq!(transposed.offset(), 0);
}

#[test]
fn slice_view_supports_negative_step() {
    let layout = TensorLayout::<Rank<1>>::compact([4]).unwrap();
    let sliced = layout
        .slice_view([SliceSpec { start: 3, end: -1, step: -2 }], 4)
        .unwrap();
    assert_eq!(sliced.shape(), &[2]);
    assert_eq!(sliced.strides(), &[-2]);
    assert_eq!(sliced.offset(), 3);
}

#[test]
fn reshape_view_as_requires_compact_layout() {
    let layout = TensorLayout::<Rank<2>>::compact([2, 3]).unwrap();
    let reshaped = layout.reshape_view_as::<Rank<1>>([6], 6).unwrap();
    assert_eq!(reshaped.shape(), &[6]);
    assert_eq!(reshaped.strides(), &[1]);
}

#[test]
fn broadcast_in_dim_view_uses_zero_strides_for_broadcast_axes() {
    let layout = TensorLayout::<Rank<1>>::compact([3]).unwrap();
    let broadcast = layout
        .broadcast_in_dim_view::<Rank<2>>([2, 3], [1], 3)
        .unwrap();
    assert_eq!(broadcast.shape(), &[2, 3]);
    assert_eq!(broadcast.strides(), &[0, 1]);
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor-core --test core view_
```

Expected: FAIL because transform methods are missing.

**Step 3: Implement transform methods**

Implement on `TensorLayout<R>`:

- `transpose_view<A>(axes: A) -> Result<TensorLayout<R2>>` using rank-preserving arrays when possible
- `slice_view(...)` using normalized start/end/step semantics and reachable-range validation
- `reshape_view_as<R2>(shape: R2::Shape, buffer_len: usize)` only when compact column-major
- `broadcast_in_dim_view<R2>(shape: R2::Shape, broadcast_dims: impl AsRef<[usize]>, buffer_len: usize)`

Prefer a small internal helper that produces `ShapeVec` and `StrideVec`, then converts through `R2::shape_from_vec` and `R2::strides_from_vec`.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor-core --test core view_
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor-core/src/layout.rs tenferro-tensor-core/tests/core.rs
git commit -m "feat: add tensor layout view transforms"
```

## Task 4: Add Mutable Layout No-Overlap Validation

**Files:**
- Modify: `tenferro-tensor-core/src/layout.rs`
- Modify: `tenferro-tensor-core/tests/core.rs`

**Step 1: Write failing mutable layout tests**

Add tests:

```rust
use tenferro_tensor_core::{DynRank, TensorLayout};

#[test]
fn mutable_layout_rejects_zero_stride_broadcast() {
    let layout = TensorLayout::<DynRank>::from_parts(vec![2].into(), vec![0].into(), 0, 1).unwrap();
    assert!(layout.validate_mutable_no_overlap().is_err());
}

#[test]
fn mutable_layout_rejects_overlapping_strides() {
    let layout = TensorLayout::<DynRank>::from_parts(vec![2, 2].into(), vec![1, 1].into(), 0, 4).unwrap();
    assert!(layout.validate_mutable_no_overlap().is_err());
}

#[test]
fn mutable_layout_accepts_reversed_non_overlapping_vector() {
    let layout = TensorLayout::<DynRank>::from_parts(vec![3].into(), vec![-1].into(), 2, 3).unwrap();
    layout.validate_mutable_no_overlap().unwrap();
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor-core --test core mutable_layout
```

Expected: FAIL because no-overlap validation is missing.

**Step 3: Implement conservative no-overlap validation**

Implement `TensorLayout::validate_mutable_no_overlap()`:

1. Return `Ok(())` for empty logical views.
2. Reject any axis with `extent > 1 && stride == 0`.
3. Accept a sufficient sorted absolute-stride span proof:

```rust
let mut axes = shape
    .iter()
    .zip(strides)
    .filter(|(&extent, _)| extent > 1)
    .map(|(&extent, &stride)| (extent, stride.unsigned_abs()))
    .collect::<SmallVec<[(usize, usize); 8]>>();
axes.sort_by_key(|&(_, stride)| stride);
let mut span = 0usize;
for (extent, stride) in axes {
    if stride <= span {
        return exact_or_reject(...);
    }
    span = span
        .checked_add((extent - 1).checked_mul(stride).ok_or(Error::IntegerOverflow)?)
        .ok_or(Error::IntegerOverflow)?;
}
Ok(())
```

4. For small views, enumerate physical offsets in a `HashSet` and reject duplicates.
5. For large views that fail the sufficient proof, return an error instead of accepting.

Keep the exact-enumeration threshold private and documented. Start with a conservative value such as `4096` logical elements.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor-core --test core mutable_layout
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor-core/src/layout.rs tenferro-tensor-core/tests/core.rs
git commit -m "feat: validate mutable tensor layout overlap"
```

## Task 5: Rename Core Host-Only Tensor Types

**Files:**
- Modify: `tenferro-tensor-core/src/lib.rs`
- Modify: `tenferro-tensor-core/tests/core.rs`
- Modify: docs and rustdoc examples that import `tenferro_tensor_core::TypedTensor`

**Step 1: Write failing public-name tests**

In `tenferro-tensor-core/tests/core.rs`, add:

```rust
use tenferro_tensor_core::{HostTensor, HostTensorView};

#[test]
fn host_tensor_uses_host_specific_public_name() {
    let tensor = HostTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let view: HostTensorView<'_, f64> = tensor.as_view();
    assert_eq!(view.shape(), &[2]);
    assert_eq!(view.as_slice().unwrap(), &[1.0, 2.0]);
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor-core host_tensor_uses_host_specific_public_name
```

Expected: FAIL because `HostTensor` and `HostTensorView` are not exported.

**Step 3: Rename host-only core types**

In `tenferro-tensor-core/src/lib.rs`, rename public host-only types:

- `TypedTensor<T>` -> `HostTensor<T, R = DynRank>` if using rank-generic host tensors immediately, or `HostTensor<T>` as an intermediate if the migration is staged inside this task.
- `TypedTensorView<'a, T>` -> `HostTensorView<'a, T, R = DynRank>`
- future mutable host view -> `HostTensorViewMut<'a, T, R = DynRank>`
- `Tensor` variants should wrap `HostTensor<T, DynRank>` after the rank-generic form exists.

Do not leave public aliases named `TypedTensor` in `tenferro-tensor-core`.

Update crate docs and examples in `tenferro-tensor-core/src/lib.rs` to use the host-specific names.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor-core
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor-core/src/lib.rs tenferro-tensor-core/tests/core.rs
git commit -m "refactor: rename core host tensor types"
```

## Task 6: Migrate Execution TypedTensor To Rank-Generic Layout

**Files:**
- Modify: `tenferro-tensor/src/types.rs`
- Modify: `tenferro-tensor/src/types/accessors.rs`
- Modify: `tenferro-tensor/src/types/shape_packing.rs`
- Modify: `tenferro-tensor/src/tests/types_tests.rs`

**Step 1: Write failing rank-generic execution tensor tests**

Add tests:

```rust
use tenferro_tensor::{Rank, TensorLayout, TypedTensor};

#[test]
fn typed_tensor_static_rank_constructs_compact_layout() {
    let tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.layout().strides(), &[1, 2]);
    assert!(tensor.layout().is_compact_col_major());
}

#[test]
fn typed_tensor_owned_layout_is_always_compact() {
    let tensor = TypedTensor::<i32>::from_vec_col_major(vec![3], vec![1, 2, 3]);
    assert_eq!(tensor.layout(), &TensorLayout::compact(vec![3]).unwrap());
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor typed_tensor_
```

Expected: FAIL because `TypedTensor` is not rank-generic and has no `layout()`.

**Step 3: Implement rank-generic `TypedTensor`**

Update `tenferro-tensor/src/types.rs`:

- re-export `DynRank`, `Rank`, `TensorLayout`, and `TensorRank` from `tenferro_tensor_core`
- change `TypedTensor<T>` to `TypedTensor<T, R = DynRank>`
- replace `pub shape: Vec<usize>` with `layout: TensorLayout<R>`
- add `shape()`, `rank()`, `layout()`, and `into_layout()` accessors
- update constructors so owned tensors always call `TensorLayout::compact(shape)`
- keep `Buffer<T>` and `Placement` in `tenferro-tensor`

Update `accessors.rs` and `shape_packing.rs` to call `self.shape()` instead of reaching into `self.shape`.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor typed_tensor_
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/types.rs tenferro-tensor/src/types/accessors.rs tenferro-tensor/src/types/shape_packing.rs tenferro-tensor/src/tests/types_tests.rs
git commit -m "feat: make execution typed tensors rank generic"
```

## Task 7: Update Workspace Shape Field Call Sites

**Files:**
- Modify: `tenferro-tensor/src/**/*.rs`
- Modify: `tenferro-runtime/src/**/*.rs`
- Modify: `tenferro-ad/src/**/*.rs`
- Modify: `tenferro-einsum/src/**/*.rs`
- Modify: `tenferro-linalg/src/**/*.rs`
- Modify: `tenferro-fft/src/**/*.rs`
- Modify: `tenferro-gpu/src/**/*.rs`

**Step 1: Run check to collect compile errors**

Run:

```bash
cargo check --workspace
```

Expected: FAIL with direct `.shape` field access errors where `TypedTensor` now exposes `shape()`.

**Step 2: Mechanically update call sites**

Replace direct `typed.shape` reads with `typed.shape()` or `typed.shape().to_vec()` as needed.

Examples:

```rust
// before
if input.shape.len() != 2 { ... }
let rows = input.shape[0];

// after
if input.shape().len() != 2 { ... }
let rows = input.shape()[0];
```

For constructors that build typed tensors from existing shape vectors:

```rust
TypedTensor::from_vec_col_major(shape, data)
```

Keep passing shape values through the constructor so compact layout is validated there.

**Step 3: Run check again**

Run:

```bash
cargo check --workspace
```

Expected: PASS or only errors related to view-type migration handled in the next task.

**Step 4: Commit**

```bash
git add tenferro-tensor tenferro-runtime tenferro-ad tenferro-einsum tenferro-linalg tenferro-fft tenferro-gpu
git commit -m "refactor: use typed tensor shape accessors"
```

## Task 8: Replace Strided Adapter Types With Canonical Typed Views

**Files:**
- Modify: `tenferro-tensor/src/types.rs`
- Modify: `tenferro-tensor/src/types/strided_view.rs`
- Modify: `tenferro-tensor/src/tests/types_tests.rs`
- Modify: `tenferro-tensor/src/tests/types_tests/strided_dynamic.rs`

**Step 1: Write failing canonical view tests**

Add tests:

```rust
use tenferro_tensor::{Rank, TypedTensor, TypedTensorView, TypedTensorViewMut};

#[test]
fn typed_tensor_as_view_preserves_rank_and_layout() {
    let tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let view = tensor.as_view();
    assert_eq!(view.shape(), &[2, 2]);
    assert_eq!(view.strides(), &[1, 2]);
    assert_eq!(view.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn typed_tensor_view_transpose_is_metadata_only() {
    let tensor = TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 3], vec![1, 2, 3, 4, 5, 6]);
    let view = tensor.as_view().transpose_view([1, 0]).unwrap();
    assert_eq!(view.shape(), &[3, 2]);
    assert_eq!(view.strides(), &[2, 1]);
    assert_eq!(view.get(&[2, 1]), Some(&6));
}

#[test]
fn mutable_typed_tensor_view_rejects_overlapping_layout() {
    let mut data = vec![1_i32, 2, 3, 4];
    assert!(TypedTensorViewMut::from_slice(vec![2, 2], vec![1, 1], 0, &mut data).is_err());
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor typed_tensor_view
```

Expected: FAIL because canonical typed views do not yet support arbitrary strides.

**Step 3: Implement canonical typed views**

In `types.rs`, add:

- `TensorBufferRef<'a, T>`
- `TensorBufferRefMut<'a, T>`
- rank-generic `TypedTensorView<'a, T, R = DynRank>`
- rank-generic `TypedTensorViewMut<'a, T, R = DynRank>`

Implement:

- `TypedTensor::as_view()`
- `TypedTensor::as_view_mut()`
- `TypedTensorView::shape`, `strides`, `offset`, `layout`, `placement`, `get`, `try_linear_offset`, `as_slice`
- `TypedTensorViewMut::get_mut`, `as_read_only`, `into_read_only`
- metadata-only view transforms by delegating to `TensorLayout`

Move useful code from `types/strided_view.rs` into the canonical view implementations. Do not keep public `TypedStridedTensorView*` compatibility aliases unless a compile-blocking downstream API forces a temporary private bridge.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor typed_tensor_view
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/types.rs tenferro-tensor/src/types/strided_view.rs tenferro-tensor/src/tests/types_tests.rs tenferro-tensor/src/tests/types_tests/strided_dynamic.rs
git commit -m "feat: unify typed tensor view surface"
```

## Task 9: Implement Same-Placement Host Canonicalization And Copy-Back

**Files:**
- Modify: `tenferro-tensor/src/types.rs`
- Modify: `tenferro-tensor/src/cpu/mod.rs`
- Modify: `tenferro-tensor/src/cpu/structural.rs`
- Modify: `tenferro-tensor/src/tests/types_tests.rs`
- Modify: `tenferro-tensor/src/tests/cpu_tests/basic_ops.rs`

**Step 1: Write failing host canonicalization tests**

Add tests:

```rust
#[test]
fn non_contiguous_host_view_to_contiguous_preserves_column_major_order() {
    let tensor = TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 3], vec![1, 2, 3, 4, 5, 6]);
    let view = tensor.as_view().transpose_view([1, 0]).unwrap();
    let compact = view.to_contiguous().unwrap();
    assert_eq!(compact.shape(), &[3, 2]);
    assert_eq!(compact.as_slice(), &[1, 3, 5, 2, 4, 6]);
}

#[test]
fn mutable_host_copy_back_writes_strided_output() {
    let mut tensor = TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 2], vec![0, 0, 0, 0]);
    {
        let mut out = tensor.as_view_mut().transpose_view([1, 0]).unwrap();
        let scratch = TypedTensor::<i32, Rank<2>>::from_vec_col_major([2, 2], vec![1, 2, 3, 4]);
        out.copy_from_contiguous(&scratch).unwrap();
    }
    assert_eq!(tensor.as_slice(), &[1, 3, 2, 4]);
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor non_contiguous_host_view
cargo test -p tenferro-tensor mutable_host_copy_back
```

Expected: FAIL because `to_contiguous` and `copy_from_contiguous` are missing.

**Step 3: Implement host canonicalization**

Implement on typed views:

- `to_contiguous() -> Result<TypedTensor<T, R>>` for `T: Clone`
- `copy_from_contiguous(&mut self, src: &TypedTensor<T, R>) -> Result<()>` for `T: Clone`

For host buffers, traverse logical indices using incremental offset helpers where practical. Avoid per-element allocation. Use `Vec::with_capacity`.

For backend buffers, return `Error::BackendFailure` until backend-specific same-placement copies are implemented in Task 10.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor non_contiguous_host_view
cargo test -p tenferro-tensor mutable_host_copy_back
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/types.rs tenferro-tensor/src/cpu/mod.rs tenferro-tensor/src/cpu/structural.rs tenferro-tensor/src/tests/types_tests.rs tenferro-tensor/src/tests/cpu_tests/basic_ops.rs
git commit -m "feat: canonicalize host tensor views"
```

## Task 10: Add Backend Canonicalization Contracts

**Files:**
- Modify: `tenferro-tensor/src/backend.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`
- Modify: `tenferro-gpu/src/lib.rs`
- Modify: `tenferro-gpu/src/**/*.rs`
- Modify: `tenferro-tensor/src/tests/cpu_stub_tests.rs`

**Step 1: Write failing backend contract tests**

Add CPU stub tests showing CPU backend rejects backend buffers instead of downloading:

```rust
#[test]
fn cpu_backend_rejects_backend_view_without_download() {
    let handle: Arc<dyn BackendBuffer<f64>> = Arc::new(BufferHandle::<f64>::new(7));
    let tensor = TypedTensor::<f64>::from_backend_buffer(vec![2], handle, gpu_placement());
    let err = CpuBackend::new().to_contiguous(&tensor.as_view()).unwrap_err();
    assert!(err.to_string().contains("download"));
}
```

Add CUDA ignored tests, if the existing GPU test harness supports them, for same-device view canonicalization:

```rust
#[test]
#[ignore]
fn cuda_to_contiguous_keeps_tensor_on_cuda() {
    // upload compact tensor, make metadata-only transpose view, call to_contiguous
    // assert output placement remains CUDA and downloaded values match expectation
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor cpu_backend_rejects_backend_view_without_download
```

Expected: FAIL because backend canonicalization contract APIs are missing.

**Step 3: Add backend trait hooks**

Add explicit backend operations such as:

```rust
fn canonicalize_view<T: TensorScalar>(&mut self, view: &TypedTensorView<'_, T>) -> Result<TypedTensor<T>>;
fn copy_contiguous_to_view<T: TensorScalar>(&mut self, src: &TypedTensor<T>, dst: &mut TypedTensorViewMut<'_, T>) -> Result<()>;
```

Keep CPU and GPU implementations separate. CPU must reject backend buffers with diagnostics that tell users to download explicitly before CPU execution. GPU must never download to host as part of canonicalization.

**Step 4: Implement CUDA same-device copy path**

In `tenferro-gpu`, add a layout-copy kernel whose launch domain covers the logical output domain. It reads source layout shape/strides/offset and writes compact output, or scatters compact scratch into a mutable output view.

Set `CUBECL_DEBUG_LOG=0` in local test commands to avoid large logs.

**Step 5: Run targeted checks**

Run CPU test:

```bash
cargo test -p tenferro-tensor cpu_backend_rejects_backend_view_without_download
```

Run CUDA ignored test only on a CUDA machine:

```bash
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.8 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
cargo test -p tenferro-gpu --features cuda cuda_to_contiguous_keeps_tensor_on_cuda -- --ignored
```

Expected: PASS where hardware is available; otherwise document skipped CUDA verification.

**Step 6: Commit**

```bash
git add tenferro-tensor/src/backend.rs tenferro-tensor/src/cpu/backend.rs tenferro-gpu tenferro-tensor/src/tests/cpu_stub_tests.rs
git commit -m "feat: add same-placement view canonicalization"
```

## Task 11: Integrate Views At Execution Boundaries

**Files:**
- Modify: `tenferro-tensor/src/cpu/elementwise.rs`
- Modify: `tenferro-tensor/src/cpu/reduction.rs`
- Modify: `tenferro-tensor/src/cpu/gemm/mod.rs`
- Modify: `tenferro-tensor/src/cpu/structural.rs`
- Modify: `tenferro-linalg/src/backend.rs`
- Modify: `tenferro-linalg/src/cpu/backend.rs`
- Modify: `tenferro-linalg/src/cpu/linalg/**/*.rs`
- Modify: relevant tests under `tenferro-tensor/src/tests/` and `tenferro-linalg/src/cpu/tests/`

**Step 1: Write failing execution-boundary tests**

Add tests:

```rust
#[test]
fn elementwise_add_accepts_transposed_host_view_input() {
    let a = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = a.as_view().transpose_view([1, 0]).unwrap();
    let out = tensor::add(&b, &b, &mut CpuBackend::new()).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 6.0, 4.0, 8.0]);
}

#[test]
fn svd_canonicalizes_transposed_host_view_before_lapack() {
    let a = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0, 0.0, 0.0, 2.0]);
    let view = a.as_view().transpose_view([1, 0]).unwrap();
    let outputs = CpuBackend::new().svd_view(&view).unwrap();
    assert_eq!(outputs[1].shape(), &[2]);
}
```

Adjust exact function names to the final public API chosen in earlier tasks.

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor elementwise_add_accepts_transposed_host_view_input
cargo test -p tenferro-linalg svd_canonicalizes_transposed_host_view_before_lapack
```

Expected: FAIL because backend ops still accept only owned compact tensors.

**Step 3: Update CPU execution boundaries**

Use the canonicalization helpers for ops that are compact-only. For ops that can use strided views through `strided-kernel`, pass layout metadata directly instead of copying.

Do not add broad public `ArrayLike`/`TensorLike` traits for linalg. Keep SVD as an execution op and use view-specific overloads only if needed.

**Step 4: Run targeted tests**

Run:

```bash
cargo test -p tenferro-tensor elementwise_add_accepts_transposed_host_view_input
cargo test -p tenferro-linalg svd_canonicalizes_transposed_host_view_before_lapack
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/cpu tenferro-linalg/src tenferro-tensor/src/tests tenferro-linalg/src/cpu/tests
git commit -m "feat: accept typed views at execution boundaries"
```

## Task 12: Update Repository Rules And Public Docs

**Files:**
- Modify: `REPOSITORY_RULES.md`
- Modify: `README.md`
- Modify: `tenferro-tensor-core/src/lib.rs`
- Modify: `tenferro-tensor/src/types.rs`
- Modify: `docs/guides/*.md` files that mention `TypedTensor`, strided views, or memory order
- Modify: `docs/design/` or `docs/architecture/` files that describe tensor-core and tensor boundaries

**Step 1: Update repository rules**

Update these sections in `REPOSITORY_RULES.md`:

- `Public Surface Discipline`: add the tensor operation vocabulary rule and `_view` suffix rule.
- `Dense Layout And Linear Algebra`: state owned tensors remain compact column-major.
- `Range Checks And Slicing`: replace any v1 negative-stride rejection language with reachable-range validation.
- `Device Transfer And Backend Buffer Errors`: state same-placement canonicalization is allowed and hidden CPU-GPU transfer remains forbidden.

**Step 2: Update rustdoc examples**

Every public type/function added in this migration must include `/// # Examples`. Ensure examples compile as doctests and do not use `ignore` or `no_run`.

Examples should show:

- `TypedTensor::<f64, Rank<2>>::from_vec_col_major(...)`
- `TypedTensor::try_into_rank::<2>()`
- metadata-only `transpose_view`
- same-placement `to_contiguous`
- dtype-erased `Tensor` remaining dynamic-rank

**Step 3: Update user docs**

Update README and guides so they do not claim `tenferro-tensor-core::TypedTensor` exists. Explain:

- `TypedTensor<T, R>` can be host-backed or backend-backed.
- `tenferro-tensor-core` owns rank/layout metadata and optional host-only adapters.
- arbitrary strides live on views.
- compact-only operations may canonicalize within the same placement.
- tenferro never silently transfers between CPU and GPU.

**Step 4: Run docs checks**

Run:

```bash
cargo test --doc -p tenferro-tensor-core
cargo test --doc -p tenferro-tensor
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add REPOSITORY_RULES.md README.md tenferro-tensor-core/src/lib.rs tenferro-tensor/src/types.rs docs
git commit -m "docs: document typed tensor rank and view contract"
```

## Task 13: Remove Obsolete Strided Public Surface And Run Contract Tests

**Files:**
- Modify: `tenferro-tensor/src/types.rs`
- Modify: `tenferro-tensor/src/types/strided_view.rs`
- Modify: `tenferro-tensor/src/lib.rs`
- Modify: `tenferro-tensor/src/tests/op_vocabulary_contract_tests.rs`
- Modify: `tenferro-tensor/src/tests/types_tests.rs`

**Step 1: Search for obsolete names**

Run:

```bash
rg -n "TypedStridedTensorView|StridedTensorView|TypedTensorView::new|permute_view|try_permute_axes|tenferro_tensor_core::TypedTensor|tenferro_tensor_core::TypedTensorView"
```

Expected: any remaining matches are either tests that need updating or private migration helpers.

**Step 2: Remove or privatize obsolete APIs**

Remove public exports for obsolete strided adapter names. If a helper remains useful internally, make it `pub(crate)` and move it under the canonical view implementation.

Rename user-facing `permute_view`/`try_permute_axes` APIs to `transpose_view` unless an internal helper is intentionally private.

**Step 3: Run contract tests**

Run:

```bash
cargo test -p tenferro-tensor op_vocabulary_contract_tests
cargo test -p tenferro-tensor types_tests
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tenferro-tensor/src/types.rs tenferro-tensor/src/types/strided_view.rs tenferro-tensor/src/lib.rs tenferro-tensor/src/tests
git commit -m "refactor: remove obsolete strided view surface"
```

## Task 14: Workspace Verification

**Files:**
- No source edits unless verification exposes failures.

**Step 1: Format**

Run:

```bash
cargo fmt --all --check
```

Expected: PASS. If it fails, run `cargo fmt --all`, then rerun the check.

**Step 2: Run targeted crate tests**

Run:

```bash
cargo test -p tenferro-tensor-core
cargo test -p tenferro-tensor
cargo test -p tenferro-linalg
```

Expected: PASS.

**Step 3: Run full release tests**

Run:

```bash
cargo test --workspace --release
```

Expected: PASS. If this is too slow locally, run `cargo nextest run --workspace --release --no-fail-fast` if available and document the substitution.

**Step 4: Run docs and coverage gates**

Run:

```bash
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS. Add focused tests for any modified file below the coverage threshold.

**Step 5: Optional CUDA verification**

Run only on CUDA 12.8+ machines:

```bash
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.8 \
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
cargo test -p tenferro-gpu --features cuda -- --ignored
```

Expected: PASS where hardware is available. If unavailable, document that CUDA ignored tests were not run.

**Step 6: Commit verification-only fixes**

If verification required code/doc fixes:

```bash
git add <fixed files>
git commit -m "fix: resolve typed tensor migration verification issues"
```

If no fixes were required, do not create an empty commit.
