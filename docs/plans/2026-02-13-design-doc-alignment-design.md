# Design: Align Codebase with tensor4all-meta PR #1

Date: 2026-02-13

## Goal

Align the tenferro-rs codebase with the updated design documents in
[tensor4all/tensor4all-meta PR #1](https://github.com/tensor4all/tensor4all-meta/pull/1).
This is an API-only change: all function bodies remain `todo!()`.

## Decisions

- **Rename** `tenferro-tensorops` → `tenferro-prims` (via `git mv`)
- **Rename** `TensorOps` → `TensorPrims<A>` (algebra-parameterized)
- **Rename** `OpDescriptor` → `PrimDescriptor`
- **New crate** `tenferro-algebra` (HasAlgebra, Semiring, Standard)
- **`tenferro-tensor`** does NOT depend on `tenferro-algebra`
- **`Semiring`** trait defined in POC (with `todo!()` impls)
- **`num-complex`** dependency added to `tenferro-algebra` for Complex32/Complex64 impls

## Crate Structure (after)

```
tenferro-device      — Device enum, Error, Result
tenferro-algebra     — HasAlgebra, Semiring, Standard  [NEW]
tenferro-prims       — TensorPrims<A>, PrimDescriptor, CpuBackend  [RENAMED from tenferro-tensorops]
tenferro-tensor      — Tensor<T>, DataBuffer, view ops  [unchanged API]
tenferro-einsum      — einsum(), Subscripts, ContractionTree  [updated trait bounds]
```

## Dependency Graph

```
tenferro-device (← strided-view, thiserror)
    │
    ↓
tenferro-algebra (← strided-traits, num-complex)
    │
    ├─────────────┐
    ↓             │
tenferro-prims   │   tenferro-tensor
    │  (← strided-view,    │  (← tenferro-device,
    │   strided-traits,     │   strided-view,
    │   tenferro-algebra)   │   strided-traits,
    │                       │   num-traits)
    └──────────┬────────────┘
               ↓
          tenferro-einsum
              (← tenferro-prims, tenferro-tensor,
               tenferro-device, strided-traits)
```

Note: `tenferro-tensor` does NOT depend on `tenferro-algebra`.

## API: tenferro-algebra

```rust
use num_complex::{Complex32, Complex64};
use strided_traits::ScalarBase;

pub trait HasAlgebra {
    type Algebra;
}

pub struct Standard;

impl HasAlgebra for f32 { type Algebra = Standard; }
impl HasAlgebra for f64 { type Algebra = Standard; }
impl HasAlgebra for Complex32 { type Algebra = Standard; }
impl HasAlgebra for Complex64 { type Algebra = Standard; }

pub trait Semiring {
    type Scalar: ScalarBase;
    fn zero() -> Self::Scalar;
    fn one() -> Self::Scalar;
    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
}
```

## API: tenferro-prims (renamed from tenferro-tensorops)

Replaces the old `TensorOps` / `ContractionDescriptor` / `ContractionPlan` API.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp { Sum, Max, Min }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension { Contract, ElementwiseMul }

pub enum PrimDescriptor {
    // Core
    BatchedGemm { batch_dims: Vec<usize>, m: usize, n: usize, k: usize },
    Reduce { modes_a: Vec<u32>, modes_c: Vec<u32>, op: ReduceOp },
    Trace { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    Permute { modes_a: Vec<u32>, modes_b: Vec<u32> },
    AntiTrace { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    AntiDiag { modes_a: Vec<u32>, modes_c: Vec<u32>, paired: Vec<(u32, u32)> },
    // Extended (dynamically queried)
    Contract { modes_a: Vec<u32>, modes_b: Vec<u32>, modes_c: Vec<u32> },
    ElementwiseMul,
}

pub trait TensorPrims<A> {
    type Plan<T: ScalarBase>;

    fn plan<T: ScalarBase>(
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan<T>>;

    fn execute<T: ScalarBase>(
        plan: &Self::Plan<T>,
        alpha: T,
        inputs: &[&StridedView<T>],
        beta: T,
        output: &mut StridedViewMut<T>,
    ) -> Result<()>;

    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool;
}

pub struct CpuBackend;

impl TensorPrims<Standard> for CpuBackend {
    type Plan<T: ScalarBase> = CpuPlan<T>;
    // all methods: todo!()
}
```

## API: tenferro-einsum changes

Add `HasAlgebra` trait bound to all einsum functions:

```rust
pub fn einsum<T: ScalarBase + HasAlgebra>(
    subscripts: &str,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

pub fn einsum_with_subscripts<T: ScalarBase + HasAlgebra>(
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;

pub fn einsum_with_plan<T: ScalarBase + HasAlgebra>(
    tree: &ContractionTree,
    operands: &[&Tensor<T>],
) -> Result<Tensor<T>>;
```

## API: tenferro-tensor

No API changes. Only dependency references updated if needed.
