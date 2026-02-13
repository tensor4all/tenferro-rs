# Design Doc Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align the tenferro-rs codebase with tensor4all-meta PR #1 — rename crates/traits, add tenferro-algebra, update all APIs. Bodies remain `todo!()`.

**Architecture:** Rename `tenferro-tensorops` → `tenferro-prims`, add new `tenferro-algebra` crate with `HasAlgebra`/`Semiring`/`Standard`, rewrite `TensorPrims<A>` trait with unified plan-based API, update einsum trait bounds.

**Tech Stack:** Rust workspace, strided-rs (git dep), num-complex, thiserror

**Design doc:** `docs/plans/2026-02-13-design-doc-alignment-design.md`

---

### Task 1: Rename tenferro-tensorops directory to tenferro-prims

**Files:**
- Rename: `tenferro-tensorops/` → `tenferro-prims/`
- Modify: `Cargo.toml` (root, line 3)
- Modify: `tenferro-prims/Cargo.toml` (line 2)
- Modify: `tenferro-einsum/Cargo.toml` (line 11)

**Step 1: git mv the directory**

```bash
git mv tenferro-tensorops tenferro-prims
```

**Step 2: Update root Cargo.toml members**

Change line 3 in `Cargo.toml`:
```toml
# Before:
members = [
    "tenferro-device",
    "tenferro-tensorops",
    "tenferro-tensor",
    "tenferro-einsum",
]

# After:
members = [
    "tenferro-device",
    "tenferro-prims",
    "tenferro-tensor",
    "tenferro-einsum",
]
```

**Step 3: Update tenferro-prims/Cargo.toml package name**

In `tenferro-prims/Cargo.toml`, change:
```toml
# Before:
name = "tenferro-tensorops"
description = "cuTENSOR-compatible tensor operation protocol (TensorOps trait) for the tenferro workspace."

# After:
name = "tenferro-prims"
description = "Tensor primitive operations (TensorPrims trait) for the tenferro workspace."
```

**Step 4: Update tenferro-einsum/Cargo.toml dependency**

In `tenferro-einsum/Cargo.toml`, change line 11:
```toml
# Before:
tenferro-tensorops = { path = "../tenferro-tensorops" }

# After:
tenferro-prims = { path = "../tenferro-prims" }
```

**Step 5: Verify build compiles**

```bash
cargo build
```

Expected: Build succeeds (tenferro-einsum doesn't actually use tenferro-tensorops types in its `todo!()` body, so the rename should be seamless. If there are unused import warnings, that's OK for now.)

**Step 6: Commit**

```bash
git add -A && git commit -m "refactor: rename tenferro-tensorops to tenferro-prims"
```

---

### Task 2: Create tenferro-algebra crate

**Files:**
- Create: `tenferro-algebra/Cargo.toml`
- Create: `tenferro-algebra/src/lib.rs`
- Modify: `Cargo.toml` (root — add to members)

**Step 1: Create Cargo.toml**

Create `tenferro-algebra/Cargo.toml`:
```toml
[package]
name = "tenferro-algebra"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "Algebra traits (HasAlgebra, Semiring, Standard) for the tenferro workspace."
publish = false

[dependencies]
strided-traits = { git = "https://github.com/tensor4all/strided-rs" }
num-complex = "0.4"
```

**Step 2: Create src/lib.rs**

Create `tenferro-algebra/src/lib.rs`:
```rust
//! Algebra traits for the tenferro workspace.
//!
//! This crate provides the minimal algebra foundation for [`TensorPrims<A>`](tenferro_prims::TensorPrims):
//!
//! - [`HasAlgebra`]: Maps a scalar type `T` to its default algebra `A`.
//!   Enables automatic inference: `Tensor<f64>` → `Standard`,
//!   `Tensor<MaxPlus<f64>>` → `MaxPlus` (in external crate).
//! - [`Semiring`]: Defines zero, one, add, mul for algebra-generic operations.
//! - [`Standard`]: Standard arithmetic algebra (add = `+`, mul = `*`).
//!
//! # Extensibility
//!
//! External crates define new algebras by implementing `HasAlgebra` for their
//! scalar types and `TensorPrims<MyAlgebra>` for `CpuBackend` (orphan rule
//! compatible). For example, `tenferro-tropical` defines `MaxPlus<T>`.
//!
//! # Examples
//!
//! ```
//! use tenferro_algebra::{HasAlgebra, Standard};
//!
//! // f64 maps to Standard algebra automatically
//! fn check_algebra<T: HasAlgebra<Algebra = Standard>>() {}
//! check_algebra::<f64>();
//! check_algebra::<f32>();
//! ```

use num_complex::{Complex32, Complex64};
use strided_traits::ScalarBase;

/// Maps a scalar type `T` to its default algebra `A`.
///
/// Enables automatic algebra inference: `Tensor<f64>` → `Standard`,
/// `Tensor<MaxPlus<f64>>` → `MaxPlus` (in external crate).
///
/// # Implementing for custom types
///
/// ```ignore
/// struct MyScalar(f64);
/// struct MyAlgebra;
///
/// impl HasAlgebra for MyScalar {
///     type Algebra = MyAlgebra;
/// }
/// ```
pub trait HasAlgebra {
    /// The algebra associated with this scalar type.
    type Algebra;
}

/// Standard arithmetic algebra (add = `+`, mul = `*`).
///
/// This is the default algebra for built-in numeric types (`f32`, `f64`,
/// `Complex32`, `Complex64`).
pub struct Standard;

impl HasAlgebra for f32 {
    type Algebra = Standard;
}

impl HasAlgebra for f64 {
    type Algebra = Standard;
}

impl HasAlgebra for Complex32 {
    type Algebra = Standard;
}

impl HasAlgebra for Complex64 {
    type Algebra = Standard;
}

/// Semiring trait for algebra-generic operations.
///
/// Defines the four fundamental operations needed for tensor contractions
/// under a given algebra:
///
/// - `zero()`: Additive identity
/// - `one()`: Multiplicative identity
/// - `add(a, b)`: Semiring addition (e.g., `+` for Standard, `max` for MaxPlus)
/// - `mul(a, b)`: Semiring multiplication (e.g., `*` for Standard, `+` for MaxPlus)
///
/// # Examples
///
/// Standard arithmetic:
/// - `zero() = 0`, `one() = 1`, `add = +`, `mul = *`
///
/// Tropical (MaxPlus) semiring (in external crate):
/// - `zero() = -∞`, `one() = 0`, `add = max`, `mul = +`
pub trait Semiring {
    /// The scalar type for this semiring.
    type Scalar: ScalarBase;

    /// Additive identity element.
    fn zero() -> Self::Scalar;

    /// Multiplicative identity element.
    fn one() -> Self::Scalar;

    /// Semiring addition.
    fn add(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;

    /// Semiring multiplication.
    fn mul(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
}
```

**Step 3: Add to root Cargo.toml**

In `Cargo.toml`, add `"tenferro-algebra"` to members:
```toml
members = [
    "tenferro-device",
    "tenferro-algebra",
    "tenferro-prims",
    "tenferro-tensor",
    "tenferro-einsum",
]
```

**Step 4: Verify build**

```bash
cargo build
```

Expected: Build succeeds.

**Step 5: Run tests**

```bash
cargo test -p tenferro-algebra
```

Expected: The doc test for `HasAlgebra` should pass.

**Step 6: Commit**

```bash
git add -A && git commit -m "feat: add tenferro-algebra crate with HasAlgebra, Semiring, Standard"
```

---

### Task 3: Rewrite tenferro-prims with TensorPrims\<A\> API

**Files:**
- Modify: `tenferro-prims/Cargo.toml` (add tenferro-algebra dep)
- Modify: `tenferro-prims/src/lib.rs` (full rewrite)

**Step 1: Add tenferro-algebra dependency**

In `tenferro-prims/Cargo.toml`, add:
```toml
[dependencies]
tenferro-device = { path = "../tenferro-device" }
tenferro-algebra = { path = "../tenferro-algebra" }
strided-view = { git = "https://github.com/tensor4all/strided-rs" }
strided-traits = { git = "https://github.com/tensor4all/strided-rs" }
```

**Step 2: Rewrite src/lib.rs**

Replace the entire content of `tenferro-prims/src/lib.rs` with:

```rust
//! Tensor primitive operations for the tenferro workspace.
//!
//! This crate defines the [`TensorPrims<A>`] trait, a backend-agnostic interface
//! parameterized by algebra `A`. The API follows the cuTENSOR plan-based execution
//! pattern:
//!
//! 1. Create a [`PrimDescriptor`] specifying the operation and index modes
//! 2. Build a plan via [`TensorPrims::plan`] (pre-computes kernel selection)
//! 3. Execute the plan via [`TensorPrims::execute`]
//!
//! # Operation categories
//!
//! **Core operations** (every backend must implement):
//! - [`BatchedGemm`](PrimDescriptor::BatchedGemm): Batched matrix multiplication
//! - [`Reduce`](PrimDescriptor::Reduce): Sum/max/min reduction over modes
//! - [`Trace`](PrimDescriptor::Trace): Trace (contraction of paired diagonal modes)
//! - [`Permute`](PrimDescriptor::Permute): Mode reordering
//! - [`AntiTrace`](PrimDescriptor::AntiTrace): Scatter-add to diagonal (AD backward of trace)
//! - [`AntiDiag`](PrimDescriptor::AntiDiag): Write to diagonal positions (AD backward of diag)
//!
//! **Extended operations** (dynamically queried via [`TensorPrims::has_extension_for`]):
//! - [`Contract`](PrimDescriptor::Contract): Fused permute + GEMM contraction
//! - [`ElementwiseMul`](PrimDescriptor::ElementwiseMul): Element-wise multiplication
//!
//! # Algebra parameterization
//!
//! `TensorPrims<A>` is parameterized by algebra `A` (e.g., `Standard`, `MaxPlus`).
//! External crates implement `TensorPrims<MyAlgebra> for CpuBackend` (orphan rule
//! compatible). The [`HasAlgebra`](tenferro_algebra::HasAlgebra) trait on scalar types
//! enables automatic inference: `Tensor<f64>` → `Standard`.
//!
//! # Backend dispatch
//!
//! `TensorPrims<A>` is used **internally** for dispatch. User-facing APIs
//! (e.g., `tenferro-einsum`) select the backend automatically based on
//! the tensor's [`Device`](tenferro_device::Device).
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_prims::{CpuBackend, TensorPrims, PrimDescriptor, ReduceOp, Extension};
//! use tenferro_algebra::Standard;
//! use strided_view::StridedArray;
//!
//! // Plan + execute: GEMM
//! let desc = PrimDescriptor::BatchedGemm {
//!     batch_dims: vec![], m: 3, n: 5, k: 4,
//! };
//! let plan = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[4, 5], &[3, 5]]).unwrap();
//! CpuBackend::execute(&plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut()).unwrap();
//!
//! // Plan + execute: Reduction
//! let desc = PrimDescriptor::Reduce {
//!     modes_a: vec![0, 1], modes_c: vec![0], op: ReduceOp::Sum,
//! };
//! let plan = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[3]]).unwrap();
//! CpuBackend::execute(&plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut()).unwrap();
//!
//! // Dynamic extension check
//! if CpuBackend::has_extension_for::<f64>(Extension::Contract) {
//!     let desc = PrimDescriptor::Contract {
//!         modes_a: vec![0, 1], modes_b: vec![1, 2], modes_c: vec![0, 2],
//!     };
//!     let plan = CpuBackend::plan::<f64>(&desc, &[&[3, 4], &[4, 5], &[3, 5]]).unwrap();
//!     CpuBackend::execute(&plan, 1.0, &[&a.view(), &b.view()], 0.0, &mut c.view_mut()).unwrap();
//! }
//! ```

use std::marker::PhantomData;

use strided_traits::ScalarBase;
use strided_view::{StridedView, StridedViewMut};
use tenferro_algebra::Standard;
use tenferro_device::Result;

/// Reduction operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum reduction.
    Sum,
    /// Maximum value reduction.
    Max,
    /// Minimum value reduction.
    Min,
}

/// Extended operation identifiers for dynamic capability query.
///
/// Used with [`TensorPrims::has_extension_for`] to check at runtime whether
/// a backend supports an optimized extended operation for a given scalar type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extension {
    /// Fused contraction (permute + GEMM). Maps to `cutensorContract` on GPU.
    Contract,
    /// Element-wise multiplication. Maps to `cutensorElementwiseBinary` on GPU.
    ElementwiseMul,
}

/// Describes a tensor primitive operation.
///
/// All operations follow the cuTENSOR pattern: describe → plan → execute.
/// Core operations must be supported by every backend. Extended operations
/// are dynamically queried via [`TensorPrims::has_extension_for`].
///
/// Modes are `u32` integer labels matching cuTENSOR conventions. Modes
/// shared between input and output tensors are batch/free dimensions;
/// modes present only in inputs are contracted.
pub enum PrimDescriptor {
    // ====================================================================
    // Core operations (every backend must implement)
    // ====================================================================

    /// Batched matrix multiplication.
    ///
    /// `C[batch, m, n] = alpha * A[batch, m, k] * B[batch, k, n] + beta * C[batch, m, n]`
    BatchedGemm {
        /// Batch dimension sizes.
        batch_dims: Vec<usize>,
        /// Number of rows in A / C.
        m: usize,
        /// Number of columns in B / C.
        n: usize,
        /// Contraction dimension (columns of A / rows of B).
        k: usize,
    },

    /// Reduction over modes not present in the output.
    ///
    /// `C[modes_c] = alpha * reduce_op(A[modes_a]) + beta * C[modes_c]`
    Reduce {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C (subset of modes_a).
        modes_c: Vec<u32>,
        /// Reduction operation (Sum, Max, Min).
        op: ReduceOp,
    },

    /// Trace: contraction of paired diagonal modes.
    ///
    /// For each pair `(i, j)`, sums over the diagonal where mode i == mode j.
    Trace {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
        /// Pairs of modes to trace over.
        paired: Vec<(u32, u32)>,
    },

    /// Permute (reorder) tensor modes.
    ///
    /// `B[modes_b] = alpha * A[modes_a]`
    Permute {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor B (same labels, different order).
        modes_b: Vec<u32>,
    },

    /// Anti-trace: scatter-add gradient to diagonal (AD backward of trace).
    AntiTrace {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
        /// Pairs of modes for diagonal scatter.
        paired: Vec<(u32, u32)>,
    },

    /// Anti-diag: write gradient to diagonal positions (AD backward of diag).
    AntiDiag {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
        /// Pairs of modes for diagonal write.
        paired: Vec<(u32, u32)>,
    },

    // ====================================================================
    // Extended operations (dynamically queried)
    // ====================================================================

    /// Fused contraction: permute + GEMM in one operation.
    ///
    /// `C[modes_c] = alpha * contract(A[modes_a], B[modes_b]) + beta * C[modes_c]`
    ///
    /// Available when `has_extension_for::<T>(Extension::Contract)` returns true.
    Contract {
        /// Mode labels for input tensor A.
        modes_a: Vec<u32>,
        /// Mode labels for input tensor B.
        modes_b: Vec<u32>,
        /// Mode labels for output tensor C.
        modes_c: Vec<u32>,
    },

    /// Element-wise multiplication of two tensors.
    ///
    /// Available when `has_extension_for::<T>(Extension::ElementwiseMul)` returns true.
    ElementwiseMul,
}

/// Backend trait for tensor primitive operations, parameterized by algebra `A`.
///
/// Provides a cuTENSOR-compatible plan-based execution model for all
/// operations. Core ops (batched_gemm, reduce, trace, permute, anti_trace,
/// anti_diag) must be implemented. Extended ops (contract, elementwise_mul)
/// have default implementations that decompose into core ops.
///
/// # Algebra parameterization
///
/// The algebra parameter `A` enables extensibility: external crates can
/// implement `TensorPrims<MyAlgebra> for CpuBackend` (orphan rule compatible).
///
/// # Associated functions (not methods)
///
/// All functions are associated functions (no `&self` receiver). Call as
/// `CpuBackend::plan::<f64>(...)` instead of `backend.plan(...)`.
pub trait TensorPrims<A> {
    /// Backend-specific plan type (no type erasure).
    type Plan<T: ScalarBase>;

    /// Create an execution plan from an operation descriptor.
    ///
    /// The plan pre-computes kernel selection and workspace sizes.
    /// `shapes` contains the shape of each tensor involved in the operation
    /// (inputs first, then output).
    fn plan<T: ScalarBase>(
        desc: &PrimDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan<T>>;

    /// Execute a plan with the given scaling factors and tensor views.
    ///
    /// Follows the BLAS/cuTENSOR pattern:
    /// `output = alpha * op(inputs) + beta * output`
    fn execute<T: ScalarBase>(
        plan: &Self::Plan<T>,
        alpha: T,
        inputs: &[&StridedView<T>],
        beta: T,
        output: &mut StridedViewMut<T>,
    ) -> Result<()>;

    /// Query whether an extended operation is available for scalar type `T`.
    ///
    /// Returns `true` if the backend supports the given extended operation
    /// for the specified scalar type. This enables dynamic dispatch:
    /// GPU may support Contract for f64 but not for tropical types.
    fn has_extension_for<T: ScalarBase>(ext: Extension) -> bool;
}

/// CPU plan — concrete enum, no type erasure.
pub enum CpuPlan<T: ScalarBase> {
    /// Plan for batched GEMM.
    BatchedGemm {
        /// Number of rows.
        m: usize,
        /// Number of columns.
        n: usize,
        /// Contraction dimension.
        k: usize,
        _marker: PhantomData<T>,
    },
    /// Plan for reduction.
    Reduce {
        /// Axis to reduce over.
        axis: usize,
        /// Reduction operation.
        op: ReduceOp,
        _marker: PhantomData<T>,
    },
    /// Plan for trace.
    Trace {
        /// Paired modes.
        paired: Vec<(u32, u32)>,
        _marker: PhantomData<T>,
    },
    /// Plan for permutation.
    Permute {
        /// Permutation mapping.
        perm: Vec<usize>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-trace (AD backward).
    AntiTrace {
        /// Paired modes.
        paired: Vec<(u32, u32)>,
        _marker: PhantomData<T>,
    },
    /// Plan for anti-diag (AD backward).
    AntiDiag {
        /// Paired modes.
        paired: Vec<(u32, u32)>,
        _marker: PhantomData<T>,
    },
    /// Plan for fused contraction (extended op).
    Contract {
        _marker: PhantomData<T>,
    },
    /// Plan for element-wise multiplication (extended op).
    ElementwiseMul {
        _marker: PhantomData<T>,
    },
}

/// CPU backend using strided-kernel and GEMM.
///
/// Dispatched automatically when tensors reside on [`Device::Cpu`](tenferro_device::Device::Cpu).
/// Implements [`TensorPrims<Standard>`] for standard arithmetic.
pub struct CpuBackend;

impl TensorPrims<Standard> for CpuBackend {
    type Plan<T: ScalarBase> = CpuPlan<T>;

    fn plan<T: ScalarBase>(
        _desc: &PrimDescriptor,
        _shapes: &[&[usize]],
    ) -> Result<CpuPlan<T>> {
        todo!()
    }

    fn execute<T: ScalarBase>(
        _plan: &CpuPlan<T>,
        _alpha: T,
        _inputs: &[&StridedView<T>],
        _beta: T,
        _output: &mut StridedViewMut<T>,
    ) -> Result<()> {
        todo!()
    }

    fn has_extension_for<T: ScalarBase>(_ext: Extension) -> bool {
        todo!()
    }
}
```

**Step 3: Verify build**

```bash
cargo build
```

Expected: Build succeeds.

**Step 4: Commit**

```bash
git add -A && git commit -m "refactor: rewrite tenferro-prims with TensorPrims<A> and PrimDescriptor"
```

---

### Task 4: Update tenferro-tensor doc comment

**Files:**
- Modify: `tenferro-tensor/src/lib.rs` (line 12 — references `TensorOps`)

**Step 1: Fix doc reference**

In `tenferro-tensor/src/lib.rs`, line 12, change:
```rust
// Before:
//!   [`TensorOps`](tenferro_tensorops::TensorOps) backends

// After:
//!   [`TensorPrims`](tenferro_prims::TensorPrims) backends
```

**Step 2: Verify build**

```bash
cargo build
```

**Step 3: Commit**

```bash
git add -A && git commit -m "docs: update tenferro-tensor doc reference to TensorPrims"
```

---

### Task 5: Update tenferro-einsum with HasAlgebra trait bounds

**Files:**
- Modify: `tenferro-einsum/Cargo.toml` (add tenferro-algebra dep)
- Modify: `tenferro-einsum/src/lib.rs` (trait bounds + imports)

**Step 1: Add tenferro-algebra dependency to Cargo.toml**

In `tenferro-einsum/Cargo.toml`:
```toml
[dependencies]
tenferro-device = { path = "../tenferro-device" }
tenferro-algebra = { path = "../tenferro-algebra" }
tenferro-prims = { path = "../tenferro-prims" }
tenferro-tensor = { path = "../tenferro-tensor" }
strided-traits = { git = "https://github.com/tensor4all/strided-rs" }
```

**Step 2: Update imports in src/lib.rs**

At the top of `tenferro-einsum/src/lib.rs`, change:
```rust
// Before:
use strided_traits::ScalarBase;
use tenferro_device::Result;
use tenferro_tensor::Tensor;

// After:
use strided_traits::ScalarBase;
use tenferro_algebra::HasAlgebra;
use tenferro_device::Result;
use tenferro_tensor::Tensor;
```

**Step 3: Update einsum function signatures**

Change three function signatures to add `+ HasAlgebra` bound:

```rust
// Before:
pub fn einsum<T: ScalarBase>(subscripts: &str, operands: &[&Tensor<T>]) -> Result<Tensor<T>> {

// After:
pub fn einsum<T: ScalarBase + HasAlgebra>(subscripts: &str, operands: &[&Tensor<T>]) -> Result<Tensor<T>> {
```

```rust
// Before:
pub fn einsum_with_subscripts<T: ScalarBase>(

// After:
pub fn einsum_with_subscripts<T: ScalarBase + HasAlgebra>(
```

```rust
// Before:
pub fn einsum_with_plan<T: ScalarBase>(

// After:
pub fn einsum_with_plan<T: ScalarBase + HasAlgebra>(
```

**Step 4: Verify build**

```bash
cargo build
```

Expected: Build succeeds.

**Step 5: Commit**

```bash
git add -A && git commit -m "refactor: add HasAlgebra trait bound to einsum functions"
```

---

### Task 6: Update AGENTS.md and final verification

**Files:**
- Modify: `AGENTS.md`

**Step 1: AGENTS.md is already updated**

The AGENTS.md was updated at the start of this session. Verify it reflects the current state (tenferro-prims, tenferro-algebra, TensorPrims<A>).

**Step 2: Run cargo fmt**

```bash
cargo fmt
```

**Step 3: Run full verification**

```bash
cargo fmt --check && cargo test
```

Expected: Both pass. `cargo test` runs doc tests (the `HasAlgebra` doc test in tenferro-algebra should pass). Other crates have no runnable tests (all bodies are `todo!()`).

**Step 4: Commit if any formatting changes**

```bash
git add -A && git commit -m "style: apply cargo fmt"
```

(Skip if no changes.)
