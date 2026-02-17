# tenferro-mdarray Bridge Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a bridge crate providing `From`/`Into` conversions between mdarray `Array<T, DynRank>` and tenferro `Tensor<T>`.

**Architecture:** A thin extension crate `tenferro-mdarray` at `extension/tenferro-mdarray/` implementing `From`/`Into` traits. POC skeleton phase — all bodies are `todo!()`.

**Tech Stack:** Rust, mdarray 0.8 (crates.io), tenferro-tensor, tenferro-algebra, tenferro-device

---

### Task 1: Scaffold crate and register in workspace

**Files:**
- Create: `extension/tenferro-mdarray/Cargo.toml`
- Create: `extension/tenferro-mdarray/src/lib.rs`
- Modify: `Cargo.toml` (workspace root, add member)

**Step 1: Create `extension/tenferro-mdarray/Cargo.toml`**

```toml
[package]
name = "tenferro-mdarray"
version = "0.1.0"
edition = "2024"
rust-version = "1.89"
license = "MIT OR Apache-2.0"
description = "Bridge between mdarray multidimensional arrays and tenferro tensors."
publish = false

[dependencies]
tenferro-tensor = { path = "../../tenferro-tensor" }
tenferro-algebra = { path = "../../tenferro-algebra" }
tenferro-device = { path = "../../tenferro-device" }
mdarray = "0.8"
```

**Step 2: Create minimal `extension/tenferro-mdarray/src/lib.rs`**

```rust
//! Bridge between [mdarray](https://crates.io/crates/mdarray) multidimensional
//! arrays and tenferro tensors.
//!
//! Provides bidirectional `From`/`Into` conversions between
//! [`mdarray::Array<T, DynRank>`] and [`tenferro_tensor::Tensor<T>`],
//! enabling users to work with mdarray as a familiar array interface
//! while using tenferro's einsum and linalg operations for computation.
//!
//! # Examples
//!
//! ```ignore
//! use mdarray::{Array, DynRank};
//! use tenferro_tensor::Tensor;
//!
//! // mdarray → tenferro
//! let a: Array<f64, DynRank> = Array::from_fn([3, 4], |idx| idx[0] as f64);
//! let t: Tensor<f64> = a.into();
//!
//! // tenferro → mdarray
//! let b: Array<f64, DynRank> = t.into();
//! ```
```

**Step 3: Add member to workspace `Cargo.toml`**

Add `"extension/tenferro-mdarray"` to the `members` list, after `"extension/tenferro-burn"`:

```toml
[workspace]
members = [
    # ...
    "extension/tenferro-burn",
    "extension/tenferro-mdarray",
    # ...
]
```

**Step 4: Verify it compiles**

Run: `cargo check -p tenferro-mdarray`
Expected: success (empty crate)

**Step 5: Commit**

```bash
git add extension/tenferro-mdarray/Cargo.toml extension/tenferro-mdarray/src/lib.rs Cargo.toml
git commit -m "feat: scaffold tenferro-mdarray bridge crate"
```

---

### Task 2: Implement From<Array> for Tensor (mdarray → tenferro)

**Files:**
- Modify: `extension/tenferro-mdarray/src/lib.rs`

**Step 1: Add the `From` impl to `lib.rs`**

Append after the crate-level docs:

```rust
use mdarray::{Array, DynRank};
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

/// Convert an mdarray `Array<T, DynRank>` into a tenferro `Tensor<T>`.
///
/// The array data is moved (not copied) when possible. mdarray stores
/// data in row-major (C) order; the resulting tensor preserves this
/// layout with corresponding strides.
///
/// # Examples
///
/// ```ignore
/// use mdarray::{Array, DynRank};
/// use tenferro_tensor::Tensor;
///
/// let a: Array<f64, DynRank> = Array::from_fn([3, 4], |idx| idx[0] as f64);
/// let t: Tensor<f64> = a.into();
/// assert_eq!(t.dims(), &[3, 4]);
/// ```
impl<T: Scalar> From<Array<T, DynRank>> for Tensor<T> {
    fn from(_array: Array<T, DynRank>) -> Self {
        todo!()
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo check -p tenferro-mdarray`
Expected: success

**Step 3: Commit**

```bash
git add extension/tenferro-mdarray/src/lib.rs
git commit -m "feat(tenferro-mdarray): add From<Array<T, DynRank>> for Tensor<T>"
```

---

### Task 3: Implement From<Tensor> for Array (tenferro → mdarray)

**Files:**
- Modify: `extension/tenferro-mdarray/src/lib.rs`

**Step 1: Add the `From` impl to `lib.rs`**

Append after the previous impl:

```rust
/// Convert a tenferro `Tensor<T>` into an mdarray `Array<T, DynRank>`.
///
/// The tensor is first made contiguous in row-major order (matching
/// mdarray's native C layout), then the data is moved into the array.
///
/// # Examples
///
/// ```ignore
/// use mdarray::{Array, DynRank};
/// use tenferro_tensor::Tensor;
/// use tenferro_device::LogicalMemorySpace;
/// use tenferro_tensor::MemoryOrder;
///
/// let t = Tensor::<f64>::zeros(&[2, 3], LogicalMemorySpace::MainMemory, MemoryOrder::RowMajor);
/// let a: Array<f64, DynRank> = t.into();
/// assert_eq!(a.dims(), &[2, 3]);
/// ```
impl<T: Scalar> From<Tensor<T>> for Array<T, DynRank> {
    fn from(_tensor: Tensor<T>) -> Self {
        todo!()
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo check -p tenferro-mdarray`
Expected: success

**Step 3: Commit**

```bash
git add extension/tenferro-mdarray/src/lib.rs
git commit -m "feat(tenferro-mdarray): add From<Tensor<T>> for Array<T, DynRank>"
```

---

### Task 4: Verify full workspace build and update API index

**Files:**
- Modify: `docs/api_index.md` (add tenferro-mdarray entry to Extension Crates section)

**Step 1: Verify full workspace compiles**

Run: `cargo check --workspace`
Expected: success

**Step 2: Add tenferro-mdarray to API index**

In `docs/api_index.md`, add after the `tenferro-burn` entry in the "Extension Crates (extension/)" section:

```markdown
<a id="tenferro-mdarray"></a>
### [tenferro-mdarray](tenferro_mdarray/index.html) <small>(Extension)</small>

Bridge between [mdarray](https://crates.io/crates/mdarray) multidimensional
arrays and tenferro tensors. Provides `From`/`Into` conversions between
`Array<T, DynRank>` and `Tensor<T>` for bidirectional interop.
```

Also add `tenferro-mdarray` to the workspace architecture text block:

```
Extension:  ...
            tenferro-mdarray        mdarray multidimensional array bridge
```

**Step 3: Verify dependency graph script**

Run: `python3 scripts/gen_dep_graph.py`
Expected: output includes `"tenferro-mdarray"` node in `cluster_extension`

**Step 4: Commit**

```bash
git add docs/api_index.md
git commit -m "docs: add tenferro-mdarray to API index"
```

**Step 5: Push**

```bash
git push
```
