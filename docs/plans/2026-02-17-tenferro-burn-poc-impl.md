# tenferro-burn POC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create an `extension/tenferro-burn` crate that compiles with Burn Backend Extension Trait pattern — all `todo!()` bodies, validating the type-level wiring between Burn's AD system and tenferro's einsum.

**Architecture:** `TensorNetworkOps` trait extends Burn's `Backend`. Concrete forward impl for `NdArray<f64>`, autodiff impl for `Autodiff<B, C>` with `Backward` trait skeleton. Conversion functions bridge Burn ↔ tenferro tensor types.

**Tech Stack:** Rust, burn 0.21.0-pre.1 (git dep from local clone or crates.io pre-release), tenferro-tensor, tenferro-einsum

---

### Task 1: Create crate scaffold and Cargo.toml

**Files:**
- Create: `extension/tenferro-burn/Cargo.toml`
- Create: `extension/tenferro-burn/src/lib.rs` (empty placeholder)
- Modify: `Cargo.toml` (workspace root — add member)

**Step 1: Create directory**

```bash
mkdir -p extension/tenferro-burn/src
```

**Step 2: Write Cargo.toml**

Create `extension/tenferro-burn/Cargo.toml`:

```toml
[package]
name = "tenferro-burn"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "Bridge between Burn deep learning framework and tenferro tensor network operations."
publish = false

[dependencies]
tenferro-tensor = { path = "../../tenferro-tensor" }
tenferro-einsum = { path = "../../tenferro-einsum" }
tenferro-device = { path = "../../tenferro-device" }
tenferro-algebra = { path = "../../tenferro-algebra" }

burn = { version = "0.21.0-pre.1", default-features = false, features = ["ndarray", "autodiff"] }
```

Note: If `0.21.0-pre.1` is not resolvable from crates.io, fall back to a git
dependency: `burn = { git = "https://github.com/tracel-ai/burn", default-features = false, features = ["ndarray", "autodiff"] }`.

**Step 3: Write minimal lib.rs**

Create `extension/tenferro-burn/src/lib.rs`:

```rust
//! Bridge between Burn and tenferro tensor network operations.

mod convert;
mod forward;
mod backward;
```

**Step 4: Create empty module files**

Create `extension/tenferro-burn/src/convert.rs`:

```rust
//! Burn tensor ↔ tenferro tensor conversion.
```

Create `extension/tenferro-burn/src/forward.rs`:

```rust
//! Forward implementations for concrete backends.
```

Create `extension/tenferro-burn/src/backward.rs`:

```rust
//! Autodiff backward implementations.
```

**Step 5: Add to workspace**

In root `Cargo.toml`, add `"extension/tenferro-burn"` to the `members` list:

```toml
members = [
    # core
    "tenferro-device",
    "tenferro-algebra",
    "tenferro-prims",
    "tenferro-tensor",
    "tenferro-einsum",
    "tenferro-linalg",
    "tenferro-capi",
    # extension
    "extension/tenferro-tropical",
    "extension/tenferro-tropical-capi",
    "extension/tenferro-burn",
    # extern (general-purpose, non-tenferro)
    "extern/chainrules-core",
    "extern/chainrules",
]
```

**Step 6: Verify compilation**

Run: `cargo build -p tenferro-burn`
Expected: Compiles successfully (empty crate)

**Step 7: Commit**

```bash
git add extension/tenferro-burn/ Cargo.toml
git commit -m "feat(tenferro-burn): scaffold crate with Burn dependency"
```

---

### Task 2: Define TensorNetworkOps trait and public einsum API

**Files:**
- Modify: `extension/tenferro-burn/src/lib.rs`

**Step 1: Write the trait and public API**

Replace `extension/tenferro-burn/src/lib.rs` with:

```rust
//! Bridge between Burn and tenferro tensor network operations.
//!
//! This crate provides [`TensorNetworkOps`], a Burn backend extension trait
//! that adds tensor network operations (einsum, SVD, etc.) to any Burn backend.
//! Forward computation delegates to tenferro-einsum; backward passes use
//! tenferro's rrule (VJP) logic, integrated into Burn's autodiff tape.
//!
//! # Examples
//!
//! ```ignore
//! use burn::tensor::Tensor;
//! use burn::backend::{Autodiff, NdArray};
//! use tenferro_burn::{TensorNetworkOps, einsum};
//!
//! type B = Autodiff<NdArray<f64>>;
//!
//! let a = Tensor::<B, 2>::ones([2, 3], &Default::default());
//! let b = Tensor::<B, 2>::ones([3, 4], &Default::default());
//! let c = einsum::<B, 2>("ij,jk->ik", vec![a, b]);
//! ```

pub mod convert;
mod forward;
mod backward;

use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;

/// Extension trait adding tensor network operations to a Burn backend.
///
/// Implement this trait for concrete backends (e.g., `NdArray<f64>`) to provide
/// the forward computation, and for `Autodiff<B, C>` to provide backward
/// (gradient) computation.
///
/// # Examples
///
/// ```ignore
/// use tenferro_burn::TensorNetworkOps;
/// use burn::backend::NdArray;
///
/// // NdArray<f64> implements TensorNetworkOps
/// let out = <NdArray<f64> as TensorNetworkOps>::tn_einsum(
///     "ij,jk->ik",
///     vec![lhs_primitive, rhs_primitive],
/// );
/// ```
pub trait TensorNetworkOps: Backend {
    /// N-ary einsum contraction at the primitive tensor level.
    ///
    /// `subscripts` uses Einstein notation (e.g., `"ij,jk->ik"`).
    /// `inputs` are backend-specific float tensor primitives.
    fn tn_einsum(
        subscripts: &str,
        inputs: Vec<FloatTensor<Self>>,
    ) -> FloatTensor<Self>;
}

/// High-level einsum on Burn `Tensor` values.
///
/// Wraps [`TensorNetworkOps::tn_einsum`] with `Tensor<B, D>` ergonomics.
/// Internally converts to/from primitive tensors.
///
/// # Examples
///
/// ```ignore
/// use burn::tensor::Tensor;
/// use burn::backend::NdArray;
/// use tenferro_burn::einsum;
///
/// let a = Tensor::<NdArray<f64>, 2>::ones([2, 3], &Default::default());
/// let b = Tensor::<NdArray<f64>, 2>::ones([3, 4], &Default::default());
/// let c = einsum::<NdArray<f64>, 2>("ij,jk->ik", vec![a, b]);
/// ```
pub fn einsum<B: TensorNetworkOps, const D: usize>(
    subscripts: &str,
    inputs: Vec<burn::tensor::Tensor<B, D>>,
) -> burn::tensor::Tensor<B, D> {
    let primitives: Vec<FloatTensor<B>> = inputs
        .into_iter()
        .map(|t| t.into_primitive().tensor())
        .collect();
    let output = B::tn_einsum(subscripts, primitives);
    burn::tensor::Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(output))
}
```

**Step 2: Verify compilation**

Run: `cargo build -p tenferro-burn`
Expected: Compiles (forward.rs and backward.rs are empty modules, convert.rs is pub mod)

**Step 3: Commit**

```bash
git add extension/tenferro-burn/src/lib.rs
git commit -m "feat(tenferro-burn): define TensorNetworkOps trait and einsum API"
```

---

### Task 3: Implement conversion function signatures

**Files:**
- Modify: `extension/tenferro-burn/src/convert.rs`

**Step 1: Write conversion functions**

Replace `extension/tenferro-burn/src/convert.rs` with:

```rust
//! Burn tensor ↔ tenferro tensor conversion.
//!
//! These functions convert between Burn's `FloatTensor<B>` (backend-specific
//! primitive) and tenferro's `Tensor<f64>`. The current implementation is
//! copy-based; zero-copy optimization can be added later if profiling warrants.

use burn::tensor::backend::Backend;
use burn::tensor::ops::FloatTensor;
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

/// Convert a Burn float tensor primitive to a tenferro tensor.
///
/// Extracts data from the Burn tensor via `TensorData`, interprets it as
/// `f64` elements, and constructs a tenferro `Tensor<f64>` in row-major
/// order.
///
/// # Examples
///
/// ```ignore
/// use tenferro_burn::convert::burn_to_tenferro;
/// use burn::backend::NdArray;
///
/// let burn_tensor: FloatTensor<NdArray<f64>> = /* ... */;
/// let tf_tensor = burn_to_tenferro::<NdArray<f64>>(burn_tensor);
/// ```
pub fn burn_to_tenferro<B: Backend>(
    _tensor: FloatTensor<B>,
) -> Tensor<f64> {
    // B::float_into_data(tensor) → TensorData
    // data.as_slice::<f64>() → &[f64]
    // Tensor::from_slice(slice, shape, MemoryOrder::RowMajor)
    todo!()
}

/// Convert a tenferro tensor to a Burn float tensor primitive.
///
/// Copies data from tenferro's `Tensor<f64>` into a `TensorData` and
/// constructs a Burn backend tensor on the specified device.
///
/// # Examples
///
/// ```ignore
/// use tenferro_burn::convert::tenferro_to_burn;
/// use burn::backend::NdArray;
///
/// let tf_tensor = tenferro_tensor::Tensor::<f64>::zeros(/* ... */);
/// let burn_tensor = tenferro_to_burn::<NdArray<f64>>(tf_tensor, &device);
/// ```
pub fn tenferro_to_burn<B: Backend>(
    _tensor: Tensor<f64>,
    _device: &B::Device,
) -> FloatTensor<B> {
    // tensor.contiguous(RowMajor)
    // buffer.as_slice() → &[f64]
    // TensorData::new(slice.to_vec(), shape)
    // B::float_from_data(data, device)
    todo!()
}
```

**Step 2: Verify compilation**

Run: `cargo build -p tenferro-burn`
Expected: Compiles. `Scalar` import may need adjustment if not re-exported — use `tenferro_algebra::Scalar` or adjust to match the actual path.

**Step 3: Commit**

```bash
git add extension/tenferro-burn/src/convert.rs
git commit -m "feat(tenferro-burn): add burn_to_tenferro / tenferro_to_burn conversion stubs"
```

---

### Task 4: Implement forward for NdArray<f64>

**Files:**
- Modify: `extension/tenferro-burn/src/forward.rs`

**Step 1: Write forward impl**

Replace `extension/tenferro-burn/src/forward.rs` with:

```rust
//! Forward implementations of [`TensorNetworkOps`] for concrete backends.

use burn::tensor::ops::FloatTensor;
use burn::backend::NdArray;
use crate::TensorNetworkOps;

impl TensorNetworkOps for NdArray<f64> {
    fn tn_einsum(
        _subscripts: &str,
        _inputs: Vec<FloatTensor<Self>>,
    ) -> FloatTensor<Self> {
        // 1. Convert each input: burn_to_tenferro(input)
        // 2. Call tenferro_einsum::traced_tensor::einsum(subscripts, &tf_inputs)
        // 3. Convert result: tenferro_to_burn(result, device)
        todo!()
    }
}
```

**Step 2: Verify compilation**

Run: `cargo build -p tenferro-burn`
Expected: Compiles. The orphan rule is satisfied because `NdArray<f64>` is from
`burn-ndarray` (external) and `TensorNetworkOps` is defined in this crate.

**Step 3: Commit**

```bash
git add extension/tenferro-burn/src/forward.rs
git commit -m "feat(tenferro-burn): TensorNetworkOps forward impl for NdArray<f64>"
```

---

### Task 5: Implement backward for Autodiff<B, C>

**Files:**
- Modify: `extension/tenferro-burn/src/backward.rs`

**Step 1: Write backward impl**

Replace `extension/tenferro-burn/src/backward.rs` with:

```rust
//! Autodiff backward implementation of [`TensorNetworkOps`].
//!
//! Wraps the forward computation and registers a [`Backward`] step on Burn's
//! computation graph. The backward step calls tenferro's einsum rrule pullback
//! to compute input gradients.

use burn::backend::autodiff::Autodiff;
use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::autodiff::grads::Gradients;
use burn::backend::autodiff::checkpoint::base::Checkpointer;
use burn::backend::autodiff::ops::{Backward, Ops, OpsKind};
use burn::tensor::ops::FloatTensor;
use crate::TensorNetworkOps;

impl<B, C> TensorNetworkOps for Autodiff<B, C>
where
    B: TensorNetworkOps,
    C: CheckpointStrategy,
{
    fn tn_einsum(
        _subscripts: &str,
        _inputs: Vec<FloatTensor<Self>>,
    ) -> FloatTensor<Self> {
        // 1. Extract primitive tensors and node refs from AutodiffTensor inputs
        // 2. Define EinsumBackward zero-sized struct
        // 3. Implement Backward<B, N> for EinsumBackward:
        //    - In backward(): convert Burn grad → tenferro, call einsum rrule
        //      pullback, convert tenferro grads → Burn, register with grads
        // 4. Use OpsPrep pipeline:
        //    - Tracked: checkpoint inputs, compute forward, finish with state
        //    - UnTracked: compute forward, finish without state
        todo!()
    }
}
```

Note: The `Backward` trait requires a const generic `N` for the number of inputs.
For einsum with variable input count, this is a design challenge. In the POC
(`todo!()`), we defer this — the full implementation will likely use a fixed
maximum (e.g., `Backward<B, 2>` for binary einsum) or a dynamic approach.

**Step 2: Verify compilation**

Run: `cargo build -p tenferro-burn`
Expected: Compiles. The impl is for `Autodiff<B, C>` where `B: TensorNetworkOps`
and `C: CheckpointStrategy`. The `Backward` imports may need path adjustments
depending on Burn's exact re-export structure. If `burn::backend::autodiff` is
not the right path, try `burn_autodiff` directly.

**Step 3: Commit**

```bash
git add extension/tenferro-burn/src/backward.rs
git commit -m "feat(tenferro-burn): TensorNetworkOps backward impl for Autodiff<B, C>"
```

---

### Task 6: Verify full workspace build and format

**Step 1: Build entire workspace**

Run: `cargo build --workspace`
Expected: All crates compile, including tenferro-burn

**Step 2: Check formatting**

Run: `cargo fmt --all --check`
If fails, run: `cargo fmt --all`

**Step 3: Run workspace tests**

Run: `cargo test --workspace`
Expected: All existing tests pass; tenferro-burn has no tests (compilation-only POC)

**Step 4: Final commit (if fmt changes needed)**

```bash
git add -A
git commit -m "style: format tenferro-burn"
```

**Step 5: Push**

```bash
git push origin main
```
