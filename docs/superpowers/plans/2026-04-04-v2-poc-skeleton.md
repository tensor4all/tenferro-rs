# v2 POC Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create v2 skeleton types, traits, and function signatures with `todo!()` bodies to validate crate boundaries compile (tenferro-rs#620). No docs, no sample code — structure first, iterate to refine.

**Architecture:** 3-layer IR (Tenferro IR → StableHLO IR → Execution IR). New `tenferro-ops` crate owns `StdTensorOp` and `SemiringOp<T>`. Existing crates get `v2` modules. External deps: `computegraph` (Fragment/GraphOp), `chainrules-core` (PrimitiveOp), `tidu` (differentiate/transpose).

**Tech Stack:** Rust workspace, computegraph-rs, chainrules-rs, tidu-rs

---

## File Structure

### New crate: `tenferro-ops/`
- `tenferro-ops/Cargo.toml`
- `tenferro-ops/src/lib.rs` — re-exports
- `tenferro-ops/src/config.rs` — `DotGeneralConfig`, `CompareDir`, placeholder configs
- `tenferro-ops/src/semiring_op_kind.rs` — `SemiringOpKind` enum
- `tenferro-ops/src/semiring_ops.rs` — `SemiringOps` trait
- `tenferro-ops/src/input_key.rs` — `TensorInputKey` + `ADKey` impl
- `tenferro-ops/src/std_tensor_op.rs` — `StdTensorOp` enum + `GraphOp` + `PrimitiveOp` + `SemiringOps`
- `tenferro-ops/src/semiring_op.rs` — `SemiringOp<T>` + `GraphOp` + `SemiringOps`

### Modified: `tenferro-tensor/src/v2/`
- `tenferro-tensor/src/v2/mod.rs` — module root
- `tenferro-tensor/src/v2/types.rs` — `TypedTensor<T>`, `Tensor`, `Buffer`, `Placement`, `MemoryKind`, `DType`
- `tenferro-tensor/src/v2/tensor_data.rs` — `TensorData` trait (buffer access)
- `tenferro-tensor/src/v2/operand.rs` — `impl Operand for Tensor` + structural ops

### Modified: `tenferro-einsum/src/v2/`
- `tenferro-einsum/src/v2/mod.rs`
- `tenferro-einsum/src/v2/types.rs` — `Subscripts`, `ContractionPath`
- `tenferro-einsum/src/v2/builder.rs` — `build_einsum_fragment<Op: SemiringOps>()`

### Modified: `tenferro/src/v2/`
- `tenferro/src/v2/mod.rs`
- `tenferro/src/v2/stablehlo.rs` — `StableHloOp`, `StableHloProgram`, `StableHloInstruction`
- `tenferro/src/v2/exec.rs` — `ExecOp`, `ExecProgram`, `ExecInstruction`
- `tenferro/src/v2/compiler.rs` — `lower_to_stablehlo()`, `compile_to_exec()`
- `tenferro/src/v2/backend.rs` — `SemiringCore<Alg>`, `SemiringFastPath<Alg>`
- `tenferro/src/v2/engine.rs` — `Engine`
- `tenferro/src/v2/traced.rs` — `TracedTensor`, `eval()`, `eval_all()`

---

## Task 1: Branch + workspace dependency setup

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Create: `tenferro-ops/Cargo.toml`
- Create: `tenferro-ops/src/lib.rs`

- [ ] **Step 1: Create feature branch**

```bash
cd /home/shinaoka/tensor4all/tenferro-rs
git fetch origin && git checkout -b v2-poc-skeleton origin/main
```

- [ ] **Step 2: Add `computegraph` workspace dependency and `tenferro-ops` member**

Add to workspace `Cargo.toml`:

Under `[workspace.dependencies]`:
```toml
computegraph = { path = "../computegraph-rs" }
```

Under `members`:
```toml
"tenferro-ops",
```

- [ ] **Step 3: Create `tenferro-ops/Cargo.toml`**

```toml
[package]
name = "tenferro-ops"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
publish.workspace = true
description = "v2 operation types: StdTensorOp, SemiringOp<T>, and traits."

[dependencies]
tenferro-tensor = { path = "../tenferro-tensor" }
computegraph.workspace = true
chainrules-core.workspace = true
num-complex.workspace = true
```

- [ ] **Step 4: Create `tenferro-ops/src/lib.rs`** (empty placeholder)

```rust
pub mod config;
pub mod input_key;
pub mod semiring_op;
pub mod semiring_op_kind;
pub mod semiring_ops;
pub mod std_tensor_op;
```

- [ ] **Step 5: Add `computegraph` dependency to `tenferro-tensor/Cargo.toml`**

```toml
computegraph.workspace = true
```

- [ ] **Step 6: Add `tenferro-ops` + `computegraph` dependency to `tenferro-einsum/Cargo.toml`**

```toml
tenferro-ops = { path = "../tenferro-ops" }
computegraph.workspace = true
```

- [ ] **Step 7: Add v2 dependencies to `tenferro/Cargo.toml`** (under `[dependencies]`)

```toml
tenferro-ops = { path = "../tenferro-ops" }
tenferro-einsum = { path = "../tenferro-einsum" }
computegraph.workspace = true
chainrules-core.workspace = true
tidu.workspace = true
```

- [ ] **Step 8: Verify workspace compiles**

```bash
cargo check --workspace 2>&1 | tail -5
```

Expected: may have errors from empty modules; fix iteratively.

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "scaffold: add tenferro-ops crate and v2 workspace deps"
```

---

## Task 2: tenferro-tensor v2 types

**Files:**
- Modify: `tenferro-tensor/src/lib.rs` (add `pub mod v2;`)
- Create: `tenferro-tensor/src/v2/mod.rs`
- Create: `tenferro-tensor/src/v2/types.rs`
- Create: `tenferro-tensor/src/v2/tensor_data.rs`
- Create: `tenferro-tensor/src/v2/operand.rs`

- [ ] **Step 1: Add v2 module to `tenferro-tensor/src/lib.rs`**

Append: `pub mod v2;`

- [ ] **Step 2: Create `tenferro-tensor/src/v2/mod.rs`**

```rust
pub mod operand;
pub mod tensor_data;
pub mod types;

pub use types::*;
```

- [ ] **Step 3: Create `tenferro-tensor/src/v2/types.rs`**

```rust
use num_complex::Complex;

#[derive(Clone, Debug)]
pub enum MemoryKind {
    Device,
    PinnedHost,
    UnpinnedHost,
    Other(String),
}

#[derive(Clone, Debug)]
pub struct ComputeDevice {
    pub kind: String,
    pub ordinal: usize,
}

#[derive(Clone, Debug)]
pub struct Placement {
    pub memory_kind: MemoryKind,
    pub resident_device: Option<ComputeDevice>,
}

#[derive(Clone, Debug)]
pub enum Buffer<T> {
    Host(Vec<T>),
}

#[derive(Clone, Debug)]
pub struct TypedTensor<T> {
    pub buffer: Buffer<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub placement: Placement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    C32,
    C64,
}

#[derive(Clone, Debug)]
pub enum Tensor {
    F32(TypedTensor<f32>),
    F64(TypedTensor<f64>),
    C32(TypedTensor<Complex<f32>>),
    C64(TypedTensor<Complex<f64>>),
}
```

- [ ] **Step 4: Create `tenferro-tensor/src/v2/tensor_data.rs`**

```rust
pub trait TensorData {
    type Scalar;
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[isize];
    fn as_slice(&self) -> &[Self::Scalar];
    fn from_dense(shape: Vec<usize>, data: Vec<Self::Scalar>) -> Self;
}
```

- [ ] **Step 5: Create `tenferro-tensor/src/v2/operand.rs`**

```rust
use computegraph::Operand;
use super::types::Tensor;

impl Operand for Tensor {
    fn zero(shape: &[usize]) -> Self { todo!() }
    fn one(shape: &[usize]) -> Self { todo!() }
    fn reshape(&self, shape: &[usize]) -> Self { todo!() }
    fn broadcast_in_dim(&self, shape: &[usize], dims: &[usize]) -> Self { todo!() }
    fn add(&self, other: &Self) -> Self { todo!() }
    fn multiply(&self, other: &Self) -> Self { todo!() }
    fn reduce_sum(&self, axes: &[usize]) -> Self { todo!() }
    fn dot_general(
        &self,
        other: &Self,
        lhs_contracting: &[usize],
        rhs_contracting: &[usize],
        lhs_batch: &[usize],
        rhs_batch: &[usize],
    ) -> Self { todo!() }
    fn conj(&self) -> Self { todo!() }
}

pub fn generic_transpose<T: TensorData>(_t: &T, _perm: &[usize]) -> T { todo!() }
pub fn generic_reshape<T: TensorData>(_t: &T, _shape: &[usize]) -> T { todo!() }
pub fn generic_broadcast_in_dim<T: TensorData>(_t: &T, _shape: &[usize], _dims: &[usize]) -> T { todo!() }

use super::tensor_data::TensorData;
```

- [ ] **Step 6: `cargo check -p tenferro-tensor`**

- [ ] **Step 7: Commit**

```bash
git add tenferro-tensor/src/v2/ tenferro-tensor/src/lib.rs
git commit -m "scaffold: tenferro-tensor v2 types (Tensor, Operand, TensorData)"
```

---

## Task 3: tenferro-ops — all types and trait impls

**Files:**
- Create: all files under `tenferro-ops/src/`

- [ ] **Step 1: Create `tenferro-ops/src/config.rs`**

```rust
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct DotGeneralConfig {
    pub lhs_contracting_dims: Vec<usize>,
    pub rhs_contracting_dims: Vec<usize>,
    pub lhs_batch_dims: Vec<usize>,
    pub rhs_batch_dims: Vec<usize>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum CompareDir {
    Eq,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct GatherConfig { /* placeholder */ }

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ScatterConfig { /* placeholder */ }

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SliceConfig {
    pub starts: Vec<usize>,
    pub limits: Vec<usize>,
    pub strides: Vec<usize>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PadConfig { /* placeholder */ }
```

- [ ] **Step 2: Create `tenferro-ops/src/input_key.rs`**

```rust
use chainrules_core::{ADKey, DiffPassId};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct TensorInputKey {
    pub id: u64,
}

impl ADKey for TensorInputKey {
    fn tangent_of(&self, pass: DiffPassId) -> Self {
        todo!()
    }
}
```

- [ ] **Step 3: Create `tenferro-ops/src/semiring_op_kind.rs`**

```rust
use crate::config::DotGeneralConfig;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum SemiringOpKind {
    Add,
    Mul,
    DotGeneral(DotGeneralConfig),
    ReduceSum { axes: Vec<usize> },
    Transpose { perm: Vec<usize> },
    Reshape { shape: Vec<usize> },
    BroadcastInDim { shape: Vec<usize>, dims: Vec<usize> },
}
```

- [ ] **Step 4: Create `tenferro-ops/src/semiring_ops.rs`**

```rust
use computegraph::GraphOp;
use crate::config::DotGeneralConfig;

pub trait SemiringOps: GraphOp {
    fn add_op() -> Self;
    fn mul_op() -> Self;
    fn dot_general(config: DotGeneralConfig) -> Self;
    fn reduce_sum(axes: Vec<usize>) -> Self;
    fn transpose_op(perm: Vec<usize>) -> Self;
    fn reshape(shape: Vec<usize>) -> Self;
    fn broadcast_in_dim(shape: Vec<usize>, dims: Vec<usize>) -> Self;
}
```

- [ ] **Step 5: Create `tenferro-ops/src/std_tensor_op.rs`**

```rust
use computegraph::{FragmentBuilder, GlobalValKey, GraphOp, LocalValId, OpMode, ValRef};
use chainrules_core::PrimitiveOp;
use tenferro_tensor::v2::Tensor;
use crate::config::*;
use crate::input_key::TensorInputKey;
use crate::semiring_ops::SemiringOps;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum StdTensorOp {
    // --- AD-closed graph core (Tier 1) ---
    Add,
    Mul,
    Neg,
    Conj,
    DotGeneral(DotGeneralConfig),
    Transpose { perm: Vec<usize> },
    Reshape { shape: Vec<usize> },
    BroadcastInDim { shape: Vec<usize>, dims: Vec<usize> },
    ReduceSum { axes: Vec<usize> },

    // --- Standard arithmetic only (Tier 2) ---
    // Elementwise
    Div,
    Abs,
    Sign,
    Maximum,
    Minimum,
    Compare(CompareDir),
    Select,
    Clamp,

    // Analytic
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Rsqrt,
    Pow,
    Expm1,
    Log1p,

    // Indexing
    Gather(GatherConfig),
    Scatter(ScatterConfig),
    Slice(SliceConfig),
    DynamicSlice,
    Pad(PadConfig),
    Concatenate { axis: usize },
    Reverse { axes: Vec<usize> },

    // Additional reductions
    ReduceProd { axes: Vec<usize> },
    ReduceMax { axes: Vec<usize> },
    ReduceMin { axes: Vec<usize> },

    // Linalg
    Cholesky,
    Svd,
    Qr,
    Eigh,
    Solve,
}

impl GraphOp for StdTensorOp {
    type Operand = Tensor;
    type Context = ();
    type InputKey = TensorInputKey;

    fn n_inputs(&self) -> usize { todo!() }
    fn n_outputs(&self) -> usize { todo!() }
    fn eval(&self, _ctx: &mut (), _inputs: &[&Tensor]) -> Vec<Tensor> { todo!() }
}

impl PrimitiveOp for StdTensorOp {
    fn add() -> Self { StdTensorOp::Add }

    fn linearize(
        &self,
        _builder: &mut FragmentBuilder<Self>,
        _primal_in: &[GlobalValKey<Self>],
        _primal_out: &[GlobalValKey<Self>],
        _tangent_in: &[Option<LocalValId>],
    ) -> Vec<Option<LocalValId>> {
        todo!()
    }

    fn transpose_rule(
        &self,
        _builder: &mut FragmentBuilder<Self>,
        _cotangent_out: &[Option<LocalValId>],
        _inputs: &[ValRef<Self>],
        _mode: &OpMode,
    ) -> Vec<Option<LocalValId>> {
        todo!()
    }
}

impl SemiringOps for StdTensorOp {
    fn add_op() -> Self { StdTensorOp::Add }
    fn mul_op() -> Self { StdTensorOp::Mul }
    fn dot_general(config: DotGeneralConfig) -> Self { StdTensorOp::DotGeneral(config) }
    fn reduce_sum(axes: Vec<usize>) -> Self { StdTensorOp::ReduceSum { axes } }
    fn transpose_op(perm: Vec<usize>) -> Self { StdTensorOp::Transpose { perm } }
    fn reshape(shape: Vec<usize>) -> Self { StdTensorOp::Reshape { shape } }
    fn broadcast_in_dim(shape: Vec<usize>, dims: Vec<usize>) -> Self {
        StdTensorOp::BroadcastInDim { shape, dims }
    }
}
```

- [ ] **Step 6: Create `tenferro-ops/src/semiring_op.rs`**

```rust
use std::marker::PhantomData;
use computegraph::{GraphOp, Operand};
use crate::config::DotGeneralConfig;
use crate::semiring_op_kind::SemiringOpKind;
use crate::semiring_ops::SemiringOps;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SemiringInputKey(pub u64);

#[derive(Clone, Debug)]
pub struct SemiringOp<T: Operand> {
    pub kind: SemiringOpKind,
    _phantom: PhantomData<T>,
}

impl<T: Operand> SemiringOp<T> {
    pub fn new(kind: SemiringOpKind) -> Self {
        Self { kind, _phantom: PhantomData }
    }
}

impl<T: Operand> PartialEq for SemiringOp<T> {
    fn eq(&self, other: &Self) -> bool { self.kind == other.kind }
}

impl<T: Operand> Eq for SemiringOp<T> {}

impl<T: Operand> std::hash::Hash for SemiringOp<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.kind.hash(state) }
}

impl<T: Operand + std::fmt::Debug> GraphOp for SemiringOp<T> {
    type Operand = T;
    type Context = ();
    type InputKey = SemiringInputKey;

    fn n_inputs(&self) -> usize { todo!() }
    fn n_outputs(&self) -> usize { todo!() }
    fn eval(&self, _ctx: &mut (), _inputs: &[&T]) -> Vec<T> { todo!() }
}

impl<T: Operand + std::fmt::Debug> SemiringOps for SemiringOp<T> {
    fn add_op() -> Self { Self::new(SemiringOpKind::Add) }
    fn mul_op() -> Self { Self::new(SemiringOpKind::Mul) }
    fn dot_general(config: DotGeneralConfig) -> Self {
        Self::new(SemiringOpKind::DotGeneral(config))
    }
    fn reduce_sum(axes: Vec<usize>) -> Self {
        Self::new(SemiringOpKind::ReduceSum { axes })
    }
    fn transpose_op(perm: Vec<usize>) -> Self {
        Self::new(SemiringOpKind::Transpose { perm })
    }
    fn reshape(shape: Vec<usize>) -> Self {
        Self::new(SemiringOpKind::Reshape { shape })
    }
    fn broadcast_in_dim(shape: Vec<usize>, dims: Vec<usize>) -> Self {
        Self::new(SemiringOpKind::BroadcastInDim { shape, dims })
    }
}
```

- [ ] **Step 7: `cargo check -p tenferro-ops`**

- [ ] **Step 8: Commit**

```bash
git add tenferro-ops/
git commit -m "scaffold: tenferro-ops crate (StdTensorOp, SemiringOp<T>, SemiringOps)"
```

---

## Task 4: tenferro-einsum v2

**Files:**
- Modify: `tenferro-einsum/src/lib.rs` (add `pub mod v2;`)
- Create: `tenferro-einsum/src/v2/mod.rs`
- Create: `tenferro-einsum/src/v2/types.rs`
- Create: `tenferro-einsum/src/v2/builder.rs`

- [ ] **Step 1: Add v2 module to `tenferro-einsum/src/lib.rs`**

Append: `pub mod v2;`

- [ ] **Step 2: Create `tenferro-einsum/src/v2/mod.rs`**

```rust
pub mod builder;
pub mod types;

pub use types::*;
```

- [ ] **Step 3: Create `tenferro-einsum/src/v2/types.rs`**

```rust
#[derive(Clone, Debug)]
pub struct Subscripts {
    pub input_indices: Vec<Vec<u32>>,
    pub output_indices: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct ContractionStep {
    pub inputs: (usize, usize),
    pub result_subscript: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct ContractionPath {
    pub steps: Vec<ContractionStep>,
}
```

- [ ] **Step 4: Create `tenferro-einsum/src/v2/builder.rs`**

```rust
use computegraph::{FragmentBuilder, LocalValId, ValRef};
use tenferro_ops::semiring_ops::SemiringOps;
use super::types::{ContractionPath, Subscripts};

pub fn build_einsum_fragment<Op: SemiringOps>(
    _builder: &mut FragmentBuilder<Op>,
    _path: &ContractionPath,
    _inputs: &[ValRef<Op>],
) -> LocalValId {
    todo!()
}

pub fn optimize_contraction_path(
    _subscripts: &Subscripts,
    _shapes: &[&[usize]],
) -> ContractionPath {
    todo!()
}
```

- [ ] **Step 5: `cargo check -p tenferro-einsum`**

- [ ] **Step 6: Commit**

```bash
git add tenferro-einsum/src/v2/ tenferro-einsum/src/lib.rs
git commit -m "scaffold: tenferro-einsum v2 (Subscripts, ContractionPath, build_einsum_fragment)"
```

---

## Task 5: tenferro v2 — IR types + backend + engine

**Files:**
- Modify: `tenferro/src/lib.rs` (add `pub mod v2;`)
- Create: all files under `tenferro/src/v2/`

- [ ] **Step 1: Add v2 module to `tenferro/src/lib.rs`**

Append: `pub mod v2;`

- [ ] **Step 2: Create `tenferro/src/v2/mod.rs`**

```rust
pub mod backend;
pub mod compiler;
pub mod engine;
pub mod exec;
pub mod stablehlo;
pub mod traced;
```

- [ ] **Step 3: Create `tenferro/src/v2/stablehlo.rs`**

```rust
use tenferro_ops::config::*;

#[derive(Clone, Debug)]
pub enum StableHloOp {
    // Tier 1 (AD-closed core)
    Add,
    Multiply,
    Negate,
    DotGeneral(DotGeneralConfig),
    Transpose { perm: Vec<usize> },
    Reshape { shape: Vec<usize> },
    BroadcastInDim { shape: Vec<usize>, dims: Vec<usize> },
    ReduceSum { axes: Vec<usize> },
    // Tier 2 (standard only)
    Divide,
    Abs,
    Exponential,
    Log,
    Sine,
    Cosine,
    Tanh,
    Sqrt,
    Rsqrt,
    Power,
    Compare(CompareDir),
    Select,
    Clamp,
    Gather(GatherConfig),
    Scatter(ScatterConfig),
    Slice(SliceConfig),
    Pad(PadConfig),
    Concatenate { axis: usize },
    Reverse { axes: Vec<usize> },
    ReduceMax { axes: Vec<usize> },
    ReduceMin { axes: Vec<usize> },
    // Linalg
    Cholesky,
    CustomCall { target: String },
    // Multi-output
    GetTupleElement { index: usize },
    // Constant
    Constant,
}

#[derive(Clone, Debug)]
pub struct StableHloInstruction {
    pub op: StableHloOp,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct StableHloProgram {
    pub instructions: Vec<StableHloInstruction>,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub n_slots: usize,
}
```

- [ ] **Step 4: Create `tenferro/src/v2/exec.rs`**

```rust
use tenferro_ops::config::*;

#[derive(Clone, Debug)]
pub enum ExecOp {
    // Structural (common infrastructure)
    Permute { perm: Vec<usize> },
    Reshape { shape: Vec<usize> },
    BroadcastInDim { shape: Vec<usize>, dims: Vec<usize> },
    // Semiring
    BatchedGemm(DotGeneralConfig),
    ReduceSum { axes: Vec<usize> },
    Add,
    Mul,
    // Standard elementwise
    Neg,
    Div,
    Abs,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Rsqrt,
    Pow,
    Compare(CompareDir),
    Select,
    Clamp,
    // Indexing
    Gather(GatherConfig),
    Scatter(ScatterConfig),
    Slice(SliceConfig),
    Pad(PadConfig),
    Concatenate { axis: usize },
    Reverse { axes: Vec<usize> },
    // Reductions
    ReduceProd { axes: Vec<usize> },
    ReduceMax { axes: Vec<usize> },
    ReduceMin { axes: Vec<usize> },
    // Linalg
    Cholesky,
    CustomCall { target: String },
}

#[derive(Clone, Debug)]
pub struct ExecInstruction {
    pub op: ExecOp,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub last_use: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct ExecProgram {
    pub instructions: Vec<ExecInstruction>,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub n_slots: usize,
}
```

- [ ] **Step 5: Create `tenferro/src/v2/compiler.rs`**

```rust
use computegraph::CompiledProgram;
use tenferro_ops::std_tensor_op::StdTensorOp;
use super::stablehlo::StableHloProgram;
use super::exec::ExecProgram;

pub fn lower_to_stablehlo(_prog: &CompiledProgram<StdTensorOp>) -> StableHloProgram {
    todo!()
}

pub fn compile_to_exec(_stablehlo: &StableHloProgram) -> ExecProgram {
    todo!()
}
```

- [ ] **Step 6: Create `tenferro/src/v2/backend.rs`**

```rust
use super::exec::ExecProgram;

pub trait SemiringCore {
    type Operand;

    fn batched_gemm(&self, lhs: &Self::Operand, rhs: &Self::Operand) -> Self::Operand;
    fn reduce_sum(&self, input: &Self::Operand, axes: &[usize]) -> Self::Operand;
}

pub trait SemiringFastPath: SemiringCore {
    fn contract(&self, lhs: &Self::Operand, rhs: &Self::Operand) -> Option<Self::Operand> {
        None
    }
    fn elementwise_mul(&self, lhs: &Self::Operand, rhs: &Self::Operand) -> Option<Self::Operand> {
        None
    }
    fn elementwise_add(&self, lhs: &Self::Operand, rhs: &Self::Operand) -> Option<Self::Operand> {
        None
    }
}

pub fn eval_exec_ir<B: SemiringCore>(
    _backend: &B,
    _program: &ExecProgram,
    _inputs: Vec<B::Operand>,
) -> Vec<B::Operand> {
    todo!()
}
```

- [ ] **Step 7: Create `tenferro/src/v2/engine.rs`**

```rust
use tenferro_tensor::v2::Tensor;
use super::stablehlo::StableHloProgram;

pub struct Engine {
    _compile_cache: Vec<StableHloProgram>, // placeholder
}

impl Engine {
    pub fn new() -> Self {
        Self { _compile_cache: Vec::new() }
    }
}
```

- [ ] **Step 8: Create `tenferro/src/v2/traced.rs`**

```rust
use std::sync::Arc;
use computegraph::{Fragment, LocalValId};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::v2::{DType, Tensor};
use super::engine::Engine;

pub struct TracedTensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub fragment: Arc<Fragment<StdTensorOp>>,
    pub val: LocalValId,
    pub data: Option<Tensor>,
}

impl TracedTensor {
    pub fn from_tensor(_tensor: Tensor) -> Self { todo!() }

    pub fn eval(&mut self, _engine: &mut Engine) -> &Tensor { todo!() }

    pub fn grad(&self, _wrt: &TracedTensor) -> TracedTensor { todo!() }

    pub fn jvp(&self, _wrt: &TracedTensor, _tangent: &TracedTensor) -> TracedTensor { todo!() }
}

pub fn eval_all(_engine: &mut Engine, _outputs: &mut [&mut TracedTensor]) -> Vec<Tensor> {
    todo!()
}
```

- [ ] **Step 9: `cargo check -p tenferro`**

- [ ] **Step 10: Commit**

```bash
git add tenferro/src/v2/ tenferro/src/lib.rs tenferro/Cargo.toml
git commit -m "scaffold: tenferro v2 (StableHLO IR, ExecIR, backends, Engine, TracedTensor)"
```

---

## Task 6: Final verification

- [ ] **Step 1: Full workspace check**

```bash
cargo check --workspace
```

- [ ] **Step 2: Fix any compilation errors iteratively**

- [ ] **Step 3: Commit final fixes**

```bash
git add -A && git commit -m "fix: resolve compilation issues in v2 skeleton"
```

---

## Dependency Order

```
Task 1 (workspace setup)
  → Task 2 (tenferro-tensor v2)
    → Task 3 (tenferro-ops)
      → Task 4 (tenferro-einsum v2)
      → Task 5 (tenferro v2)
        → Task 6 (final verify)
```

Tasks 4 and 5 are independent of each other (both depend on Task 3).
