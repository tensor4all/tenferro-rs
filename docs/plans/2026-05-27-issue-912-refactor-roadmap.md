# Issue 912 Refactor Roadmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete issue #912 by finishing the clean endpoint for issues #903 through #911: core-op catalog, backend session cleanup, linalg ownership, runtime/GPU dispatch cleanup, API consistency, and file/test audit.

**Architecture:** Add `tenferro-core-ops` as the source of truth for core primitive op identity and metadata. Keep `tenferro-internal-ops` as graph integration and AD emission, standard extensions as separate operation-family crates, and GPU/runtime dispatch as descriptor-driven lookups keyed by catalog or extension family descriptors. Replace `TensorExec` with a lightweight associated backend session surface and move linalg out of tensor/gpu public backend traits.

**Tech Stack:** Rust 2021 workspace, `computegraph`, `chainrules-core`, `tidu`, `strided-*`, CubeCL/CUDA, `cargo test`, `cargo llvm-cov`, rustdoc, docs-site checks.

---

## Preconditions

Current branch: `issue-912-refactor-roadmap`.

The worktree already contains many uncommitted #912-related changes. Do not
revert them. Treat them as the current working baseline, and stage only files
explicitly changed by the active task.

Before implementation, re-read:

- `docs/design/refactor-roadmap-912.md`
- `AGENTS.md`
- `REPOSITORY_RULES.md`
- `docs/design/gpu-backend-design.md` before GPU code changes

## Task 1: Baseline Inventory

**Files:**
- Read: `docs/design/refactor-roadmap-912.md`
- Read: `Cargo.toml`
- Read: `tenferro-tensor/src/backend.rs`
- Read: `tenferro-runtime/src/exec.rs`
- Read: `tenferro-runtime/src/exec/dispatch.rs`
- Read: `tenferro-internal-ops/src/std_tensor_op.rs`
- Read: `tenferro-gpu/src/cubecl/mod.rs`

**Step 1: Capture current status**

Run:

```bash
git status --short
git diff --stat
```

Expected: many existing dirty files. Do not clean or revert them.

**Step 2: Establish compile baseline**

Run:

```bash
cargo check --workspace
```

Expected: PASS. If it fails, record the failure and fix only if the failure
blocks the first implementation task.

**Step 3: Check existing inline test state**

Run:

```bash
rg -n "#\\[cfg\\(test\\)\\]\\s*mod tests\\s*\\{" tenferro-* -g'*.rs'
```

Expected: no normal-module inline test blocks. Tiny leaf-module exceptions must
be listed in the final audit if they remain.

**Step 4: Commit nothing**

This task is inventory only.

## Task 2: Add `tenferro-core-ops`

**Files:**
- Modify: `Cargo.toml`
- Create: `tenferro-core-ops/Cargo.toml`
- Create: `tenferro-core-ops/src/lib.rs`
- Create: `tenferro-core-ops/src/catalog.rs`
- Create: `tenferro-core-ops/tests/catalog.rs`

**Step 1: Write catalog tests first**

Create `tenferro-core-ops/tests/catalog.rs`:

```rust
use tenferro_core_ops::{
    all_primitive_descriptors, descriptor, DTypePolicy, OpCategory, PrimitiveOpKind,
};

#[test]
fn catalog_contains_core_primitives_only() {
    let names: Vec<_> = all_primitive_descriptors()
        .iter()
        .map(|entry| entry.name)
        .collect();

    assert!(names.contains(&"add"));
    assert!(names.contains(&"dot_general"));
    assert!(names.contains(&"dynamic_update_slice"));
    assert!(!names.iter().any(|name| name.contains("svd")));
    assert!(!names.iter().any(|name| name.contains("fft")));
    assert!(!names.iter().any(|name| name.contains("einsum")));
}

#[test]
fn descriptor_lookup_is_total_for_declared_kinds() {
    for entry in all_primitive_descriptors() {
        assert_eq!(descriptor(entry.kind).kind, entry.kind);
        assert!(!entry.name.is_empty());
    }
}

#[test]
fn representative_dtype_policies_are_explicit() {
    assert_eq!(
        descriptor(PrimitiveOpKind::Add).dtype_policy,
        DTypePolicy::SameNumeric
    );
    assert_eq!(
        descriptor(PrimitiveOpKind::Compare).dtype_policy,
        DTypePolicy::CompareToBool
    );
    assert_eq!(
        descriptor(PrimitiveOpKind::ShapeOf).category,
        OpCategory::Host
    );
}
```

**Step 2: Run the failing test**

Run:

```bash
cargo test -p tenferro-core-ops
```

Expected: FAIL because the crate does not exist.

**Step 3: Add workspace member and crate**

In root `Cargo.toml`, add `"tenferro-core-ops"` to `[workspace].members`.

Create `tenferro-core-ops/Cargo.toml`:

```toml
[package]
name = "tenferro-core-ops"
version.workspace = true
edition.workspace = true
license.workspace = true
authors.workspace = true
publish.workspace = true
description = "Core primitive operation catalog for the tenferro workspace."

[lib]
name = "tenferro_core_ops"
path = "src/lib.rs"
```

Create `tenferro-core-ops/src/lib.rs`:

```rust
//! Core primitive operation catalog for tenferro.
//!
//! This crate intentionally excludes standard extension families such as
//! linalg, FFT, and einsum.

mod catalog;

pub use catalog::{
    all_primitive_descriptors, descriptor, DTypePolicy, OpCategory, PrimitiveOpDescriptor,
    PrimitiveOpKind,
};
```

Create `tenferro-core-ops/src/catalog.rs` with a single macro source of truth:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpCategory {
    Elementwise,
    Analytic,
    Structural,
    Reduction,
    Contraction,
    Indexing,
    Dynamic,
    Host,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DTypePolicy {
    SameAny,
    SameNumeric,
    SameFloat,
    SameFloatOrComplex,
    CompareToBool,
    BoolSelect,
    Convert,
    Shape,
    Constant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrimitiveOpDescriptor {
    pub kind: PrimitiveOpKind,
    pub name: &'static str,
    pub category: OpCategory,
    pub dtype_policy: DTypePolicy,
    pub min_inputs: u8,
    pub max_inputs: u8,
    pub host_only: bool,
}

macro_rules! primitive_ops {
    ($macro:ident) => {
        $macro! {
            Add, "add", Elementwise, SameNumeric, 2, 2, false;
            Mul, "mul", Elementwise, SameNumeric, 2, 2, false;
            Neg, "neg", Elementwise, SameNumeric, 1, 1, false;
            Conj, "conj", Elementwise, SameFloatOrComplex, 1, 1, false;
            Div, "div", Elementwise, SameFloatOrComplex, 2, 2, false;
            Abs, "abs", Elementwise, SameFloat, 1, 1, false;
            Sign, "sign", Elementwise, SameFloat, 1, 1, false;
            Maximum, "maximum", Elementwise, SameFloat, 2, 2, false;
            Minimum, "minimum", Elementwise, SameFloat, 2, 2, false;
            Compare, "compare", Elementwise, CompareToBool, 2, 2, false;
            Select, "select", Elementwise, BoolSelect, 3, 3, false;
            Clamp, "clamp", Elementwise, SameFloat, 3, 3, false;
            Exp, "exp", Analytic, SameFloatOrComplex, 1, 1, false;
            Log, "log", Analytic, SameFloatOrComplex, 1, 1, false;
            Sin, "sin", Analytic, SameFloatOrComplex, 1, 1, false;
            Cos, "cos", Analytic, SameFloatOrComplex, 1, 1, false;
            Tanh, "tanh", Analytic, SameFloatOrComplex, 1, 1, false;
            Sqrt, "sqrt", Analytic, SameFloatOrComplex, 1, 1, false;
            Rsqrt, "rsqrt", Analytic, SameFloatOrComplex, 1, 1, false;
            Pow, "pow", Analytic, SameFloatOrComplex, 2, 2, false;
            Expm1, "expm1", Analytic, SameFloatOrComplex, 1, 1, false;
            Log1p, "log1p", Analytic, SameFloatOrComplex, 1, 1, false;
            DotGeneral, "dot_general", Contraction, SameFloatOrComplex, 2, 2, false;
            ReduceSum, "reduce_sum", Reduction, SameNumeric, 1, 1, false;
            ReduceProd, "reduce_prod", Reduction, SameNumeric, 1, 1, false;
            ReduceMax, "reduce_max", Reduction, SameFloat, 1, 1, false;
            ReduceMin, "reduce_min", Reduction, SameFloat, 1, 1, false;
            Transpose, "transpose", Structural, SameAny, 1, 1, false;
            Reshape, "reshape", Structural, SameAny, 1, 1, false;
            BroadcastInDim, "broadcast_in_dim", Structural, SameAny, 1, 1, false;
            Convert, "convert", Structural, Convert, 1, 1, false;
            ExtractDiag, "extract_diag", Structural, SameAny, 1, 1, false;
            EmbedDiag, "embed_diag", Structural, SameAny, 1, 1, false;
            Tril, "tril", Structural, SameAny, 1, 1, false;
            Triu, "triu", Structural, SameAny, 1, 1, false;
            Gather, "gather", Indexing, SameAny, 2, 2, false;
            GatherDynamicSliceSizes, "gather_dynamic_slice_sizes", Indexing, SameAny, 2, 2, false;
            Scatter, "scatter", Indexing, SameAny, 3, 3, false;
            Slice, "slice", Indexing, SameAny, 1, 1, false;
            DynamicSlice, "dynamic_slice", Indexing, SameAny, 2, 2, false;
            DynamicUpdateSlice, "dynamic_update_slice", Indexing, SameAny, 3, 3, false;
            Pad, "pad", Indexing, SameAny, 1, 1, false;
            Concatenate, "concatenate", Indexing, SameAny, 1, u8::MAX, false;
            Reverse, "reverse", Indexing, SameAny, 1, 1, false;
            ShapeOf, "shape_of", Host, Shape, 1, 1, true;
            DynamicTruncate, "dynamic_truncate", Dynamic, SameAny, 2, 2, true;
            PadToMatch, "pad_to_match", Dynamic, SameAny, 2, 2, true;
            Constant, "constant", Host, Constant, 0, 0, true;
        }
    };
}

macro_rules! define_kind {
    ($( $variant:ident, $name:literal, $category:ident, $policy:ident, $min:expr, $max:expr, $host:expr; )*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum PrimitiveOpKind {
            $( $variant, )*
        }
    };
}

primitive_ops!(define_kind);

macro_rules! define_descriptors {
    ($( $variant:ident, $name:literal, $category:ident, $policy:ident, $min:expr, $max:expr, $host:expr; )*) => {
        const DESCRIPTORS: &[PrimitiveOpDescriptor] = &[
            $(
                PrimitiveOpDescriptor {
                    kind: PrimitiveOpKind::$variant,
                    name: $name,
                    category: OpCategory::$category,
                    dtype_policy: DTypePolicy::$policy,
                    min_inputs: $min,
                    max_inputs: $max,
                    host_only: $host,
                },
            )*
        ];

        pub fn descriptor(kind: PrimitiveOpKind) -> &'static PrimitiveOpDescriptor {
            match kind {
                $(
                    PrimitiveOpKind::$variant => &DESCRIPTORS[PrimitiveOpKind::$variant as usize],
                )*
            }
        }
    };
}

primitive_ops!(define_descriptors);

pub fn all_primitive_descriptors() -> &'static [PrimitiveOpDescriptor] {
    DESCRIPTORS
}
```

If the `as usize` indexing is rejected because enum discriminants are not
guaranteed, replace the body of `descriptor` with a macro-generated match arm
returning each descriptor by position.

**Step 4: Run tests**

Run:

```bash
cargo test -p tenferro-core-ops
cargo check --workspace
```

Expected: PASS.

**Step 5: Commit**

```bash
git add Cargo.toml tenferro-core-ops
git commit -m "feat: add core primitive op catalog"
```

## Task 3: Wire `StdTensorOp` To `PrimitiveOpKind`

**Files:**
- Modify: `tenferro-internal-ops/Cargo.toml`
- Modify: `tenferro-internal-ops/src/std_tensor_op.rs`
- Modify: `tenferro-internal-ops/src/tests/std_tensor_op_tests.rs`
- Maybe create: `tenferro-internal-ops/src/tests/std_tensor_op_tests/catalog.rs`

**Step 1: Add failing tests**

Add tests asserting representative mappings:

```rust
use tenferro_core_ops::PrimitiveOpKind;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::{CompareDir, DType};

#[test]
fn std_tensor_op_maps_core_ops_to_catalog_kinds() {
    assert_eq!(StdTensorOp::Add.primitive_kind(), Some(PrimitiveOpKind::Add));
    assert_eq!(StdTensorOp::Compare(CompareDir::Lt).primitive_kind(), Some(PrimitiveOpKind::Compare));
    assert_eq!(
        StdTensorOp::Convert { from: DType::F32, to: DType::F64 }.primitive_kind(),
        Some(PrimitiveOpKind::Convert)
    );
}

#[test]
fn extension_ops_do_not_claim_core_kind() {
    // Use an existing test extension payload helper from ext_op tests, or keep
    // this assertion in the extension test module where a payload already exists.
}
```

**Step 2: Run failing tests**

Run:

```bash
cargo test -p tenferro-internal-ops std_tensor_op
```

Expected: FAIL because `primitive_kind` does not exist.

**Step 3: Add dependency and method**

In `tenferro-internal-ops/Cargo.toml`, add:

```toml
tenferro-core-ops = { path = "../tenferro-core-ops" }
```

In `std_tensor_op.rs`, add:

```rust
use tenferro_core_ops::PrimitiveOpKind;
```

Add:

```rust
impl StdTensorOp {
    pub fn primitive_kind(&self) -> Option<PrimitiveOpKind> {
        Some(match self {
            StdTensorOp::Add => PrimitiveOpKind::Add,
            StdTensorOp::Mul => PrimitiveOpKind::Mul,
            StdTensorOp::Neg => PrimitiveOpKind::Neg,
            StdTensorOp::Conj => PrimitiveOpKind::Conj,
            StdTensorOp::DotGeneral { .. } => PrimitiveOpKind::DotGeneral,
            StdTensorOp::Transpose { .. } => PrimitiveOpKind::Transpose,
            StdTensorOp::Reshape { .. } => PrimitiveOpKind::Reshape,
            StdTensorOp::BroadcastInDim { .. } => PrimitiveOpKind::BroadcastInDim,
            StdTensorOp::Convert { .. } => PrimitiveOpKind::Convert,
            StdTensorOp::Constant { .. } => PrimitiveOpKind::Constant,
            StdTensorOp::ReduceSum { .. } => PrimitiveOpKind::ReduceSum,
            StdTensorOp::Div => PrimitiveOpKind::Div,
            StdTensorOp::Abs => PrimitiveOpKind::Abs,
            StdTensorOp::Sign => PrimitiveOpKind::Sign,
            StdTensorOp::Maximum => PrimitiveOpKind::Maximum,
            StdTensorOp::Minimum => PrimitiveOpKind::Minimum,
            StdTensorOp::Compare(_) => PrimitiveOpKind::Compare,
            StdTensorOp::Select => PrimitiveOpKind::Select,
            StdTensorOp::Clamp => PrimitiveOpKind::Clamp,
            StdTensorOp::Exp => PrimitiveOpKind::Exp,
            StdTensorOp::Log => PrimitiveOpKind::Log,
            StdTensorOp::Sin => PrimitiveOpKind::Sin,
            StdTensorOp::Cos => PrimitiveOpKind::Cos,
            StdTensorOp::Tanh => PrimitiveOpKind::Tanh,
            StdTensorOp::Sqrt => PrimitiveOpKind::Sqrt,
            StdTensorOp::Rsqrt => PrimitiveOpKind::Rsqrt,
            StdTensorOp::Pow => PrimitiveOpKind::Pow,
            StdTensorOp::Expm1 => PrimitiveOpKind::Expm1,
            StdTensorOp::Log1p => PrimitiveOpKind::Log1p,
            StdTensorOp::ExtractDiag { .. } => PrimitiveOpKind::ExtractDiag,
            StdTensorOp::EmbedDiag { .. } => PrimitiveOpKind::EmbedDiag,
            StdTensorOp::Tril { .. } => PrimitiveOpKind::Tril,
            StdTensorOp::Triu { .. } => PrimitiveOpKind::Triu,
            StdTensorOp::Gather(_) => PrimitiveOpKind::Gather,
            StdTensorOp::GatherDynamicSliceSizes { .. } => PrimitiveOpKind::GatherDynamicSliceSizes,
            StdTensorOp::Scatter(_) => PrimitiveOpKind::Scatter,
            StdTensorOp::Slice(_) => PrimitiveOpKind::Slice,
            StdTensorOp::DynamicSlice { .. } => PrimitiveOpKind::DynamicSlice,
            StdTensorOp::DynamicUpdateSlice => PrimitiveOpKind::DynamicUpdateSlice,
            StdTensorOp::Pad(_) => PrimitiveOpKind::Pad,
            StdTensorOp::Concatenate { .. } => PrimitiveOpKind::Concatenate,
            StdTensorOp::Reverse { .. } => PrimitiveOpKind::Reverse,
            StdTensorOp::ShapeOf { .. } => PrimitiveOpKind::ShapeOf,
            StdTensorOp::DynamicTruncate { .. } => PrimitiveOpKind::DynamicTruncate,
            StdTensorOp::PadToMatch { .. } => PrimitiveOpKind::PadToMatch,
            StdTensorOp::ReduceProd { .. } => PrimitiveOpKind::ReduceProd,
            StdTensorOp::ReduceMax { .. } => PrimitiveOpKind::ReduceMax,
            StdTensorOp::ReduceMin { .. } => PrimitiveOpKind::ReduceMin,
            StdTensorOp::Extension(_) => return None,
        })
    }
}
```

This is the first and only direct match for mapping carrier payload variants to
catalog identity. Later tasks should reuse it.

**Step 4: Run tests**

```bash
cargo test -p tenferro-internal-ops std_tensor_op
cargo check --workspace
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-internal-ops
git commit -m "feat: map std tensor ops to core catalog kinds"
```

## Task 4: Replace Primitive AD Match Lookup With A Registry

**Files:**
- Create: `tenferro-internal-ops/src/ad/registry.rs`
- Modify: `tenferro-internal-ops/src/ad/mod.rs`
- Modify: `tenferro-internal-ops/src/ad/tests/mod.rs`
- Modify or create targeted AD tests under: `tenferro-internal-ops/src/ad/tests/`

**Step 1: Add registry coverage test**

Create or extend an AD test that asserts representative non-extension kinds
have registry entries and extension ops are delegated separately:

```rust
use tenferro_core_ops::PrimitiveOpKind;
use tenferro_ops::ad::registry::primitive_ad_rule;

#[test]
fn primitive_ad_registry_has_representative_rules() {
    assert!(primitive_ad_rule(PrimitiveOpKind::Add).is_some());
    assert!(primitive_ad_rule(PrimitiveOpKind::DotGeneral).is_some());
    assert!(primitive_ad_rule(PrimitiveOpKind::DynamicUpdateSlice).is_some());
    assert!(primitive_ad_rule(PrimitiveOpKind::ShapeOf).is_some());
}
```

If `ad::registry` should not be public, expose a `#[cfg(test)]` helper in
`ad/mod.rs` instead of making implementation details public.

**Step 2: Run failing test**

```bash
cargo test -p tenferro-internal-ops primitive_ad_registry
```

Expected: FAIL because registry does not exist.

**Step 3: Introduce registry wrappers**

In `ad/mod.rs`, make category modules visible to `registry.rs` with
`pub(crate)` where needed.

Create `ad/registry.rs`:

```rust
use chainrules_core::ADRuleResult;
use computegraph::fragment::FragmentBuilder;
use computegraph::types::{GlobalValKey, LocalValId, OpMode, ValRef};
use computegraph::OpEmitter;
use tenferro_core_ops::PrimitiveOpKind;

use crate::ad::context::ShapeGuardContext;
use crate::std_tensor_op::StdTensorOp;

pub(crate) type LinearizeFn = fn(
    &StdTensorOp,
    &mut FragmentBuilder<StdTensorOp>,
    &[GlobalValKey<StdTensorOp>],
    &[GlobalValKey<StdTensorOp>],
    &[Option<LocalValId>],
    &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>>;

pub(crate) type TransposeFn = fn(
    &StdTensorOp,
    &mut dyn OpEmitter<StdTensorOp>,
    &[Option<LocalValId>],
    &[ValRef<StdTensorOp>],
    &OpMode,
    &mut ShapeGuardContext,
) -> ADRuleResult<Vec<Option<LocalValId>>>;

pub(crate) struct PrimitiveAdRule {
    pub(crate) kind: PrimitiveOpKind,
    pub(crate) linearize: LinearizeFn,
    pub(crate) transpose: TransposeFn,
}

pub(crate) fn primitive_ad_rule(kind: PrimitiveOpKind) -> Option<&'static PrimitiveAdRule> {
    PRIMITIVE_AD_RULES.iter().find(|rule| rule.kind == kind)
}

static PRIMITIVE_AD_RULES: &[PrimitiveAdRule] = &[
    // Fill by grouping current match arms into small wrapper functions.
];
```

Move the current `try_linearize` and `try_transpose_rule` match arm bodies into
small private wrapper functions. Keep `StdTensorOp::Extension(_)` handling in
`ad/mod.rs`, outside the primitive registry.

**Step 4: Replace lookup**

In `try_linearize`:

```rust
let Some(kind) = op.primitive_kind() else {
    if let StdTensorOp::Extension(ext) = op {
        return linearize_extension_rule(...);
    }
    unreachable!("non-extension op without primitive kind");
};
let Some(rule) = registry::primitive_ad_rule(kind) else {
    return Err(format!("missing primitive AD rule for {kind:?}").into());
};
(rule.linearize)(op, builder, primal_in, primal_out, tangent_in, ctx)
```

Apply the same shape to transpose.

**Step 5: Run AD tests**

```bash
cargo test -p tenferro-internal-ops --features autodiff ad
cargo test -p tenferro-ad ad
cargo check --workspace
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro-internal-ops tenferro-ad
git commit -m "refactor: dispatch primitive ad rules through registry"
```

## Task 5: Make Runtime Dispatch Use Catalog Keys

**Files:**
- Modify: `tenferro-runtime/Cargo.toml`
- Modify: `tenferro-runtime/src/exec.rs`
- Modify: `tenferro-runtime/src/exec/dispatch.rs`
- Modify: `tenferro-runtime/src/exec/tests.rs`

**Step 1: Add dispatch table tests**

In `tenferro-runtime/src/exec/tests.rs`, add tests for representative mapping:

```rust
use tenferro_core_ops::PrimitiveOpKind;
use crate::exec::ExecOp;

#[test]
fn exec_op_maps_to_catalog_kind_for_core_backend_ops() {
    assert_eq!(ExecOp::Add.primitive_kind(), Some(PrimitiveOpKind::Add));
    assert_eq!(ExecOp::Negate.primitive_kind(), Some(PrimitiveOpKind::Neg));
    assert_eq!(ExecOp::ShapeOf { axis: 0 }.primitive_kind(), Some(PrimitiveOpKind::ShapeOf));
}
```

**Step 2: Run failing test**

```bash
cargo test -p tenferro-runtime exec_op_maps_to_catalog_kind
```

Expected: FAIL.

**Step 3: Add dependency and mapping**

Add `tenferro-core-ops` dependency to `tenferro-runtime/Cargo.toml`.

Add `ExecOp::primitive_kind(&self) -> Option<PrimitiveOpKind>` in
`exec.rs`. This match should mirror the `StdTensorOp` carrier mapping and
return `None` for `ExecOp::Extension(_)`.

**Step 4: Change backend dispatch key**

In `exec/dispatch.rs`, replace `BackendDispatchKey` with
`PrimitiveOpKind` for core backend ops. Keep separate `FfiDispatchKey` and
`HostDispatchKey` until those can be cleanly folded by descriptor category.

The table entries should look like:

```rust
BackendDispatchEntry {
    key: PrimitiveOpKind::Add,
    execute: execute_add,
}
```

`backend_dispatch_entry` should call `op.primitive_kind()` and look up the
entry by kind.

**Step 5: Run runtime tests**

```bash
cargo test -p tenferro-runtime --lib
cargo test -p tenferro-runtime --tests
cargo check --workspace
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro-runtime
git commit -m "refactor: key runtime dispatch by primitive op catalog"
```

## Task 6: Rename `TensorExec` To `BackendSession`

**Files:**
- Modify: `tenferro-tensor/src/backend.rs`
- Modify: `tenferro-tensor/src/cpu/exec_session.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`
- Modify: `tenferro-runtime/src/exec.rs`
- Modify: `tenferro-runtime/src/exec/dispatch.rs`
- Modify: `tenferro-runtime/src/segment.rs`
- Modify: `tenferro-runtime/src/tensor.rs`
- Modify: `tenferro-einsum/src/eager.rs`
- Modify: `tenferro-ad/src/eager_backend.rs`
- Modify: `tenferro-ad/src/eager_exec.rs`
- Modify tests/docs that mention `TensorExec`

**Step 1: Add compile-fail target by search**

Run:

```bash
rg -n "TensorExec|with_exec_session" tenferro-* docs README.md
```

Expected: many hits. Save this list as the rename checklist in your notes, not
in the repository.

**Step 2: Rename public trait and methods**

In `tenferro-tensor/src/backend.rs`:

- Rename `TensorExec` to `BackendSession`.
- Rename `BackendExecAdapter` to `BackendSessionAdapter`.
- Rename `default_exec_session` to `default_backend_session`.
- Rename `with_exec_session` to `with_backend_session`.
- Rename `with_exec_session_cached` to `with_backend_session_cached`.

Do not leave type aliases or deprecated shims.

**Step 3: Update all call sites**

Replace call sites mechanically, then fix imports.

Examples:

```rust
backend.with_backend_session(|session| session.add(&lhs, &rhs))
```

```rust
type BackendDispatchFn =
    fn(&mut dyn BackendSession, &[Option<Tensor>], &ExecInstruction) -> Result<Tensor>;
```

**Step 4: Run targeted checks**

```bash
cargo check -p tenferro-tensor
cargo check -p tenferro-runtime
cargo check --workspace
```

Expected: PASS.

**Step 5: Remove stale references**

Run:

```bash
rg -n "TensorExec|with_exec_session" tenferro-* README.md docs
```

Expected: only historical docs under `docs/plans/` may remain. Update
`docs/design/`, `docs/spec/`, README, and public rustdoc references.

**Step 6: Commit**

```bash
git add tenferro-* README.md docs
git commit -m "refactor: rename tensor exec surface to backend sessions"
```

## Task 7: Remove Linalg From Tensor Backend Traits

**Files:**
- Modify: `tenferro-tensor/src/backend.rs`
- Modify: `tenferro-tensor/src/cpu/exec_session.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`
- Modify: `tenferro-tensor/src/types.rs`
- Modify: `tenferro-tensor/src/typed_linalg.rs`
- Modify: `tenferro-tensor/src/lib.rs`
- Modify: `tenferro-linalg/src/extension.rs`
- Modify: `tenferro-linalg/src/lib.rs`
- Create: `tenferro-linalg/src/backend.rs`

**Step 1: Add negative search gate**

Run before editing:

```bash
rg -n "fn (cholesky|triangular_solve|lu|full_piv_lu|full_piv_lu_solve|svd|qr|eigh|eig|solve)\\(" tenferro-tensor/src/backend.rs tenferro-tensor/src/cpu/exec_session.rs
```

Expected: linalg methods exist.

**Step 2: Create linalg backend trait**

Create `tenferro-linalg/src/backend.rs`:

```rust
use tenferro_tensor::{Tensor, TensorBackend};

pub trait LinalgBackend: TensorBackend {
    fn cholesky(&mut self, input: &Tensor) -> tenferro_tensor::Result<Tensor>;
    fn triangular_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> tenferro_tensor::Result<Tensor>;
    fn lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
    fn full_piv_lu(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> tenferro_tensor::Result<Tensor>;
    fn svd(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
    fn qr(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
    fn eigh(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
    fn eig(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>>;
    fn solve(&mut self, a: &Tensor, b: &Tensor) -> tenferro_tensor::Result<Tensor>;
}
```

Keep this trait in `tenferro-linalg`, not `tenferro-tensor`.

**Step 3: Move implementation call target**

Change `tenferro-linalg/src/extension.rs`:

```rust
use crate::backend::LinalgBackend;

impl<B: LinalgBackend + 'static> ExtensionRuntime<B> for LinalgRuntime { ... }

pub fn register_runtime<B: LinalgBackend + 'static>(
    executor: &mut ExtensionExecutor<B>,
) -> Result<(), ExtensionRuntimeRegistryError> { ... }

fn execute_linalg<B: LinalgBackend>(...) -> tenferro_tensor::Result<Vec<Tensor>> { ... }
```

**Step 4: Delete linalg methods from tensor backend/session traits**

Remove linalg methods from `BackendSession` and `TensorBackend` in
`tenferro-tensor/src/backend.rs`.

Remove linalg delegations and macros from `tenferro-tensor/src/cpu/exec_session.rs`.

Temporarily keep CPU linalg implementation methods on `CpuBackend` as inherent
methods if needed for the next task, but they must not be in `TensorBackend`.

**Step 5: Remove direct tensor linalg methods**

Delete from `tenferro-tensor/src/types.rs`:

- `Tensor::svd`
- `Tensor::qr`
- `Tensor::lu`
- `Tensor::full_piv_lu`
- `Tensor::cholesky`
- `Tensor::eigh`
- `Tensor::eig`
- `Tensor::triangular_solve`
- `Tensor::solve`

Delete `tenferro-tensor/src/typed_linalg.rs` and its export from
`tenferro-tensor/src/lib.rs` if it only provides linalg convenience APIs.

**Step 6: Update tests**

Move linalg tests out of `tenferro-tensor` into `tenferro-linalg` where they
test linalg behavior. Tensor tests should keep only core tensor/GEMM behavior.

**Step 7: Run checks**

```bash
cargo check -p tenferro-tensor
cargo check -p tenferro-linalg
cargo check --workspace
```

Expected: PASS.

**Step 8: Search gate**

```bash
rg -n "Tensor::(svd|qr|cholesky|eigh|eig|triangular_solve|solve)|\\.svd\\(&mut|fn svd\\(" tenferro-tensor README.md docs
```

Expected: no current docs/source references outside historical `docs/plans/`.

**Step 9: Commit**

```bash
git add tenferro-tensor tenferro-linalg README.md docs
git commit -m "refactor: move linalg out of tensor backend surface"
```

## Task 8: Move CPU Linalg Implementation To `tenferro-linalg`

**Files:**
- Move: `tenferro-tensor/src/cpu/linalg/` -> `tenferro-linalg/src/cpu/`
- Modify: `tenferro-tensor/src/cpu/mod.rs`
- Modify: `tenferro-tensor/src/cpu/backend.rs`
- Modify: `tenferro-linalg/src/lib.rs`
- Modify: `tenferro-linalg/Cargo.toml`
- Modify: `tenferro-linalg/src/backend.rs`

**Step 1: Move files without changing behavior**

Use `git mv` where possible:

```bash
mkdir -p tenferro-linalg/src/cpu
git mv tenferro-tensor/src/cpu/linalg tenferro-linalg/src/cpu/linalg
```

If directories already changed in the dirty worktree, move with care and stage
only the intended paths.

**Step 2: Update imports and dependencies**

In `tenferro-linalg/Cargo.toml`, add workspace deps required by moved code:

```toml
faer.workspace = true
lapack.workspace = true
num-traits.workspace = true
```

Only add deps actually used after the move. Keep `cpu-faer` and `cpu-blas`
feature gates in `tenferro-linalg`.

**Step 3: Implement `LinalgBackend` for `CpuBackend`**

In a linalg-owned module, implement:

```rust
impl crate::backend::LinalgBackend for tenferro_tensor::cpu::CpuBackend {
    fn svd(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        // Use moved CPU linalg kernels and the backend's public/core session
        // helpers for reshape/transpose as needed.
    }
}
```

Use existing CPU buffer-pool and context APIs. If private access is required,
prefer adding a narrow crate-public helper in `tenferro-tensor` over making
large internals public.

**Step 4: Remove tensor linalg module**

Remove `pub(crate) mod linalg;` from `tenferro-tensor/src/cpu/mod.rs`.

Run:

```bash
rg -n "cpu::linalg|super::.*linalg|mod linalg" tenferro-tensor/src
```

Expected: no remaining tensor-owned linalg implementation references.

**Step 5: Run tests**

```bash
cargo test -p tenferro-linalg
cargo test -p tenferro-tensor --lib
cargo check --workspace
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro-tensor tenferro-linalg
git commit -m "refactor: move cpu linalg kernels to linalg crate"
```

## Task 9: Move GPU Linalg Implementation To `tenferro-linalg`

**Files:**
- Read: `docs/design/gpu-backend-design.md`
- Move: `tenferro-gpu/src/cubecl/linalg.rs` -> `tenferro-linalg/src/gpu/`
- Modify: `tenferro-gpu/src/cubecl/mod.rs`
- Modify: `tenferro-gpu/src/lib.rs`
- Modify: `tenferro-linalg/Cargo.toml`
- Modify: `tenferro-linalg/src/lib.rs`
- Modify: `tenferro-linalg/src/backend.rs`
- Move/update tests: `tenferro-gpu/src/cubecl/tests/linalg_tests.rs` -> `tenferro-linalg/tests/` or `tenferro-linalg/src/gpu/tests.rs`

**Step 1: Read GPU rules**

Run:

```bash
sed -n '1,260p' docs/design/gpu-backend-design.md
```

**Step 2: Update feature dependency**

In `tenferro-linalg/Cargo.toml`:

```toml
cuda = ["tenferro-ad?/cuda", "tenferro-internal-ops/cuda", "dep:tenferro-gpu", "tenferro-gpu/cuda"]
tenferro-gpu = { path = "../tenferro-gpu", default-features = false, optional = true }
```

Remove linalg-specific feature dependency from `tenferro-gpu` if it exists.

**Step 3: Move GPU linalg kernels**

Move GPU linalg code into `tenferro-linalg/src/gpu/`. Keep the CubeCL backend,
runtime, memory, upload/download, and FFI library loading in `tenferro-gpu`.

**Step 4: Implement `LinalgBackend` for CubeCL backend**

Under `#[cfg(feature = "cuda")]`:

```rust
impl crate::backend::LinalgBackend for tenferro_gpu::cubecl::CubeclBackend {
    fn svd(&mut self, input: &Tensor) -> tenferro_tensor::Result<Vec<Tensor>> {
        crate::gpu::svd(self, input)
    }
}
```

Expose only narrow GPU backend accessors required by the linalg crate.

**Step 5: Run CPU and CUDA checks**

```bash
cargo check -p tenferro-linalg --features cuda
cargo test -p tenferro-linalg
CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.0 LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-linalg --features cuda -- --ignored
```

Expected: PASS on the available CUDA machine.

**Step 6: Search gate**

```bash
rg -n "fn (svd|qr|cholesky|eigh|eig|triangular_solve|solve)\\(" tenferro-gpu/src
```

Expected: no linalg operation implementations in `tenferro-gpu`.

**Step 7: Commit**

```bash
git add tenferro-gpu tenferro-linalg
git commit -m "refactor: move gpu linalg kernels to linalg crate"
```

## Task 10: Extension Runtime Macro

**Files:**
- Modify: `tenferro-internal-extension-macros/src/lib.rs`
- Modify: `tenferro-internal-extension-macros/src/tests.rs`
- Modify: `tenferro-linalg/src/extension.rs`
- Modify: `tenferro-fft/src/lib.rs`
- Modify: `tenferro-einsum/src/extension.rs`

**Step 1: Add macro tests**

In `tenferro-internal-extension-macros/src/tests.rs`, add a test that expands
`define_extension_runtime!` for a tiny fake op/runtime and verifies
registration is idempotent or returns the existing duplicate behavior.

**Step 2: Run failing test**

```bash
cargo test -p tenferro-internal-extension-macros
```

Expected: FAIL because macro does not generate the new runtime shape.

**Step 3: Implement function-like macro**

Implement:

```rust
define_extension_runtime! {
    runtime = LinalgRuntime,
    family_id = LINALG_EXTENSION_FAMILY_ID,
    op_type = LinalgExtensionOp,
    execute = execute_linalg_extension,
    register_fn = register_runtime,
}
```

Prefer a narrow syntax matching current extension runtime traits. The macro
must generate the `ExtensionRuntime<B>` impl and `register_runtime` function.

**Step 4: Migrate standard extensions**

Apply the macro to linalg first, then FFT/einsum. Keep op-specific metadata and
execution functions in the owning extension crate.

**Step 5: Run tests**

```bash
cargo test -p tenferro-internal-extension-macros
cargo test -p tenferro-linalg
cargo test -p tenferro-fft
cargo test -p tenferro-einsum
cargo check --workspace
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro-internal-extension-macros tenferro-linalg tenferro-fft tenferro-einsum
git commit -m "refactor: generate standard extension runtimes with macro"
```

## Task 11: GPU Primitive Dispatch Descriptors

**Files:**
- Read: `docs/design/gpu-backend-design.md`
- Modify: `tenferro-gpu/Cargo.toml`
- Modify: `tenferro-gpu/src/cubecl/dispatch.rs`
- Modify: `tenferro-gpu/src/cubecl/mod.rs`
- Maybe create: `tenferro-gpu/src/cubecl/op_descriptor.rs`
- Modify: `tenferro-gpu/src/cubecl/tests/elementwise_tests.rs`
- Modify: `tenferro-gpu/src/cubecl/tests/reduction_tests.rs`
- Modify: `tenferro-gpu/src/cubecl/tests/structural_tests.rs`

**Step 1: Add descriptor tests**

Create tests that assert representative primitive kinds map to GPU descriptors:

```rust
use tenferro_core_ops::{DTypePolicy, PrimitiveOpKind};
use crate::cubecl::op_descriptor::gpu_descriptor;

#[test]
fn gpu_descriptors_have_catalog_dtype_policy() {
    let add = gpu_descriptor(PrimitiveOpKind::Add).unwrap();
    assert_eq!(add.dtype_policy, DTypePolicy::SameNumeric);
    let compare = gpu_descriptor(PrimitiveOpKind::Compare).unwrap();
    assert_eq!(compare.dtype_policy, DTypePolicy::CompareToBool);
}
```

Keep the descriptor module `pub(crate)` and put tests inside the crate.

**Step 2: Run failing test**

```bash
cargo test -p tenferro-gpu --features cuda gpu_descriptors_have_catalog_dtype_policy
```

Expected: FAIL.

**Step 3: Add dependency**

Add `tenferro-core-ops` to `tenferro-gpu/Cargo.toml`.

**Step 4: Implement `GpuOpDescriptor`**

Add:

```rust
pub(crate) struct GpuOpDescriptor {
    pub(crate) kind: PrimitiveOpKind,
    pub(crate) name: &'static str,
    pub(crate) dtype_policy: DTypePolicy,
    pub(crate) launch: GpuLaunchKind,
}
```

Use this descriptor as the entry point for elementwise/analytic/reduction dtype
dispatch. Keep typed kernel launches generic; the dynamic part is selecting the
descriptor for one op.

**Step 5: Remove op-local dtype matches incrementally**

Start with elementwise unary/binary ops already covered by macros. Then handle
analytic and reductions. Leave a short local comment only where FFI or complex
layout forces a special typed path.

Search gate:

```bash
rg -n "match input|match \\(lhs, rhs\\)|Tensor::F32|Tensor::F64|Tensor::C32|Tensor::C64" tenferro-gpu/src/cubecl/mod.rs tenferro-gpu/src/cubecl/dispatch.rs
```

Expected: remaining matches are either descriptor boundaries, FFI boundaries,
or documented special cases.

**Step 6: Run GPU checks**

```bash
cargo test -p tenferro-gpu --features cuda
CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.0 LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-gpu --features cuda -- --ignored
cargo check --workspace
```

Expected: PASS.

**Step 7: Commit**

```bash
git add tenferro-gpu
git commit -m "refactor: drive gpu primitive dispatch from descriptors"
```

## Task 12: API Naming And Error Consistency

**Files:**
- Modify: `tenferro-ad/src/context.rs`
- Modify: `tenferro-ad/src/traced.rs`
- Modify: `tenferro-ad/tests/`
- Modify: `tenferro-tensor/src/error.rs`
- Modify: all production files constructing `Error::BackendFailure`
- Modify: README/docs/rustdoc references

**Step 1: Add search gates**

Run:

```bash
rg -n "try_grad|Error::BackendFailure\\s*\\{" tenferro-* README.md docs -g'*.rs' -g'*.md'
```

Expected: current references remain.

**Step 2: Rename `try_grad`**

Rename public API:

- `try_grad` -> `grad_optional`
- `try_grad_with_rules` -> `grad_optional_with_rules`

Do not leave aliases.

Update tests:

```rust
let maybe = loss.grad_optional(&b).expect("grad_optional b");
```

**Step 3: Replace production backend error construction**

Production code should use:

```rust
Error::backend_failure("op_name", message)
```

Tests may pattern-match on `Error::BackendFailure`. Test expected construction
should use the helper.

**Step 4: Run checks**

```bash
cargo test -p tenferro-ad
cargo test -p tenferro-tensor
cargo check --workspace
```

Expected: PASS.

**Step 5: Search gate**

```bash
rg -n "try_grad|Error::BackendFailure\\s*\\{" tenferro-* README.md docs -g'*.rs' -g'*.md'
```

Expected:

- no `try_grad` outside historical `docs/plans/`
- no production direct `Error::BackendFailure { ... }`
- test pattern matches may remain

**Step 6: Commit**

```bash
git add tenferro-ad tenferro-tensor tenferro-* README.md docs
git commit -m "refactor: align grad optional and backend error APIs"
```

## Task 13: Traced Extension APIs Return `Result`

**Files:**
- Modify: `tenferro-linalg/src/traced.rs`
- Modify: `tenferro-linalg/src/traced_tensor.rs`
- Modify: `tenferro-fft/src/lib.rs`
- Modify: `tenferro-einsum/src/extension.rs`
- Modify: traced extension tests
- Modify: README/docs examples

**Step 1: Add failing tests**

For linalg and FFT traced APIs, add tests that use `?`:

```rust
#[test]
fn traced_linalg_svd_is_result_returning() -> Result<(), Box<dyn std::error::Error>> {
    let x = TracedTensor::from_vec_row_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]);
    let (_u, s, _vt) = tenferro_linalg::traced_tensor::svd(&x)?;
    assert_eq!(s.shape(), &[2]);
    Ok(())
}
```

**Step 2: Run failing tests**

```bash
cargo test -p tenferro-linalg traced_linalg_svd_is_result_returning
```

Expected: FAIL if API still returns bare tensors.

**Step 3: Convert APIs**

Change public traced extension helpers to return crate-local `Result`.
Validation failures should return typed errors instead of panicking or
asserting. Keep graph construction pure after validation.

**Step 4: Update call sites and docs**

All traced extension examples should use `?`.

**Step 5: Run checks**

```bash
cargo test -p tenferro-linalg
cargo test -p tenferro-fft
cargo test -p tenferro-einsum
cargo test --doc --workspace
cargo check --workspace
```

Expected: PASS.

**Step 6: Commit**

```bash
git add tenferro-linalg tenferro-fft tenferro-einsum README.md docs
git commit -m "refactor: make traced extension APIs fallible"
```

## Task 14: Documentation Surface Cleanup

**Files:**
- Modify: `README.md`
- Modify: `docs/guides/linear-algebra.md`
- Modify: `docs/guides/eager-operations.md`
- Modify: `docs/getting-started/pytorch-jax-mapping.md`
- Modify: `docs/design/supported-ops.md`
- Modify: `docs/spec/backend-contract.md`
- Modify rustdoc in touched public APIs

**Step 1: Search stale public surface**

```bash
rg -n "TensorExec|Tensor::svd|\\.svd\\(&mut|try_grad|tenferro_core_ops|tenferro-core-ops|standard operation" README.md docs tenferro-* -g'*.md' -g'*.rs'
```

**Step 2: Update current docs**

Rules:

- Direct Tensor Execution docs show only core tensor ops.
- Linalg examples use `tenferro_linalg::eager_tensor::*` or `tenferro_linalg::traced_tensor::*`.
- Traced linalg examples use `?`.
- Internal crates are not presented as user-facing API.
- `tenferro-core-ops` is mentioned only in internals/design docs.

**Step 3: Extract changed README/getting-started examples**

Manually compile/run changed examples, or add temporary snippets to a scratch
test and remove the scratch file before commit.

**Step 4: Run docs checks**

```bash
cargo test --doc --workspace
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add README.md docs tenferro-*
git commit -m "docs: align public surface with issue 912 refactor"
```

## Task 15: File/Test Audit

**Files:**
- Modify: `docs/design/refactor-roadmap-912.md` if audit notes need a durable summary
- Modify: large files only when there is a real ownership boundary
- Modify: extracted tests only when inline tests remain

**Step 1: Find large files**

```bash
find tenferro-* -path '*/target' -prune -o -name '*.rs' -print \
  | xargs wc -l \
  | sort -nr \
  | sed -n '1,40p'
```

**Step 2: Find inline test blocks**

```bash
rg -n "#\\[cfg\\(test\\)\\]\\s*mod tests\\s*\\{" tenferro-* -g'*.rs'
```

**Step 3: Split only real boundaries**

For each large file, choose one:

- Split by existing boundary such as dispatch, validation, cache, backend glue,
  extension family, or AD rule category.
- Keep as-is and document why it is one coherent concern.

Do not create `part1`/`part2` files.

**Step 4: Run tests affected by moved files**

```bash
cargo test -p <affected-crate>
cargo check --workspace
```

**Step 5: Commit**

```bash
git add tenferro-* docs/design/refactor-roadmap-912.md
git commit -m "refactor: complete file and test organization audit"
```

## Task 16: Workspace Verification And Residual Redundancy Audit

**Files:**
- Modify only files needed to fix verification failures
- Maybe modify: `docs/design/refactor-roadmap-912.md` with final residual notes

**Step 1: Search for forbidden residuals**

Run:

```bash
rg -n "TensorExec|try_grad|Tensor::svd|\\.svd\\(&mut|Error::BackendFailure\\s*\\{" tenferro-* README.md docs -g'*.rs' -g'*.md'
rg -n "NaryEinsum|StdTensorOp::Svd|StdTensorOp::Qr|StdTensorOp::Cholesky" tenferro-* -g'*.rs'
rg -n "TypeId|type_name::<" tenferro-* -g'*.rs'
```

Expected:

- no `TensorExec`
- no current `try_grad`
- no direct tensor linalg docs/API
- no production direct backend failure construction
- no linalg as core `StdTensorOp`
- no new TypeId/type-name dispatch

**Step 2: Run final local gate**

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

**Step 3: Run CUDA gate**

```bash
CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.0 LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-gpu --features cuda -- --ignored
CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.0 LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-linalg --features cuda -- --ignored
```

**Step 4: Fix failures**

For each failure, use `superpowers:systematic-debugging` before changing code.
After each fix, rerun the smallest failing command, then rerun the final gate.

**Step 5: Final commit**

```bash
git add .
git commit -m "refactor: complete issue 912 roadmap"
```

If there is nothing to commit because all previous task commits covered the
final state, do not create an empty commit.
