# Structured Error Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace duplicated and stringly tensor errors with shared structured validation, align eager and traced failure timing, make traced reshape fallible, and enforce complete public `# Errors` / `# Panics` documentation.

**Architecture:** `tenferro-tensor-core` owns `ValidationError`, detailed shape payloads, and coarse `ErrorKind`; higher crates wrap those values with operation and runtime phase context while retaining crate-local domain errors. Implementation proceeds strictly bottom-up so every consumer migrates to one public model before release, with no compatibility aliases or deprecated variants.

**Tech Stack:** Rust 2021, `thiserror`, Cargo workspace, Clippy, rustdoc/doctests, Python repository-rules review.

## Global Constraints

- Caller-controlled invalid input returns a typed error and must not panic.
- Validate at the earliest phase with sufficient information.
- Error kind and error phase remain separate.
- Never convert a typed in-workspace error to `String`; strings are allowed only when an external/vendor API supplies no structured source, or at display/logging/FFI/serialization boundaries.
- Explicit fallible methods are canonical; `Add<Output = Result<_>>` remains convenience syntax.
- Do not implement fallible `AddAssign` or add a panicking operator alternative.
- Every public `Result` function/method/trait method has a concrete `# Errors` section; traced deferred validation also has `# Deferred errors`.
- Every documented public panic contract has `# Panics`.
- Pre-1.0 API cleanliness takes priority over compatibility: no aliases, deprecated shims, legacy error variants, or old traced reshape wrapper survive the final tree.
- Keep `tenferro-runtime` free of AD ownership and preserve the existing runtime/AD dependency boundary.
- Work test-first, keep the workspace compiling at each commit, and commit only files named by the current task.

---

## File and dependency map

| Layer | Primary files | Responsibility |
|---|---|---|
| Shared validation | `crates/tenferro-tensor-core/src/error.rs`, `src/lib.rs`, `src/layout.rs`, `src/rank.rs` | Detailed validation payloads and stable classification |
| Runtime tensor error | `crates/tenferro-tensor/src/error.rs`, `src/validate/mod.rs`, `src/types.rs`, `src/config.rs`, `src/backend.rs` | Operation context, backend/extension sources, tensor execution errors |
| Graph/runtime error | `crates/tenferro-runtime/src/error.rs`, `src/traced.rs`, `src/shape_infer.rs`, `src/graph/compiler.rs`, `src/graph/executor.rs`, `src/eager_exec.rs` | Graph phase context and earliest validation |
| AD propagation | `crates/tenferro-ad/src/ad_rule_error.rs`, `src/eager_ops.rs`, `src/eager_exec.rs`, `src/shape_packing.rs`, `src/traced.rs` | Preserve runtime/tensor classification through eager and traced AD |
| Extension ownership | `crates/tenferro-einsum/src/error.rs`, `src/concrete.rs`, `src/eager_ad.rs`, `src/traced.rs`, `src/extension.rs`; linalg/FFT/XLA and `ext/*` error sites | Local parsing/planning/numerical sources plus shared validation |
| Public explanation | `crates/tenferro-runtime/src/lib.rs`, `src/traced.rs`, `docs/getting-started/core-concepts.md`, `docs/spec/api-conventions.md` | Correct fallible operator and deferred-error examples |
| Governance | `REPOSITORY_RULES.md`, `.github/workflows/ci.yml`, `scripts/test-repository-rules-review.py` | Required docs and workspace/extension audit gates |

---

### Task 1: Introduce the shared validation vocabulary in tensor-core

**Files:**
- Create: `crates/tenferro-tensor-core/src/error.rs`
- Modify: `crates/tenferro-tensor-core/src/lib.rs:35-125`
- Modify: `crates/tenferro-tensor-core/src/layout.rs:175-535`
- Modify: `crates/tenferro-tensor-core/src/rank.rs:1-110`
- Modify: `crates/tenferro-tensor-core/tests/core.rs:1-180`

**Interfaces:**
- Produces: `ValidationError`, `ShapeMismatch`, `ValidationKind`, `ErrorKind`, and `Result<T> = core::result::Result<T, ValidationError>`.
- Produces: `ValidationError::kind(&self) -> ValidationKind`.
- Consumed by: every later task.

- [ ] **Step 1: Add failing classification and payload tests**

Add these tests to `crates/tenferro-tensor-core/tests/core.rs` and replace the existing `Error` import with `ValidationError`:

```rust
use tenferro_tensor_core::{
    ErrorKind, ShapeMismatch, ShapeVec, ValidationError, ValidationKind,
};

#[test]
fn shape_mismatch_keeps_machine_readable_payload() {
    let err = ValidationError::ShapeMismatch(ShapeMismatch::IncompatibleShapes {
        lhs: ShapeVec::from_vec(vec![2, 3]),
        rhs: ShapeVec::from_vec(vec![2, 4]),
    });

    assert_eq!(err.kind(), ValidationKind::ShapeMismatch);
    assert!(matches!(
        err,
        ValidationError::ShapeMismatch(ShapeMismatch::IncompatibleShapes {
            ref lhs,
            ref rhs,
        }) if lhs.as_slice() == [2, 3] && rhs.as_slice() == [2, 4]
    ));
}

#[test]
fn public_error_kind_can_classify_validation() {
    assert_eq!(
        ErrorKind::Validation(ValidationKind::RankMismatch),
        ErrorKind::Validation(ValidationKind::RankMismatch)
    );
}
```

- [ ] **Step 2: Run the tests to verify the new API is absent**

Run:

```bash
cargo test -p tenferro-tensor-core --test core shape_mismatch_keeps_machine_readable_payload
```

Expected: compile failure for unresolved `ValidationError`, `ShapeMismatch`, `ValidationKind`, and `ErrorKind`.

- [ ] **Step 3: Move the error model into `src/error.rs`**

Create the following public shape, retaining every existing tensor-core validation case as a structured `ValidationError` variant:

```rust
use crate::{DType, ShapeVec};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ValidationKind {
    ShapeMismatch,
    RankMismatch,
    AxisOutOfBounds,
    DTypeMismatch,
    InvalidArgument,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    Validation(ValidationKind),
    Unsupported,
    NumericalFailure,
    BackendFailure,
    Io,
    RuntimeState,
    Internal,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ShapeMismatch {
    #[error("incompatible shapes: lhs={lhs:?}, rhs={rhs:?}")]
    IncompatibleShapes { lhs: ShapeVec, rhs: ShapeVec },
    #[error("shape mismatch: expected={expected:?}, actual={actual:?}")]
    ExpectedActual { expected: ShapeVec, actual: ShapeVec },
    #[error("reshape element-count mismatch: from {from} to {to}")]
    ReshapeElementCount { from: usize, to: usize },
    #[error(
        "contracted dimensions differ: lhs axis {lhs_axis} ({lhs_size}) vs rhs axis {rhs_axis} ({rhs_size})"
    )]
    ContractedDimensions {
        lhs_axis: usize,
        lhs_size: usize,
        rhs_axis: usize,
        rhs_size: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ValidationError {
    #[error(transparent)]
    ShapeMismatch(#[from] ShapeMismatch),
    #[error("shape product {expected} does not match data length {actual}")]
    ShapeDataLengthMismatch { expected: usize, actual: usize },
    #[error("rank mismatch: expected {expected}, actual {actual}")]
    RankMismatch { expected: usize, actual: usize },
    #[error("axis {axis} out of bounds for rank {rank}")]
    AxisOutOfBounds { axis: usize, rank: usize },
    #[error("duplicate {role} axis {axis}")]
    DuplicateAxis { axis: usize, role: &'static str },
    #[error("axis {axis} appears in both {first_role} and {second_role}")]
    AxisRoleConflict {
        axis: usize,
        first_role: &'static str,
        second_role: &'static str,
    },
    #[error("invalid permutation length: expected {expected}, actual {actual}")]
    InvalidPermutationLength { expected: usize, actual: usize },
    #[error("invalid slice step {step}; zero is invalid")]
    InvalidSliceStep { step: isize },
    #[error("invalid slice bounds: start={start}, end={end}, axis_len={axis_len}")]
    InvalidSliceBounds { start: isize, end: isize, axis_len: usize },
    #[error("dtype mismatch: expected {expected:?}, actual {actual:?}")]
    DTypeMismatch { expected: DType, actual: DType },
    #[error("invalid argument {argument}: {message}")]
    InvalidArgument { argument: &'static str, message: String },
    #[error("view is not slice-contiguous")]
    NonContiguousViewAsSlice,
    #[error("view metadata is out of borrowed-slice bounds")]
    ViewOutOfBounds,
    #[error("mutable tensor layout may overlap physical elements")]
    OverlappingMutableLayout,
    #[error("integer overflow while validating tensor metadata")]
    IntegerOverflow,
}

impl ValidationError {
    pub fn kind(&self) -> ValidationKind {
        match self {
            Self::ShapeMismatch(_) | Self::ShapeDataLengthMismatch { .. } => {
                ValidationKind::ShapeMismatch
            }
            Self::RankMismatch { .. } | Self::InvalidPermutationLength { .. } => {
                ValidationKind::RankMismatch
            }
            Self::AxisOutOfBounds { .. } => ValidationKind::AxisOutOfBounds,
            Self::DTypeMismatch { .. } => ValidationKind::DTypeMismatch,
            _ => ValidationKind::InvalidArgument,
        }
    }
}
```

Move `Result<T>` to use `ValidationError`, add `mod error; pub use error::{...};` after `ShapeVec` is declared, and remove the old `Error` enum completely. Update tensor-core call sites and tests from `Error::...` to `ValidationError::...`; `DuplicateAxis` constructors now supply the role (`"permutation"` where applicable), and reshape uses `ShapeMismatch::ReshapeElementCount`.

- [ ] **Step 4: Add concrete `# Errors` sections for tensor-core**

The baseline Clippy run reports 27 missing sections in `src/layout.rs`, `src/rank.rs`, and `src/lib.rs`. For each reported public function, name the actual `ValidationError` variant. Use wording such as:

```rust
/// # Errors
///
/// Returns [`ValidationError::RankMismatch`] when the supplied shape length
/// differs from the compile-time rank.
```

Do not use “returns an error on failure.” Run Clippy repeatedly until the package has zero `missing_errors_doc` and `missing_panics_doc` findings.

- [ ] **Step 5: Verify tensor-core**

Run:

```bash
cargo test -p tenferro-tensor-core
cargo clippy -p tenferro-tensor-core --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
```

Expected: both commands pass.

- [ ] **Step 6: Commit the shared layer**

```bash
git add -- \
  crates/tenferro-tensor-core/src/error.rs \
  crates/tenferro-tensor-core/src/lib.rs \
  crates/tenferro-tensor-core/src/layout.rs \
  crates/tenferro-tensor-core/src/rank.rs \
  crates/tenferro-tensor-core/tests/core.rs
git commit -m "refactor: centralize tensor validation errors"
```

---

### Task 2: Wrap shared validation in tenferro-tensor without losing sources

**Files:**
- Modify: `crates/tenferro-tensor/src/error.rs:1-113`
- Modify: `crates/tenferro-tensor/src/lib.rs:35-65`
- Modify: `crates/tenferro-tensor/src/validate/mod.rs:1-240`
- Modify: `crates/tenferro-tensor/src/validate/tests.rs:1-80`
- Modify: `crates/tenferro-tensor/src/types.rs:3431-3467`
- Modify: `crates/tenferro-tensor/src/config.rs`
- Modify: `crates/tenferro-tensor/src/backend.rs`
- Test: `crates/tenferro-tensor/src/tests/types_tests.rs`

**Interfaces:**
- Consumes: Task 1 `ValidationError`, `ShapeMismatch`, `ValidationKind`, `ErrorKind`.
- Produces: `tenferro_tensor::Error::{Validation, UnsupportedDTypeConversion, BackendFailure, BackendSource, Extension, MissingValue, Internal}`.
- Produces: constructors `validation`, `invalid_argument`, `backend_failure`, `backend_source`, `extension`, plus `kind()`.

- [ ] **Step 1: Add failing tensor-wrapper tests**

Add to `crates/tenferro-tensor/src/tests/types_tests.rs`:

```rust
use std::error::Error as _;
use tenferro_tensor::{
    Error, ErrorKind, ShapeMismatch, ShapeVec, ValidationError, ValidationKind,
};

#[test]
fn tensor_error_preserves_shared_validation_source() {
    let err = Error::validation(
        "add",
        ShapeMismatch::IncompatibleShapes {
            lhs: ShapeVec::from_vec(vec![2]),
            rhs: ShapeVec::from_vec(vec![3]),
        }
        .into(),
    );

    assert_eq!(
        err.kind(),
        ErrorKind::Validation(ValidationKind::ShapeMismatch)
    );
    assert!(err.source().is_some());
    assert!(matches!(
        err,
        Error::Validation {
            op: "add",
            source: ValidationError::ShapeMismatch(_),
        }
    ));
}

#[test]
fn typed_backend_source_is_not_formatted_away() {
    let err = Error::backend_source("load", std::io::Error::other("device read failed"));
    assert_eq!(err.kind(), ErrorKind::BackendFailure);
    assert!(err.source().is_some());
}
```

- [ ] **Step 2: Verify the tests fail against the old enum**

Run:

```bash
cargo test -p tenferro-tensor tensor_error_preserves_shared_validation_source
```

Expected: compile failure because the constructors and wrapper variants do not exist.

- [ ] **Step 3: Replace duplicated tensor validation variants**

Implement this outer shape in `src/error.rs`:

```rust
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{op}: {source}")]
    Validation {
        op: &'static str,
        #[source]
        source: ValidationError,
    },
    #[error("{op}: unsupported dtype conversion from {from:?} to {to:?}: {message}")]
    UnsupportedDTypeConversion {
        op: &'static str,
        from: DType,
        to: DType,
        message: String,
    },
    #[error("{op}: backend failure: {message}")]
    BackendFailure { op: &'static str, message: String },
    #[error("{op}: backend failure: {source}")]
    BackendSource {
        op: &'static str,
        #[source]
        source: BoxError,
    },
    #[error("{op}: extension {family} failed: {source}")]
    Extension {
        op: &'static str,
        family: &'static str,
        kind: ErrorKind,
        #[source]
        source: BoxError,
    },
    #[error("missing runtime value for slot {slot}")]
    MissingValue { slot: usize },
    #[error("internal tensor error: {0}")]
    Internal(String),
}
```

Implement:

```rust
pub fn validation(op: &'static str, source: ValidationError) -> Self;
pub fn invalid_argument(
    op: &'static str,
    argument: &'static str,
    message: impl Into<String>,
) -> Self;
pub fn backend_failure(op: &'static str, message: impl Into<String>) -> Self;
pub fn backend_source<E>(op: &'static str, source: E) -> Self
where
    E: std::error::Error + Send + Sync + 'static;
pub fn extension<E>(
    op: &'static str,
    family: &'static str,
    kind: ErrorKind,
    source: E,
) -> Self
where
    E: std::error::Error + Send + Sync + 'static;
pub fn kind(&self) -> ErrorKind;
```

`backend_failure` deliberately accepts `Into<String>`, not `Display`, so passing an arbitrary typed error does not silently stringify it. `backend_source` and `extension` box the original source. Drop `Clone`, `Eq`, and `PartialEq` from the outer error instead of sacrificing sources.

- [ ] **Step 4: Migrate tensor validation producers**

Use this mapping throughout the named tensor files:

| Old tensor variant | New representation |
|---|---|
| `AxisOutOfBounds` | `Error::validation(op, ValidationError::AxisOutOfBounds { axis, rank })` |
| `DuplicateAxis` | `Error::validation(op, ValidationError::DuplicateAxis { axis, role })` |
| `AxisRoleConflict` | `Error::validation(op, ValidationError::AxisRoleConflict { ... })` |
| `ShapeMismatch` | `Error::validation(op, ShapeMismatch::IncompatibleShapes { ... }.into())` |
| `RankMismatch` | `Error::validation(op, ValidationError::RankMismatch { ... })` |
| `DTypeMismatch` | `Error::validation(op, ValidationError::DTypeMismatch { expected, actual })` |
| `InvalidConfig` | `Error::invalid_argument(op, argument_name, message)` |
| typed error passed to `backend_failure` | `Error::backend_source(op, error)` |
| backend/vendor text only | `Error::backend_failure(op, message)` |

Change `tensor_layout_error` to wrap the complete tensor-core source instead of matching and stringifying unmatched variants:

```rust
fn tensor_layout_error(op: &'static str, err: ValidationError) -> crate::Error {
    crate::Error::validation(op, err)
}
```

Top-level reexports must include `ErrorKind`, `ShapeMismatch`, `ValidationError`, and `ValidationKind` so normal tensor and extension users do not need to reach through `tenferro_tensor::core`.

- [ ] **Step 5: Update tensor rustdoc and tests to match structured variants**

Replace string-format assertions with `kind()` and payload matching. Add concrete `# Errors` sections for all public `Result` APIs reported by package Clippy, naming the exact validation/backend variant.

- [ ] **Step 6: Verify and commit tensor**

Run:

```bash
cargo test -p tenferro-tensor
cargo clippy -p tenferro-tensor --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
```

Expected: pass.

```bash
git add -- \
  crates/tenferro-tensor/src/error.rs \
  crates/tenferro-tensor/src/lib.rs \
  crates/tenferro-tensor/src/validate/mod.rs \
  crates/tenferro-tensor/src/validate/tests.rs \
  crates/tenferro-tensor/src/types.rs \
  crates/tenferro-tensor/src/config.rs \
  crates/tenferro-tensor/src/backend.rs \
  crates/tenferro-tensor/src/tests/types_tests.rs
git commit -m "refactor: preserve structured tensor error sources"
```

---

### Task 3: Add runtime phase context and remove stringly graph validation

**Files:**
- Modify: `crates/tenferro-runtime/src/error.rs:1-150`
- Modify: `crates/tenferro-runtime/src/traced.rs`
- Modify: `crates/tenferro-runtime/src/shape_infer.rs`
- Modify: `crates/tenferro-runtime/src/shape_packing.rs`
- Modify: `crates/tenferro-runtime/src/graph/compiler.rs`
- Modify: `crates/tenferro-runtime/src/graph/executor.rs`
- Modify: `crates/tenferro-runtime/src/eager_exec.rs`
- Modify: `crates/tenferro-runtime/tests/runtime_public_api.rs`
- Modify: `crates/tenferro-runtime/src/traced/tests.rs`
- Modify: `crates/tenferro-runtime/src/shape_infer/tests.rs`

**Interfaces:**
- Consumes: Tasks 1-2 error types.
- Produces: `ErrorPhase::{GraphBuild, Compile, Execution}`.
- Produces: `Error::validation(op, phase, source)`, `Error::kind()`, and `Error::phase()`.

- [ ] **Step 1: Add failing runtime classification tests**

Add to `crates/tenferro-runtime/tests/runtime_public_api.rs`:

```rust
use tenferro_runtime::{Error, ErrorPhase};
use tenferro_tensor::{ErrorKind, ShapeMismatch, ValidationKind};

#[test]
fn runtime_validation_separates_kind_from_phase() {
    let err = Error::validation(
        "reshape",
        ErrorPhase::GraphBuild,
        ShapeMismatch::ReshapeElementCount { from: 2, to: 3 }.into(),
    );

    assert_eq!(
        err.kind(),
        ErrorKind::Validation(ValidationKind::ShapeMismatch)
    );
    assert_eq!(err.phase(), Some(ErrorPhase::GraphBuild));
}
```

- [ ] **Step 2: Verify the runtime API test fails**

Run:

```bash
cargo test -p tenferro-runtime --test runtime_public_api runtime_validation_separates_kind_from_phase
```

Expected: compile failure for missing `ErrorPhase` and `Error::validation`.

- [ ] **Step 3: Implement phase-aware runtime errors**

Add:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorPhase {
    GraphBuild,
    Compile,
    Execution,
}

#[error("{op} ({phase:?}): {source}")]
Validation {
    op: &'static str,
    phase: ErrorPhase,
    #[source]
    source: ValidationError,
},
```

Keep `TensorRuntime(#[from] tenferro_tensor::Error)` as the typed execution boundary; `kind()` delegates to the tensor error and `phase()` returns `Some(ErrorPhase::Execution)` for it. Runtime-state variants such as context mismatch and poisoned/missing bindings map to `ErrorKind::RuntimeState`. `UnsupportedAdRule` maps to `ErrorKind::Unsupported`; `Internal` maps to `ErrorKind::Internal`.

During this task, leave the einsum-owned legacy `InvalidSubscripts` and `ContractionError` variants only as a temporary compile bridge. Task 6 removes both before the final release. Do not add any new call sites.

- [ ] **Step 4: Replace graph validation strings with shared payloads**

For caller-controlled shape/rank/axis/dtype/config failures in `traced.rs`, `shape_infer.rs`, `shape_packing.rs`, and graph binding validation, construct `Error::Validation` with the earliest phase. Convert:

- concrete graph-builder failures to `GraphBuild`;
- failures found while lowering/inference with concrete input specs to `Compile`;
- binding/executor failures to `Execution`.

Use specialized shared variants where facts exist. Use `ValidationError::InvalidArgument { argument, message }` only for caller-invalid configurations that do not have a more specific shared payload. Graph corruption or impossible instruction shapes become `Error::Internal`, not `InvalidArgument`.

Remove `shape_infer_from_tensor_error(err.to_string())`; match `tenferro_tensor::Error::Validation { source, .. }` and retain `source`, otherwise retain the complete tensor error through `TensorRuntime` or an internal typed source. Replace the extension wrapper in `eager_exec.rs` with `tenferro_tensor::Error::extension(...)` rather than formatting `family_id` and the source.

- [ ] **Step 5: Update runtime tests to assert kinds, phases, and sources**

Replace tests that match `InvalidGraphBuild` or inspect messages for known validation with structured matches. Keep message assertions only for genuinely opaque external messages.

- [ ] **Step 6: Add complete runtime rustdoc**

Run package Clippy with the two documentation lints and add `# Errors` to every reported public `Result` API. For traced functions that may defer symbolic checks, add `# Deferred errors` separately.

- [ ] **Step 7: Verify and commit runtime phase support**

Run:

```bash
cargo test -p tenferro-runtime
cargo clippy -p tenferro-runtime --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
```

Expected: pass.

```bash
git add -- \
  crates/tenferro-runtime/src/error.rs \
  crates/tenferro-runtime/src/traced.rs \
  crates/tenferro-runtime/src/shape_infer.rs \
  crates/tenferro-runtime/src/shape_packing.rs \
  crates/tenferro-runtime/src/graph/compiler.rs \
  crates/tenferro-runtime/src/graph/executor.rs \
  crates/tenferro-runtime/src/eager_exec.rs \
  crates/tenferro-runtime/tests/runtime_public_api.rs \
  crates/tenferro-runtime/src/traced/tests.rs \
  crates/tenferro-runtime/src/shape_infer/tests.rs
git commit -m "refactor: classify runtime errors by phase"
```

---

### Task 4: Make traced reshape fallible and validate at the earliest phase

**Files:**
- Modify: `crates/tenferro-runtime/src/traced.rs:1463-1610`
- Modify: `crates/tenferro-runtime/src/shape_infer.rs:220-250`
- Modify: `crates/tenferro-runtime/src/graph/compiler.rs:386-431`
- Modify: `crates/tenferro-runtime/src/traced/tests.rs`
- Modify: `crates/tenferro-runtime/src/shape_infer/tests.rs`
- Modify: all workspace call sites that consume `TracedTensor::reshape`

**Interfaces:**
- Changes: `TracedTensor::reshape(&self, shape: &[usize]) -> Result<TracedTensor>`.
- Preserves: `reshape_sym` as fallible; both concrete and symbolic APIs use the same element-count validation.

- [ ] **Step 1: Add graph-build and compile-phase reshape tests**

Add to `crates/tenferro-runtime/src/traced/tests.rs`:

```rust
use crate::{Error, ErrorPhase, GraphCompiler};
use tenferro_tensor::{ShapeMismatch, ValidationError};

#[test]
fn concrete_reshape_rejects_element_count_at_graph_build() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let err = x.reshape(&[3]).unwrap_err();

    assert!(matches!(
        err,
        Error::Validation {
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::ShapeMismatch(
                ShapeMismatch::ReshapeElementCount { from: 2, to: 3 }
            ),
            ..
        }
    ));
}

#[test]
fn symbolic_reshape_defers_until_input_specs_are_concrete() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let y = x.reshape(&[3]).unwrap();
    let mut compiler = GraphCompiler::new();

    let err = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .unwrap_err();

    assert!(matches!(
        err,
        Error::Validation {
            phase: ErrorPhase::Compile,
            source: ValidationError::ShapeMismatch(
                ShapeMismatch::ReshapeElementCount { from: 2, to: 3 }
            ),
            ..
        }
    ));
}
```

- [ ] **Step 2: Verify the new contract fails**

Run:

```bash
cargo test -p tenferro-runtime concrete_reshape_rejects_element_count_at_graph_build
```

Expected: compile failure because `reshape` still returns `TracedTensor`.

- [ ] **Step 3: Implement one reshape validation helper**

Create a private helper in `traced.rs` that uses checked products and returns:

```rust
Error::validation(
    "TracedTensor::reshape",
    ErrorPhase::GraphBuild,
    ShapeMismatch::ReshapeElementCount { from, to }.into(),
)
```

when both shapes are concrete and products differ. Shape-product overflow returns `ValidationError::IntegerOverflow` at the same phase. `reshape` calls the helper before creating the node and returns `Ok(apply_unary(...))`.

In `shape_infer.rs`, the `StdTensorOp::Reshape` arm compares products whenever the input and target `DimExpr` values are all constants. It returns the same `ShapeMismatch::ReshapeElementCount` under `ErrorPhase::Compile`. When any dimension remains symbolic, it returns the target expression and leaves the existing reshape node as the deferred execution constraint. The backend reshape path must already return the same shared payload after Task 2; retain it as the execution fallback.

- [ ] **Step 4: Migrate all call sites without a shim**

Change successful traced calls to `x.reshape(shape)?`, `.reshape(shape).unwrap()`, or `.reshape(shape).expect("...")` only in tests/internal proven setup. Do not add `reshape_unchecked`, `reshape_infallible`, or a deprecated wrapper. Use:

```bash
rg -n '\.reshape\(' crates docs samples ext
```

and let `cargo check --workspace --all-targets` identify type-directed traced call sites that the textual search cannot distinguish.

- [ ] **Step 5: Document immediate and deferred errors**

The `reshape` rustdoc must contain exact `# Errors` and `# Deferred errors` sections, naming `ShapeMismatch::ReshapeElementCount` and integer overflow. Update `reshape_sym` with the same distinction.

- [ ] **Step 6: Verify and commit reshape**

Run:

```bash
cargo test -p tenferro-runtime concrete_reshape_rejects_element_count_at_graph_build
cargo test -p tenferro-runtime symbolic_reshape_defers_until_input_specs_are_concrete
cargo check --workspace --all-targets
```

Expected: pass.

```bash
git ls-files --modified --others --exclude-standard -z -- \
  crates docs samples ext | xargs -0 git add --
git diff --cached --name-only
git commit -m "refactor: make traced reshape validate fallibly"
```

The staged-name review must contain only the runtime reshape files and the
compile-discovered call sites changed in Step 4. Unstage anything unrelated
before committing.

---

### Task 5: Align eager and traced AD error propagation

**Files:**
- Modify: `crates/tenferro-ad/src/ad_rule_error.rs`
- Modify: `crates/tenferro-ad/src/eager_ops.rs`
- Modify: `crates/tenferro-ad/src/eager_exec.rs`
- Modify: `crates/tenferro-ad/src/shape_packing.rs`
- Modify: `crates/tenferro-ad/src/traced.rs`
- Modify: `crates/tenferro-ad/tests/fallible_api.rs`
- Modify: `crates/tenferro-ad/tests/dot_general_validation.rs`

**Interfaces:**
- Consumes: runtime `Error::validation`, tensor shared payloads.
- Produces: eager and traced public paths with the same `ValidationKind` for equivalent concrete invalid input.

- [ ] **Step 1: Add a failing eager/traced parity test**

Extend `crates/tenferro-ad/tests/fallible_api.rs` so `eager_binary_methods_return_shape_errors` matches the new tensor wrapper and add:

```rust
#[test]
fn eager_and_traced_shape_mismatch_share_validation_kind() {
    let ctx = EagerRuntime::new();
    let eager_lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let eager_rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap(),
        ctx,
    )
    .unwrap();
    let eager_err = eager_lhs.add(&eager_rhs).unwrap_err();

    let traced_lhs = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();
    let traced_rhs = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
    let traced_err = traced_lhs.add(&traced_rhs).unwrap_err();

    assert_eq!(eager_err.kind(), traced_err.kind());
    assert_eq!(
        traced_err.kind(),
        tenferro_tensor::ErrorKind::Validation(
            tenferro_tensor::ValidationKind::ShapeMismatch
        )
    );
}
```

- [ ] **Step 2: Run the parity test and observe the mismatch**

Run:

```bash
cargo test -p tenferro-ad --test fallible_api eager_and_traced_shape_mismatch_share_validation_kind
```

Expected: failure until both surfaces expose the shared kind.

- [ ] **Step 3: Preserve typed AD categories**

Map `tidu::ADRuleError::InvalidInput { op, message, .. }` to `Error::validation(transform, ErrorPhase::GraphBuild, ValidationError::InvalidArgument { argument: "ad rule input", message: format!("{op}: {message}") })`. Map `Unsupported` to `ErrorKind::Unsupported` through the existing `UnsupportedAdRule` variant.

Where `tidu` itself exposes only a message field, preserve its structured `InvalidInput` versus `Unsupported` category and treat the message as an external dependency boundary; do not stringify another Tenferro error before constructing it.

Update eager validation helpers to construct tensor shared payloads and traced helpers to construct runtime phase-aware payloads. Replace message-only assertions for known rank/axis/shape cases.

- [ ] **Step 4: Add `# Errors` / `# Panics` sections throughout tenferro-ad**

Run package Clippy and fix every reported public item. `# Errors` must name shared variants or delegated runtime kinds; `# Panics` must describe actual contract preconditions, never implementation accidents.

- [ ] **Step 5: Verify and commit AD**

Run:

```bash
cargo test -p tenferro-ad --test fallible_api
cargo test -p tenferro-ad --test dot_general_validation
cargo clippy -p tenferro-ad --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
```

Expected: pass.

```bash
git add -- \
  crates/tenferro-ad/src/ad_rule_error.rs \
  crates/tenferro-ad/src/eager_ops.rs \
  crates/tenferro-ad/src/eager_exec.rs \
  crates/tenferro-ad/src/shape_packing.rs \
  crates/tenferro-ad/src/traced.rs \
  crates/tenferro-ad/tests/fallible_api.rs \
  crates/tenferro-ad/tests/dot_general_validation.rs
git commit -m "refactor: align eager and traced validation errors"
```

---

### Task 6: Make einsum own its domain errors and preserve them at runtime boundaries

**Files:**
- Modify: `crates/tenferro-einsum/src/error.rs`
- Modify: `crates/tenferro-einsum/src/concrete.rs`
- Modify: `crates/tenferro-einsum/src/eager.rs`
- Modify: `crates/tenferro-einsum/src/eager_ad.rs`
- Modify: `crates/tenferro-einsum/src/traced.rs`
- Modify: `crates/tenferro-einsum/src/tensordot.rs`
- Modify: `crates/tenferro-einsum/src/extension.rs`
- Modify: `crates/tenferro-einsum/tests/error_public.rs`
- Modify: `crates/tenferro-einsum/tests/public_surface_contract.rs`
- Modify: `crates/tenferro-runtime/src/error.rs`

**Interfaces:**
- Produces: one `tenferro_einsum::Error` across concrete, eager, and traced public extension APIs.
- Produces: `Error::kind()` and consuming `into_tensor_error(op)`.
- Removes: runtime `InvalidSubscripts` and `ContractionError` variants.

- [ ] **Step 1: Replace the lossy conversion test with source-preservation tests**

Replace `local_error_maps_to_tensor_backend_failure` in `tests/error_public.rs` with:

```rust
#[test]
fn shared_einsum_validation_promotes_directly_to_tensor_validation() {
    let err = Error::validation(
        "einsum",
        tenferro_tensor::ShapeMismatch::ExpectedActual {
            expected: tenferro_tensor::ShapeVec::from_vec(vec![2, 3]),
            actual: tenferro_tensor::ShapeVec::from_vec(vec![2, 4]),
        }
        .into(),
    );

    let tensor_err = err.into_tensor_error("einsum_extension");
    assert!(matches!(tensor_err, TensorError::Validation { .. }));
}

#[test]
fn einsum_planning_error_remains_a_typed_extension_source() {
    let err = Error::planning("no valid contraction path");
    let tensor_err = err.into_tensor_error("einsum_extension");

    assert_eq!(tensor_err.kind(), tenferro_tensor::ErrorKind::RuntimeState);
    assert!(matches!(tensor_err, TensorError::Extension { .. }));
}
```

- [ ] **Step 2: Verify the source-preservation tests fail**

Run:

```bash
cargo test -p tenferro-einsum --test error_public
```

Expected: compile failure because `validation`, `planning`, and `into_tensor_error` do not exist.

- [ ] **Step 3: Redesign the einsum outer error**

Use these owned categories:

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("{op}: {source}")]
    Validation {
        op: &'static str,
        #[source]
        source: ValidationError,
    },
    #[error("invalid einsum subscripts: {message}")]
    InvalidSubscripts { message: String },
    #[error("einsum planning failed: {message}")]
    Planning { message: String },
    #[error("einsum numerical failure: {message}")]
    Numerical { message: String },
    #[error(transparent)]
    Tensor(#[from] tenferro_tensor::Error),
    #[error(transparent)]
    Runtime(#[from] tenferro_runtime::Error),
    #[cfg(feature = "autodiff")]
    #[error(transparent)]
    Ad(#[from] tenferro_ad::Error),
}
```

The transparent tuple variants provide both `#[source]` and `From`. `kind()` delegates to nested Tenferro errors and maps invalid subscripts to validation-invalid-argument, planning to runtime state, and numerical to numerical failure.

`into_tensor_error(self, op)` consumes `self`: promote `Validation` directly to `TensorError::validation`; unwrap `Tensor`; convert all remaining variants with `TensorError::extension(op, EINSUM_EXTENSION_FAMILY_ID, self.kind(), self)`. Remove the old borrowing `to_tensor_error(&self, ...)` so the source cannot be formatted away.

- [ ] **Step 4: Unify public Result ownership**

Concrete, eager, and traced einsum extension traits and free functions return `crate::Result<T>` and use `?` through `From` implementations for tensor/runtime/AD errors. Parsing and planning remain local variants. The type-erased extension runtime still satisfies `tenferro_tensor::Result` by calling `into_tensor_error` at the final registry boundary.

Remove `to_tenferro_error(error.to_string())`, `ContractionError(String)`, and every other string conversion of an `EinsumError`. Once all call sites compile, delete the temporary runtime `InvalidSubscripts` and `ContractionError` variants.

- [ ] **Step 5: Update einsum rustdoc and test surfaces**

All public trait methods and functions returning `crate::Result` receive exact `# Errors`. Examples use `# Ok::<(), tenferro_einsum::Error>(())`. Tests match local variants, shared validation, `kind()`, or `source()` rather than formatted text.

- [ ] **Step 6: Verify and commit einsum ownership**

Run:

```bash
cargo test -p tenferro-einsum --all-features
cargo test -p tenferro-runtime
cargo clippy -p tenferro-einsum --all-targets --all-features -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
```

Expected: pass.

```bash
git add -- \
  crates/tenferro-einsum/src/error.rs \
  crates/tenferro-einsum/src/concrete.rs \
  crates/tenferro-einsum/src/eager.rs \
  crates/tenferro-einsum/src/eager_ad.rs \
  crates/tenferro-einsum/src/traced.rs \
  crates/tenferro-einsum/src/tensordot.rs \
  crates/tenferro-einsum/src/extension.rs \
  crates/tenferro-einsum/tests/error_public.rs \
  crates/tenferro-einsum/tests/public_surface_contract.rs \
  crates/tenferro-runtime/src/error.rs
git commit -m "refactor: preserve einsum domain errors"
```

---

### Task 7: Migrate linalg, FFT, XLA, CPU/GPU, and standalone extensions

**Files:**
- Create: `crates/tenferro-linalg/src/error.rs`
- Modify: `crates/tenferro-linalg/src/lib.rs`
- Modify: `crates/tenferro-linalg/src/ad.rs`
- Modify: `crates/tenferro-linalg/src/backend.rs`
- Modify: `crates/tenferro-linalg/src/cpu/backend.rs`
- Modify: `crates/tenferro-linalg/src/cpu/linalg/faer_linalg.rs`
- Modify: `crates/tenferro-linalg/src/cpu/linalg/lapack_linalg/cholesky.rs`
- Modify: `crates/tenferro-linalg/src/cpu/linalg/lapack_linalg/eig.rs`
- Modify: `crates/tenferro-linalg/src/cpu/linalg/lapack_linalg/eigh.rs`
- Modify: `crates/tenferro-linalg/src/cpu/linalg/lapack_linalg/full_piv_lu.rs`
- Modify: `crates/tenferro-linalg/src/cpu/linalg/lapack_linalg/helpers.rs`
- Modify: `crates/tenferro-linalg/src/cpu/linalg/lapack_linalg/lu.rs`
- Modify: `crates/tenferro-linalg/src/cpu/linalg/lapack_linalg/solve.rs`
- Modify: `crates/tenferro-linalg/src/cpu/linalg/lapack_linalg/triangular_solve.rs`
- Modify: `crates/tenferro-linalg/src/eager_ext.rs`
- Modify: `crates/tenferro-linalg/src/extension.rs`
- Modify: `crates/tenferro-linalg/src/gpu/ffi/cusolver.rs`
- Modify: `crates/tenferro-linalg/src/gpu/ffi/mod.rs`
- Modify: `crates/tenferro-linalg/src/gpu/linalg.rs`
- Modify: `crates/tenferro-linalg/src/traced.rs`
- Modify: `crates/tenferro-linalg/tests/backend_errors.rs`
- Modify: `crates/tenferro-fft/src/lib.rs`
- Modify: `crates/tenferro-fft/src/concrete_tests.rs`
- Modify: `crates/tenferro-xla/src/error.rs`
- Modify: `crates/tenferro-xla/tests/public_api.rs`
- Modify: the error-producing CPU files printed by `rg -l 'backend_failure|InvalidConfig|to_string\(\)' crates/tenferro-cpu/src`, currently `analytic.rs`, `backend.rs`, `context.rs`, `elementwise.rs`, `exec_session.rs`, `gemm/{blas_gemm,mod,strided_dot}.rs`, `indexing.rs`, `indexing_alloc.rs`, `inject.rs`, `lib.rs`, `reduction.rs`, and `structural.rs`, plus their colocated tests when assertions change
- Modify: the error-producing GPU files printed by the same scan under `crates/tenferro-gpu/src`, currently `cubecl/{dispatch,gemm,interop,memory,mod,op_descriptor,runtime}.rs`, `cubecl/ffi/{cutensor,mod}.rs`, `cubecl/fusion/classify.rs`, and `webgpu/{gemm,memory,mod,runtime}.rs`, plus their colocated tests when assertions change
- Modify: `ext/tropical/src/cpu.rs`
- Modify: `ext/tropical/src/einsum.rs`
- Modify: `ext/tropical/src/extension.rs`
- Modify: `ext/tropical/src/traced.rs`
- Modify: `ext/sparse/src/extension.rs`
- Modify: `ext/sparse/src/sparse.rs`
- Test: existing package and extension error tests named below

**Interfaces:**
- Consumes: shared validation and tensor extension-source constructors.
- Produces: local `tenferro_linalg::Error` payloads for numerical/unsupported linalg failures and `kind()` for XLA errors.

- [ ] **Step 1: Add cross-extension classification regressions**

Add focused assertions to existing tests:

- `tenferro-linalg/tests/backend_errors.rs`: singular/non-converging failures report `ErrorKind::NumericalFailure` and retain a `tenferro_linalg::Error` source.
- `tenferro-fft/src/concrete_tests.rs`: invalid FFT axis reports `ValidationKind::AxisOutOfBounds`, not `InvalidConfig(String)`.
- `tenferro-xla/tests/public_api.rs`: unsupported dtype reports `ErrorKind::Unsupported`.
- `ext/tropical/tests/tropical_einsum.rs`: incompatible tensor shapes report shared `ValidationKind::ShapeMismatch`.
- `ext/sparse/tests/sparse_ad.rs`: invalid sparse metadata reports shared validation, not backend failure.

Run each targeted test and confirm it fails against the old classifications.

- [ ] **Step 2: Add linalg domain sources**

Create this exact initial public linalg error set:

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("{op} did not converge")]
    NonConvergence { op: &'static str },
    #[error("{op} is singular")]
    Singular { op: &'static str },
    #[error("{op} does not support dtype {dtype:?}")]
    UnsupportedDType { op: &'static str, dtype: DType },
}
```

Implement `kind()`: non-convergence/singular are numerical failures and unsupported dtype is unsupported. At tensor/runtime boundaries wrap these with `TensorError::extension(..., LINALG_EXTENSION_FAMILY_ID, error.kind(), error)` instead of calling `backend_failure(error.to_string())`. Shared rank/shape/axis/dtype validation bypasses the extension box and uses `TensorError::validation`.

- [ ] **Step 3: Migrate FFT and XLA**

Replace `tensor_fft_config_error` and `fft_config_error` with shared validation constructors. Negative/out-of-range axes use `AxisOutOfBounds`; invalid lengths and norm/config values use `InvalidArgument`; incompatible real/complex dtype relations use `DTypeMismatch` or the existing structured unsupported category.

Add `tenferro_xla::Error::kind() -> ErrorKind`; retain structured local variants. Replace any conversion from a typed I/O, plugin, runtime, or tensor error to a message with a `#[source]` variant. Vendor/PJRT text that has no source may remain text and maps to backend failure.

- [ ] **Step 4: Migrate CPU, GPU, tropical, and sparse boundaries**

Use this decision rule at every current `backend_failure`, `InvalidConfig`, or `to_string()` call site:

1. caller tensor relationship known: shared `ValidationError`;
2. typed in-workspace kernel/extension error: `backend_source` or `extension` with its source;
3. vendor status/message only: `BackendFailure` text is allowed;
4. impossible internal state: typed `Internal`, not validation.

Do not rewrite display-only `to_string()` calls in tests, logs, cache keys, or actual FFI message extraction. Update tests that match removed variants.

- [ ] **Step 5: Complete public error documentation package by package**

Run Clippy separately for `tenferro-linalg`, `tenferro-fft`, `tenferro-xla`, `tenferro-cpu`, `tenferro-gpu`, `ext/tropical`, and `ext/sparse`; add concrete `# Errors` / `# Panics` sections until every command is clean.

- [ ] **Step 6: Verify extension migration**

Run:

```bash
cargo test -p tenferro-linalg --all-features
cargo test -p tenferro-fft --all-features
cargo test -p tenferro-xla
cargo test -p tenferro-cpu
cargo test -p tenferro-gpu
cargo test --manifest-path ext/tropical/Cargo.toml --features autodiff
cargo test --manifest-path ext/sparse/Cargo.toml --features autodiff
```

Expected: pass on available non-hardware tests; feature-gated GPU tests that require hardware remain under their existing CI lanes.

- [ ] **Step 7: Commit extension migration**

```bash
git ls-files --modified --others --exclude-standard -z -- \
  crates/tenferro-linalg crates/tenferro-fft crates/tenferro-xla \
  crates/tenferro-cpu crates/tenferro-gpu ext/tropical ext/sparse \
  | xargs -0 git add --
git diff --cached --name-only
git commit -m "refactor: preserve structured extension failures"
```

The staged-name review must contain only the files listed in Task 7 and
colocated assertion tests changed by the classification migration.

---

### Task 8: Explain fallible operators and deferred errors in online documentation

**Files:**
- Modify: `crates/tenferro-runtime/src/lib.rs:1-40`
- Modify: `crates/tenferro-runtime/src/traced.rs:780-875`
- Modify: `docs/getting-started/core-concepts.md:136-175`
- Modify: `docs/spec/api-conventions.md:50-80`
- Modify: the current user-facing reshape examples identified in Task 4; do not edit historical files under `docs/plans/` or `docs/superpowers/specs/`

**Interfaces:**
- Documents: explicit methods as canonical, fallible operators as sugar, `a + b + c` limitation, and immediate/deferred validation.

- [ ] **Step 1: Add a documentation consistency regression**

Extend `scripts/test-doc-consistency.py` to assert that `crates/tenferro-runtime/src/traced.rs` contains all of:

```python
required = [
    "let ab = (&a + &b)?;",
    "let abc = (&ab + &c)?;",
    "robust error handling",
    "# Deferred errors",
]
```

Run `python3 scripts/test-doc-consistency.py`; expect failure before the docs change.

- [ ] **Step 2: Correct the crate-root and method examples**

The runtime crate-root example uses:

```rust
let y = (&x + &x)?;
# Ok::<(), tenferro_runtime::Error>(())
```

The `TracedTensor::add` docs show both canonical and operator forms plus composition:

```rust
let method = a.add(&b)?.add(&c)?;
let ab = (&a + &b)?;
let abc = (&ab + &c)?;
assert_eq!(method.rank, abc.rank);
```

State verbatim: “Tenferro prioritizes robust error handling over the conciseness of chained operator notation.” Explain that `a + b + c` parses as `(a + b) + c`, while the first expression is a `Result`. Mention that generic `T: Add<Output = T>` and fallible assignment operators are not supported for dynamic tensors.

- [ ] **Step 3: Document traced timing**

In `core-concepts.md` and `api-conventions.md`, explain:

- concrete invalid relationships fail during graph build;
- symbolic relationships record constraints and may fail during compile/run;
- both paths expose the same validation kind, with `ErrorPhase` identifying timing.

Update every affected reshape and extension example to handle `Result`; do not hide fallibility solely with unexplained `unwrap()` in user guides.

- [ ] **Step 4: Verify and commit docs**

Run:

```bash
python3 scripts/test-doc-consistency.py
python3 scripts/test-check-docs-site.py
cargo test -p tenferro-runtime --doc
```

Expected: pass.

```bash
git add -- \
  crates/tenferro-runtime/src/lib.rs \
  crates/tenferro-runtime/src/traced.rs \
  docs/getting-started/core-concepts.md \
  docs/spec/api-conventions.md \
  scripts/test-doc-consistency.py
git commit -m "docs: explain fallible tensor operators"
```

---

### Task 9: Enforce repository rules and Clippy documentation gates

**Files:**
- Modify: `REPOSITORY_RULES.md:60-135`
- Modify: `.github/workflows/ci.yml:30-43`
- Modify: `scripts/test-repository-rules-review.py`

**Interfaces:**
- Produces: mandatory `# Errors` / `# Panics` repository policy.
- Produces: deterministic Clippy gates for workspace, tropical, and sparse.
- Preserves: semantic review through the always-selected `Public Boundary Safety Audits` section.

- [ ] **Step 1: Add failing policy and workflow source tests**

Add to `scripts/test-repository-rules-review.py`:

```python
def test_public_boundary_rules_are_always_selected() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["crates/tenferro-runtime/src/traced.rs"])
    assert "Public Boundary Safety Audits" in sections
```

Extend `scripts/test-doc-consistency.py` to assert that `.github/workflows/ci.yml` contains both lint names and both standalone extension manifests. Run both scripts; expect the workflow assertion to fail.

- [ ] **Step 2: Add the normative repository rule**

Insert under `Public Boundary Safety Audits`:

```text
- Every public function, inherent method, and workspace-owned trait method
  returning `Result` must document concrete failure conditions under `# Errors`
  and name the public error variant or stable kind. Generic text such as
  “returns an error on failure” is insufficient. Public operator APIs whose
  effective `Output` is `Result` must document the same behavior on the owning
  type/operator surface. Traced APIs must additionally use `# Deferred errors`
  when symbolic validation may fail during compilation or execution.
- Every public API with an intentional panic contract must document the exact
  precondition under `# Panics`. Caller-controlled invalid input must remain a
  typed error.
- Crate-boundary conversion must preserve structured error kind, payload, and
  `source()` chain. Convert to text only for display/logging/FFI/serialization
  or when an external API supplied no structured source.
```

Because this section is in `ALWAYS_SECTIONS`, the existing LLM review automatically performs the requested semantic audit; do not add a hand-written Rust parser.

- [ ] **Step 3: Strengthen the CI lint commands**

Change the workspace command to:

```yaml
run: >-
  cargo clippy --workspace --all-targets --
  -D warnings
  -D clippy::missing_errors_doc
  -D clippy::missing_panics_doc
```

Use the same flags for `ext/tropical/Cargo.toml` and add a third command for `ext/sparse/Cargo.toml`.

- [ ] **Step 4: Clear the complete lint inventory**

Run the exact workspace command. The pre-change baseline stops first at 27 tensor-core errors; after earlier tasks it must continue through every crate. Fix every remaining public item with concrete documentation. Do not add crate-level `allow`, item-level `allow`, or a baseline file.

- [ ] **Step 5: Verify and commit governance**

Run:

```bash
python3 scripts/test-repository-rules-review.py
python3 scripts/test-doc-consistency.py
cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo clippy --manifest-path ext/sparse/Cargo.toml --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
```

Expected: pass.

```bash
git add -- \
  REPOSITORY_RULES.md \
  .github/workflows/ci.yml \
  scripts/test-repository-rules-review.py \
  scripts/test-doc-consistency.py
git commit -m "ci: enforce public error documentation"
```

---

### Task 10: Remove transitional surfaces and run the complete release gate

**Files:**
- Create: `docs/worklogs/2026-07-17-structured-error-model.md`

No source change is expected in this task. If a forbidden scan finds a match,
return to the task that owns that exact file, fix and verify it there, then
restart Task 10 from Step 1.

**Interfaces:**
- Removes: every legacy variant, alias, lossy conversion, and compatibility shim.
- Produces: a clean pre-1.0 public error model ready for review.

- [ ] **Step 1: Run final forbidden-pattern scans**

Run:

```bash
rg -n 'InvalidGraphBuild|InvalidCompiledGraph|ContractionError\(String\)|pub type Error = ValidationError|to_tensor_error\(&self' crates ext
rg -n 'backend_failure\([^\n]*\.to_string\(\)|map_err\([^\n]*\.to_string\(\)' crates ext
rg -n 'pub fn reshape\(&self, shape: &\[usize\]\) -> TracedTensor' crates
```

Expected: no matches. Review any broader `to_string()` results manually and retain only display/logging/cache-key/vendor-message boundaries.

- [ ] **Step 2: Write the migration worklog**

Record:

- old-to-new public error mapping;
- traced reshape signature change;
- fallible operator composition example;
- eager/traced detection-phase matrix;
- extension source-preservation behavior;
- verification commands and exact outcomes;
- explicit statement that no compatibility layer remains.

- [ ] **Step 3: Run formatting, tests, docs, and lints**

Run:

```bash
cargo fmt --all --check
cargo test --workspace --all-targets
cargo test --workspace --doc
cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo test --manifest-path ext/tropical/Cargo.toml --features autodiff
cargo test --manifest-path ext/sparse/Cargo.toml --features autodiff
cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo clippy --manifest-path ext/sparse/Cargo.toml --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
python3 scripts/test-repository-rules-review.py
python3 scripts/test-doc-consistency.py
python3 scripts/test-check-docs-site.py
```

Expected: every command passes. If a hardware-only feature cannot run locally, record the exact skipped command and leave it to its existing CI lane; do not claim it passed.

- [ ] **Step 4: Run the repository-rules deterministic review over the implementation diff**

Run:

```bash
impl_base=$(git merge-base HEAD origin/main)
python3 scripts/repository-rules-review.py \
  --base "$impl_base" \
  --head HEAD \
  --dry-run \
  --llm-skipped-reason "local deterministic preflight; CI performs semantic review"
```

Expected: deterministic verdict passes.

- [ ] **Step 5: Commit the final cleanup and worklog**

```bash
git add -- docs/worklogs/2026-07-17-structured-error-model.md
git commit -m "docs: record structured error migration"
```

---

## Plan self-review checklist

- Shared payload ownership is implemented before any consumer migration.
- Runtime phase never enters tensor-core payloads.
- Tensor, runtime, AD, and extension boundaries retain `source()`.
- Concrete traced reshape fails at graph build; symbolic reshape fails only when concrete facts arrive.
- Explicit methods remain canonical and operator notation remains fallible.
- `a + b + c`, generic `Add<Output = T>`, and `AddAssign` limitations are documented.
- Repository policy, Clippy presence checks, semantic review, and behavior tests all enforce the model.
- The final scan forbids compatibility aliases and legacy string variants.
