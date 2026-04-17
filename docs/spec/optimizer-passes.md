# Optimizing Compiler: Pass Design

**Date:** 2026-04-04
**Parent:** `../index.md`
**Related:** `primitive-catalog.md`, `backend-contract.md`, `../architecture/tenferro-crates.md`

---

## I. Purpose

This document specifies the optimization passes that run while lowering
computegraph `CompiledProgram` values into tenferro's execution IR.

The current compiler entry points are:

- `compile_std_to_exec()`
- `compile_semiring_to_exec()`

Both lower directly to `ExecProgram`. Passes run on `ExecProgram` in place;
there is no intermediate `StableHloProgram` / `StableHloOp` layer anymore.

Before any pass runs, lowering also computes:

- `ExecInstruction::dtype`
- `ExecInstruction::output_shapes`
- `ExecInstruction::last_use`

The new `output_shapes` field is the key enabler for future `DotDecomposer`
work: the compiler can now reason about reshapes and contraction outputs
without reconstructing shapes from scratch downstream.

---

## II. Current Status

### Active passes

| Pass | Status | Purpose |
|---|---|---|
| `DotDimensionSorter` | active | Sort contracting dimensions on `ExecOp::DotGeneral` to stabilize canonical order |
| `TransposeFolding` | active | Fold producer `ExecOp::Transpose` into downstream `ExecOp::DotGeneral` dimension numbers |

### Deferred pass

| Pass | Status | Notes |
|---|---|---|
| `DotDecomposer` | deferred | Shape tracking is now available on `ExecInstruction`, but the canonicalization pass itself is still tracked in `tensor4all/tenferro-rs#729` |

### Removed pass

| Pass | Status | Reason |
|---|---|---|
| `ReductionSimplification` | removed | The pass was deleted during the 1-layer IR refactor; it is not part of the current compiler pipeline |

### No-op category removed by IR cleanup

Structured linalg operations no longer travel through string `CustomCall`
targets, so there is no separate "linalg passthrough" pass in the current
pipeline. `Svd`, `Qr`, `Lu`, `Eigh`, `Eig`, `TriangularSolve`, and
`ValidateNonsingular` are first-class `ExecOp` variants.

---

## III. Pass Specifications

### III.1 DotDimensionSorter

**Scope:** `ExecProgram`

**Purpose:** Normalize `DotGeneral` contracting dimensions into a stable
sorted order when they are already consecutive up to permutation.

**Why it still matters:** `TransposeFolding` and future `DotDecomposer`
logic both behave more predictably when contracting dimensions already have a
canonical ordering.

**When to fire:**

- `ExecInstruction.op` is `ExecOp::DotGeneral(config)`
- one side's contracting dimensions are consecutive if sorted
- that side's contracting dimensions are currently unsorted

**Algorithm:**

```text
PROCEDURE SortDotDimensions(program):
  FOR each instruction IN program.instructions:
    IF instruction.op is not DotGeneral:
      CONTINUE

    lhs = instruction.op.lhs_contracting_dims
    rhs = instruction.op.rhs_contracting_dims

    IF lhs is consecutive_if_sorted AND lhs is not sorted:
      perm = argsort(lhs)
      lhs = apply_perm(lhs, perm)
      rhs = apply_perm(rhs, perm)
    ELSE IF rhs is consecutive_if_sorted AND rhs is not sorted:
      perm = argsort(rhs)
      lhs = apply_perm(lhs, perm)
      rhs = apply_perm(rhs, perm)
```

**Example:**

```text
Before:
  lhs_contracting = [3, 2]
  rhs_contracting = [2, 1]

After:
  lhs_contracting = [2, 3]
  rhs_contracting = [1, 2]
```

### III.2 TransposeFolding

**Scope:** `ExecProgram`

**Purpose:** Remove producer `Transpose` instructions that feed directly into
`DotGeneral` by rewriting the dot dimension numbers and bypassing the
transpose output slot.

**When to fire:**

- the consumer instruction is `ExecOp::DotGeneral`
- the selected operand is produced by `ExecOp::Transpose`
- the permutation is valid for the operand rank
- the foldability checks pass

**Current foldability rule:** batch dimensions must remain fixed, and the
operand must expose exactly one contracting dimension after the rewrite.

**Algorithm:**

```text
PROCEDURE FoldTransposeIntoDot(program):
  REPEAT until fixed point:
    changed = false

    FOR each DotGeneral instruction IN program.instructions:
      FOR operand_idx IN {0, 1}:
        producer = instruction producer for operand_idx
        IF producer is not Transpose:
          CONTINUE

        perm = producer.permutation
        IF transpose is foldable for this operand:
          rewrite DotGeneral dimension numbers with perm
          replace operand slot with the transpose input slot
          changed = true

    IF changed is false:
      BREAK
```

**Example:**

```text
Before:
  t0 = Transpose(A, [1, 0])
  t1 = DotGeneral(t0, B)

After:
  t1 = DotGeneral(A, B) with rewritten dimension numbers
```

The pass runs to a fixed point so later `DotGeneral` instructions can absorb
transposes exposed by earlier folds.

### III.3 DotDecomposer

**Scope:** planned for `ExecProgram`

**Status:** deferred to `tensor4all/tenferro-rs#729`

**Goal:** Canonicalize arbitrary `ExecOp::DotGeneral` contractions into a
shape more directly consumable by backend GEMM paths.

The 1-layer IR refactor intentionally landed the prerequisites first:

- direct `StdTensorOp -> ExecProgram` lowering
- per-instruction dtype inference
- per-instruction `output_shapes`

That metadata is now available to support decomposition, reshape insertion,
and downstream validation. The transformation itself is still out of scope for
the current branch and is not part of the live compiler pipeline.

### III.4 ReductionSimplification

`ReductionSimplification` no longer exists in the compiler. Historical notes
about hoisting independent reductions remain useful background, but there is no
implementation, no tests, and no pass-order slot for it in the current code.

---

## IV. Pass Order

The current lowering pipeline is:

```text
CompiledProgram<StdTensorOp or SemiringOp<_>>
    │
    │ lower directly to ExecProgram
    │ infer dtype + output_shapes per instruction
    ▼
ExecProgram
    │
    │ 1. DotDimensionSorter
    │ 2. TransposeFolding (fixed point)
    │ 3. populate_last_use
    ▼
Ready for execution
```

`DotDecomposer` is intentionally absent from this list until issue `#729`
lands. `ReductionSimplification` is intentionally absent because the pass was
deleted.

---

## V. Comparison to the Previous Execution Engine

| Previous behavior | Current status |
|---|---|
| Contracting-dimension normalization during execution planning | Moved into `DotDimensionSorter` on `ExecProgram` |
| Lazy transpose absorption around GEMM | Moved into `TransposeFolding` on `ExecProgram` |
| Multi-step `DotGeneral` canonicalization | Deferred to `DotDecomposer` in `#729` |
| Reduction hoisting before contraction | No dedicated pass today |
| Linalg passthrough via string `CustomCall` | Removed; linalg ops are structured `ExecOp` variants |

---

## VI. Testing Expectations

The pass test suite should focus on:

- direct `ExecProgram` fixtures
- exact rewrites of `ExecOp::DotGeneral` dimension metadata
- fixed-point behavior for `TransposeFolding`
- preservation of output slots and `output_shapes`

Regression tests live in `tenferro/tests/compiler_passes.rs`.
