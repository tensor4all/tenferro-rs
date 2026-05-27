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
| `DotDecomposer` | active | Canonicalize `ExecOp::DotGeneral` into `[M?, K, B...] × [K, N?, B...] → [M?, N?, B...]` by emitting explicit `Transpose`/`Reshape` wrappers |
| `DeadCodeElimination` | active | Drop instructions whose outputs are never consumed by a program output or later instruction |

### Removed pass

| Pass | Status | Reason |
|---|---|---|
| `ReductionSimplification` | removed | The pass was deleted during the 1-layer IR refactor; it is not part of the current compiler pipeline |

### No-op category removed by IR cleanup

Structured extension operations no longer travel through string `CustomCall`
targets, so there is no separate extension passthrough pass in the current
pipeline. Linalg, einsum, and FFT use `ExecOp::Extension` and are handled as
single-instruction extension boundaries.

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

**Scope:** `ExecProgram`

**Goal:** Canonicalize arbitrary `ExecOp::DotGeneral` contractions into a
shape directly consumable by canonical batched GEMM kernels. Runs after
`TransposeFolding`, so pre-existing foldable transposes are already absorbed
when decomposition kicks in.

**Canonical form** (tenferro column-major with batch trailing, per
`AGENTS.md`):

```text
LHS shape: [M?, K, B0, B1, ...]
  lhs_contracting_dims = [|free_L|']
  lhs_batch_dims       = [|free_L|' + 1, ..., |free_L|' + nb]
  |free_L|' = 1 if the original LHS had any free dim, else 0.

RHS shape: [K, N?, B0, B1, ...]
  rhs_contracting_dims = [0]
  rhs_batch_dims       = [|free_R|' + 1, ..., |free_R|' + nb]
  |free_R|' = 1 if the original RHS had any free dim, else 0.
```

Output shape (from `shape_infer::dot_general_shape`): `[M?, N?, B0, B1, ...]`
with 0-or-1 `M` dim, 0-or-1 `N` dim, and the batch sizes trailing.

**When to fire:** any `ExecOp::DotGeneral` whose `(config, lhs_rank, rhs_rank)`
is not already in canonical form.

**Algorithm** per non-canonical DotGeneral (inputs: LHS and RHS slots with
shapes in `slot_shapes`):

1. **LHS Transpose** to target order `free_L ++ contracting_L ++ batch_L`.
   Skip if already identity.
2. **LHS Reshape** to `[M?, K, B...]` by merging multiple free/contracting
   dims. Skip if `|free_L| ≤ 1` and `|contracting_L| == 1`.
3. **RHS Transpose** to target order `contracting_R ++ free_R ++ batch_R`.
   Skip if already identity.
4. **RHS Reshape** to `[K, N?, B...]`. Same skip rule as LHS.
5. **Canonical `DotGeneral`** with the dim-number positions described above.
6. **Output Reshape** — the XLA Step 5 that the pre-#729 skeleton was
   missing — restores the original output shape when either side had
   multiple free dims. It takes the canonical DotGeneral output and the
   original LHS and RHS slots as shape-providing inputs, so the dynamic
   free-dim sizes can be recovered at runtime.

Skipping the output Reshape (step 6) is what caused the historical
downstream-Permute bug: downstream consumers would observe the merged rank
instead of the expected unmerged rank, and subsequent `Transpose`/`Reshape`
users would fall out of bounds. With the Reshape in place, downstream ops
see the original output shape and no further rewriting is required.

**Example** (single non-canonical case):

```text
Before:
  DotGeneral lhs=[a, b, M, K1, K2] rhs=[a, b, K1, K2, N]
             lhs_batch=[0, 1] rhs_batch=[0, 1]
             lhs_cont=[3, 4]  rhs_cont=[2, 3]

After:
  Transpose(lhs, perm=[2, 3, 4, 0, 1])  -> [M, K1, K2, a, b]
  Reshape(...)                           -> [M, K1*K2, a, b]
  Transpose(rhs, perm=[2, 3, 4, 0, 1])  -> [K1, K2, N, a, b]
  Reshape(...)                           -> [K1*K2, N, a, b]
  DotGeneral(canonical)                  -> [M, N, a, b]
```

**Interaction with `TransposeFolding`:** `TransposeFolding` runs first and
may leave dead `Transpose` producers (its folded-away producer is bypassed
but not removed). The subsequent `DeadCodeElimination` reclaims that runtime
cost. For programs where the user already wrote a canonical `Transpose →
DotGeneral`, the fold+decomp combination is net-identity (the fold absorbs,
the decomp re-emits), so the compiled result is equivalent to the input up
to dead-code cleanup.

### III.4 DeadCodeElimination

**Scope:** `ExecProgram`

**Purpose:** Remove instructions whose outputs are not consumed by any
downstream instruction or program output.

**When to fire:** always. In practice, instructions die after
`DotDecomposer` re-canonicalizes a DotGeneral whose upstream Transpose was
already folded into dim-numbers by `TransposeFolding`.

### III.5 ReductionSimplification

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
    │ 3. DotDecomposer
    │ 4. DeadCodeElimination
    │ 5. populate_last_use
    ▼
Ready for execution
```

`DotDecomposer` runs after `TransposeFolding` so that any pre-existing
foldable transpose is absorbed into the DotGeneral dim numbers before the
decomposition step canonicalizes the result. `DeadCodeElimination` runs last
among the optimizations to remove bypassed instructions (typically
`Transpose` producers the fold left behind) before `populate_last_use`
records the final liveness map.

`ReductionSimplification` is intentionally absent because the pass was
deleted.

---

## V. Comparison to the Previous Execution Runtime

| Previous behavior | Current status |
|---|---|
| Contracting-dimension normalization during execution planning | Moved into `DotDimensionSorter` on `ExecProgram` |
| Lazy transpose absorption around GEMM | Moved into `TransposeFolding` on `ExecProgram` |
| Multi-step `DotGeneral` canonicalization | Implemented in `DotDecomposer` (#729) with the full XLA-style algorithm, including the output Reshape step |
| Reduction hoisting before contraction | No dedicated pass today |
| Linalg passthrough via string `CustomCall` | Removed; linalg lowers to `ExecOp::Extension` and dispatches through the registered linalg runtime |

---

## VI. Testing Expectations

The pass test suite should focus on:

- direct `ExecProgram` fixtures
- exact rewrites of `ExecOp::DotGeneral` dimension metadata
- fixed-point behavior for `TransposeFolding`
- preservation of output slots and `output_shapes`

Regression tests live with the runtime/compiler crate tests.
