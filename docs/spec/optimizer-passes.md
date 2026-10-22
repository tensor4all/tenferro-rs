# Optimizing Compiler: Pass Design

**Date:** 2026-05-28
**Parent:** `../index.md`
**Related:** `primitive-catalog.md`, `backend-contract.md`, `../architecture/tenferro-crates.md`

---

## I. Purpose

This document specifies the optimization passes that run while lowering
computegraph `CompiledProgram` values into tenferro's execution IR.

This is the runtime compiler optimizer, not the AD graph optimizer. Traced AD
also runs a smaller materialize-pre optimizer on graphs produced by
`linearize`/`linear_transpose`; see `docs/architecture/ad-pipeline.md` for that
pipeline. The AD optimizer owns backend-independent derivative graph cleanup
such as transform-graph DCE and local algebraic identities. The runtime
compiler still owns execution-IR layout, dot, backend, and last-use passes.

The current compiler entry point is:

- `compile_std_to_exec()`

It lowers directly to `ExecProgram`. Passes run on `ExecProgram` in place;
there is no intermediate `StableHloProgram` / `StableHloOp` layer.

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
| `MetadataViewFolding` | active (at prepare) | Turn a single-consumer `Transpose`/`Reshape`/`BroadcastInDim`/`Slice` into an input view of the consumer that reads it, so it is not a scheduled command |

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

### III.5 MetadataViewFolding (#2226)

**Scope:** `ExecProgram`, at preparation (`PreparedProgramRoot::new`), after
engine placement is known.

**Purpose:** A metadata-only operation moves no bytes, yet as a scheduled
command it costs a full dispatch and, on an owned intermediate, a full copy.
When its result has exactly one consumer that reads the operand as a
`TensorRead`, the transform becomes an entry of that consumer's
`ExecInstruction::input_views` and the instruction is removed. The consumer
derives the view when it reads the operand (`InputView::apply`, the same
derivation the Read-slot path uses for a metadata command over a borrowed
input).

**When to fire:** an instruction folds when all of the following hold:

- `InputView::from_op` accepts its op (`Transpose`, `Reshape`,
  `BroadcastInDim`, `Slice`), with one input and one output;
- its dtype is not `DType::External` (a caller-owned payload has no borrowed
  view);
- its output is not a program output (an owned output must be materialized);
- exactly one instruction uses the output, at exactly one input position;
- that consumer reads the input through a view (`reads_input_view`: every
  backend dispatch entry except `Convert`, `DotGeneral`, extensions, and
  `ShapeOf`);
- producer and consumer share an execution location (a value that crosses a
  transfer boundary keeps materializing, so the transfer carries the result
  rather than the source), and that location's engine reports
  `LayoutRuntime::folds_metadata_views` (its `*_read` entries consume a
  strided view in place; CPU does, the GPU engines keep the default `false`
  and an unchanged schedule);
- for `Reshape`: the source is the output of an instruction that is not itself
  a view, and the reshape carries no pending view of its own (a reshape of a
  strided view has no view form in general; a non-view instruction result is a
  compact engine result).

**Algorithm:**

```text
PROCEDURE FoldMetadataViews(program, same_placement):
  REPEAT until no instruction folds:
    FOR each instruction i IN program.instructions:
      IF i is not a foldable view (rules above): CONTINUE
      (j, k) = the single use of i.output
      IF NOT reads_input_view(j.op, k) OR NOT same_placement(i, j): CONTINUE
      j.input_slots[k] = i.input_slots[0]
      j.input_views += i.input_views (re-indexed to k), then (k, view of i)
      remove i
      RESTART the scan
  populate_last_use(program); validate(program)
```

Chains fold in turn (`transpose → slice → consumer` leaves the consumer with
two views on one input, applied in order). One forward pass suffices, since a
folded instruction's consumer is visited later in the same pass; the pass is
linear in the instruction count plus the folded views.

**Interaction with region planning:** a folded `BroadcastInDim` on a region
input is the same broadcast view the elementwise planner derives from a
broadcast instruction and keeps the region fusable; any other folded view keeps
the region on the per-instruction path, which reads views itself. The folded
`broadcast-multiply` form keeps the segmentation boundary of the unfolded
triplet/pair.

**Example:**

```text
Before (scheduled commands: 3):
  t0 = Negate(A)
  t1 = Transpose(t0, [1, 0])
  t2 = Add(B, t1)

After (scheduled commands: 2):
  t0 = Negate(A)
  t2 = Add(B, t0 viewed as Transpose([1, 0]))
```

### III.6 ReductionSimplification

`ReductionSimplification` no longer exists in the compiler. Historical notes
about hoisting independent reductions remain useful background, but there is no
implementation, no tests, and no pass-order slot for it in the current code.

---

## IV. Pass Order

The current lowering pipeline is:

```text
CompiledProgram<StdTensorOp>
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
Staged program (identity-level, placement-independent)
    │
    │ at Runtime::prepare, once the engine placement is resolved:
    │ 6. MetadataViewFolding (same-location pairs), then populate_last_use
    ▼
Ready for execution
```

`MetadataViewFolding` runs at preparation rather than in the staging pipeline
because its placement rule needs the resolved operation locations; the staged
program the preparation cache identifies stays placement-independent, and each
prepared root executes its own folded copy.

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
| Metadata-only commands (`Transpose`, `BroadcastInDim`, ...) scheduled and materialized per run | Folded into the consumer's input view at preparation (`MetadataViewFolding`, #2226) |
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
