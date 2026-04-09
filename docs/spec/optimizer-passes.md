# Optimizing Compiler: Pass Design

**Date:** 2026-04-04
**Parent:** `../index.md`
**Related:** `primitive-catalog.md`, `backend-contract.md`, `../architecture/tenferro-crates.md`

---

## I. Purpose

This document specifies the optimization passes in tenferro's optimizing
compiler. The compiler transforms StableHLO IR into Execution IR for non-XLA
backends.

The pass design is based on two sources:

1. **XLA's HLO optimization pipeline** (`xla/service/gpu/gpu_compiler.cc`)
2. **the previous einsum execution optimizations** (`tenferro-einsum/src/`)

The goal is to adopt the minimum set of XLA-style passes that covers the
performance characteristics of the previous hand-tuned execution engine.

---

## II. Previous Optimizations and Required Passes

### Mapping table

| Previous optimization | Previous location | What it does | Pass needed |
|----------------|-------------|--------------|---------------|
| Lazy permutation | `dispatch.rs:446-454` | Return non-contiguous view instead of physical copy after GEMM | TransposeFolding |
| Fusability check | `layout.rs:1-33` | Check if dim groups can collapse into one GEMM dimension without copy | DotDecomposer |
| Partial materialization | `prepare.rs:97-140` | Copy only unfusable dimension group, not both | DotDecomposer + engine |
| Direct output writing | `dispatch.rs:331-401` | GEMM writes directly to output buffer via fused strides | DotDimensionSorter |
| Lazy output propagation | `execute.rs:205-228` | Intermediates in N-ary chain stay non-contiguous | TransposeFolding (cascaded) |
| Pre-reduction | `dispatch.rs:121-139` | Reduce unique-only axes before GEMM to shrink tensor | ReductionSimplification |
| Diagonal extraction | `dispatch.rs:105-119` | Zero-copy diagonal view | Not needed (diagonal lowered to Gather/ReduceSum at graph level) |
| Buffer pooling | `execute.rs:232-247` | Reuse buffers via Arc refcount + pool | Execution engine (liveness + pool) |
| Backend fast-path | `dispatch.rs:195-218` | Delegate to cuTENSOR etc. | SemiringFastPath trait |

### Minimum passes

1. **DotDimensionSorter** — sort contracting dims to minimize transposes
2. **TransposeFolding** — absorb Transpose into DotGeneral dimension_numbers
3. **DotDecomposer** — canonicalize DotGeneral to `[batch, M, K] × [batch, K, N]`
4. **ReductionSimplification** — hoist independent reductions before contractions
5. **LinalgPassthrough** — pass CustomCall (linalg) through unchanged

---

## III. Pass Specifications

### III.1 DotDimensionSorter

**Source:** XLA `xla/backends/gpu/transforms/dot_dimension_sorter.cc`

**Purpose:** Sort contracting dimensions so that DotDecomposer can canonicalize
without inserting unnecessary transposes.

**When to fire:** contracting dimensions are consecutive (no gaps) but not
sorted.

**Algorithm:**

```
PROCEDURE SortDotDimensions(dot):
  lhs_con = dot.dimension_numbers.lhs_contracting_dims
  rhs_con = dot.dimension_numbers.rhs_contracting_dims

  IF consecutive_if_sorted(lhs_con) AND NOT sorted(lhs_con):
    sort_key = lhs_con
  ELIF consecutive_if_sorted(rhs_con) AND NOT sorted(rhs_con):
    sort_key = rhs_con
  ELSE:
    RETURN  // nothing to do

  perm = argsort(sort_key)
  new_lhs_con = apply_perm(lhs_con, perm)
  new_rhs_con = apply_perm(rhs_con, perm)

  REPLACE dot with dot(lhs, rhs, new_dimension_numbers)
```

**Helper:**

```
FUNCTION consecutive_if_sorted(dims):
  RETURN max(dims) - min(dims) == len(dims) - 1
```

**Example:**

```
Before: dot(A, B), lhs_contracting={3,2}, rhs_contracting={2,1}
After:  dot(A, B), lhs_contracting={2,3}, rhs_contracting={1,2}
```

Now DotDecomposer can canonicalize without inserting a Transpose for the
contracting group, because {2,3} is already in increasing order.

---

### III.2 TransposeFolding

**Source:** XLA `xla/service/transpose_folding.cc`

**Purpose:** Eliminate Transpose instructions that feed directly into
DotGeneral by absorbing the permutation into dimension_numbers.

**When to fire:** a DotGeneral input is a Transpose instruction, and the
permutation is foldable (batch dimensions are unchanged by the transpose).

**Algorithm:**

```
PROCEDURE FoldTransposeIntoDot(dot):
  FOR operand_idx IN {0 (lhs), 1 (rhs)}:
    IF dot.operand(operand_idx) is NOT Transpose:
      CONTINUE

    perm = dot.operand(operand_idx).permutation
    original_input = dot.operand(operand_idx).operand(0)

    IF NOT is_foldable(dot, operand_idx, perm):
      CONTINUE

    // Apply inverse permutation to dimension_numbers
    IF operand_idx == 0:
      new_lhs_contracting = [perm[d] for d in dot.lhs_contracting]
      new_lhs_batch       = [perm[d] for d in dot.lhs_batch]
    ELSE:
      new_rhs_contracting = [perm[d] for d in dot.rhs_contracting]
      new_rhs_batch       = [perm[d] for d in dot.rhs_batch]

    // Unwrap: replace transpose(x) with x
    dot.set_operand(operand_idx, original_input)
    dot.set_dimension_numbers(new_dimension_numbers)

  RETURN modified dot (transposes removed)
```

**Foldability check:**

```
FUNCTION is_foldable(dot, operand_idx, perm):
  batch_dims = dot.lhs_batch if operand_idx == 0 else dot.rhs_batch
  contracting_dims = dot.lhs_contracting if operand_idx == 0
                     else dot.rhs_contracting

  // Batch dims must be unchanged by the permutation
  FOR d IN batch_dims:
    IF perm[d] != d:
      RETURN false

  // Must have exactly 1 contracting dimension
  IF len(contracting_dims) != 1:
    RETURN false

  RETURN true
```

**Example:**

```
Before: dot(transpose(A, {1,0}), B)
        lhs_contracting={1}, rhs_contracting={0}

After:  dot(A, B)
        lhs_contracting={0}, rhs_contracting={0}
        // perm={1,0} applied: dim 1 → perm[1] = 0
```

**Why this covers the previous lazy permutation:** Previously, the einsum engine defers
permutations as stride rewrites and only materializes at GEMM time. Now,
TransposeFolding achieves the same effect at the IR level — the Transpose
instruction is eliminated and the GEMM (DotGeneral) directly reads the
original layout through adjusted dimension_numbers.

---

### III.3 DotDecomposer

**Source:** XLA `xla/hlo/transforms/expanders/dot_decomposer.cc`

**Purpose:** Canonicalize DotGeneral with arbitrary dimension_numbers into a
form that maps directly to BatchedGemm:

```
Canonical form:
  LHS: [batch..., M, K]
  RHS: [batch..., K, N]
  Out: [batch..., M, N]

  lhs_batch = [0, 1, ..., nb-1]
  rhs_batch = [0, 1, ..., nb-1]
  lhs_contracting = [nb + 1]    (K is the last dim of LHS)
  rhs_contracting = [nb]        (K is the first non-batch dim of RHS)
```

**When to fire:** DotGeneral is not already in canonical form. Specifically:
- batch dims are not leading, OR
- there are multiple contracting dims, OR
- contracting dim is not in canonical position.

**Canonicality check:**

```
FUNCTION is_canonical(operand_shape, batch_dims, contracting_dims):
  RETURN len(contracting_dims) == 1
     AND batch_dims == [0, 1, ..., len(batch_dims)-1]
     AND operand_rank <= len(batch_dims) + 2
```

**Algorithm:**

```
PROCEDURE CanonicalizeDot(dot):
  FOR each operand (lhs, rhs):
    batch_dims = operand's batch dims
    contracting_dims = operand's contracting dims
    non_contracting_dims = all other dims

    // Step 1: Build target axis order
    IF operand is LHS:
      target_order = batch_dims + non_contracting_dims + contracting_dims
    ELSE (RHS):
      target_order = batch_dims + contracting_dims + non_contracting_dims

    // Step 2: Insert Transpose if order differs
    IF target_order != [0, 1, ..., rank-1]:
      operand = Transpose(operand, target_order)

    // Step 3: Reshape to fuse dimension groups
    batch_size = product of batch dim sizes
    IF operand is LHS:
      M = product of non_contracting dim sizes  (may be 1)
      K = product of contracting dim sizes
      new_shape = [batch_sizes..., M, K]
    ELSE:
      K = product of contracting dim sizes
      N = product of non_contracting dim sizes  (may be 1)
      new_shape = [batch_sizes..., K, N]

    operand = Reshape(operand, new_shape)

  // Step 4: Create canonical DotGeneral
  canonical_dot = DotGeneral(reshaped_lhs, reshaped_rhs, canonical_dim_numbers)

  // Step 5: Reshape output back to original shape
  output = Reshape(canonical_dot, original_output_shape)

  REPLACE original dot with output
```

**Example (multi-contracting-dim):**

```
Before: dot(A[512,32,32], B[1024,512])
        lhs_contracting={0}, rhs_contracting={1}

Step 1 (LHS): target = [1,2,0] (non-contracting=[1,2], contracting=[0])
              → Transpose(A, {1,2,0}) → [32,32,512]
Step 2 (LHS): Reshape → [1024, 512]  (M=32*32=1024, K=512)
Step 1 (RHS): target = [1,0] (contracting=[1], non-contracting=[0])
              → Transpose(B, {1,0}) → [512, 1024]
Step 2 (RHS): already [K=512, N=1024]
Step 3: canonical dot([1024,512], [512,1024]) → [1024, 1024]
Step 4: Reshape → [32, 32, 1024]
```

**Why this covers the previous fusability check:** Previously, `try_fuse_group_in_target_order`
checks if dimension groups can collapse into single GEMM dimensions. DotDecomposer
does the same via Reshape — multiple non-contracting dims are fused into M, multiple
contracting dims are fused into K. If the underlying strides happen to be
contiguous, Reshape is a no-op in the Execution IR.

**Why this covers the previous partial materialization:** Previously,
`prepare_one_operand` copies only the unfusable dimension group. In the current design,
DotDecomposer inserts a Transpose only for the axis group that needs
reordering. TransposeFolding may absorb it; otherwise the Execution IR
emits a Permute (physical copy) only for that operand.

---

### III.4 ReductionSimplification

**Source:** XLA `AlgebraicSimplifier` (reduction hoisting patterns)

**Purpose:** When the einsum builder emits `ReduceSum + DotGeneral` as
separate nodes (because an axis appears in only one operand and not in
the output), this pass ensures the ReduceSum runs before the DotGeneral,
shrinking the tensor before contraction.

Note: `einsum("ijk,jl->il")` cannot be a single `DotGeneral` — k is in
LHS only and not in the output. The einsum builder decomposes this into
`ReduceSum(A, axis=k)` followed by `DotGeneral`. This pass is about
ensuring that decomposition is optimal (reduction before contraction, not
after).

**When to fire:** a `ReduceSum` feeds into a `DotGeneral`, and the reduced
axes are independent of the contraction (not batch dims, not contracting
dims).

**Algorithm:**

```
PROCEDURE HoistIndependentReductions(program):
  FOR each ReduceSum → DotGeneral sequence:
    reduced_axes = axes being reduced
    dot_used_dims = batch_dims ∪ contracting_dims

    IF reduced_axes ∩ dot_used_dims is empty:
      // Reduction is independent of contraction — can run first
      // (einsum builder already emits this order, but this pass
      //  verifies and enforces it after other transforms)
      ENSURE ReduceSum precedes DotGeneral in topological order
```

**Example:**

```
einsum("ijk,jl->il", A[2,3,4], B[3,5])

Einsum builder emits:
  t0 = ReduceSum(A, axes=[2])          → t0[2,3]  (reduce k)
  t1 = DotGeneral(t0[2,3], B[3,5])     → t1[2,5]  (contract j)

ReductionSimplification verifies this order is preserved after
other transforms (e.g., if a prior pass reordered instructions).
```

**Why this covers the previous pre-reduction:** Previously, `execute_reduce_with_plan` was called
for unique-only axes before GEMM (`dispatch.rs:121-139`). This pass does the
same at the IR level.

---

### III.5 LinalgPassthrough

**Purpose:** Pass CustomCall instructions (linalg ops like SVD, QR, etc.)
through the compiler unchanged. They are not involved in transpose optimization
and are dispatched to the kernel registry at execution time.

**Algorithm:** Identity — no transformation.

---

## IV. Pass Execution Order

```
StableHLO IR (input)
    │
    │  1. DotDimensionSorter
    │     Sort contracting dims → reduces transposes from DotDecomposer
    │
    │  2. ReductionSimplification
    │     Hoist independent reductions before DotGeneral
    │
    │  3. DotDecomposer
    │     Canonicalize DotGeneral → [batch, M, K] × [batch, K, N]
    │     May insert Transpose + Reshape instructions
    │
    │  4. TransposeFolding
    │     Absorb remaining Transpose into DotGeneral dimension_numbers
    │     May run multiple iterations until fixed point
    │
    │  5. LinalgPassthrough
    │     CustomCall instructions pass through unchanged
    │
    │  6. Lower to Execution IR
    │     DotGeneral (canonical) → BatchedGemm
    │     All other ops → pass through unchanged (same op, same parameters)
    │     (Transpose → Permute is a rename; everything else is identity)
    ↓
Execution IR (output, stride-aware engine dispatch)
```

**Why this order matters:**

- **DotDimensionSorter before DotDecomposer:** sorting contracting dims means
  DotDecomposer can often avoid inserting Transpose for the contracting group.
- **ReductionSimplification before DotDecomposer:** reduces tensor rank/size
  before canonicalization, leading to simpler canonical form.
- **DotDecomposer before TransposeFolding:** DotDecomposer may insert new
  Transpose instructions that TransposeFolding can then absorb.
- **TransposeFolding last (among structural passes):** absorbs all accumulated
  Transpose instructions, including those inserted by DotDecomposer.

---

## V. Comparison: Previous Execution vs Compiled

| Step | Previous (runtime, per-contraction) | Current (compile-time, on IR) |
|------|-------------------------------|--------------------------|
| Axis classification | `classify.rs:12-54` at plan time | DotDecomposer identifies batch/contracting/non-contracting |
| Dim sorting | implicit (modes already ordered) | DotDimensionSorter |
| Lazy permute | `tensor.permute()` returns strided view | TransposeFolding eliminates Transpose from IR |
| Fusability check | `try_fuse_group_in_target_order` | DotDecomposer's Reshape (fuses dim groups) |
| Partial materialization | `prepare_one_operand` copies unfusable side | DotDecomposer inserts Transpose only where needed |
| Pre-reduction | `execute_reduce_with_plan` before GEMM | ReductionSimplification hoists ReduceSum |
| Direct output write | c_direct path checks output fusability | TransposeFolding on output Transpose |
| Buffer reuse | `TensorBufferPool` + `try_into_data_vec` | Liveness analysis + buffer pool (execution engine) |

**Key difference:** The previous design made these decisions at runtime per contraction step.
The current design makes them at compile time on the full IR graph, which enables cross-step
optimization (e.g., a Transpose inserted by one DotDecomposer step may be
absorbed by a subsequent DotGeneral via TransposeFolding).

---

## VI. What We Do NOT Adopt from XLA

| XLA pass | Reason for exclusion |
|----------|---------------------|
| DotMerger | Merges dots sharing an operand. Useful for XLA's JIT but not needed for step-by-step interpreter. |
| TransposeFolding for Convolution | No convolution support in initial scope. |
| AlgebraicSimplifier (full) | Most patterns are XLA-specific. We adopt only ReductionSimplification. |
| LayoutAssignment | XLA-specific. Engine-produced data is column-major; inputs are stride-aware. |
| Kernel fusion | No kernel fusion in the step-by-step interpreter. |

These can be added later if needed (e.g., DotMerger when implementing
checkpoint scheduling or operator fusion).

---

## VII. Implementation Notes

### Iteration

TransposeFolding should run in a loop until no more folds are possible (fixed
point). In practice 2-3 iterations suffice for einsum-derived graphs.

### Correctness testing

Each pass should preserve program semantics. Test strategy:
- Generate random StableHLO programs from einsum patterns
- Run with and without each pass
- Compare output numerically (or exactly for integer types)

### Profiling

The previous profiling counters (`PREPARE_ZEROCOPY`, `PREPARE_FALLBACK`,
`PREPARE_FALLBACK_ELEMS`, `GEMM_NS`, `PERMUTE_NS`) should have equivalents:
- Count of Transpose instructions before/after TransposeFolding
- Count of Permute instructions in final Execution IR
- Total elements physically copied by Permute instructions
