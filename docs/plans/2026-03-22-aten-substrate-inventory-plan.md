# ATen-Aligned Low-Level Substrate Inventory Plan

**Goal:** Define a substrate-first plan for the next CUDA linalg tranche by tracing the low-level building blocks that PyTorch/ATen actually uses for `svd`, `svdvals`, `lu`, `qr`, `cholesky`, `solve`, `solve_triangular`, `det`, `slogdet`, `pinv`, `norm`, and `matrix_exp`, then mapping the reusable subset into tenferro's layering.

**Why this plan exists:** The previous linalg CUDA tranche proved that implementing composite ops directly is the wrong order once the missing piece is a reusable lower-layer primitive. The next phase should not start from individual high-level ops. It should start from an inventory of the reusable tensor/device/prims/linalg-prims substrate that those ops depend on.

**Key idea:** Do not merely union:

- "what PyTorch `svd`, `lu`, ... use"
- "what ATen low-level substrate exists"

Instead, compute the **transitive substrate closure** of a bounded target op set, then add a second tier of **adjacent high-value substrate** that is not strictly required by the first target set but is clearly reusable and ATen-proven. Then turn both tiers into a **tenferro gap matrix** and a **priority order**. That keeps the work aligned with PyTorch without committing tenferro to reimplementing all of ATen.

## Locked design decisions

- `tenferro-linalg` remains CPU/GPU generic and must not regain normal-path GPU->CPU payload fallback.
- `tenferro-device` remains Layer 0 runtime substrate.
- `tenferro-tensor` owns tensor-basic materialization / structural helpers on top of Layer 0.
- `tenferro-prims` owns scalar / analytic / semiring family execution.
- `tenferro-linalg-prims` owns cuBLAS / cuSOLVER contracts, working-copy preparation, and status plumbing.
- The inventory must be driven by real PyTorch/ATen call paths, not by speculative "maybe useful later" substrate.
- Useful adjacent substrate may be included, but it must be called out explicitly as a second tier rather than mixed into the must-have core.

## Better plan than a plain 1+2 union

The proposed two investigations are both necessary:

1. trace PyTorch linalg ops down to the low-level helpers they actually call
2. inventory the ATen low-level substrate that exists

But that alone is not sufficient, because ATen contains much more than tenferro needs right now. The improved plan adds a third required output:

3. build an **op -> substrate -> tenferro layer gap matrix**

This third step is what makes the investigation actionable. Without it, the result is just a long list of ATen utilities. With it, we can identify:

- the smallest reusable substrate set that unlocks the most blocked tenferro ops
- the adjacent substrate that is worth pulling in now because it is likely to be needed immediately after the first tranche

## Scope

The bounded target-op set for this investigation is the **Tier A core set**:

- `svd`
- `svdvals`
- `lu`
- `lu_factor_ex`
- `qr`
- `cholesky`
- `cholesky_ex`
- `solve`
- `solve_ex`
- `solve_triangular`
- `det`
- `slogdet`
- `pinv`
- `norm`
- `matrix_exp`

In addition to Tier A, the investigation should record a **Tier B adjacent-useful set**. These are substrate items that:

- are not strictly required by the Tier A closure
- are already present as reusable ATen substrate near the traced call paths
- are very likely to be needed by the next neighboring tranche
- still belong to the correct tenferro layer if implemented

Examples of Tier B candidates:

- `view_as_real`-style helpers
- `real` / `imag` views or materialization helpers
- `where` / mask-select / masked-fill style substrate
- alias-safe `copy_` / `clone` conventions
- additional layout helpers around transposed wide-matrix handling
- batched small-matrix special-case infrastructure

Rule:

- Tier A is the execution-driving must-have set
- Tier B is an explicitly separate backlog of high-value adjacent substrate

## Outputs

This plan should produce five concrete artifacts:

1. **PyTorch call-path notes**
   - for each target op, identify the concrete PyTorch/ATen source files and the low-level helpers used
2. **ATen substrate catalog**
   - group low-level helpers by function rather than file
   - examples: batched column-major copy, triangular cleanup, `abs`/`sum`/`max`, `real`/`imag`, alias-safe copy, `info` handling
3. **tenferro gap matrix**
   - for each substrate item, record:
     - analogous tenferro layer
     - existing implementation status
     - current blockers
     - whether the gap is same-dtype, cross-dtype, structural, or status-related
4. **Tier B adjacent-useful substrate list**
   - explicitly separated from the Tier A must-have closure
   - each item must include a short justification for why it is worth carrying now
5. **execution order**
   - an ordered substrate backlog, prioritized by how many blocked target ops each item unlocks

## Investigation questions

For each target op in PyTorch/ATen, answer:

- Which top-level entrypoint implements the op?
- Which helper layers does it traverse?
- Which reusable lower-level operations are required?
- Which outputs are same-dtype vs cross-dtype?
- Which status / `info` paths are used?
- Which helpers are layout-only, structural, scalar-family, analytic-family, or linalg-kernel specific?
- Which helpers are CPU/GPU shared in semantics but backend-specific in execution?

For each ATen low-level helper found, classify it into one of these buckets:

- **Layout/materialization**
  - example: batched column-major copy, contiguous materialization, alias-safe clone/copy
- **Structural tensor ops**
  - example: `tril`, `triu`, diagonal extraction, trailing zero-fill, packing/unpacking
- **Scalar same-dtype**
  - example: `mul`, `div`, `pow`, `sum`, `max`, `prod`
- **Cross-dtype scalar/analytic**
  - example: complex `abs -> real`, `real`, `imag`, real reductions over complex-derived tensors
- **Status / control**
  - example: `info` tensors, convergence checks, fallback decisions
- **Linalg working-copy / wrapper**
  - example: `cloneBatchedColumnMajor`, transposed wide-matrix handling, workspace sizing

For every helper that is not in the Tier A closure but still looks valuable, also answer:

- Is it adjacent to a traced Tier A path, or is it just generally interesting?
- Which likely next tranche would consume it?
- Would adding it now simplify layering, or merely broaden scope?

## Execution phases

### Phase 1: Target-op call-path tracing

Start from PyTorch implementations of the bounded target-op set and trace downward only as far as needed to identify reusable substrate.

For each op, record:

- top-level file/function
- immediate helper layer(s)
- low-level substrate dependencies
- notable GPU-specific divergences

Expected files include:

- `../pytorch/aten/src/ATen/native/BatchLinearAlgebra.cpp`
- `../pytorch/aten/src/ATen/native/LinearAlgebra.cpp`
- `../pytorch/aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.cpp`
- `../pytorch/aten/src/ATen/native/cuda/linalg/CUDASolver.cpp`
- `../pytorch/aten/src/ATen/native/LinearAlgebraUtils.h`
- `../pytorch/aten/src/ATen/native/cuda/TriangularOps.cu`
- any directly referenced helper headers / source files needed for substrate identification

Deliverable:

- one short note section per target op
- no implementation yet

### Phase 2: ATen substrate catalog

Build a de-duplicated catalog of the low-level substrate encountered in Phase 1.

Important rule:

- catalog substrate by **capability**, not by file name

Examples:

- batched column-major working copy
- structural triangular cleanup
- same-dtype pointwise/reduction family
- cross-dtype complex-to-real family
- `info`/status tensor handling
- conjugation/view resolution
- alias-safe out/in-place copy semantics

Deliverable:

- a grouped substrate catalog with ATen file references
- with each item marked `Tier A` or `Tier B`

### Phase 3: tenferro crosswalk

For every substrate item in the catalog, map it onto tenferro:

- target layer:
  - `tenferro-device`
  - `tenferro-tensor`
  - `tenferro-prims`
  - `tenferro-linalg-prims`
  - `tenferro-linalg`
- current status:
  - absent
  - partial
  - present but wrong layer
  - present but same-dtype only
  - present but CPU-only
- current blocker examples
- public/composite ops blocked by the gap

This phase should explicitly identify reusable substrate already present in tenferro so it is not redesigned unnecessarily.

Examples likely to appear:

- Layer 0 shared runtime in `tenferro-device`
- tensor-level structural helpers such as triangular cleanup / trailing zero-fill
- same-dtype CUDA scalar family in `tenferro-prims`
- missing cross-dtype complex->real substrate in `tenferro-prims`
- working-copy and `info` plumbing in `tenferro-linalg-prims`

Deliverable:

- a gap matrix table

### Phase 4: tenferro crosswalk

### Phase 5: Execution order

Convert the gap matrix into an ordered backlog.

Prioritize by:

1. how many target ops a substrate item unlocks
2. whether it belongs to the correct layer
3. whether it removes a current source of ad hoc cleanup
4. whether it is a prerequisite for other substrate items
5. for Tier B only: whether it is likely to be needed in the very next tranche

Expected early candidates:

- cross-dtype complex->real unary substrate
- cross-dtype real reduction over complex-derived tensors
- `real` / `imag` / `abs_real` style helpers
- linalg `info`-bearing contracts where still missing
- remaining structural tensor helpers needed for fixed-shape postprocessing

Deliverable:

- a numbered execution sequence with stop points
- clearly split into:
  - Tier A core execution order
  - Tier B adjacent-useful follow-on order

## Concrete deliverable format

The final plan note produced from this investigation should contain:

1. **Target Ops**
   - one row per target op
   - PyTorch entrypoint
   - reusable substrate dependencies
2. **Substrate Catalog**
   - grouped by capability bucket
3. **tenferro Crosswalk**
   - substrate item
   - tenferro layer
   - status
   - blocked ops
4. **Tier B Adjacent-Useful Substrate**
   - separated from the must-have core
   - each item justified
5. **Execution Order**
   - Tier A first
   - Tier B second

## Why this is better than only "survey PyTorch" + "survey ATen"

A plain two-list union is too broad. It risks importing ATen's entire internal toolbox into the plan.

The better approach is:

- start from a bounded op set
- trace the transitive substrate closure actually used
- separately record adjacent ATen-proven helpers that are probably next
- catalog the substrate
- crosswalk it against tenferro layers
- then prioritize by unlock value

That keeps the plan PyTorch-aligned, but still tenferro-specific and layering-correct.

## Immediate next step

Execute **Phase 1** first and stop after producing the per-op call-path notes. Do not start implementation from this note directly.
