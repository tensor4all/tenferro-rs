# Note: Batch-Axis Ordering and Materialization in Binary Contraction

This note compares `omeinsum-rs` and `strided-opteinsum` with focus on one
question:

When batch axes exist, does the implementation pay extra permutation/copy cost
after each binary contraction?

## Concrete Example: Batch-First Output Causes Repeated Reorder

Consider pairwise evaluation of:

`A[b,i,j], B[b,j,k], C[b,k,l] -> Y[b,i,l]`

If intermediate outputs follow user-facing order (`batch` first), the first
binary step naturally asks for:

`T[b,i,k] = contract(A, B)`

and second step asks for:

`Y[b,i,l] = contract(T, C)`.

### In `omeinsum-rs`

Inside each binary `contract`, the GEMM result order is fixed to:

`[left, right, batch]`

(`current_order = left + right + batch`, see
`../omeinsum-rs/src/backend/cpu/contract.rs`).

For step 1 above:

- requested output `modes_c = [b,i,k]`
- internal `current_order = [i,k,b]`
- mismatch (`current_order != modes_c`) triggers `permute_data` copy

For step 2:

- requested output `modes_c = [b,i,l]`
- internal `current_order = [i,l,b]`
- mismatch again, so another `permute_data` copy

So batch-first requested order can force repeated output-side materialization.

## Difference in Strategy

### 1. `omeinsum-rs` (eager materialization)

Per binary contraction step:

1. `ensure_contiguous(a/b)` on inputs
2. `permute_data` for A/B into GEMM canonical order
3. GEMM
4. `permute_data` again if output order differs

Relevant code paths:

- `../omeinsum-rs/src/backend/cpu/contract.rs`
- `../omeinsum-rs/src/einsum/engine.rs`

Result: axis-order mismatch is resolved immediately by data movement.

### 2. `strided-opteinsum` + `strided-einsum2` (lazy permute + conditional copy)

Key differences:

- Permutation-only rewrites are often metadata-only (`permuted(...)`), not
  immediate data copy.
- Binary node output IDs are canonicalized to `[lo, ro, batch]` by
  `compute_binary_output_ids`, which keeps internal tree evaluation in a stable
  batch-last order.
- Final output reorder can remain metadata-only when it is a pure permutation.

Relevant code paths:

- `../strided-rs/strided-opteinsum/src/expr.rs`
- `../strided-rs/strided-opteinsum/src/operand.rs`
- `../strided-rs/strided-einsum2/src/lib.rs`
- `../strided-rs/strided-einsum2/src/contiguous.rs`

Important caveat:

Lazy permutation does not mean "never copy". During binary contraction,
`prepare_input_owned`/`prepare_input_view` still materialize contiguous
buffers when stride fusability checks fail, when backend stride constraints
require it, or when conjugation must be materialized.

## Practical Takeaway

- `omeinsum-rs`: reorder mismatch is paid eagerly at each binary step.
- `strided-opteinsum`: permutation is deferred as metadata where possible; copy
  is pushed to the contraction boundary and performed only when required by the
  backend/stride condition.

For batch-heavy contractions, this difference can remove repeated
"output-order-only" copies in intermediate steps.
