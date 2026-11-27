# Sampled einsum v1 — the plan (#2675)

Written and committed **before any v1 code**, so that a lost session cannot lose the design.
Status of every claim is marked: **[read]** = checked in the tree at `57a6f31f`,
**[derived]** = follows from what was read but not yet run, **[open]** = decided only by a
measurement v1 has not made yet.

Corrections to the brief (it says `tensor4all-rs`; the issue, v0 `#2739` and `#2753` are in
`tenferro-rs`): the CPU job type is `cpueinsum::GroupedGemmJob` / `GroupedPlan` (cpueinsum 0.13.1,
already a dependency of `tenferro-einsum`); `t4a-desc` is not in this dependency graph **[read]**.

## 1. Where the contiguity comes from

v0's pair step computes `out[.., p, ..] = Σ left[.., lr[p], ..] · right[.., rr[p], ..]` and gathers
rows `lr[p]`, `rr[p]` of two column-major slots into per-chunk copies **[read: `exec.rs`
`PairChunks::hold`]**. In column-major storage, row `r` of a slot whose configuration axis is
the **last** axis is the contiguous block `[r·blk, (r+1)·blk)`. That is the whole idea, and the
planner decision that gives it is: **the configuration axis is placed last in every slot a pair
step reads or writes**, and the remaining modes of a block are ordered so the block is a compact
column-major matrix in the role the consumer needs:

- left operand block: `[M…, K…, B…]`, right: `[K…, N…, B…]`, output: `[M…, N…, B…]`
  (`M` = labels only on the left, `N` only on the right, `K` contracted, `B` labels shared by both
  and kept — batch). Each pair is then `|B|` GEMMs of one geometry `(m,k,n)` at plain offsets.
- `cpueinsum::GroupedGemmJob` takes **compact column-major jobs with no transpose flags**
  **[read]**, so this order is a requirement, not a preference.
- **Producers we control** (an `Einsum` instruction's output order is the planner's choice) get the
  consumer's order for free. **A pair step's output is fixed** by the no-transpose GEMM to
  `[M, N, B, P]` **[derived]**.

**When it cannot be had.** A pair step's consumer needs its operand as `[M',K',B',P]`; that is
satisfiable from `[M,N,B,P]` iff the producer's labels, in order, are already grouped
`M'* K'* B'*` by the consumer's roles. This is a plan-time check. If it fails for a step, **that
step falls back to the v0 gathered path** (per step, not per network) and the report records
which steps fell back and why. Not "refuse": a network where one step in 50 cannot be contiguous
should still run with 49 gather-free steps. Inserting a permuting einsum on the operand (one
copy of the slot, not a gather per chunk) is the alternative; **[open]** whether to add it
depends on how many `n30m14` steps fail the check — v1 measures that count first (plan-only, no
execution) and reports it before choosing.

## 2. What the CPU path hands to `GroupedPlan`

- **Batch shape:** one `GroupedGemmJob` per `(pair p, batch index b)`:
  `lhs_offset = lr[p]·blk_l + b·m·k`, `rhs_offset = rr[p]·blk_r + b·k·n`,
  `out_offset = p·blk_o + b·m·n`, dims `(m, k, n)` **[derived]**. The lhs/rhs buffers are the two
  slots' own storage (no copy); the output buffer is the step result.
- **Equal-sized blocks.** Within one step every job has the same `(m,k,n)`, because geometry is a
  property of the labels, not of the pair — so `GroupedPlan` holds a single child plan.
  Raggedness only exists *across* steps (each step is its own `GroupedPlan`) and is expressed
  by the step boundary, not by variable jobs inside one call. The chunk-size distinction of v0
  (`full` vs `tail` contraction) disappears, because nothing is held per chunk.
- **Accumulation:** none across jobs. Each pair owns a distinct output block, and `K` is summed
  inside the GEMM. The jobs therefore cover the output exactly once, in order from 0, which is
  exactly `GroupedPlan::execute_uninit`'s fresh-output contract (`beta = 0`, `FreshCoverage`)
  **[read]** — so v1 also drops v0's `fill_scalar` zero-fill of the step result.
- **`run_chunks`.** The chunk loop stays as the memory/job-list bound but its `ChunkSource::hold`
  returns nothing (no gathered copies). If `GroupedPlan` needs the whole job list at once (it
  sorts and checks overlap), the chunk becomes a range of jobs; the interface change, if any,
  is made in `crate::chunked` and justified in the PR, with `sliced_einsum` / `sliced_memory`
  unchanged **[open until written]**.
- **Plan reuse:** `GroupedPlan` is built once per step at plan time, not per run.

## 3. What the CUDA path passes to `cublasGemmBatchedEx`

- `cublasGemmBatchedEx` exists in the vendored cudarc binding **[read]**; `vendor.rs` currently
  has only dense and **strided**-batched GEMM **[read]**, so this is new code in
  `tenferro-gpu`. The rank-8 view ABI is **not** touched.
- **Pointer array:** host `Vec<u64>` of `base + offset·sizeof` for A, B and C (three arrays, one
  entry per job), uploaded to a device buffer from the runtime's pool.
- **Lifetime:** the device pointer arrays are registered in the **same raw scope** as the three
  operand buffers (`with_raw_locked` in `vendor.rs`), which keeps them alive until the scope's
  stream completion **[read: that is what the strided call does for A, B, C]**. They are not freed
  at return of the call.
- **Group size:** `cublasGemmBatchedEx` takes one `(m,n,k,lda,ldb,ldc)` for the whole call, so one
  call = one step's jobs (uniform by §2). `batchCount ≤ i32::MAX`: a longer job list is split into
  consecutive calls. **Different sizes in one call are not expressible**; they are different calls.
  `cublasGemmGroupedBatchedEx` (CUDA ≥ 12.5) would allow it and is **not** used.
- **Verification limit, stated now:** this machine has **no GPU** (`nvidia-smi` absent). The CUDA
  path will compile and be unit-tested on host-side pointer-array construction; **it will not
  have run on a GPU when the PR opens** unless the RunPod gate runs it. New CUDA tests need
  `scripts/ci/cuda_test_partition.tsv` rows.

## 4. What remains from v0

v0 is unchanged and stays the default; v1 is an additional step lowering chosen at plan time.
Still using the gather (to be **confirmed by the gather counter v1 adds first**, not asserted
here **[open]**):
- the leaf `IndexSelect` that restricts a leaf by the samples (once per leaf, not per chunk; leaves
  are small);
- one-sided / full-product-then-select steps;
- any pair step that fails the §1 contiguity check (the fallback);
- **the eager (autodiff) route** — v1 is concrete-only, because the eager layer differentiates
  through `index_select`'s existing scatter-add rule and v1 adds no new AD rule.

Not a regression because v0's code path and its tests are untouched, and the per-step choice
only moves a step *off* the gather when the contiguity check passes.

## Acceptance instruments (added before the lowering)

- `ExecStats` gains `gathered_rows` (rows passed through `index_select` in pair steps). v0 must
  report > 0 on a pair workload, v1 must report **0 on the steps it lowers** — the test that the
  gather is gone, distinct from "faster".
- v1 vs v0 numerical agreement on the same plan/samples, plus the existing dense reference.

## Benchmark protocol and its limits

Workload: `data/n30m14` + `examples/circuit_amplitudes.rs` of `Fanerst/sampled-einsum`
(reference amplitudes and artensor via `scripts/export_circuit.py`). Report planning and execution
time separately, GEMM count and `(m,k,n,batch)` distribution, peak memory, wall time next to v0
and artensor, anchored to the `#2706` numbers.

Limits known now: that repository and data are **not on this machine**, and this machine has
30 GiB RAM against the prototype's **48 GiB live / 83 GiB RSS** at complex128. The full run
needs the large evo host; if it cannot be run there the PR says so and reports what *was*
measured (plan-only step/fallback counts, a reduced circuit), not an extrapolation.
A negative result (planning dominates, or most steps fall back) will be reported as such.
