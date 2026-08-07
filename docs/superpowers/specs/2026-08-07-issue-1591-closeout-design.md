# Issue #1591 Closeout Design

**Date:** 2026-08-07

## Context

Tenferro issue #1591 asks to absorb the remaining `strided-einsum2`
dot-general and batched-GEMM machinery into `tenferro-cpu`, remove the external
dependency, carry forward identified correctness and safety fixes, and unblock
retirement of the upstream crate.

That description no longer matches the current repository state. An earlier
local implementation/adaptation, commit `eb689172666004ca70618757c62188181635429f`,
moved the dot-general-specific Faer preparation algorithm into `tenferro-cpu`
and removed the old `gemm/strided_dot.rs` adapter; its historical record is
`docs/worklogs/2026-06-23-strided-einsum2-removal.md`. Later, PR #1553, commit
`6255590e76d21f3ec7ba2a7feaa7e160baecabc1`, removed the stale dependency
declarations and feature wiring after that local provider path already existed.
These are distinct steps: #1553 removed stale build-graph references; it did
not create the earlier local adaptation. At the design baseline, `origin/main`
is `166abc167bb09b12b3a6a80761e817a92ec072f0`; it has no active
`strided-einsum2` dependency and no `gemm/strided_dot.rs` adapter.

The local working branch is hundreds of commits behind `origin/main` and
contains unrelated unresolved changes. All evidence collection and document
work therefore uses an isolated worktree based on the pinned baseline.

## Goal

Close #1591 only after producing an auditable requirement-by-requirement record
showing that each item is either:

1. **Satisfied** by current code, tests, and repository history;
2. **Not applicable** because the retired upstream execution path is absent; or
3. **A separate gap** recorded as a focused follow-up issue.

After #1591 is closed, update the related strided-rs issues so that retirement
issue #202 no longer treats tenferro migration as a blocker.

## Non-goals

- Re-import or transplant retired `strided-einsum2` implementation code.
- Add speculative GEMM abstractions, uninitialized-output APIs, or tests for
  paths that do not exist in tenferro.
- Execute the remaining strided-rs crate retirement work in #202.
- Resolve tenferro-rs #1592, which requires an independent scope decision.
- Modify the user's existing working tree.

## Evidence Model

The audit uses four evidence layers.

### 1. Dependency evidence

Confirm from the pinned baseline that:

- workspace and `tenferro-cpu` manifests do not declare `strided-einsum2`;
- Cargo metadata contains no such dependency;
- active source references are limited to contracts preventing reintroduction;
- `gemm/strided_dot.rs` is absent; and
- PR #1553 / commit `6255590e` records the removal.

### 2. Ownership evidence

Trace the active dot-general route through the CPU runtime, layout preparation,
and local Faer/BLAS providers. Record source locations proving that:

- validation occurs before provider execution;
- batch descriptors and checked offsets are tenferro-owned;
- provider threading comes from `CpuExecutionContext`; and
- output allocation and writeback use tenferro's buffer and tensor machinery.

The closeout must describe the actual resolution accurately: the stale Cargo
edge was removed after local provider ownership already existed. It must not
claim that PR #1553 copied the upstream implementation.

### 3. Known-problem disposition

Map every known problem listed in #1591 to current code and tests:

- independent batch-axis fusion and logical batch pairing;
- negative destination strides and provider fallback;
- rank/config validation before indexing or provider selection;
- checked offset and size overflow;
- Faer parallelism policy;
- conjugation without unnecessary full-buffer cloning;
- provider rejection before output mutation; and
- uninitialized-output contracts.

An upstream-only problem is marked **not applicable** only when source tracing
shows that the corresponding execution or storage path is absent. In
particular, the current CPU route uses initialized tensor outputs, so upstream
Faer `MaybeUninit` support and the upstream O(elements) uninitialized-output
injectivity check are not migration prerequisites.

Run the smallest focused existing tests that substantiate the matrix. Add no
code by default. A missing behavioral guarantee becomes a proposed follow-up
issue rather than an unreviewed expansion of #1591.

### 4. Provenance evidence

Inspect relevant file history and work logs to distinguish original tenferro
code from adapted code. In particular, cite commit
`eb689172666004ca70618757c62188181635429f` and
`docs/worklogs/2026-06-23-strided-einsum2-removal.md`: they document the earlier
local implementation/adaptation that moved Faer preparation into tenferro.
Distinguish that lineage from PR #1553 / commit
`6255590e76d21f3ec7ba2a7feaa7e160baecabc1`, which later removed stale dependency
declarations and feature wiring. Do not claim that no adapted code ever
existed; record which step owns the adaptation and which step removed the stale
build-graph edge. If copied or materially adapted implementation lacks
sufficient attribution, #1591 remains open until provenance is resolved.

## GitHub Update Flow

Apply issue updates only after the evidence matrix is complete:

1. Post a closeout comment on tenferro-rs #1591 containing the conclusion,
   requirement matrix, exact verification commands, and any follow-up links.
2. Check the rendered comment and links, then close #1591.
3. Update strided-rs #199 with the actual Phase 1 resolution.
4. Update strided-rs #202 to state that the tenferro dependency blocker is
   cleared; do not close #202 because crate retirement remains.
5. Cross-link strided-rs #198 and #201, recording that #198 is not transferred
   to tenferro because tenferro does not expose the upstream uninitialized Faer
   destination path.
6. Leave tenferro-rs #1592 unchanged.

Prefer comments over rewriting historical issue descriptions. Edit checklist
boxes only when permissions are available and the corresponding statement is
fully evidenced.

## Stop Conditions

Do not close #1591 if any of the following is true:

- an active dependency or adapter path remains;
- a correctness or safety requirement cannot be classified with evidence;
- a required focused test fails;
- relevant copied code lacks confirmed provenance; or
- GitHub state differs materially from the issue relationships assumed here.

Issue changes are ordered comment-first and close-last so they remain readable
and reversible. Record the resulting issue URLs in the execution report.

## Deliverables

- This approved design.
- A step-by-step execution plan with exact files, commands, expected outcomes,
  issue-comment templates, and stop conditions.
- During later execution, an evidence matrix and the resulting GitHub issue
  update URLs.
