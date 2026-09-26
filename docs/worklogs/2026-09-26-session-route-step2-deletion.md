# Deleting the operation one-shot spelling (#1926 / umbrella #1929)

## Session summary

Issue #1926 requires one operation interface for CPU and GPU. The one-shot
spelling — a value-returning operation method on the backend owner — was the
duplicate: it had a `_read` sibling from Step 1, so every one of the 31 paired
operations could be deleted once its callers moved to a session.

This record covers Step 2 of the migration: 31 operation deletions, the tooling
that made them tractable, the two real defects the deletions exposed, and what
is left (B3/B4/B5 and the Phase-C measurement).

## What the slice deletes, and what it must not change

Deleted: `TensorAnalytic` (9), `TensorElementwise` (14), `TensorStructural`
(3), `TensorReduction` (4), and `TensorDot::dot_general` — 31 pairs in total.
Retained, because they have no `_read` sibling and are therefore capabilities
rather than spellings: `TensorStructural`'s `cast`/`convert`/`extract_diagonal`/
`embed_diagonal`/`tril`/`triu`, the eight `TensorIndexing` operations,
`TensorElementwise::rem` and the `*_into` family, `TensorDot::dot_general_with_conj`,
and the cached dot trait.

Invariants each slice had to keep: values, dtype lattice, validation order,
typed errors, failure-output immutability, AD semantics, provider exclusion,
permit lifetime (CPU), GPU device ordering, and cache/plan lifetime.

## Chosen approach

Per *operation*, smallest call-site count first, one commit per operation, with
the deletion in the same commit as its callers so the tree compiles after every
slice. Family-sized slices were rejected: they would have mixed 1500 `add`-scale
call sites with a single deletion and made bisection useless.

The work list came from `cargo check` diagnostics, never from a name grep. The
pilot record already established why: of 35 non-test `\.dot_general\(` matches
only five were the deleted method, the rest being different APIs
(`EagerTensor::dot_general`, `TracedTensor::dot_general`, capability queries,
provider requests, `gemm::*` free functions).

Three mechanical aids, all outside the repository:

- a diagnostic-driven migrator that rewrites the span the compiler reports,
  wrapping only the arguments whose `_read` parameter is a `TensorRead`, and
  classifying a receiver from its own text (`self`/session receivers take the
  read form, owners take an explicit `with_backend_session` boundary);
- per-operation structural edits (trait item, CUDA body absorption into the read
  half, CPU session delegation, runtime extension, eager dispatcher), driven by
  an explicit parameter table with asserted counts;
- a mask-aware delimiter matcher over the original text, because string
  contents, raw strings and comments had repeatedly skewed positional splicing.

## Defects the deletions exposed

Both were invisible to `cargo check` and were only caught by running the suites:

- **A real implementation can hide behind a one-shot.** `WebGpuExecSession`'s
  `transpose_read` reported `unsupported!` while the working device transpose
  lived in the one-shot. Deleting the one-shot therefore turned a supported
  operation into an unsupported one and failed five WebGPU structural tests. The
  read half now calls the restored `structural::transpose`.
- **A migrated read half can call itself.** Where a read half delegated to the
  one-shot and the one-shot was removed first, rewriting the delegate produced
  `fn op_read(..) { .. self.op_read(..) }` in ten test fixtures. A repository-wide
  scan for self-recursive `*_read` bodies is now part of the audit; fixtures that
  had one carry the removed behaviour directly (delegate to the wrapped backend,
  return the fixture result, or keep the same rejection after materializing a
  borrowed view).

Two source-text contract tests also had to be retargeted to entries that outlive
the deletion (`cubecl_launch_contract` section ends, `public_surface_contract`
runtime-materialization section, `backend_capability_contracts` function list,
`webgpu_backend_contract` read-entry signature).

## Verification

- `cargo check --workspace --all-targets`: clean, warning-free.
- `cargo test --workspace`: 5207 pass; the only failure is the pre-existing
  environmental `tenferro-ad` trybuild span mismatch
  (`eager_backend_capability_contract`), which reproduces with the session's
  changes stashed.
- `cargo test --workspace --doc`: green, including the new `compile_fail`
  fixtures that pin the deleted spellings.
- Focused suites per slice (`tenferro-cpu`, `tenferro-tensor`,
  `tenferro-gpu --lib --tests`, `tenferro-einsum`, `tenferro-runtime`,
  `tenferro-ad`, `tenferro-linalg`, `tenferro-fft`, `tenferro-df64-proof`).

## Residual risks and open work

- **B3 has not landed.** `TensorBackend` still inherits `BackendSession`,
  `TensorBackendOps` and `BackendCachedDot`, so an owner can still be used as a
  session. Removing them was attempted and reverted; the design document records
  the measured work list (the `SessionCachedDot` blanket bound and the
  `default_backend_session` factory inside `tenferro-tensor`; the runtime FFI
  dispatch, `HostExecution`, the cached-dot bound and `reclaim_buffer` in
  `tenferro-runtime`). B3(i) cannot ship without B3(ii), because the runtime
  dispatch layer has to become session-taking in the same commit.
- **B4 and B5 remain**, and the Phase-C measurement has not been re-run since
  the deletions. The route baseline (`docs/testing/session-route-baseline.json`)
  still holds, and the before-only one-shot cases are expected to be reported as
  `DELETED` by the comparator.
- The session-entry audit that B3 will shrink is in place early, so this work
  cannot regress silently: any new entry mechanism, including through a renamed
  import, fails `scripts/check-pr-fast.sh`.
