# Bounded compiler metadata passes

## Decisions

- tensor4all-rs#1408 exposed backward graph preparation dominated by layout
  simplification: the supplied macOS sample attributed 1171 of 1370 backward
  samples to this pass. Removing the outer fixpoint in #1260 left a whole-program
  input/output scan for every identity reshape, identity transpose, or cancelling
  transpose pair.
- Keep the existing algebra and replace eager use rewriting with a forward
  canonical-slot table. Semantic staging rejects unavailable inputs and duplicate
  producers, and lowering consumes input metadata before recording outputs. Thus
  aliases always point to sources already visited: each instruction input and
  program output needs one lookup, without recursive resolution or use-site lists.
  This keeps the change portable to the downstream pinned runtime revision.
- The first paired release experiment remains inconclusive and does not establish
  the end-to-end acceptance gate. A subsequent release profile attributed 391 of
  461 samples in the linear-solve AD segment to conjugation sinking, meeting the
  predeclared additional-path investigation threshold. Redirect resolution
  repeatedly allocated a slot-sized cycle table, while metadata collection and
  producer registration copied symbolic shape trees and operations on every pass.
- Resolve redirects with an allocation-free hop budget. The pass writes resolved
  targets and processes producers before consumers, so valid production redirects
  stay canonical. Keep bounded chain traversal and cycle rejection for malformed
  redirect tables.
- Record metadata locations in the owning IR rather than copies of every slot's
  shapes, extents, and producer operation. Extension dtype inference borrows those
  shapes; only an actually emitted conjugate copies its source metadata. The
  existing sinking algebra and fixpoint remain unchanged.
- Existing Complexity Budget guidance already prohibits both defects; no
  additional policy is needed.

## Verification conclusions and constraints

- Added structural regressions for mixed identity chains, duplicate/fan-out uses,
  program-output aliases, transpose composition across aliases, and the complete
  standard-operation compilation entry point. A debug assertion rejects programs
  that read a slot before its producer, which the forward alias table requires.
- No test inspects the compiler source text: such checks pin the spelling of the
  implementation rather than its behavior. The ignored release probes measure
  the scaling property instead.
- The ignored release probe measures only the private compiler pass, with setup
  and program cloning outside the timed interval, over 64/256/1024/4096 operations.
  No numerical backend or new public benchmark API is involved.
- The initial layout change's focused compiler tests pass on current main (86 checks) and on the downstream
  pinned runtime with Accelerate (85 checks). The local PR gate, including
  workspace and extension Clippy, passes.
- Independent review of the initial layout change found no actionable defects. The implementation and tests
  are byte-identical in the downstream backport; its dependency versions and
  public APIs remain unchanged.
- Follow-up conjugation regressions check redirect boundaries/cycles, borrowed
  metadata, shared conjugates, all supported conjugation dtypes, and exact
  symbolic shapes/extents, including a mixed-output extension. Its additional
  diagnostic scaling probe uses the same pass APIs on the original pin and the
  candidate, without changing the original paired suite or acceptance thresholds.
- The follow-up passes 94 compiler checks and the upstream local PR gate,
  including strict workspace and extension Clippy. Independent review found no
  actionable correctness defects.
- The downstream backport passes 93 compiler checks in release mode, and all
  five compiler/test files match the reviewed upstream implementation. The
  downstream five-crate ordinary suite passes 2,072 checks; all eight retained
  heavy witnesses also pass.
- The complete paired rerun meets the unchanged local primary thresholds and
  load/noise checks: both unchanged AD tests take about 14% of baseline time.
  [Full experiment evidence](../performance/compiler-preparation-2026-11-20/README.md)
  summarizes the first inconclusive run, every case, and the diagnostic; the raw
  samples and intervals are linked at their immutable pre-removal commit.
  These are unpinned macOS results; pinned-core idle observations and
  affinity-controlled Linux validation required by the full upstream host
  protocol remain unavailable. Do not claim that full gate is satisfied.
