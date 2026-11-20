# Bounded layout simplification

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
- Existing Complexity Budget guidance already prohibits the defect; no additional
  policy is needed. The nearby conjugation pass still allocates a slot-sized cycle
  table for each redirect resolution. That separate path remains unchanged pending
  evidence from the post-fix end-to-end profile.

## Verification conclusions and constraints

- Added structural regressions for mixed identity chains, duplicate/fan-out uses,
  program-output aliases, transpose composition across aliases, and the complete
  standard-operation compilation entry point. The source-contract negative control
  detects the old per-rewrite scan without a wall-clock assertion.
- The ignored release probe measures only the private compiler pass, with setup
  and program cloning outside the timed interval, over 64/256/1024/4096 operations.
  No numerical backend or new public benchmark API is involved.
- Focused compiler tests pass on current main (86 checks) and on the downstream
  pinned runtime with Accelerate (85 checks). The local PR gate, including
  workspace and extension Clippy, passes. The Rust source-contract check rejects
  the original implementation and passes with canonical aliases.
- Independent review found no actionable defects. The implementation and tests
  are byte-identical in the downstream backport; its dependency versions and
  public APIs remain unchanged.
- Paired release measurements are pending. No speedup is claimed yet. Local
  validation is being staged to fit the host's limited free disk.
