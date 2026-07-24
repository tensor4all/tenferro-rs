# Agent Workflow Lessons

These lessons apply to long-running, multi-phase implementation work. They are
model-independent process checks, not requirements to use a specific agent
surface or delegation strategy.

## Process Self-Audit

At task or phase boundaries, evaluate whether the current workflow is reducing
uncertainty or adding coordination overhead. If progress is stalling, pause and
simplify the workflow before adding more implementation.

Use these checks:

- Is the next implementation slice small enough to verify independently?
- Are RED/GREEN failures diagnostic, or are unrelated failures mixed in?
- Is delegation, review, or additional planning reducing risk enough to justify
  its coordination cost?
- Is the root agent maintaining a short execution ledger instead of carrying all
  prior context forward?
- Are reviewer or delegated-agent findings being verified as technical claims
  before changing code?

## Delegation

Use delegation only when it reduces effective context load or adds independent
review signal. Keep delegated tasks bounded, with explicit inputs, outputs,
non-goals, and stop conditions.

Treat delegated findings as claims, not facts. The root agent remains
responsible for technical judgment, verification evidence, commits, pushes, and
phase advancement.

## TDD Slice Hygiene

Prefer small RED → GREEN slices. Do not batch unrelated RED tests when each
requires separate production APIs. Rust integration test targets compile all
registered modules, so unrelated unresolved imports can obscure the intended
failure.

For RED-only reviews, explicitly state that production files may be
intentionally absent and compile failure can be expected. The review target is
whether the tests specify the intended contract, not whether the final program
already compiles.

Keep source-contract tests narrow. They should guard specific high-risk
regressions and should not overfit function order, byte windows, or helper
structure when behavior tests can express the contract.
