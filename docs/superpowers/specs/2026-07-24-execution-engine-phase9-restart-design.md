# Execution Engine Phase 9 Restart Design

Date: 2026-07-24

## Purpose

Restart the execution-engine implementation after the Phase 2E campaign
failure, preserve the high-quality Phase 1/2 Rust work, repair the
multi-operation session-entry regression, and continue through the accepted
architecture phases to the Phase 9 exact-commit audit.

This restart deliberately treats benchmark infrastructure as supporting
equipment rather than as the main product. Correctness remains a hard gate.
Noisy microbenchmarks provide regression evidence without requiring every
measurement to prove five-percent non-inferiority.

## Authoritative Inputs

- Issue #1433, including the accepted umbrella architecture contracts.
- Issue #1436 and the six preserved negative Phase 2E runs.
- The maintainer-provided 2026-07-24 external audit.
- `HANDOFF-2026-07-24-tenferro-phase9-restart.md`.
- Workspace and tenferro repository rules.
- Accepted child-phase designs, beginning with Phase 3 Issue #1449.

When these sources disagree, the maintainer audit and later explicit
maintainer decisions control the restart.

## Current State Preservation

The stopped branch and worktree remain forensic inputs.

- Do not launch run-0007.
- Do not commit the six uncommitted Phase 2E harness files to a Rust
  integration branch.
- Do not delete or rewrite the six preserved negative evidence roots.
- Do not continue inode, descriptor, manifest, or sealing hardening unless a
  maintainer separately decides to retain that threat model.
- Do not use passing harness unit tests as evidence of progress toward Phase 9.

The stopped branch may be tagged or retained as an archive. Integration work
starts from a clean branch based on the latest `origin/main`.

## Integration Structure

### Rust integration branch

Create one clean integration branch for reviewable Rust implementation:

1. start from the latest `origin/main`;
2. reconstruct the accepted Phase 1/2 Rust implementation with coherent
   commits;
3. include only curated design, API, test, and work-log documentation;
4. exclude raw benchmark output, repeated lockfiles, preserved run trees, and
   the current large Phase 2E Python harness;
5. keep phase boundaries visible in commit history.

The reconstruction must preserve behavior and authorship where practical. It
must not silently rewrite semantics merely to make cherry-picking easier.

### Harness and evidence

The benchmark harness has an identity independent of the Rust candidate.
Harness fixes do not require source changes or a new candidate identity.

Raw outputs belong outside the Rust integration history. Git retains only:

- the public measurement protocol;
- the minimal runner and classifier needed to reproduce a decision;
- small representative fixtures;
- a concise result summary with candidate, harness, environment, and artifact
  digests.

## Multi-Operation Session Repair

Before architecture Phase 3 production implementation, repair the confirmed
session-entry regression.

`origin/main` enters the Faer/Rayon execution context once around a backend
session. The stopped branch constructs one backend session but each contained
operation reaches `CpuOperationEntry::enter -> install_scoped ->
executor.install`, potentially paying pool entry once per operation.

The repaired contract is:

- explicit backend sessions on Tenferro-managed executors enter the selected
  execution domain at most once for a multi-operation closure;
- compiled multi-operation programs enter at most once per compatible
  execution scope;
- eager standalone operations retain a cheap single-operation path;
- nested provider work reuses the active compatible execution scope;
- sequential, inner-parallel, and outer-parallel ownership remain explicit
  and do not double-fan out.

Fallible external executors retain operation-level entry until the generic
`BackendSessionHost` callback contract can represent a typed admission failure
before the callback begins. The repair must not replace that failure with a
panic, a hidden fallback, or an untyped side channel.

Implementation follows test-driven development. A regression test must first
demonstrate that two or more operations in one session cause repeated entry on
the stopped implementation, then pass with exactly one compatible entry after
the repair.

## Practical Performance Protocol

### Hard gates

The following remain blocking:

- numerical or semantic incorrectness;
- panic, deadlock, resource leak, or invalid ownership;
- a reproducible API or placement contract violation;
- failure of repository-mandated correctness tests.

### Microsecond overhead gate

For dispatch, session-entry, and orchestration benchmarks whose baseline
median is at most 10 microseconds:

- a regression is blocking only when the candidate is at least 50 percent
  slower on a predeclared primary case and the slowdown reproduces in a second
  complete paired A/B run;
- a single noisy slowdown is not blocking;
- `INCONCLUSIVE` is not failure and does not close the candidate;
- the same candidate may be rerun without source modification;
- secondary cases are diagnostic and do not all need to pass one uniform
  threshold.

The primary Phase 2 restart cases are:

1. explicit multi-operation CPU backend session;
2. compiled multi-operation CPU program;
3. representative standalone eager operation.

Large-kernel throughput and scaling benchmarks do not inherit the 50-percent
microbenchmark allowance. Each later phase must predeclare a
workload-appropriate practical threshold when performance is part of its
acceptance contract.

### Reachability and retry policy

Before candidate data is collected:

- run the complete harness end to end on a dummy candidate;
- confirm that harness identity and candidate identity are independent;
- confirm that PASS, regression, and INCONCLUSIVE outcomes are all reachable
  with synthetic inputs;
- publish the protocol amendment on the owning issue.

At most two campaign attempts may fail before producing a primary performance
sample. A second harness-originated failure stops the campaign and returns the
decision to the maintainer. It does not authorize further autonomous
hardening.

## Phase Progression

After the Phase 1/2 Rust unit and session repair are reviewable:

1. implement the accepted Phase 3 child design;
2. verify the phase-specific correctness and public contracts;
3. record a concise work log and remaining risks;
4. proceed through Phases 4-8 using the same checkpoint;
5. stop for maintainer direction if a phase requires unaccepted public API,
   architecture, backend, dependency, feature, or AD semantic changes.

No benchmark harness problem may silently expand an architecture phase.
Generated evidence volume is not a progress metric.

## Phase 9 Audit

Phase 9 runs only after Phases 3-8 are implemented and the required repository
verification is green.

- Freeze one exact candidate commit.
- Run the repository-mandated cross-phase multi-agent audit against that
  commit.
- Give reviewers separate bounded concerns: architecture and layering,
  correctness and error behavior, performance and resource ownership,
  documentation and public API consistency, and test/benchmark quality.
- Aggregate findings once; do not create independent fix loops per reviewer.
- Resolve Blocking and Important findings with focused changes and rerun the
  affected verification.
- Repeat the exact-commit audit only when the candidate changes materially.

Phase 9 completes when the exact candidate has no unresolved Blocking or
Important findings, repository verification passes, and residual risks are
explicitly documented.

## Anti-Runaway Controls

- Two consecutive failed fix or measurement attempts trigger an architectural
  or maintainer checkpoint.
- New security or durability threat models require explicit maintainer
  approval.
- Raw evidence and generated bulk cannot enter the Rust integration branch.
- Every phase reports progress in production contracts implemented, not in
  commits, tests added, artifacts preserved, or harness lines written.
- Independent review must ask whether the mechanism is proportionate and
  necessary, not only whether it is internally correct.
- A terminal instruction to continue through Phase 9 does not authorize new
  public semantics outside accepted issues and designs.

## Completion Criteria

The restart is complete when:

- the Phase 1/2 Rust unit is separated and reviewable;
- multi-operation execution enters the CPU session once per compatible scope;
- the practical benchmark protocol is published and dry-run;
- no reproducible 50-percent microsecond-overhead regression remains in the
  predeclared primary cases;
- Phases 3-8 are implemented under their accepted contracts;
- Phase 9 exact-commit audit and repository verification pass;
- the large stopped harness and evidence bulk remain outside the integration
  unit.
