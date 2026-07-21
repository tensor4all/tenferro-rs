# Final Cross-Phase Multi-Agent Audit Design

**Status:** Accepted design for the phase 0--9 execution-engine program.

## Purpose

Repository-scale changes can pass local task reviews while retaining conflicts
between phases, backends, or resource-ownership contracts. Before such a program
is declared complete, the repository must require a final audit performed by
independent agents against one exact candidate revision.

The normative checklist belongs in one `Final cross-phase multi-agent audit`
section in `REPOSITORY_RULES.md`. That section references the existing safety,
performance, testing, and documentation rules instead of copying them.

## Trigger And Independence

The final audit runs after all implementation phases and their local reviews are
complete, but before the umbrella issue or implementation branch is declared
ready for integration. Every report names the exact candidate commit. A final
auditor must not audit a lane whose implementation or local review it performed.
The lanes may run in several batches when agent concurrency is limited.

## Required Audit Lanes

Use a distinct independent auditor for each lane:

1. **Specification and architecture:** accepted issues, phase acceptance
   criteria, eager/graph semantic parity, extension lowering, and migration
   compatibility.
2. **Rust safety and resource lifecycle:** aliasing, unsafe boundaries,
   lifetimes, permits, locks, buffers, caches, identifiers, and success, error,
   cancellation, and unwind cleanup.
3. **Performance and parallelism:** the current-main baseline, eager fast path,
   allocations, request/container overhead, nested fan-out, provider worker
   ownership, thread-count and placement control, and CPU/GPU synchronization.
4. **Public API and documentation:** facade boundaries, operation-family traits,
   typed errors, feature combinations, runnable examples, online parallelism
   documentation, and source/checker consistency.
5. **CPU and NUMA:** managed and external domains, strict versus advisory
   placement, `ResourceArbiter`, faer/BLAS/strided behavior, multiple sockets,
   re-entry, fairness, and failure recovery.
6. **GPU, XLA, and multi-GPU:** context/stream/event ownership, backend-neutral
   artifacts, compiler and prepared-operation caches, device placement,
   independent devices, and cross-device failure handling.

After all lane reports, a separate integration auditor checks cross-phase
invariants, duplicated or contradictory findings, and the closure evidence.

## Evidence And Severity

Every lane report records:

- candidate commit and relevant feature/toolchain/hardware configuration;
- files, public contracts, and issue acceptance criteria inspected;
- fresh commands and their complete result classification;
- findings classified as `Critical`, `Important`, or `Minor`; and
- explicit limitations, skipped hardware paths, and performance results as
  `PASS`, `FAIL`, or `INCONCLUSIVE`.

No lane may infer a pass from an implementer's earlier run. Source scanners and
mutation tests are supporting evidence, not substitutes for call-path review and
runtime tests.

## Closure

The final audit passes only when:

- every `Critical` and `Important` finding is fixed and re-reviewed by the
  auditor that raised it or another independent auditor;
- every `Minor` finding is fixed or has a written rationale and accepted
  tracking issue;
- a required performance gate is `PASS`; `INCONCLUSIVE` blocks promotion until
  a valid rerun or an explicit accepted scope decision is recorded;
- environment-limited CPU, GPU, XLA, and multi-device paths have reproducible
  diagnostics and an identified verification owner; and
- the integration auditor reports no unresolved cross-phase contradiction.

The final worklog links every lane report, the integration report, the exact
candidate commit, and the final verification commands.

## Non-Goals

This rule does not replace task-local TDD, spec review, code-quality review, CI,
or performance gates. It does not require all auditors to run concurrently, and
it does not permit audit agents to silently modify the candidate while reviewing
it.
