# Issue #1906 binary einsum dispatch

## Decisions

Use a checked no-tree binary path for supported ordinary two-operand notation: unique labels, shared omitted contracting labels, remaining labels exactly matching the output, and valid dot-general extents. Reuse `BinaryDotPlan`, try its existing operand-order logic for exact output order, and dispatch directly to backend read-into. Unsupported notation retains the general tree path. Prepared plans should retain executable binary metadata and compare borrowed shapes/dtypes without rebuilding heap input specs or output shapes.

Rejected a parser/cache rewrite and new backend execution contracts in this slice. If the existing parser or backend metadata keeps the performance target out of reach, measure and report it rather than expanding scope across crate boundaries.

## Verification conclusions and constraints

The concrete read-into wrappers now attempt the existing exact-output binary dot plan before constructing a contraction tree. Prepared plans retain the binary descriptor and output shape; execution compares borrowed dtype/shape metadata directly and reuses those values. Unsupported shapes/notation continue through the original tree fallback. Tests cover swapped `ij,pj->pi`, offset/strided output, prepared reuse, shape/dtype validation, and existing crate behavior.

The supervisor's independent semantic-parity suite found two regressions in the initial implementation. Added runtime tests first reproduced both defects: the operand-order test panicked at `eager.rs` indexing a rank-2 shape with axis 2, and the extra-input typed-read case returned success and changed output. The shape planner now checks dimensions in the order selected by `BinaryDotPlan`; the read-into planner now refuses any input count other than two, preserving general validation before output mutation. New tests also verify that singleton contracting extents fall back to general einsum broadcasting.

Focused RED/GREEN: each regression test failed before the fix and passes after it. After the fix, all three focused tests pass; `cargo test -j 16 -p tenferro-einsum` passes all 316 tests, and `cargo fmt --all -- --check` plus `git diff --check` pass. A post-fix `autodiff`-enabled binary-dot filter passes all five matching tests. No release or paired benchmark was run in this worker scope; the parent owns it. The current parser and `BinaryDotPlan` use heap-backed metadata, and typed-view wrappers still collect `TensorRead`s, so zero hot-path allocations and the <=1us target are not established. Backend layout-analysis overhead, GPU behavior, and f32 swapped-output remain unverified. Baseline source investigation and the predeclared paired performance protocol are described in `/tmp/tenferro-1906-source/docs/worklogs/2026-09-25-issue-1906-dispatch-investigation.md` and `/tmp/tenferro-1906-implementation/protocol.md`.
