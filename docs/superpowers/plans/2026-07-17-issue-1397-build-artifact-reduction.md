# Issue #1397 build-artifact reduction implementation plan

1. Capture a fresh four-job release-test baseline and inventory test targets,
   test executables, largest artifacts, dependency features, and duplicates.
2. Add source-contract tests for the intended consolidated harness count, then
   move integration tests behind one explicit harness per compatible crate.
3. Rebuild from a fresh target and record the test-target-only delta.
4. Fix the `strided-einsum2` faer dependency contract upstream with feature
   combination tests, then update tenferro to the corrected dependency.
5. Add dependency-contract tests for the minimal faer feature set and verify
   removed packages with `cargo tree`.
6. Add failing linalg provider-isolation contract tests, make `faer` and
   `lapack` optional under their provider features, and compile faer-only,
   BLAS-only, both-provider, and no-default configurations.
7. Audit direct CubeCL/cudarc imports and resolved CUDA features, add dependency
   ownership tests, remove redundant operation-crate edges, and compile CUDA
   targets without executing GPU tests.
8. Repeat fresh measurements after every cumulative stage and write the tables,
   decisions, rejected alternatives, and residual risks to a work log.
9. Run formatting, focused tests, feature checks, the local debug PR gate, full
   release tests, coverage, docs, clippy parity, and repository-rules review.
10. Commit coherent stages, create the tenferro PR (and prerequisite strided-rs
    PR when required), enable the repository-prescribed auto-merge mode, and
    monitor CI through completion.
