# Phase 2E dispatch and characterization evidence

Date: 2026-07-22

## Scope

Task 7 adds crate-owned dispatch evidence without introducing a CPU-to-AD
dependency or production counters. The CPU artifact owns 29 characterization
rows and ten direct/borrowed-session vectors. The AD artifact owns 18 public
placement-bound rows and five ordinary eager single-session-entry proofs. The
gate composes exactly 47 unique canonical keys and rejects duplicates,
omissions, wrong ownership, count/mode mismatches, failed recovery fields, and
untyped hardware skips.

The row fixtures are executable rather than declarative. Module-local executor
wrappers count real `install` and `submit` calls and record the CPUs observed by
their jobs; an explicit pool-affinity audit samples the actual worker CPUs for
every ownership/budget fixture, and module-local GEMM wrappers count calls
while delegating to faer.
External exact fixtures pin their caller-owned Rayon workers when enough CPUs
are visible, while advisory fixtures retain observations without upgrading the
placement claim. D-N uses nonuniform `f64[65536]` inputs with exact comparison,
D-D/E-D use nonuniform `f64[128,128]` inputs with a `1e-12` relative Frobenius
bound, and G-O executes `J = 2B + 1` real grouped requests. U-O executes the
typed pre-submit rejection and U-I executes the no-inner sequential fallback.
Every success row injects a typed operation error and unwind before a successful
post-recovery operation.

The complete `RecordingBackend` fixture now lives in the module-local
`eager_backend/tests.rs`. Production `eager_backend.rs` retains only the test
module declaration, private test-only type reference, and dispatch arms.

## TDD record

The plan-prescribed initial commands produced the following baseline:

- CPU and AD filtered Cargo tests reported zero matching tests and exited zero,
  which is Cargo's normal missing-filter behavior.
- `python3 -m unittest scripts/test_run_phase2e_gates.py -v` failed to import the
  nonexistent module.

After registering the Rust tests, the compile RED exposed obsolete
`SliceConfig` field names (`start_indices` and `limit_indices`); these were
replaced by `starts` and `limits`. The first CPU Criterion compile then exposed
the missing `TensorElementwise` trait import. Both defects were fixed before
the corresponding GREEN runs.

## Provenance and deadlines

`phase2e_build.py` now owns the exact locked CPU/AD library-test and Criterion
build argv, manifest locations, executable selection, and executable/source/
lock/feature-graph/environment digest fields. `run_phase2e_gates.py` executes
the hashed test binaries directly with the evidence filters and `--nocapture`;
it does not invoke Cargo during evidence collection.

Correctness executables have a 120-second deadline. Each Criterion row has a
30-second deadline. Process groups receive a five-second termination grace
before forced kill. The two Criterion harnesses use fixed two-second warmup,
five-second measurement, 100 samples, and 95% confidence.

## Verification

- CPU and AD focused evidence tests passed and their emitted JSON composed to
  exactly 47 rows.
- All ten gate-wrapper contract tests passed.
- All 83 `phase2e_build.py` contract tests passed (93 Python contract tests in
  the combined invocation).
- The full CPU suite passed 512 tests and the full AD suite passed 71 tests.
- The focused provider filter passed 57 tests and the dot-runtime filter passed
  43 tests.
- Both characterization Criterion binaries compiled with `cpu-faer` and no
  default features.
- Clippy passed for both crates across library, tests, and benches with warnings
  denied; rustfmt and `git diff --check` passed.
- The repository-rules review passed with no findings.
- Direct `--list` execution found all 27 CPU and 18 eager benchmark row ids;
  benchmark setup successfully constructed every managed/exact/advisory fixture.
