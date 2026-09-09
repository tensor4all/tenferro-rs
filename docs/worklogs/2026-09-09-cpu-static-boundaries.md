# CPU static replay and basic/fused boundaries

## Scope and provenance

This work follows the maintainer-approved compile-time investigation: use crate
boundaries for lightweight CPU use, retain ordinary CPU automatic fusion, and
share typed replay without adding dynamic callbacks. Crate-path compatibility
is not a requirement. No package publication is authorized by this work.

The implementation branch starts at tenferro `a096f280ab0dd774f92f533aa0f38b663a4c94d3`;
its paired strided worktree starts at `dc0a8e03286c61a84d56446b5cc2c53295f75d76`.
The earlier experiments used tenferro `e5b8c65ee6b1e2c418b1b23e5b91feb6dc68893c`
and strided `b29e7601ba090aa5eafc65b9bde5d9450282e0d8`.

Read: current repository rules and PERFORMANCE_TIPS, current kernel/analytic
entrypoints and callers, buffer guard source contracts, and the experimental
static-sharing transforms. Existing unpublished user changes in the original
worktree were not copied or modified.

## Implemented so far

The static replay changes were reapplied by function to current main, not by
copying experimental crate files. They share owned/read replay for arithmetic,
analytic/pow, unary, select/clamp and ordered min/max. Method function items
replace repeated expression-site closures. Existing allocation/check ordering,
integer domain checks, complex abs output dtype, and specialized multiplication
and comparison kernels remain. No unsafe block or dynamic callback was added.

The ordinary CPU fusion hook is unchanged. A regression test uses an input at
the existing 16K fusion threshold and checks both fused outputs through the
ordinary backend, then reclaims them. Small inputs intentionally need not fuse.
The buffer-guard source contract now verifies that add delegates to the guarded
binary helper, rather than requiring obsolete unsafe comments in the wrapper.

Focused checks passed:

- `cargo test -p tenferro-internal-cpu-kernels --lib -- --test-threads=1`: 59 tests.
- `cargo test -p tenferro-cpu --test integration -- --test-threads=1`:
  50 tests, including the ordinary fusion hook.
- `cargo test -p tenferro-cpu --lib -- --test-threads=1`: 538 tests.

Test execution used RAYON_NUM_THREADS=1 and explicit one-thread CPU backends.
Cargo build concurrency was 4, not a runtime throughput measurement.

## Remaining before merge readiness

The basic/fused crate extraction, narrow shared resource APIs, tensor read-into
backend contract, downstream adapters and publication DAG are not yet complete.
Tests must move with implementation ownership. Final coverage review, local PR
gate, documentation/skill updates and performance validation must run on the
submitted candidate. The historical experimental timings do not certify this
new main-based, fusion-preserving candidate. In particular, earlier scalar div
measurements included a slower case whose cause was not established.

Issue #1706 predates this lightweight-use design and has different quantitative
acceptance criteria. Its original acceptance criteria must not be silently
reported as satisfied by the exploratory measurements.
