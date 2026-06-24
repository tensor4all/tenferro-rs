# Buffer Pool Panic Replenishment

## Summary

Fixes #1164 by making `CpuBackend` pool loans replenish retained buffer
capacity when a panic unwinds through the loan. The fix preserves the safety
rule that partially initialized in-flight vectors are not reinserted into the
pool.

## Context Read

- `crates/tenferro-cpu/src/buffer_pool.rs`
- `crates/tenferro-cpu/src/backend.rs`
- `crates/tenferro-cpu/src/backend/tests.rs`
- `crates/tenferro-cpu/src/buffer_pool/tests.rs`
- GitHub issue #1164

## Root Cause

`PoolScalar::pool_acquire` removes a retained vector from the pool and returns
a plain `Vec<T>`. If a kernel panics before the vector is returned with
`pool_release`, the vector is dropped during unwinding. `BufferPoolLoan`
restored the loaned pool object to the backend, but it had no record of retained
capacities that were checked out and then lost.

## Decision

Track only pool-backed in-flight capacities inside `BufferPool`. A successful
`pool_release` decrements the matching in-flight count. When `BufferPoolLoan`
drops during panic unwinding, it replenishes remaining in-flight capacities with
empty same-capacity replacement vectors, then restores the pool to the backend.
On normal loan completion, it clears in-flight records so escaped output buffers
are not duplicated back into the pool.

This avoids retaining a potentially partially initialized vector while still
preventing repeated caught panics from progressively draining the backend's hot
buffer pool.

## Verification

- `cargo fmt --all --check`
- `cargo test -p tenferro-cpu linalg_pool_acquire_then_panic_replenishes_retained_buffer -- --nocapture`
- `cargo test -p tenferro-cpu replenish_in_flight_retained -- --nocapture`
- `cargo test -p tenferro-cpu`
- `cargo clippy -p tenferro-cpu --all-targets -- -D warnings`
- `cargo test -p tenferro-cpu --release`
- `cargo llvm-cov -p tenferro-cpu --release --json --output-path /tmp/tenferro-cpu-coverage-1164.json`
- `cargo llvm-cov --workspace --release --json --output-path /tmp/tenferro-workspace-coverage-1164.json`
- `python3 scripts/check-coverage.py /tmp/tenferro-workspace-coverage-1164.json`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo test --workspace --release`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`

## Residual Risk

The replenishment hook is tied to unwinding through `BufferPoolLoan`. If a
caller catches a panic entirely inside a loaned closure and continues normally,
the loan cannot distinguish that internal panic from an ordinary successful
operation. That pattern is outside the reported backend boundary and should be
handled by the caller if needed.
