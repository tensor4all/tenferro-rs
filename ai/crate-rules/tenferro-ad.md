# tenferro-ad crate rules

Template and policy: [`README.md`](README.md).

### A test must assert the change's property at a case where the change runs

- **Applies to:** tests added with a gated or conditional path: a fast path, a
  cache, a fused or in-place variant, a session reuse, a threading threshold.
  The common shape here is a bit-identity test that compares the new path with
  a reference, as in `tests/integration/eager_elementwise_into.rs`, which
  compares `to_bits()` of the `_into` result with the allocating result.
- **Check:** two review questions, each answered from the code, not the test
  name. (1) For every case in the test, does the gate let the new path run?
  Read the gate's condition and make the test fail if the path did not run
  (see the next entry for how to count that). (2) Is the asserted property
  one the code has on every host and run? In tenferro, bitwise equality is
  promised at a fixed thread count only, never across thread counts
  (`docs/spec/reproducibility.md`, restated in `PERFORMANCE_TIPS.md`).
  Comparing `to_bits()` of a threaded path with the serial path, or of two
  thread counts, asserts a property the code does not have; use a tolerance
  with a reported residual there, and keep `to_bits()` for two paths at one
  thread count with the same reduction order. Run the test several times and
  with `--test-threads=1` before trusting it.
- **Incident:**
  [colmat-rs#244](https://github.com/tensor4all/colmat-rs/pull/244) -- a
  bit-identity test compared the team (parallel) LU with `Par::Seq`. The cell
  that failed on aarch64 was team width 3, where the look-ahead under test never
  ran ("the per-call count was 0, and width 3 is below `LOOKAHEAD_MIN_WIDTH`"),
  and the team product itself was "scheduling-dependent summation", so the
  comparison "sometimes differed ... and sometimes matched". The replacement
  test, look-ahead on against look-ahead off, then
  [failed at n = 257, width 4](https://github.com/tensor4all/colmat-rs/pull/244#issuecomment-7130147755)
  on the same host: bit identity was still not a property the code had there.

### A counter a test asserts belongs to the call, not to the process

- **Applies to:** any "did the path run" or "how many times" check in tests:
  `static` atomics, thread-locals, and global stats. `src/eager.rs` defines
  a `#[cfg(test)] static EAGER_SEMANTIC_VJP_EXECUTIONS: AtomicUsize`, which the
  semantic VJP path increments; nothing asserts on it today, and a new
  assertion must not start doing so.
- **Check:** `cargo test` runs the tests of one binary on concurrent threads, so
  another test can move a process-global counter. Use a counter owned by the
  call: an `Arc<AtomicUsize>` held by an engine or context the test builds, as
  `eager_materialization_uses_the_engine` in `src/eager/tests.rs` does with
  `RecordingEngine`, or per-instance stats such as `AdContext::cache_stats()`.
  A thread-local is not a fix when the increment can happen on a worker thread.
  Confirm with both `cargo test -p tenferro-ad <name>` and
  `cargo test -p tenferro-ad <name> -- --test-threads=1`; a test that passes
  only with concurrency is reading someone else's increment.
- **Incident:**
  [colmat-rs#244 review](https://github.com/tensor4all/colmat-rs/pull/244#issuecomment-7087305856)
  -- the "the look-ahead ran" guard read a process-global `AtomicUsize`, which a
  concurrent test bumped, so with the row-major half of the feature disabled the
  test still passed with default test threads and failed with
  `--test-threads=1`. The
  [follow-up](https://github.com/tensor4all/colmat-rs/pull/244#issuecomment-7087357244)
  found that a thread-local made it worse, "because the increment happens on the
  leader thread inside the team, not on the test's thread"; the counter had to
  be per-invocation and plumbed through the call.
