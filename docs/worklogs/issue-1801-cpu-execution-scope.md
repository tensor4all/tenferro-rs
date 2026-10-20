# CPU eager/trace shared execution scope

## Decisions

- Implemented the accepted [issue #1801](https://github.com/tensor4all/tenferro-rs/issues/1801)
  at the existing CPU admission boundary. Ordinary eager, AD and prepared trace
  callers keep their APIs; no Tensor unification or GPU scope was introduced.
- A callback-lifetime admission record retains the immutable backend witness
  and permit, while a separate operation loan preserves exclusive resource
  borrowing and nested-entry rejection. This avoids borrowing one mutable
  backend session across arbitrary high-level calls. It is not a TLS cache.
- External executor admission remains operation-scoped. Infallible session
  entry retains its established panic boundary rather than changing a public
  trait in this feature. The shared-scope design is retired by #2206; see the [engine ownership record](../design/execution-engine-ownership.md).

## Verification conclusions and constraints

- Disabling entered-context reuse makes the entry regression report **65
  installations instead of 1**. With reuse enabled, repeated native operations,
  cached/uncached sessions, GEMM and linalg-context access install once at
  explicit CPU budgets 1 and 4, for the compiled Faer and BLAS providers.
- Temporary executor-boundary logging on `a2b4cc34` also observed exactly one
  entry for each of twelve actual high-level scopes: Faer/MKL, CPU budgets 1/4,
  and the three matmul shapes below. Each scope repeats eager primal/JVP/VJP
  and prepared primal/JVP/VJP twice. Diagnostic edits were removed afterward;
  no public instrumentation API or production logging was added.
- Independent scalar-loop references verify eager and prepared trace primal,
  JVP, VJP and stored-gradient backward for 2x2 batch16, 4x4 batch3 and 16x16
  batch1 matmul. Elementwise
  square/reduction AD covers lengths 3, 64 and 1024. Wrong witness/provider,
  wrong prepared-runtime identity, nested entry, errors, unwinding, child
  workers, provider exclusion and post-scope execution are covered.
- Host checks passed all 543 CPU unit tests. The MKL devcontainer passed all
  559 CPU unit tests and 355 AD integration tests with Rust 1.98.1; focused
  release checks passed with effective MKL thread counts 1 and 4. The public
  example and doctest run. Host Rust 1.96/1.97 emits a different caret layout
  for one existing compile-fail snapshot, also on the unchanged base. The
  checked-in snapshot is correct for Rust 1.98.1 and remains unchanged.
- Linux MKL source-linked tests require the installed MKL core/thread and
  Intel OpenMP shared libraries to be loaded together. Test runs used those
  libraries from the same oneAPI installation, not an alternative provider.
- This establishes execution/lifecycle behavior, not a latency speedup.
  Downstream idle-core MKL benchmarks must use the actual merged revision and
  retain all 1T/4T outcomes, including regressions and inconclusive results.
