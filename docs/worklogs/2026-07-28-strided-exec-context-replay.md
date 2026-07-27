# strided ExecContext Replay Adoption

## Summary

Updated tenferro-rs to consume `tensor4all/strided-rs@f1343692f8afbc1d1c1f4614478772ae5683dcbb`
and removed the tenferro-side `ExecContext::serial()` hardcoding from CPU
hot paths delegated through strided erased plans.

## Context Read

- `AGENTS.md`
- `REPOSITORY_RULES.md`
- `docs/guides/parallelism-and-caching.md`
- `crates/tenferro-cpu/src/provider.rs`
- `crates/tenferro-cpu/src/backend.rs`
- `crates/tenferro-cpu/src/exec_session.rs`
- `crates/tenferro-cpu/src/reduction.rs`
- `crates/tenferro-cpu/src/indexing.rs`
- `crates/tenferro-internal-cpu-kernels/src/elementwise.rs`
- strided-rs PR #162 / issue #161

## Decisions

- Keep `CpuExecutionContext` as the single owner of CPU native parallelism
  policy. It now exposes a crate-private `strided_exec_context()` helper that
  maps Rayon-capable `Inner` execution to `ExecContext::max_threads(n)` and all
  other contexts to `ExecContext::serial()`.
- `ExecContext::max_threads(n)` is a strided replay-policy limit, not a
  thread-pool constructor. Tenferro enters its owned `CpuContext` pool first
  via `with_native_parallelism`, then strided observes the bounded policy
  inside that scope.
- Pass that explicit context into erased strided replay for sum/product
  reduction, gather/scatter, dynamic slice/update, and fused elementwise replay.
- Do not use `ExecContext::ambient()`. The CPU backend must not fall back to an
  ambient global Rayon pool.
- Do not keep compatibility wrappers for the old internal helper signatures.
  Tests that call internal helpers directly now pass an explicit serial
  context.
- Do not change faer parallelism. faer continues to receive its policy only
  through `CpuExecutionContext::faer_parallelism()`.

## Residual Risks

- This PR removes tenferro's forced serial replay boundary. It does not claim
  that every strided indexed plan has a partitioned parallel algorithm. Indexed
  gather/scatter and dynamic slice/update still depend on upstream strided plan
  coverage for actual parallel fanout.
- Full benchmark evidence should be collected in the follow-up benchmark
  campaign. This change is a correctness repair for the execution-policy
  plumbing regression, not a performance-claim PR.

## Verification

- `cargo fmt --all -- --check`
- `cargo check -p tenferro-cpu --features cpu-faer,cpu-blas --tests --quiet`
- `python3 -m unittest discover -s scripts/ci/tests -v`
  - 159 CI configuration contract tests passed after updating the pinned
    strided-rs revision expectation.
- `cargo test -p tenferro-internal-cpu-kernels --quiet`
  - 38 unit tests and 19 doctests passed.
- `cargo test -p tenferro-cpu --features cpu-faer --quiet`
  - 488 library tests, 1 install allocation test, 47 integration tests, 2
    provider-boundary allocation tests, and 175 doctests passed.
- `cargo bench -p tenferro-runtime --features __bench_unification_run_compiled_api --bench elementwise_fusion -- --quick`
  - Sanity check only; not a formal paired benchmark campaign.
  - `add_mul/1048576`: 1.4434 ms to 1.4467 ms.
  - `broadcast_mul_add/1024x1024`: 5.3913 ms to 5.6427 ms.
