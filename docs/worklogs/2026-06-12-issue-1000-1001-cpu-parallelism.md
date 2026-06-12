# Issues 1000 And 1001 CPU Parallelism Remediation

## Session Summary

This slice resolves GitHub issue #1001 and narrows the CPU-performance part of
the #1000 umbrella audit. It wires tenferro-owned CPU tensor kernels to the
existing `CpuContext` parallelism contract, fixes the cached faer GEMM helper
so it enters the owned Rayon pool, and updates active docs/rules that still
described the older no-owned-pool design.

The PR should use `Closes #1001` and `Refs #1000`. #1000 remains an umbrella
backlog with GPU placement, AD oracle, public API, broader materialization, and
docs/tooling items that need separate focused verification.

## Context Read

- `AGENTS.md`
- `CONTRIBUTING.md`
- `REPOSITORY_RULES.md`
- `ai/contribution-workflows/repository-remediation.md`
- GitHub issues #1000 and #1001
- Online tensor4all shared rules:
  - `rules/index.md`
  - `rules/common/repository.md`
  - `rules/common/performance.md`
  - `rules/common/docs-and-tests.md`
  - `rules/rust/index.md`
  - `rules/rust/performance.md`
- `crates/tenferro-cpu/src/context.rs`
- `crates/tenferro-cpu/src/backend.rs`
- `crates/tenferro-cpu/src/elementwise.rs`
- `crates/tenferro-cpu/src/analytic.rs`
- `crates/tenferro-cpu/src/reduction.rs`
- `crates/tenferro-cpu/src/structural.rs`
- `crates/tenferro-cpu/src/indexing.rs`
- `docs/guides/parallelism-and-caching.md`
- `docs/design/tensor-prims.md`
- `docs/design/exec-session.md`
- `docs/design/dot-general-overhead.md`
- `docs/design/contraction-pipeline.md`

## Reference Code

- strided-rs `main` at `71bdd913158a87437e51f4f9b69cba4cac6f5082`.
- `strided-kernel` at that revision has a `parallel` feature, Rayon-backed
  threading helpers, `MaybeSendSync` / `MaybeSync` bounds for parallel builds,
  and large-array tests that exercise parallel `map_into` paths.
- `strided-einsum2/parallel` propagates `strided-kernel/parallel` and
  `strided-perm/parallel`.

## Classification Ledger

| Finding | Classification | Current action |
| --- | --- | --- |
| `strided-kernel` lacked the `parallel` feature in the tenferro workspace dependency graph | Auto Fix / Fixed | Updated the strided-rs pin to current `main` and enabled `strided-kernel/parallel` in the workspace dependency. Added a source-contract test that checks the manifest wiring. |
| `cpu-faer` did not propagate `strided-einsum2/parallel` | Auto Fix / Fixed | Added `strided-einsum2/parallel` to `cpu-faer` and to provider feature aliases that enable `strided-einsum2`. Verified with `cargo tree -p tenferro-cpu -e features -i strided-kernel`. |
| Cached faer GEMM helper did not enter the owned `CpuContext` Rayon pool | Auto Fix / Fixed | Changed `install_with_pool_and_gemm_cache` to run through `ctx.install(...)`. Added a regression test that observes the configured non-ambient Rayon thread count. |
| Elementwise, analytic, reduction, and structural materialization kernels delegated to strided-kernel but could not compile with `parallel` bounds | Auto Fix / Fixed | Added the necessary `Send` / `Sync` bounds to sealed CPU scalar and private helper surfaces so the parallel strided-kernel API is actually compiled. |
| `embed_diagonal`, triangular masks, and indexing-family kernels remain dedicated sequential loops | Intentional Sequential / Documented | Added nearby source comments explaining that these loops do not yet map to a strided-kernel or backend-native parallel primitive. They still run inside `CpuContext::install`, so a future parallel implementation can use the same policy. |
| BLAS/LAPACK threading | Provider-owned | Left unchanged. Docs now distinguish provider-owned BLAS/LAPACK threading from Rayon-backed tenferro/faer work. |
| Active docs said `CpuContext` does not own a Rayon pool | Auto Fix / Fixed | Updated active guide/design docs to match `CpuContext` ownership and session entry behavior. |
| #1000 AD oracle/support coverage | Verify First / Out of scope for this PR | Full-pivot LU oracle coverage landed in #1016. Broader oracle/support alignment remains outside this CPU-parallelism slice. |
| #1000 CUDA placement diagnostics | Verify First / Deferred | Not touched; needs CUDA-specific behavior tests. |
| #1000 public API and extension-boundary panic risks | Partially fixed by #1015 / Deferred here | Not touched in this PR. |
| #1000 broader performance/materialization risks | Partially narrowed | This PR fixes the concrete CPU parallelism source-risk tracked as #1001. Other performance/materialization findings need focused tests or benchmarks. |
| #1000 broader docs/tooling drift | Partially narrowed | This PR fixes stale `CpuContext` parallelism docs. Snippet/API tooling expansion remains deferred. |

## Decisions Made

- Used the current strided-rs `main` revision because it contains the parallel
  feature wiring and Rayon-backed strided-kernel implementation required by
  #1001.
- Kept `CpuContext` as the only tenferro-owned CPU thread policy source.
- Kept faer using `Par::rayon(0)` only after entering `CpuContext::install`,
  so faer joins the backend-owned Rayon pool rather than selecting a separate
  pool or thread count.
- Left BLAS/LAPACK provider threading provider-owned.
- Did not parallelize the indexing-family and triangular/embedding loops in
  this PR. Their indexing patterns need separate design or backend-native
  helpers before parallelization would be reviewable.
- Did not close #1000 as a whole because the remaining findings are unrelated
  verify-first or design-gated slices.

## Verification Performed

- RED: `cargo test -p tenferro-cpu cpu_tensor_kernel_parallel_features_are_wired`
  failed before the implementation because `strided-kernel/parallel` was not
  enabled.
- RED: `cargo test -p tenferro-cpu cached_faer_gemm_pool_helper_enters_owned_rayon_pool`
  failed before the implementation because the cached faer helper observed the
  ambient Rayon pool instead of the configured `CpuContext` pool.
- GREEN: `cargo test -p tenferro-cpu`
- Feature graph check:
  `cargo tree -p tenferro-cpu -e features -i strided-kernel`
- GREEN: `cargo fmt --all --check`
- GREEN: `cargo clippy --workspace --all-targets -- -D warnings`
- GREEN:
  `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- GREEN: `cargo test --workspace --release`
- GREEN: `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- GREEN: `python3 scripts/check-coverage.py coverage.json`
- GREEN: `cargo doc --workspace --no-deps`
- GREEN: `python3 scripts/check-doc-snippets.py --check`
- GREEN: `python3 scripts/check-docs-site.py`
- GREEN:
  `cargo test -p tenferro-cpu --test inject_tests --release --no-default-features --features "cpu-blas,provider-inject"`
- GREEN: `git diff --check`

## Remaining Risks

- No strict speedup assertion was added. The regression coverage verifies
  feature wiring and pool entry; performance benchmarking should be done with
  pinned thread counts when reviewing actual speedups.
- Dedicated indexing and triangular/embedding loops remain sequential by
  design for this PR.
- #1000 remains open for GPU, AD, public API, and broader performance/docs
  follow-up slices.
