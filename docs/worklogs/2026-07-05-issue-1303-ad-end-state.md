# Issue 1303 AD end-state work log

Date: 2026-07-05

## Session summary

Issue #1303 moves tenferro eager AD to the same rule-set architecture as
traced AD:

- tidu now provides eager forward and backward trace walkers plus an executor
  hook for recorded-graph linearization caching.
- tenferro eager tensors support `backward_with(seed)`, functional
  `grad`/`vjp`/`jvp`, and `no_grad`.
- Functional eager transforms return ordinary eager tensors and can compose,
  including Hessian-vector products written as `jvp(grad(f))`.
- Eager runtime cache stats now include a bounded Tier-1 AD transform cache for
  repeated linearization of the same recorded eager graph node.
- `tenferro-ad` has a Criterion microbenchmark for the same-tape Tier-1 cache
  hit path, comparing cold cache-cleared backward against warm repeated
  backward.
- CUDA eager reverse-mode smoke coverage was added for supported GPU builds.

## Context read

| Source | Why it was read | Decision impact |
| --- | --- | --- |
| Issue #1303 | Source of the requested AD architecture end state. | Drove the eager `{JVP,VJP,grad}` work, composability tests, cache hook, and docs updates. |
| `AGENTS.md` and `REPOSITORY_RULES.md` | Confirm repository rules for public APIs, docs, coverage, work logs, and tidu updates. | Added API examples, updated architecture/spec/guides, and made the needed tidu change upstream instead of shimming in tenferro. |
| Shared tensor4all rules | Confirm Rust, performance, docs, numerical, benchmark, and workflow expectations. | Kept cache ownership explicit and documented residual cache scope. |
| tidu eager AD code | Locate traversal ownership boundaries. | Added the linearization-cache hook to tidu executors while leaving traversal in tidu. |
| Existing tenferro eager reverse-mode implementation | Find how tensors, gradient slots, metadata, and extension executors are owned. | Reused existing runtime ownership and callback execution instead of adding a second eager AD engine. |
| `docs/spec/ad-contract.md` and `docs/architecture/ad-pipeline.md` | Locate normative AD wording that would become stale. | Updated the one-rule-set contract and eager/traced interpreter split. |

## Decisions made

- **tidu keeps traversal ownership.** Eager forward and backward walkers decide
  trace order and active slots. tenferro supplies concrete execution hooks and
  cache storage.
- **Functional eager transforms record derivative execution.** `grad`, `vjp`,
  and `jvp` do not mutate gradient slots; they return eager tensors that remain
  traceable when the derivative depends on tracked eager values.
- **Stateful backward stays explicit.** `backward()` and `backward_with(seed)`
  are the APIs that accumulate into `grad()` slots.
- **Eager value records bridge saved concrete tensors back to eager tensors.**
  The runtime keeps weak registries by `ValueKey` and materialized tensor
  pointer so derivative execution can recover the original eager trace when
  tidu saved a concrete `Arc<Tensor>`.
- **Synthetic tangent zeros use fresh eager keys.** Reusing the primal input
  key for a missing tangent zero caused later derivative execution to recover
  the zero instead of the saved primal tensor. Fresh keys avoid that collision.
- **Tier-1 cache scope is same-tape recorded graph reuse.** The cache key is a
  recorded graph structural fingerprint plus requested output slots.
  Generalized shape-template caching and whole-program backward caching are
  left as future work.
- **`no_grad` is a thread-local nesting guard.** It suppresses eager operation
  recording while still computing concrete eager values.

## Verification performed

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo clippy --manifest-path ext/tropical/Cargo.toml --all-targets -- -D warnings`
- `cargo test -p tenferro-ad eager_ --release -- --nocapture`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --worktree --output-json /tmp/repository-rules-review-worktree.json`
- `cargo bench -p tenferro-ad --bench eager_ad_transform_cache --no-run`
- `cargo bench -p tenferro-ad --bench eager_ad_transform_cache -- --sample-size 10 --warm-up-time 0.2 --measurement-time 0.2`
  - cold clear-cache backward: 213.79-218.58 us
  - warm cached-linearization backward: 196.66-200.24 us
- `cargo test -p tenferro-ad --features cuda test_gpu_eager_backward_smoke --release --no-run`
- `CUBECL_DEBUG_LOG=0 CUDA_PATH=/usr/local/cuda-12.6 LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH cargo test -p tenferro-ad --features cuda test_gpu_eager_backward_smoke --release -- --nocapture`
- tidu PR #44 CI passed before merging into the tidu AD-rule branch:
  - coverage
  - docs-site
  - nextest on macOS and Ubuntu
  - rustfmt
- tidu PR #33 merged the AD-rule branch to `main`; tenferro now pins tidu
  `998dbac4ca24442433ab9a52a182040dda2d8eea`.

## Remaining risks

- The eager AD transform cache is intentionally a per-recorded-graph Tier-1
  cache. It does not yet provide whole-program backward caching or reuse across
  structurally equivalent tapes.
- CUDA eager AD coverage is a smoke test, not an exhaustive GPU AD oracle.
- `cargo clippy -p tenferro-ad --features cuda --all-targets -- -D warnings`
  currently reaches existing `tenferro-gpu` CUDA-feature lint findings outside
  this change. The CUDA feature test build and smoke execution pass.
