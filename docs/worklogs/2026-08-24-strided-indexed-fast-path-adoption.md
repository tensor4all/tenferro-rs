# strided indexed fast-path adoption

## Summary

Advanced all four workspace `strided-rs` dependencies from release commit
`b29e7601ba090aa5eafc65b9bde5d9450282e0d8` to merged commit
`39111bd7b397c54402d1d9370bdd27a6c04023ed`, the squash merge of
[strided-rs PR #236](https://github.com/tensor4all/strided-rs/pull/236).
The upstream change specializes the compact rank-one gather and ordered
additive-scatter layouts used by tenferro's CPU indexed APIs without changing
the public API or the generic fallback.

## Context reviewed

- tenferro issue [#1719](https://github.com/tensor4all/tenferro-rs/issues/1719)
  and its pinned EPYC measurements
- strided-rs issue [#213](https://github.com/tensor4all/strided-rs/issues/213)
  and merged PR #236
- strided-rs `docs/design/erased-execution-policy.md` and
  `docs/worklogs/2026-08-23-issue-213-indexed-fast-paths.md`
- tenferro `AGENTS.md`, `REPOSITORY_RULES.md`, the shared Rust/repository/docs
  rules, and the CPU indexed delegation and test paths

## Decisions

- Pin the merged upstream commit rather than an implementation branch.
- Keep the registry identity at `0.4.0`; the merged revision still declares
  that workspace version for `strided-view`, `strided-traits`, `strided-perm`,
  and `strided-kernel`.
- Update the build-artifact source-contract test together with the workspace
  manifest. No tenferro source, API, feature, or durable architecture change is
  required.
- Do not duplicate the upstream fast path in tenferro. The reusable indexed
  replay remains owned by `strided-rs`.

## Verification

Focused correctness on the candidate dependency graph:

- `python3 -m unittest scripts.ci.tests.test_build_artifact_contracts`: 9 passed
- `cargo test -p tenferro-cpu 'tests::indexing::'`: 28 passed
- `cargo test -p tenferro-cpu gather_delegates_bulk_traversal_to_strided_kernel_plan`: 1 passed
- `cargo test -p tenferro-cpu scatter_negative_start_indices_clamp_like_dynamic_slice`: 1 passed

A release probe of the public `CpuBackend` and direct strided plan at
`N = 262,144` ran sequentially after four-second topology/load gates on an AMD
EPYC 7713P. The accepted one-thread process was pinned to CPU 34 in L3 domain
32-39; the accepted four-thread process was pinned to CPUs 48-51 in L3 domain
48-55. Every selected core was below 2% busy and every sibling in the L3 domain
was below 20% busy before timing.

| context | public gather | public additive scatter | direct gather | direct additive scatter |
|---|---:|---:|---:|---:|
| 1 thread | 0.511 ms | 0.926 ms | 0.394 ms | 0.860 ms |
| 4 threads | 0.279 ms | 0.906 ms | 0.190 ms | 0.643 ms |

The earlier pinned one-thread issue measurements were 6.624 ms for public
gather and 5.320 ms for public additive scatter. The adoption therefore closes
the measured lower-layer replay bottleneck while retaining tenferro's expected
allocation/index-conversion overhead and ordered additive-scatter semantics.
The four-thread direct-scatter timing can benefit from the plan's parallel
operand-copy phase; repeated-index update replay itself remains serial and
order-preserving.

The focused local PR gate passed with coverage explicitly reviewed:

```text
bash scripts/check-pr-fast.sh --coverage-reviewed \
  --test "cargo test -p tenferro-cpu 'tests::indexing::'"
```

This included root and standalone-extension formatting, documentation-snippet
checks, CI-parity clippy for the workspace and standalone extension manifests,
and the 28 focused CPU indexing tests.

The root `Cargo.lock` is intentionally ignored by `.gitignore`; tenferro does
not commit a workspace lockfile. It was regenerated locally and resolved all
four strided packages to `39111bd7`, but it is not a PR artifact. This also
records the evidence for rejecting a review false positive that assumed
`origin/main` tracked the lockfile (`git cat-file -e origin/main:Cargo.lock`
reported no such tracked path).

The exact-candidate repository-rules and independent review results are
recorded in the PR body because adding a commit identity here would make this
file recursively change that identity.

## Residual scope

Arbitrary-rank indexed replay remains tracked by strided-rs #213. Additive
scatter remains deterministic serial replay because repeated indices are
order-sensitive. This pin does not attempt a uniqueness contract or change
scatter to set semantics.
