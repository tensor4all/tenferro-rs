# CI Cache Trust And Immutable GPU Artifact Reuse

Status: active. Implements issue #1403 under the #1401 CI performance
umbrella. Companion contract tests live in
`scripts/ci/tests/test_workflow_contracts.py` and
`scripts/ci/tests/test_find_archive_artifact.py`.

## Threat model

GitHub Actions caches are shared mutable state. A cache entry saved into the
default-branch scope is readable by every later run, including trusted RunPod
GPU runs that execute restored binaries on a paid self-hosted pod. Two paths
could publish attacker-influenceable bytes into that scope if left open:

1. A job that **executes PR-controlled code** while holding a cache-write
   credential. `runpod-gpu-test.yml` runs on the `workflow_run` event, so
   `github.ref` is `refs/heads/main` and any cache it saved would land in the
   privileged main scope — while its `cuda-archive` job compiles
   PR-controlled sources (`build.rs`, proc macros) that can execute arbitrary
   code inside the job.
2. A job **on the self-hosted RunPod pod**. The pod executes the checked-out
   ref as root in the same container as the runner process, so any credential
   visible to the runner is visible to test code.

`pull_request`-event workflows are not part of this problem: GitHub isolates
their cache writes to the PR's own merge-ref scope, which cannot be read from
main-scope runs, and their workflow definitions never receive RunPod secrets.

## Namespace and role model

| Actor | Cache role | Why |
|---|---|---|
| `ci-cache-publish.yml` (push to main, schedule, dispatch from main) | **Only shared-cache writer** | Definition and built code are both default-branch content |
| `runpod-gpu-test.yml` `cuda-archive` | Reader only | Builds PR-controlled code in the main-scoped context |
| `runpod-gpu-test.yml` `run-gpu-tests` | Reader only | Runs on the self-hosted pod |
| `ci.yml` / `ci-pr-workspace-tests.yml` PR jobs | Platform-isolated writers | Saves go to the PR merge-ref scope, never to main scope |
| `ci.yml` / `ci-pr-workspace-tests.yml` push-to-main jobs | Trusted writers (CPU lanes) | Push runs build default-branch code |

Attacker-controlled contents cannot reach the privileged namespace because
the only jobs holding a write-capable posture in the main scope never execute
non-default-branch code, and every job that does execute PR code or run on
the pod uses `actions/cache/restore` plus rust-cache `save-if: false`, with
read-only workflow permissions (`cuda-archive` adds `actions: read` only, for
artifact lookup). These invariants are contract-tested.

## Key derivation

The CUDA/PJRT archive key is derived from material compilation inputs only:

- `Cargo.lock`, every `Cargo.toml`, `src/**`, `tests/**`, `examples/**`,
  `benches/**`, and every `build.rs` (hashed) — everything
  `cargo nextest archive` compiles, including required-feature CUDA
  examples that no non-GPU lane checks,
- `scripts/ci/**`, `.cargo/**`, and `rust-toolchain*` (hashed) — these are
  executed or read from the PR checkout during the archive build, so a key
  match must prove they were identical; without this, a PR that only edits
  an install script could poison an artifact under an unchanged name,
- the Rust toolchain version (`rustc -V`),
- the cudarc binding and PTX toolkit versions (explicit key prefix),
- a manually bumped `vN` component covering the archive build commands and
  toolkit selection themselves.

Workflow YAML is deliberately excluded: PR-triggered runs always execute the
default-branch workflow definition, so hashing it would only cause unrelated
invalidation. (`runpod_config.json` is hashed as part of `scripts/ci/**`;
the occasional spurious rebuild is accepted in exchange for the simpler
"hash everything the checkout contributes to the build" rule.) The consumer
(`runpod-gpu-test.yml`) and publisher (`ci-cache-publish.yml`) compute the
key with byte-identical expressions over an identical checkout layout;
a contract test compares the two lines.

The pod-side cuTENSOR and CUDA-runtime-tree caches key on OS, architecture,
component version, and a manual `vN`. Their paths live under the fixed root
`/opt/tenferro-ci` so absolute paths restored from hosted-runner saves line
up on the pod.

Every cache namespace was rotated (`vN` bumps across the archive, Rust
build, runtime-tree, and cuTENSOR keys) when the trusted publisher took
ownership: entries under the previous names could have been populated while
write-capable jobs still built PR code, so consumers must never restore
them.

## Immutable GPU artifact reuse across retries

Every `cuda-archive` run uploads the two nextest archives as a per-run
artifact whose **name is the content key**. A GPU allocation retry (a fresh
`workflow_dispatch` through `recover_runpod_pr.py`, or any rerun with
unchanged sources) resolves its own content key, then asks
`scripts/ci/find_archive_artifact.py` for an existing unexpired artifact with
that exact name before building anything.

The finder only accepts artifacts whose producing run:

- belongs to this repository (`head_repository` check),
- ran a trusted workflow file (`runpod-gpu-test.yml` or
  `ci-cache-publish.yml`),
- was triggered by an event whose definition comes from the default branch
  (`workflow_run`, `workflow_dispatch`, `push`, `schedule`) — never
  `pull_request`, whose definitions are PR-controlled,
- for directly-triggered events (`workflow_dispatch`, `push`, `schedule`),
  ran on the default branch — a dispatch aimed at another ref is rejected,
  and the `ci-cache-publish.yml` jobs additionally refuse to run at all off
  `refs/heads/main`.

Name-collision attacks fail on the producer check; content-substitution
attacks fail because the key is computed by the trusted workflow from its own
pinned checkout, so a name match implies identical build inputs. Reuse is an
optimization: any lookup or download failure falls back to a fresh build.

On success the archives are re-uploaded to the current run so the pod's
`download-artifact` step and per-run retention semantics are unchanged, and
no Cargo compilation happens on the retry path.

## Observability and bounds

- The archive job logs its source (cache hit / reused artifact with run id /
  fresh build) plus archive sizes and free disk.
- The pod logs cuTENSOR and runtime-tree cache hit state before falling back
  to direct downloads.
- The publisher logs whether a key was already published (no-op) or built.
- Artifacts retain for 7 days; caches use GitHub's 7-day LRU eviction with a
  weekly scheduled publish refresh. Denied writes cannot occur silently:
  write paths are removed from untrusted jobs entirely, and the contract
  tests fail if one is reintroduced.

## Residual risks

- The `vN` key components must be bumped by hand when archive build commands
  or toolkit selection change; a forgotten bump can reuse a stale archive
  until any source file changes.
- `actions: read` on `cuda-archive` exposes read access to run metadata and
  artifacts of this repository to PR-controlled build code. This grants no
  write capability and the repository is public.
- End-to-end cache-hit and retry behavior on paid RunPod hardware can only be
  demonstrated in live CI runs.
- The manual PR-recovery path (`workflow_dispatch` with `pr_number`) builds
  and runs PR-authored code in a run whose ref is `main`. If GitHub's cache
  service granted that run an implicit main-scope cache-write credential
  independent of `GITHUB_TOKEN` permissions, PR build scripts could attempt
  cache poisoning from inside the job. Observed behavior contradicts this:
  the read-only-token runs of this workflow had their cache saves rejected
  (#1403), indicating cache writes follow token permissions here. This
  pre-existing recovery path is tracked under the #1322 hardening umbrella;
  revisit if GitHub's cache-credential model changes.
