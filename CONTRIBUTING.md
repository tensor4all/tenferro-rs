<!-- unified contribution policy: ../../docs/contributing/rust-repositories.md in tensor4all/tensor4all-meta -->
# Contributing

Thanks for helping improve tenferro-rs. This file describes the external
contribution path. Repository-specific implementation rules live in
`REPOSITORY_RULES.md` and, for performance contracts, `PERFORMANCE_TIPS.md`.

Governance, maintainer roles, merge authority, and project-direction decisions
are defined in [GOVERNANCE.md](GOVERNANCE.md). The maintainer list is
maintained in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Agent-assisted workflows

Agent-assisted issue and bug-fix PR preparation is supported.

The canonical contribution policy is this file plus `REPOSITORY_RULES.md`.
Repository-local AI workflows are helpers for collecting the right information
and applying the contribution boundary; they do not replace maintainer review,
merge authority, or roadmap decisions.

Supported entry points:

- Codex CLI: use the `tenferro-issue-intake` or `tenferro-bugfix-pr` skill
  from `.agents/skills/`.
- Claude Code: invoke `/tenferro-issue-intake` or `/tenferro-bugfix-pr`; the
  project skills live in `.claude/skills/`.
- OpenCode: invoke `/tenferro-issue-intake` or `/tenferro-bugfix-pr`; the
  project commands live in `.opencode/commands/`.
- Kimi CLI: invoke `/skill:tenferro-issue-intake` or
  `/skill:tenferro-bugfix-pr`; the project skills live in `.kimi/skills/`.

The shared workflow bodies live in `ai/contribution-workflows/`.

Open issues carrying the maintainer-controlled `ai-ready` label have completed
human readiness review and may be selected for autonomous AI implementation.
The label authorizes implementation and preparation of a reviewable pull
request within the accepted issue scope; it does not authorize merge, release,
publication, or bypassing human review and required CI.

Use the issue-intake workflow for bug reports, feature requests, design
discussion issues, and documentation or article topic issues. Use the
bug-fix PR workflow only for fixes to existing intended behavior. If a proposed
bug-fix PR needs a new public API, operation family, backend, dependency,
feature flag, architectural layer, or AD semantics change, move it to an issue
first.

## Bug fixes

Bug reports, minimal reproducers, and regression tests are welcome in issues.

Bug-fix pull requests are welcome from any account when they fix behavior that
is already intended by current docs, specs, or tests. A bug-fix PR should not
introduce a new public API, operation family, backend, dependency, feature
flag, or architectural layer; open an issue for that decision instead.

## Pull requests from any account

Pull requests are accepted from any GitHub account, forks included, and you do
not need collaborator access to open one. A prior issue is recommended rather
than required: it is the best place to agree the API, dependency, backend, and
roadmap decisions before implementation. Maintainers may ask for an issue or a
short design before reviewing a large or risky change, and may close a pull
request that is out of scope for the project, with an explanation of why.

New features, substantial behavior changes, new public APIs, new backends, new
dependencies, and architectural changes are the cases where that issue matters
most. An accepted issue carries the intended behavior, the API, and the
non-goals, and it stays the source of truth for the implementation, the tests,
and the roadmap decision; the project keeps development coherent when the
implementation inside the repository follows that accepted design.

Requests, prototype code, exploratory branches, gists, external repositories,
and focused unit tests are all useful contributions. Link them from the issue
or the pull request and explain what behavior they demonstrate. Prototype code
may inform an implementation when its license and provenance are compatible
with this repository.

The shared tensor4all policy asks for an issue before every non-trivial change
(`docs/contributing/rust-repositories.md` in `tensor4all/tensor4all-meta`).
tenferro-rs keeps that step as a recommendation for external work and lets
maintainer review ask for the issue when a change needs one; the files in this
repository are the operative policy.

## CI on pull requests

A `pull_request` run from a fork receives no repository secrets and a read-only
token. GitHub additionally holds `pull_request` runs from external contributors
until a maintainer approves them: *Settings > Actions > General > Fork pull
request workflows from outside collaborators* is set to the policy
`all_external_contributors`, recorded as `fork_pr_contributor_approval` in
`ai/repo-settings.json` and verified by `scripts/check-repo-settings.sh`. An
unapproved run does not execute, so it cannot consume repository CI or read a
secret, and the required checks stay queued until a maintainer approves the run
on the pull request.

External pull requests also get the deterministic `repository rules review
gate`. That workflow runs on `pull_request_target`, so it cannot rely on the
approval hold above; it is safe for a fork by construction instead. It checks
out the **base** revision and treats pull request content as data, so nothing
from an external branch is executed, it reads no secret, and its token is scoped
to reading contents and writing the review comment. Every secret-bearing path
starts from trusted `main`: RunPod GPU execution authorizes `admin`/`maintain`
actors (`.github/workflows/runpod-gpu-test.yml`), and the cache publisher runs
only on main-scoped triggers. A fork PR never starts RunPod by itself; its GPU
gate passes once a maintainer has reviewed its head and approved that exact
commit to run on RunPod (see "GPU validation of a fork pull request" below). See
[Change-aware CI and trusted RunPod recovery](docs/design/change-aware-ci.md).

## Local PR gate and hosted CI profiles

For code changes, run the non-release local gate with at least one focused test.
The gate also runs the CI-parity formatting and clippy profiles for the root
workspace plus the standalone tropical and sparse extension manifests:

```bash
bash scripts/check-pr-fast.sh \
  --coverage-reviewed \
  --test 'cargo test -p tenferro-tensor checked_convert_follows_dtype_promotion_lattice'
```

The default dev/test profiles use `opt-level=0`, `debug=0`, and
`incremental=true`, with debug assertions and overflow checks enabled. Enable
debug symbols for one command with `CARGO_PROFILE_DEV_DEBUG=1` or
`CARGO_PROFILE_TEST_DEBUG=1` when using a debugger.

For documentation-only changes, run `bash scripts/check-pr-fast.sh`; it does not
compile Rust or require a coverage acknowledgement. CI-only changes require a
focused CI helper command through `--test`.

Hosted CI remains responsible for complete workspace tests, coverage, backend
variants, documentation builds, GPU validation, and clean builds through the
workspace `[profile.ci]` (`opt-level=0`, `debug=0`, `incremental=false`,
`strip="symbols"`). Local `dev`/`test` profiles stay incremental, while the
local gate shares hosted CI's formatting and clippy command groups. Run release locally when
validating performance, a release-only failure, unsafe or optimization-sensitive
behavior, or an explicit maintainer request.

The exact command groups used by hosted CI are available locally:

```bash
python3 scripts/ci/run_profile.py --list
python3 scripts/ci/run_profile.py fmt
python3 scripts/ci/run_profile.py clippy
python3 scripts/ci/run_profile.py workspace-faer
python3 scripts/ci/run_profile.py docs
```

`full` expands every hosted profile once. Use `--dry-run` to inspect commands
without executing them.
`scripts/check-pr-fast.sh` accepts repeatable `--ci-profile NAME` options;
prefer these profiles over copying command lists into local scripts.

### Optional developer-local sccache

Developers who explicitly run non-incremental workspace-wide builds across
multiple worktrees may share compatible compiler outputs through a local
sccache. Keep the wrapper scoped to that explicit command:

```bash
RUSTC_WRAPPER=sccache \
SCCACHE_DIR="$HOME/.cache/tensor4all/sccache" \
SCCACHE_CACHE_SIZE=20G \
  CARGO_INCREMENTAL=0 cargo test --workspace

SCCACHE_DIR="$HOME/.cache/tensor4all/sccache" sccache --show-stats
```

This is an optional local optimization. Do not configure a shared remote cache,
and do not rely on cache hits for correctness. Ordinary focused local
development, including AI-assisted edit-test loops, should use Cargo
incremental compilation through the default dev/test profiles. Do not set
`RUSTC_WRAPPER=sccache` globally for those loops. Disable sccache when measuring
clean-build performance.

Pull-request CI classifies a diff conservatively as code, docs-only, or
CI-only. Docs-only changes run documentation validation. CI-only changes run
CI helper tests and actionlint; changes to the RunPod control plane also keep
the GPU gate. Mixed docs and CI changes run both lightweight suites. Empty
diffs, unknown paths, and compiled-code changes use full validation. Pushes to
`main` always run the comprehensive non-GPU matrix. Required check names remain
present and succeed with an explicit “not required” result when a lane is
irrelevant.

Maintainers can recover a same-repository PR GPU gate from its PR number:

```bash
python3 scripts/ci/recover_runpod_pr.py 1379 --wait
```

The command always dispatches the secret-bearing workflow from trusted
`main`. The workflow resolves and rechecks the open PR head itself; do not run
a PR-branch copy of that workflow with repository secrets. See
[Change-aware CI and trusted RunPod recovery](docs/design/change-aware-ci.md)
for the durable design and trust boundaries.

### GPU validation of a fork pull request

A fork PR's `CI GPU gate` waits for a maintainer's approval. After reviewing the
code at the PR's current head, an admin or maintainer approves that exact
commit:

```bash
python3 scripts/ci/recover_runpod_pr.py <number> --approve-head <full head SHA> --wait
```

Dispatching it is the approval to run the fork's code on RunPod and on the
hosted jobs that build it. The workflow refuses any other head, so a push after
the review needs a new review and approval. The approved run publishes no check
on the fork commit; its `Fork PR GPU verdict #<number> @ <sha>` job is what the
PR's own `CI GPU gate` job reads (`scripts/ci/fork_gpu_gate.py`). That job waits
up to five and a half hours, and an approval that lands later re-runs it. A
change that needs no GPU validation passes without an approval.

For the Actions cache, an approval is close to a merge into `main`. The
approved code runs inside a `workflow_dispatch` run on `refs/heads/main`.
Code that runs there can take the runner's runtime token and create
main-scoped cache entries under keys that later trusted `main` workflows
restore. Cache entries cannot be overwritten, but future keys can be created
in advance. The run also builds the fork's code on the hosted archive job and
on the pod. Before approving, review everything that runs during the build and
the tests, not only the library change:

- `build.rs` files and proc-macro crates;
- new or changed dependencies in `Cargo.lock` (and any `Cargo.lock` under
  `ext/` or `tests/fixtures/`);
- `.cargo/` configuration (build scripts, runners, aliases, `rustflags`);
- `scripts/ci/**`, which the RunPod jobs run from the fork's checkout;
- test code, including integration tests and benchmarks run by the GPU lane.

If you are unsure about any of these, do not approve. Ask for the change to be
split, or check it locally first.

### Local GPU validation instead of the paid gate

The `CI GPU gate` normally needs a RunPod pod. A repository admin or maintainer
with a local NVIDIA GPU can satisfy it instead, with no pod at all:

1. Label the PR `gpu-validated-locally`. For a labelled same-repository PR the
   RunPod workflow runs no job and publishes no `CI GPU gate`, and the
   PR-attached gate waits for the local publication instead.
2. At the exact commit you push, on the GPU host, with a clean worktree:

   ```bash
   CUDA_PATH=/usr/local/cuda-12.6 \
   TENFERRO_LOCAL_PJRT_ROOT=<site-packages with jax_plugins/xla_cuda12, nvidia/cudnn, nvidia/cuda_nvcc> \
     python3 scripts/ci/local_gpu_gate.py run --jobs 4
   ```

   This builds the archives the hosted CUDA archive job builds and runs the same
   lanes: the CUDA and PJRT host and device partitions
   (`scripts/ci/gpu_test_partition.py`) and the CUDA tutorial.
   `TENFERRO_CUDA_ARTIFACT_CACHE` defaults to `~/.cache/tenferro/t4a-cuda-artifacts`
   so later runs reuse compiled kernels. The evidence (GPU, driver and toolkit
   versions, every command, the nextest summaries) is written to
   `target/local-gpu-gate/<sha>.json`. Do not touch the worktree while it runs: a
   changed HEAD or tree voids the evidence. The script refuses to start while
   another process uses the GPU (`--allow-busy-gpu` overrides).
3. After the commit is the PR head:

   ```bash
   python3 scripts/ci/local_gpu_gate.py publish --pr <number>
   ```

   It posts the evidence as a PR comment and dispatches
   `.github/workflows/local-gpu-gate.yml` on `main`. That workflow verifies that
   the dispatching account is an admin or maintainer, the PR is open, from this
   repository and labelled, its head is the tested SHA, and the comment is that
   account's passing record for that SHA, then publishes `CI GPU gate` on the head
   with the evidence as the check body.

The gate is bound to the tested SHA: after any push, run and publish again. Fork
PRs cannot use it; they take the approved RunPod run described above. Removing
the label after a push leaves no publisher;
recover through the paid gate with `scripts/ci/recover_runpod_pr.py`. The
publication must come from an Actions run because branch protection pins
`CI GPU gate` to the GitHub Actions app.

## Prototype code and provenance

By submitting code directly to this repository, you represent that you have the
right to submit it under this repository's license, `MIT OR Apache-2.0`.

If you link prototype code from an issue, clearly state its license if it is
not `MIT OR Apache-2.0` or if the project should not use it as an
implementation reference.

When maintainers implement an accepted issue, they may rewrite the design from
scratch, use the prototype only as a behavioral reference, or take over a
prototype branch. If the project takes over a prototype branch, maintainers
should preserve the contributor's original commits where practical and add new
commits on top. If the final implementation is otherwise based on contributed
prototype code, including by rewriting it manually or with AI assistance, the
project will preserve appropriate copyright notices, license obligations,
attribution, and links to the original prototype or issue discussion.

If your prototype is only meant to illustrate behavior and must not be used as
an implementation reference, say so explicitly in the issue.

## Contributors

Contributors may be listed in `CONTRIBUTORS.md`. Contributor recognition does
not imply maintainer status, merge authority, copyright transfer, or ownership
of project direction.
