# Contributing

Thanks for helping improve tenferro-rs. This file describes the external
contribution path. Repository-specific implementation rules live in
`REPOSITORY_RULES.md`.

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

Use the issue-intake workflow for bug reports, feature requests, design
discussion issues, and documentation or article topic issues. Use the
bug-fix PR workflow only for fixes to existing intended behavior. If a proposed
bug-fix PR needs a new public API, operation family, backend, dependency,
feature flag, architectural layer, or AD semantics change, move it to an issue
first.

## Bug fixes

Bug reports, minimal reproducers, and regression tests are welcome in issues.

Bug-fix pull requests from collaborators are welcome when they fix behavior
that is already intended by current docs, specs, or tests. A bug-fix PR should
not introduce a new public API, operation family, backend, dependency, feature
flag, or architectural layer.

This repository restricts pull request creation to collaborators. If you are
not a collaborator, please open an issue with the reproducer, proposed test,
or prototype branch instead of opening an implementation PR.

## Feature requests, prototypes, and implementation ownership

New features, substantial behavior changes, new public APIs, new backends, new
dependencies, and architectural changes must start as an issue before an
implementation pull request is opened.

This boundary is intended to keep development fast and coherent in an
agentic-coding workflow. The project preserves API consistency, internal
architecture, test strategy, backend behavior, and long-term maintainability
best when maintainers and active collaborators own the final implementation
inside the repository.

Requests, prototype code, exploratory branches, gists, external repositories,
and focused unit tests are still useful contributions. Please link them from
the issue and explain what behavior they demonstrate. The issue remains the
source of truth for the accepted API, dependency impact, backend behavior, AD
behavior, tests, implementation plan, and roadmap decision.

Implementation PRs for new features or substantial changes that are opened
before an accepted issue may be closed with a request to continue the
discussion in an issue first.

## Local PR gate and hosted CI profiles

For code changes, run the non-release local gate with at least one focused test:

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
`strip="symbols"`). Local `dev`/`test` profiles stay incremental. Run release
locally when validating performance, a release-only failure, unsafe or
optimization-sensitive behavior, or an explicit maintainer request.

The exact command groups used by hosted CI are available locally:

```bash
python3 scripts/ci/run_profile.py --list
python3 scripts/ci/run_profile.py workspace-faer
python3 scripts/ci/run_profile.py workspace-blas
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
