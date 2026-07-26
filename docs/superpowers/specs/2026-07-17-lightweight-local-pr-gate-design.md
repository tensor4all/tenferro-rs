# Lightweight Local Pull-Request Gate Design

## Scope

Local development and pull-request preparation should optimize for short edit-test
cycles. GitHub-hosted CI remains the comprehensive correctness gate. A contributor
must not rebuild and execute the complete Rust workspace locally merely to open or
update a pull request.

This change updates Cargo development profiles, the local PR helper scripts,
repository policy, and their contract tests. It does not reduce hosted CI test,
coverage, backend, documentation, or GPU requirements.

## Cargo profile contract

The default development and test profiles use:

```toml
opt-level = 0
debug = 0
debug-assertions = true
overflow-checks = true
incremental = true
```

This is the default for ordinary local builds, focused tests, and AI-assisted
edit-test loops. Developers who need debugger symbols may override `debug` for an
individual command through Cargo profile environment variables. Release mode is
reserved for benchmarks, performance validation, release-only failures, unsafe or
optimization-sensitive changes, and explicit maintainer requests.

The custom non-incremental `local-gate` profile is removed. Comprehensive hosted
CI builds use profiles with incremental compilation disabled. Existing release
profiles already have that Cargo default; future unoptimized hosted-CI profiles
must set `incremental = false` explicitly.

## Local gate contract

`scripts/check-pr-fast.sh` classifies the complete local diff, including committed,
staged, unstaged, and untracked paths, using the same conservative path policy as
hosted CI.

### Code or unknown changes

The gate runs whitespace checks, relevant documentation snippet checks, the
CI-parity formatting and clippy command groups, and one or more
contributor-selected focused verification commands. At least one `--test
COMMAND` or `--ci-profile NAME` is required. Focused test commands use the
default incremental dev/test profiles unless the contributor deliberately
requests another profile. The formatting and clippy command groups cover the
root workspace and the standalone tropical and sparse extension manifests
because those manifests are outside the root Cargo workspace.

Manual coverage review remains required for code or unknown changes. The review
confirms that new branches, errors, dtypes, ranks, shapes, devices, and AD paths
have suitable focused coverage or an explicit reason to rely on hosted coverage.

### Documentation-only changes

The gate runs whitespace checks and the documentation snippet synchronization
checks selected by the changed paths. It does not require Rust compilation,
focused Rust tests, or a code-coverage review acknowledgement. Hosted CI continues
to run the complete documentation lane selected by the shared change policy.

### CI-only changes

The gate runs whitespace and formatting checks as applicable and requires a
focused CI helper verification command. Changes to CI control-plane code are not
treated as documentation-only. Hosted CI continues to run CI configuration checks
and any GPU control-plane checks selected by policy.

Empty or unrecognized diffs remain conservatively classified as code changes.

## Pull-request creation

`scripts/create-pr.sh` accepts repeatable `--test COMMAND` arguments and forwards
them to `scripts/check-pr-fast.sh`. It no longer invokes the workspace-wide
`local-gate` profile. It still requires a clean named branch, checks repository
settings, runs the repository-rules review on the committed head, pushes, creates
the PR, enables auto-merge for ordinary PRs, and monitors required checks.

Generated PR verification text records the focused commands actually supplied.
For documentation-only changes it records the lightweight documentation gate and
does not claim that Rust workspace tests ran locally.

The repository-remediation workflow adopts the same local contract: focused
incremental tests before PR creation and comprehensive validation in hosted CI.

## Hosted CI ownership

GitHub-hosted CI remains responsible for:

- complete workspace tests and doctests;
- coverage generation and threshold enforcement;
- BLAS and extension feature variants;
- formatting and clippy replay and documentation-site validation;
- CUDA/PJRT archive and GPU execution; and
- clean-build behavior with incremental compilation disabled.

Documentation-only pull requests continue to bypass Rust and GPU lanes through the
existing change policy while running the hosted documentation lane. The local
classification and hosted classification must agree for representative code,
documentation-only, CI-only, empty, and unknown path sets.

## Rule ownership

The tenferro implementation PR updates `AGENTS.md`, `CONTRIBUTING.md`, the previous
local-gate design record, repository-remediation guidance, and source-contract
tests. The old design record remains historical and receives a supersession note
rather than being rewritten as if it had always described the new policy.

A separate documentation-only PR to `tensor4all-agent-rules` records the durable
cross-repository principle:

- ordinary local development and focused tests use non-release incremental builds;
- local PR preparation is proportional to the changed surface;
- comprehensive clean, non-incremental validation belongs to hosted CI; and
- release mode is required only when its optimization semantics are relevant.

Repository-local policy remains authoritative when a project needs stricter gates.

## Verification

Contract tests cover:

- exact default dev/test Cargo profile fields;
- removal of the `local-gate` profile and profile runner entry;
- focused-test requirements for code and CI-only changes;
- documentation-only bypass of Rust tests and coverage acknowledgement;
- forwarding and PR-body recording of focused commands;
- repository-remediation guidance; and
- unchanged hosted CI path classification and required-lane behavior.

The implementation is verified with formatting, CI helper unit tests, shell syntax
checks, representative dry runs for code/docs-only/CI-only diffs, and focused Cargo
tests. The PR itself relies on hosted CI for the comprehensive workspace gate.
