# Local PR Gate Design

## Scope

Local pull-request preparation should catch ordinary Rust correctness failures
without requiring contributors to repeat the repository's release, coverage,
backend-matrix, or GPU validation before every push. Hosted CI remains the
owner of those comprehensive checks.

This change is local-development policy only. It does not change GitHub Actions,
RunPod, CI caches, required checks, or the release profiles used by hosted CI.

## Local profile

The workspace defines a `local-gate` Cargo profile with non-optimized debug
semantics but without debug symbols or incremental state:

```toml
[profile.local-gate]
inherits = "dev"
opt-level = 0
debug = 0
debug-assertions = true
overflow-checks = true
incremental = false
```

This preserves debug assertions and overflow checks while avoiding the disk
cost of full debug information. Disabling incremental compilation also makes
the profile compatible with developer-local sccache. The profile is not a
performance benchmark profile and must not be used for performance claims.

## Local PR contract

`scripts/ci/run_profile.py local-gate` runs workspace nextest tests and
workspace doctests with the Cargo `local-gate` profile. The existing
`scripts/check-pr-fast.sh` profile delegation remains the entry point:

```bash
bash scripts/check-pr-fast.sh \
  --coverage-reviewed \
  --ci-profile local-gate
```

Local release validation is required only for performance-sensitive changes,
release-only bug reproduction, unsafe or optimization-sensitive work, or an
explicit maintainer request. Hosted CI owns full release workspace tests,
coverage, BLAS/backend variants, documentation builds, and GPU validation.

## Local sccache policy

Before a workspace-wide local Rust build, agents check whether `sccache` is
available and enabled. If not, they recommend the documented developer-local
setup once. Agents do not install software, edit global Cargo configuration,
or enable a remote cache without explicit approval.

Each developer owns an independent local cache. The repository does not define
or recommend a shared remote sccache. Correctness checks must work on a cache
miss, and clean-build measurements must disable sccache and state their cache
condition.

## Verification

Repository tests lock the exact local profile commands and Cargo profile
settings. CI helper tests, formatting, profile dry-run, and the local fast
preflight validate the implementation. No hosted workflow file changes in this
work.
