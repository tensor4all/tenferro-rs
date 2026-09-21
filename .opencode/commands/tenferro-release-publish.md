---
description: Release a new tenferro-rs version - bump the workspace version on main, tag the merged commit, publish crates to crates.io in dependency order, and verify provenance. Maintainer-only; publishing is irreversible.
---

Use `$ARGUMENTS` as the target version or initial request if present.

Follow `ai/contribution-workflows/release-publish.md` as the canonical
workflow. Read it fully before acting. Derive and present its SemVer proposal
before editing; stop for explicit confirmation when the requested target
differs.

Hard invariants (abort the release if any would be violated): publish only
from a pushed, tagged, main-lineage commit; land the version bump on `main`
before publishing; never edit manifests at publish time (fix on `main` and
re-tag instead); git-pinned workspace dependencies must resolve, at publish
time, to a crates.io release whose contents match the pinned revision — version
existence alone is not enough, because `scripts/check-git-pin-content.py`
compares the pinned tree and dependency wiring against the crate archive Cargo
resolves and only reviewed entries in
`scripts/git-pin-content-exceptions.toml` may deviate.

Proceed through the version-bump PR and tag. At Phase 3, execute publication
yourself, one irreversible step at a time, after the maintainer approves that
step in the conversation; they never copy commands. Run the preflight without
`--execute` first, present exactly what it will publish (including
per-package new-package approval), and only then run
`python3 scripts/release-publish.py X.Y.Z --execute` with exactly those
approvals. Stop and report when any invariant fails, and never work around a
failed check; a maintainer who prefers to run it personally can generate the
guarded handoff script with
`python3 scripts/release-publish.py X.Y.Z --generate-script PATH` that re-runs
the preflight and requires one exact lowercase `y` at a TTY before `--execute`.
Phase 3 validation is change-aware (`scripts/release-validation-policy.py`); a
rerun is skipped only when the exact-SHA CI check passes
(`verify_release_ci` in `scripts/release-publish.py`). After publication,
perform the canonical post-publish verification.
