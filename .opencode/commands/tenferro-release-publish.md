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
re-tag instead); git-pinned workspace dependencies must pin revs whose
declared versions exist on crates.io.

Proceed through the version-bump PR and tag, then stop after validation; a
human maintainer runs Phase 3 publication from the tag. The maintainer can
generate a guarded handoff script with
`python3 scripts/release-publish.py X.Y.Z --generate-script PATH` that re-runs the
preflight and requires one exact lowercase `y` at a TTY before `--execute`;
agents never run publication and never type that confirmation. This
target-neutral example intentionally omits `--approve-new-package`; add one
`--approve-new-package PACKAGE` only for each genuinely new package named by
the user's explicit approval. Phase 3
validation is change-aware (`scripts/release-validation-policy.py`); a rerun
is skipped only when the exact-SHA CI check passes
(`verify_release_ci` in `scripts/release-publish.py`). After human
publication, perform the canonical post-publish verification.
