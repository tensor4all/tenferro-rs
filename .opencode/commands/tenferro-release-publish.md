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

Proceed phase by phase — version-bump PR, tag, dependency-order publish from
a worktree of the tag, post-publish verification — and confirm with the user
immediately before the first `cargo publish`.
