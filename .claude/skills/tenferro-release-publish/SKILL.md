---
name: tenferro-release-publish
description: Release a new tenferro-rs version. Use for workspace version bumps, tagging, crates.io publication in dependency order, and post-publish provenance verification. Publishing is irreversible and requires maintainer crates.io ownership. Do not use for ordinary feature or bug-fix PRs, and never publish from an unpushed or untagged commit.
---

# Tenferro Release And Publish

Follow `ai/contribution-workflows/release-publish.md` as the canonical
workflow. Read it fully before acting.

Hard invariants (abort the release if any would be violated):

1. Publish only from a pushed, tagged, main-lineage commit.
2. The version bump merges to `main` before anything is published.
3. No manifest edits at publish time; fix on `main` and re-tag instead.
4. Git-pinned workspace dependencies must pin revs whose declared versions
   exist on crates.io.

Keep the interaction incremental:

1. Derive and present the canonical workflow's SemVer proposal before editing;
   stop for explicit confirmation when the requested target differs.
2. Phase 1: version-bump PR (workspace version plus every internal
   cross-crate `version = "..."` requirement), curated user-facing GitHub
   release notes, and the full pre-push checklist. Release notes describe only
   shipped outcomes, never experiments or implementation history.
3. Phase 2: tag the merged commit, push the tag, and create the GitHub release
   from the reviewed notes file.
4. At Phase 3, stop after validation; a human maintainer runs Phase 3
   publication from the tag. The maintainer can generate a guarded handoff
   script with
   `python3 scripts/release-publish.py X.Y.Z --generate-script PATH`
   that re-runs the fail-closed preflight and requires one exact lowercase
   `y` at a TTY before invoking the helper with `--execute`; agents never run
   publication and never type that confirmation. Phase 3 validation is
   change-aware (`scripts/release-validation-policy.py`): helper-only or
   publication-metadata-only diffs run focused lanes, and a rerun is skipped
   only when the exact-SHA CI check passes
   (`verify_release_ci` in `scripts/release-publish.py`); Rust source or
   ambiguous diffs run the full validation.
5. After human publication, Phase 4 verifies crates.io versions and
   `.cargo_vcs_info.json` provenance, then cleans up.
