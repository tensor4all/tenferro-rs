# 2026-07-04 Release/publish workflow and adapters

## Summary

Added `ai/contribution-workflows/release-publish.md` as the canonical
maintainer release workflow, with thin adapters
(`tenferro-release-publish`) under `.claude/skills/`, `.agents/skills/`,
`.kimi/skills/`, and `.opencode/commands/`, and registered the workflow in
`AGENTS.md` and `ai/README.md`.

## Context

Investigation of the crates.io v0.2.0 release found it was published
(2026-06-28) from a local-only branch `release/v0.2.0` with publish-time
dependency rewrites; the branch was neither pushed, tagged, nor merged, so
`main` stayed at 0.1.0 and the published source had no commit on GitHub. As
cleanup, the branch was pushed, `v0.2.0` was tagged on the publish commit
(`cee2a6d4`), and a companion PR reflects version 0.2.0 on `main` via an
`-s ours` lineage merge.

## Chosen design

- Four-phase flow: version-bump PR to `main` → tag the merged commit →
  publish from a worktree of the tag in dependency order → provenance
  verification against `.cargo_vcs_info.json`.
- Invariants forbid publishing from unpushed/untagged commits and forbid
  publish-time manifest edits (fix on `main`, re-tag a patch version).
- The version bump must update both `[workspace.package] version` and every
  internal cross-crate `version = "..."` requirement; missing the latter
  breaks dependency resolution (hit during the companion cleanup PR).

## Residual risks

- The publish order list is a snapshot; the workflow instructs recomputing
  it from `cargo metadata` when workspace membership changes.
- `cargo publish --dry-run` cannot fully verify deep crates before their
  dependencies exist on the registry at the new version; deep crates are
  verified live during Phase 3.
- Test suite not rerun locally for this docs-only branch (code tree
  identical to `origin/main`); PR CI gates the merge.
