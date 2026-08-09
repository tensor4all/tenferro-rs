# Release And Publish Workflow

Maintainer workflow for releasing a new tenferro-rs version to crates.io.
Publishing is irreversible: published versions cannot be replaced, only
yanked. Only maintainers with crates.io ownership run Phase 3.

## Why This Workflow Exists

The v0.2.0 release (2026-06-28) was published from a local-only branch
(`release/v0.2.0`) with post-hoc dependency rewrites. The branch was not
pushed, not tagged, and not merged, so `main` kept version 0.1.0 and the
published source had no corresponding commit on GitHub until cleanup a week
later. The invariants below make that failure mode structurally impossible.

## Invariants

Violating any of these aborts the release.

1. Publish only from a commit that is pushed to `origin`, tagged `vX.Y.Z`,
   and reachable from `main`.
2. The version bump lands on `main` before anything is published.
3. No manifest edits at publish time. The tagged tree must be publishable
   as-is; if a crate needs an edit to publish, abort, fix on `main`, and
   re-tag as a new patch version.
4. Every git-pinned dependency in `[workspace.dependencies]` must pin a rev
   whose declared `version` exists on crates.io, because `cargo publish`
   strips the `git` source and keeps only `version`.

## Phase 0 — Preconditions

- Fresh worktree of the latest `origin/main`; clean tree; CI green on
  `origin/main`.
- Pick the new version `X.Y.Z`. Pre-1.0 convention: breaking changes bump
  the minor version; compatible fixes bump the patch version.
- Run `python3 scripts/check-publish-layout.py` and fix findings first.

## Phase 1 — Version-Bump PR To Main

1. Branch `release/vX.Y.Z-prep` from `origin/main`.
2. Bump `[workspace.package] version` in the root `Cargo.toml`.
3. Bump every internal cross-crate requirement in `crates/*/Cargo.toml` to
   the same version. These look like
   `tenferro-foo = { path = "../tenferro-foo", version = "X.Y.Z", ... }`
   and MUST match the new workspace version, or dependency resolution
   fails. Find them with:

   ```bash
   grep -rn 'tenferro-.*path = "\.\./' crates/*/Cargo.toml
   ```

   (With `cargo-edit` installed, `cargo set-version --workspace X.Y.Z`
   performs steps 2-3 in one command.)
4. Verify locally:

   ```bash
   cargo metadata --format-version 1 > /dev/null   # resolution sanity
   python3 scripts/check-publish-layout.py
   ```

   then run the full pre-push checklist from `AGENTS.md`. Note:
   `cargo publish --dry-run` works only for crates whose internal
   dependencies are already on the registry at the new version, so deep
   crates are verified live in Phase 3; that is expected.
5. Open the PR to `main` and merge it through the normal auto-merge flow.

## Phase 2 — Tag

1. `git fetch origin main` and identify the merged bump commit.
2. `git tag vX.Y.Z <merged-commit> && git push origin vX.Y.Z`.
3. Recommended: `gh release create vX.Y.Z --generate-notes`.

## Phase 3 — Publish From The Tag

1. `git worktree add <fresh-path> vX.Y.Z`; confirm the tree is clean and
   `python3 scripts/check-publish-layout.py` passes unchanged.
2. Publish publishable crates in dependency order. As of 2026-08 the order
   is:

   ```text
   tenferro-core-ops
   tenferro-internal-extension-macros
   tenferro-tensor-core
   tenferro-tensor
   tenferro-internal-cpu-kernels
   tenferro-internal-ops
   tenferro-runtime
   tenferro-cpu
   tenferro-gpu
   tenferro-xla
   tenferro-ad
   tenferro-einsum
   tenferro-fft
   tenferro-linalg
   ```

   Crates with `publish = false` (currently `tenferro-tutorial-code` and
   standalone examples under `ext/`) are skipped. `t4a-tblis-src` is not part
   of this order until a maintainer explicitly approves publishing that new
   package. When workspace membership or dependencies change, recompute the
   order from `cargo metadata` (topological sort of workspace members over
   their `tenferro-*` dependencies) instead of trusting this list.
3. For each crate run `cargo publish -p <crate>`. A dependent crate fails
   until the registry index has its dependencies; wait roughly 30 seconds
   and retry before treating a failure as real.
4. If any crate cannot publish without a manifest change: stop publishing,
   fix the manifest on `main` through Phase 1, and restart at Phase 2 with
   the next patch version. Crates already published stay published; the
   remaining crates ship only at the new version, so prefer aborting before
   any crate has been published when possible.

## Phase 4 — Verify And Close

1. For every published crate, confirm crates.io reports
   `max_version == X.Y.Z`.
2. Fetch one published crate's source (registry cache under
   `~/.cargo/registry/src/` after any consumer build, or the `.crate` file)
   and confirm `.cargo_vcs_info.json` records the tagged commit, and that
   the commit is on `origin` and reachable from `main`.
3. Remove the release worktree. Update downstream pins and announcements
   (Matrix, blog for major releases) as appropriate.
