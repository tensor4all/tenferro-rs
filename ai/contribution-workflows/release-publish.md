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

## Phase 0 — SemVer Proposal And Preconditions

Complete this phase before changing manifests or making any other release edit.

1. Query crates.io for each existing publishable workspace crate and establish
   the latest matching published stable baseline and its provenance tag. An
   approved package that has never been published does not participate in
   baseline agreement. If the published versions or their provenance do not
   agree, stop and resolve the mismatch. A newer tag that was not published is
   an anomaly, not a baseline; stop and resolve it before continuing.
2. Inspect changes actually merged since the matching provenance tag, using
   merged PRs and release evidence. Classify the highest-impact merged change as
   `breaking`, `feature`, or `fix-only`. Accepted issues may explain a
   classification only when linked to merged implementation; unimplemented
   accepted issues do not affect the release classification.
3. Apply this SemVer table to the latest published stable version:

   | Baseline | `breaking` | `feature` | `fix-only` |
   | --- | --- | --- | --- |
   | `0.Y.Z` | `0.(Y+1).0` | `0.(Y+1).0` | `0.Y.(Z+1)` |
   | `X.Y.Z`, `X >= 1` | `(X+1).0.0` | `X.(Y+1).0` | `X.Y.(Z+1)` |

4. Present the baseline and provenance, classified evidence, and one proposed
   target before changing manifests. If the user supplied a different target,
   stop for explicit confirmation and a reason; proceed only after recording
   the confirmed override. Never align independently versioned repositories:
   a dependency's version is not evidence for this repository's target.
5. Use a fresh worktree of the latest `origin/main`; require a clean tree and
   green CI on `origin/main`.
6. Run `python3 scripts/check-publish-layout.py` and fix findings first.

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
4. Record the tag commit SHA (`git rev-parse vX.Y.Z`) and the required CI job
   names (the workspace gate jobs that must pass before publication, e.g.
   `workspace-faer`, `workspace-blas`, `extensions`, `docs`, `coverage`,
   `ci-config`). Phase 3 reuses CI results only for exactly this SHA.

## Phase 3 — Publish From The Tag

Only a human maintainer with crates.io ownership runs the publication helper.
Agents must stop after validation and must never execute a publication.

1. Fetch the remote state and create a detached worktree at the pushed tag:

   ```bash
   git fetch origin main --tags
   git worktree add --detach <fresh-path> vX.Y.Z
   cd <fresh-path>
   ```

   Do not make any changes in this worktree. The helper fetches `origin` again
   and aborts before publication unless the worktree is clean and detached,
   `HEAD` is the exact pushed remote `vX.Y.Z` tag commit, and that commit is an
   ancestor of `origin/main`.
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
3. Run the fail-closed preflight without `--execute` first. It structurally
   parses every git dependency in `[workspace.dependencies]`, checks its exact
   registry package/version and pinned-revision manifest, queries crates.io,
   validates the remote tag/clean checkout invariants, and verifies provenance
   for any target versions already present on crates.io:

   ```bash
   python3 scripts/release-publish.py X.Y.Z
   ```

   **New-package gate:** `tenferro-internal-cpu-kernels` is new for v0.3.0.
   The command above must abort before publishing anything unless the user's
   approval names exactly `tenferro-internal-cpu-kernels`. After recording that
   exact approval, the human operator asserts it concretely with:

   ```bash
   python3 scripts/release-publish.py X.Y.Z \
     --approve-new-package tenferro-internal-cpu-kernels
   ```

   Never infer this approval from a general release request. On restart, keep
   the exact command unchanged: the helper accepts that approval after the
   package exists only when crates.io also has the target version, whose archive
   must still pass all checks before it is skipped. It rejects the approval as
   stale when the package exists but the target version does not.

   **Generated handoff script (recommended):** instead of typing the two
   commands below by hand, generate a guarded handoff script that re-runs the
   same preflight and requires one exact lowercase `y` at a TTY before it
   invokes the helper with `--execute`:

   ```bash
   python3 scripts/release-publish.py X.Y.Z \
     --approve-new-package tenferro-internal-cpu-kernels \
     --generate-script "$TMPDIR/publish-X.Y.Z.sh"
   ```

   The generator refuses a path inside the release worktree (the helper aborts
   on untracked files). The generated script pins SHA-256 checksums of the
   helper and this workflow, aborts unless stdin is a TTY and the worktree is
   clean and detached, and carries exactly the `--approve-new-package` values
   passed at generation. Run it from the release worktree root:

   ```bash
   cd <fresh-path> && bash "$TMPDIR/publish-X.Y.Z.sh"
   ```

   Regenerate the script after any helper or workflow change; a checksum
   mismatch aborts before anything runs.

   **Change-aware revalidation and exact-SHA CI reuse:** classify what
   actually changed since the previous release with
   `python3 scripts/release-validation-policy.py --base <prev-tag> --head <this-tag>`
   (or repeated `--change PATH OLD NEW` triples for explicit old/new content).
   A helper-or-workflow-only diff needs only the focused `ci-config` lane; a
   publication-metadata-only diff needs metadata, publish-layout, and
   archive/dry-run checks; a semantic manifest diff needs affected tests plus
   the applicable CI tier; Rust source or ambiguous diffs need the full normal
   validation. Before skipping a rerun on the strength of CI already passed,
   verify every required check run for the exact tag commit with the canonical
   query
   `gh api repos/tensor4all/tenferro-rs/commits/<SHA>/check-runs --paginate --slurp`
   and require, per required check name, `head_sha == <SHA>`,
   `status == "completed"`, and `conclusion == "success"`. Any required check
   that is missing, ran on another commit, is not completed, or did not
   succeed fails closed: rerun the applicable tier. The release helper encodes
   this procedure as `verify_release_ci(commit, required_checks)`; the
   maintainer executes it — it is not part of the helper's automatic preflight
   (the helper never blocks on CI state by itself).
4. After the preflight passes, the human operator repeats the same exact command
   with `--execute` (or runs the generated handoff script, which does the same
   after the single exact `y`):

   ```bash
   python3 scripts/release-publish.py X.Y.Z \
     --approve-new-package tenferro-internal-cpu-kernels \
     --execute
   ```

   Before each `cargo publish`, the helper waits for every prerequisite registry
   archive, then runs `cargo package` and inspects that exact local `.crate` file.
   It checks the file list and normalized metadata (name, version, description,
   license, repository, homepage, documentation, README, `rust-version`,
   keywords, categories, and `include`/`exclude`), verifies
   `.cargo_vcs_info.json` equals the clean tagged commit, and compares every
   source-derived regular archive file byte-for-byte with the tagged package
   tree. The file list must equal `cargo package --list` for that tag.
   `Cargo.toml.orig` must equal the tagged source manifest. The helper retains
   every regular file's exact bytes, including Cargo's normalized `Cargo.toml`,
   generated `Cargo.lock`, and VCS file. Dry-run and immediate post-upload
   registry archives must match every member exactly. For target versions and
   prerequisites already registry-visible when the helper starts, only
   `Cargo.lock` bytes may differ from the fresh local package; the identical file
   set (including `Cargo.lock`) and every other comparison remain required.
   Unmapped, changed, or required missing files abort. It packages, dry-runs,
   and publishes one crate at a time.
   Do not pre-package the whole workspace: lower-layer registry versions must
   exist before dependent packages can resolve.
5. The helper is restart-safe and fail-closed: it skips an already-published
   target version only after recreating the local archive from the clean tag,
   then downloading its registry archive and matching every regular file's bytes
   except the generated `Cargo.lock` restart exception above, as well as
   tagged-source provenance. Any missing approval, metadata/provenance
   mismatch, network ambiguity, command failure, or required manifest change
   aborts publication. Fix manifest problems on `main` through Phase 1 and
   restart at Phase 2 with the next patch version.

## Phase 4 — Verify And Close

1. For every published crate, confirm crates.io reports
   `max_version == X.Y.Z`.
2. Fetch one published crate's source (registry cache under
   `~/.cargo/registry/src/` after any consumer build, or the `.crate` file)
   and confirm `.cargo_vcs_info.json` records the tagged commit, and that
   the commit is on `origin` and reachable from `main`.
3. Remove the release worktree. Update downstream pins and announcements
   (Matrix, blog for major releases) as appropriate.
