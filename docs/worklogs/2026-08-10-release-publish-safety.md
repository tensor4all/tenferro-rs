# Release publish safety (issue #1650)

Work log for the release-workflow hardening in issue
[tensor4all/tenferro-rs#1650](https://github.com/tensor4all/tenferro-rs/issues/1650):
forbid publication cycles and forward versioned dev-dependencies, fix the two
violations, generate a guarded human publication script, add change-aware
release validation, and move reusable policy into the shared agent rules.

## Outcome

- `scripts/check-publish-layout.py` now rejects forward **versioned**
  normal/build/dev dependencies and publication cycles over versioned edges;
  unversioned (path-only) dev-dependencies are allowed and explained.
- The two manifest violations are fixed with path-only dev-dependencies
  (below), not by moving tests.
- `scripts/release-publish.py --generate-script` emits a guarded human
  handoff script; `verify_release_ci` encodes the exact-SHA CI reuse gate.
- `scripts/release-validation-policy.py` classifies the release-time
  validation lane.
- Shared policy added to `tensor4all-agent-rules`
  (`rules/common/repository.md`), sibling PR #10.

## Decision: path-only dev-dependencies instead of the issue's move-tests approach

**Deviation that needs maintainer sign-off.** The issue prefers moving
cross-layer tests into `publish = false` test crates. That is infeasible for
`tenferro-runtime`: its doctests and inline `#[cfg(test)]` unit tests in
`src/*.rs` use `tenferro_cpu` via the dev-dependency and cannot relocate
(doctests and private-access tests are crate-local). A test-crate migration of
roughly 5200 lines was implemented and then reverted; the migration survives
in the reflog and git history of the pre-branch session if it is ever wanted.

The decisive experiment (run before this work log was written):

- runtime dev-dep `tenferro-cpu = { path, version = "99.99.99" }` →
  `cargo package -p tenferro-runtime --no-verify --locked` fails to resolve
  (the exact v0.3.0 failure mode).
- runtime dev-dep `tenferro-cpu = { path = "../tenferro-cpu",
  default-features = false }` (path-only) → `cargo package` succeeds
  (112 files).
- xla with path-only einsum dev-dep → `cargo package` succeeds (30 files).

Adopted rule: cross-crate dev-dependencies among publishable crates may be
path-only (unversioned). Dev-dependencies are stripped from published
manifests and never registry-resolve for consumers, so unversioned forward
dev-edges (`tenferro-xla` → `tenferro-einsum`) and dev-edges that close a
cycle (`tenferro-runtime` → `tenferro-cpu`, whose reverse edge is a normal
dependency) are safe. The checker enforces that only *versioned* edges
participate in order/cycle constraints.

**Note:** removing the xla `tenferro-cpu` dev-dependency entirely (not just
unversioning it) broke `cargo test -p tenferro-xla`: the einsum dev-dep links
`tenferro-cpu`, which cannot compile without a provider feature
(`CpuBackendKind::default_compiled()` has no no-provider fallback; pre-existing
latent gap, `cargo check -p tenferro-cpu --no-default-features` fails on
main). The xla dev-dependency on einsum therefore carries
`features = ["cpu-faer"]` (a pure feature forward; no test code uses the CPU
backend directly). The latent `tenferro-cpu` no-provider compile failure is
left in place as out of scope.

## Reordering cannot fix the violations

Canonical Phase 3 order: `tenferro-runtime` (6) before `tenferro-cpu` (7);
`tenferro-xla` (9) before `tenferro-einsum` (11). `tenferro-cpu` has a hard
normal dependency on `tenferro-runtime`, and user crates must precede
extensions, so no reordering can satisfy a versioned forward/cyclic dev-edge.

## Generated handoff script (`--generate-script`)

Design, verified by executing the generated script against a stub helper in
tests:

- Output path must be outside the release worktree (the release helper aborts
  on untracked files; the script must not dirty the immutable tag checkout).
- The helper and canonical workflow are pinned with SHA-256 checksums computed
  at generation with Python `hashlib` (no external `sha256sum` binary) and
  re-verified by the script before anything else runs; a later helper change
  cannot silently alter publication behavior.
- Script is mode 0700 (chmod at generation + runtime re-check), non-append,
  deterministic, `bash -n` clean (generator runs the parse check).
- Guards, in order: checksum pin → owner-only mode → TTY → clean detached
  worktree (helper re-verifies the exact pushed tag and main lineage
  authoritatively) → fail-closed preflight without `--execute` → one exact
  lowercase `y` (single `read`; `n`/`Y`/EOF abort) → helper invocation with
  `--execute` carrying exactly the `--approve-new-package` values passed at
  generation.
- Restart-safe: a failed/aborted run can be re-run; the helper skips
  already-published target versions only after full re-attestation.

## Change-aware validation and exact-SHA CI reuse

- `scripts/release-validation-policy.py` classifies the diff between the
  previous release and the tag into lanes by old/new manifest content:
  helper-or-workflow-only → focused `ci-config` lane (no full workspace, no
  CPU/GPU); publication-metadata-only (version, description, homepage,
  keywords, categories, documentation, readme in `[package]` or
  `[workspace.package]`) → metadata + publish-layout + archive/dry-run only;
  semantic manifest (dependency source/version/features, targets, build.rs,
  native libs, profiles, added/removed manifests) → affected tests plus the
  applicable CI tier; rust source or ambiguous → full validation. Mixed sets
  take the strongest lane.
- Before skipping a rerun on the strength of previously passed CI, verify
  every required check run for the exact release commit with the canonical
  query `gh api repos/tensor4all/tenferro-rs/commits/<SHA>/check-runs
  --paginate`, requiring per check `head_sha == <SHA>`, `status ==
  "completed"`, `conclusion == "success"`; anything else fails closed and
  reruns. `verify_release_ci(commit, required_checks)` in
  `scripts/release-publish.py` encodes this with the injected-runner pattern;
  the human maintainer records the tag SHA and required job names at Phase 2.

## Shared rule placement

Cross-repository policy lives in `tensor4all-agent-rules`
(`rules/common/repository.md`, "Publication And Release Safety") — sibling PR
#10, with the tenferro review case documented in the commit message. The
tenferro `REPOSITORY_RULES.md` section keeps only tenferro-specific
constraints (crate names, canonical order, approvals, checker names) and
references the shared rules without vendoring their normative text. Sibling
repos (Tensor4all.jl, BubbleTeaCI, tidu-rs, chainrules-rs) were checked for
conflicting publication guidance: none found; the shared rules are generic and
do not impose tenferro topology.

## Deviation summary for the maintainer

1. Path-only dev-dependencies replace the move-tests approach (see above).
2. `tenferro-xla` keeps a path-only `tenferro-einsum` dev-dependency with
   `cpu-faer` forwarded for buildability.
3. The latent `tenferro-cpu` no-provider-feature compile failure is noted but
   not fixed here.

## Post-review corrections (reviewer-gpt)

- Removed the now-obsolete `[patch.crates-io]` bootstrap (`--no-verify`
  + `--config`) for runtime/xla from `publish_release`: the path-only
  dev-dependencies made it unnecessary, and it contradicted the
  packageable-as-is rule. Orchestration tests now assert no patch is used.
- Fixed multi-approval generation in the handoff script (approval groups are
  joined with a space, not `", "`); added a two-approval execution test.
- `verify_release_ci` now flattens paginated page objects; added a multi-page
  fixture test. Exact-SHA verification remains a human-executed Phase 3 step
  by plan design (documented in the workflow).
- Release-helper classification now allowlists the exact release test files
  instead of the `scripts/test-*` wildcard; unrelated test changes classify
  as full.
- Handoff-script checksum mismatch test also covers the canonical workflow
  file; skills document the versioned `--generate-script` command.

## Next-release approval cleanup

The generic Phase 3 preflight, handoff-generation, and execution examples now
omit new-package approval arguments. They add `--approve-new-package PACKAGE`
only when a release really contains a new package and the user explicitly names
it. This keeps existing packages out of the approval set while preserving the
helper's fail-closed gate for future new packages; helper semantics did not
change. The three byte-identical skills and the OpenCode adapter carry the same
target-neutral rule, guarded by a focused documentation contract test.

The shared-rule dependency is resolved: tensor4all-agent-rules PR #10 merged as
`5cff8254`, with its `validate-rules` check successful. The current shared
`rules/common/repository.md` contains the publication and release safety policy
used here.

Focused verification passed for the documentation and no-approval handoff
contracts, the existing exact/stale new-package approval gate, publish layout,
release validation policy, Python byte compilation, formatting, adapter byte
identity, deterministic repository-rules review, and `git diff --check`.
`ci-config` reaches the same release-helper suite but its interactive handoff
tests require GNU `script -qec` and a Git `master` default branch, which are not
available in this macOS environment; the affected changed tests pass when run
without those Linux-only interactive cases.
