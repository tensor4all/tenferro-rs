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

## #1608 follow-up: published crate metadata

Added the accepted 14-crate metadata contract: workspace MSRV `1.96`, inherited
`rust-version`, crates.io keywords/categories, exact docs.rs URLs, and docs.rs
`rustdoc-args`. The checker and focused importlib tests reject missing metadata,
invalid counts or keyword syntax, invalid URLs, invalid docs.rs feature modes,
and undefined explicit docs.rs features. Published discovery uses Cargo
metadata's `package.publish` semantics, with the 14-crate order above as the
allowlisted publishable set.

| Crate | Keywords | Categories | docs.rs mode |
| --- | --- | --- | --- |
| tenferro-tensor | tensor, numerical, array | science, mathematics, data-structures | all features |
| tenferro-cpu | tensor, cpu, linear-algebra, numerical | science, mathematics, hardware-support | `cpu-faer` |
| tenferro-gpu | tensor, gpu, cuda, webgpu | science, hardware-support, mathematics | `cpu-faer`, `cuda`, `webgpu` |
| tenferro-runtime | tensor, runtime, graph, numerical | science, mathematics, data-structures | `cpu-faer` |
| tenferro-ad | tensor, autodiff, numerical, graph | science, mathematics, algorithms | `cpu-faer`, `cuda`, `webgpu` |
| tenferro-xla | tensor, xla, stablehlo, compiler | science, mathematics, development-tools | all features |
| tenferro-linalg | linear-algebra, tensor, mathematics, numerical | science, mathematics, algorithms | `autodiff`, `cpu-faer`, `cuda`, `webgpu` |
| tenferro-einsum | tensor, einsum, contraction, numerical | science, mathematics, algorithms | `autodiff`, `cpu-faer`, `cuda`, `webgpu` |
| tenferro-fft | tensor, fft, signal-processing, numerical | science, mathematics | `autodiff`, `cpu-faer`, `cuda`, `webgpu` |
| tenferro-tensor-core | tensor, data-structures, numerical | science, mathematics, data-structures | all features |
| tenferro-core-ops | tensor, operations, graph | science, mathematics, algorithms | all features |
| tenferro-internal-cpu-kernels | tensor, kernels, cpu, numerical | science, mathematics, hardware-support | all features |
| tenferro-internal-ops | tensor, operations, autodiff, graph | science, mathematics, algorithms | all features |
| tenferro-internal-extension-macros | tensor, macros, code-generation | development-tools, development-tools::procedural-macro-helpers | all features |

The explicit docs.rs lists avoid mutually exclusive CPU/BLAS providers and
rocm/provider-inject features. The CUDA/WebGPU combinations for GPU, AD, and
all three public extensions compile in hardware-free nightly documentation
builds, so those extension APIs remain discoverable. Faer is the only CPU
provider selected for docs. The exact `cargo +1.96.0 check --workspace`,
metadata discovery (14 manifests), focused checker tests/check, and nightly
GPU/AD/extension docs builds passed. `cargo package` produced and inspected the
tensor, runtime, linalg, tensor-core, and internal-ops package manifests;
metadata survived and packaged manifests had no git-only dependency entries.
Verification of tensor/runtime/linalg and internal-ops tarballs is blocked by
registry drift: their existing published 0.2.0 dependency artifacts lack
current workspace symbols; tensor-core verification passed. No dependency,
README, service, facade, feature-default, or MSRV-matrix change was made.
Residual: publish deep crates only after matching dependencies are available on
crates.io; package verification should then be rerun.

## 2026-08-09 v0.3 final-review hardening

Added `scripts/release-publish.py` as the fail-closed human operator path for
Phase 3. It verifies the clean detached checkout against the pushed remote tag
and `origin/main`, structurally validates every workspace git pin against both
its revision manifest and exact crates.io version, and requires exact approval
for each package that is new to crates.io. In particular, v0.3 publication
cannot proceed without an approval assertion naming
`tenferro-internal-cpu-kernels` exactly.

Publication now proceeds one DAG node at a time. Each node is packaged and its
actual archive files, normalized metadata, README, and tagged-commit provenance
are checked before and after `cargo publish --dry-run`; dependent packaging
starts only after prerequisite registry archives are visible with matching
provenance. A restarted run skips an existing target version only after the
same registry-archive provenance check. Focused tests use injected transports
and in-memory archives, so unit tests perform no network access. Live validation
also confirmed all current strided, CubeCL, Cubek, and computegraph pin
manifests and declared registry versions.

### Final-review round 1 corrections

Restart approval now distinguishes package existence from exact target-version
existence. The documented approval remains valid after the newly approved crate
has published, but only when its target version exists and its downloaded
archive passes full verification; an existing package without that version is a
stale approval and aborts.

Archive provenance no longer trusts the VCS marker alone. Every source-derived
regular archive member is mapped to the tagged package tree and compared
byte-for-byte, and the archive list must equal Cargo's tagged package selection.
Normalized `Cargo.toml`, generated `Cargo.lock`, and
`.cargo_vcs_info.json` have explicit generated-file handling, while
`Cargo.toml.orig` must match the tagged source manifest. This intentionally does
not require source files omitted by Cargo's packaging rules. Injected command,
registry, archive, source-tree, clock, checkout, metadata, and order hooks cover
restart, propagation, failure, and irreversible-command ordering without
network access or publication. These tests and the publish-layout tests run in
the normal `ci-config` profile.

### Final-review round 2 corrections

Every registry comparison now starts from a newly generated and inspected local
archive from the clean tag. The helper retains the complete regular-file byte
map and requires both dry-run and crates.io archives to match it exactly, closing
the gap for normalized dependency changes, `Cargo.lock`, and semantically valid
but byte-different VCS metadata. Resume follows the same DAG sequence: attest
prerequisite registry archives, create the local exact-tag archive, then compare
the downloaded target before skip.

Tar member prefix, traversal, and duplicate validation now runs before directory
entries are skipped. Network response truncation and HTTP protocol/read errors
are converted to bounded, actionable release failures. Focused regressions cover
generated-file mutations, malicious traversal directories, HTTP failures, exact
resume comparison, and prerequisite-before-dependent packaging.

## 2026-08-09 SemVer proposal gate

The release guidance previously accepted a requested target before deriving an
independent SemVer target. That allowed a dependency release to be mistaken for
a reason to align tenferro with an independently versioned repository.

Phase 0 now derives one proposal from the latest matching published stable
baseline and its provenance tag, then classifies changes actually merged since
that tag as breaking, feature, or fix-only. Accepted issues count only when they
link to merged implementation. The explicit pre-1.0 and 1.0+ table makes the
result reviewable. A conflicting supplied target requires explicit confirmation
and a recorded reason before edits. Approved never-published packages are
excluded from baseline agreement, while unpublished newer tags are treated as
anomalies rather than baselines.

The human-only publication boundary is unchanged: agents stop after validation,
and only a maintainer with crates.io ownership runs Phase 3 publication.

The RED pressure scenario omitted the skill and requested tenferro 0.4.0 after
strided-rs 0.4.0. The response treated that request as sufficient instead of
deriving and proposing tenferro 0.3.0 from the published 0.2.0 baseline. The
new documentation contract test was then observed failing before the workflow
and adapters were updated.

Four fresh GREEN pressure scenarios passed against the updated guidance: a
0.2.0 feature proposed 0.3.0 and stopped on a requested 0.4.0 conflict; a
0.2.3 fix proposed 0.2.4; a 1.4.2 breaking change proposed 2.0.0; and an
explicitly reasoned 0.4.0 override proceeded with a recording requirement.

The initial `ci-config` run passed all Python/config suites but stopped because
`actionlint` was unavailable. The controller's subsequent run and the fix-round
run with pinned actionlint 1.7.7 completed the full profile successfully.

A quality-review correction aligned every adapter with the canonical human-only
boundary: agents stop after validation, and a human maintainer alone runs Phase
3 publication. Contract tests now assert the exact SemVer rows, ordering of
provenance before the proposal, the conflict halt, the human-only boundary, and
byte equality of the three skill adapters.

Validation commands:

- `python3 scripts/test-release-publish.py`
- `python3 scripts/ci/run_profile.py ci-config`
- byte comparisons of the three `SKILL.md` adapters
- `git diff --check`

## 2026-08-09 Cargo 1.97 dry-run staging fix

The first human v0.3 publication attempt stopped safely before uploading any
crate. `cargo publish --dry-run` succeeded, but Cargo 1.97 staged its archive at
both `target/package/tmp-crate/` and `target/package/tmp-registry/` instead of
retaining `target/package/<crate>-<version>.crate`. The helper removed and then
checked only the latter path, so it reported a false missing-archive failure.

The shared archive inspection boundary now clears and discovers all three exact
Cargo locations, rejects a missing archive, requires multiple generated copies
to be byte-identical, and then applies the existing full archive and expected
content checks. Read failures identify the exact candidate path. This preserves
the fail-closed contract rather than skipping dry-run archive attestation.

TDD regressions model Cargo's staged paths, reject differing copies, and verify
read-error attribution. A real Cargo 1.97 run on `tenferro-core-ops` confirmed
that `cargo package`, `tmp-crate`, and `tmp-registry` archives were byte-identical
and that the repaired helper completed package-to-dry-run attestation without
uploading. The focused release-helper suite passed with 30 tests.

## 2026-08-09 v0.3 runtime/CPU cycle recovery

The v0.3.0 human run published and verified the first six crates through
`tenferro-internal-ops`, then stopped before uploading `tenferro-runtime`.
The immutable tag is commit `9505d5bfd60c890212f0ec44aa9cf07fef185f63`.
Runtime has a versioned path dev-dependency on `tenferro-cpu 0.3.0`, while CPU
has a normal dependency on runtime 0.3.0, so Cargo's ordinary package ordering
cannot bootstrap either absent registry version.

Clean-tag experiments established the narrow recovery: runtime package,
publish dry-run, and publish use `--no-verify` plus
`--config 'patch.crates-io.tenferro-cpu.path="ABS_TAG_PATH/crates/tenferro-cpu"'`
whenever CPU 0.3.0 is registry-absent, including a restart after runtime is
registry-visible. The path comes from the selected release checkout's Cargo
metadata and is encoded as an inline TOML string. The helper still inspects and
compares the package, `tmp-crate`, and `tmp-registry` archives; the experiment
found all three byte-identical. The existing prerequisite wait permits CPU to
package and publish normally, without the patch. Once CPU is registry-visible,
runtime packaging no longer needs the bootstrap patch.

`--release-root PATH` lets the corrected helper on `origin/main` operate on the
clean detached immutable tag checkout. `verify_release_checkout(root=...)`
remains authoritative for cleanliness, exact remote tag identity, and
reachability from `origin/main`. The new-package approval gate and human-only
`--execute` boundary are unchanged.

`--no-verify` is deliberately limited to runtime's three bootstrap commands: it
skips Cargo's package verification build because that build hits the cyclic
lockfile collision. It does not weaken archive, tagged-source, normalized
manifest, registry-byte, or provenance comparisons. The source tree was already
covered by the release's workspace, PR, and main checks. Residual risk remains
that crates.io may reject the real upload despite the successful dry-run; only
the human operator may cross that server-acceptance boundary.
