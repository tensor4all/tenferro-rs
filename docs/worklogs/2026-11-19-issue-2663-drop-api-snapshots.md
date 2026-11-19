# Drop the checked-in `api/*.txt` snapshots (#2663)

Issue: https://github.com/tensor4all/tenferro-rs/issues/2663
Branch: `chore/2663-drop-api-snapshots` (off `origin/main` `d48982d3`).
Plan: `IMPLEMENTATION-PLAN.md`, reviewed before implementation by
GLM-5.3-flash (`reviewer-glm`, `ollama-cloud/glm-5.3-flash:high`) — round 1
`VERDICT: CHANGES` (BLOCKER: `scripts/check-pr-fast.sh` also invoked the checker
against the committed snapshots; IMPORTANT: the durable design doc
`docs/design/api-grammar.md` and the `tenferro-ad` registry module doc still
named the old model), round 2 `VERDICT: CHANGES` (api-grammar.md still
uncovered), final round `VERDICT: APPROVE`.

## What changed

- The 23 checked-in `api/tenferro*.txt` snapshots are deleted; `api/` keeps
  `README.md` (rewritten) and `allowlist.txt` (the E06 symbol list, still read
  by `crates/tenferro-runtime/src/api/tests/registry.rs`).
- `scripts/api-snapshot.sh` lost `--write`/`--check`: it now takes no
  arguments, compiles the unchanged 23-combo matrix into `mktemp -d`, runs
  `scripts/check-api-registry-conformance.py --surface-dir <tmp>`, removes the
  dir on success and keeps it (printing the path) on failure.
- `scripts/check-api-registry-conformance.py`: rules unchanged; `--surface-dir`
  is now required (the old default `api/` holds no snapshots); docstring
  rewritten for the generated surface.
- `scripts/test-check-api-registry-conformance.py` synthesizes its baseline
  surface from the parsed `api_ops!` registries (all declared signatures per
  crate plus every `tenferro_ad` exception line, in both `tenferro-ad.txt` and
  `tenferro-ad+cuda.txt`; empty `tenferro-linalg.txt` for the extension-trait
  case) instead of copying the deleted files; the positive case became
  `test_baseline_surface_conforms`.
- CI (`ci-pr-workspace-tests.yml`, job `api-surface`, renamed "Public API
  surface and registry conformance"): the "Committed snapshots conform" step is
  gone; `scripts/api-snapshot.sh` (no `--check`) is the conformance command; the
  `ci-gate` echo and the no-op lane message follow. `ci.yml` comment reworded.
- `scripts/check-pr-fast.sh` keeps only the checker self-test for code changes
  (the compiled-surface conformance is CI's `api-surface` job; locally
  `scripts/api-snapshot.sh` does it). The removed `-d api` guard had also kept
  the local-gate fixture (`scripts/ci/tests/test_change_policy.py`,
  `LocalGateTests`) from running that block, so the fixture now provisions an
  always-succeeding stub for the self-test, like its stub `cargo` (found by
  the post-review).
- Post-review (independent, `gpt-6.1-sol` via `scripts/pi_review.sh`):
  the gate-fixture regression above was its one real finding (fixed, all 43
  `scripts.ci.tests.test_change_policy` cases pass); the temp directory now
  cleans up on any early exit and is kept only for a conformance failure;
  the checker docstring and the self-test baseline duplication it flagged
  were fixed; `IMPLEMENTATION-PLAN.md` (this task's plan) stays out of the PR
  per `REPOSITORY_RULES.md` ("Do not include AI-generated analysis, task, or
  verification reports as standalone files in PRs") — the decisions and
  verdicts are recorded here.
- Docs and comments that named the files: `api/README.md`,
  `docs/spec/api-conventions.md`, `docs/design/api-grammar.md` (Enforcement),
  `docs/spec/generated/api-ops.md` (regenerated via
  `UPDATE_API_DOCS=1 cargo test -p tenferro-runtime api_docs_are_current`,
  E04 reason/verification strings),
  `crates/tenferro-ad/src/api/registry.rs` module doc,
  `crates/tenferro-runtime/src/api/registry.rs` E04 strings,
  `crates/tenferro-internal-extension-macros/src/{api_ops.rs,lib.rs}` doc
  comments, `crates/tenferro-cpu/Cargo.toml`.
- Untouched on purpose: `api/allowlist.txt` and the E06 test, the checker's
  rules, `COMBOS` (the documented matrix, now in `api/README.md`'s table), the
  pinned `nightly-2026-07-18` + `cargo-public-api@0.52.0`, historical records
  (`docs/design/api-grammar/`, `docs/worklogs/`, old CHANGELOG entries).

## Verification (local, A100 host, toolchain nightly-2026-07-18,
cargo-public-api 0.52.0)

- `python3 scripts/test-check-api-registry-conformance.py` — all 7 cases ok
  (baseline conforms; every reintroduced forbidden shape fails).
- Generated-surface mismatch proof (the issue's requirement):
  `RUSTUP_TOOLCHAIN=nightly-2026-07-18 cargo public-api -p tenferro-ad -ss`
  and `--features cuda` into a temp dir;
  `python3 scripts/check-api-registry-conformance.py --surface-dir <tmp>` →
  `ok`; after appending the legacy engine-taking `EagerTensor::backward` line
  to `tenferro-ad.txt`, the same command fails with both "undeclared eager
  entry point" and "names an engine outside the declared session boundary"
  (exit 1).
- `scripts/api-snapshot.sh` (the exact CI command, all 23 combos) —
  `api-registry-conformance: ok`, `api-snapshot: ok` (54 s warm).
- `UPDATE_API_DOCS=1 cargo test -p tenferro-runtime api_docs_are_current` and
  plain `cargo test -p tenferro-runtime --lib api` — 11 passed (E06 allowlist
  included); `cargo test -p tenferro-internal-extension-macros` — 23 passed.
- `cargo fmt --all --check` clean; `cargo clippy -p tenferro-runtime
  --all-targets` and `-p tenferro-internal-extension-macros --all-targets`
  exit 0 (two pre-existing unused-workspace-dependency manifest warnings,
  unchanged by this diff).
- `python3 -m unittest scripts.ci.tests.test_change_policy` — 43 tests OK
  (after the fixture stub fix).
- Stale-reference sweep: `git grep -nE 'api/tenferro[^ ]*\.txt|
  api-snapshot\.sh --(check|write)'` outside historical dirs — no hits.

Not checked: the hosted CI lanes themselves (RunPod GPU gate untouched; the
`api-surface` job commands were run locally exactly as CI runs them).
