# Drop the vendor BLAS/LAPACK CPU providers (#2608)

## Scope

The CPU stack becomes one unconditional backend: faer-backed `tlinalg` for linalg and
`cpueinsum`/`tcontract` for contraction. The `native`/`blas` dual-backend split, the
OpenBLAS/Accelerate/MKL provider features, and the `provider-inject`/`provider-src` test seams are
deleted, not aliased: a dependent crate that still lists them fails with "feature does not exist".

Out of scope by the issue's own terms:

- the `t4a-faer` → `colmat` package rename (checked in `Cargo.lock`: the registry has no `colmat`
  release), so `tenferro-cpu` still re-exports the linked build as `tenferro_cpu::faer`;
- collapsing the owned/view twin entry points.

## Step 0 — the required CI context

`cargo test (blas inject)` was removed from `ai/repo-settings.json` and from the live branch
protection before any source edit, by `bash scripts/configure-repo-settings.sh --repo
tensor4all/tenferro-rs` (it printed `repo-settings-ok`). `gh api` then listed seven required
contexts with `blas inject` gone. No admin-rights refusal occurred.

## Decisions

- **`tenferro-cpu` and `tenferro-runtime` declare no features at all.** Their `default` feature was
  the provider selector, so it goes with it. `docs.rs` metadata for both now sets
  `all-features = true` (the same build every dependent gets) because
  `scripts/check-publish-layout.py` requires exactly one of `all-features`/`features` and there is
  no feature list to name.
- **`faer` and `tlinalg` become non-optional in `tenferro-linalg`**, and `tcontract-exec` gains
  `features = ["faer"]`. One faer identity, one linalg provider.
- **The API snapshots keep their content and change their names.** Dropping `native` from the five
  docs.rs feature sets produces byte-identical surfaces, so `+native-cuda` → `+cuda` (and
  `+autodiff-native-cuda` → `+autodiff-cuda`) is a pure rename. `native_engine` in
  `tenferro-tensor` is an API name, not a feature, and survives untouched.
- **The macOS lane is renamed `macos`.** It used `--no-default-features --features
  blas-accelerate,...`; with Accelerate gone it is a second-architecture/second-libc run of the
  same faer stack, so the profile and its cache key lose the vendor name.
- **`scripts/ci/trybuild-cargo.py` is deleted** with the `workspace-blas` profile that was its only
  caller (it existed to add the OpenBLAS/LAPACK link flags to nested trybuild Cargo invocations).
- **`tenferro-cpu` stays a normal dependency of `tenferro-gpu`** even though only gpu's tests use
  it. Moving it to `[dev-dependencies]` would delete a production edge that
  `docs/architecture/tenferro-crates.md` documents and `scripts/test-doc-consistency.py` asserts.
  The `unused dependency` warning it produces is pre-existing (gpu's `src` never referenced it).
- **The architecture diagram names the real stack.** `docs/assets/tenferro-architecture.svg` had
  `faer | BLAS/LAPACK` in the `tenferro-cpu` box; it now reads `faer · tlinalg · cpueinsum`.
  `scripts/test-doc-consistency.py` asserts that label and rejects `BLAS/LAPACK`, `OpenBLAS`, `MKL`,
  `Accelerate` and `native` anywhere in the file, so the diagram cannot drift back.
- **The vendor-BLAS analysis in `docs/design/execution-engine-provider-architecture.md` is kept and
  marked historical.** It is the record of why the pluggable provider was removed (#2004, #2608);
  deleting it would delete the reasoning, and the document already opens with a superseded notice.

## Behaviour that changes for a dependent crate

- Dependency lists naming `native`, `blas`, `blas-openblas`, `blas-accelerate`, `blas-mkl`,
  `provider-inject`, `provider-src`, `cpueinsum-blas` or `tlinalg-blas` stop resolving.
- `Cargo.lock` no longer contains `cblas-sys`, `openblas-src`, `openblas-build`, `intel-mkl-src`,
  `accelerate-src`, `blas-src`, `lapack-src` or the `lapack` FFI crate.
- The `doctor`-visible CI contract loses one required context and two profiles.
- Nothing in the public API changes: every `api/*.txt` snapshot is content-identical.

## Verification

All commands ran in the worktree `/home/shinaoka/work/tf-2608` on the branch
`chore/2608-drop-vendor-blas-lapack` (base `be356ad1`).

| Gate | Command | Result |
|---|---|---|
| build | `cargo build --workspace --all-targets` | pass |
| tests | `cargo nextest run --workspace --cargo-profile ci --no-fail-fast` | 3575 passed, 267 skipped, 0 failed |
| doctests | `cargo test --doc --workspace --profile ci` | pass (all binaries `ok`) |
| fmt | `python3 scripts/ci/run_profile.py fmt` | pass |
| clippy | `python3 scripts/ci/run_profile.py clippy` | pass |
| extensions | `python3 scripts/ci/run_profile.py extensions` | pass |
| coverage | `python3 scripts/ci/run_profile.py coverage` | 264/264 files meet thresholds |
| ci-config | `python3 scripts/ci/run_profile.py ci-config` | pass (`actionlint` 1.7.7 installed locally) |
| API snapshot | `bash scripts/api-snapshot.sh --check` | `api-snapshot: ok`, `api-registry-conformance: ok` |
| deny | `cargo deny check bans` (what CI runs) | pass |
| tensor-core deps | `bash scripts/check-tensor-core-deps.sh` | `tensor-core-deps-ok` |
| docs consistency | `python3 scripts/test-doc-consistency.py` | pass |
| docs site inputs | `python3 scripts/test-check-docs-site.py`, `python3 scripts/test-gen-dep-graph.py` | pass |
| guide snippets | `python3 scripts/check-guide-dependency-snippets.py` (+ its self-test) | `4 guides verified` |
| agent skills | `python3 scripts/check-agent-skills.py --root-dir .` | `agent-skills-ok` |
| interop example | `cargo run -p tenferro-tutorial-code --profile ci --no-default-features --bin faer_interop` | `all checks passed` |
| CUDA example | `cargo check -p tenferro-tutorial-code --profile ci --no-default-features --features cuda --bin cuda_vendor_interop` | pass |
| repo settings | `bash scripts/configure-repo-settings.sh --repo tensor4all/tenferro-rs` | `repo-settings-ok` |

Not run, with the reason:

- `bash scripts/build_docs_site.sh` — no `quarto` on this host. The profile steps that do not need
  it ran individually (table above).
- `cargo deny check` (full) — `licenses` and `advisories` reject plain `MIT`/`Apache-2.0` in this
  environment's registry set, which is independent of this change; CI runs `check bans` only, and
  that passes.
- The hosted CUDA lanes — no CUDA work is touched (the four `--features cuda,native` sites become
  `--features cuda`), so the local `cargo check` above is the evidence.

## Review

- Reviewer model: `ollama-cloud/glm-5.3-flash` (`glm-brainstormer`, resolved
  `ollama-cloud/glm-5.3-flash:high`), read-only, launched from the implementing pi session.
- Pre-implementation plan review: two rounds, both `VERDICT: CHANGES`; every finding verified
  against the repository and dispositioned in `IMPLEMENTATION-PLAN.md` §12. No finding was rejected.
- Post-implementation review of the finished diff: `VERDICT: CHANGES`, one `BLOCKER`, one
  `IMPORTANT`, seven `NIT`. All nine verified against the source and fixed; disposition table in
  `IMPLEMENTATION-PLAN.md` §13. The `BLOCKER` was a copy-pasteable
  `--features cuda,native` in `docs/guides/external-linalg-interop.md`, a feature that no longer
  exists — documentation commands are outside the plan's grep gates, which is why it survived the
  first pass.

## What the gate holes were, so the next removal does not repeat them

Two classes of residue survived the mechanical sweeps and were caught only by reading:

1. **Runnable commands inside documentation.** Gates #13/#14b grep `crates ext tests scripts
   .github api`, so a `cargo run … --features <deleted>` in a guide passes every check and fails
   the user. Sweep `--features` and `--no-default-features` occurrences in `docs/` by hand.
2. **Normative prose about threading.** The feature name is easy to grep; the *consequence*
   ("the linked provider keeps its own thread controls", "it can oversubscribe an outer batch")
   is not, and it survives in `docs/spec/reproducibility.md`, `docs/guides/linear-algebra.md` and
   the troubleshooting guide until someone reads them.
