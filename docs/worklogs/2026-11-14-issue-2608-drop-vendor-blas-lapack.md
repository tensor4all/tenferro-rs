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

- **`tenferro-cpu` and `tenferro-runtime` lose their `default` feature.** It *was* the provider
  selector, so it goes with it. `tenferro-runtime` ends up with no features at all;
  `tenferro-cpu` keeps `ledger` and `route-report`, the off-by-default test and host-check
  hooks, which gate `#[doc(hidden)]` items and change no numerical path. `docs.rs` metadata for
  both sets `all-features = true` because `scripts/check-publish-layout.py` requires exactly one
  of `all-features`/`features`: for `tenferro-runtime` that is the whole build, and for
  `tenferro-cpu` it documents the same surface as the default build, because
  `cargo public-api` omits `#[doc(hidden)]` items.
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

## Second review round (partial local-diff review at `6336e10b`)

`ollama-cloud/glm-5.3-flash` reviewed the diff locally (the GitHub diff endpoint
returns 406 above 20,000 lines) and returned four blocking findings plus minor
stale prose. All were verified against the source and fixed:

1. **BLOCKER — `docs/oracle/tensor-ad-oracles-support.md:99`** ran
   `--features autodiff,native` for the incremental-Householder-QR oracle replay.
   Same class as the first round's `external-linalg-interop.md`: a copy-pasteable
   command naming a deleted feature, in a page in the published nav, outside every
   grep gate. Fixed → `--features autodiff`, and `cargo tree -p tenferro-linalg
   --features autodiff` resolves.
2. **BLOCKER — `docs/performance/tt-inner-product-overhead.md`** described
   `benches/openblas_direct_overhead.rs` and gave ~7 runnable
   `--features blas-openblas` / `--bench openblas_direct_overhead` commands. Its
   banner covered #2206 only. Fixed → the banner now also records #2608 (features
   and bench target gone, "tenferro OpenBLAS" columns no longer reproducible), the
   two dead command blocks became one sentence, the verification list keeps the
   commands as *what was run* rather than as instructions, and the benchmark entry
   is past tense with a "(removed … #2608)" marker. The faer command that still
   works stays runnable.
3. **BLOCKER — a stale `source_section` end marker made a source contract vacuous.**
   `linalg_internal_path_contract.rs:601` ended its slice at
   `self.with_linalg_pool_checked(`, deleted with the no-op wrapper. `source_section`
   silently fell back to the end of the file, so the "guarded ineligible branch"
   assertion covered the whole rest of `backend.rs` and still passed.
   Fixed at both levels: the marker is now `"}\n\n        self.with_linalg_pool("` and
   the assertion also requires `solve_read_into_direct_eligible` in the slice; and
   **`source_section` in all three contract files now panics on a missing end
   marker** instead of widening the slice. That strictness immediately found three
   more stale markers nobody had noticed — `cpu_linalg_source_contract.rs`
   (`cpu_backend_buffer_error` → `cpu_negative_integer_exponent`),
   `gpu_linalg_source_contract.rs` (`fn solve_typed` → `fn lu_typed`;
   `fn raw_stream` → `fn matrix_slice_config`) and
   `linalg_internal_path_contract.rs` (`fn count_nonzero<B: LinalgBackend>` →
   `… + ?Sized>`). Four vacuous assertions repaired by one helper change.
4. **BLOCKER — "`tenferro-cpu` declares no features at all" was false.**
   `ledger` and `route-report` remain. Corrected in `CHANGELOG.md`,
   `crates/tenferro-cpu/src/lib.rs`, `api/README.md`, `crates/tenferro-cpu/Cargo.toml`,
   this worklog and `IMPLEMENTATION-PLAN.md`, and the docs.rs/API-snapshot relation
   is now stated and **verified**: `cargo public-api -p tenferro-cpu -ss --features
   ledger,route-report` is byte-identical to `api/tenferro-cpu.txt`, because both
   features gate `#[doc(hidden)]` items (`take_ledger_*`, `take_last_plan_route`)
   and `cargo public-api` omits those. So no second snapshot row is needed, and
   `all-features = true` documents exactly the snapshotted surface.

Also fixed from the same review: `error.rs`'s `mod tests` was gated on
`feature = "cuda"`, so `invalid_workspace_keeps_typed_backend_source` ran in no
default lane although `invalid_workspace` is now unconditional and reaches the faer
route (`cpu/tlinalg_error.rs:87`); it is `#[cfg(test)]` again and the run count went
3575 → 3576. Stale prose: `docs/guides/eager-operations.md` ("default `native`
stack"), `REPOSITORY_RULES.md:568` (`lapack_linalg/*` → `cusolver.rs`/`cuda/ffi`),
`full_svd_read.rs` ("The LAPACK provider packs all three"),
`docs/design/{dot-general-overhead,testing,tensor,scalar-composition,tensor-session-redesign-1938}.md`,
and `#2608` banners on the three design records that still present CPU-BLAS as
current (`incremental-householder-qr.md`, `rank-revealing-qr.md`,
`incremental-householder-qr-performance-gate.md`).

Follow-up on the same page (#2609 third round): the sentence I added pointed at "the
faer run above", but the block immediately above it was the `cpu_install_overhead`
command, whose target was deleted in #2206 — my own sentence inherited the defect it
was meant to fix. Fixed → the dead block is deleted and both dead runs (the
#2206 install-overhead target and the #2608 vendor runs) are named as historical in
one paragraph; the pointer now names the two script runs that work; and
`cargo bench --bench cpu_install_overhead --no-run` moved out of the "Verification
Performed" list into the same historical note.

Verifying that pointer found a pre-existing bug: `scripts/bench-pairwise-contraction.sh`
ran `cargo bench -p tenferro-tensor --bench pairwise_contraction`, and that target
lives in `tenferro-cpu` (it moved there before #2206), so the script's `--kind rust`
leg failed with "no bench target named `pairwise_contraction`". One-word fix to
`-p tenferro-cpu` (the criterion ids the script's own example filters on,
`pairwise_contraction/c64/one_thread/normal_per_call`, are the ones that bench
defines), then `bash scripts/bench-pairwise-contraction.sh --kind rust -- pairwise_contraction/c64/one_thread/normal_per_call`
runs a filtered smoke pass end to end. No numbers from that run are recorded or
published: it used a 0.2 s warm-up / 0.5 s measurement budget and 10 samples, so it
proves the leg starts and finishes, nothing more.

Deliberately not changed: `docs/testing/storage-traversal-performance.md:87-90`. Its
`thread_environment` block is a captured measurement record, and
`scripts/verify-storage-traversal-performance.py:14` lists those five variables as
required environment keys; dropping them means recapturing the baseline and editing
the capture/verify pair. The vendor entries are inert now, not wrong, so that is a
separate change with its own measurement cost.

## What the gate holes were, so the next removal does not repeat them

Two classes of residue survived the mechanical sweeps and were caught only by reading:

1. **Runnable commands inside documentation.** Gates #13/#14b grep `crates ext tests scripts
   .github api`, so a `cargo run … --features <deleted>` in a guide passes every check and fails
   the user. Sweep `--features` and `--no-default-features` occurrences in `docs/` by hand.
2. **Normative prose about threading.** The feature name is easy to grep; the *consequence*
   ("the linked provider keeps its own thread controls", "it can oversubscribe an outer batch")
   is not, and it survives in `docs/spec/reproducibility.md`, `docs/guides/linear-algebra.md` and
   the troubleshooting guide until someone reads them.
