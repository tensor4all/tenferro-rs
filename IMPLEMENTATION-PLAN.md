# IMPLEMENTATION-PLAN — tenferro-rs#2608

Drop vendor BLAS/LAPACK providers and collapse the dual-backend CPU/linalg layers.

- Issue: https://github.com/tensor4all/tenferro-rs/issues/2608 (approved `ai-ready` by maintainer)
- Worktree: `/home/shinaoka/work/tf-2608`, branch `chore/2608-drop-vendor-blas-lapack`
- Base: `origin/main` = `be356ad1` — **identical to the survey base**, so every line number in the
  issue is valid at this base. Re-verified by `git rev-parse origin/main`.
- Authorisation to implement: maintainer comment on the issue (do not re-ask).

## 0. Goal and non-goals

**Goal.** `colmat`/`cpueinsum`/`tlinalg` (faer) becomes the single unconditional CPU backend.
Features `blas`, `provider-src`, `provider-inject`, `blas-openblas`, `blas-accelerate`, `blas-mkl`
and `native` are deleted, not aliased. All dual-backend cfg gates and BLAS-only source, tests and
benches go. CI keeps one CPU configuration. Docs and agent skills stop describing a second backend.

**Non-goals (deliberate, recorded so they are not silently done):**

1. Swapping `faer = { package = "t4a-faer" }` for `colmat` (root `Cargo.toml:104`). Issue: "not a
   step in this PR" unless the upstreams (`tlinalg` 0.11.0, `tcontract` 0.12.0, `cpueinsum` 0.12.0)
   release a `colmat` dependency first. They have not (checked `Cargo.lock`). **No dependency
   version changes at all except removals.**
2. The owned/view twin collapse listed as "optionally fold in" in §3 of the issue. Not folded in —
   it is an independent refactor and would make this diff unreviewable.
3. Archiving/deprecating the upstream `cpueinsum-blas`, `tlinalg-blas`, `cblas-inject`,
   `lapack-inject` crates. That is the issue's "After merge" section.
4. GPU: keep `tenferro-gpu`'s `blas1_cuda` and cuBLAS code untouched.
5. `tenferro-cpu/src/blas1.rs`, `benches/blas1.rs`, `blas1_*` tests: **kept** (colmat-path level-1
   kernels). This is the issue's explicit caution and the main grep-and-delete hazard.
6. `OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` pins in perf-gate scripts that compare against external
   libraries: kept.

## 1. Step 0 — repository settings (blocker, first)

Current behaviour: `ai/repo-settings.json:10` lists `"cargo test (blas inject)"` in
`required_status_checks.contexts`; live branch protection on `main` lists the same 8 contexts
(verified with `gh api repos/tensor4all/tenferro-rs/branches/main/protection`). If left, the PR and
every later PR blocks on a job this PR deletes.

Change:
- Delete the `"cargo test (blas inject)",` line from `ai/repo-settings.json`.
- Apply it to live protection:
  `bash scripts/configure-repo-settings.sh --repo tensor4all/tenferro-rs --settings ai/repo-settings.json`
  from the worktree.
- Verify: `bash scripts/check-repo-settings.sh --repo tensor4all/tenferro-rs` and re-read
  `gh api .../branches/main/protection --jq '.required_status_checks.contexts'`.

Admin rights confirmed available (`gh auth status`: token scopes include `admin:org`, `repo`, and
the account owns the repo; `enforce_admins=false`). If any API call is refused anyway, stop and
report the exact command — do not edit source.

Note: after this, `main`'s committed `ai/repo-settings.json` is stale until the PR merges. That is
the intended state; `check-repo-settings.sh` is run against the worktree file.

## 2. Manifests

Files and current behaviour → target:

| File | Current | Target |
|---|---|---|
| `Cargo.toml:87,90,105-110` | 8 workspace deps for the vendor stack | delete `cpueinsum-blas`, `tlinalg-blas`, `cblas-sys`, `cblas-inject`, `blas-src`, `lapack`, `lapack-src`, `lapack-inject`; reword the comment at `:83-86` to describe one faer identity |
| `crates/tenferro-cpu/Cargo.toml:25-56` | `default=["native"]`, `native=["dep:faer"]`, `blas`, `provider-src`, `provider-inject`, `blas-{openblas,accelerate,mkl}` | delete `default`, `native` and all six provider features; `faer` becomes a plain non-optional dep |
| `crates/tenferro-cpu/Cargo.toml:68,79-84` | optional vendor deps | delete; `cpueinsum` stays non-optional |
| `crates/tenferro-cpu/Cargo.toml:19-20` | docs.rs `features=["native"]` | drop `features` (keep `rustdoc-args`) |
| `crates/tenferro-cpu/Cargo.toml:126-131` | `[[bench]] openblas_direct_overhead` + `required-features=["blas"]` | delete target and `benches/openblas_direct_overhead.rs` |
| `crates/tenferro-cpu/Cargo.toml:133-139` | `required-features=["native"]` on `direct_faer_on_engine_pool`, `["native","route-report"]` on `route_faerlimit` | `direct_faer_on_engine_pool` drops it; `route_faerlimit` keeps `["route-report"]` |
| `crates/tenferro-linalg/Cargo.toml:20,25-80` | `default=["native"]`, `native`, `blas`, three provider features, `provider-inject` | delete all; `native`'s deps (`faer`, `tcontract-exec/faer`, `tlinalg`) become unconditional |
| `crates/tenferro-linalg/Cargo.toml:45-47,72-76,95-109` | optional vendor deps | delete |
| `crates/tenferro-linalg/Cargo.toml:167-170` | `[[test]] lapack_allocation` with `required-features=["blas"]` | delete target and `tests/lapack_allocation.rs` |
| **all** `[package.metadata.docs.rs] features = [...]` tables | 6 crates list `native` in the docs.rs feature set: `-cpu:20`, `-linalg:20`, `-einsum:20`, `-gpu:20`, `-ad:20`, `-fft:19` (the `-runtime:20` table too) | drop `native` from each list; `tenferro-cpu`/`tenferro-runtime` drop the key entirely (empty list is rejected) |
| `crates/tenferro-runtime/Cargo.toml:81-84` | `required-features=["native"]` on the `[[test]] session_softmax_broadcast_routes` | delete the key |
| `crates/tenferro-einsum/Cargo.toml:20,25-…` | `default=["native"]`, `native`, `blas`, providers | delete; `cpueinsum-blas` dep (`:70`) deleted |
| `crates/tenferro-ad`, `-runtime`, `-fft`, `-gpu`, `-xla` `Cargo.toml` | `default=["native"]` + forward features | delete the `default` line and the provider forwards; the `tenferro-cpu` dependency becomes default-featured |
| `docs/tutorial-code/Cargo.toml` | `default=["native","xla-pjrt","doc-snippets"]`, `native`, provider features, `[[bin]] faer_interop` at `:81-84` with `required-features=["native"]`, `[[bin]] blas_interop` at `:86-89` with `required-features=["blas"]` | drop `native` from `default` and all provider features; delete the `blas_interop` bin + `src/bin/blas_interop.rs` + its vendor deps (`cblas-sys`, `lapack`, `openblas-src`); **delete `required-features=["native"]` from `faer_interop` (`:84`), keeping the bin** |
| `ext/{bf16-proof,df64-proof,scalar-consumer-algorithm,scalar-consumer-application}/Cargo.toml` | each has `default=["native"]` + a `native` forward, with the "forcing `native` so a `blas` workspace build does not collide" comment | delete the feature, the `default` line and the comment; depend on the CPU crates with default features |
| `ext/sparse/Cargo.toml:31-32` | `default-features=false, features=["native"]` | drop `features=["native"]` |
| `tests/fixtures/extension-authoring/Cargo.toml:10,12` | same | drop `features=["native"]` |
| `Cargo.lock` | contains the vendor stack | regenerate with `cargo update -w --offline`-equivalent: `cargo metadata`/`cargo build` after the manifest edits, then `cargo update --workspace` only if needed; must lose `openblas-src`, `openblas-build`, `intel-mkl-src`, `accelerate-src`, `lapack-sys` |

`crates/tenferro-linalg/Cargo.toml:101` carries a comment explaining "neither depends on the
other" about `tlinalg` vs `tlinalg-blas` — deleted with `tlinalg-blas`.

## 3. `tenferro-cpu`

### 3.1 Delete outright
- `src/inject.rs` (1052 lines) and `mod inject` (`lib.rs:186`-ish, `#[cfg(feature="provider-inject")]`).
- `src/contraction/blas_plan.rs` (175 lines) and `mod blas_plan` / `use blas_plan::BinaryPlan`
  (`contraction.rs:3-4`).
- `lib.rs:166-173` `extern crate blas_src / cblas_inject / lapack_inject / lapack_src`.
- `lib.rs:25-36` the two `cfg_attr` allows (`all(not(native), not(blas))` and
  `all(test, provider-inject)`).
- `lib.rs:67-90`: all four `compile_error!` guards (`native`+`blas`, `provider-inject` without
  `blas`, two explicit providers, `provider-inject`+explicit provider) plus the comment block at
  `:63-66` that explains the "neither backend" build.
- `benches/openblas_direct_overhead.rs`.
- `tests/provider_inject*.rs` (in cpu and linalg) + `PROVIDER_INJECT_TEST_LOCK`.

### 3.2 `src/contraction.rs` (1037 lines)
Current behaviour: `BinaryPlan`, `Scalar` and `record_contraction_route` are cfg-split over
`cpueinsum` vs `cpueinsum-blas`; ~12 items are gated on `any(native, blas)` and have no-backend
stubs at `:985-1037`; BLAS builds thread an extra `work` buffer through `:754-776, 811-832`.

Changes:
- `:28-31` → `use cpueinsum::BinaryPlan;` unconditionally (delete the `blas_plan` alias).
- `:107-117`: keep exactly one `record_contraction_route(route: cpueinsum::ExecutionRoute)` (the
  `not(blas)` body) and delete the `cpueinsum_blas` twin.
- `:40-45`: `ContractionRoute` is test-only and has one non-`Vendor` variant left. Its only
  consumer is `backend/tests/batch_policy.rs:164` asserting `Some(ContractionRoute::Native)`.
  Delete the enum, `LAST_CONTRACTION_ROUTE`, `take_last_contraction_route`, and change that
  assertion to the existing `take_last_route_was_faer()` (`Some(true)`) — the faer route is what it
  actually meant. Keep `LAST_FAER`/`take_last_route_was_faer`.
- `:123-126`: one `trait Scalar: PoolScalar + cpueinsum::Scalar {}` + blanket impl.
- `:175-190`: delete `impl LowerContractionError for cpueinsum_blas::BlasError`.
- `:513-941`: ungate the ~12 `cfg(any(native, blas))` items; `:68,88,642` `cfg(all(feature="native",
  any(test, feature="route-report")))` → `cfg(any(test, feature="route-report"))`.
- `:754-776, 811-832`: delete the BLAS-only `work` buffer plumbing, keeping the native path. Then
  check whether `buffers` is still read in the affected functions (`initialized` et al.); if not,
  rename to `_buffers` or drop the parameter, following whatever the native arm already does.
- `:985-1037`: delete the no-backend stubs entirely.
- `default_faer_limit` becomes `budget == 1` as the issue specifies — verified by reading the
  current function before editing.
- `:41` `#[cfg_attr(not(feature="blas"), allow(dead_code))]` goes with the enum.

### 3.3 `src/lib.rs`
- `cpu_provider_id()` (`:141-156`) becomes `pub fn cpu_provider_id() -> &'static str
  { "tenferro.cpu.faer" }`. Grep for `"tenferro.cpu.blas"`/`"tenferro.cpu.none"` — only
  `lib.rs` and `docs/guides/choosing-a-backend.md:99` mention them; the doc line is rewritten in
  §7. The id is also the runtime `ProviderId`; no runtime code compares the two other literals, so
  no compat shim is needed.
- `:262-280` `pub use faer` loses `#[cfg(feature="native")]`, `#[cfg_attr(docsrs, doc(cfg(...)))]`
  and the `# #[cfg(feature = "native")] # {` / `# }` doc-test wrappers.
- `cfg(all(test, not(provider-inject)))` → `cfg(test)` at `lib.rs:723`, `engine.rs:950`,
  `backend/tests.rs:132`, `exec_session/tests.rs:102`.
- `src/tests.rs`, `src/tests/cpu_tests/dot_structural_analytic.rs`,
  `src/runtime_adapter/tests.rs`, `src/engine/tests.rs`, `src/contraction/tests.rs` native/blas
  gates: ungate.

### 3.4 Docs inside sources
- `exec_session.rs:560-576` "Lanes and the vendor provider" → describes lanes as always working.
- `batch_policy.rs:44-51` "The BLAS linalg lane" → forced strategies always honoured.
- `capability.rs` / `provider.rs` / `runtime_adapter.rs`: drop any text describing two providers.

## 4. `tenferro-linalg`

### 4.1 Delete
- `src/cpu/linalg/lapack_linalg.rs` + `src/cpu/linalg/lapack_linalg/` (2184 lines).
- `src/cpu/tlinalg_blas.rs` (142), `src/cpu/tlinalg_workspace.rs` (82).
- `map_blas_error` (`tlinalg_error.rs:127-135`).
- `lib.rs:192-196` `compile_error!` guards.

### 4.2 `src/cpu/backend.rs` (3283 lines, 108 cfg sites)
Current behaviour: every `*_entered` function has the shape
```
{
    #[cfg(feature="native")] { #[cfg(feature="native")] { <faer arm> } }
    #[cfg(feature="blas")]   { #[cfg(feature="blas")]   { <blas arm> } }
}
```
and the 20 traits impls choose per build. Target: each function keeps only the faer arm, dedented;
`with_linalg_pool_checked` drops the `reject_forced_batch_strategy` wrapper (`:179-190`) and becomes
a call to `with_linalg_pool` — or is deleted and inlined, whichever leaves fewer lines; the
`CpuBackendLinalgAffinityExt` trait goes if it has no other purpose.
- Delete the 20 blas arms (`:1169-2732` region as surveyed; locate by `#[cfg(feature = "blas")]`).
- Ungate the native fast paths (`:442-708`) and `faer_*_host_ok` helpers (`:2759-2969`).
- 108 cfg sites: 0 remaining.

### 4.3 Smaller pieces
- `src/cpu/linalg/mod.rs:151-152`: drop the `faer`/`blas` alias module layer, use `faer_linalg`'s
  items directly.
- `src/cpu/linalg/faer_linalg.rs` + `faer_linalg/`: 8 cfg sites → 0.
- `src/cpu/backend/packed_lu.rs:72-142`: inline `CpuPackedLu` into direct
  `tlinalg::solve_prepared_batch` / `factor_solve_batch` calls.
- `src/cpu/tlinalg_error.rs:15-46`: inline `map_provider_error!` into `map_error`, drop the
  `extra_pat` hook.
- `src/lane.rs`: delete `mod unavailable`, inline `mod native`; ungate `tensor_ext.rs:60-233`.
- `src/error.rs`: ungate `map_error` and `BackendError`; revisit `InvalidWorkspace`'s wording
  ("both CPU provider routes") — keep the variant, reword the doc.
- `src/api/registry.rs`: drop the 13 "The BLAS provider gathers every operand…" `layout` strings and
  keep/rewrite the faer wording. **This changes the generated API snapshot** — see §6 accept-bar.
- `src/cpu/tlinalg.rs`: 0 cfg sites remain.
- `src/api/tests/*`, `src/cpu/tests/*`, `src/extension.rs`, `src/householder.rs`, `src/validation.rs`,
  `src/prelude.rs`, `src/eager_ext.rs`: ungate native, delete blas-only arms.

### 4.4 LAPACK contracts that must survive (they are asserted, not incidental)
- 1-based pivots: `src/cpu/tests/batched.rs:252-280`, `src/extension.rs:469`. **Check what the faer
  path currently returns and keep the test asserting 1-based**, since the test must not change
  semantics silently. If faer is already 1-based, nothing to do; if the test was LAPACK-only,
  convert it to assert the post-change behaviour and say so in the PR body.
- eig `NonConvergence` mapping.
- real-spectrum behaviour (CHANGELOG:480).
- Negative-stride `solve`: LAPACK refused it; now accepted → **new test** in
  `src/cpu/tests/` proving a negative-stride `solve` succeeds and matches the contiguous result.

## 5. `tenferro-einsum`

- `cpu_concrete.rs:9-12` `type NumericalPlan<T>` and `:95-102` `trait Scalar`: keep only the
  `cpueinsum` versions as concrete (`EinsumPlan<T>`), delete the `cpueinsum_blas::Blas` type
  parameter entirely.
- Keep `EinsumPlan::new` (`:246-248`).
- Ungate `:112, 166, 173, 196, 322`; delete the `Unsupported` stubs at `:155` and `:379`.
- `lib.rs:85`: remove the now-unneeded `allow`.

## 6. Tests, CI and scripts

### 6.1 Delete BLAS-only tests
`inject` integration tests (cpu + linalg) + `PROVIDER_INJECT_TEST_LOCK`;
`tests/lapack_allocation.rs`; `contraction/tests.rs::blas_thin_c64`;
`test_dot_general_read_blas_negative_stride_view_falls_back`; the two forced-strategy refusal tests
in `linalg/src/cpu/tests/batch_policy.rs`; `svd_full_lapack_provider_returns_square_unitary_factors`.

### 6.2 Single-backend cleanup
`small_linalg_allocation.rs:152-213` drop blas ceilings; `cpu_linalg_allocation.rs` drop
`blas_backend()` and the loops at `:404`, `:550`; `borrowed_linalg_parity.rs` flatten `providers()`.
Shrink the source-text contract tests: `cpu_linalg_source_contract.rs`,
`linalg_internal_path_contract.rs:417-426`, `src/cpu/tests/linalg.rs:456-470`.
Rename the five `..._before_lapack` tests (drop the LAPACK reference from the name).
Ungate `native`/`any(native,blas)` test modules: `eager_dot_plan_reuse.rs:163-303`,
`tutorial_binaries.rs:94-99`, and the rest.

### 6.3 CI
- `scripts/ci/run_profile.py` — **five distinct edits, all of them currently green-tested:**
  - delete the `workspace-blas` (`:57-69`), `blas-inject` (`:79-84`) and `macos-accelerate`
    (`:71-77`) profiles and the `workspace-blas` branch (`:255-259`); drop all three from
    `FULL_PROFILE` (`:168-177`).
  - `docs` profile (`:114-117`): the first command is
    `cargo run … --no-default-features --features native --bin faer_interop` — **drop `--features
    native`** (keep the bin, run it with the now-unconditional backend). The second,
    `… --features blas --bin blas_interop` with the OpenBLAS `RUSTFLAGS`, is deleted with the bin.
    The third keeps `cuda` but **drops `native`**: `--features cuda --bin cuda_vendor_interop`
    (that bin's `required-features` is `["cuda"]` only, so it survives).
  - `_TRYBUILD_CARGO_WRAPPER`: check users; the wrapper exists for BLAS link flags — if no profile
    still needs it, delete `scripts/ci/trybuild-cargo.py` and the constant.
  - give the macOS job a colmat test set (reuse `workspace-faer`'s command shape so macOS still
    tests `-p tenferro-gpu -p tenferro-fft -p tenferro-linalg`).
- `.github/workflows/ci.yml`: remove the `test-inject-blas` job (`:151-177`), the `run_blas` policy
  output (`:24`) and the docs-site BLAS install step (`:273`).
- **Four build sites pass `--features cuda,native`** and would fail immediately after merge:
  `.github/workflows/runpod-gpu-test.yml:676`, `ci-cache-publish.yml:179`, `CI_gpu.yml:232` and
  `scripts/ci/local_gpu_gate.py:325` (the `CUDA tutorial build` step) — all
  `… --no-default-features --features cuda,native --bin cuda_tutorial`. Drop `,native` in each; the
  `cuda_tutorial` bin's `required-features` is `["cuda"]`.
- `scripts/bench-mps-inner-product.sh`: `BLAS_FEATURE=blas-accelerate` (`:8`), `--backend blas|both`
  (`:25, 114, 151-172`). Delete the BLAS arm and the `--backend both` mode; `faer` is the only
  value left. These run *our* `blas` feature, so the "keep the external-library thread pins"
  non-goal does not cover them. **Note**: the feature reaches cargo through the shell variable
  `"$BLAS_FEATURE"` (`:155, 174`), so a literal `--features <name>` grep cannot see it — hence
  gate #14b below has a second pattern keyed on the variable and the feature-name literals.
- `scripts/incremental-householder-qr-performance.py:21, 72-73`: drop `"blas"` from `BACKENDS` and
  the `--features blas,blas-openblas` command.
- `.github/workflows/ci-pr-workspace-tests.yml`: remove the `cpu-blas` matrix leg and its
  install/verify steps (`:50, 55, 143-150`); the backend name feeds the cache key and target dir, so
  the matrix keeps exactly one entry.
- `scripts/ci/change_policy.py`: drop `run_blas`.
- CI self-tests: `test_run_profile.py` (expectations at `:68, 70, 74, 166, 172, 177, 188`),
  `test_workflow_contracts.py:129,166,248`, `test_change_policy.py`,
  `test_build_artifact_contracts.py:167` (currently KeyErrors on `cpueinsum-blas`/`tlinalg-blas`).
- `scripts/test-check-publish-layout.py` fixtures hardcode `native = []` (`:31, 35, 484-494`) and
  `scripts/test-release-validation-policy.py:91` rewrites `default = ["native"]`; both are
  *synthetic fixture manifests*, not ours, and still exercise their checkers. Left alone.
- `scripts/check-tensor-core-deps.sh:11-12`: drop the `cblas-sys`/`lapack` ban lines.

### 6.4 Repo-settings script
`ai/repo-settings.json` per §1. `scripts/check-repo-settings.sh` needs no change.

## 7. Docs and agent skills

Top-level: `README.md:183`, `PERFORMANCE_TIPS.md:191-192,255,591-609`,
`REPOSITORY_RULES.md:558,568,654-655,716-720`, `CONTRIBUTING.md:144` (the
`workspace-blas` line in the profile list), `ai/contribution-workflows/{issue-intake,release-publish}.md`,
`api/README.md:15-43`.

Guides/design: `docs/getting-started/index.md`, `guides/{troubleshooting,choosing-a-backend,cpu-execution,parallelism-and-caching}.md`,
`architecture/tenferro-crates.md`, `design/{cpu-backend-execution,execution-engine-provider-architecture}.md`,
`docs/llms.txt`, `docs/index.md`, regenerate `docs/spec/generated/api-ops.md`.

Live guides that carry runnable `--features` commands and **must** be edited (found by grep; the
issue does not list them): `docs/guides/devices-and-gpu.md:167` (`--features cuda,native`),
`docs/guides/tenferro-fft.md:235` (`--features native,cuda,doc-snippets`),
`docs/guides/external-linalg-interop.md:29` (`--features blas-openblas --bin blas_interop`) and
`:605` (`--features cuda,native`).

`docs/guides/external-linalg-interop.md`: keep the colmat half, drop the `cblas-sys`/`lapack` half
together with `blas_interop.rs` and its `_quarto.yml` entry.

**API snapshots (the issue names only the exclusion line; the machinery itself breaks).**
`scripts/api-snapshot.sh:40-63` builds every snapshot with a feature set; the sets
`tenferro-gpu native,cuda`, `tenferro-ad native,cuda`, and `tenferro-{einsum,linalg,fft}
autodiff,native,cuda` no longer resolve once `native` is gone, and the comment at `:40` says
`native` is the default for `tenferro-cpu`/`tenferro-runtime`. Change: drop `native` from those
five feature lists, rename the five files
`api/tenferro-{gpu,ad}+native-cuda.txt` → `+cuda.txt` and
`api/tenferro-{einsum,linalg,fft}+autodiff-native-cuda.txt` → `+autodiff-cuda.txt`, update the
`COMBOS` suffixes to match, update the comment at `:40` ("`native` is the default" for
`tenferro-cpu`/`tenferro-runtime` — stale once §2 drops those docs.rs keys), and update
`api/README.md:20-30` (table plus the "`native` is the default set" paragraph) and the
BLAS-provider exclusion at `:43-44`.
`scripts/test-check-api-registry-conformance.py:92,128` hardcode `tenferro-ad+native-cuda.txt` →
`tenferro-ad+cuda.txt`.

**Left alone deliberately**: `docs/worklogs/*` (except the new one for this PR),
`docs/superpowers/plans/*`, `docs/oracle/*`, `docs/performance/tt-inner-product-overhead.md` (a
measurement record whose commands are historical), release notes, and the `native_engine` public
API name of `tenferro-tensor` (`api/tenferro-tensor.txt:99,3452`) — that is an API name, not a
feature, and must survive the sweep in §9.12. Same for `coverage-thresholds.json:118`, which keys
on `crates/tenferro-cpu/src/blas1.rs` (a kept file).

Agent skill `tenferro-compute` in all three copies (`.claude/`, `.agents/`, `.kimi/`):
`references/crate-selection.md:49-62`, `pitfalls.md:37-42`, `performance-idioms.md:133-134`,
`api-cheatsheet.md:211`.

`scripts/test-doc-consistency.py:41,187,190`.

`CHANGELOG.md`: one breaking entry under `## [Unreleased]`. Historical worklogs, plans and release
notes untouched (they legitimately mention the old provider ids).

## 8. Semantics that change (documented, not accidental)

- threadteam owns every CPU thread; the "budget not honoured under `blas`" caveat and the
  `OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` exceptions in the *product* docs go.
- Lanes always work.
- Forced batch strategies are always honoured (the `reject_forced_batch_strategy` refusal goes).
- The CPU provider id is the constant `tenferro.cpu.faer`. No `native`/`blas`/`none` variant.

## 9. Verification — exact commands and acceptance bars

All commands run in `/home/shinaoka/work/tf-2608`. `nextest` is used where the repo's profiles use
it. **Baseline is `be356ad1`; no measurement is compared across hosts and no measuring host is
touched** (no benchmark runs at all — see "measurement" below).

Cargo builds here run through kache; `kache doctor --verify` already passed (22388/22388 valid
entries, daemon v2.1.1).

Gates, in this order:

1. `cargo build --workspace --all-targets` — bar: exit 0.
2. `python3 scripts/ci/run_profile.py fmt` — bar: exit 0.
3. `python3 scripts/ci/run_profile.py clippy` — bar: exit 0 with
   `-D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc`.
4. `python3 scripts/ci/run_profile.py workspace-faer` — bar: exit 0, zero failures, zero ignored
   growth beyond what exists at base. (This is the full workspace test gate: nextest `--workspace`,
   the `tenferro-fft/autodiff` acceptance, doctests, and the scalar-composition check.)
5. CUDA configuration: `cargo build -p tenferro-ad -p tenferro-fft -p tenferro-gpu
   --no-default-features --features cuda` — bar: exit 0. These used to compile only thanks to the
   no-CPU-backend stubs, so this is the regression this step protects.
5b. `cargo build -p tenferro-tutorial-code --no-default-features --features cuda --bin
   cuda_vendor_interop` and `cargo run -p tenferro-tutorial-code --no-default-features --bin
   faer_interop` — bar: exit 0. This is what the `docs` profile runs after §6.3's edit.
6. `cargo test -p tenferro-cpu --all-targets` and `cargo test -p tenferro-linalg --all-targets` —
   bar: exit 0; the new negative-stride `solve` test present and passing.
7. `cargo deny check` — bar: exit 0 (fewer dependencies, so this can only get easier).
8. API snapshots: `bash scripts/api-snapshot.sh --check` (the `placement` job,
   `ci-pr-workspace-tests.yml:224`) **and** `python3 scripts/check-api-registry-conformance.py` —
   bar: exit 0 with the five renamed files committed. `bash scripts/api-snapshot.sh --write`
   regenerates them; the diff must be only the five renamed files plus any real surface change from
   §4.3's registry `layout` rewording.
9. `python3 scripts/check-agent-skills.py` — bar: exit 0.
10. `python3 scripts/test-doc-consistency.py` — bar: exit 0.
11. `python3 scripts/ci/run_profile.py ci-config` — bar: exit 0. This is a hosted-CI gate
    (`ci.yml:335`) that runs `check-publish-layout.py`, `check-duplicate-test-messages.py` (which
    the §6.2 renames feed) and `python3 -m unittest discover -s scripts/ci/tests -v` (the four
    updated self-tests in §6.3) plus `actionlint`.
12. `python3 scripts/ci/run_profile.py coverage` — bar: exit 0, or a stated reason if the local
    llvm-cov run is not reproducible off the CI image; `scripts/check-coverage.py` reads
    `coverage-thresholds.json`, which keeps a `blas1.rs` entry (§7).
13. `rg -n 'feature = "(blas|native|provider-)' crates ext docs` — bar: **no output**.
14. `rg -n '(cpueinsum-blas|tlinalg-blas|cblas-sys|cblas-inject|blas-src|lapack-src|lapack-inject)'
    --glob '!Cargo.lock' --glob '!docs/worklogs/**' --glob '!docs/superpowers/**'
    --glob '!docs/oracle/**' --glob '!docs/performance/**' .` — bar: no output.
14b. `rg -n -- '--features [^ ]*(native|blas\b)' scripts .github api` **and**
    `rg -n '(BLAS_FEATURE|blas-accelerate|blas-mkl|blas-openblas|provider-inject|provider-src)'
    scripts .github api` — bar: the first returns no output; the second returns only the
    `blas1`-related and external-library thread-pin lines that §0 keeps. (The second pattern exists
    because `bench-mps-inner-product.sh` forwards the feature through `"$BLAS_FEATURE"`.)
15. `rg -n 'openblas|intel-mkl-src|accelerate-src|lapack-sys' Cargo.lock` — bar: no output.
16. `bash scripts/check-tensor-core-deps.sh` — bar: `tensor-core-deps-ok`.
17. `bash scripts/check-repo-settings.sh` — bar: exit 0 after §1.
18. Deliberately kept: `rg -n 'blas1' crates/tenferro-cpu` still finds `blas1.rs`, `benches/blas1.rs`,
    `blas1_*` tests, and `rg -n 'blas1_cuda' crates/tenferro-gpu` still finds the cuBLAS code —
    bar: both present.

**Measurement.** This PR changes no numerical kernel, so it runs **no benchmarks** and reports no
timings. The only "measurement" is the deletion counts, taken mechanically:
`git diff --stat` and `git grep -c 'cfg(feature'` before/after. Acceptance: ≥3000 lines of BLAS-only
source deleted; `cfg(feature = "blas"|"native"|"provider-")` count 346 → 0.

**Independent review.** The repo requires a different-provider review at the head SHA with
`xreview:done`; this plan's own GLM plan review is separate from that. If the different-provider
review is unavailable, the PR gets `xreview:requested` and is **not merged**.

## 10. Risks

1. **Silent semantic drift in the `*_entered` dedent.** The 108 cfg sites in `backend.rs` are
   mechanically dedented; a mis-drop could leave a `blas`-only call to a deleted function (caught by
   compile) or swap an arm (caught only by tests). Mitigation: the workspace test gate (#4) is the
   authority; the LAPACK contract tests in §4.4 are named individually and checked one by one.
2. **Allocation-count ceilings.** `small_linalg_allocation.rs`, `cpu_linalg_allocation.rs` and
   `cpu_build_allocation.rs` assert exact byte allocation counts. Removing a layer can *lower* them,
   and the tests may assert `<=` (fine) or `==` (must be updated with a stated reason). Mitigation:
   read each assertion; where a number moves, record old→new in the PR body.
3. **`Cargo.lock` regeneration on a kache machine.** A failed partial `cargo update` can leave the
   lock inconsistent. Mitigation: edit manifests first, then a single `cargo metadata`-driven lock
   update, then a clean `cargo build --workspace --all-targets` to prove resolution.
4. **The 346 cfg sites include `ledger`/`route-report`/`autodiff`/`cuda`, which must survive.** A
   blind regex on `cfg(feature` would destroy them. Mitigation: every edit is keyed on the exact
   feature names `blas`, `native`, `provider-src`, `provider-inject`, `blas-openblas`,
   `blas-accelerate`, `blas-mkl`, never on `cfg(feature` alone; the acceptance grep in #13 is
   restricted to those prefixes and `ledger` etc. are re-grepped as still present.
5. **Live repo-settings change is global before the PR merges.** If this branch is abandoned, `main`
   is left without the `blas inject` required check while the yml on `main` still creates it — a job
   that would no longer be required, which is harmless. Mitigation: the issue explicitly orders this
   change first; and the job itself is removed in the same PR.
6. **`docs/tutorial-code` `blas_interop` removal** also removes the only *link-level* proof that the
   interop surface works. Accepted: the issue says to drop it together with the `cblas-sys`/`lapack`
   half of `external-linalg-interop.md`; the colmat half remains documented and exercised.
7. **Feature-resolution fallout in `ext/sparse` and `ext/tropical`** (path-excluded, own lockfiles).
   Mitigation: `fmt`/`clippy` profiles cover them; their `Cargo.toml` references are updated per §2.
8. **Failures that are silent in local gates but loud in hosted CI** — the reviews' central finding,
   now folded into the plan: the API-snapshot feature sets (§7), the `docs` profile's
   `--features native` (§6.3), the four `--features cuda,native` build sites (§6.3), the docs.rs
   feature tables in six crates (§2), and the two bench/perf scripts that forward the feature through
   a shell variable (§6.3). None of them is exercised by `cargo build --workspace` or by the
   workspace test profile. Mitigation: gates #5b, #8, #11 and the explicit `rg` sweeps #13/#14/#14b.
9. **Renaming five API snapshot files changes reviewable paths.** Mitigation: one `git mv` per file
   with no content change unless §4.3's registry rewording really changes the surface, so the diff
   stays readable.
10. **A blind regex on `cfg(feature` (§10.4) is the one edit that can silently destroy `ledger`,
    `route-report`, `autodiff` and `cuda` gating.** Mitigation as stated: every edit keys on the
    exact seven feature names, and gate #18 re-checks that the kept ones survived.

## 11. Alternative rejected

**Keep `native` as an alias of "faer backend" and only delete the six vendor features.** Rejected
because the issue is explicit ("`native` is deleted outright, not kept as an alias") and because a
`default = []` crate with an always-on backend is the same thing spelled with more code — the 221
`native` cfg sites would stay and §3/§4's collapse would not happen. Also rejected: folding in the
`t4a-faer`→`colmat` switch (§0.1) and the owned/view twin collapse (§0.2) — both would change what
the reviewer must verify, for no requirement in this issue.

## 12. Review record

**Round 1 — plan review before any code.**

- Reviewer: **`ollama-cloud/glm-5.3-flash`** (agent `glm-brainstormer`, resolved model
  `ollama-cloud/glm-5.3-flash:high`), read-only, launched from this pi session with the `subagent`
  tool. It did not edit files.
- Verdict: **`VERDICT: CHANGES`**.
- Findings and disposition (every citation was re-verified against this base before accepting):
  - `BLOCKER` API-snapshot machinery breaks: `scripts/api-snapshot.sh:40-63` feature sets contain
    `native`, five snapshot files are named `+native-cuda.txt`, and
    `scripts/test-check-api-registry-conformance.py:92,128` hardcodes the name. **Verified, accepted**
    → new §7 "API snapshots" paragraph, gates #8.
  - `BLOCKER` `run_profile.py` `docs` profile keeps `--features native` (`:114-117`) and
    `--features cuda,native` (`:121-123`); `docs/tutorial-code/Cargo.toml:84` `faer_interop`
    `required-features=["native"]`. **Verified, accepted** → §2 row rewritten, §6.3 expanded,
    gate #5b.
  - `BLOCKER` docs.rs `features=[…"native"…]` in six crates; `tenferro-runtime/Cargo.toml:84`
    `required-features=["native"]`; `scripts/check-publish-layout.py:438-444` rejects undefined
    docs.rs features. **Verified, accepted** → two new §2 rows.
  - `BLOCKER` bench/perf scripts still build the deleted features:
    `scripts/bench-mps-inner-product.sh:8,25` and
    `scripts/incremental-householder-qr-performance.py:72-73`. **Verified, accepted** → §6.3, and
    the sweep gates #14/#14b widened so they cannot slip through again.
  - `IMPORTANT` verification set omits the repo's `ci-config` (and `coverage`) profile, which runs
    `check-publish-layout.py`, `check-duplicate-test-messages.py` and the CI self-tests. **Verified,
    accepted** → gates #11, #12.
  - `IMPORTANT` three GPU workflows pass `--features cuda,native` (found by me while verifying this
    finding, not named by the reviewer): `runpod-gpu-test.yml:676`, `ci-cache-publish.yml:179`,
    `CI_gpu.yml:232`. **Verified, accepted** → §6.3.
  - `IMPORTANT` note which `native` occurrences are *not* features and must survive:
    `tenferro-tensor`'s `native_engine` API. **Accepted** → §7 "Left alone deliberately", gate
    #13's scoping.
  - `NIT` add the snapshot rename and the docs-profile `--features native` as explicit risks.
    **Accepted** → §10 items 8 and 9.
- Reviewer findings rejected: none. One finding was over-broad and was narrowed instead of applied:
  "widen gate #12/13 to `rg … scripts .github api`" would also flag the deliberate `native_engine`
  API name and the synthetic publish-layout fixtures, so the sweep is scoped by `--glob` exclusions
  and split into #13/#14/#14b.

**Round 2 — confirmation review of the revised plan.**

- Reviewer: same model, same read-only agent, same session path.
- Verdict: **`VERDICT: CHANGES`** (narrow: two `IMPORTANT`, three `NIT`; it explicitly confirmed the
  §7 snapshot plan, the §6.3 `docs`-profile commands, the §2 docs.rs/runtime/linalg rows and the
  gate-#13 scoping as correct).
- Disposition:
  - `IMPORTANT` `scripts/ci/local_gpu_gate.py:325` passes `--features cuda,native`. **Verified,
    accepted** → §6.3 now lists four sites, §10.8 says "four".
  - `IMPORTANT` gate #14b cannot see `bench-mps-inner-product.sh`'s `"$BLAS_FEATURE"`. **Verified,
    accepted** → #14b gained a second pattern and a stated bar that allows the kept lines.
  - `NIT` `api/README.md` bullet is `:43-44` and the stale `native`-is-default comment at
    `api-snapshot.sh:40` belongs in the change list. **Accepted** → §7 text updated.
  - `NIT` `tenferro-runtime/Cargo.toml:84` is a `[[test]]`, not a bench/example. **Verified,
    accepted** → §2 row corrected to `:81-84`, target `session_softmax_broadcast_routes`.
  - `NIT` stale cross-reference to #12 in §10 risk 4. **Accepted** → now #13.
- Reviewer findings rejected: none. Line-range NITs in `run_profile.py` were not actioned verbatim —
  the plan's ranges are within a line or two and the named profile entries identify the edit
  unambiguously, so churning the numbers would add no information.

---

## 13. Worklog

`docs/worklogs/2026-11-14-issue-2608-drop-vendor-blas-lapack.md` records the reviewer model, the
verdict, the disposition above, and the final verification conclusions.

### Scope added during implementation (maintainer request)

`docs/assets/tenferro-architecture.svg` is in this PR. Its `tenferro-cpu` box said
`faer | BLAS/LAPACK`, which framed the deleted split as a free choice; it now reads
`faer · tlinalg · cpueinsum`. `scripts/test-doc-consistency.py` asserts that label and rejects
`BLAS/LAPACK`, `OpenBLAS`, `MKL`, `Accelerate` and `native` anywhere in the SVG, so the diagram
cannot drift back. The three pages that embed it (`README.md`, `docs/index.md`,
`docs/architecture/tenferro-crates.md`) and `docs/llms.txt` were corrected in the same pass.

### Deliberate deviations from this plan

1. **`tenferro-gpu` keeps `tenferro-cpu` as a normal dependency** instead of moving it to
   `[dev-dependencies]`. The move deletes a production edge that
   `docs/architecture/tenferro-crates.md` documents and `scripts/test-doc-consistency.py` asserts,
   and the unused-dependency warning it would fix is pre-existing. Reverted.
2. **`all-features = true` in `tenferro-cpu`/`tenferro-runtime` docs.rs metadata** instead of
   deleting the `features` key. `scripts/check-publish-layout.py` requires exactly one of
   `all-features`/`features`; a featureless crate has no list to name.
3. **`scripts/check-tensor-core-deps.sh` keeps `cblas-sys` and `lapack`** in its forbidden list.
   They are unreachable now, and keeping them keeps the negative rule mechanically enforced.
4. **`docs/design/execution-engine-provider-architecture.md` keeps its vendor-BLAS analysis**,
   marked historical in the superseded notice and in the section header, rather than deleting the
   record of why the provider was removed.
5. **The macOS profile is renamed `macos`** (and its cache key `macos-v1`) rather than kept as
   `macos-accelerate` with the vendor feature removed.

### Final verification

Recorded in `docs/worklogs/2026-11-14-issue-2608-drop-vendor-blas-lapack.md` with the exact
commands and results, including the gates that could not run on this host (full `cargo deny
check` outside `bans`) and why each is not evidence of a regression.

### Post-implementation review

- Reviewer: `ollama-cloud/glm-5.3-flash` (`glm-brainstormer`, read-only), same model as the plan
  rounds, on the staged diff.
- Verdict: **`VERDICT: CHANGES`** — one `BLOCKER`, one `IMPORTANT`, seven `NIT`; it explicitly
  cleared the feature sweeps, the deleted files, the API/CI contract agreement, the SVG, and
  confirmed no unrecorded behaviour change.
- Disposition, every finding accepted and fixed:
  - `BLOCKER` `docs/guides/external-linalg-interop.md:437` still ran
    `--features cuda,native`, a now-nonexistent feature, in a copy-pasteable console block.
    **Verified, fixed** → `--features cuda`. This is a real gate hole: #14b sweeps
    `scripts .github api`, so documentation commands escape it. `docs/spec/reproducibility.md`
    has the same shape of residue and is fixed below.
  - `IMPORTANT` `docs/spec/reproducibility.md:16-22` told the reader to pin
    `OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`OMP_NUM_THREADS`/`VECLIB_MAXIMUM_THREADS` for a
    `blas` build and linked a heading that no longer exists. **Verified, fixed** → the pool is
    stated as the only CPU thread control, dead link removed.
  - `NIT` `cpu_provider_id` still had the `{ "…" }` block and "compile-time-selected" doc left
    over from the deleted cfg dispatch. **Fixed** → literal + corrected doc.
  - `NIT` `packed_lu.rs` carried `#[allow(unused_imports)] use super::linalg;` hiding a now-dead
    import created by inlining `CpuPackedLu`. **Verified by grep, fixed** → both lines deleted.
  - `NIT` plural-provider prose in `cpu/linalg/mod.rs`, `eig_normalize.rs`, `faer_linalg.rs`,
    `api/tests/registry.rs`. **Fixed** → singular, or "the numeric library" where the sentence is
    about what faer guarantees rather than about a provider seam.
  - `NIT` one-element `Vec` provider helpers in `cpu/tests/linalg.rs`, `full_svd_read.rs`,
    `householder_qr.rs` kept the dual-backend frame (this was §6.2's own requirement, partially
    applied). **Fixed** → helpers and all 18 loops deleted, `CpuEngine::serial()` called directly,
    three test names renamed `…_for_compiled_providers` → `…_on_the_cpu_backend`.
  - `NIT` the QR bench still accepted the `native` spelling for `--backend`. **Fixed** → `faer`
    only. The `openblas_num_threads`/`mkl_num_threads` provenance fields stay: they record what
    the host's environment set, which is what makes a baseline reproducible.
  - `NIT` `docs/tutorials/index.md:17` still advertised "Calling faer or BLAS/LAPACK directly"
    for a guide that now opens by saying a vendor call is not part of it. **Fixed** → retitled.
  - `NIT` `docs/design/linalg.md:236-238` described deleted batched LAPACK kernels and `getrs` in
    a present-tense status section. **Fixed**.
- Findings rejected: none. Two findings that the reviewer had itself doubted were checked and
  confirmed before fixing (the `packed_lu.rs` import by grep, the `reject_negative_strides`
  behaviour as a deliberate keep).
- Also found while verifying that `BLOCKER`'s class: `docs/guides/linear-algebra.md:133` still
  promised "A BLAS/LAPACK provider may use its own worker pool … can oversubscribe an outer
  batch", and `docs/design/session-blas1.md:84` still listed "CPU faer and BLAS feature tests" as
  the verification recipe. Both **fixed**.
