# Work log: scalar composition handoff (Stage 1 + Stage 2, PR #1800)

Handoff note for the next session on the scalar-composition plan recorded in
`docs/design/scalar-composition.md`. Read this file first, then the design sections it
points at. Everything below was measured on the head named here; nothing is assumed.

## Where things are

| Item | Value |
| --- | --- |
| Repository | `/home/shinaoka/tensor4all/tenferro-rs` |
| Worktree | `.worktrees/scalar-composition-stage1` (work here, not in the primary checkout) |
| Branch | `agent/scalar-composition-stage1`, pushed to `origin` |
| PR | **#1800 — draft, auto-merge must stay disabled** |
| Head at handoff | `48bb5031` — 256 commits, **0 behind `origin/main`**, tree clean |
| Merge base | `3230ec6f` (`git merge-base origin/main HEAD`) |
| Diff size | 181 files, +27632 / −4000 |
| Design source of truth | `docs/design/scalar-composition.md` (§5.16a, §5.17a/b, §5.20b, §5.20c are the ones in play) |
| Earlier work logs | `docs/worklogs/2026-09-15-scalar-composition-{stage1a,1b-and-2-1,extension-execution}.md` |

Cargo is run with `-j 8` and `CARGO_BUILD_JOBS=8` (user-mandated). **Never run two cargo
commands at once against this target directory.**

## Objective, compressed

Stage 1b: open scalar contract (`Scalar`, `ScalarArithmetic`, integer wrapping), one preset
table, `TensorScalar` as a thin erased adapter, one shared dispatch mechanism,
caller-destination entry points, exactly one `ad_admission` with three typed errors, plus an
external proof crate whose own scalar satisfies the contract end to end.

Stage 2: `Tensor<S: ScalarSet = DefaultScalars>` with `define_scalar_set!`; convert the
per-variant arms to tag-based dispatch with a single erased payload in descending arm-density
order; resolve the storage/reinterpretation, IR, cache-identity, typed-error, C-API/XLA/
serialization boundaries with explicit conversion-or-rejection; keep `ScalarSet` out of
numerical bodies and prove no set-induced specialization. **Remove the seven `Tensor`
variants last.** Ends when the acceptance criteria of #1785, #1788, #1789, #1790 and #1793 are
met.

## Status

| Area | State |
| --- | --- |
| Stage 1b | **Done and verified** |
| Stage 2 parameterization, boundary decisions, kernel-sharing evidence | **Done and verified** |
| Module conversion, all four objective-named files + every remaining 5+-arm file + both seams + the three deliberate exceptions | **Done and verified** |
| `remove the seven Tensor variants last` | **Not implemented** — blocked on #1789's resource-boundary contract (measured trade-off below) |
| #1793 ordinary eager einsum routing | **Not implemented** — the issue text does not authorize a feature PR |
| Coverage item of the verification list | **In flight** — 80.5% of added instrumented lines (this figure predates the last commit) |

Measured state of the conversion: in shipping library code (excluding `types.rs`, tests and
benches) **four** `Tensor` variant pattern sites remain — two in a `#[cfg(test)]` module inside
`crates/tenferro-linalg/src/ad.rs` and two in `crates/tenferro-cpu/benches/grouped_gemm.rs`.
Everything else is tag dispatch. `crates/tenferro-tensor/src/types.rs`, the enum definition
itself, still carries 271 arm-head lines that name variants: that is the removal's surface.

### Verified evidence at the handoff head

- `cargo check --workspace --all-targets`: 0 errors, 0 warnings.
- Workspace suite: 3418 run, 3417 passed, 1 failed — the pre-existing trybuild
  `storage_ui_compile_contracts` (A/B verified against `origin/main`).
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test '<focused test>'`: passes.
- `python3 scripts/check-coverage.py <json>`: 226/226 files at their thresholds.
- `python3 scripts/repository-rules-review.py --base origin/main --head HEAD --dry-run --llm-skipped-reason "local deterministic review"`: pass.
- CI profiles `docs`, `fmt`, `ci-config`, `extensions`: pass. `workspace-faer`: 3417/3418 (the
  same trybuild failure).
- Kernel-sharing evidence is wired into CI (debug 366 vs release 4 instantiations).
- Per-stage source line counts and the per-area table are in the design doc.

## Not implemented, with the exact unblocking input

### 1. Removing the seven `Tensor` variants

The single erased payload the objective names can be expressed today as
`(DType, Box<dyn Any + Send + Sync>)`, and the trade-off is measured rather than argued
(`ext/df64-proof/tests/erased_payload_allocation.rs`,
`report_erased_payload_allocation_cost`):

| Representation | Measured |
| --- | --- |
| Today: seven variants holding `TypedTensor<T>` inline | `size_of::<Tensor>()` = **1464 bytes**, `Tensor::from_typed` = **0 allocations / 0 bytes** |
| Single payload behind a box | **1 allocation of 1456 bytes per erased tensor**, outside every pool the erased layer accounts for |

Choosing between them is the resource boundary the objective leaves with #1789
("keep the sealed pool as a resource boundary owned by #1789 until that issue's contract is
agreed"). **Unblocking input:** whether the erased layer may allocate tensor storage outside
the accounted pool. If yes, the swap is mechanical — the definition plus the 271 arms in
`types.rs`, plus the two non-library sites — because every other naming site is already gone.
If no, the inline storage-aware erased payload has to be designed first.

### 2. #1793 routing an external scalar through the ordinary eager einsum surface

The issue states verbatim that it "does not authorize a feature implementation PR". The missing
piece is a scalar-generic contraction slot, which is a public-contract change.
**Unblocking input:** authorization for that feature PR.

## In flight: coverage of the lines this branch added

The objective's "90% line coverage on changed files" is this repository's own `AGENTS.md`
§Test Coverage Target (90%+ per file; cover new paths; add tests when modifying a file below
90%; the linalg AD rules are excepted because their guarantee is numerical). Three readings,
measured:

1. **Enforced gate**: `coverage-thresholds.json` with default 80 and per-file overrides that are
   mostly frozen pre-existing baselines (values of 0, 8, 13 exist under `_comment_*_baseline`
   headings). This branch does not edit that file; the gate passes 226/226.
2. **Whole-file 90%**: of the 63 changed `crates/**/*.rs` files the profile instruments, 16 are
   at or above 90%; the rest sit mostly in the 74–87% band, dominated by code this branch did
   not write (`tenferro-tensor/src/types.rs` is 74.4% over 3928 lines). Not met, not claimed.
3. **Added lines** (what "cover new paths" asks for): **2903 of 3606 instrumented added lines =
   80.5%** at `cov66`; 342 more lines would be needed for 90%. 8675 further added lines carry no
   line record in this profile because they are feature-gated or `cfg`-excluded.

Tests added for this, all passing: the elementwise same-dtype matrix and all 49 ordered dtype
pairs; the reduction `reduce_sum_read`/`reduce_prod_read` view arms and the sum-of-squares
refusals; the grouped-gemm outer scheduling table for f32, c32 and c64; `eig`'s zero-extent
f32/c32 path; the linalg `tensor_write_view` table; the faer f32 `svd_values`/`eigh_values`
entries; the analytic read view across every preset scalar.

### Two things the next session should fix before adding more

**Weak tests (near padding).** Three of the added tests assert only that a call returned:
`dot_runtime`'s outer tests (the spy provider does no arithmetic), the product read view
matrix and the analytic read view matrix. `GemmSpy` already records `grouped_calls`, `parallelism` and
`grouped_job_counts`, so the outer tests can assert that the **outer path was actually taken**;
the product view test can assert a value (`prod([2,3]) == 6`). Do that rather than adding more
tests of the same shape.

**Two structural caps, both measured, neither fixable by tests.**

- Each converted arm's operand extraction is a provable no-op at run time: the arm is selected
  by `lhs.dtype()`/`rhs.dtype()`, which `Tensor` derives from the same payload the extraction
  reads, so `pair_operand::<T>(...)?` cannot fail inside its arm. In `elementwise.rs` that is
  217 uncovered lines. Reaching 90% needs the extraction to become total, which means a panic on
  a caller-reachable path (rejected by design) or the single payload above.
- Integer and boolean operand-layout arms of the dot validators are unreachable because
  `DotGeneralAccumulation::overwrite` refuses those dtypes first (the scalar identities are
  float/complex only).

### Proposed next moves, in order

1. Strengthen the three weak tests with real assertions (spy call counts, a product value).
2. Add **one shared test-side macro** (`for_each_preset_scalar!`, `for_each_preset_pair!`) in a
   `pub(crate)` `#[cfg(test)]` support module and replace the five near-identical 7-arm macros
   each added test currently spells out.
3. Generate the remaining **hand-written per-dtype tables** from a library macro instead of
   chasing them dtype by dtype: `tensor_write_view` and `read_as_analytic_view`
   (`tenferro-linalg/src/cpu/backend.rs`, `tenferro-cpu/src/analytic.rs`),
   `execute_grouped_outer_typed`'s dispatch (`tenferro-cpu/src/dot_runtime.rs`),
   `validate_read_layout`/`validate_write_layout` (same file), `reclaim_*` and
   `pooled_zero_tensor` (same file), and the `types.rs` arms together with the removal.
   Evidence that this pays: the macro-driven seam `tenferro-tensor/src/dispatch.rs` measures
   **100%** line coverage (70/70), while the hand-written table in `elementwise.rs` sits at
   **81.39%** with 217 unreachable arm lines. Macro expansion attributes its regions to the
   macro, so one exercised path covers the definition instead of one line per dtype.

Do not edit `coverage-thresholds.json` to make the gate or the file-level figure look better.

## Commands

```bash
cd /home/shinaoka/tensor4all/tenferro-rs/.worktrees/scalar-composition-stage1
export CARGO_BUILD_JOBS=8

# compile gate
cargo check -j 8 --workspace --all-targets

# workspace suite
cargo nextest run -j 8 --workspace --cargo-profile ci --no-fail-fast

# the local PR gate (runs default features only — see the per-feature list below)
bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -j 8 -p <crate> --lib <filter>'

# repository rules review
python3 scripts/repository-rules-review.py --base origin/main --head HEAD --dry-run \
  --llm-skipped-reason "local deterministic review"

# CI profiles
python3 scripts/ci/run_profile.py {docs,extensions,ci-config,coverage,fmt,workspace-faer,workspace-blas}

# per-feature compile checks (the gate only builds default features, and two real breakages
# have hidden behind that: gpu/mod.rs 22 sites and cpu/backend.rs 29 pattern positions)
cargo check -j 8 -p tenferro-gpu --features cuda   --all-targets
cargo check -j 8 -p tenferro-gpu --features webgpu --all-targets
cargo check -j 8 -p tenferro-linalg --features cpu-faer --all-targets
cargo check -j 8 -p tenferro-linalg --features cpu-blas --all-targets
cargo check -j 8 -p tenferro-fft --features autodiff --all-targets
```

### Coverage measurement (the recipe that works)

```bash
# 1. instrumented + executed lines
cargo llvm-cov -j 8 --workspace --exclude tenferro-tutorial-code --profile ci \
  --ignore-run-fail --json --output-path /tmp/covNN.json
# 2. llvm-cov's own per-line data — the `--profile ci` is required, without it every count is 0
cargo llvm-cov report --profile ci --lcov --output-path /tmp/covNN.lcov
# 3. the gate
python3 scripts/check-coverage.py /tmp/covNN.json
# 4. added-line coverage against the branch diff
python3 - /tmp/covNN.lcov <<'PY'
import subprocess, re, pathlib, collections, sys
lcov_path = sys.argv[1]
base = subprocess.run(["git","merge-base","origin/main","HEAD"],capture_output=True,text=True).stdout.strip()
diff = subprocess.run(["git","diff","-U0",base,"HEAD","--","crates"],capture_output=True,text=True).stdout
added = collections.defaultdict(set); cur = None
for line in diff.splitlines():
    if line.startswith("+++ b/"): cur = line[6:]
    elif line.startswith("@@"):
        m = re.search(r"\+(\d+)(?:,(\d+))?", line)
        if m and cur: added[cur].update(range(int(m.group(1)), int(m.group(1)) + int(m.group(2) or 1)))
lcov = collections.defaultdict(dict); f = None
for line in pathlib.Path(lcov_path).read_text().splitlines():
    if line.startswith("SF:"): f = line[3:]
    elif line.startswith("DA:") and f:
        q = line[3:].split(","); lcov[f][int(q[0])] = int(q[1])
    elif line == "end_of_record": f = None
tot = cov = 0
for rel, lns in added.items():
    if not rel.endswith(".rs"): continue
    m = [k for k in lcov if k.endswith("/" + rel) or k == rel]
    if not m: continue
    d = lcov[m[0]]; ins = [x for x in lns if x in d]
    cov += sum(1 for x in ins if d[x] > 0); tot += len(ins)
print(f"added-line coverage: {cov}/{tot} = {cov / tot * 100:.1f}%")
PY
```

Add `--ignore-run-fail` to every `llvm-cov` run; the harness reports test files separately
(226 files, none of them tests).

## Hazards and lessons (each cost real time)

1. **Per-feature checks are mandatory.** `check-pr-fast` builds default features only. A
   feature-gated hole (`tenferro-fft --features autodiff`: the in-place arms had been routed
   through the shared accessor, so the plan could not write through the operand) was caught by
   the `docs` profile's negative compile check, not by the gate.
2. **`#[macro_export]` detaches if any item is inserted between its doc/attribute and the
   `macro_rules!`** — happened twice, both times silently (the macro simply stopped being
   exported). Check the attribute is adjacent after any scripted insertion.
3. **A `?` inside a macro runs in the caller's function.** Use `.and_then(|$x| $body)` instead.
   Type names injected by a macro resolve in the caller's scope: primitives are bare, other
   types need `$crate::`.
4. **The counting allocator is process-wide.** One allocation-measuring test per binary; adding
   a second to the same file perturbs the first one's numbers (that is why
   `erased_payload_allocation.rs` exists separately).
5. **llvm-cov does not instrument doctests**, and `cfg`-excluded lines are absent from the
   report. A naive added-line fraction therefore counts absent lines as uncovered; only join
   against lines that carry a record.
6. **`scripts/check-coverage.py` is a ratchet** over `coverage-thresholds.json`, whose values are
   frozen pre-existing baselines. Policy lives in `AGENTS.md`, not in that file.
7. **Generated artifacts must be regenerated, not hand-edited**: after a public-surface change
   run `python3 scripts/check-public-boundary-inventory.py --generate` (the `docs` profile
   reports it stale), and regenerate `docs/assets/dependency-footprint.svg` with the repo
   generator plus the Graphviz-WASM shim at `/tmp/vizjs/dot-shim.js`.
8. **Source-text contract tests must be re-pointed when a spelling changes** —
   `cubecl_launch_contract`, `gpu_linalg_source_contract`, and the `docs` profile's negative
   compile check (`scripts/test-doc-consistency.py`). Update a needle only in the file it names.
9. **The uniform arm template can change behaviour.** In `analytic.rs::pow_with_pool` the integer
   arms call a different body than the float arms; the crate's tests plus clippy's unused-item
   warning caught it.
10. **Never sweep patterns crate-wide.** A compile stops at the first error class, so a whole-
    crate pass reported "clean" with 67 errors outstanding. Convert one file at a time and let
    `E0164` name the pattern positions to revert.
11. `--all-features` pulls the Apple-only `accelerate-src`; `--no-default-features` intentionally
    `compile_error!`s; four pre-existing doc warnings and 43 pre-existing cuda clippy lints exist.

## Conversion recipe for the remaining arm work

1. Split the match at its arm heads; inline-substitute names in the helper argument list rather
   than renaming bindings (avoids the `scatter` binding/parameter collision).
2. Per file: rewrite the constructions, let the compiler report `E0164` for pattern positions,
   revert exactly those lines. Never crate-wide.
3. For a brace-bodied arm, insert `let` + accessor immediately after the arm's `{` — do not move
   braces; generated templates repeatedly mispaired them.
4. For exported macros use a nested accessor match with a `_ => $fallback` arm so the caller's
   error type is never named and no panic is added.
5. Extract one helper per module, in that module's own type-resolution form: fully-qualified
   `tenferro_tensor::…` where neither the trait nor the type is imported; a `type` alias where
   clippy's type-complexity fires; inline accessors only inside `FaerLinalg` impls, where the
   associated type shadows `TypedTensor`.
6. A helper's failure branch that is unreachable by construction must be reported as measured
   and documented, not papered over with `unreachable!` or hidden behind a test that cannot
   reach it.

## Pre-existing failures and quirks (not regressions)

- Trybuild: `eager_backend_capability_contract::eager_backend_capability_boundary`,
  `session_contract::execution_session_capability_cannot_project_or_escape_owner_borrow`,
  `storage_ui_compile_contracts` (the one that fails the workspace run).
- `workspace-blas`: six `full_svd_lstsq` tests plus `faer_full_svd_enters_once` — the CPU LAPACK
  provider does not implement full-matrices SVD; use the faer provider.
- Whole-workspace builds with one crate's GPU feature fail inside `tenferro-einsum`.

## Next steps, priority order

1. Strengthen the three weak tests (spy call counts, product value) — minutes.
2. Shared test-side preset macros, replacing five duplicated 7-arm macros.
3. Macro-generate the remaining hand-written per-dtype tables (list in the coverage section).
4. Once #1789's contract lands: the enum swap — `crates/tenferro-tensor/src/types.rs`
   (definition + 271 arms), the two non-library sites, and any hand-written table still naming
   variants. One indivisible change; re-run the per-feature matrix, the suite and coverage.
5. Once authorized: #1793's scalar-generic contraction slot.

The goal stays incomplete while items 4 and 5 are open, and while the coverage item is
partially met.
