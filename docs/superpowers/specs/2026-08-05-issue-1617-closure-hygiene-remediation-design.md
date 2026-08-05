# Issue #1617 Closure Hygiene Remediation Design

Date: 2026-08-05

Status: approved design

Authority: Issue #1617, umbrella Issue #1555, Issues #1564–#1569, and the accepted P11/P13 storage-ownership designs

## Purpose

Repair the post-merge closure defects found after PR #1616 without changing the storage-ownership architecture. The implementation remains sound; this remediation fixes the macOS-only test breakage, makes final evidence bind to one actual product candidate, adds a bounded fresh-execution closure mode, records real Apple/Metal evidence, strengthens the static-rank obligation, and applies verified local hygiene fixes.

The work uses one remediation branch and one non-squash PR with coherent commits. The final PR includes product/checker commits first and evidence-only descendants second.

## Finding classification

| Finding | Classification | Disposition |
|---|---|---|
| T0 Metal test does not compile | Auto Fix | Correct the mutable borrow and add a Linux cross-target type-check guard. |
| T1 evidence commit binding differs from merged product | Auto Fix | Freeze a new product/checker candidate and regenerate every dependent report against it. |
| T2 closure validates recorded documents without a fresh mode | Auto Fix | Add an explicit bounded `--reproduce` mode while retaining receipt-only default behavior. |
| T3 final Metal lane is a structured skip | Auto Fix | Run WebGPU and Metal on the Apple M5 Max and require all final lanes to pass. |
| T4 static-rank test is runtime-only | Auto Fix | Add one existing-`trybuild` compile-fail case for `Rank<2>` versus `Rank<3>`. |
| T5 `NonNull` reborrows are verbose | Auto Fix | Replace `NonNull::from(&mut *owner)` with `NonNull::from(owner)`. |
| T5 `cast_host_vec` relies on debug-only checks | Auto Fix, narrowed | Use release-active size and alignment assertions at the existing unsafe boundary; do not add a new type framework. |
| T5 `sha2` is unused | Stale / Out of Scope | Keep it: `tenferro-xla/src/stablehlo.rs` imports `Sha256`, and `cargo tree -i sha2 --workspace` resolves it through `tenferro-xla`. Record this evidence in the work log and PR ledger. |

## Global constraints

- Preserve one move-only owner per physical allocation.
- Keep duplicate, upload, download, synchronization, and materialization explicit.
- Do not change CUDA, WebGPU, or Apple provider namespaces.
- Do not add compatibility shims, hidden transfers, repeated hot-path validation, or new public APIs.
- Do not add hashes, signatures, attestations, nonces, hostile-runner defenses, or a new generic command framework.
- Identify tracked evidence with an exact Git commit, a clean tracked tree, and repository-relative paths.
- Keep receipt-only checking as the ordinary CI default; use fresh reproduction only for final closure or an explicit release-candidate audit.
- A Metal skip, zero executed Metal tests, or unavailable Apple compiler/device blocks final closure.

## Candidate and evidence model

The corrected model has two identities:

1. **Product candidate C** is the last commit containing Rust code, Python checkers, tests, CI configuration, public documentation, and durable design text.
2. **Evidence HEAD E** is a descendant containing only approved generated evidence, ledger-state updates, and the final work log.

C cannot contain a report that names C without a Git self-reference. Therefore generated benchmark, codegen, hardware, documentation-audit, freeze, and closure reports are legitimate evidence-only descendants. The accepted P13 design and the contract document will be corrected to use an explicit closed evidence allowlist that includes these report paths.

The closure checker must reject any `C..E` change outside that allowlist. In particular, evidence descendants cannot change Rust/Python implementation, checker semantics, tests, CI, public documentation, or durable design contracts.

Every generated report field named `candidate_commit` records C. The final ownership receipt is generated and checked on E because its existing contract binds execution to the current clean HEAD. Documentation will distinguish the product candidate from the evidence HEAD rather than calling both the candidate.

Any product/checker/test/CI/design correction after C creates a new C and invalidates affected evidence. A hardware rerun that changes no product file does not create a new candidate.

## Product fixes

### Metal mutable borrow

In `crates/tenferro-gpu/tests/integration/apple_context.rs`, bind the `Tensor::F32` inner tensor through `&mut managed` before calling `with_host_write(&mut self)`. The existing four Apple context tests remain the behavioral acceptance target.

### macOS-gated type-check guard

Add a separate Ubuntu CI job gated by the existing Rust change policy. It installs the `aarch64-apple-darwin` Rust target and runs:

```text
cargo check -p tenferro-gpu --features webgpu --test integration --target aarch64-apple-darwin
```

This is compile-only and does not claim Metal execution. A workflow-contract unit test pins the job and command so target-gated test code cannot silently disappear from CI.

### Static-rank compile contract

Add one UI fixture to the existing storage `trybuild` set. It constructs a `TypedTensor<f64, Rank<2>>`, obtains its view, and attempts to pass that view to a function requiring `TypedTensorView<'_, f64, Rank<3>>`. Compilation must fail with a type mismatch. No dependency is added.

### Storage hygiene

Replace all three `NonNull::from(&mut *owner)` expressions in `storage/group.rs` with `NonNull::from(owner)`.

Keep the existing generic `cast_host_vec` implementation. Replace its debug-only size and alignment checks with ordinary `assert_eq!` checks immediately before `Vec::from_raw_parts`. These checks run once at owner construction, not in an element loop, and prevent an internal invariant violation from reaching unsafe reinterpretation in release builds. The existing sealed `TensorScalar` and matching dtype dispatch remain the proof source; no new trait or conversion abstraction is introduced.

## Bounded closure reproduction

The closure checker interface becomes:

```text
python3 scripts/check-storage-redesign-closure.py \
  --report docs/worklogs/storage-redesign-closure.md

python3 scripts/check-storage-redesign-closure.py \
  --report docs/worklogs/storage-redesign-closure.md \
  --reproduce \
  --receipt /tmp/storage-ownership-receipt.json
```

Default mode remains low-cost: it validates the tracked freeze, performance, static-rank, hardware, documentation-audit, and closure records without rerunning heavy commands. The existing ownership checker remains responsible for receipt-only validation in ordinary CI.

`--reproduce` runs only this fixed set of existing commands:

```text
cargo test -p tenferro-tensor --test storage_public_api
cargo test -p tenferro-tensor --test storage_traversal_resolution
cargo test -p tenferro-tensor --test storage_static_rank
cargo test -p tenferro-tensor --test storage_compile_contract
cargo test -p tenferro-runtime scoped_immediate_provider_returns_borrowed_output
python3 scripts/ci/run_profile.py coverage
```

Before reproduction, the closure checker delegates the complete receipt, including obligation, argv, cwd, artifact, candidate, and exit-status validation, to the existing ownership checker. It does not duplicate receipt parsing or validation. The focused runtime command and coverage command are closure-only reproductions and must exit zero.

The checker collects results in memory and writes a passing closure report only after every command succeeds. The report records exact argv and exit status, not full logs or new attestations. A nonzero result, receipt mismatch, missing command, or interrupted run returns failure and does not write a passing report.

The contract document states:

- hosted ordinary CI uses receipt-only/default mode;
- the final closure audit and explicit release-candidate audit use `--reproduce`;
- coverage remains owned by the existing coverage profile and thresholds.

## Multi-host hardware evidence

The hardware checker continues to own execution and final validation. It gains one repeatable `--merge-report` input for combining temporary partial reports. No remote service or artifact digest is introduced.

The evidence sequence is:

```text
# Linux/CUDA host, checked out at C
python3 scripts/check-storage-hardware-matrix.py \
  --report /tmp/storage-hardware-linux.md \
  --lanes cpu,cuda2,cuda-ad

# Apple M5 Max, checked out at C
python3 scripts/check-storage-hardware-matrix.py \
  --report /tmp/storage-hardware-apple.md \
  --lanes webgpu,metal

# Evidence worktree descended from C
python3 scripts/check-storage-hardware-matrix.py \
  --report docs/testing/storage-hardware-matrix.md \
  --merge-report /tmp/storage-hardware-linux.md \
  --merge-report /tmp/storage-hardware-apple.md \
  --required-mode
```

A partial report records C and per-lane host, OS, architecture, command, device facts, test counts, outcome, and output tail. It is marked incomplete and is not accepted as final closure evidence.

Final merge performs only the checks needed for correctness:

- every partial report names C;
- each required lane occurs exactly once;
- no required lane is missing;
- every required lane has status `pass` and a positive executed test count;
- the final report records all per-host environments.

A skip remains available for ordinary partial development runs but is rejected by `--required-mode` and by final closure. Linux and Apple temporary reports need not be committed; the merged tracked report contains their concrete lane records.

## Evidence regeneration

After C is fixed, run all evidence producers against C and generate evidence-only descendants in this order:

1. contract freeze;
2. traversal performance;
3. static-rank codegen;
4. Linux/CUDA and Apple WebGPU/Metal hardware captures and merged matrix;
5. rendered documentation checks and source-blind audit candidate binding;
6. generate the default recorded-evidence closure report and commit all reports as a clean pre-reproduction evidence head;
7. generate a receipt bound to that clean head and run the closure checker in `--reproduce` mode;
8. commit the reproduced closure report and independent integration audit as final evidence HEAD E;
9. generate and validate a fresh final ownership receipt bound to clean E, then rerun the default closure validation without rewriting the report.

The default closure report in step 6 is the low-cost recorded-evidence result needed by the already-active closure obligation. The reproduction in step 7 replaces it with fresh command outcomes before final promotion. This order avoids a receipt/report self-reference: the receipt consumed by reproduction proves the clean pre-reproduction head, while the final receipt proves clean E. Every applicable report records C. The final work log records C, both receipt roles, hardware/toolchain facts, exact commands, finding dispositions, and limitations; it does not attempt to contain the hash of its own commit.

After E exists, Issue #1555 receives a correction comment naming C, E, the merged PR, and the final evidence paths. Issue #1617 records each T0–T5 disposition, including the retained `sha2` dependency evidence.

## Failure behavior

- If the cross-target Apple check fails, C is not frozen.
- If any product, checker, test, CI, or design file changes after C, create a new C and rerun affected evidence.
- If Metal compiles but runs zero tests, reports skip, or lacks a provider device, final hardware validation fails.
- If coverage, traversal, static-rank codegen, receipt validation, or reproduction fails, closure remains non-passing.
- If partial hardware reports disagree on C or duplicate a required lane, merge fails.
- If evidence descendants change a non-allowlisted path, closure fails.

No fallback relabels a skip or inconclusive result as pass.

## Verification

Focused checks run before broad checks:

```text
cargo test -p tenferro-gpu --features webgpu --test integration -- apple --nocapture
cargo check -p tenferro-gpu --features webgpu --test integration --target aarch64-apple-darwin
cargo test -p tenferro-tensor --test storage_compile_contract
cargo test -p tenferro-tensor --test storage_static_rank
cargo test -p tenferro-tensor
python3 -m unittest discover -s scripts/ci/tests
python3 scripts/test-storage-ownership-contracts-v2.py
```

PR-ready verification includes formatting, workspace and extension Clippy, the focused local PR gate, storage checker tests, freeze/static-rank/traversal/docs/closure checkers, coverage, trusted CUDA, Apple M5 Max WebGPU/Metal, repository-rules review, `git diff --check`, and a clean worktree.

## Commit and PR structure

The single remediation PR keeps these review units separate:

1. Metal mutable-borrow fix plus cross-target CI guard;
2. closure reproduction and multi-host hardware checker behavior plus tests and contract updates;
3. static-rank compile contract and storage hygiene fixes;
4. final work log and product/checker candidate C selection;
5. freeze/performance/codegen/documentation evidence;
6. merged Linux/CUDA/Apple hardware evidence;
7. reproduction receipt and independent closure evidence.

The PR is merged without squash so C and its evidence descendants remain identifiable.

## Completion criteria

The remediation is complete only when:

- every valid T0–T5 finding is fixed and independently reviewed;
- `sha2` is explicitly closed as stale with current source and dependency-tree evidence;
- all tracked evidence reports name the same C;
- every required hardware lane, including Metal, passes with a positive test count;
- closure `--reproduce` and the final ownership receipt pass;
- `C..E` contains only approved evidence changes;
- required PR checks pass and the remediation PR is merged;
- Issues #1617 and #1555 record the corrected candidate and evidence state.
