# Issues 1708–1713 Downstream Gaps

## Session summary

This single-PR stream addresses six accepted gaps reported by the completed latticeqcd-rs port: extension authoring, borrowed extension reads, multi-input AD, checked residual access, structured runtime diagnostics, and session-scoped BLAS-1 operations.

## Issue 1708: extension authoring facade

### Context read

- GitHub issue #1708
- `AGENTS.md`, relevant `REPOSITORY_RULES.md` public-surface, extension, documentation, test, and worklog rules
- `crates/tenferro-runtime/src/extension.rs`
- `crates/tenferro-runtime/src/runtime/extension.rs`
- `crates/tenferro-internal-extension-macros/src/lib.rs`
- `docs/guides/custom-operations.md`
- existing standard extension macro invocations and runtime tests

### Design and review gate

- Design: [`../design/extension-authoring-facade.md`](../design/extension-authoring-facade.md)
- Reviewer: `reviewer-flash-opencode-go` (`opencode-go/deepseek-v4-flash:medium`), read-only
- Rounds 1–2 timed out without reading all required artifacts; neither cleared the gate.
- Round 3 used the complete design plus bounded verified source facts.
- Pre-implementation verdict: **Correct-to-merge**.

### Decisions

- Reuse `tenferro_runtime::extension`; do not create another facade crate or framework.
- Re-export the existing author-facing contracts and proc macro while keeping generated adapters private.
- Make the macro's already-unused legacy `execute` argument optional rather than requiring downstream boilerplate.
- Prove downstream usability with an external fixture containing two independently generated families.
- Keep runtime/module/backend ownership explicit and preserve typed errors.

### Rejected or deferred

- A global registry, discovery, host-reference compatibility shim, generated public adapter types, implicit transfer/fallback, and new naming DSL were rejected by the accepted issue and design.
- Borrow/allocation optimization remains issue #1709; #1708 only establishes the public read-based adapter path.

### Verification

- `cargo test -p tenferro-internal-extension-macros`: 11 passed.
- `bash tests/run-extension-authoring-fixture.sh`: passed.
- `cargo test -p tenferro-runtime --test integration`: 146 passed.
- `cargo test -p tenferro-runtime --doc`: 414 passed.
- Targeted runtime/macro and external-fixture clippy: passed with warnings denied.
- Runtime package creation, doc snippets, public error docs, workflow contract tests, and actionlint: passed.
- Post-implementation reviewer: `reviewer-flash-opencode-go`; verdict **Correct-to-merge** after bounded follow-up evidence resolved the initial missing-evidence blockers.

The combined-PR local gate, coverage run, deterministic repository-rules review, and hosted docs build remain final-stream checks after issues #1709–#1713 are integrated.

### Remaining risks

No issue-#1708 blocker remains. Issue #1709 owns allocation behavior beyond establishing the borrowed `execute_reads` boundary.

## Issue 1709: borrowed extension reads

### Design and review gate

- Design: [`../design/borrowed-extension-reads.md`](../design/borrowed-extension-reads.md)
- Reviewer: `reviewer-flash-opencode-go`, read-only
- Pre-implementation verdict: **Correct-to-merge** after explicitly recording the shallow-clone/original-storage lifetime invariant.

### Decisions

- Reuse the existing read-based generated executor; the runtime already forwards `TensorRead` unchanged.
- Add only `TensorRead::as_slice<T>` as an allocation-free convenience over the existing `TensorView::as_slice<T>` contract.
- Keep noncompact materialization and backend transfer explicit; no macro/runtime policy or fallback was added.
- Use pointer identity and explicit callback counters as deterministic evidence. Allocator bytes compare borrowed execution against an equivalent output-construction baseline so output/runtime bookkeeping is not misclassified as input duplication.

### Verification

- `cargo test -p tenferro-tensor tensor_read_as_slice`: 5 passed.
- `cargo test -p tenferro-tensor --doc`: 327 passed.
- `bash tests/run-extension-authoring-fixture.sh`: passed with four-input pointer checks, explicit materialization count, backend rejection, and allocation comparison.
- Targeted tensor and external-fixture clippy: passed with warnings denied.
- Doc snippets and public error docs: passed.

Post-implementation reviewer: `reviewer-flash-opencode-go`; verdict **Correct-to-merge**. Coverage review and combined-PR gates remain pending.

## Issue 1710: multi-input traced JVP and VJP

### Design and review gate

- Design: [`../design/multi-input-traced-ad.md`](../design/multi-input-traced-ad.md)
- Reviewer: `reviewer-flash-opencode-go`, read-only
- Pre-implementation verdict: **Correct-to-merge**.

### Decisions

- Compile one semantic source, union one activity mask, run one cached semantic transform, and bind all derivative seeds once.
- Build all requested VJP traces from one shared derivative graph and one metadata/constraint analysis.
- Preserve request order and `None` for unreachable leaves; repeat duplicate VJP results without re-accumulation.
- Reject duplicate JVP leaves from the raw request before compilation; distinct tangents form one directional derivative.
- Keep single-input APIs source-compatible by delegating through the many-input helpers.

### Verification

- Multi-input integration tests cover two/four inputs, empty and unreachable requests, duplicate policies, metadata errors, transform-cache entry counts, and shared graph identity.
- A Wilson-like four-input action direct-VJP rule emits one four-output force extension; `compile_many` execution reports exactly one force callback and correct outputs.
- `cargo test -p tenferro-ad`: 589 passed across unit, integration, and doctest suites.
- Targeted clippy, doc snippets, and public error docs passed.

Post-implementation reviewer: `reviewer-flash-opencode-go`; verdict **Correct-to-merge**. Coverage review and combined-PR gates remain pending.

## Issue 1711: checked semantic AD residual access

### Design and review gate

- Design: [`../design/checked-semantic-ad-residual-access.md`](../design/checked-semantic-ad-residual-access.md)
- Reviewer: `reviewer-flash-opencode-go`, read-only
- Pre-implementation verdict: **Correct-to-merge**.

### Decisions

- Remove raw primal value slices from transpose/direct primal-VJP requests; linearization remains unrestricted.
- Add bounds-first, mask-checked value access with typed family/kind/index errors.
- Snapshot dtype/shape metadata into request-owned boxes so metadata remains available without exposing `ProgramValue` or aliasing the mutable builder.
- Keep `ResidualSpec` as the only tensor-retention authority; metadata snapshots retain no tensor data.
- Migrate standard and nested extension rules with no compatibility shim.

### Verification

- Focused semantic-extension tests cover undeclared input/output access, metadata-only access, bounds precedence, and absence of raw request accessors.
- Wilson-like four-input direct VJP covers inactive inputs and a non-unit cotangent while executing one force node.
- Debug standard-crate suites: 1347 passed; sparse autodiff: 25 passed; tropical autodiff: 77 passed.
- Release-mode semantic-extension tests: 5 passed.
- Workspace/nested checks, formatting, public error docs, and warning-denied clippy passed.

Post-implementation reviewer: `reviewer-flash-opencode-go`; verdict **Correct-to-merge**. Coverage review and combined-PR gates remain pending.

## Issue 1712: structured runtime failure reasons

### Design and review gate

- Design: [`../design/structured-runtime-failure-reasons.md`](../design/structured-runtime-failure-reasons.md)
- Reviewer: `reviewer-flash-opencode-go`, read-only
- Pre-implementation verdict: **Correct-to-merge**.

### Decisions

- Add one borrowed, non-exhaustive reason view while retaining every owned error and source link.
- Replace the ambiguous missing-extension `PrepareError::Unsupported` at the already-known compiled-preparation seam with a precise `PrepareError::MissingExtension` source.
- Classify primary errors before suppressed errors; traverse typed sources without strings or allocation.
- Keep real extension/provider unsupported failures distinct from missing registration.

### Verification

- Runtime error tests cover missing extension, no input ingress, unsupported operation, nested wrappers, suppressed-primary precedence, and `Other`.
- Runtime integration tests consume no-ingress reasons without downcasts; the external fixture verifies exact missing-extension family through the real compiled path.
- `cargo test -p tenferro-runtime`: 973 passed; external fixture passed.
- Runtime/fixture warning-denied clippy, formatting, and diff checks passed.

Post-implementation reviewer: `reviewer-flash-opencode-go`; verdict **Correct-to-merge**. Coverage review and combined-PR gates remain pending.

## Issue 1713: session-scoped BLAS-1

### Design and review gate

- Design: [`../design/session-blas1.md`](../design/session-blas1.md)
- Reviewer: `reviewer-flash-opencode-go`, read-only
- Pre-implementation verdict: **Correct-to-merge**.

### Decisions

- Add object-safe default-unsupported methods directly to `BackendSession`; unsupported backends cannot transfer or fall back.
- Reuse all-axis dot-general with lhs conjugation for VDOT and strided-kernel reduction for real norm-squared output.
- Reuse `ContractionScalar` with exact dtype matching.
- Implement fused compact-destination AXPBY as the accepted issue-specific RMW exception because strided-rs has no in-place read-modify-write primitive; validate shape/dtype/placement/compactness/overlap before mutation.
- Borrow compact x, materialize a noncompact same-placement x exactly once, and reject noncompact y.

### Verification and measurement

- Tensor/CPU suites: 1461 passed; focused BLAS-1 tests cover F32/F64/C32/C64, lhs-only complex conjugation, rank-N/empty/strided paths, invalid requests, safe backend-alias rejection, unsupported defaults, and a CG microfixture.
- Steady-state allocation probe over 100 compact 65,536-element AXPBY calls: cpu-faer allocated 0 bytes; cpu-blas retained one 1,520-byte provider bookkeeping allocation. Both are far below the 524,288-byte vector size and prove no full-size temporary. The cross-provider gate intentionally allows at most one allocation while requiring total bytes below one vector; it guards the accepted no-full-size-temporary contract rather than an inaccurate provider-independent absolute-zero claim.
- Default cpu-faer, cpu-blas-only, and combined cpu-faer+cpu-blas configurations check successfully.
- Warning-denied all-target clippy, CPU/tensor doctests, formatting, and diff checks passed.
- Release Criterion full 100-sample intervals: length 1,024 — fused 9.602–9.802 µs (1 thread), 13.967–14.200 µs (4 threads), manual 0.855–0.870 µs (1 thread), 1.246–1.272 µs (4 threads); length 65,536 — fused 21.585–22.025 µs (1 thread), 24.619–24.788 µs (4 threads), manual 58.489–59.790 µs (1 thread), 58.159–59.111 µs (4 threads). The session/provider boundary dominates tiny vectors; the fused path is 2.4–2.7x faster for the representative large vector. No multithread speedup claim is made for these sizes.
- VDOT follows provider reduction order; norm-squared deliberately returns the sum of squared magnitudes without `sqrt` and follows strided-kernel reduction order; tests use dtype-appropriate tolerances. AXPBY deliberately rejects noncompact y and materializes only noncompact x.
- Coverage: `semantic_extension.rs` 92.6%; `blas1.rs` 90.5%; the complete repository coverage gate passes.

Post-implementation reviewer: `reviewer-flash-opencode-go`; verdict **Correct-to-merge**, including a follow-up soundness verdict on compact mutable-view pointer provenance.

## Final combined verification and cross-phase audit

Candidate `a608aab624f14e00bdd3570087edae655e8e6a9a` was audited read-only after all task-local gates:

- **Specification and architecture** (`reviewer-gpt`): PASS; #1708–#1713 acceptance mappings, dependency ordering, extension/AD/runtime/session boundaries, and migration compatibility closed.
- **Rust safety and lifecycle** (`reviewer-flash`): PASS; TensorRead lifetime, checked-request ownership, AXPBY alias/pointer proof, and pooled rank-0 initialization were sound. Empty rank-0 shape product is exactly one.
- **Performance and parallelism** (`deepseek-brainstormer`): PASS against the accepted allocation/one-pass criteria; provider-specific bounded allocation evidence and full Criterion intervals are recorded above. Small-vector and multithread overhead are explicit limitations, not hidden claims.
- **Public API and documentation** (built-in `reviewer`): PASS; facades, typed errors, object safety, runnable examples, guides, architecture map, and feature combinations aligned.
- **CPU and NUMA** (built-in `scout`): PASS; production session routes, provider/strided ownership, placement rejection, serial/owned-Rayon paths, re-entry, and failure behavior aligned. Unchanged multi-socket behavior was not benchmarked.
- **GPU, XLA, and multi-GPU** (`gpt-brainstormer`): static/API/build PASS; hardware execution unavailable and unaffected by the diff because only CPU overrides the new default-unsupported methods. Hosted CI is the verification owner.
- **Integration auditor** (built-in `worker`): PASS; no Critical, Important, Minor, or cross-lane contradiction remained. Hardware-only limitations were classified as unaffected-by-diff and non-blocking.

Fresh combined gates:

- `python3 scripts/ci/run_profile.py coverage`: PASS, 193/193 files; `semantic_extension.rs` 92.6%, `blas1.rs` 90.5%.
- `python3 scripts/ci/run_profile.py docs`: PASS, including faer/BLAS tutorial binaries and rendered site checks.
- `bash scripts/check-pr-fast.sh --coverage-reviewed --test 'cargo test -p tenferro-cpu blas1'`: PASS.
- Committed-head deterministic repository-rules review: PASS; only the required recorded external-LLM-skip notice remained.
- Branch was behind `origin/main` by zero commits at audit time.
