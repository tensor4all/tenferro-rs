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
