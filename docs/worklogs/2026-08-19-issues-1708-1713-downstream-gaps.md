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
