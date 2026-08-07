# Issue #1591 Evidence-Backed Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close tenferro-rs #1591 with auditable evidence and clear the corresponding tenferro blocker from strided-rs retirement issues without changing production code.

**Architecture:** Audit a clean checkout of the latest `origin/main`, classify every #1591 requirement against current dependency, source, test, and provenance evidence, then publish the evidence before changing issue state. Treat absent upstream-only paths as not applicable, and draft a focused follow-up issue for any non-blocking regression-test gap rather than reviving the obsolete migration epic.

**Tech Stack:** Git, Cargo, Rust unit/integration tests, GitHub CLI (`gh`), Markdown, POSIX shell, Python 3 for Cargo metadata inspection.

## Global Constraints

- Work only in an isolated worktree based on the latest `origin/main`; do not modify `/home/shinaoka/tensor4all/tenferro-rs` or its unresolved files.
- Record the exact audited `origin/main` SHA. It must contain PR #1553 commit `6255590e`.
- Do not add, port, or modify production code while executing this closeout plan.
- Do not close #1591 if an active dependency/adapter remains, a correctness or safety item lacks evidence, a focused test fails, or copied-code provenance is unresolved.
- Keep strided-rs #202 open; this plan clears only its tenferro dependency blocker.
- Do not modify tenferro-rs #1592.
- Follow `ai/contribution-workflows/issue-intake.md` and obtain user confirmation before creating any newly proposed follow-up issue.
- Put build products under `/tmp/tenferro-1591-target` and remove them, generated `Cargo.lock`, and all temporary drafts before completion.
- Shared Cargo caches under `~/.cargo` must remain untouched.

---

## File and State Map

**Repository files inspected, never modified during execution:**

- `Cargo.toml` — workspace dependency declarations.
- `crates/tenferro-cpu/Cargo.toml` — CPU features and direct dependencies.
- `crates/tenferro-cpu/src/backend/tests.rs` — contract preventing dependency reintroduction.
- `crates/tenferro-cpu/src/exec_session.rs` — initialized dot-output allocation and session entry.
- `crates/tenferro-cpu/src/dot_runtime.rs` — validation, provider routing, canonical temporary allocation, and fallback policy.
- `crates/tenferro-cpu/src/gemm/mod.rs` — batch-layout analysis, checked offsets, Faer/BLAS descriptors, and provider execution.
- `crates/tenferro-cpu/src/provider.rs` — provider request and layout contracts.
- `crates/tenferro-cpu/src/dot_runtime/tests.rs` — rank, layout, and mutation-order tests.
- `crates/tenferro-cpu/src/gemm/tests.rs` — fusion, overflow, provider, and accumulation tests.
- `crates/tenferro-cpu/src/provider/tests.rs` — direct Faer/BLAS provider tests.
- `crates/tenferro-cpu/src/tests/cpu_tests/dot_structural_analytic.rs` — end-to-end batch fallback, negative-stride input, and dot-general tests.
- `crates/tenferro-cpu/tests/integration/inject_tests.rs` — injected BLAS call-through test.
- `docs/design/einsum.md` and `docs/worklogs/2026-07-31-issue-1546-faer-policy.md` — active ownership and historical rationale.

**Temporary local artifacts created and deleted:**

- `/tmp/tenferro-1591-evidence.md` — audit matrix and command results.
- `/tmp/tenferro-1591-closeout.md` — final #1591 comment body.
- `/tmp/tenferro-1591-followup.md` — exact draft for any test-only follow-up.
- `/tmp/tenferro-1591-target/` — isolated Cargo target directory.
- `/tmp/tenferro-issue-{1591,1592,198,199,201,202}-before.json` — pre-change GitHub snapshots.

**Remote state modified only after review:**

- `tensor4all/tenferro-rs#1591` — evidence comment, then close as completed.
- `tensor4all/strided-rs#199` — Phase 1 resolution comment.
- `tensor4all/strided-rs#202` — blocker-cleared comment; remains open.
- `tensor4all/strided-rs#198` and `#201` — cross-linked disposition comments.

---

### Task 1: Pin the Audit Baseline and Prove Dependency Removal

**Files:**
- Inspect: `Cargo.toml`
- Inspect: `crates/tenferro-cpu/Cargo.toml`
- Inspect: `crates/tenferro-cpu/src/backend/tests.rs`
- Create temporarily: `/tmp/tenferro-1591-evidence.md`

**Interfaces:**
- Consumes: latest `origin/main`, PR #1553 commit `6255590e`, GitHub issue states.
- Produces: a pinned SHA, dependency proof, pre-change issue snapshots, and a clean stop/go decision for Task 2.

- [ ] **Step 1: Verify isolation and refresh the remote baseline**

Run:

```bash
set -euo pipefail
GIT_DIR=$(cd "$(git rev-parse --git-dir)" && pwd -P)
GIT_COMMON=$(cd "$(git rev-parse --git-common-dir)" && pwd -P)
test "$GIT_DIR" != "$GIT_COMMON"
test -z "$(git rev-parse --show-superproject-working-tree 2>/dev/null)"
git fetch origin
MAIN_SHA=$(git rev-parse origin/main)
printf '%s\n' "$MAIN_SHA"
git merge-base --is-ancestor 6255590e "$MAIN_SHA"
```

Expected: isolation assertions pass, one 40-character SHA is printed, and the ancestry check exits 0. If not, stop.

- [ ] **Step 2: Confirm the execution checkout has current main source**

Run:

```bash
set -euo pipefail
MAIN_SHA=$(git rev-parse origin/main)
git diff --quiet "$MAIN_SHA" -- \
  Cargo.toml \
  crates/tenferro-cpu/Cargo.toml \
  crates/tenferro-cpu/src \
  docs/design/einsum.md \
  docs/worklogs/2026-07-31-issue-1546-faer-policy.md
```

Expected: exit 0. Documentation-only plan/spec commits may differ outside these paths. If source differs, rebase the documentation branch onto `origin/main` or create a fresh managed worktree before continuing.

- [ ] **Step 3: Snapshot all relevant GitHub issues before mutation**

Run:

```bash
set -euo pipefail
gh issue view 1591 --repo tensor4all/tenferro-rs \
  --json number,title,state,url,body,comments > /tmp/tenferro-issue-1591-before.json
gh issue view 1592 --repo tensor4all/tenferro-rs \
  --json number,title,state,url,body,comments > /tmp/tenferro-issue-1592-before.json
for issue in 198 199 201 202; do
  gh issue view "$issue" --repo tensor4all/strided-rs \
    --json number,title,state,url,body,comments \
    > "/tmp/tenferro-issue-${issue}-before.json"
done
python3 - <<'PY'
import json
from pathlib import Path
for path in sorted(Path('/tmp').glob('tenferro-issue-*-before.json')):
    issue = json.loads(path.read_text())
    print(issue['url'], issue['state'], issue['title'])
PY
```

Expected: all six URLs and current states print. #1591, #198, #199, #201, and #202 should still be open. If an expected state or relationship changed, stop and re-review the design before posting.

- [ ] **Step 4: Prove manifests and active source no longer depend on the crate**

Run:

```bash
set -euo pipefail
test ! -e crates/tenferro-cpu/src/gemm/strided_dot.rs
! grep -n 'strided-einsum2' Cargo.toml crates/tenferro-cpu/Cargo.toml
git grep -n 'strided[-_]einsum2' -- \
  Cargo.toml \
  crates/tenferro-cpu \
  REPOSITORY_RULES.md \
  docs/design \
  docs/worklogs || true
```

Expected: the first two checks exit 0. Remaining grep output may contain only intentional absence assertions and historical attribution, notably `crates/tenferro-cpu/src/backend/tests.rs` and the issue #1546 work log. Any active adapter, dependency, or rule requiring `strided-einsum2` is a stop condition.

- [ ] **Step 5: Verify Cargo metadata independently of text search**

Run:

```bash
set -euo pipefail
export CARGO_TARGET_DIR=/tmp/tenferro-1591-target
test ! -e Cargo.lock
cargo metadata --format-version 1 --no-deps | python3 -c '
import json, sys
metadata = json.load(sys.stdin)
found = [
    (package["name"], dependency["name"])
    for package in metadata["packages"]
    for dependency in package.get("dependencies", [])
    if dependency["name"] == "strided-einsum2"
]
assert not found, found
print("strided-einsum2 dependencies: 0")
'
```

Expected: `strided-einsum2 dependencies: 0`. Preserve any generated `Cargo.lock` only until Task 4 cleanup; never commit it.

- [ ] **Step 6: Record the baseline and dependency evidence**

Run:

```bash
MAIN_SHA=$(git rev-parse origin/main)
cat > /tmp/tenferro-1591-evidence.md <<EOF
# tenferro-rs #1591 closeout evidence

- Audited origin/main: \`$MAIN_SHA\`
- PR #1553 removal commit: \`6255590e\`
- \`6255590e\` is an ancestor of the audited main: yes
- Active \`strided-einsum2\` Cargo dependencies: 0
- \`crates/tenferro-cpu/src/gemm/strided_dot.rs\`: absent
- Production-code changes made by this audit: none

## Requirement matrix

| Requirement | Initial status | Evidence |
|---|---|---|
| Remove tenferro dependency | Satisfied | manifests, Cargo metadata, absence contract |

The remaining ownership, carry-list, provenance, and blocker dispositions are
recorded in the Task 2 sections below after source tracing and tests complete.
EOF
```

Expected: the file contains the exact audited SHA and no unresolved shell variables.

---

### Task 2: Trace Ownership, Classify the Carry List, and Run Focused Verification

**Files:**
- Inspect: `crates/tenferro-cpu/src/dot_runtime.rs`
- Inspect: `crates/tenferro-cpu/src/gemm/mod.rs`
- Inspect: `crates/tenferro-cpu/src/provider.rs`
- Inspect: test and documentation files listed in the file map
- Modify temporarily: `/tmp/tenferro-1591-evidence.md`
- Create temporarily: `/tmp/tenferro-1591-followup.md`

**Interfaces:**
- Consumes: Task 1's pinned baseline and dependency proof.
- Produces: a complete requirement matrix, focused test results, a provenance conclusion, and—only if confirmed necessary—an exact test-hardening issue draft.

- [ ] **Step 1: Trace the active local execution path**

Run:

```bash
git grep -n -E \
  'allocate_dot_output|pool_acquire_zeroed|execute_dot_general_into|execute_gemm_plan|execute_faer_request_typed|execute_blas_request_typed|blas_descriptor_unsupported|allocate_canonical_operand|pooled_zero_tensor|faer_parallelism' \
  -- crates/tenferro-cpu/src/exec_session.rs \
     crates/tenferro-cpu/src/dot_runtime.rs \
     crates/tenferro-cpu/src/gemm/mod.rs \
     crates/tenferro-cpu/src/provider.rs \
     crates/tenferro-cpu/src/context.rs
git grep -n -E \
  'validate_dot_general|validate_axis_groups|try_fuse_dims|checked_batch_offset|checked_view_batch_offset|blas_output_layout_supported' \
  -- crates/tenferro-cpu/src/dot_runtime.rs \
     crates/tenferro-cpu/src/gemm/mod.rs
```

Expected: all routing, validation, allocation, and provider symbols resolve under `tenferro-cpu`; no call resolves through `strided-einsum2`.

- [ ] **Step 2: Classify each known upstream problem against current source**

Read the matched functions completely, then append this evidenced disposition to `/tmp/tenferro-1591-evidence.md`:

```bash
cat >> /tmp/tenferro-1591-evidence.md <<'EOF'
## Current-path disposition

| #1591 carry item | Status | Current-path evidence |
|---|---|---|
| Independent fusion orders can mispair batches | Satisfied by a different design | `analyse_gemm` accepts a direct batched plan only after each batch layout has identity stride order and can be fused; otherwise the engine canonicalizes operands. `test_dot_general_falls_back_for_unfusable_lhs_batch_layout` checks logical-coordinate results. |
| Negative destination stride becomes a positive BLAS leading dimension | Not applicable to the retired adapter; safely rejected locally | `blas_output_layout_supported` requires positive supported output layout and `blas_descriptor_unsupported` returns `Layout(Output)` before pointer execution. Generic layout validation separately checks reachable negative-stride views. |
| Rank arrays are indexed before validation | Satisfied | `validate_axis_groups`/`validate_dot_general` precede plan/provider construction; rank-parity tests cover competing invalid configurations. |
| O(elements) `HashSet` injectivity preflight | Not applicable | The upstream uninitialized destination validator is absent from the tenferro dot-general route. Tenferro uses validated tensor layouts and initialized output storage. |
| Conjugation clones a non-conjugated full backing buffer | Not applicable to current implementation | Faer receives conjugation flags directly; canonical materialization fuses only the required transform. |
| `beta == 0` may read uninitialized BLAS output | Not a memory-safety path in tenferro | Dot-general temporaries are acquired zero-initialized and providers receive initialized `TensorWrite` destinations. No upstream overwrite-only API is used. |
| Initialized BLAS layout path panics | Satisfied | Provider capability checks return typed `CpuProviderUnsupported` before mutation. |
| Batch/group offset overflow | Satisfied | `checked_batch_offset` and `checked_view_batch_offset` reject conversion, multiplication, and addition overflow before provider execution. |
| Faer uses ambient/ad-hoc parallelism | Satisfied | `execute_faer_request_typed` receives `CpuExecutionContext::faer_parallelism`; issue #1546 work log and source-contract tests record the bounded policy. |
| Faer typed `MaybeUninit` support from strided-rs #198 | Not transferred | The current tenferro Faer dot path consumes initialized `TensorWrite`, so #198 is not a downstream prerequisite. |
EOF
```

Expected: every row is supported by a named function or test. If reading the complete functions contradicts any row, correct the row; if it cannot be reclassified with evidence, stop and keep #1591 open.

- [ ] **Step 3: Check provenance and record the actual history**

Run:

```bash
set -euo pipefail
git show --stat --oneline 6255590e
git show --format=fuller --no-ext-diff 6255590e -- \
  Cargo.toml \
  crates/tenferro-cpu/Cargo.toml \
  crates/tenferro-cpu/src/backend/tests.rs \
  docs/design/einsum.md \
  docs/worklogs/2026-07-31-issue-1546-faer-policy.md
git log --follow --oneline -- crates/tenferro-cpu/src/gemm/mod.rs | head -40
git log --follow --oneline -- crates/tenferro-cpu/src/dot_runtime.rs | head -40
git blame -L 1552,1642 -- crates/tenferro-cpu/src/gemm/mod.rs
```

Then append:

```bash
cat >> /tmp/tenferro-1591-evidence.md <<'EOF'
## Provenance

PR #1553 / `6255590e` removed stale manifests, feature wiring, and obsolete
contract text after the active local Faer/BLAS path already existed. It did not
copy the retired upstream crate into tenferro. Current implementation ownership
and rationale remain visible through file history and
`docs/worklogs/2026-07-31-issue-1546-faer-policy.md`.
EOF
```

Expected: history supports the statement. If blame/history shows materially copied implementation not covered by existing attribution, stop and resolve provenance before closure.

- [ ] **Step 4: Run focused default/Faer tests**

Run:

```bash
set -euo pipefail
export CARGO_TARGET_DIR=/tmp/tenferro-1591-target
cargo test -p tenferro-cpu --lib cpu_tensor_kernel_parallel_features_are_wired
cargo test -p tenferro-cpu --lib axis_groups_match_existing_rank_validation_through_rank_seventy
cargo test -p tenferro-cpu --lib dot_general_validation_accepts_checked_negative_stride_output
cargo test -p tenferro-cpu --lib checked_batch_offset_reports_batch_conversion_overflow
cargo test -p tenferro-cpu --lib test_dot_general_falls_back_for_unfusable_lhs_batch_layout
cargo test -p tenferro-cpu --lib faer_provider_covers_f32_c32_and_c64_conjugation
cargo test -p tenferro-cpu --lib faer_provider_executes_non_unit_strides_and_strided_batches
```

Expected: every command exits 0 and reports its named test passed. Any failure is a stop condition; do not weaken or skip the failing test.

- [ ] **Step 5: Run the injected BLAS call-through lane**

Run:

```bash
set -euo pipefail
export CARGO_TARGET_DIR=/tmp/tenferro-1591-target
cargo test -p tenferro-cpu \
  --no-default-features \
  --features cpu-blas,provider-inject \
  --test integration \
  provider_inject_dot_general_uses_registered_blas
```

Expected: the registered BLAS provider is called and the named integration test passes. Failure is a stop condition.

- [ ] **Step 6: Confirm whether exact carry-list regression tests are absent**

Run:

```bash
set -euo pipefail
! git grep -n -E \
  'negative.*(destination|output).*(BLAS|blas)|BLAS.*negative.*(destination|output)' \
  -- crates/tenferro-cpu/src/provider/tests.rs \
     crates/tenferro-cpu/src/tests/cpu_tests/dot_structural_analytic.rs
! git grep -n -E \
  'batch.*\[2, *3\].*\[3, *1\].*\[1, *2\]|independent.*fusion.*order' \
  -- crates/tenferro-cpu/src
```

Expected at the design baseline: both commands exit 0, confirming that source behavior is safe but the issue's exact negative-output and mixed-fusion regression fixtures were not carried verbatim. If either test now exists, record it as stronger evidence and omit the corresponding item from the follow-up draft.

- [ ] **Step 7: Draft one focused test-hardening issue and pause for confirmation**

First read `ai/contribution-workflows/issue-intake.md` completely. If Step 6 confirms both gaps, write:

```bash
cat > /tmp/tenferro-1591-followup.md <<'EOF'
## Goal

Lock in the current CPU dot-general safety behavior with provider-level regressions for the two concrete cases called out during the retired `strided-einsum2` closeout.

## Current behavior

The current implementation is source-safe:

- `analyse_gemm` rejects independently unfusable batch layouts and canonicalizes them instead of independently fusing A/B/C batch orders.
- `blas_descriptor_unsupported` rejects unsupported output layouts before entering BLAS, and `blas_output_layout_supported` rejects non-positive output leading strides.

Existing tests cover unfusable logical batches, generic negative-stride output validation, negative-stride input fallback, and output-layout rejection without mutation, but not the exact upstream regression fixtures.

## Requested regression coverage

1. Exercise batch shape `[2, 3]` with logical A batch strides `[3, 1]` and B batch strides `[1, 2]`; assert every logical batch coordinate matches a scalar/reference calculation.
2. Construct a negative-stride BLAS destination request; assert `CpuProviderUnsupported::Layout(CpuOperand::Output)` is returned before provider mutation and the destination remains unchanged.

## Acceptance criteria

- Both tests fail if independent batch layouts are blindly fused or a negative output stride reaches BLAS as a positive leading dimension.
- Tests use existing private provider/runtime APIs; no new public API or dependency is added.
- Focused Faer/default and `provider-inject` BLAS lanes pass.

## Provenance

The fixtures are derived from the bug descriptions recorded in tenferro-rs #1591 and the linked strided-rs issues. No upstream implementation code is required.
EOF
```

Present the title `test(cpu): cover retired strided-einsum2 carry-list regressions` and this body to the user. Do **not** run `gh issue create` without explicit confirmation. If confirmed, create it exactly with:

```bash
FOLLOWUP_URL=$(gh issue create \
  --repo tensor4all/tenferro-rs \
  --title 'test(cpu): cover retired strided-einsum2 carry-list regressions' \
  --body-file /tmp/tenferro-1591-followup.md)
printf '%s\n' "$FOLLOWUP_URL" | tee /tmp/tenferro-1591-followup-url
```

Expected: either the user declines and the draft remains a documented residual test gap, or one new issue URL is recorded. The source-safety findings remain the basis for #1591 closeout.

- [ ] **Step 8: Record verification results and final stop/go decision**

Because every earlier test command used `set -e`, reaching this step proves all named commands exited 0. Record them exactly, then check the audit artifact and worktree:

```bash
cat >> /tmp/tenferro-1591-evidence.md <<'EOF'
## Focused verification

Passed:

- `cargo test -p tenferro-cpu --lib cpu_tensor_kernel_parallel_features_are_wired`
- `cargo test -p tenferro-cpu --lib axis_groups_match_existing_rank_validation_through_rank_seventy`
- `cargo test -p tenferro-cpu --lib dot_general_validation_accepts_checked_negative_stride_output`
- `cargo test -p tenferro-cpu --lib checked_batch_offset_reports_batch_conversion_overflow`
- `cargo test -p tenferro-cpu --lib test_dot_general_falls_back_for_unfusable_lhs_batch_layout`
- `cargo test -p tenferro-cpu --lib faer_provider_covers_f32_c32_and_c64_conjugation`
- `cargo test -p tenferro-cpu --lib faer_provider_executes_non_unit_strides_and_strided_batches`
- `cargo test -p tenferro-cpu --no-default-features --features cpu-blas,provider-inject --test integration provider_inject_dot_general_uses_registered_blas`
EOF

grep -n -E 'Pending|TBD|TODO|unknown|unresolved' /tmp/tenferro-1591-evidence.md && exit 1 || true
git diff --check
git status --short
```

Expected: no unresolved marker in the evidence, no repository diff, and only an untracked generated `Cargo.lock` if Cargo created one. Any production or tracked-file change must be reverted or explained before Task 3.

---

### Task 3: Publish the tenferro Closeout and Close #1591

**Files:**
- Consume: `/tmp/tenferro-1591-evidence.md`
- Create temporarily: `/tmp/tenferro-1591-closeout.md`
- Modify remotely: `tensor4all/tenferro-rs#1591`

**Interfaces:**
- Consumes: Task 2's complete matrix, passing tests, provenance conclusion, and optional confirmed follow-up URL.
- Produces: a rendered evidence comment and a closed #1591 with a recorded comment URL.

- [ ] **Step 1: Build the exact closeout comment**

Use the confirmed issue URL when one was created; otherwise record that only the reviewed draft exists:

```bash
MAIN_SHA=$(git rev-parse origin/main)
if test -f /tmp/tenferro-1591-followup-url; then
  FOLLOWUP_LINE="Residual test hardening: $(cat /tmp/tenferro-1591-followup-url)"
else
  FOLLOWUP_LINE='Residual test hardening: exact fixture draft reviewed; no new issue was authorized during this closeout.'
fi
cat > /tmp/tenferro-1591-closeout.md <<EOF
The tenferro side of this migration is complete at \`origin/main\` \`$MAIN_SHA\`.

PR #1553 / commit \`6255590e\` removed the stale \`strided-einsum2\` dependency after confirming that active Faer and BLAS dot-general execution was already owned by \`tenferro-cpu\`. No upstream source transplant was needed for that removal.

### Requirement disposition

| Requirement | Result |
|---|---|
| Remove workspace/CPU dependency and adapter | Complete: no Cargo dependency and no \`gemm/strided_dot.rs\` |
| Keep dot-general and batched GEMM ownership in tenferro-cpu | Complete: validation, planning, checked batch offsets, provider dispatch, and writeback are local |
| Independent batch fusion ordering | Current direct planning requires fusible identity order; unfusable layouts use canonical materialization |
| Negative BLAS destination stride | Retired adapter path is absent; local BLAS capability checks reject unsupported output layout before execution |
| Rank/config validation and overflow | Shared validation and checked offset/product paths run before provider execution |
| Faer parallelism | Comes from \`CpuExecutionContext::faer_parallelism\`, as audited in #1546 |
| Upstream uninitialized-output/HashSet/\`beta == 0\` concerns | Not applicable: current tenferro dot-general providers receive initialized \`TensorWrite\` output |
| Conjugation cloning concern | Not applicable: local providers carry conjugation flags and layout materialization fuses required transforms |
| strided-rs #198 transfer | Not required: tenferro does not consume the upstream uninitialized Faer destination API |
| Provenance | #1553 removed declarations and stale text; file history and the #1546 work log preserve the actual local-ownership history |

### Focused verification

- dependency absence contract
- rank-validation parity through rank 70
- checked negative-stride output metadata
- checked batch-offset overflow
- unfusable logical batch fallback
- Faer complex conjugation and strided-batch execution
- injected BLAS dot-general call-through

$FOLLOWUP_LINE

This closes the tenferro migration issue. The remaining deletion/release work stays in tensor4all/strided-rs#202.
EOF
```

Expected: the file contains the actual SHA, no shell placeholders, and no claim stronger than `/tmp/tenferro-1591-evidence.md`.

- [ ] **Step 2: Review the rendered body before posting**

Run:

```bash
cat /tmp/tenferro-1591-closeout.md
grep -n -E '\$\{|<[^>]+>|TBD|TODO|unknown|unresolved' /tmp/tenferro-1591-closeout.md && exit 1 || true
gh issue view 1591 --repo tensor4all/tenferro-rs --json state --jq .state
```

Expected: no placeholder-like text and issue state `OPEN`. Compare every row with the evidence file. If the issue is no longer open or the body overstates evidence, stop.

- [ ] **Step 3: Post the comment and capture its URL**

Run:

```bash
CLOSEOUT_COMMENT_URL=$(gh issue comment 1591 \
  --repo tensor4all/tenferro-rs \
  --body-file /tmp/tenferro-1591-closeout.md)
printf '%s\n' "$CLOSEOUT_COMMENT_URL" | tee /tmp/tenferro-1591-closeout-comment-url
```

Expected: one GitHub comment URL under `tensor4all/tenferro-rs/issues/1591`.

- [ ] **Step 4: Verify the comment, then close as completed**

Run:

```bash
set -euo pipefail
gh issue view 1591 --repo tensor4all/tenferro-rs --comments | tail -120
gh issue close 1591 --repo tensor4all/tenferro-rs --reason completed
test "$(gh issue view 1591 --repo tensor4all/tenferro-rs --json state --jq .state)" = CLOSED
```

Expected: the rendered comment is intact and #1591 becomes `CLOSED`. If rendering or links are wrong, edit the comment before closing rather than adding corrective noise.

---

### Task 4: Clear the strided-rs Blocker, Cross-Link Disposition, and Clean Up

**Files:**
- Modify remotely: `tensor4all/strided-rs#198`, `#199`, `#201`, `#202`
- Verify unchanged remotely: `tensor4all/tenferro-rs#1592`
- Delete: all `/tmp/tenferro-1591-*` artifacts and any generated `Cargo.lock`

**Interfaces:**
- Consumes: closed tenferro-rs #1591 and its closeout comment URL.
- Produces: four strided-rs update URLs, an open #202 without the tenferro blocker, a clean worktree, and a concise final report.

- [ ] **Step 1: Post the Phase 1 resolution to strided-rs #199**

Run:

```bash
CLOSEOUT_COMMENT_URL=$(cat /tmp/tenferro-1591-closeout-comment-url)
cat > /tmp/tenferro-1591-strided-199.md <<EOF
Downstream Phase 1 status: tenferro-rs removed its stale \`strided-einsum2\` dependency in tensor4all/tenferro-rs#1553 (commit \`6255590e\`) after confirming that active Faer/BLAS contraction execution was already local.

The evidence-backed closeout is tensor4all/tenferro-rs#1591; evidence comment: $CLOSEOUT_COMMENT_URL.

No source transplant or Faer \`MaybeUninit\` API transfer was required. The tenferro dependency condition for retirement is complete.
EOF
STRIDED_199_URL=$(gh issue comment 199 --repo tensor4all/strided-rs \
  --body-file /tmp/tenferro-1591-strided-199.md)
printf '%s\n' "$STRIDED_199_URL" | tee /tmp/tenferro-1591-strided-199-url
```

Expected: one comment URL for strided-rs #199.

- [ ] **Step 2: Clear only the tenferro blocker on strided-rs #202**

Run:

```bash
CLOSEOUT_COMMENT_URL=$(cat /tmp/tenferro-1591-closeout-comment-url)
cat > /tmp/tenferro-1591-strided-202.md <<EOF
The tenferro dependency blocker is cleared: tensor4all/tenferro-rs#1591 is closed with evidence at $CLOSEOUT_COMMENT_URL, and PR tensor4all/tenferro-rs#1553 removed the stale Cargo edge.

This issue should remain open for its remaining crate deletion, release, and repository cleanup steps; this update does not claim those steps are complete.
EOF
STRIDED_202_URL=$(gh issue comment 202 --repo tensor4all/strided-rs \
  --body-file /tmp/tenferro-1591-strided-202.md)
printf '%s\n' "$STRIDED_202_URL" | tee /tmp/tenferro-1591-strided-202-url
test "$(gh issue view 202 --repo tensor4all/strided-rs --json state --jq .state)" = OPEN
```

Expected: the comment posts and #202 remains `OPEN`.

- [ ] **Step 3: Record #198's non-transfer disposition on both linked issues**

Run:

```bash
CLOSEOUT_COMMENT_URL=$(cat /tmp/tenferro-1591-closeout-comment-url)
cat > /tmp/tenferro-1591-strided-198.md <<EOF
The tenferro closeout found no downstream transfer target for this API. Current tenferro CPU Faer dot-general execution consumes initialized \`TensorWrite\` destinations and does not use the retired crate's uninitialized Faer destination path.

Evidence: $CLOSEOUT_COMMENT_URL. Please handle this issue's final close/retain decision as part of #201/#202 rather than transferring it to tenferro.
EOF
STRIDED_198_URL=$(gh issue comment 198 --repo tensor4all/strided-rs \
  --body-file /tmp/tenferro-1591-strided-198.md)
cat > /tmp/tenferro-1591-strided-201.md <<EOF
Cross-repository disposition for #198: no tenferro transfer is required because tenferro's active Faer dot path uses initialized output storage. The supporting audit is $CLOSEOUT_COMMENT_URL.

The remaining #198 disposition can therefore be resolved locally alongside this transfer/closure ledger and #202.
EOF
STRIDED_201_URL=$(gh issue comment 201 --repo tensor4all/strided-rs \
  --body-file /tmp/tenferro-1591-strided-201.md)
printf '%s\n%s\n' "$STRIDED_198_URL" "$STRIDED_201_URL" \
  | tee /tmp/tenferro-1591-strided-198-201-urls
```

Expected: one comment URL for each of #198 and #201. Do not close either issue unless its owners separately request that action.

- [ ] **Step 4: Verify all final remote states and links**

Run:

```bash
set -euo pipefail
test "$(gh issue view 1591 --repo tensor4all/tenferro-rs --json state --jq .state)" = CLOSED
test "$(gh issue view 202 --repo tensor4all/strided-rs --json state --jq .state)" = OPEN
for spec in \
  'tensor4all/tenferro-rs 1591' \
  'tensor4all/strided-rs 198' \
  'tensor4all/strided-rs 199' \
  'tensor4all/strided-rs 201' \
  'tensor4all/strided-rs 202'; do
  set -- $spec
  gh issue view "$2" --repo "$1" --comments | tail -80
done
gh issue view 1592 --repo tensor4all/tenferro-rs --json number,state,url,title
```

Expected: comments render with valid cross-links, #1591 is closed, #202 remains open, and #1592 has received no update from this plan.

- [ ] **Step 5: Remove all generated build and temporary artifacts**

Run:

```bash
set -euo pipefail
rm -rf /tmp/tenferro-1591-target
rm -f Cargo.lock
rm -f /tmp/tenferro-1591-*.md \
      /tmp/tenferro-1591-*-url \
      /tmp/tenferro-1591-*-urls \
      /tmp/tenferro-issue-*-before.json
test ! -e /tmp/tenferro-1591-target
test -z "$(find /tmp -maxdepth 1 -name 'tenferro-1591-*' -print -quit)"
git status --short
git diff --check
```

Expected: no temporary audit/build artifacts remain and the repository has no execution-time changes. The already committed design and plan documents are the only branch difference from `origin/main`.

- [ ] **Step 6: Report the closeout concisely**

Report:

- audited `origin/main` SHA;
- #1591 closeout comment URL and closed state;
- #198/#199/#201/#202 update URLs;
- optional follow-up issue URL or the fact that only a draft was produced;
- focused test commands and pass/fail counts;
- confirmation that #202 remains open and #1592 was untouched;
- confirmation that `/tmp` build/audit artifacts and generated `Cargo.lock` were removed;
- residual risks, limited to any declined test-hardening draft or unexecuted platform-specific BLAS lane.

No commit is created for Tasks 1-4 because they intentionally make no repository change.
