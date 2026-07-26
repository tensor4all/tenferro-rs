# Execution Engine Rust Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct the accepted Phase 1/2 execution-engine Rust implementation on a clean `origin/main` branch without the Phase 2E harness or generated evidence bulk.

**Architecture:** Preserve the stopped worktree as forensic state and create a new isolated integration worktree. Replay only the curated Phase 1/2 design, Rust, tests, and repository-contract commits through the last audited Rust marker, then prove that the reconstructed Rust tree matches that marker while excluded Phase 2E paths are absent.

**Tech Stack:** Git worktrees, Rust/Cargo, repository Python contract tests, CodeGraph

---

## File Structure

The reconstruction intentionally creates no new production source files.

- Replayed Phase 1/2 implementation: `crates/tenferro-cpu/**`,
  `crates/tenferro-tensor/**`, `crates/tenferro-ad/**`, and the narrow
  integration updates in runtime/linalg/fft/GPU tests.
- Replayed curated architecture: `docs/design/execution-engine-provider-architecture.md`,
  `docs/design/cpu-backend-execution.md`, and Phase 1/2 specifications,
  plans, guides, and work logs.
- Replayed repository contracts: `REPOSITORY_RULES.md`,
  `scripts/test-repository-rules-review.py`, and the narrow source-contract
  tests introduced by the Phase 1/2 implementation.
- Restart control documents:
  `docs/superpowers/specs/2026-07-24-execution-engine-phase9-restart-design.md`
  and this plan.
- Explicitly excluded: `scripts/phase2e*`, `scripts/run_phase2e*`,
  `docs/worklogs/artifacts/**`, and the six stopped uncommitted harness files.

### Task 1: Create the clean integration worktree

**Files:**
- Preserve: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/execution-engine-provider-design/**`
- Create worktree: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/execution-engine-phase9-restart`

- [ ] **Step 1: Confirm the stopped worktree still contains only the six expected uncommitted files**

Run:

```bash
git -C /home/shinaoka/tensor4all/tenferro-rs/.worktrees/execution-engine-provider-design status --short
```

Expected: the six modified Phase 2E harness files and no unreviewed production
Rust changes.

- [ ] **Step 2: Refresh remote state**

Run:

```bash
git -C /home/shinaoka/tensor4all/tenferro-rs fetch origin
git -C /home/shinaoka/tensor4all/tenferro-rs rev-parse origin/main
```

Expected: fetch succeeds and prints one exact `origin/main` commit.

- [ ] **Step 3: Create the integration branch from the refreshed main**

Run:

```bash
git -C /home/shinaoka/tensor4all/tenferro-rs worktree add \
  /home/shinaoka/tensor4all/tenferro-rs/.worktrees/execution-engine-phase9-restart \
  -b codex/execution-engine-phase9-restart \
  origin/main
```

Expected: a new clean worktree on
`codex/execution-engine-phase9-restart`.

- [ ] **Step 4: Re-read repository instructions in the new worktree**

Run:

```bash
sed -n '1,260p' /home/shinaoka/tensor4all/tenferro-rs/.worktrees/execution-engine-phase9-restart/AGENTS.md
sed -n '1,900p' /home/shinaoka/tensor4all/tenferro-rs/.worktrees/execution-engine-phase9-restart/REPOSITORY_RULES.md
```

Expected: the complete local rules are available before replaying commits.

### Task 2: Replay the curated Phase 1 implementation

**Files:**
- Modify: `crates/tenferro-cpu/**`
- Modify: `crates/tenferro-tensor/src/backend.rs`
- Modify: `crates/tenferro-runtime/src/graph/executor/tests/preflight.rs`
- Modify: `crates/tenferro-ad/benches/eager_dispatch_baseline.rs`
- Modify: curated Phase 1 design, plan, guide, and work-log files

- [ ] **Step 1: Replay the Phase 1 design and baseline**

Run in the integration worktree:

```bash
git cherry-pick \
  d9859191 \
  045472bd \
  474ed072 \
  e5a16a65
```

Expected: four commits apply without adding Phase 2E files.

- [ ] **Step 2: Replay the Phase 1 provider implementation**

Run:

```bash
git cherry-pick \
  00695184 \
  f3619e44 \
  a8c6d9a8 \
  9c42317b \
  3a9b2572 \
  8058828e \
  8118e105 \
  66e2a618 \
  9ffe1daa
```

Expected: nine coherent implementation/test commits apply.

- [ ] **Step 3: Run focused Phase 1 tests**

Run:

```bash
cargo test -p tenferro-cpu --lib --release
cargo test -p tenferro-cpu --test integration --release
```

Expected: both commands pass with zero failed tests.

- [ ] **Step 4: Confirm Phase 2E bulk is absent**

Run:

```bash
test ! -e scripts/run_phase2e.py
test -z "$(git ls-files 'docs/worklogs/artifacts/**')"
```

Expected: both checks exit successfully.

### Task 3: Replay the curated Phase 2 implementation

**Files:**
- Modify: `crates/tenferro-cpu/**`
- Modify: `crates/tenferro-tensor/src/types.rs`
- Modify: `crates/tenferro-ad/**`
- Modify: narrow runtime/linalg/fft/GPU integration tests
- Modify: curated Phase 2 specs, plans, design docs, guides, and work logs

- [ ] **Step 1: Replay the Phase 2 design and domain foundation**

Run:

```bash
git cherry-pick \
  49b84905 \
  2b34be26 \
  963e426c \
  c137e331 \
  64897b41 \
  e6dd298a \
  4689472a \
  52d34d19 \
  8d472857 \
  f76fa31c \
  f5e0f8fd
```

Expected: eleven commits apply and the domain-executor tests compile.

- [ ] **Step 2: Replay the audit contract and operation-entry implementation**

Run:

```bash
git cherry-pick \
  f04969ab \
  c4793020 \
  a54b7c29 \
  28c1fd77 \
  97410519 \
  48b95864 \
  8e5db120 \
  b48c4b06
```

Expected: eight commits apply through the audited Rust marker
`b48c4b06`.

- [ ] **Step 3: Run focused Phase 2 tests**

Run:

```bash
cargo test -p tenferro-cpu --lib --release
cargo test -p tenferro-ad --lib --release
cargo test -p tenferro-ad --test integration placement_bound --release
```

Expected: all focused CPU and placement-bound eager tests pass.

### Task 4: Prove reconstruction fidelity and add restart controls

**Files:**
- Create: `docs/superpowers/specs/2026-07-24-execution-engine-phase9-restart-design.md`
- Create: `docs/superpowers/plans/2026-07-24-execution-engine-rust-reconstruction.md`

- [ ] **Step 1: Prove the selected patches were replayed faithfully**

The audited marker predates the current `origin/main` full-SVD/lstsq merge.
Therefore an exact tree comparison with `b48c4b06` would incorrectly reject
the required current-main changes. Compare the replayed commits by stable
patch ID instead, then inspect the two commits whose surrounding SVD context
changed.

Run the stable patch-ID comparison for the 32 curated original/replayed commit
pairs, followed by:

```bash
git range-diff a54b7c29^! <replayed-operation-entry-commit>^!
git range-diff b48c4b06^! <replayed-output-affinity-commit>^!
```

Expected: 30 stable patch IDs match exactly. The operation-entry difference
retains current-main full/thin SVD selection while replacing `CpuContext`
entry with `CpuExecutionContext`; the output-affinity difference is only the
same patch under the current-main `svd_full` method context.

- [ ] **Step 2: Copy only the approved restart documents**

Run:

```bash
git restore \
  --source=codex/execution-engine-through-phase9 \
  -- \
  docs/superpowers/specs/2026-07-24-execution-engine-phase9-restart-design.md \
  docs/superpowers/plans/2026-07-24-execution-engine-rust-reconstruction.md
git add \
  docs/superpowers/specs/2026-07-24-execution-engine-phase9-restart-design.md \
  docs/superpowers/plans/2026-07-24-execution-engine-rust-reconstruction.md
git diff --cached --check
git commit -m "docs: define execution engine restart"
```

Expected: exactly the two restart documents are committed.

- [ ] **Step 3: Enforce the excluded-path contract**

Run:

```bash
test -z "$(git diff --name-only origin/main...HEAD | rg \
  '^(scripts/(phase2e|run_phase2e)|docs/worklogs/artifacts/)')"
```

Expected: no excluded path is present in the integration delta.

- [ ] **Step 4: Record the reconstruction range**

Run:

```bash
git log --reverse --oneline origin/main..HEAD
git diff --shortstat origin/main...HEAD
git status --short
```

Expected: only curated Phase 1/2 commits plus one restart-document commit,
and a clean worktree.

### Task 5: Verify the reconstructed integration unit

**Files:**
- Verify: all reconstructed Phase 1/2 files

- [ ] **Step 1: Check formatting**

Run:

```bash
cargo fmt --all --check
```

Expected: exit status 0.

- [ ] **Step 2: Run the complete release test suite**

Run:

```bash
cargo test --workspace --release
```

Expected: all workspace tests and doctests pass.

- [ ] **Step 3: Build documentation**

Run:

```bash
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: rustdoc builds and the docs-site check passes.

- [ ] **Step 4: Run repository contract review**

Run:

```bash
python3 scripts/repository-rules-review.py \
  --base origin/main \
  --head HEAD \
  --output-json /tmp/execution-engine-reconstruction-review.json
```

Expected: the committed-head review reports no Blocking or Important
findings.

- [ ] **Step 5: Handle any verification failure without widening scope**

If and only if a verification command exposes a reconstruction error, first
record the failing command and diagnostic in the plan checklist. Write a
focused TDD amendment naming the exact test and production files before
editing code. Make the smallest correction, rerun the affected command, and
commit only the named files with commit subject
`fix: repair execution engine reconstruction`.

Expected: no opportunistic refactor or Phase 2E harness change enters this
commit.

### Task 6: Prepare the multi-operation regression plan

**Files:**
- Read: `crates/tenferro-cpu/src/backend.rs`
- Read: `crates/tenferro-cpu/src/exec_session.rs`
- Read: `crates/tenferro-cpu/src/provider.rs`
- Read: `crates/tenferro-cpu/src/context.rs`
- Create: `docs/superpowers/plans/2026-07-24-multi-operation-session-entry.md`

- [ ] **Step 1: Refresh CodeGraph and trace all three execution surfaces**

Run:

```bash
codegraph init
codegraph explore \
  "BackendSessionHost CpuExecSession CpuOperationEntry::enter GraphExecutor multi-operation eager explicit-session compiled-program"
```

Expected: exact eager, explicit-session, and compiled-program call paths are
available from the reconstructed tree.

- [ ] **Step 2: Write a separate TDD plan**

The plan must define exact failing tests for:

- two native operations inside one explicit backend session;
- two operations inside one compiled graph program;
- one standalone eager operation;
- sequential and inner-parallel domain modes;
- external executor behavior without assuming Rayon internals.

The implementation design must preserve provider-selected outer scheduling and
must not replace operation-level mode selection with one unconditional session
mode.

- [ ] **Step 3: Apply the practical performance contract**

The follow-up plan must use the restart design's microbenchmark rule:
a predeclared primary case is blocking only for a slowdown of at least 50
percent that reproduces in a second complete paired A/B run. `INCONCLUSIVE`
does not close the candidate.
