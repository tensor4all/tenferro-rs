# Tensor AD Oracles Surface Gap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Classify the published `tensor-ad-oracles` surface against current tenferro public APIs, open implementation issues only for real public-surface gaps, and leave replay expansion focused on already-expressible families.

**Architecture:** Use the vendored oracle case tree as the source of truth, compare each family against the current public APIs in `tenferro-linalg`, `tenferro-tensor`, and `tenferro-einsum`, record the canonical bucket mapping in docs, then create issue-ready backlog entries only for families that require new product surface.

**Tech Stack:** Rust workspace docs, vendored `tensor-ad-oracles` JSONL database, GitHub issues via `gh`, repository design docs.

---

### Task 1: Freeze the current oracle family inventory

**Files:**
- Modify: `docs/plans/2026-03-10-tensor-ad-oracles-surface-gap-design.md`

**Step 1: Extract the current published oracle family inventory**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import json
root = Path("third_party/tensor-ad-oracles/cases")
for opdir in sorted(root.iterdir()):
    if not opdir.is_dir():
        continue
    for file in sorted(opdir.glob("*.jsonl")):
        count = sum(1 for line in file.open() if line.strip())
        print(f"{opdir.name}\t{file.stem}\t{count}")
PY
```

Expected: a stable list of published `(op, family)` pairs with counts.

**Step 2: Confirm the design doc inventory matches the vendored subtree**

Check the family lists in:

- `docs/plans/2026-03-10-tensor-ad-oracles-surface-gap-design.md`

Expected: every unsupported family in the vendored subtree appears in one of the
two buckets.

**Step 3: Commit the inventory confirmation**

```bash
git add docs/plans/2026-03-10-tensor-ad-oracles-surface-gap-design.md
git commit -m "docs: capture oracle surface-gap taxonomy"
```

### Task 2: Resolve the decision bucket

**Files:**
- Modify: `docs/plans/2026-03-10-tensor-ad-oracles-surface-gap-design.md`

**Step 1: Re-audit the ambiguous families against current public APIs**

Inspect:

- `tenferro-linalg/src/lib.rs`
- `tenferro-tensor/src/lib.rs`
- `tenferro-einsum/src/lib.rs`
- `extension/tenferro-dyadtensor/src/api/mod.rs`

Focus on:

- `lu_factor`
- `multi_dot`
- `pinv_hermitian`
- `vecdot`

**Step 2: Write the final decision into the design doc**

For each family, move it to exactly one final bucket:

- `Replay only`
- `Needs public API issue`

Expected: no remaining ambiguous bucket.

**Step 3: Commit the resolved taxonomy**

```bash
git add docs/plans/2026-03-10-tensor-ad-oracles-surface-gap-design.md
git commit -m "docs: resolve oracle surface-gap decisions"
```

### Task 3: Draft the missing-surface issue backlog

**Files:**
- Create: `docs/plans/2026-03-10-tensor-ad-oracles-surface-gap-issues.md`

**Step 1: Write issue-ready entries for each missing public-surface family**

For each family or grouped feature slice, include:

- problem statement
- required public API contract
- likely implementation crate
- oracle family names affected
- acceptance criteria

Keep the entries concise and implementation-oriented.

**Step 2: Verify the draft only includes real public-surface gaps**

Expected: no `Replay only` family appears in the issue draft.

**Step 3: Commit the issue draft**

```bash
git add docs/plans/2026-03-10-tensor-ad-oracles-surface-gap-issues.md
git commit -m "docs: draft oracle surface-gap issue backlog"
```

### Task 4: Open GitHub issues for missing public surface

**Files:**
- Modify: `docs/plans/2026-03-10-tensor-ad-oracles-surface-gap-issues.md`

**Step 1: Open one GitHub issue per approved backlog item**

Run for each issue:

```bash
gh issue create --title "<title>" --body-file <body-file>
```

Expected: issue URLs are returned for all missing-surface items.

**Step 2: Record created issue URLs in the issue draft**

Append the created links under each backlog item in:

- `docs/plans/2026-03-10-tensor-ad-oracles-surface-gap-issues.md`

**Step 3: Commit the issue links**

```bash
git add docs/plans/2026-03-10-tensor-ad-oracles-surface-gap-issues.md
git commit -m "docs: link oracle surface-gap implementation issues"
```

### Task 5: Hand off the replay-only backlog

**Files:**
- Modify: `docs/plans/2026-03-10-tensor-ad-oracles-surface-gap-design.md`
- Modify: `docs/plans/2026-03-10-tensor-ad-oracles-replay.md`

**Step 1: Add a final replay-only family list to the design doc**

Expected: the replay backlog contains only families already expressible by the
current tenferro public APIs.

**Step 2: Update the replay implementation plan to reference the new taxonomy**

Add a short note to:

- `docs/plans/2026-03-10-tensor-ad-oracles-replay.md`

Expected: replay work explicitly excludes families that still need product API
issues.

**Step 3: Commit the handoff**

```bash
git add docs/plans/2026-03-10-tensor-ad-oracles-surface-gap-design.md \
  docs/plans/2026-03-10-tensor-ad-oracles-replay.md
git commit -m "docs: hand off replay-only oracle backlog"
```
