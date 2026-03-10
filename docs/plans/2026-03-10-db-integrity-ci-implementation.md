# Database Integrity CI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add lightweight repository guards that make published oracle cases reproducible, cross-oracle consistent, and review-protected.

**Architecture:** Strengthen the replay validator to enforce live oracle agreement, add a regeneration checker that rebuilds the full case tree into a temporary directory and diffs it against `cases/`, and wire both checks into GitHub Actions. Pin the PyTorch dependency exactly so generation and replay remain stable across CI runs.

**Tech Stack:** Python 3.12, `uv`, `torch`, `jsonschema`, `unittest`, GitHub Actions

---

### Task 1: Lock the PyTorch dependency contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Test: `tests/test_repo_config.py`

**Step 1:** Add a failing repo-config test that requires an exact `torch` pin.

**Step 2:** Pin `torch` exactly in `pyproject.toml` and document the rule.

**Step 3:** Refresh `uv.lock`.

### Task 2: Tighten replay validation

**Files:**
- Modify: `validators/replay.py`
- Modify: `tests/test_db_replay.py`

**Step 1:** Add a failing test that requires the replay path to enforce live
cross-oracle JVP agreement and adjoint consistency.

**Step 2:** Implement the additional replay checks.

### Task 3: Add regeneration diff validation

**Files:**
- Create: `scripts/check_regeneration.py`
- Modify: `generators/pytorch_v1.py`
- Modify: `tests/test_scripts.py`

**Step 1:** Add a failing test for a regeneration checker helper that detects
content mismatch.

**Step 2:** Add a generator entrypoint that can materialize the full published
tree deterministically.

**Step 3:** Implement the regeneration diff script and its CLI.

### Task 4: Add repository policy files

**Files:**
- Create: `.github/CODEOWNERS`
- Create: `.github/workflows/oracle-integrity.yml`
- Create: `.github/workflows/oracle-regeneration.yml`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Test: `tests/test_repo_config.py`

**Step 1:** Add failing tests for workflow and CODEOWNERS presence.

**Step 2:** Add the workflows and CODEOWNERS entries.

**Step 3:** Document the CI contract and branch-protection expectation.

### Task 5: Verify end to end

**Files:**
- Modify as needed based on failures

**Step 1:** Run targeted unit tests.

**Step 2:** Run the full test suite through `uv`.

**Step 3:** Run the regeneration checker and replay validator directly.
