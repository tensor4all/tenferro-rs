# Structured Input Semantics README Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Document the Hermitian gradcheck-wrapper semantics for published `eigh` and `cholesky` families so downstream consumers can correctly interpret the v1 database contract.

**Architecture:** Update only the public README contract. Add one short section that defines structured-input semantics for wrapper-backed families and cross-link that meaning with the replay-validation description. Leave schema, records, and generator behavior unchanged.

**Tech Stack:** Markdown, `uv`, Python `unittest`.

---

### Task 1: Add a focused README contract section

**Files:**
- Modify: `README.md`
- Test: `tests/test_repo_config.py`

**Step 1: Write the failing test**

Add a repository-config test that requires the README to mention:

- structured-input or gradcheck-wrapper semantics
- `eigh`
- `cholesky`

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_repo_config.RepoConfigTests.test_readme_documents_structured_input_semantics -v`

Expected: FAIL because the README does not describe the Hermitian wrapper contract.

**Step 3: Write minimal implementation**

Update `README.md` with a short `Structured Input Semantics` section that:

- states some families are published under upstream gradcheck-wrapper semantics
- names `eigh` and `cholesky`
- clarifies that serialized payloads are interpreted through Hermitian
  structure-preserving wrapper semantics during oracle evaluation
- states that v1 does not yet expose a machine-readable schema field for this
  contract

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_repo_config.RepoConfigTests.test_readme_documents_structured_input_semantics -v`

Expected: PASS

**Step 5: Commit**

```bash
git add README.md tests/test_repo_config.py
git commit -m "docs: clarify structured input semantics"
```

### Task 2: Run repository verification for the doc-only change

**Files:**
- Verify: `README.md`
- Verify: `tests/test_repo_config.py`

**Step 1: Run targeted tests**

Run: `uv run python -m unittest tests.test_repo_config -v`

Expected: PASS

**Step 2: Run the full unit test suite**

Run: `uv run python -m unittest discover -s tests -v`

Expected: PASS

**Step 3: Run docs and syntax checks**

Run: `python3 -m py_compile generators/*.py scripts/*.py tests/*.py validators/*.py`

Expected: PASS

**Step 4: Commit verification-clean state**

```bash
git add README.md tests/test_repo_config.py
git commit --amend --no-edit
```
