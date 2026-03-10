# Materialization and UV Pin Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the repository's `uv` environment reliably usable for PyTorch-backed generation, then implement the first real case materialization helpers.

**Architecture:** Pin the repo to a specific CPython patch version so `uv` chooses a managed interpreter instead of a broken external 3.12 build. Once the environment can import `torch`, add small, test-driven materialization helpers that encode tensors, create stable probes, and write concrete case records.

**Tech Stack:** Python 3.12, `uv`, PyTorch, NumPy, JSON Schema, `unittest`

---

### Task 1: Pin the UV Interpreter

**Files:**
- Modify: `.python-version`
- Modify: `README.md`
- Test: `tests/test_repo_config.py`

**Step 1: Write the failing test**

Add a test that requires `.python-version` to be patch-pinned (for example `3.12.12`, not `3.12`).

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_repo_config.RepoConfigTests.test_python_version_file_is_patch_pinned -v`
Expected: FAIL because the file only contains `3.12`

**Step 3: Write minimal implementation**

Set `.python-version` to the pinned patch version and document why the repo checks in an exact interpreter version.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_repo_config.RepoConfigTests.test_python_version_file_is_patch_pinned -v`
Expected: PASS

### Task 2: Add Tensor Encoding Helpers

**Files:**
- Create: `generators/encoding.py`
- Modify: `generators/pytorch_v1.py`
- Test: `tests/test_encoding.py`

**Step 1: Write the failing test**

Add tests for:
- row-major tensor encoding from real tensors
- complex tensor encoding as `[re, im]`
- scalar encoding with `shape=[]`

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_encoding -v`
Expected: FAIL with import or missing symbol errors

**Step 3: Write minimal implementation**

Implement a small encoder that converts PyTorch tensors into the repository JSON tensor format.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_encoding -v`
Expected: PASS

### Task 3: Materialize One Success Family

**Files:**
- Modify: `generators/pytorch_v1.py`
- Modify: `generators/probes.py`
- Test: `tests/test_materialize.py`

**Step 1: Write the failing test**

Add a test that builds one concrete `solve/identity` success case from encoded inputs, probe payloads, and provenance.

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_materialize.MaterializeTests.test_materialize_solve_success_case -v`
Expected: FAIL because the materialization helper does not exist

**Step 3: Write minimal implementation**

Implement a helper that assembles one fully encoded success case record using the v1 schema contract.

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_materialize.MaterializeTests.test_materialize_solve_success_case -v`
Expected: PASS

### Task 4: Verify End-to-End

**Files:**
- Modify: none
- Test: `tests/`

**Step 1: Run the focused suite**

Run: `python3 -m unittest discover -s tests -v`
Expected: PASS

**Step 2: Run the UV-managed suite**

Run: `uv sync --locked --all-groups`
Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS

**Step 3: Smoke-test the generator CLI**

Run: `uv run python -m generators.pytorch_v1 --list`
Expected: PASS and print the v1 op/family registry
