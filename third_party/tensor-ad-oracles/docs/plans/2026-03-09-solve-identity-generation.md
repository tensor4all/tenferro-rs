# Solve Identity Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Materialize the first real JSONL dataset for `cases/solve/identity.jsonl` from PyTorch sample families.

**Architecture:** Use PyTorch internal `sample_inputs_linalg_solve` on CPU with `float64`, derive one deterministic probe per case, compute JVP/VJP/FD-JVP for the `identity` observable, and write validated JSONL records via the existing schema helpers.

**Tech Stack:** Python 3.12, `uv`, PyTorch, NumPy, JSON Schema, `unittest`

---

### Task 1: Add a failing integration test for solve generation

**Files:**
- Create: `tests/test_solve_generation.py`
- Modify: `generators/pytorch_v1.py`

**Step 1: Write the failing test**

Add a `uv`-only integration test that calls a new helper to materialize one `solve/identity` case and asserts:
- one record is returned
- `op == "solve"`
- `family == "identity"`
- `probes` has length 1
- the record can be written to `cases/solve/identity.jsonl`

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_solve_generation -v`
Expected: FAIL because the helper does not exist

### Task 2: Implement minimal solve generation

**Files:**
- Modify: `generators/pytorch_v1.py`
- Modify: `generators/fd.py`
- Modify: `generators/observables.py`

**Step 1: Write minimal implementation**

Implement:
- a deterministic solve sample loader using PyTorch internal `sample_inputs_linalg_solve`
- one paired probe generator for raw tensor inputs
- JVP via `torch.func.jvp`
- VJP via `torch.autograd.grad`
- FD-JVP via second-order central difference
- a helper that returns one materialized `solve/identity` record

**Step 2: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_solve_generation -v`
Expected: PASS

### Task 3: Add CLI write path

**Files:**
- Modify: `generators/pytorch_v1.py`
- Modify: `README.md`
- Test: `tests/test_pytorch_v1.py`

**Step 1: Write the failing test**

Add a test for a CLI/helper path that writes `cases/solve/identity.jsonl` with `limit=1`.

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_pytorch_v1 -v`
Expected: FAIL because the write path is missing

**Step 3: Write minimal implementation**

Add a small CLI flag or helper to materialize the first solve family into the canonical JSONL file.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_pytorch_v1 -v`
Expected: PASS

### Task 4: Verify end-to-end

**Files:**
- Modify: none

**Step 1: Run verification**

Run:
- `python3 -m unittest discover -s tests -v`
- `uv sync --locked --all-groups`
- `uv run python -m unittest discover -s tests -v`
- `uv run python -m generators.pytorch_v1 --materialize solve --family identity --limit 1`

Expected:
- test suites pass
- `cases/solve/identity.jsonl` is created with one record
