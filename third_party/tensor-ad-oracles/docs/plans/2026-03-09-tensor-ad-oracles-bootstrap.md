# Tensor AD Oracles Bootstrap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bootstrap `tensor-ad-oracles` with the schema-first baseline for PyTorch-aligned derivative-correctness cases.

**Architecture:** Start with a machine-readable contract before any case materialization. The repository begins with a top-level README, a formal JSON schema for `success` and `error` cases, and a minimal PyTorch generator entrypoint that will later expand the agreed case families into JSONL files.

**Tech Stack:** Python 3.11+, JSON Schema 2020-12, PyTorch, NumPy

---

### Task 1: Bootstrap Repository Metadata

**Files:**
- Create: `/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/README.md`
- Create: `/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/pyproject.toml`

**Step 1: Write the files**

- Add a README that defines the repository purpose, oracle policy, verification contract, and file layout.
- Add a minimal `pyproject.toml` with project metadata and dependencies required for schema validation and future generation.

**Step 2: Verify the files are syntactically valid**

Run:

```bash
python - <<'PY'
from pathlib import Path
import tomllib

tomllib.loads(Path("/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/pyproject.toml").read_text())
PY
```

Expected: no output and exit code 0.

### Task 2: Add the Case Schema

**Files:**
- Create: `/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/schema/case.schema.json`

**Step 1: Write the schema**

- Define a `oneOf` schema covering:
  - `success` cases with paired probes and both `pytorch_ref` and `fd_ref`
  - `error` cases with `expect_error`
- Add shared tensor encoding with `dtype`, `shape`, `order`, and `data`.
- Freeze the v1 enums for target ops, observables, and comparison kinds.

**Step 2: Verify the schema parses**

Run: `python -m json.tool /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/schema/case.schema.json >/dev/null`

Expected: exit code 0.

### Task 3: Add the PyTorch Generator Entrypoint

**Files:**
- Create: `/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/generators/__init__.py`
- Create: `/sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/generators/pytorch_v1.py`

**Step 1: Write the minimal entrypoint**

- Define the fixed v1 target ops and the schema path.
- Expose a small CLI entrypoint that makes it clear generation is not implemented yet.

**Step 2: Verify the module imports**

Run: `python -m py_compile /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/generators/__init__.py /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles/generators/pytorch_v1.py`

Expected: exit code 0.
