# Dense Scalar OpInfo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `tensor-ad-oracles` so it can materialize dense generic PyTorch scalar/tensor-scalarizable `OpInfo` families, including elementwise, reduction, and `special.*` ops, into the existing JSON oracle database format.

**Architecture:** Add a generic upstream inventory and a generic runtime call-spec layer that complements the current linalg machinery. Keep the current success/error oracle contract, add machine-readable non-differentiable call metadata, and drive full case generation mechanically from upstream `OpInfo` definitions rather than hand-coded family lists.

**Tech Stack:** Python 3.12, `uv`, pinned `torch==2.10.0`, `jsonschema`, `unittest`, GitHub Actions.

---

### Task 1: Add schema support for generic call metadata

**Files:**
- Modify: `schema/case.schema.json`
- Modify: `tests/test_schema_contract.py`

**Step 1: Write the failing test**

Add schema tests that require success cases to accept:

- `op_args`
- `op_kwargs`

with JSON-serializable scalar/integer/bool values and nested list forms needed
for generic OpInfo replay.

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_schema_contract -v`

Expected: FAIL because the schema does not yet allow generic call metadata.

**Step 3: Write minimal implementation**

Update `schema/case.schema.json` to allow:

- optional `op_args`
- optional `op_kwargs`

without changing existing linalg records.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_schema_contract -v`

Expected: PASS

**Step 5: Commit**

```bash
git add schema/case.schema.json tests/test_schema_contract.py
git commit -m "feat: add generic call metadata to case schema"
```

### Task 2: Add generic upstream inventory normalization

**Files:**
- Create: `generators/upstream_scalar_inventory.py`
- Create: `tests/test_upstream_scalar_inventory.py`
- Modify: `tests/test_family_mapping.py`

**Step 1: Write the failing test**

Add tests that require the new inventory to collect dense AD-relevant generic
PyTorch `OpInfo` entries and classify them as:

- unary
- binary
- reduction
- generic dense op

for `float32`, `float64`, `complex64`, and `complex128`.

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_upstream_scalar_inventory tests.test_family_mapping -v`

Expected: FAIL because no scalar/generic inventory exists.

**Step 3: Write minimal implementation**

Implement a normalized inventory layer that records:

- upstream name and variant
- sample-input function name
- output-process function names
- gradcheck wrapper
- first-order and second-order support flags
- explicit tolerance overrides
- family class

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_upstream_scalar_inventory tests.test_family_mapping -v`

Expected: PASS

**Step 5: Commit**

```bash
git add generators/upstream_scalar_inventory.py tests/test_upstream_scalar_inventory.py tests/test_family_mapping.py
git commit -m "feat: add dense scalar upstream inventory"
```

### Task 3: Add generic runtime call reconstruction

**Files:**
- Modify: `generators/runtime.py`
- Create: `tests/test_runtime_generic_scalar.py`

**Step 1: Write the failing test**

Require the runtime to rebuild a generic operation call from:

- differentiable tensor `inputs`
- non-differentiable `op_args`
- non-differentiable `op_kwargs`

and to execute the pinned upstream operator through one common path.

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_runtime_generic_scalar -v`

Expected: FAIL because runtime reconstruction is still linalg-oriented.

**Step 3: Write minimal implementation**

Add generic helpers in `generators/runtime.py` to:

- materialize scalar kwargs and args
- call upstream ops through a generic invocation path
- keep current linalg behavior unchanged

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_runtime_generic_scalar -v`

Expected: PASS

**Step 5: Commit**

```bash
git add generators/runtime.py tests/test_runtime_generic_scalar.py
git commit -m "feat: add generic scalar runtime execution"
```

### Task 4: Add generic family mapping and registry support

**Files:**
- Modify: `generators/pytorch_v1.py`
- Modify: `tests/test_pytorch_v1.py`

**Step 1: Write the failing test**

Add tests that require the registry to include representative generic scalar
families from unary, binary, and reduction classes while preserving current
linalg families.

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_pytorch_v1 -v`

Expected: FAIL because the registry only materializes linalg families.

**Step 3: Write minimal implementation**

Extend the registry and spec model so generic scalar families can record:

- upstream metadata
- observable kind
- `op_args`
- `op_kwargs`
- dtype support and HVP eligibility policy

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_pytorch_v1 -v`

Expected: PASS

**Step 5: Commit**

```bash
git add generators/pytorch_v1.py tests/test_pytorch_v1.py
git commit -m "feat: add scalar family registry mapping"
```

### Task 5: Materialize representative scalar families first

**Files:**
- Modify: `generators/pytorch_v1.py`
- Create: `tests/test_scalar_generation.py`

**Step 1: Write the failing test**

Add generation tests for one representative family from each class:

- one unary family
- one binary family
- one reduction family

Require generated records to include generic call metadata and valid first-order
oracle payloads.

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_scalar_generation -v`

Expected: FAIL because generic scalar generation is not implemented.

**Step 3: Write minimal implementation**

Implement enough generation machinery to materialize those representative
families end to end, reusing the existing probe/oracle flow.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_scalar_generation -v`

Expected: PASS

**Step 5: Commit**

```bash
git add generators/pytorch_v1.py tests/test_scalar_generation.py
git commit -m "feat: materialize representative scalar families"
```

### Task 6: Extend replay validation for generic scalar families

**Files:**
- Modify: `validators/replay.py`
- Modify: `tests/test_db_replay.py`

**Step 1: Write the failing test**

Require replay validation to reproduce at least the representative unary,
binary, and reduction families using `op_args` / `op_kwargs`.

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_db_replay -v`

Expected: FAIL because replay cannot reconstruct generic scalar calls.

**Step 3: Write minimal implementation**

Extend replay to:

- reconstruct generic upstream calls
- preserve current linalg replay behavior
- reuse the same first-order and second-order validation logic

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_db_replay -v`

Expected: PASS

**Step 5: Commit**

```bash
git add validators/replay.py tests/test_db_replay.py
git commit -m "feat: replay generic scalar oracle families"
```

### Task 7: Import the full dense scalar family set

**Files:**
- Modify: `generators/pytorch_v1.py`
- Modify: `generators/tolerance_audit.py`
- Modify: `scripts/check_tolerances.py`
- Modify: `scripts/check_upstream_ad_tolerances.py`
- Modify: `README.md`

**Step 1: Write the failing test**

Add or extend tests so the full dense scalar inventory must be either:

- mapped into DB families, or
- explicitly excluded with a reason

and so tolerance audit covers new scalar families and dtypes.

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_family_mapping tests.test_tolerance_audit tests.test_upstream_ad_tolerance_script -v`

Expected: FAIL because the generic inventory is not yet fully mapped.

**Step 3: Write minimal implementation**

Complete the full mapping/import, add dtype-aware first-order and second-order
tolerance handling, and document the scalar expansion in `README.md`.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_family_mapping tests.test_tolerance_audit tests.test_upstream_ad_tolerance_script -v`

Expected: PASS

**Step 5: Commit**

```bash
git add generators/pytorch_v1.py generators/tolerance_audit.py scripts/check_tolerances.py scripts/check_upstream_ad_tolerances.py README.md
git commit -m "feat: import dense scalar OpInfo families"
```

### Task 8: Regenerate the database and run full verification

**Files:**
- Modify: `cases/**/*.jsonl`
- Verify: `schema/case.schema.json`
- Verify: `generators/*.py`
- Verify: `validators/*.py`
- Verify: `scripts/*.py`

**Step 1: Regenerate the full database**

Run: `uv run python -m generators.pytorch_v1 --materialize-all`

Expected: all supported linalg and dense scalar families regenerate successfully.

**Step 2: Run verification commands**

Run:

```bash
uv run python -m unittest discover -s tests -v
uv run python scripts/validate_schema.py
uv run python scripts/verify_cases.py
uv run python scripts/check_replay.py
uv run python scripts/check_regeneration.py
uv run python scripts/check_tolerances.py
uv run python scripts/check_upstream_ad_tolerances.py
python3 -m py_compile generators/*.py scripts/*.py tests/*.py validators/*.py
```

Expected: PASS

**Step 3: Commit regenerated artifacts**

```bash
git add cases schema/case.schema.json generators scripts tests validators README.md
git commit -m "feat: publish dense scalar oracle database"
```
