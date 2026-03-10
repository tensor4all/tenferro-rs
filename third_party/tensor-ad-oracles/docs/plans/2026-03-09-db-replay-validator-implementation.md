# DB Replay Validator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a replay validator that recomputes observables, JVPs, VJPs, FD-JVPs, and expected error cases from the published JSON database and verifies the stored references are reproducible.

**Architecture:** Keep the validator inside `tensor-ad-oracles` and reuse the existing observable/runtime helpers so generation-time and replay-time semantics stay identical. Build the validator in small layers: tensor decoding, case loading, replay execution, then family-by-family tests ending in a full-database integration check.

**Tech Stack:** Python 3.12, `uv`, `torch`, `jsonschema`, `unittest`

---

### Task 1: Add failing tests for JSON tensor decode and case loading

**Files:**
- Create: `validators/__init__.py`
- Create: `validators/tests_placeholder.txt`
- Create: `tests/test_validator_encoding.py`
- Create: `tests/test_case_loader.py`

**Step 1: Write the failing test**

```python
import unittest

from validators import case_loader, encoding


class ValidatorEncodingTests(unittest.TestCase):
    def test_decode_real_tensor_round_trips_shape_and_dtype(self) -> None:
        encoded = {
            "dtype": "float64",
            "shape": [2, 2],
            "order": "row_major",
            "data": [1.0, 2.0, 3.0, 4.0],
        }
        tensor = encoding.decode_tensor(encoded)
        self.assertEqual(tuple(tensor.shape), (2, 2))


class CaseLoaderTests(unittest.TestCase):
    def test_load_case_file_reads_jsonl_records(self) -> None:
        records = case_loader.load_case_file("cases/solve/identity.jsonl")
        self.assertGreaterEqual(len(records), 1)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_validator_encoding tests.test_case_loader -v`
Expected: FAIL with missing `validators` module or missing functions.

**Step 3: Write minimal implementation**

Create a `validators` package with:

- `decode_tensor(encoded: dict) -> torch.Tensor`
- `decode_tensor_map(encoded: dict[str, dict]) -> dict[str, torch.Tensor]`
- `load_case_file(path) -> list[dict]`
- `iter_case_files(root) -> list[Path]`

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_validator_encoding tests.test_case_loader -v`
Expected: PASS

**Step 5: Commit**

```bash
git -C /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles add \
  validators/__init__.py \
  tests/test_validator_encoding.py \
  tests/test_case_loader.py
git -C /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles commit -m "test: add validator tensor decode and case loading"
```

### Task 2: Add a failing replay test for `solve/identity`

**Files:**
- Create: `validators/replay.py`
- Create: `tests/test_db_replay.py`
- Modify: `generators/runtime.py`

**Step 1: Write the failing test**

```python
import unittest

from validators import replay


class DbReplayTests(unittest.TestCase):
    def test_replay_solve_identity_case_matches_stored_references(self) -> None:
        result = replay.replay_case_file("cases/solve/identity.jsonl", limit=1)
        self.assertEqual(result.checked, 1)
        self.assertEqual(result.failures, [])
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_db_replay.DbReplayTests.test_replay_solve_identity_case_matches_stored_references -v`
Expected: FAIL because replay entrypoints do not exist.

**Step 3: Write minimal implementation**

Implement:

- a replay result type or dict with `checked` and `failures`
- success-case replay for `solve/identity`
- observable reconstruction via existing runtime helpers
- recomputation of:
  - observable
  - `pytorch_ref.jvp`
  - `pytorch_ref.vjp`
  - `fd_ref.jvp`

Use stored `fd_ref.step` from JSON instead of recomputing a new step.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_db_replay.DbReplayTests.test_replay_solve_identity_case_matches_stored_references -v`
Expected: PASS

**Step 5: Commit**

```bash
git -C /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles add \
  validators/replay.py \
  tests/test_db_replay.py \
  generators/runtime.py
git -C /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles commit -m "feat: replay solve identity cases from the JSON database"
```

### Task 3: Extend replay coverage to all success families

**Files:**
- Modify: `validators/replay.py`
- Modify: `validators/__init__.py`
- Modify: `tests/test_db_replay.py`

**Step 1: Write the failing test**

```python
def test_replay_one_case_from_each_success_family(self) -> None:
    families = [
        "cases/svd/u_abs.jsonl",
        "cases/svd/s.jsonl",
        "cases/svd/vh_abs.jsonl",
        "cases/svd/uvh_product.jsonl",
        "cases/eigh/values_vectors_abs.jsonl",
        "cases/solve/identity.jsonl",
        "cases/cholesky/identity.jsonl",
        "cases/qr/identity.jsonl",
        "cases/pinv_singular/identity.jsonl",
    ]
    for path in families:
        result = replay.replay_case_file(path, limit=1)
        assert result.failures == [], path
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_db_replay.DbReplayTests.test_replay_one_case_from_each_success_family -v`
Expected: FAIL on the first unsupported op family.

**Step 3: Write minimal implementation**

Add op-family replay dispatch for:

- `svd`
- `eigh`
- `cholesky`
- `qr`
- `pinv_singular`

Reuse `generators.runtime.apply_spec_observable()` and the same op-specific input structure as generation.

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_db_replay.DbReplayTests.test_replay_one_case_from_each_success_family -v`
Expected: PASS

**Step 5: Commit**

```bash
git -C /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles add \
  validators/replay.py \
  tests/test_db_replay.py \
  validators/__init__.py
git -C /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles commit -m "feat: replay all success-family oracle cases"
```

### Task 4: Add replay validation for expected error families

**Files:**
- Modify: `validators/replay.py`
- Modify: `tests/test_db_replay.py`

**Step 1: Write the failing test**

```python
def test_replay_gauge_ill_defined_cases_raise_expected_error(self) -> None:
    for path in [
        "cases/svd/gauge_ill_defined.jsonl",
        "cases/eigh/gauge_ill_defined.jsonl",
    ]:
        result = replay.replay_case_file(path, limit=1)
        assert result.failures == [], path
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_db_replay.DbReplayTests.test_replay_gauge_ill_defined_cases_raise_expected_error -v`
Expected: FAIL because error-case replay is not implemented.

**Step 3: Write minimal implementation**

Implement error-case replay that:

- reconstructs stored input tensors
- executes the gauge-dependent backward path
- checks the runtime error message includes `"ill-defined"`
- reports a structured failure if no error is raised

**Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_db_replay.DbReplayTests.test_replay_gauge_ill_defined_cases_raise_expected_error -v`
Expected: PASS

**Step 5: Commit**

```bash
git -C /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles add \
  validators/replay.py \
  tests/test_db_replay.py
git -C /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles commit -m "feat: replay expected spectral error cases"
```

### Task 5: Add full-database integration coverage and documentation

**Files:**
- Modify: `tests/test_db_replay.py`
- Modify: `README.md`
- Modify: `AGENTS.md`

**Step 1: Write the failing test**

```python
def test_replay_all_published_case_files(self) -> None:
    result = replay.replay_all_cases("cases")
    self.assertEqual(result.failures, [])
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_db_replay.DbReplayTests.test_replay_all_published_case_files -v`
Expected: FAIL because the repository-wide replay entrypoint does not exist.

**Step 3: Write minimal implementation**

Add:

- `replay_all_cases(root="cases")`
- concise failure reporting
- README usage for replay validation
- AGENTS note that replay validation is part of the repository contract

**Step 4: Run test to verify it passes**

Run:

```bash
uv run python -m unittest discover -s tests -v
python3 -m py_compile generators/*.py scripts/*.py tests/*.py validators/*.py
uv run python - <<'PY'
from validators.replay import replay_all_cases
result = replay_all_cases("cases")
assert not result.failures, result.failures
print(result.checked)
PY
```

Expected: all commands succeed; replay reports all published cases checked.

**Step 5: Commit**

```bash
git -C /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles add \
  tests/test_db_replay.py \
  README.md \
  AGENTS.md \
  validators
git -C /sharehome/shinaoka/projects/tensor4all/tensor-ad-oracles commit -m "feat: validate published oracle cases by replay"
```
