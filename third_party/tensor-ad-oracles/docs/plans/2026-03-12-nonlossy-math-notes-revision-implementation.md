# Non-Lossy Math Notes Revision Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Revise `docs/math/*.md` so the published math notes preserve the mathematical and implementation-relevant detail from `../tenferro-rs/docs/AD/*.md` while aligning the detailed formulas with PyTorch's manual AD implementation.

**Architecture:** Treat `../tenferro-rs/docs/AD/*.md` as the non-lossy content baseline, treat PyTorch's `derivatives.yaml` and `FunctionsManual.cpp` as the implementation-alignment reference, and revise `docs/math/*.md` in place without changing registry anchors. Add note-completeness tests that protect representative helper definitions, case splits, and implementation correspondence from being compressed away again.

**Tech Stack:** Markdown docs, Python 3.12, `uv`, `unittest`, local sibling repositories `../tenferro-rs` and `../pytorch`.

---

### Task 1: Add non-lossy note completeness tests

**Files:**
- Modify: `tests/test_math_registry.py`

**Step 1: Write the failing test**

Extend `tests/test_math_registry.py` with representative note-content checks for
the most detail-sensitive notes.

Add assertions such as:

- `docs/math/svd.md` contains:
  - `F_{ij}`
  - `S_inv` or `S_{\\text{inv}`
  - non-square correction terms
  - gauge ambiguity discussion
- `docs/math/qr.md` contains:
  - `copyltu`
  - full-rank case
  - wide reduced QR case
  - triangular solve discussion

Example:

```python
svd_text = (REPO_ROOT / "docs" / "math" / "svd.md").read_text(encoding="utf-8")
self.assertIn("F_{ij}", svd_text)
self.assertIn("gauge", svd_text)

qr_text = (REPO_ROOT / "docs" / "math" / "qr.md").read_text(encoding="utf-8")
self.assertIn("copyltu", qr_text)
self.assertIn("wide", qr_text)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: FAIL because the current condensed notes do not contain enough of the
required detailed content.

**Step 3: Write minimal implementation**

Do not revise the notes yet. Only land the failing expectations in the test file
so the non-lossy bar is explicit.

**Step 4: Re-run to confirm failure remains**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: FAIL for missing detailed note content.

**Step 5: Commit**

```bash
git add tests/test_math_registry.py
git commit -m "test: define non-lossy math note expectations"
```

### Task 2: Restore full detail for `svd.md` and `qr.md`

**Files:**
- Modify: `docs/math/svd.md`
- Modify: `docs/math/qr.md`
- Test: `tests/test_math_registry.py`

**Step 1: Use the reference sources**

Read in full before editing:

- `../tenferro-rs/docs/AD/svd.md`
- `../tenferro-rs/docs/AD/qr.md`
- `../pytorch/tools/autograd/derivatives.yaml`
- `../pytorch/torch/csrc/autograd/FunctionsManual.cpp`

Focus on the SVD and QR manual formula sections.

**Step 2: Write the minimal non-lossy revisions**

Revise `docs/math/svd.md` to restore:

- detailed helper matrix definitions
- inverse singular-value helper
- non-square correction terms
- complex gauge caveat
- implementation correspondence note

Revise `docs/math/qr.md` to restore:

- `copyltu`
- full-rank QR backward formula
- wide reduced-QR case
- triangular solve interpretation
- implementation correspondence note

Preserve existing DB-family anchors.

**Step 3: Run the targeted test**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: PASS for the new `svd` and `qr` completeness assertions.

**Step 4: Review against references**

Manually compare the revised notes against:

- `../tenferro-rs/docs/AD/svd.md`
- `../tenferro-rs/docs/AD/qr.md`

Ensure no meaningful formula/helper/case information was dropped.

**Step 5: Commit**

```bash
git add docs/math/svd.md docs/math/qr.md tests/test_math_registry.py
git commit -m "docs: restore detailed svd and qr notes"
```

### Task 3: Expand the Tier A linalg notes

**Files:**
- Modify: `docs/math/lu.md`
- Modify: `docs/math/eig.md`
- Modify: `docs/math/eigen.md`
- Modify: `docs/math/solve.md`
- Modify: `docs/math/lstsq.md`
- Modify: `docs/math/pinv.md`
- Modify: `docs/math/norm.md`
- Test: `tests/test_math_registry.py`

**Step 1: Extend the test with representative expectations**

Add targeted assertions for representative Tier A notes, for example:

- `solve.md` contains block / adjoint structure or explicit linearized solve form
- `eig.md` and `eigen.md` remain distinct and contain spectral-gap style detail
- `pinv.md` references its factorization-driven or spectral derivation path
- `norm.md` distinguishes matrix and vector norm differentiation caveats

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: FAIL because the current notes remain too compressed.

**Step 3: Revise the notes non-lossily**

For each note, compare against:

- the corresponding `../tenferro-rs/docs/AD/*.md`
- relevant PyTorch manual formula sections where applicable

Restore missing:

- helper operators
- intermediate equations
- case splits
- explicit assumptions / domain restrictions
- short implementation correspondence notes

Preserve registry anchors and DB-family sections.

**Step 4: Run the targeted tests**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: PASS

**Step 5: Commit**

```bash
git add docs/math/lu.md docs/math/eig.md docs/math/eigen.md docs/math/solve.md docs/math/lstsq.md docs/math/pinv.md docs/math/norm.md tests/test_math_registry.py
git commit -m "docs: expand detailed linalg notes"
```

### Task 4: Expand the Tier B notes and reorganize shared notes

**Files:**
- Modify: `docs/math/cholesky.md`
- Modify: `docs/math/inv.md`
- Modify: `docs/math/det.md`
- Modify: `docs/math/matrix_exp.md`
- Modify: `docs/math/dyadtensor_reverse.md`
- Modify: `docs/math/scalar_ops.md`
- Test: `tests/test_math_registry.py`

**Step 1: Add the next round of representative tests**

Add assertions that protect against over-compression in the remaining notes.

Examples:

- `cholesky.md` contains Hermitian / triangular structure caveats
- `matrix_exp.md` contains the derivative strategy and current DB status
- `scalar_ops.md` contains shared derivative pattern descriptions rather than
  only a thin op list

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: FAIL

**Step 3: Revise the notes**

Restore detail from `../tenferro-rs/docs/AD/*.md` and enrich where needed so
the notes remain mathematically complete without trying to force all files to
match Tier A density.

For `scalar_ops.md`, preserve all current anchors while making the shared
patterns explicit:

- unary analytic ops
- binary ops
- reductions
- broadcasting
- nondifferentiable / filtered outputs

**Step 4: Run the targeted tests**

Run: `uv run python -m unittest tests.test_math_registry -v`

Expected: PASS

**Step 5: Commit**

```bash
git add docs/math/cholesky.md docs/math/inv.md docs/math/det.md docs/math/matrix_exp.md docs/math/dyadtensor_reverse.md docs/math/scalar_ops.md tests/test_math_registry.py
git commit -m "docs: restore remaining math note detail"
```

### Task 5: Refresh cross-links and note index wording

**Files:**
- Modify: `docs/math/index.md`
- Modify: `docs/math-registry.md`
- Modify: `README.md`
- Test: `tests/test_repo_config.py`

**Step 1: Write the failing test**

Extend `tests/test_repo_config.py` to require the repository docs to describe
the notes as the detailed canonical AD reference rather than a lightweight note
set.

Example assertions:

```python
readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
self.assertIn("detailed", readme)
self.assertIn("docs/math/", readme)
```

**Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_repo_config -v`

Expected: FAIL because the public wording still underspecifies the new note
depth.

**Step 3: Update the docs wording**

Refresh:

- `docs/math/index.md`
- `docs/math-registry.md`
- `README.md`

so they describe the math corpus as the detailed canonical reference linked to
the DB, not just a short note corpus.

**Step 4: Run the targeted test**

Run: `uv run python -m unittest tests.test_repo_config -v`

Expected: PASS

**Step 5: Commit**

```bash
git add docs/math/index.md docs/math-registry.md README.md tests/test_repo_config.py
git commit -m "docs: describe math notes as canonical detailed reference"
```

### Task 6: Run full verification and perform the non-lossy comparison pass

**Files:**
- Verify: `docs/math/*.md`
- Verify: `tests/test_math_registry.py`
- Verify: `tests/test_repo_config.py`

**Step 1: Run focused test coverage**

Run:

```bash
uv run python -m unittest tests.test_math_registry tests.test_repo_config -v
```

Expected: PASS

**Step 2: Run the full repository suite**

Run:

```bash
uv run python -m unittest discover -s tests -v
```

Expected: PASS

**Step 3: Do the manual non-lossy review**

For each revised note, compare against its `../tenferro-rs/docs/AD/*.md`
counterpart and confirm:

- no meaningful helper definition was lost
- no meaningful case split was lost
- no explicit formula path was lost

For implementation-heavy notes, also verify the note still corresponds to the
relevant PyTorch manual formula entry points.

**Step 4: Commit**

```bash
git add docs/math README.md tests
git commit -m "docs: complete non-lossy math note revision"
```
