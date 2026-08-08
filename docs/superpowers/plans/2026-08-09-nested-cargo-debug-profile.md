# Nested Cargo Debug Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make independent guide and trybuild Cargo fixtures inherit tenferro's debug-free dev default without overriding explicit debugger requests.

**Architecture:** Repository-local Cargo environment configuration propagates the dev-profile default to Cargo-launched test processes and nested trybuild Cargo. The directly launched Python guide checker supplies the same set-if-absent default at its subprocess boundary.

**Tech Stack:** Cargo profiles/config, Python 3.11+, `unittest`, Rust trybuild, ELF inspection tools.

## Global Constraints

- Preserve `CARGO_PROFILE_DEV_DEBUG=1` or `2` supplied by the caller.
- Do not change optimization, assertions, overflow checks, incremental behavior, feature selection, diagnostics, or public APIs.
- Do not add dependencies, shared worktree target directories, sccache, or automatic garbage collection.
- Use fresh `origin/main`-based isolation and never touch the user's existing checkout.

---

### Task 1: Capture cold baseline evidence

**Files:**
- Produce outside repository: `/tmp/tenferro-build-artifact-measurements/baseline-*`

**Interfaces:**
- Consumes: unmodified `origin/main` source.
- Produces: allocated-byte, timing, largest-file, and ELF-section evidence used in the issue and PR.

- [ ] **Step 1: Record environment**

```bash
mkdir -p /tmp/tenferro-build-artifact-measurements
rustc -Vv > /tmp/tenferro-build-artifact-measurements/toolchain.txt
cargo -V >> /tmp/tenferro-build-artifact-measurements/toolchain.txt
uname -a > /tmp/tenferro-build-artifact-measurements/system.txt
```

- [ ] **Step 2: Measure the guide fixture from a fresh target**

```bash
rm -rf target/guide-snippet-check
/usr/bin/time -v -o /tmp/tenferro-build-artifact-measurements/baseline-guide-time.txt \
  env CARGO_BUILD_JOBS=4 RUSTC_WRAPPER= \
  python3 scripts/check-guide-dependency-snippets.py
du -s --block-size=1 target/guide-snippet-check \
  > /tmp/tenferro-build-artifact-measurements/baseline-guide-du.txt
```

- [ ] **Step 3: Measure the focused trybuild fixture from a fresh target**

```bash
rm -rf /tmp/tenferro-trybuild-baseline
/usr/bin/time -v -o /tmp/tenferro-build-artifact-measurements/baseline-trybuild-time.txt \
  env CARGO_TARGET_DIR=/tmp/tenferro-trybuild-baseline CARGO_BUILD_JOBS=4 RUSTC_WRAPPER= \
  cargo test -p tenferro-tensor --test storage_compile_contract \
  storage_ui_compile_contracts -- --exact
du -s --block-size=1 /tmp/tenferro-trybuild-baseline \
  > /tmp/tenferro-build-artifact-measurements/baseline-trybuild-du.txt
```

- [ ] **Step 4: Record largest files and confirm baseline DWARF**

```bash
find target/guide-snippet-check /tmp/tenferro-trybuild-baseline -type f \
  -printf '%s %p\n' | sort -nr | head -30 \
  > /tmp/tenferro-build-artifact-measurements/baseline-largest.txt
```

Choose a representative ELF or rlib member from the inventory and record its `.debug_*` sections with `readelf -SW`.

### Task 2: Add failing configuration contracts

**Files:**
- Modify: `scripts/ci/tests/test_build_artifact_contracts.py`
- Create: `.cargo/config.toml`
- Modify: `scripts/check-guide-dependency-snippets.py`

**Interfaces:**
- Produces: `cargo_environment(target_dir: pathlib.Path) -> dict[str, str]`.

- [ ] **Step 1: Add failing repository-config contract**

Add a test that parses `.cargo/config.toml` and expects:

```python
config = tomllib.loads((ROOT / ".cargo" / "config.toml").read_text())
self.assertEqual(
    config["env"]["CARGO_PROFILE_DEV_DEBUG"],
    {"value": "0", "force": False},
)
```

- [ ] **Step 2: Add failing guide-environment contracts**

Load `scripts/check-guide-dependency-snippets.py` with `runpy.run_path`. Assert that `cargo_environment(Path("/tmp/target"))` supplies `CARGO_PROFILE_DEV_DEBUG="0"` when absent and preserves `"2"` under `patch.dict(os.environ, ..., clear=True)`.

- [ ] **Step 3: Run the tests and verify failure**

```bash
python3 -m unittest scripts.ci.tests.test_build_artifact_contracts -v
```

Expected: failure because `.cargo/config.toml` and `cargo_environment` do not exist.

### Task 3: Implement profile propagation

**Files:**
- Create: `.cargo/config.toml`
- Modify: `scripts/check-guide-dependency-snippets.py:305-340`
- Test: `scripts/ci/tests/test_build_artifact_contracts.py`

**Interfaces:**
- `cargo_environment(target_dir)` returns a copied environment with an explicit target and set-if-absent dev debug level.

- [ ] **Step 1: Add repository-local nested-Cargo default**

```toml
# Keep independently generated Cargo fixtures aligned with the root dev profile.
# A caller-provided debugger level remains authoritative.
[env]
CARGO_PROFILE_DEV_DEBUG = { value = "0", force = false }
```

- [ ] **Step 2: Add the guide subprocess helper**

```python
def cargo_environment(target_dir: pathlib.Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CARGO_PROFILE_DEV_DEBUG", "0")
    env["CARGO_TARGET_DIR"] = str(target_dir)
    return env
```

Replace the inline environment construction in `run_case` with `env=cargo_environment(target_dir)`.

- [ ] **Step 3: Run focused contracts**

```bash
python3 -m unittest scripts.ci.tests.test_build_artifact_contracts -v
```

Expected: all tests pass.

- [ ] **Step 4: Verify actual nested fixtures**

```bash
python3 scripts/check-guide-dependency-snippets.py
CARGO_TARGET_DIR=/tmp/tenferro-trybuild-check \
  cargo test -p tenferro-tensor --test storage_compile_contract \
  storage_ui_compile_contracts -- --exact
```

Expected: both pass without snapshot changes.

- [ ] **Step 5: Verify explicit debugger override**

Run the guide and focused trybuild commands with `CARGO_PROFILE_DEV_DEBUG=2`; inspect one generated artifact and confirm `.debug_*` sections exist.

- [ ] **Step 6: Commit implementation**

```bash
git add .cargo/config.toml scripts/check-guide-dependency-snippets.py \
  scripts/ci/tests/test_build_artifact_contracts.py
git commit -m "build: keep nested Cargo fixtures debug-free"
```

### Task 4: Measure the implemented effect and document evidence

**Files:**
- Modify: `docs/worklogs/2026-08-09-nested-cargo-debug-profile.md`

**Interfaces:**
- Consumes: Task 1 baseline and Task 3 implementation.
- Produces: reviewer-facing measurements, decisions, verification, and residual risks.

- [ ] **Step 1: Repeat Task 1 with fresh candidate targets**

Use `target/guide-snippet-check` after deleting it and `/tmp/tenferro-trybuild-candidate`. Save outputs as `candidate-*`.

- [ ] **Step 2: Confirm candidate artifacts omit DWARF**

Use `readelf -SW` on representative candidate artifacts and record the absence of `.debug_*` sections.

- [ ] **Step 3: Write the work log**

Record commit/toolchain/system, exact commands, allocated bytes, wall time, percentage deltas, largest-artifact change, explicit override verification, design choice, rejected alternatives, and timing caveat.

- [ ] **Step 4: Commit evidence**

```bash
git add docs/worklogs/2026-08-09-nested-cargo-debug-profile.md
git commit -m "docs: record nested Cargo artifact reduction"
```

### Task 5: Review, verify, and deliver

**Files:**
- Review all branch changes.

- [ ] **Step 1: Run repository-required local verification**

```bash
python3 scripts/ci/run_profile.py fmt
python3 -m unittest scripts.ci.tests.test_build_artifact_contracts -v
bash scripts/check-pr-fast.sh --coverage-reviewed \
  --test 'python3 -m unittest scripts.ci.tests.test_build_artifact_contracts -v'
python3 scripts/repository-rules-review.py \
  --base origin/main --head HEAD \
  --output-json /tmp/tenferro-nested-cargo-rules-review.json
```

- [ ] **Step 2: Run independent specification and code-quality reviews**

Use fresh read-only reviewers. Fix confirmed findings, rerun focused and repository verification, and obtain clean follow-up review.

- [ ] **Step 3: Create the GitHub issue and PR**

Create a concise performance/bug issue with baseline evidence and acceptance criteria. Build the PR body from the repository template, link the issue and work log, push the branch, and create the PR with `gh pr create`.

- [ ] **Step 4: Enable and verify prescribed auto-merge**

```bash
gh pr merge --auto --squash --delete-branch <PR>
gh pr view <PR> --json autoMergeRequest,mergeStateStatus,state
```

- [ ] **Step 5: Babysit to merge**

Inspect every check. For failures, read logs, fix the cause in the isolated worktree, rerun invalidated local checks, push, and continue. If `origin/main` advances and the PR becomes behind, rebase/merge current `origin/main` as repository policy requires and rerun affected verification. Finish only when GitHub reports `state=MERGED` and record the merged commit and issue status.
