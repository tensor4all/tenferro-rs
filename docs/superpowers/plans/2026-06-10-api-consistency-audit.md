# API Consistency Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repeatable first-pass audit tool that inventories lexically public APIs and reports release-freeze API convention inconsistencies as reviewable candidates.

**Architecture:** Keep the first slice as a repository-local Python script under `scripts/` so it matches existing boundary checks and avoids adding a workspace crate. The script derives lexically public items from `crates/*/src/**/*.rs`, scans user-facing docs and published-crate features, groups same-concept API surfaces, and prints a markdown audit report without modifying source files.

**Tech Stack:** Python 3.11+ standard library (`argparse`, `dataclasses`, `pathlib`, `re`, `tomllib`), existing Cargo workspace layout, existing docs convention from `docs/design/api-and-convention-freeze.md`.

---

### Task 1: Add Public API Consistency Script

**Files:**
- Create: `scripts/check-api-consistency.py`

- [ ] **Step 1: Create the script skeleton**

Create `scripts/check-api-consistency.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class PublicItem:
    crate: str
    crate_path: pathlib.Path
    file: pathlib.Path
    line: int
    kind: str
    name: str
    signature: str


@dataclasses.dataclass(frozen=True)
class Finding:
    category: str
    location: str
    evidence: str
    expected: str
```

- [ ] **Step 2: Implement workspace discovery**

Add `workspace_crates(root: pathlib.Path) -> list[tuple[str, pathlib.Path]]` that reads the workspace members from `Cargo.toml`, loads each member `Cargo.toml`, and returns library crates only. It must skip `docs/tutorial-code` because it is not a release API crate.

- [ ] **Step 3: Implement public item collection**

Add `collect_public_items(root, crates)` that scans each crate's `src/**/*.rs`, skips path components named `tests`, and records `pub` functions, structs, enums, traits, type aliases, constants, modules, and reexports while excluding `pub(crate)`, `pub(super)`, `pub(self)`, and `pub(in ...)`.

- [ ] **Step 4: Implement convention findings**

Add checks for:

```text
traced_prefix: public item name starts with traced_
read_suffix_without_read_input: public function name ends _read but signature does not mention TensorRead or Read
per_dtype_constructor: public function name looks like from_f32/from_f64/from_i32/from_i64/from_bool/from_c32/from_c64
public_gpu_feature: a crate Cargo.toml exposes a feature literally named gpu
facade_path_in_user_docs: README.md or docs/guides/*.md contains tenferro::
internal_jargon_in_user_docs: README.md or docs/guides/*.md contains internal crate paths or internal graph/IR vocabulary
```

- [ ] **Step 5: Implement concept-family output**

Group public function names for these concepts and print the affected surfaces:

```python
CONCEPT_PATTERNS = {
    "reshape": re.compile(r"reshape"),
    "transpose": re.compile(r"transpose"),
    "slice": re.compile(r"slice"),
    "broadcast": re.compile(r"broadcast"),
    "matmul/dot_general": re.compile(r"matmul|dot_general"),
    "reduce": re.compile(r"reduce_(sum|prod|max|min)"),
    "gather/scatter": re.compile(r"gather|scatter"),
    "pad/concatenate/reverse": re.compile(r"pad|concatenate|reverse"),
    "convert": re.compile(r"convert"),
    "flat-buffer constructors": re.compile(r"from_vec_(col|row)_major|into_vec_(col|row)_major"),
}
```

- [ ] **Step 6: Implement CLI**

Support:

```bash
python3 scripts/check-api-consistency.py
python3 scripts/check-api-consistency.py --fail-on-findings
python3 scripts/check-api-consistency.py --output /tmp/api-consistency.md
```

Default mode exits `0` and prints an audit report. `--fail-on-findings` exits
`1` when convention findings are present.

### Task 2: Verify Script Behavior

**Files:**
- Modify: `scripts/check-api-consistency.py`

- [ ] **Step 1: Compile-check Python**

Run:

```bash
python3 -m py_compile scripts/check-api-consistency.py
```

Expected: exit code `0`.

- [ ] **Step 2: Run default audit**

Run:

```bash
python3 scripts/check-api-consistency.py --output /tmp/tenferro-api-consistency.md
```

Expected: exit code `0`, stdout includes `api-consistency-report:`, and `/tmp/tenferro-api-consistency.md` exists.

- [ ] **Step 3: Run failing check mode**

Run:

```bash
python3 scripts/check-api-consistency.py --fail-on-findings
```

Expected: exit code `1` if current API convention findings exist, otherwise `0`. Do not wire this into CI until findings are triaged.

### Task 3: Commit First Audit Tooling Slice

**Files:**
- Add: `scripts/check-api-consistency.py`
- Add: `docs/superpowers/plans/2026-06-10-api-consistency-audit.md`

- [ ] **Step 1: Review diff**

Run:

```bash
git diff -- scripts/check-api-consistency.py docs/superpowers/plans/2026-06-10-api-consistency-audit.md
```

Expected: only the new script and this plan are changed.

- [ ] **Step 2: Commit**

Run:

```bash
git add scripts/check-api-consistency.py docs/superpowers/plans/2026-06-10-api-consistency-audit.md
git commit -m "tools: add api consistency audit"
```

Expected: commit succeeds.
