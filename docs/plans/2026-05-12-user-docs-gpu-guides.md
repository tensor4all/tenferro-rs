# User Documentation and GPU Guides Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `/tenferro-rs/` a beginner-friendly user documentation landing page, add checked CPU/CUDA quickstarts, and remove stale or implementation-leaking public docs.

**Architecture:** Treat executable examples as the source of truth for guide snippets. Expose a user-facing `tenferro::cuda` facade over the existing CubeCL backend while keeping `cubecl` and `tenferro-cubecl` as internal implementation names. Let Quarto own the public site root and keep rustdoc under `/api/`.

**Tech Stack:** Rust, Cargo examples, Quarto docs, Python snippet sync/check script, existing `scripts/build_docs_site.sh` and `scripts/check-docs-site.py`.

---

## Precondition

- Current branch: `docs/user-docs-gpu-guides`.
- Design commit exists: `d641407 docs: design user documentation refresh`.
- Design doc: `docs/plans/2026-05-12-user-docs-gpu-guides-design.md`.

## Task 1: Add Public CUDA Facade And Checked Examples

**Files:**
- Modify: `tenferro/Cargo.toml`
- Modify: `tenferro/src/lib.rs`
- Create: `tenferro/examples/cpu_quickstart.rs`
- Create: `tenferro/examples/cuda_quickstart.rs`

**Step 1: Run the expected missing-example checks**

Run:

```bash
cargo check -p tenferro --example cpu_quickstart
cargo check -p tenferro --features cuda --example cuda_quickstart
```

Expected: both fail because the examples do not exist and the `cuda` feature
alias does not exist yet.

**Step 2: Add the `cuda` feature alias**

In `tenferro/Cargo.toml`, add:

```toml
cuda = ["cubecl"]
```

Keep the existing `cubecl = [...]` feature unchanged.

**Step 3: Add the facade module**

In `tenferro/src/lib.rs`, below the existing public re-exports, add:

```rust
#[cfg(feature = "cubecl")]
pub mod cuda {
    pub use tenferro_tensor::cubecl::{
        download_tensor, gpu_available, upload_tensor, CubeclBackend as CudaBackend,
    };
}
```

Do not rename the internal `tenferro-tensor/src/cubecl/` module in this task.
The user-facing alias is enough for docs and examples.

**Step 4: Add the CPU quickstart example**

Create `tenferro/examples/cpu_quickstart.rs`:

```rust
use tenferro::{CpuBackend, Tensor, TensorBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::new();

    let a = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
    let b = Tensor::from_vec(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]);

    let c = a.matmul(&b, &mut backend)?;

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.as_slice::<f64>().unwrap(), &[19.0, 43.0, 22.0, 50.0]);

    Ok(())
}
```

The flat data is column-major. The logical matrices are:

```text
a = [[1, 2],
     [3, 4]]
b = [[5, 6],
     [7, 8]]
c = [[19, 22],
     [43, 50]]
```

**Step 5: Add the CUDA quickstart example**

Create `tenferro/examples/cuda_quickstart.rs`:

```rust
use tenferro::cuda::{download_tensor, upload_tensor, CudaBackend};
use tenferro::{Tensor, TensorBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CudaBackend::new(0)?;

    let a = Tensor::from_vec(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let b = Tensor::from_vec(vec![3], vec![4.0_f64, 5.0, 6.0]);

    let gpu_a = upload_tensor(backend.runtime(), &a)?;
    let gpu_b = upload_tensor(backend.runtime(), &b)?;
    let gpu_c = backend.add(&gpu_a, &gpu_b)?;
    let c = download_tensor(backend.runtime(), &gpu_c)?;

    assert_eq!(c.shape(), &[3]);
    assert_eq!(c.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);

    Ok(())
}
```

This example intentionally uses a simple supported operation so it is a stable
first GPU smoke test.

**Step 6: Verify examples compile**

Run:

```bash
cargo check -p tenferro --example cpu_quickstart
cargo check -p tenferro --features cuda --example cuda_quickstart
```

Expected: both pass.

**Step 7: Verify CPU quickstart runs**

Run:

```bash
cargo run -p tenferro --example cpu_quickstart
```

Expected: exits successfully with no assertion failure.

**Step 8: Commit**

```bash
git add tenferro/Cargo.toml tenferro/src/lib.rs tenferro/examples/cpu_quickstart.rs tenferro/examples/cuda_quickstart.rs
git commit -m "docs: add checked cpu and cuda quickstarts"
```

## Task 2: Add Snippet Synchronization Check

**Files:**
- Create: `scripts/check-doc-snippets.py`
- Modify: `scripts/check-docs-site.py`
- Modify: `scripts/build_docs_site.sh`

**Step 1: Create the snippet checker**

Create `scripts/check-doc-snippets.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import re
import sys

START_RE = re.compile(r"<!--\s*snippet-source:\s*([^>]+?)\s*-->")
END_RE = re.compile(r"<!--\s*end-snippet-source\s*-->")


def fenced(source: pathlib.Path) -> str:
    return "```rust\n" + source.read_text().rstrip() + "\n```\n"


def rewrite_doc(root: pathlib.Path, doc: pathlib.Path) -> tuple[str, bool]:
    text = doc.read_text()
    out: list[str] = []
    pos = 0
    changed = False
    while True:
        start = START_RE.search(text, pos)
        if not start:
            out.append(text[pos:])
            break
        end = END_RE.search(text, start.end())
        if not end:
            raise ValueError(f"{doc}: missing end-snippet-source marker")
        source_rel = start.group(1).strip()
        source = (root / source_rel).resolve()
        if not source.is_file():
            raise ValueError(f"{doc}: snippet source does not exist: {source_rel}")
        replacement = (
            text[start.start() : start.end()]
            + "\n"
            + fenced(source)
            + text[end.start() : end.end()]
        )
        current = text[start.start() : end.end()]
        out.append(text[pos : start.start()])
        out.append(replacement)
        changed = changed or current != replacement
        pos = end.end()
    return "".join(out), changed


def user_facing_docs(root: pathlib.Path) -> list[pathlib.Path]:
    docs_root = root / "docs"
    excluded_parts = {"plans", "superpowers", "design", "architecture", "spec", "reference", "oracle"}
    docs: list[pathlib.Path] = []
    for path in sorted(docs_root.rglob("*.md")):
        relative = path.relative_to(docs_root)
        if relative.parts and relative.parts[0] in excluded_parts:
            continue
        docs.append(path)
    return docs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = pathlib.Path(args.root_dir).resolve()
    changed_docs: list[pathlib.Path] = []
    for doc in user_facing_docs(root):
        new_text, changed = rewrite_doc(root, doc)
        if changed:
            changed_docs.append(doc)
            if not args.check:
                doc.write_text(new_text)

    if changed_docs and args.check:
        print("stale doc snippets:", file=sys.stderr)
        for doc in changed_docs:
            print(f"- {doc.relative_to(root)}", file=sys.stderr)
        print("run: python3 scripts/check-doc-snippets.py", file=sys.stderr)
        return 1

    if changed_docs:
        print(f"updated {len(changed_docs)} doc snippet file(s)")
    else:
        print("doc-snippets-ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 2: Make it executable**

Run:

```bash
chmod +x scripts/check-doc-snippets.py
```

**Step 3: Wire snippet checking into docs checks**

In `scripts/check-docs-site.py`, near the start of `main()` after `root` is
computed, run the checker:

```python
    snippet_check = subprocess.run(
        [sys.executable, str(root / "scripts" / "check-doc-snippets.py"), "--root-dir", str(root), "--check"],
        check=False,
    )
    if snippet_check.returncode != 0:
        return snippet_check.returncode
```

Also add `import subprocess` at the top.

**Step 4: Wire snippet checking into full docs build**

In `scripts/build_docs_site.sh`, before `[1/5] Building rustdoc`, add:

```bash
echo "[0/6] Checking user-facing snippets"
python3 "$ROOT_DIR/scripts/check-doc-snippets.py" --root-dir "$ROOT_DIR" --check
```

Do not auto-rewrite snippets inside the build script; CI should fail on drift.

**Step 5: Verify checker succeeds before docs use markers**

Run:

```bash
python3 scripts/check-doc-snippets.py --check
python3 scripts/check-docs-site.py
```

Expected: both pass.

**Step 6: Commit**

```bash
git add scripts/check-doc-snippets.py scripts/check-docs-site.py scripts/build_docs_site.sh
git commit -m "docs: check user guide snippets"
```

## Task 3: Rewrite Landing And Getting Started Around Checked CPU Example

**Files:**
- Modify: `docs/index.md`
- Modify: `docs/getting-started/index.md`
- Modify: `scripts/build_docs_site.sh`

**Step 1: Stop overwriting the Quarto root page**

Modify the last section of `scripts/build_docs_site.sh` so it only writes a
fallback `index.html` when Quarto did not render one:

```bash
echo "[6/6] Verifying site top page"
if [[ -f "$OUT_DIR/index.html" ]]; then
  echo "  Using Quarto-rendered site index."
else
  echo "  Quarto index missing; generating fallback top page."
  cat >"$OUT_DIR/index.html" <<EOF
<!doctype html>
<html lang="en">
  <head><meta charset="utf-8" /><title>${REPO_TITLE} docs</title></head>
  <body>
    <main>
      <h1>${REPO_TITLE} Documentation</h1>
      <p>Build the full documentation with Quarto to see the user guide landing page.</p>
      <p><a href="./api/index.html">API Reference</a></p>
    </main>
  </body>
</html>
EOF
fi
```

Remove the old generated API/design selector HTML from this step.

**Step 2: Rewrite `docs/index.md`**

Replace stale root content with:

````markdown
# tenferro

tenferro is a dense tensor computation library for Rust users who want
PyTorch- and JAX-style tensor workflows with explicit Rust ownership and
backend control.

It supports:

- eager CPU tensor operations with `Tensor`, `TypedTensor`, and `CpuBackend`,
- eager scalar-loss reverse-mode AD with `EagerTensor`,
- lazy traced execution with `TracedTensor` and `Engine`,
- transform AD with `grad`, `vjp`, `jvp`, and HVP composition,
- einsum and linear algebra,
- experimental CUDA execution for selected operations.

## Start Here

- [Getting Started](getting-started/index.md)
- [Choosing an API](guides/choosing-an-api.md)
- [Devices and GPU](guides/devices-and-gpu.md)
- [API Reference](api/index.md)

## First CPU Example

<!-- snippet-source: tenferro/examples/cpu_quickstart.rs -->
```rust
placeholder
```
<!-- end-snippet-source -->

## Mental Model

| Workflow | Use |
| --- | --- |
| Direct CPU computation | `Tensor` or `TypedTensor` with `CpuBackend` |
| Scalar-loss eager AD | `EagerTensor` with `EagerContext` |
| Transform AD and graph optimization | `TracedTensor` with `Engine` |
| CUDA execution | `tenferro::cuda::CudaBackend` with explicit upload/download |
````

Then run `python3 scripts/check-doc-snippets.py` to replace the placeholder
with the executable example.

**Step 3: Rewrite `docs/getting-started/index.md`**

Keep installation content, but:

- use only `use tenferro::{...}` imports,
- remove the recommendation to depend on `tenferro-tensor` in the beginner
  path,
- include the same CPU quickstart snippet through the marker,
- link to `guides/devices-and-gpu.md` for CUDA.

Add this section after installation:

````markdown
## First CPU Program

<!-- snippet-source: tenferro/examples/cpu_quickstart.rs -->
```rust
placeholder
```
<!-- end-snippet-source -->
````

**Step 4: Sync snippets**

Run:

```bash
python3 scripts/check-doc-snippets.py
python3 scripts/check-doc-snippets.py --check
```

Expected: first command updates docs, second command passes.

**Step 5: Build docs site**

Run:

```bash
bash scripts/build_docs_site.sh
python3 scripts/check-docs-site.py
```

Expected:

- docs build succeeds,
- `target/docs-site/index.html` is the Quarto-rendered tenferro landing page,
- docs-site check passes.

**Step 6: Commit**

```bash
git add docs/index.md docs/getting-started/index.md scripts/build_docs_site.sh
git commit -m "docs: make site root a user landing page"
```

## Task 4: Add Beginner Guides

**Files:**
- Create: `docs/guides/choosing-an-api.md`
- Create: `docs/guides/installation.md`
- Create: `docs/guides/devices-and-gpu.md`
- Create: `docs/guides/memory-order.md`
- Create: `docs/guides/troubleshooting.md`
- Modify: `docs/_quarto.yml`
- Modify: `docs/getting-started/pytorch-jax-mapping.md`
- Modify: `docs/guides/performance.md`

**Step 1: Add guide pages to Quarto nav**

In `docs/_quarto.yml`, update the Guides section to:

```yaml
      - section: "Guides"
        contents:
          - guides/choosing-an-api.md
          - guides/installation.md
          - guides/eager-operations.md
          - guides/tensor-operations.md
          - guides/einsum.md
          - guides/linear-algebra.md
          - guides/autodiff.md
          - guides/devices-and-gpu.md
          - guides/memory-order.md
          - guides/performance.md
          - guides/troubleshooting.md
```

**Step 2: Create `guides/choosing-an-api.md`**

Include:

```markdown
# Choosing an API

Use the simplest tensor layer that matches the workflow.

| Need | Use |
| --- | --- |
| Direct concrete computation | `Tensor` + `CpuBackend` |
| Compile-time scalar type while still owning dense data | `TypedTensor<T>` |
| PyTorch-style scalar-loss `backward()` | `EagerTensor` + `EagerContext` |
| `grad`, `vjp`, `jvp`, HVP, graph optimization | `TracedTensor` + `Engine` |

## Rule of Thumb

Start with `Tensor` for concrete CPU work. Move to `EagerTensor` when you need
gradient accumulation, and move to `TracedTensor` when you need transform AD or
graph reuse.
```

**Step 3: Create `guides/installation.md`**

Document:

- default `cpu-faer`,
- `cpu-blas`,
- `src-openblas`,
- `cuda` feature alias,
- CUDA runtime env vars,
- link to `devices-and-gpu.md`.

Use only `tenferro` dependency snippets.

**Step 4: Create `guides/devices-and-gpu.md`**

Include this CUDA quickstart marker:

````markdown
# Devices and GPU

tenferro follows the PyTorch convention: no implicit CPU/GPU transfer. Upload
CPU tensors before CUDA backend operations and download results before host
inspection.

CUDA support is partial and experimental. It currently targets NVIDIA CUDA.
AMD/ROCm is not a supported execution path yet.

## CUDA Quickstart

<!-- snippet-source: tenferro/examples/cuda_quickstart.rs -->
```rust
placeholder
```
<!-- end-snippet-source -->

Compile-check the example without requiring a GPU:

```bash
cargo check -p tenferro --features cuda --example cuda_quickstart
```

Run it on a configured CUDA machine:

```bash
CUBECL_DEBUG_LOG=0 \
CUDA_PATH=/usr/local/cuda-12.0 \
LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH \
  cargo run -p tenferro --features cuda --example cuda_quickstart
```

The example downloads the result back to CPU and asserts the expected values.
````

Also include a short coverage table:

```markdown
| Area | Status |
| --- | --- |
| Allocation and transfer | CUDA supported |
| Elementwise and reductions | partial |
| Structural/indexing | partial |
| Contractions | selected cuTENSOR/cuBLAS-backed paths |
| Linalg | selected cuSOLVER/cuBLAS-backed paths |
| General `eig` | not supported by cuSOLVER; download to CPU |
| AMD/ROCm | not supported yet |
```

**Step 5: Create `guides/memory-order.md`**

Document:

- default column-major construction,
- row-major owned import via `Tensor::from_vec_row_major`,
- conversion via `to_col_major()` and `to_row_major()`,
- zero-copy owned export only when requested order matches.

Include a small CPU example with assertions:

```rust
use tenferro::{MemoryOrder, Tensor};

let row = Tensor::from_vec_row_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
assert_eq!(row.order(), MemoryOrder::RowMajor);

let col = row.to_col_major().unwrap();
assert_eq!(col.order(), MemoryOrder::ColMajor);
assert_eq!(col.as_slice::<f64>().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
```

**Step 6: Create `guides/troubleshooting.md`**

Cover:

- CUDA library load failures and env vars,
- `expected GPU tensor ... use upload_tensor()` errors,
- panic on host access to GPU tensors and `download_tensor`,
- dtype mismatch,
- column-major/row-major confusion,
- CPU backend feature conflicts.

**Step 7: Refresh existing mapping/performance pages**

In `docs/getting-started/pytorch-jax-mapping.md`:

- add CUDA row in device/runtime mapping,
- mention explicit upload/download,
- mention row-major import helper.

In `docs/guides/performance.md`:

- update column-major section to mention row-major owned import/export,
- add a short "CUDA transfer boundaries" section linking to
  `devices-and-gpu.md`.

**Step 8: Sync snippets and build docs**

Run:

```bash
python3 scripts/check-doc-snippets.py
python3 scripts/check-doc-snippets.py --check
bash scripts/build_docs_site.sh
python3 scripts/check-docs-site.py
```

Expected: all pass.

**Step 9: Commit**

```bash
git add docs/_quarto.yml docs/guides docs/getting-started/pytorch-jax-mapping.md docs/guides/performance.md
git commit -m "docs: add beginner gpu and api guides"
```

## Task 5: Clean API And Internal Naming In Public Docs

**Files:**
- Modify: `docs/api/index.md`
- Modify: `README.md`
- Modify: `docs/design/gpu-backend-design.md`
- Modify: `docs/design/supported-ops.md`
- Modify as needed: other files found by `rg`

**Step 1: Search for stale public claims**

Run:

```bash
rg -n "GPU support is planned|GPU backend stubs|tenferro-algebra|tenferro_tensor|tenferro-tensor|CubeCL|cubecl|tenferro-cubecl" README.md docs/index.md docs/getting-started docs/guides docs/api
```

Expected: results identify stale or implementation-leaking public docs.

**Step 2: Update `docs/api/index.md`**

Replace the current workspace crate bullets with current crates:

```markdown
- [tenferro](../rustdoc/tenferro/index.html): user-facing facade for eager
  tensors, traced tensors, einsum, linalg, AD, and backend selection
- [tenferro-tensor](../rustdoc/tenferro_tensor/index.html): dense runtime
  tensors, backend traits, CPU backend, and internal CUDA backend integration
- [tenferro-einsum](../rustdoc/tenferro_einsum/index.html): subscripts,
  contraction planning, and lowering helpers
- [tenferro-ops](../rustdoc/tenferro_ops/index.html): graph op vocabulary and
  AD rule implementations
- [tenferro-device](../rustdoc/tenferro_device/index.html): shared device and
  error infrastructure
- [tenferro-cubecl](../rustdoc/tenferro_cubecl/index.html): internal CubeCL
  kernel crate used by the CUDA backend
- [tenferro-extension-macros](../rustdoc/tenferro_extension_macros/index.html):
  procedural macros for extension-op registration
```

Remove `tenferro-algebra` from the public API page if it is not a workspace
member.

**Step 3: Update `README.md` docs bullets**

Make the docs links match the new guide set:

```markdown
- [Getting Started](https://tensor4all.org/tenferro-rs/getting-started/) — install and run the first checked CPU example
- [Guides](https://tensor4all.org/tenferro-rs/guides/choosing-an-api.html) — API selection, tensor ops, einsum, linalg, autodiff, memory order, and CUDA
- [API Reference](https://tensor4all.org/tenferro-rs/api/) — rustdoc links for every crate
- [Internals](https://tensor4all.org/tenferro-rs/internals/) — architecture, specification, contributor pointers
```

**Step 4: Keep CubeCL in developer docs only**

In `docs/design/gpu-backend-design.md` and `docs/design/supported-ops.md`, make
the split explicit:

- user-facing docs say CUDA,
- developer docs say CubeCL implementation,
- `tenferro-cubecl` is an internal kernel crate.

**Step 5: Verify public docs no longer leak internals**

Run:

```bash
rg -n "tenferro_tensor|tenferro-tensor|CubeCL|cubecl|tenferro-cubecl" README.md docs/index.md docs/getting-started docs/guides
```

Expected: no results except intentional explanations in troubleshooting or
GPU guide if needed. Prefer "CUDA backend" in user docs.

**Step 6: Build docs**

Run:

```bash
python3 scripts/check-doc-snippets.py --check
bash scripts/build_docs_site.sh
python3 scripts/check-docs-site.py
```

Expected: all pass.

**Step 7: Commit**

```bash
git add README.md docs/api/index.md docs/design/gpu-backend-design.md docs/design/supported-ops.md docs/index.md docs/getting-started docs/guides
git commit -m "docs: align public api and gpu terminology"
```

## Task 6: Final Verification

**Files:**
- No new files unless fixes are required.

**Step 1: Run formatting**

```bash
cargo fmt --all --check
```

Expected: pass. If it fails, run `cargo fmt --all`, inspect the diff, and
commit formatting with the affected task.

**Step 2: Run example checks**

```bash
cargo check -p tenferro --example cpu_quickstart
cargo run -p tenferro --example cpu_quickstart
cargo check -p tenferro --features cuda --example cuda_quickstart
```

Expected: all pass.

**Step 3: Run docs checks**

```bash
python3 scripts/check-doc-snippets.py --check
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
bash scripts/build_docs_site.sh
```

Expected: all pass.

**Step 4: Run focused tests**

```bash
cargo test -p tenferro --release
cargo test -p tenferro-tensor --features cubecl metadata_tests
```

Expected: all pass. The CubeCL metadata test compiles the CubeCL feature
without requiring a CUDA device.

**Step 5: Inspect final diff**

```bash
git status -sb
git diff --stat origin/main...HEAD
rg -n "GPU support is planned|GPU backend stubs|tenferro-algebra" README.md docs
```

Expected:

- worktree clean except intentional uncommitted fixes before the final commit,
- no stale public claims,
- final diff contains docs, examples, facade alias, and snippet-check script.

**Step 6: Commit final fixes if needed**

If Step 5 found small residual fixes:

```bash
git add <files>
git commit -m "docs: finalize user docs refresh"
```

Do not create a PR until the repository PR checklist has been rerun or the user
explicitly asks for PR creation.
