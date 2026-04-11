# Documentation Restructure Plan

> **IMPORTANT**: Do NOT auto-implement. An agent must discuss the plan with a
> human reviewer and get explicit approval before writing any code.

## Principles

1. **Source of truth = source code** — doc comments on types/traits/functions
2. **Online docs = user-facing only** — `tenferro` facade crate の使い方
3. **No internal jargon in user docs** — Fragment, StableHLO, ExecOp 等は出さない
4. **All imports via `use tenferro::...`** — 内部 crate を直接参照しない
5. **All examples run as doctests** — `ignore` / `no_run` 禁止
6. **Internals page = pointer table** — 説明はソースコードの doc comments に
7. **AI agentic coding 前提** — 開発者は AGENTS.md → ソースコード

## Target site structure

```
tenferro-rs/ (http://tensor4all.org/tenferro-rs/)
├── index.html                        ← site top (API + Design cards)
├── api/                              ← Rustdoc (auto-generated, unchanged)
└── design/                           ← Quarto-rendered pages (below)
    ├── index.md                      ← tenferro とは + PyTorch/JAX 対応表
    ├── getting-started/
    │   ├── index.md                  ← インストール + 最初の例 (30秒)
    │   ├── core-concepts.md          ← TracedTensor / Engine / eval
    │   └── pytorch-jax-mapping.md    ← 操作対応表
    ├── guides/
    │   ├── tensor-operations.md      ← 作成、演算、reshape、broadcast、reduce
    │   ├── einsum.md                 ← einsum の使い方
    │   ├── linear-algebra.md         ← svd, qr, solve 等
    │   ├── autodiff.md              ← grad, vjp, jvp
    │   └── performance.md           ← column-major、スレッド数設定
    ├── api/
    │   └── index.md                  ← Rustdoc リンク (= 現 api_index.md)
    └── internals/
        └── index.md                  ← ポインタ表 + "AI agentic coding 前提"
```

## Step 1: Update REPOSITORY_RULES.md

Add documentation example rules (already drafted):
- `ignore` / `no_run` 禁止
- All examples must run as doctests

**Files**: `REPOSITORY_RULES.md`

## Step 2: Create new page files

### `docs/index.md` — Landing page (rewrite)

Content:
- "tenferro — General-purpose tensor computation in Rust"
- 3-sentence description (lazy eval, einsum, AD)
- PyTorch / JAX / tenferro comparison table (5 rows)
- Links to Getting Started, Guides, API, Internals

### `docs/getting-started/index.md` — Quick start (rewrite)

Content:
- Installation (Cargo.toml dependency)
- "Hello einsum" — minimal working example (5 lines)
- "Hello grad" — minimal gradient example (5 lines)
- Links to core-concepts, guides

Rules:
- All imports: `use tenferro::{...}`
- No `Tensor::F64(TypedTensor::from_vec(...))` — add convenience fn if needed
- All examples: no `ignore`, must run as doctest

### `docs/getting-started/core-concepts.md` — NEW

Content:
- TracedTensor: "operations record a graph, nothing executes yet"
  - Compare: PyTorch eager vs JAX jit vs tenferro lazy
- Engine: "holds backend + caches, executes graphs"
- eval(): "materializes the result"
- Typical flow diagram (text, no ASCII box art)

Rules:
- NO mention of Fragment, StableHLO, ExecProgram, computegraph
- NO mention of tenferro-tensor, tenferro-ops, etc.

### `docs/getting-started/pytorch-jax-mapping.md` — NEW

Content:
- Table: concept mapping (tensor creation, ops, einsum, autodiff, device)
- Table: function mapping (matmul, reshape, transpose, broadcast, reduce, etc.)
- Key differences section (column-major, lazy eval, engine ownership)

### `docs/guides/tensor-operations.md` — NEW (replaces quick-start ops section)

Content:
- Creating tensors (from data, zeros, ones)
- Elementwise: add, mul, exp, log, sin, ...
- Shape manipulation: reshape, transpose, broadcast
- Reduction: reduce_sum, reduce_prod
- Slicing, indexing (if exposed)
- All with runnable examples

### `docs/guides/einsum.md` — rewrite of einsum-examples.md

Content:
- What is einsum (brief, with PyTorch/NumPy comparison)
- Unary: transpose, trace, diagonal
- Binary: matmul, outer product, batch matmul
- N-ary: chain contraction, automatic path optimization
- All examples runnable, imports via `use tenferro::{...}`

### `docs/guides/linear-algebra.md` — NEW

Content:
- SVD, QR, Cholesky, Eigh, LU, Solve
- Each with runnable example
- Note: eval_all for multi-output ops

### `docs/guides/autodiff.md` — rewrite of autodiff-examples.md

Content:
- grad (scalar loss → gradient)
- vjp (vector-Jacobian product)
- jvp (Jacobian-vector product)
- Higher-order: HVP via jvp(vjp(...))
- Gradient through einsum
- Gradient through linalg
- PyTorch comparison for each

### `docs/guides/performance.md` — NEW

Content:
- Column-major storage: why, and what to watch out for
- Thread count: `CpuBackend::with_threads(n)`
- Buffer reuse: automatic via BufferPool
- Einsum optimization: contraction path selection

### `docs/api/index.md` — rewrite of api_index.md

Content:
- Brief: "tenferro is the main crate. Other crates are internal."
- Link to tenferro rustdoc (primary)
- Collapsed section: internal crate links (for contributors)

### `docs/internals/index.md` — NEW

Content:
```markdown
# Internals

内部設計の Source of Truth はソースコードです。
開発は AI agentic coding を前提としています。
[AGENTS.md](https://github.com/tensor4all/tenferro-rs/blob/main/AGENTS.md)
がエントリーポイントです。

| Topic | Location |
|---|---|
| Op vocabulary | `tenferro-ops/src/std_tensor_op.rs` |
| Backend contract | `tenferro-tensor/src/backend.rs` |
| Execution session | `tenferro-tensor/src/backend.rs` |
| AD rules | `tenferro-ops/src/ad/` |
| Compilation pipeline | `tenferro/src/compiler.rs` |
| Buffer pool | `tenferro-tensor/src/buffer_pool.rs` |
| CPU context | `tenferro-tensor/src/cpu/context.rs` |
| GPU design (future) | `docs/design/exec-session.md` |
```

## Step 3: Update `_quarto.yml`

Replace sidebar with new structure. Remove all architecture/, spec/,
design/ (except exec-session.md), oracle/, reference/ from render list.

```yaml
project:
  type: website
  output-dir: ../target/docs-site/design
  render:
    - index.md
    - getting-started/**/*.md
    - guides/**/*.md
    - api/**/*.md
    - internals/**/*.md

website:
  title: "tenferro"
  sidebar:
    style: docked
    contents:
      - index.md
      - section: "Getting Started"
        contents:
          - getting-started/index.md
          - getting-started/core-concepts.md
          - getting-started/pytorch-jax-mapping.md
      - section: "Guides"
        contents:
          - guides/tensor-operations.md
          - guides/einsum.md
          - guides/linear-algebra.md
          - guides/autodiff.md
          - guides/performance.md
      - api/index.md
      - internals/index.md
```

## Step 4: Move old docs out of render path

Do NOT delete — move to `docs/_archive/` (or just remove from _quarto.yml
render list). The files stay in git for history but are not rendered.

Files to stop rendering:
- `docs/architecture/` — all 6 files
- `docs/spec/` — all 6 files
- `docs/design/` — all except `exec-session.md` and `supported-ops.md`
- `docs/oracle/` — both files
- `docs/reference/` — all 9 files

Keep rendering:
- `docs/design/exec-session.md` — future GPU design (linked from internals)
- `docs/design/supported-ops.md` — CPU/GPU coverage (linked from internals)

## Step 5: Add convenience API to tenferro facade

To make examples clean, add helpers if missing:

```rust
// tenferro/src/lib.rs or traced.rs
impl TracedTensor {
    /// Create a traced tensor from f64 data.
    /// Equivalent to TracedTensor::from_tensor(Tensor::F64(TypedTensor::from_vec(shape, data)))
    pub fn from_f64(shape: &[usize], data: &[f64]) -> Self { ... }
}
```

This makes examples:
```rust
let a = TracedTensor::from_f64(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
```

Instead of:
```rust
let a = TracedTensor::from_tensor(
    Tensor::F64(TypedTensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
);
```

## Step 6: Fix all examples to be runnable doctests

Go through every example in the new docs and ensure:
- No `ignore` or `no_run` attribute
- Imports only from `use tenferro::{...}`
- Each example compiles and runs

## Step 7: Update check-docs-site.py if needed

The script checks rustdoc output and api_index links. If `api_index.md`
moves to `docs/api/index.md`, update the script's default path.

## Step 8: Verify

```bash
cargo fmt --all
cargo test --workspace --release   # includes doctests
cargo doc --workspace --no-deps
scripts/build_docs_site.sh
python3 scripts/check-docs-site.py
```

## File change summary

| Action | Files |
|---|---|
| **Rewrite** | `docs/index.md`, `docs/getting-started/index.md` |
| **Rewrite** | `docs/getting-started/einsum-examples.md` → `docs/guides/einsum.md` |
| **Rewrite** | `docs/getting-started/autodiff-examples.md` → `docs/guides/autodiff.md` |
| **New** | `docs/getting-started/core-concepts.md` |
| **New** | `docs/getting-started/pytorch-jax-mapping.md` |
| **New** | `docs/guides/tensor-operations.md` |
| **New** | `docs/guides/linear-algebra.md` |
| **New** | `docs/guides/performance.md` |
| **New** | `docs/api/index.md` |
| **New** | `docs/internals/index.md` |
| **Rewrite** | `docs/_quarto.yml` |
| **Edit** | `REPOSITORY_RULES.md` (example rules) |
| **Edit** | `tenferro/src/lib.rs` or `traced.rs` (convenience API) |
| **Move** | Old docs out of render path |
| **Edit** | `scripts/check-docs-site.py` (path update if needed) |
