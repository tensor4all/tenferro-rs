# No-Facade Crate Boundaries Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the no-facade crate-boundary redesign from
`docs/plans/2026-05-25-no-facade-crate-boundaries-design.md` in one PR with
separate reviewable commits.

**Architecture:** Move ownership to direct public crates instead of feature
gates on the `tenferro` facade. Start with the tensor/device/GPU boundary,
then move runtime, AD, extension AD, docs, and cleanup in that order.

**Tech Stack:** Rust 2021 workspace, Cargo resolver 2, `computegraph`,
`strided-*`, `faer`/BLAS CPU backends, CubeCL/CUDA isolated in `tenferro-gpu`.

---

### Task 1: Implementation Plan Commit

**Files:**
- Create: `docs/plans/2026-05-26-no-facade-crate-boundaries-implementation.md`

**Step 1: Write the plan**

Add this implementation plan with commit boundaries, verification commands, and
the ordering below.

**Step 2: Verify docs hygiene**

Run:

```bash
LC_ALL=C rg -n "[^\x00-\x7F]" docs/plans/2026-05-26-no-facade-crate-boundaries-implementation.md || true
git diff --check
```

Expected: no output.

**Step 3: Commit**

```bash
git add docs/plans/2026-05-26-no-facade-crate-boundaries-implementation.md
git commit -m "docs: add no-facade implementation plan"
```

### Task 2: Rename Tensor Package And Absorb Device Model

**Files:**
- Modify: `Cargo.toml`
- Move: `tenferro-internal-tensor/` to `tenferro-tensor/`
- Modify: `tenferro-tensor/Cargo.toml`
- Modify: `tenferro-tensor/src/types.rs`
- Modify: `tenferro-tensor/src/error.rs`
- Move as needed: `tenferro-internal-device/src/batch_index.rs`
- Move as needed: CPU generator code from `tenferro-internal-device/src/generator.rs`
- Update imports in all workspace crates from package path

**Step 1: Write failing boundary tests**

Add tests in `tenferro-tensor/src/tests/types_tests.rs` proving:

- `DeviceId` is typed, comparable, and hashable.
- CPU default placement is `MemoryKind::UnpinnedHost` with no device.
- GPU placement uses `DeviceKind::Gpu(GpuBackendKind::Cuda)` without any
  CUDA runtime dependency.

Run:

```bash
cargo test -p tenferro-internal-tensor device_model
```

Expected: fail because the new device model does not exist yet.

**Step 2: Implement tensor-owned device model**

Add `DeviceKind`, `GpuBackendKind`, `DeviceId`, updated `MemoryKind`, and
updated `Placement` to `tenferro-tensor`. Keep it vendor-neutral. Do not move
CUDA runtime code here.

**Step 3: Rename package shell**

Rename package directory and workspace member from `tenferro-internal-tensor`
to `tenferro-tensor`. Keep `lib.name = "tenferro_tensor"`.

**Step 4: Verify**

Run:

```bash
cargo test -p tenferro-tensor types_tests
cargo check -p tenferro-tensor
```

Expected: pass.

**Step 5: Commit**

```bash
git add Cargo.toml tenferro-tensor
git commit -m "refactor(tensor): absorb device model into tensor core"
```

### Task 3: Extract GPU Backend Crate

**Files:**
- Create: `tenferro-gpu/Cargo.toml`
- Move: `tenferro-tensor/src/cubecl/**` to `tenferro-gpu/src/**`
- Move/absorb: `tenferro-internal-gpubackend/**` into `tenferro-gpu/src/**`
- Modify: `Cargo.toml`
- Modify: `tenferro-tensor/src/types.rs`

**Step 1: Write failing boundary checks**

Add a contract test or script check that `tenferro-tensor/Cargo.toml` does not
mention `cubecl`, `cudarc`, or `tenferro-gpu`.

Run:

```bash
rg -n "cubecl|cudarc|tenferro-gpu|tenferro-internal-gpubackend" tenferro-tensor/Cargo.toml tenferro-tensor/src
```

Expected before implementation: matches exist.

**Step 2: Implement `BackendBuffer<T>`**

Replace tensor-core `Buffer::Cubecl` with an opaque backend buffer trait object
owned by `tenferro-tensor`.

**Step 3: Move GPU implementation**

Move CubeCL/CUDA implementation into `tenferro-gpu`, define local
`CubeclBackend` and `CubeclBuffer<T>`, and implement `TensorBackend` plus
`BackendBuffer<T>` there.

**Step 4: Verify**

Run:

```bash
cargo check -p tenferro-tensor
cargo check -p tenferro-gpu --features cuda
python3 scripts/check-crate-boundaries.py
rg -n "cubecl|cudarc|tenferro-gpu|tenferro-internal-gpubackend" tenferro-tensor/Cargo.toml tenferro-tensor/src
```

Expected: checks pass; `rg` has no matches in tensor core.

**Step 5: Commit**

```bash
git add Cargo.toml tenferro-tensor tenferro-gpu
git rm -r tenferro-internal-gpubackend
git commit -m "refactor(gpu): extract CubeCL backend into tenferro-gpu"
```

Leave `tenferro-internal-device` for the later direct-crate cleanup commit:
today it is still the owner of einsum parsing/planning `Error`/`Result` types,
which should move with einsum rather than inside the GPU extraction.

### Task 4: Extract Runtime From Facade

**Files:**
- Move: `tenferro-internal-runtime/` to `tenferro-runtime/`
- Move from `tenferro/src`: `graph/`, `compiler/`, `traced.rs`,
  `traced_tensor.rs`, `exec.rs`, `segment.rs`, `eager_exec.rs`,
  runtime-owned metadata/cache modules, plus the runtime-facing eager,
  extension, concrete tensor helper, and typed tensor helper modules needed to
  keep the extracted crate internally coherent before the later AD split.
- Modify: `Cargo.toml`
- Modify extension crate dependencies.

**Step 1: Write failing direct-crate compile test**

Add or update a doctest/integration test using:

```rust
use tenferro_runtime::{CpuBackend, GraphCompiler, GraphExecutor, TracedTensor};
```

Expected before implementation: compile failure.

**Step 2: Move runtime modules**

Move graph/tracing/execution into `tenferro-runtime`. Re-export the existing
graph compiler/executor API directly from `tenferro_runtime`.

**Step 3: Verify**

Run:

```bash
cargo check -p tenferro-runtime
cargo test -p tenferro-runtime
```

Expected: pass. The `--no-default-features` form remains blocked at this point
by the existing tensor CPU backend compile-time contract that requires exactly
one CPU backend feature.

**Step 4: Commit**

```bash
git add Cargo.toml tenferro-runtime tenferro-tensor tenferro-einsum tenferro-linalg tenferro-fft
git commit -m "refactor(runtime): extract traced graph runtime"
```

### Task 5: Extract AD

**Files:**
- Create: `tenferro-ad/Cargo.toml`
- Move AD modules from `tenferro/src/eager*`, `tenferro/src/checkpoint.rs`,
  and `tenferro-internal-ops/src/ad/**`.
- Modify: `tenferro-ops`
- Modify: `Cargo.toml`

**Step 1: Write failing AD opt-in check**

Run:

```bash
cargo check -p tenferro-runtime --no-default-features
rg -n "chainrules|tidu|autodiff" tenferro-runtime/Cargo.toml tenferro-ops/Cargo.toml
```

Expected before implementation: AD dependencies or features still appear.

**Step 2: Move AD ownership**

Move primitive rules, AD extension traits, registries, checkpoint replay, and
user transforms into `tenferro-ad`.

**Step 3: Verify**

Run:

```bash
cargo check -p tenferro-runtime --no-default-features
cargo check -p tenferro-ad
```

Expected: pass; runtime and ops do not depend on AD crates.

**Step 4: Commit**

```bash
git add Cargo.toml tenferro-ad tenferro-runtime tenferro-ops
git commit -m "refactor(ad): extract automatic differentiation crate"
```

### Task 6: Split Linalg AD And Update Operation Crates

**Files:**
- Create: `tenferro-linalg-ad/Cargo.toml`
- Move: linalg AD rules from `tenferro-linalg/src/ad/**`
- Modify: `tenferro-linalg`, `tenferro-einsum`, `tenferro-fft`

**Step 1: Write failing feature-boundary checks**

Run:

```bash
rg -n "autodiff|tenferro/autodiff|tenferro-internal-ops/autodiff" tenferro-linalg/Cargo.toml tenferro-einsum/Cargo.toml tenferro-fft/Cargo.toml
```

Expected before implementation: matches exist.

**Step 2: Move linalg AD**

Keep primal linalg runtime registration in `tenferro-linalg`. Move AD rule
registration into `tenferro-linalg-ad`.

**Step 3: Verify**

Run:

```bash
cargo check -p tenferro-linalg --no-default-features
cargo check -p tenferro-linalg-ad
cargo check -p tenferro-einsum --no-default-features
cargo check -p tenferro-fft --no-default-features
```

Expected: pass.

**Step 4: Commit**

```bash
git add Cargo.toml tenferro-linalg tenferro-linalg-ad tenferro-einsum tenferro-fft
git commit -m "refactor(linalg): split linalg AD registration"
```

### Task 7: Remove Facade And Update Docs

**Files:**
- Modify: `AGENTS.md`
- Modify: `REPOSITORY_RULES.md`
- Modify: `README.md`
- Modify: `docs/design/**`
- Remove or empty: `tenferro/`

**Step 1: Write failing facade grep checks**

Run:

```bash
rg -n "use tenferro::|tenferro::(linalg|einsum|fft|cuda)|tenferro/" README.md docs AGENTS.md REPOSITORY_RULES.md
```

Expected before implementation: matches exist.

**Step 2: Remove facade paths**

Update examples to direct crate imports and remove the public `tenferro`
facade package from workspace members.

**Step 3: Verify**

Run:

```bash
cargo fmt --all --check
cargo check --workspace --no-default-features
cargo test --workspace --release
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: pass, subject to local tool availability.

**Step 4: Commit**

```bash
git add Cargo.toml AGENTS.md REPOSITORY_RULES.md README.md docs tenferro-* 
git commit -m "docs: update direct-crate public surface"
```

### Task 8: Final Verification And PR

**Files:**
- No planned source edits unless verification finds issues.

**Step 1: Run full required checks**

Run:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

**Step 2: Fix any failures in small commits**

Use focused commits such as `fix: update direct crate doctests`.

**Step 3: Create one PR**

Use `gh pr create`, then enable auto-merge with:

```bash
gh pr merge --auto --squash --delete-branch
```
