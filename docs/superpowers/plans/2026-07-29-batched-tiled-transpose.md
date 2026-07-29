# Batched Tiled Transpose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route compact `[1,0,2]`-class permutations through the existing compile-time shared-memory transpose kernel.

**Architecture:** Extend `NativePermutationPlan` with a checked three-dimensional tiled classification and a three-dimensional dispatch grid. Use `CUBE_POS_Z` to offset the unchanged two-dimensional tile algorithm by one compact matrix per batch. Both CUDA and wgpu launch the same CubeCL kernel and fall back to generic materialization when any grid dimension exceeds 65,535.

**Tech Stack:** Rust, CubeCL, wgpu/Metal, CubeCL CUDA, Cargo tests, tenferro-benchmark.

## Global Constraints

- Preserve profile-first ordering and the recorded Metal baseline.
- Keep tile width, block rows, padding, vector width, batch count, and batch stride as compile-time kernel metadata.
- Never introduce a host or CPU fallback.
- Preserve source/destination non-aliasing and bounds validation.
- Keep CUDA float/complex permutation on cuTENSOR.

---

### Task 1: Classify and Bound Batched Tile Plans

**Files:**
- Modify: `crates/tenferro-gpu/src/native_permutation.rs`

**Interfaces:**
- Consumes: `NativePermutationPlan::{dims,src_strides,dst_strides}` and `NativeTransposeTile`.
- Produces: `NativePermutationKind::TiledTranspose` for compact rank-three batches and `NativeTransposeTile::dispatch_grid(...) -> Result<Option<(u32, u32, u32)>>`.

- [ ] **Step 1: Write failing planner tests**

Add tests that construct `[256,256,240]` with source strides
`[256,1,65536]` and destination strides `[1,256,65536]`, expect
`TiledTranspose`, expect grid `(16,16,240)` for tile 16, and expect `None`
when the maximum dimension is below the batch count. Add a non-compact batch
stride case and expect `GenericStrided`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer native_permutation::tests::batched
```

Expected: the compact batch is classified `GenericStrided` or the grid API
does not provide a Z dimension.

- [ ] **Step 3: Implement the minimal classification**

Generalize the transpose predicate to accept rank two or rank three. For rank
three, require `src_strides[2] == dims[0] * dims[1]` and
`dst_strides[2] == dims[0] * dims[1]` using checked conversions and
multiplication. Return batch count one for rank two and `dims[2]` for rank
three. Extend `dispatch_grid` to validate X, Y, and Z against the supplied
runtime limit.

- [ ] **Step 4: Run the focused and complete planner tests**

Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer native_permutation::tests
```

Expected: all planner tests pass.

### Task 2: Launch the Batched Shared-Memory Kernel

**Files:**
- Modify: `crates/tenferro-gpu/src/kernels/structural.rs`
- Modify: `crates/tenferro-gpu/src/webgpu/structural.rs`
- Modify: `crates/tenferro-gpu/src/cubecl/mod.rs`
- Test: `crates/tenferro-gpu/tests/integration/webgpu_structural_runtime.rs`

**Interfaces:**
- Consumes: the three-dimensional grid and compact matrix batch stride from Task 1.
- Produces: `tiled_transpose_kernel` with compile-time `batch_stride`, shared by wgpu and CUDA.

- [ ] **Step 1: Add a failing WebGPU runtime assertion**

Extend the existing 3D transpose test to exercise `[1,0,2]` on a shape that
has multiple batches and partial tile edges. The existing host reference
remains the oracle.

- [ ] **Step 2: Run the focused runtime test and verify RED**

Temporarily assert the planner-selected tiled behavior through the public
contract test or kernel inventory. Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer \
  webgpu_structural_runtime::webgpu_transpose_f32_stays_on_device_and_matches_column_major_reference
```

Expected: failure until the launch signature and kernel batch offset exist.

- [ ] **Step 3: Implement batch addressing**

Add compile-time `batch_stride` to the kernel. Compute
`batch_base = CUBE_POS_Z as usize * batch_stride`, add it to `src_offset` for
loads, and add it to destination indices for stores. Pass
`dims[0] * dims[1]` from both launchers and use `CubeCount::Static(x, y, z)`.

- [ ] **Step 4: Run correctness, lint, and combined-feature checks**

Run:

```bash
cargo fmt --all -- --check
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer
cargo clippy -p tenferro-gpu --no-default-features --features webgpu,cpu-faer -- -D warnings
cargo check -p tenferro-gpu --no-default-features --features cuda,webgpu,cpu-faer
```

Expected: all commands exit zero.

### Task 3: Measure and Accept or Revert

**Files:**
- Modify: `docs/worklogs/2026-07-29-issue-1507-native-permutation-metal.md`
- Modify in tenferro-benchmark: `result/mac-gpu/gpu/permutation.md`

**Interfaces:**
- Consumes: the stable entry baseline and `16x8-p1-v1` selected tile.
- Produces: a correctness-checked Metal profile and an explicit retention decision.

- [ ] **Step 1: Run the selected Metal profile**

Run `scripts/run_gpu_permutation_mac.sh` from the benchmark worktree with the
clean tenferro revision and `TENFERRO_NATIVE_TRANSPOSE_TILE=16x8-p1-v1`.

- [ ] **Step 2: Compare stable medians**

Compare `mac_transpose_3d_102`, all other tenferro wgpu rows, and the memcpy
row against the prior stable run. Retain the change only if 3D improves and no
row violates the +20% gate under a comparable memcpy result.

- [ ] **Step 3: Record the result and commit**

Update the worklog and benchmark report with the measured result, run
`git diff --check`, and commit the accepted implementation and evidence. If
the hypothesis fails, revert the implementation commits and record the
negative experiment instead.
