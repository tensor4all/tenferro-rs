# CubeCL Native Permutation Metal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a reproducible Metal baseline, then unify and optimize tenferro's native CubeCL permutation path with shared dimension fusion and compile-time tiled transpose specializations.

**Architecture:** The work is deliberately ordered across `tenferro-rs`, `tenferro-benchmark`, and `strided-rs`: WebGPU measurement wiring first, an immutable Metal baseline second, shared host planning third, and kernel changes only afterward. CUDA's cuTENSOR route remains intact; native CUDA and WebGPU share the same CubeCL materialization kernels while runtime-specific launch code owns placement and capability checks.

**Tech Stack:** Rust 2021, CubeCL CUDA/WebGPU, wgpu Metal/Vulkan, `strided-perm`, JSONL/YAML benchmark artifacts, Python 3.12, PyTorch MPS, best-effort JAX Metal.

## Global Constraints

- Run all implementation work in isolated worktrees based on each repository's current `origin/main`.
- Do not modify the bodies, launch geometry, dimension decoding, or specialization behavior of existing structural kernels before the Step 0 Metal baseline is committed.
- The baseline dtype is `F32`; never substitute CPU execution for an unavailable Metal participant.
- Record the actual local device as Apple M5 Max; explain that the user approved it in place of the issue's named M4.
- Keep CUDA `F32`, `F64`, `C32`, and `C64` permutation on cuTENSOR.
- Share bilateral fusion through a public `strided-perm` API; do not copy the fusion algorithm into `tenferro-gpu`.
- Tile width, tile height, padding, workgroup dimensions, and vector width are CubeCL compile-time parameters.
- Validate placement, overlap, metadata, bounds, cube counts, and shared-memory limits before launch; never fall back to CPU or transfer implicitly.
- Run Linux A100 verification with both CubeCL CUDA and CubeCL wgpu/Vulkan; treat those timings as diagnostic and Metal as the optimization decision baseline.
- Stop and investigate any comparable Metal row that regresses by at least 20%.
- Inspect upstream reference paths when tenferro is more than 2x slower, or when a reference reaches at least 70% of the copy ceiling while tenferro remains below 35%.
- Add precise source provenance when third-party implementation details influence code; do not change scientific citation policy without separate approval.
- Use Python 3.12 for repository documentation gates because the system Python 3.9 lacks `enum.StrEnum`.

---

## File Map

### `tenferro-rs`

- `crates/tenferro-gpu/src/lib.rs`: compile native structural kernels for both CUDA and WebGPU.
- `crates/tenferro-gpu/src/webgpu/mod.rs`: register WebGPU structural traits and delegate focused launch work.
- `crates/tenferro-gpu/src/webgpu/structural.rs`: WebGPU structural validation, typed dispatch, and launches.
- `crates/tenferro-gpu/src/native_permutation.rs`: runtime-independent launch planning and specialization classification.
- `crates/tenferro-gpu/src/kernels/structural.rs`: generic fused materialization and tiled transpose CubeCL kernels.
- `crates/tenferro-gpu/tests/integration/webgpu_structural_runtime.rs`: end-to-end Metal/WebGPU structural correctness.
- `crates/tenferro-gpu/tests/integration/native_permutation_plan.rs`: host-plan classification and validation.
- `docs/gpu-design.md`: native permutation architecture and runtime policy.
- `docs/development-log.md`: issue chronology, commands, revisions, devices, and results.

### `tenferro-benchmark`

- `scripts/benchmark_layout.py`: accept the `mac-gpu` target profile.
- `benchmarks/gpu/permutation-mac.yaml`: F32 Metal permutation suite.
- `data/instances/gpu_permutation_mac_patterns.json`: deterministic Metal-scaled patterns.
- `src/bin/benchmark_gpu_permutation_webgpu.rs`: tenferro WebGPU and Metal-copy runner.
- `scripts/benchmark_gpu_permutation_metal.py`: PyTorch MPS and best-effort JAX Metal runner.
- `scripts/run_gpu_permutation_mac.sh`: sequential orchestration and environment capture.
- `scripts/format_gpu_permutation_results.py`: include profile-specific device/runtime metadata and unavailable reasons.
- `tests/test_mac_gpu_permutation_profile.sh`: profile, participants, schema, and no-CPU-fallback contract.
- `result/mac-gpu/gpu/permutation.md`: latest human-readable Metal report.
- `data/results/mac-gpu/gpu/permutation/<timestamp>/`: immutable raw baseline and final runs.

### `strided-rs`

- `strided-perm/src/fuse.rs`: validated bilateral fusion plan.
- `strided-perm/src/lib.rs`: public exports.
- `strided-perm/tests/bilateral_fusion_plan.rs`: public API behavior and errors.

## Task 1: Expose Existing Structural Kernels on WebGPU

**Files:**
- Modify: `crates/tenferro-gpu/src/lib.rs`
- Modify: `crates/tenferro-gpu/src/webgpu/mod.rs`
- Create: `crates/tenferro-gpu/src/webgpu/structural.rs`
- Modify: `crates/tenferro-gpu/tests/integration.rs`
- Create: `crates/tenferro-gpu/tests/integration/webgpu_structural_runtime.rs`

**Interfaces:**
- Consumes: existing `structural::transpose_kernel` and `structural::view_to_contiguous_kernel` unchanged.
- Produces: `WebGpuBackend: TensorStructural + TensorViewCanonicalization` for supported WebGPU dtypes, using only the resident WebGPU allocation.

- [ ] **Step 1: Add failing WebGPU runtime tests**

Add tests that upload `F32` tensors, call `transpose(&input, &[1, 0])` and `to_contiguous_read` on a noncompact `TensorRead`, download only after the operation, and assert:

```rust
assert_eq!(backend.to_vec_f32(&transposed)?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
assert_eq!(backend.to_vec_f32(&materialized)?, vec![1.0, 3.0, 5.0]);
assert_eq!(transposed.placement(), input.placement());
```

Include zero-size, size-one, invalid permutation, and foreign-runtime cases. Gate runtime tests with the existing WebGPU availability helper so absence is a reported skip, not a pass through CPU.

- [ ] **Step 2: Verify the tests fail on the unsupported route**

Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer \
  --test integration webgpu_structural_runtime -- --nocapture
```

Expected: the valid transpose/materialization cases fail with the current explicit WebGPU unsupported error.

- [ ] **Step 3: Compile structural kernels for WebGPU and add focused launch glue**

Change the kernel module gate to:

```rust
#[cfg(any(feature = "cuda", feature = "webgpu"))]
mod kernels;
```

In `webgpu/structural.rs`, add checked helpers that:

```rust
fn transpose_typed<T: CubeElement + CubePrimitive>(
    backend: &WebGpuBackend,
    input: &Tensor,
    permutation: &[usize],
) -> crate::Result<Tensor>;

fn to_contiguous_view_typed<T: CubeElement + CubePrimitive>(
    backend: &WebGpuBackend,
    read: TensorRead<'_>,
) -> crate::Result<Tensor>;
```

Validate permutation and view bounds on the host, resolve the existing resident buffer with `ensure_resident_on_runtime`, allocate with `alloc_output`, and launch the unchanged kernels with `cube_count_for_len`, `cube_dim_1d`, and `comptime_sequence`. Dispatch `F32`, `I32`, and `Bool` through their existing storage representations; return an explicit unsupported error for WebGPU-incompatible element representations.

- [ ] **Step 4: Implement the structural trait methods**

Make `WebGpuBackend::transpose`, `to_contiguous_read`, and the canonicalization route delegate to the typed helpers. Preserve tensor dtype, logical shape, placement, runtime identity, and out-of-place semantics. Reject overlapping writes and foreign provider/runtime allocations before launch.

- [ ] **Step 5: Run focused and simultaneous-feature checks**

Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer \
  --test integration webgpu_structural_runtime -- --nocapture
cargo check -p tenferro-gpu --no-default-features --features cuda,webgpu,cpu-faer
```

Expected: focused tests pass; both runtime feature sets compile together.

- [ ] **Step 6: Prove the entry-gate restriction and commit**

Run:

```bash
git diff 7d2095ef -- crates/tenferro-gpu/src/kernels/structural.rs
```

Expected: no output.

Commit:

```bash
git add crates/tenferro-gpu/src/lib.rs crates/tenferro-gpu/src/webgpu \
  crates/tenferro-gpu/tests/integration.rs \
  crates/tenferro-gpu/tests/integration/webgpu_structural_runtime.rs
git commit -m "feat: expose native structural kernels on WebGPU"
```

## Task 2: Add and Run the `mac-gpu` Entry-Gate Profile

**Files:**
- Create an isolated `tenferro-benchmark` worktree from `origin/main`
- Modify and create the benchmark files listed in the file map

**Interfaces:**
- Consumes: the exact tenferro Task 1 commit SHA.
- Produces: schema-valid raw records and `result/mac-gpu/gpu/permutation.md`, committed before any kernel algorithm change.

- [ ] **Step 1: Create and index the benchmark worktree**

Run:

```bash
git check-ignore -q .worktrees
git worktree add .worktrees/issue-1507-mac-gpu -b codex/issue-1507-mac-gpu origin/main
cd .worktrees/issue-1507-mac-gpu
codegraph init
```

Expected: the new branch is based on the current benchmark `origin/main`, and `.codegraph/` is created locally.

- [ ] **Step 2: Add failing profile contract tests**

The shell test must assert:

```bash
python3.12 scripts/benchmark_layout.py validate-profile mac-gpu
rg -q '"dtype": "f32"' data/instances/gpu_permutation_mac_patterns.json
rg -q 'pytorch-mps' benchmarks/gpu/permutation-mac.yaml
rg -q 'jax-metal' benchmarks/gpu/permutation-mac.yaml
rg -q 'memcpy-metal-d2d' benchmarks/gpu/permutation-mac.yaml
rg -q 'not_configured' scripts/benchmark_gpu_permutation_metal.py
```

It must also inject a CPU-only JAX stub and assert the runner emits `status: not_configured` with a reason naming the non-Metal backend.

- [ ] **Step 3: Verify the profile test fails**

Run:

```bash
bash tests/test_mac_gpu_permutation_profile.sh
```

Expected: failure because `mac-gpu` and its runners are absent.

- [ ] **Step 4: Add the profile and deterministic patterns**

Extend target-profile validation from:

```python
r"(mac-cpu|amd-cpu|nvidia-gpu)"
```

to:

```python
r"(mac-cpu|mac-gpu|amd-cpu|nvidia-gpu)"
```

Define the F32 patterns `device-copy`, `transpose-2d`, two distinct 3D permutations, rotation, high-rank reverse, high-rank cyclic, TN scattered-to-column-major, and TN contiguous-collapse. Each record contains explicit shape, permutation, source strides, destination order, deterministic index seed, read bytes, and write bytes.

- [ ] **Step 5: Implement the native runners and timing contract**

The Rust runner must warm up, verify before timing, launch without allocation in the timed loop when the case contract permits, synchronize with the CubeCL WebGPU client, and emit JSONL fields:

```json
{
  "target_profile": "mac-gpu",
  "suite_id": "gpu/permutation",
  "dtype": "f32",
  "device": "Apple M5 Max",
  "runtime": "wgpu/Metal",
  "synchronization": "CubeCL client sync",
  "allocation": "outside timed region",
  "median_ms": 0.0,
  "p25_ms": 0.0,
  "p75_ms": 0.0,
  "effective_gbps": 0.0,
  "tenferro_revision": "<40-hex commit>"
}
```

The Python runner must use `torch.mps.synchronize()` around MPS timing. It must require `jax.default_backend()` or every selected JAX device platform to be Metal; otherwise emit `not_configured` and never execute a CPU sample.

- [ ] **Step 6: Implement sequential orchestration and formatting**

`scripts/run_gpu_permutation_mac.sh` runs in this order:

```text
tenferro-webgpu-transpose-baseline
tenferro-webgpu-to-contiguous
pytorch-mps
jax-metal
memcpy-metal-d2d
```

It captures `system_profiler SPHardwareDataType SPDisplaysDataType`, `sw_vers`, tool versions, git SHAs, suite YAML, instances JSON, stdout, stderr, and JSONL records under one timestamped raw directory. The formatter computes bandwidth from exact read-plus-write bytes and preserves unavailable-participant reasons.

- [ ] **Step 7: Run validation and the Metal baseline**

Run:

```bash
bash tests/test_mac_gpu_permutation_profile.sh
bash tests/test_permutation_result_schema.sh
bash scripts/run_gpu_permutation_mac.sh \
  --tenferro-worktree /Users/hiroshi/projects/tensor4all/tenferro-rs/.worktrees/issue-1507-metal-permutation
```

Expected: correctness passes for available participants; JAX either runs on Metal or is explicitly `not_configured`; no row uses CPU; the latest report and timestamped raw directory are created.

- [ ] **Step 8: Audit metadata and commit the immutable baseline**

Run:

```bash
rg -n "Apple M5 Max|wgpu/Metal|f32|tenferro_revision|not_configured" \
  result/mac-gpu/gpu/permutation.md data/results/mac-gpu/gpu/permutation
python3.12 scripts/validate_benchmark_suite.py \
  --suite benchmarks/gpu/permutation-mac.yaml \
  --results data/results/mac-gpu/gpu/permutation
```

Expected: the actual device, runtime, dtype, revision, and any unavailable reason appear; validation succeeds.

Commit:

```bash
git add benchmarks/gpu/permutation-mac.yaml data/instances/gpu_permutation_mac_patterns.json \
  scripts src/bin tests result/mac-gpu data/results/mac-gpu
git commit -m "bench: record mac-gpu permutation baseline"
```

## Task 3: Publish Validated Bilateral Fusion Planning

**Files:**
- Create an isolated `strided-rs` worktree from `origin/main`
- Modify: `strided-perm/src/fuse.rs`
- Modify: `strided-perm/src/lib.rs`
- Create: `strided-perm/tests/bilateral_fusion_plan.rs`

**Interfaces:**
- Produces:

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BilateralFusionPlan {
    pub dims: Vec<usize>,
    pub src_strides: Vec<isize>,
    pub dst_strides: Vec<isize>,
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum FusionPlanError {
    #[error("metadata lengths differ: dims={dims}, src_strides={src_strides}, dst_strides={dst_strides}")]
    LengthMismatch { dims: usize, src_strides: usize, dst_strides: usize },
    #[error("fused dimension product overflows usize")]
    DimensionOverflow,
}

pub fn plan_bilateral_fusion(
    dims: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
) -> Result<BilateralFusionPlan, FusionPlanError>;
```

- [ ] **Step 1: Create and index the strided worktree**

Run:

```bash
git check-ignore -q .worktrees
git worktree add .worktrees/issue-1507-fusion-planner \
  -b codex/issue-1507-fusion-planner origin/main
cd .worktrees/issue-1507-fusion-planner
codegraph init
```

- [ ] **Step 2: Add public API tests**

Test exact outputs for empty/scalar metadata, size-one removal, identity collapse, partial fusion, non-fusion, 2D transpose, negative strides, rank-24 TN collapse, mismatched lengths, and `usize::MAX * 2` overflow. Assert errors by enum variant, not message text.

- [ ] **Step 3: Verify tests fail before exports exist**

Run:

```bash
cargo test -p strided-perm --test bilateral_fusion_plan
```

Expected: unresolved imports for the new public types/function.

- [ ] **Step 4: Implement validated planning and keep HPTT on the same code**

Filter axes with dimension one, reject dimensions that cannot be represented as
`isize`, then fuse adjacent axes only when:

```rust
let current_dim = isize::try_from(current_dim)
    .map_err(|_| FusionPlanError::DimensionOverflow)?;
let src_contiguous = current_src
    .checked_mul(current_dim)
    .is_some_and(|expected| next_src == expected);
let dst_contiguous = current_dst
    .checked_mul(current_dim)
    .is_some_and(|expected| next_dst == expected);
```

Use `checked_mul` for fused dimensions. Make the existing HPTT planner consume `plan_bilateral_fusion(...).expect("HPTT metadata is prevalidated")` so there is one algorithm owner. Preserve the existing Strided.jl/HPTT provenance comment.

- [ ] **Step 5: Run package tests, docs, and commit**

Run:

```bash
cargo test -p strided-perm
cargo test -p strided-perm --doc
cargo fmt --check
```

Expected: all pass.

Commit:

```bash
git add strided-perm/src/fuse.rs strided-perm/src/lib.rs \
  strided-perm/tests/bilateral_fusion_plan.rs
git commit -m "feat: expose bilateral fusion planning"
```

## Task 4: Add the Native Materialization Planner

**Files:**
- Modify: `Cargo.toml` or the workspace dependency file that pins `strided-perm`
- Create: `crates/tenferro-gpu/src/native_permutation.rs`
- Modify: `crates/tenferro-gpu/src/lib.rs`
- Modify: `crates/tenferro-gpu/tests/integration.rs`
- Create: `crates/tenferro-gpu/tests/integration/native_permutation_plan.rs`

**Interfaces:**
- Consumes: Task 3 `plan_bilateral_fusion`.
- Produces:

```rust
pub(crate) enum NativePermutationKind {
    LinearCopy,
    GenericStrided,
    TiledTranspose,
}

pub(crate) struct NativePermutationPlan {
    pub kind: NativePermutationKind,
    pub dims: Vec<usize>,
    pub src_strides: Vec<isize>,
    pub dst_strides: Vec<isize>,
    pub src_offset: isize,
    pub len: usize,
}
```

- [ ] **Step 1: Add failing planner tests**

Construct plans for identity, 2D transpose, partial fusion, rank-24 collapse, negative stride, zero-size, invalid permutation, length mismatch, product overflow, source range violation, destination overlap, and source/destination allocation overlap. Assert classification and fused metadata exactly.

- [ ] **Step 2: Point the workspace pin at the reviewed strided commit**

Update only the `strided-perm` revision to the exact Task 3 commit and run:

```bash
cargo update -p strided-perm
```

Expected: `Cargo.lock` resolves the exact reviewed revision.

- [ ] **Step 3: Implement validation and classification**

`NativePermutationPlan::new` validates metadata and allocations once, calls `plan_bilateral_fusion`, then classifies:

```rust
if len == 0 || fused_rank <= 1 && src_is_contiguous && dst_is_contiguous {
    NativePermutationKind::LinearCopy
} else if fused_rank == 2 && stride_one_axes_differ && affine_tile_axes {
    NativePermutationKind::TiledTranspose
} else {
    NativePermutationKind::GenericStrided
}
```

Keep tile selection out of this task; `TiledTranspose` identifies eligibility only.

- [ ] **Step 4: Run planner tests and feature checks**

Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer \
  --test integration native_permutation_plan
cargo check -p tenferro-gpu --no-default-features --features cuda,webgpu,cpu-faer
```

Expected: all planner tests pass and both runtime features compile.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock crates/tenferro-gpu/src/lib.rs \
  crates/tenferro-gpu/src/native_permutation.rs \
  crates/tenferro-gpu/tests/integration.rs \
  crates/tenferro-gpu/tests/integration/native_permutation_plan.rs
git commit -m "feat: plan native permutation launches"
```

## Task 5: Unify Transpose and View Materialization

**Files:**
- Modify: `crates/tenferro-gpu/src/kernels/structural.rs`
- Modify: `crates/tenferro-gpu/src/webgpu/structural.rs`
- Modify: CUDA native fallback launch code under `crates/tenferro-gpu/src/cuda/`
- Modify: WebGPU structural and native planner integration tests

**Interfaces:**
- Consumes: `NativePermutationPlan`.
- Produces: one `materialize_strided_kernel` used by native transpose and `to_contiguous`.

- [ ] **Step 1: Add route-sharing and fused-rank tests**

Add a test-only launch trace that records `NativePermutationKind` and fused rank. Assert transpose and an equivalent permuted view yield identical plan metadata. Assert the rank-24 contiguous TN pattern launches with fused rank at most one and high-rank reverse remains correct.

- [ ] **Step 2: Verify the route-sharing test fails**

Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer \
  --test integration native_materialization_route
```

Expected: failure because transpose still launches `transpose_kernel`.

- [ ] **Step 3: Replace the generic view kernel with fused bilateral metadata**

Define:

```rust
#[cube(launch_unchecked)]
pub fn materialize_strided_kernel<E: CubePrimitive>(
    dst: &mut Array<E>,
    src: &Array<E>,
    #[comptime] dims: Sequence<usize>,
    #[comptime] src_strides: Sequence<i64>,
    #[comptime] dst_strides: Sequence<i64>,
    src_offset: i64,
    dst_offset: i64,
    #[comptime] rank: usize,
)
```

Decode coordinates over `rank` fused axes once, accumulate both affine offsets, then copy one element. The host converts validated `isize` metadata to CubeCL `i64` compile-time sequences.

- [ ] **Step 4: Route native transpose through materialization**

Build transpose source strides by permuting the input's logical strides into output-axis order, use contiguous output strides, construct the common plan, and launch `LinearCopy` or `GenericStrided`. Remove calls to `transpose_kernel`; delete that kernel only after all references are gone. Leave the cuTENSOR dispatch unchanged.

- [ ] **Step 5: Run cross-dtype correctness and commit**

Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer \
  --test integration webgpu_structural_runtime -- --nocapture
cargo test -p tenferro-gpu --no-default-features --features cuda,webgpu,cpu-faer \
  native_materialization
cargo check -p tenferro-gpu --no-default-features --features cuda,webgpu,cpu-faer
```

Expected: scalar, zero-size, size-one, noncompact, negative-stride, high-rank, and supported dtype cases pass.

Commit:

```bash
git add crates/tenferro-gpu/src crates/tenferro-gpu/tests
git commit -m "feat: unify native permutation materialization"
```

## Task 6: Add Compile-Time Tiled Transpose

**Files:**
- Modify: `crates/tenferro-gpu/src/kernels/structural.rs`
- Modify: `crates/tenferro-gpu/src/native_permutation.rs`
- Modify: WebGPU/CUDA native launch wrappers
- Create: `crates/tenferro-gpu/tests/integration/tiled_transpose_runtime.rs`

**Interfaces:**
- Produces:

```rust
pub(crate) struct TileConfig {
    pub width: u32,
    pub height: u32,
    pub padding: u32,
    pub vector_width: u32,
}

pub(crate) const TILE_CANDIDATES: &[TileConfig];
```

- [ ] **Step 1: Add failing boundary and fallback tests**

Force candidate selection in tests and cover exact-multiple tiles, one-element edges, both-axis partial tiles, rectangular shapes, shared-memory limit below every candidate, and non-tileable strides. Compare every result to a host permutation and assert the selected/fallback kind.

- [ ] **Step 2: Verify the tiled tests fail**

Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer \
  --test integration tiled_transpose_runtime -- --nocapture
```

Expected: failure because no tiled specialization exists.

- [ ] **Step 3: Implement candidate validation**

Use a small set such as `(32, 8, 1, 1)`, `(16, 16, 1, 1)`, and `(16, 8, 1, 4)`. Compute:

```rust
let shared_bytes = height
    .checked_mul(width + padding)
    .and_then(|n| n.checked_mul(element_size))
    .ok_or(TileSelectionError::SharedMemoryOverflow)?;
```

Select only candidates within the runtime's `max_shared_memory_size`; otherwise return `GenericStrided`.

- [ ] **Step 4: Implement the CubeCL shared-memory kernel**

Define the kernel with `#[comptime] tile_width`, `tile_height`, `padding`, and `vector_width`. Allocate `SharedMemory::new_aligned(tile_width * (tile_height + padding), alignment)`, cooperatively load coalesced source elements, call `sync_cube()`, and cooperatively write coalesced destination elements. Guard source and destination coordinates independently for boundary tiles.

- [ ] **Step 5: Run Metal correctness, shared-feature checks, and commit**

Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer \
  --test integration tiled_transpose_runtime -- --nocapture
cargo check -p tenferro-gpu --no-default-features --features cuda,webgpu,cpu-faer
cargo fmt --check
```

Expected: boundary and fallback tests pass; simultaneous features compile.

Commit:

```bash
git add crates/tenferro-gpu/src crates/tenferro-gpu/tests
git commit -m "perf: tile native CubeCL transpose"
```

## Task 7: Verify Both Native Runtime Paths on Linux A100

**Files:**
- Create: `docs/performance/issue-1507-linux-a100.md`
- Modify: `docs/development-log.md`

**Interfaces:**
- Consumes: the Task 6 tenferro revision.
- Produces: correctness and diagnostic performance evidence for CUDA and wgpu/Vulkan using the same kernel source.

- [ ] **Step 1: Resolve the approved A100 execution environment**

Use existing repository CI, documented SSH configuration, or the project's benchmark host mechanism. Record GPU model, driver, CUDA toolkit, Vulkan adapter, CubeCL revision, Rust version, and exact tenferro commit. Do not use a different GPU while labeling it A100.

- [ ] **Step 2: Run CUDA native-fallback correctness**

Run the package's native-fallback test selector with:

```bash
cargo test -p tenferro-gpu --no-default-features --features cuda,cpu-faer \
  native_materialization -- --nocapture
```

Expected: supported native fallback dtype/layout cases pass; cuTENSOR-supported production cases remain routed to cuTENSOR.

- [ ] **Step 3: Run wgpu/Vulkan correctness**

Select the Vulkan adapter explicitly through the repository/CubeCL environment contract and run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer \
  --test integration webgpu_structural_runtime -- --nocapture
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer \
  --test integration tiled_transpose_runtime -- --nocapture
```

Expected: the adapter is reported as Vulkan on A100 and all structural cases pass.

- [ ] **Step 4: Record diagnostic results and commit**

Document commands, raw logs, environment metadata, correctness status, and diagnostic medians without using them as the Metal acceptance baseline.

Commit:

```bash
git add docs/performance/issue-1507-linux-a100.md docs/development-log.md
git commit -m "docs: record A100 native permutation verification"
```

## Task 8: Sweep Tiles and Close Reference Gaps on M5 Max

**Files:**
- Modify benchmark raw/latest results
- Modify CubeCL planner/kernel files only if an evidence-backed candidate or reference-path improvement is needed
- Modify provenance comments at the moment third-party source influences code

**Interfaces:**
- Consumes: immutable Task 2 baseline and Task 6 candidates.
- Produces: chosen compile-time tile candidates, final Metal profile, regression analysis, and reference-gap investigations.

- [ ] **Step 1: Run the complete tile sweep**

For every candidate and the generic path, run all tile-eligible Metal patterns with identical dtype, sizes, allocation policy, synchronization, warmup, and sample count. Store raw samples and report median/p25/p75/effective bandwidth.

- [ ] **Step 2: Select candidates without overfitting**

Choose the smallest candidate set that wins across pattern classes, respects the measured Metal shared-memory limit, and introduces no row with a 20% or larger regression. If a candidate regresses a comparable row, retain generic fallback or remove that candidate and rerun.

- [ ] **Step 3: Rerun the complete `mac-gpu` profile**

Run:

```bash
bash scripts/run_gpu_permutation_mac.sh \
  --tenferro-worktree /Users/hiroshi/projects/tensor4all/tenferro-rs/.worktrees/issue-1507-metal-permutation \
  --label final
```

Expected: every available participant passes correctness and every tenferro row has a baseline ratio.

- [ ] **Step 4: Identify extreme comparable gaps**

Flag a row when:

```text
tenferro_median / reference_median > 2.0
```

or:

```text
reference_gbps / copy_ceiling_gbps >= 0.70
and tenferro_gbps / copy_ceiling_gbps < 0.35
```

Exclude unavailable participants and rows with mismatched dtype, pattern, timing scope, allocation policy, or device.

- [ ] **Step 5: Inspect and apply feasible reference techniques**

For every flagged PyTorch/JAX row, locate the exact dispatch and generated/native kernel using official source repositories and runtime traces. Compare collapse, copy specialization, launch geometry, vectorization, coalescing, tiling, padding, and synchronization. Apply only runtime-generic improvements that preserve CubeCL architecture, add a provenance comment naming project/file/function and relationship, write a failing regression/performance-classification test first, then rerun the full correctness and Metal profile.

- [ ] **Step 6: Commit final benchmark evidence**

Commit raw samples, environment metadata, tile sweep, latest report, regression table, and reference-gap notes:

```bash
git add data/results/mac-gpu result/mac-gpu docs
git commit -m "bench: record optimized Metal permutation results"
```

## Task 9: Final Documentation and Repository Gates

**Files:**
- Modify: `docs/gpu-design.md`
- Modify: `docs/development-log.md`
- Modify: issue-specific performance documents

**Interfaces:**
- Produces: review-ready branches in all three repositories with exact cross-repository revision references.

- [ ] **Step 1: Update architecture and work log**

Document the profile-first order, shared `strided-perm` planner, unified native route, compile-time tile parameters, validation/fallback policy, unchanged cuTENSOR route, Linux A100 CUDA+wgpu/Vulkan development validation, Metal acceptance baseline, and the approved final M5 bounded tile sweep in place of the issue's originally named M4 sweep.

- [ ] **Step 2: Run focused and broad tenferro verification**

Run:

```bash
cargo test -p tenferro-gpu --no-default-features --features webgpu,cpu-faer
cargo check -p tenferro-gpu --no-default-features --features cuda,webgpu,cpu-faer
cargo fmt --check
cargo clippy -p tenferro-gpu --no-default-features --features webgpu,cpu-faer -- -D warnings
PATH=/tmp/tenferro-1507-python-shim:/usr/bin:/bin:/usr/sbin:/sbin \
  bash scripts/check-pr-fast.sh
```

Expected: all commands pass. The shim's `python3` resolves to `/opt/homebrew/opt/python@3.12/bin/python3.12`.

- [ ] **Step 3: Run strided and benchmark gates**

Run:

```bash
cargo test -p strided-perm
cargo fmt --check
bash tests/test_mac_gpu_permutation_profile.sh
bash tests/test_permutation_result_schema.sh
python3.12 scripts/validate_benchmark_suite.py \
  --suite benchmarks/gpu/permutation-mac.yaml \
  --results data/results/mac-gpu/gpu/permutation
```

Expected: all pass.

- [ ] **Step 4: Audit issue completion criteria**

Verify from git history and artifacts that the baseline commit predates every kernel algorithm commit; both native runtime paths pass; all comparable Metal regressions are below 20% or explained; every extreme reference gap has a recorded investigation; provenance is complete; no hidden CPU fallback exists; and exact cross-repository SHAs are recorded.

- [ ] **Step 5: Commit final documentation**

```bash
git add docs
git commit -m "docs: complete native permutation optimization record"
```

- [ ] **Step 6: Review branch state**

Run in each worktree:

```bash
git status --short
git log --oneline origin/main..HEAD
```

Expected: clean worktrees, coherent commit series, and no unrelated user changes.
