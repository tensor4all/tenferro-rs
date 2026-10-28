# HANDOFF: tenferro-rs #2208 (CUDA backend on t4a-cuda-runtime), 2026-10-23

Stopped for a power outage (maintainer request). Not committed to git; this text is posted as a
comment on #2208 and the same file lives untracked in the working tree as `HANDOFF-2208.md`.

Everything described below is **merged on `origin/main`** unless a section says otherwise. There is no
dirty state, no unpushed branch with work in it, and no open PR from this run.

## Scope of the adoption

#2208 replaces the CubeCL CUDA backend with a path over `t4a-cuda-runtime` (launch/streams/retention)
and `t4a-cuda-kernels` (ABI rev 10 kernel catalogue), behind the private feature `cuda-next`, in seven
units: U0 (dependency ABI work, in the sibling repos) → U1 seam → U2 dispatch → U3 fusion/prepare →
U4 vendor libraries → U5 tensor surface + eager/traced/AD → U6 delete CubeCL → U7 gates and the
end-to-end comparison.

## Where things are

- Repos: `~/tensor4all/tenferro-rs` (this one), `~/tensor4all/t4a-gpu-backend`, `~/tensor4all/t4a-cuda-kernels`.
- Pins in `crates/tenferro-gpu/Cargo.toml` (both publish-disabled, pinned by rev):
  - `t4a-cuda-runtime` `0a101bf19207e0abb2ea169721f63f50af8669b0` (t4a-gpu-backend#8)
  - `t4a-cuda-kernels` `5019c713c14c1f5d59ea2cf03bbfff08c40f167e` (t4a-cuda-kernels#17)
- Feature: `cuda-next = ["cuda", "dep:t4a-cuda-runtime", "dep:t4a-cuda-kernels"]`. Everything new is
  `#[doc(hidden)]` and private until U6 flips the default.
- `crates/tenferro-gpu/src/cuda_next/` (the new path):
  `runtime.rs` (CudaRuntime, submit guard, raw scope), `storage.rs` (CudaStorage), `transfer.rs`
  (upload/download, Bool included), `pending.rs`, `pinned.rs`, `publication.rs` (SourcePublication),
  `engine.rs` (CudaEngine), `launch.rs` (KernelModules, layout args, device_buffer), `dispatch.rs`
  (per-family dispatch), `fusion.rs` (PreparedFusion + cold/warm counters), `vendor.rs` (cuBLAS),
  `cutensor.rs` (cuTENSOR with shape-keyed plans), `structural.rs` (the tensor surface:
  `TensorStructural` + `TensorReduction` + `TensorElementwise`, 1322 lines).
- `crates/tenferro-gpu/src/gpu/` — the backend-neutral seam (`GpuError`, `StorageIdentity`/`Buffer`,
  `Program`/`OpDescriptor`, `KernelSource`, `VendorOps`).
- Design and worklogs: `docs/design/cuda-seam-u1-2208.md`, `docs/worklogs/2026-10-23-cuda-{seam-u1,
  dispatch-u2,fusion-u3,reduce-u4,cublas-u4,cutensor-u4,structural-u5}.md`.
- GPU test targets (all `required-features = ["cuda-next"]`): `seam_storage`, `seam_transfers`,
  `seam_publication`, `seam_staging`, `seam_copy`, `seam_launch`, `seam_dispatch`, `seam_reduce`,
  `seam_fusion`, `seam_vendor`, `seam_contraction`, `seam_structural` (14 tests — the tensor surface).

## What landed (all merged)

| PR | merge commit | content |
|---|---|---|
| t4a-cuda-kernels#15 | `6237c15` | ABI rev 9 dynamic rank: 31 `_any` twins, `LayoutRef`/`WideLayout`, `RankVariant`; `scripts/sass-diff.sh` proves rev 1–8 symbols bit-identical |
| t4a-cuda-kernels#16 | `de7241c` | ABI rev 10: `broadcast_in_dim`, `concatenate_copy`; `reverse` folded into strided movement, `scale_in_place` into `binary` MUL; 173-row pinned inventory test |
| t4a-cuda-kernels#17 | `5019c71` | `t4a_cuda_kernels::source`: ABI header + 11 translation units + `module_of`/`compose` for NVRTC |
| t4a-gpu-backend#6 | `9f75730` | by-value POD kernel params, raw scope, retirement accounting |
| t4a-gpu-backend#7 | `71768c3` | `Runtime::checkpoint()` → `Completion` for arbitrary producer work |
| t4a-gpu-backend#8 | `0a101bf` | cudarc pinned `=0.19.10` |
| tenferro-rs#2250 | `52353a89` | **U1** seam + `cuda_next` skeleton, five ported suites |
| #2251 / #2252 | `b5e8dc99` / `f59d10e3c` | **U2** KernelModules (NVRTC per module, bind on demand) and Dispatcher (unary, binary, compare, select, clamp, convert, gather, scatter_add) |
| #2253 | `e5142aa43` | **U3** fused regions, artifact preparation keyed by `ArtifactKey`, cold/warm counters, zero NVRTC on warm |
| #2254 | `2515ca33a` | **U4a** reductions: `reduce` (explicit strategy), `reduce_auto` (crate split policy, two passes on one stream) |
| #2255 | `9f84aa8d0` | **U4b** cuBLAS: handle per runtime on its own stream, plain + strided-batched GEMM, F32/F64/C32/C64 |
| #2256 | `f379b3951` | **U4c** cuTENSOR contraction with shape-keyed plans (cold/warm counters) |
| #2257 | `1ed0cad10` | **U5a** `TensorStructural`: materialize, copy, cast through the trait |
| #2258 | `f70fe6537` | **U5b** transposed, reshaped, broadcast reads |
| #2259 | `74d138135` | **U5c** diagonal extract/embed, tril/triu → `TensorStructural` complete (F32/F64) |
| #2260 | `a24fccea2` | **U5d** `TensorReduction` complete |
| #2261 | `ffac68bcb` | **U5e** `TensorElementwise`: arithmetic, unary, max/min, clamp |
| #2262 | `136f137dc` | **U5f** `compare_read`/`select_read` + `Bool` in the transfer route |
| #2263 | `281300f09` | **U5g** `rem`/`rem_read` (checked flag read back), `elementwise_read_into` |

## Evidence and how to reproduce

Host: one A100 80GB PCIe, driver 580.126.09, CUDA toolkit 12.6, cuTENSOR 2.5.0 at
`/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2`.

```bash
cargo test -j 16 -p tenferro-gpu --features cuda-next \
  --lib --test seam_structural --test seam_contraction --test seam_vendor \
  --test seam_reduce --test seam_fusion --test seam_dispatch --test seam_launch
```

Every PR in the table above ran the repository-local gate
`scripts/check-pr-fast.sh --coverage-reviewed --test "…"`, `cargo clippy -j 16 -p tenferro-gpu
--features cuda-next --all-targets -- -D warnings`,
`python3 scripts/repository-rules-review.py --base origin/main --dry-run` and
`python3 scripts/check-public-error-docs.py --changed-from origin/main`, then spent the hosted
RunPod GPU gate exactly once on its finished state.

## Decisions in force

- The ABI stays **ceiling-free**: rev 9/10 add `_any` twins taking pointer geometry in caller-owned
  device memory; `T4A_CUDA_MAX_RANK` is unchanged; rev 1–8 symbols, signatures and structs are
  untouched (proved by sass-diff).
- Fold rather than duplicate: `reverse` → `materialize_strided` with negated strides; a permutation
  (`transpose_read`) is the same strided movement; `scale_in_place` → `binary` MUL.
- No dual-storage switch: the new path is private behind `cuda-next` until U6.
- Publication frontier is `Runtime::checkpoint()`; admission never synchronises.
- Ownership by allocation domain, not device ordinal.
- The dispatcher takes its plan only from the kernel crate's `plan_*` — no second validation source.
- NVRTC needs a real `sm_<major><minor>`; `compute_XX` fails (double atomics, sm_60+ intrinsics).
- cuBLAS/cuTENSOR handles are created inside the runtime's raw scope with the runtime's own stream;
  operand pointers come from buffers registered in the same scope, so vendor calls order with the
  surrounding kernels and cannot outlive their operands.
- Warm path does no compilation and no vendor planning (U3 counters + cuTENSOR plan cache).

## Gotchas discovered (each cost a failing test)

- `clamp` takes **per-element** `lower`/`upper`, not scalars.
- A rank-0 layout is `Layout::compact(&[])`; `&[0]` gives `IndexVectorMismatch`.
- `plan_extract_diagonal` and `plan_pad` require a **compact destination**.
- The reduction kernel is **keepdims**: destination rank = input rank, extent 1 on the reduced axis.
  Multi-axis reduction therefore squeezes the unit axes with one final copy, reducing largest-axis
  first.
- `Arg` has no signed 64-bit scalar: positive `long long` (`splits`, `chunk_len`) go as `Arg::U64`
  (bit-identical); `tril`/`triu`'s `k` **can be negative** and goes as raw `i64` bytes via
  `pod_arg(&k)`.
- `Arg::I64` and `Arg::bytes` do not exist as such — `pod_arg` builds `Arg::Bytes` for `repr(C)`
  mirrors (that is how `CGatherPlan`/`CAxisValues` travel).
- cuTENSOR 2.5 rejects a **null C descriptor** at plan creation *and* a **null C pointer** at
  execution: a contraction without an addend passes the destination as C with `beta = 0` (guaranteed
  never to read C).
- `CUTENSOR_COMPUTE_DESC_32F/64F` are exported **as data whose pointee is the descriptor**; using the
  symbol's address makes `cutensorCreateContraction` answer `CUTENSOR_STATUS_NOT_INITIALIZED`.
- This cuTENSOR header has `CUTENSOR_WORKSPACE_DEFAULT/MIN/MAX` only (no `_RECOMMENDED`), and no plain
  `libcutensor.so.N` lookup finds the installed layout — bind from the located file
  (`TENFERRO_CUTENSOR_PATH`, then the versioned subdirectory).
- `TensorValue::reshape_view` yields a **view** and `into_tensor` refuses a metadata-only view: an
  owned tensor with new metadata must be a **fresh allocation**.
- `Tensor::strides()` is not on the dtype-erased tensor; strides come from the typed payload
  (`as_typed::<T>().strides()`), which forces dtype dispatch before layout construction.
- The kernel's `sign` follows the **XLA** convention (`sign(0) = 0`); `f64::signum(0)` is 1.
- Boolean device tensors were unusable until the transfer route learned `Bool` — a comparison's
  result could be produced but never read back.
- Integer remainder uses `RemChecked`, whose divide-by-zero flag is a **device word**; it must be
  read back and turned into a typed error instead of a silent zero.

## Remaining work, in dependency order

### 1. U5 tail — `TensorIndexing` (windowing family first)

`crates/tenferro-gpu/src/cuda_next/dispatch.rs` has `gather` and `scatter_add` but **not** the
windowing trio. Add them (plan functions already exist in the kernel crate):
`plan_slice` (line ~3651), `plan_dynamic_slice` (~3694), `plan_pad` (~3732); entries
`Entry::Slice`/`DynamicSlice`/`Pad`, planners documented in
`~/tensor4all/t4a-cuda-kernels/src/lib.rs`.

Then implement `TensorIndexing` on `CudaEngine` (`gather`, `scatter`, `slice`, `dynamic_slice`,
`dynamic_update_slice`) the way `structural.rs` does it: resolve via `device_read`, allocate with
`alloc_tensor`, launch through `Dispatcher`, synchronise before returning.

### 2. U5 tail — `TensorDot`

`dot_general`/`dot_general_with_conj`/`dot_general_read_into_accum` can be built directly on
`cuda_next::Cutensor` (`Contraction`/`Operand` already take explicit modes, extents and strides, and
the plan cache is shape-keyed). Map the `DotGeneralConfig` batch/contract/free dims to cuTENSOR
modes; the `out`-accumulating form needs `beta = 1`, which `Contraction` currently hardcodes to 0 —
extend the request rather than adding a second path.

### 3. U5 core — the session wiring (this is the unlock)

Only `crates/tenferro-gpu/src/cubecl/engine.rs:346` implements `tenferro_runtime::ExecutionEngine`
today. The new engine needs the same: `impl ExecutionEngine for CudaEngine` with `native_engine`,
`placement`, extension-cache store, `prepare`/`run`/`Output` plumbing, plus `TensorBackendCapability`,
`Engine` (as in `cubecl/exec_session.rs`), and the session-shaped `elementwise_read_into` scalar
forms (currently refused: operands must share the destination's shape).

This is what lets `tenferro-linalg` (cuSOLVER) and `tenferro-fft` (cuFFT) move, which is the rest of
U4: both crates are written against `tenferro_gpu::cuda::{CudaEngine, CudaRuntime}` and the
`ExecutionEngine` surface (`crates/tenferro-fft/src/cuda/plan.rs`,
`crates/tenferro-linalg/src/gpu/ffi/cusolver.rs`). Until then those two vendor libraries cannot be
ported — the ordering is a consequence, not an omission.

### 4. U6 — delete CubeCL

- Remove `crates/tenferro-gpu/src/cubecl/**`, the `webgpu` feature and the macOS/Metal lane in CI.
- Drop the `t4a-cubecl`/`t4a-cubek`/`t4a-cubecl-*` pins from the workspace `Cargo.toml`
  (`Cargo.toml:110-113`) and the patch section notes at `Cargo.toml:106-107`.
- Make `cuda-next` the only CUDA path, delete the `#[doc(hidden)]` markers and give the seam real
  doctests (the pre-cutover exemption disappears here, and `check-public-error-docs.py` plus the
  documentation-policy rules will demand examples).
- Keep `scripts/engine-entry-allowlist.json`'s `CudaRuntime::with_raw_locked` entry honest — it is the
  single raw-scope entry of the new path.

### 5. U7 — gates and the end-to-end comparison

- Repository gates for the whole workspace with CUDA on the A100, plus the RunPod gate once on the
  finished PR.
- The end-to-end comparison that decides whether the accepted performance residuals are user-visible:
  small-work overhead with an explicit 1-thread/1-worker configuration (never the machine default),
  comparing old and new paths on the same benchmark bodies; one measurement at a time, one L3 domain
  per cell, observing side pinned outside that domain.

## Open questions for the maintainer

1. **`gpu::Buffer` handle question.** The seam's `Buffer` carries an allocation identity but no
   backend handle, so `VendorOps::serve(op, &[&Buffer])` cannot be implemented against the current
   shape (its placeholder signature was deleted in #2255 rather than kept alive unused). Two options:
   give `Buffer` an opaque backend handle (a private enum behind it, as the U1 design anticipated) or
   keep vendor adapters typed per backend (`cublas`, `Cutensor`) and drop the trait. The tensor
   surface works today with the typed adapters, so this is a design call, not a blocker.
2. **Dtype scope.** The structural slice serves F32/F64 (plus Bool where a kernel demands it), and the
   transfer route serves all preset scalars. Widening `TensorStructural`/`TensorElementwise` to I32/I64/C32/C64
   is mechanical but should be decided together with the U6 doctest work.
3. **Accepted-but-recorded residuals** (from `docs/worklogs/2026-10-23-dynamic-rank.md` and the U1
   worklog): gather/scatter fast-vs-`_any` equality below rank 8, batched linalg-aux compared at ranks
   2–3, device discovery still uses the metadata probe, RRQR's whole-sequence harness anomaly (the
   sections are separate `#[test]`s). U7 should confirm these stay acceptable or file them.

## Traps

- Do not push follow-up commits to an open PR; produce one finished state and spend the RunPod GPU
  gate once.
- The cpu-faer lane flakes on
  `tenferro-runtime::eager_dot_plan_reuse::repeated_tiny_typed_matmul_keeps_only_the_one_time_plan_build`
  (allocation-count assertion, `first 1, warm 0, repeat 4`). It passed on `main` and on the adjacent PR
  runs, reproduces green locally, and re-running the failed job passed. Read the failing step before
  assuming infrastructure, but this one is environmental.
- `git add -A` in this worktree picks up untracked `HANDOFF-*.md`, `ext/*/Cargo.lock` and
  `samples/*/Cargo.lock`; stage explicit paths instead.
