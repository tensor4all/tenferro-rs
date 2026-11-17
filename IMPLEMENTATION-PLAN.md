# U4 slice 1 — `MetalContext` / `MetalEngine` foundation and the elementwise family

Tracking issue: tensor4all/tenferro-rs#2402.
Design: tensor4all-meta `8befe98`, `docs/design/metal-backend.md` §4, §5, §6, §8, §10 (U4).
Worktree: `~/projects/tensor4all/tenferro-rs-metal-u4` (branch `feat/2402-metal-u4`, rebased onto `origin/main` `a76f2f2ee`).
Host: macOS 26.7.2, Apple M5 Max, no NVIDIA device.

## 0. What already exists and what this unit adds

Merged and available to build on:

- U1 `83fe982df`: `BackendStorage::shared_host_slice` (`crates/tenferro-tensor/src/types.rs:974-981`,
  default `Unsupported`), the CPU admission of a shared allocation in
  `tenferro-cpu-basic/src/lib.rs:150,213`, `CpuEngine::with_allocation_domain`
  (`crates/tenferro-cpu/src/engine.rs:267`), `GpuBackendKind::Metal`.
- U2 `b8dee5e9` (t4a-gpu-backend): `t4a-metal-sys`, `t4a-metal-runtime`. Runtime API:
  `Runtime::open`, `alloc_zeroed`/`alloc_uninit`, `compile`, `bind`, `launch`, `commit -> Completion`,
  `synchronize`, `stats`; `Buffer::{len, allocation_id, domain_id, host_ptr}`;
  `probe()` with per-dtype MSL compile results; `Arg::Buffer/Pod`; `Completion::{poll,wait}`.
- U3 slice 1 (t4a-cuda-kernels, published as `0.3.0`; source base `e870458`): `t4a_cuda_kernels::metal` — `SIMD_WIDTH`,
  `THREADGROUP`, `ABI_REVISION`, `SYMBOLS` (fill_zero, copy_exact, add, sub, mul, div, unary),
  `DTYPES = [F32, C32]`, `UNARY_OPS`, `plan_fill_zero/plan_copy_exact/plan_elementwise/plan_unary`,
  `Plan`/`Dispatch`, `source::{ABI_HEADER, MODULES, module_of, compose}`.

Current behaviour of `tenferro-gpu`: `gpu` seam module and `cuda` module are both gated on
`feature = "cuda"` (`crates/tenferro-gpu/src/lib.rs:33-42`); there is no Metal code, no `metal`
feature, no Metal device discovery, and `docs/guides/devices-and-gpu.md:10-11,37` states Metal is
not provided.

This PR is **U4 slice 1**, the same grain as U1/U2/U3 slice 1: it makes the Metal engine exist
end to end for one family and refuses everything else with a typed error. The precedent inside
this repository is the CUDA engine's own first slice, which states exactly this contract
(`crates/tenferro-gpu/src/cuda/structural.rs:3-7`: "the materialization boundary plus the two
operations the movement and conversion kernels already serve ... Every other structural
operation answers a typed Unsupported until its kernel slice lands").

**Served in this PR**: probe/discovery, `MetalContext` (device + queue + shared allocation
domain + paired `CpuEngine`), in-domain allocation, `MetalStorage` host admission (`shared_host_slice`
waits for pending submissions), explicit `upload`/`download` with byte accounting, the flat
elementwise family the merged MSL serves, `to_contiguous`/`copy_read_into`, `cast` between
identical dtypes, and view-producing `transpose`/`reshape`/`broadcast` (metadata only, no copy).

**Refused in this PR with a typed error, never a fallback**: f64/Complex64 operands of any Metal
op (message names the paired `CpuEngine`), reductions, indexing, dot/contraction, analytic
(transcendental) family beyond the unary entries the catalogue carries, `tril`/`triu`/diagonals,
cross-dtype casts, and any length above `u32::MAX`.

Named follow-ups (recorded in the PR body, the worklog, and the issue):
- U4 slice 2: structural kernels (strided materialization, `tril`/`triu`, diagonals, f32↔c32 cast) —
  needs the U3 slice-3 MSL entries.
- U4 slice 3: reductions — MPS first (`MPSMatrixReduction`, the design's owner for reductions),
  MSL only where MPS cannot serve (`prod`).
- U5: MPS GEMM/einsum. U6: MSL fusion emitter.

## 1. Files and functions to change

### 1.0 Sibling change this slice needs: `Runtime: Send`

`t4a-metal-runtime`'s `Runtime` is `!Send`: `submission.rs:28` `Session` has no `unsafe impl Send`
while every other handle it owns does (`compile.rs:89`, `kernel.rs:182`, `memory.rs:30`,
`submission.rs:182`, `context.rs:44`). U4's domain must allocate from `&self`
(`SharedTensorAllocationDomain::allocate`, `crates/tenferro-tensor/src/types.rs:705`) while
`Runtime::alloc_zeroed` takes `&mut self` (`runtime.rs:223`), so the context needs
`Mutex<Runtime>` — and `Mutex<T>: Send + Sync` needs `T: Send`.

Change (t4a-gpu-backend, released as `t4a-metal-runtime` 0.1.2): `unsafe impl Send for
Runtime {}` with the argument the crate already uses for the other four handles — every object the
runtime owns is an Apple handle usable from any thread, and `&mut self` is what prevents concurrent
use of one command buffer — plus a `fn assert_send<T: Send>()` contract test. 0.1.1 was published
before the impl existed; 0.1.2 is the `Send` revision (`e695c24`) plus the version bump, and this PR
requires `=0.1.2` from the registry. A local `unsafe impl Send` on a wrapper in `tenferro-gpu`
was rejected: the thread-safety claim belongs to the crate that owns the handles.

### 1.1 Pins and features

`Cargo.toml` (`[workspace.dependencies]`, `:112-122`)

- Add `t4a-metal-runtime = { version = "=0.1.2" }`, the release whose sources contain
  `unsafe impl Send for Runtime`; 0.1.1 was published before that impl.
- Bump `t4a-cuda-kernels` from `0.1.2` to `=0.3.0`, the published release that contains
  `src/metal.rs`, and update the "ABI revision 13" comment to 14.
- `t4a-metal-sys` `0.1.0` comes in transitively through the Metal runtime.
- All three resolve from crates.io, so the branch pins no git revision and
  `scripts/git-pin-content-exceptions.toml` needs no entry: `scripts/check-git-pin-content.py`
  is green (0 warnings, no content drift) and `tenferro-gpu` can be republished with the same
  requirements.

`crates/tenferro-gpu/Cargo.toml`

- `metal = ["dep:t4a-metal-runtime", "dep:t4a-cuda-kernels"]`.
- `[[test]] name = "metal_engine"` with `required-features = ["metal"]`; `[[bench]] name = "metal_engine"`
  with `required-features = ["metal"]`, `harness = false`.
- `[package.metadata.docs.rs] features` unchanged (docs.rs has no Apple GPU; the module still
  documents itself).

`crates/tenferro-gpu/src/lib.rs`

- `#[cfg(any(feature = "cuda", feature = "metal"))] pub mod gpu;` (`:33-36`) — the seam is
  vendor-neutral (`gpu/mod.rs:1-9`), only its gate named CUDA.
- `#[cfg(feature = "metal")] pub mod metal;`.
- `#[cfg_attr(not(feature = "cuda"), allow(dead_code))] pub(crate) mod capability;` (`:39-42`) becomes
  `#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]`, and the
  crate-level `use tenferro_tensor::*;` gate widens the same way.
- The `#![deny(clippy::cast_ptr_alignment)]` comment block stays; the Metal module's one
  `from_raw_parts` goes through a byte pointer, so it does not need a new allowance.

### 1.2 `crates/tenferro-gpu/src/metal/` (new)

- `mod.rs` — module contract (what is served/refused, the wait rule, the no-link rule), re-exports,
  `pub struct MetalBackend;` implementing `crate::gpu::Backend { type Allocation = Arc<t4a_metal_runtime::Buffer>; }`.
- `device.rs`
  - `pub struct MetalDeviceInfo { name, unified_memory, fp32, bf16, fp16, mps, mps_graph }` built from
    `t4a_metal_runtime::probe()` / `DeviceInfo`.
  - `pub fn metal_devices() -> Result<Vec<MetalDeviceInfo>, MetalDeviceError>`; `MetalDeviceError`
    is a typed enum (`Unavailable { detail }`, `Probe { detail }`) reusing the runtime's message
    instead of panicking (design §6 "probe before constructing"; model: `cuda/device.rs:436`).
  - `pub fn metal_available() -> bool` — same `TENFERRO_REQUIRE_*` discipline as
    `cuda/mod.rs:44-58`, environment variable `TENFERRO_REQUIRE_METAL` (design §9 names it).
  - `pub fn metal_device_id() -> ...` — only one device exists; the id carries the device name.
- `memory.rs` — `MetalBuffer` (an `Arc<t4a_metal_runtime::Buffer>` plus its `PendingWrites`),
  `PendingWrites` (see `pending.rs`), and the checked `Bytes` conversions from element counts.
- `pending.rs` — `PendingTouches`: one `std::sync::Mutex<Option<Completion>>` holding the **last
  submission that touched the allocation, read or write**. One in-order command queue
  (`runtime.rs:121`, one `newCommandQueue` in `context.rs:69`) means the last submission dominates
  every earlier one, so `wait()` on it is exactly "every pending device operation on this allocation
  has finished" — which is what design §5 asks for on both sides: a CPU **read** waits for pending
  GPU writes, and a CPU **write** waits for pending GPU reads. Commit registers every allocation
  named by `Arg::Buffer` in that submission (inputs *and* outputs), so a read-only launch is
  registered too; missing that would let a `map_write` after a read-only launch race. Counters
  `waits`.
- `storage.rs` — `pub(crate) struct MetalStorage<T>` implementing `tenferro_tensor::BackendStorage<T>`:
  `backend_family() = "metal"`; `len`; `allocation_domain()`; `allocation_id()`
  (`AllocationId::from_backend_id(buffer.allocation_id())`); `shared_host_slice()` = `wait()` then
  `std::slice::from_raw_parts(host_ptr.cast(), len)` (`unsafe` with the shared-storage-mode
  argument: the pointer is the same address the GPU addresses and every submission that touched it
  has completed); `map_write()` = `wait()` then `HostWriteGuard` over `host_ptr` (`&mut [T]` under
  the ownership kernel's exclusive capability, design §4.3); `map_read()` = the same borrow as
  `shared_host_slice` behind `HostReadGuard`. `map_read`/`map_write` raise a `live_host_access`
  count for the guard's lifetime and clear it on drop.
- **Who may write an allocation while a host borrow is live.** A GPU submission that names an
  allocation with `live_host_access > 0` returns the typed `MappedForHost` error (design §5) and
  writes nothing. `shared_host_slice`'s plain borrow cannot raise that count — the U1 signature
  (`types.rs:974-981`) has no guard to drop — so slice 1 removes the window structurally instead:
  every GPU **write** goes to an allocation created inside the same engine call and returned only
  after `commit`, so no caller can hold a borrow of it yet, and the engine's only in-place write,
  `copy_read_into`, writes its destination on the **host** (`memcpy` into shared storage under the
  domain lock, no GPU launch). GPU **reads** concurrent with a host borrow are harmless. This is
  stated in `mod.rs`, and slice 2's in-place materialization kernel must revisit it.
- `domain.rs` — `pub struct MetalAllocationDomain` implementing
  `SharedTensorAllocationDomain` (`id`, `allocate`): allocates `alloc_zeroed(bytes)`, wraps it in
  `MetalStorage`, returns a `TypedTensor::from_buffer_col_major(..., StorageBuffer::Backend(...),
  Placement { memory_kind: MemoryKind::Managed, device: None, cpu_affinity: None })` (exact shape
  copied from the U1 fake provider,
  `crates/tenferro-cpu/src/tests/cpu_tests/managed_copy_boundaries.rs:150-174`). **Every dtype in
  `DType` is served here** — F32, F64, I32, I64, Bool, C32, C64
  (`crates/tenferro-tensor-core/src/lib.rs:121-136`) — because design §4.3 makes this domain the
  allocator for the paired CPU engine's outputs too: an f64 reduction on the CPU over a shared
  tensor must land in the same domain so the result chains back to Metal without an upload.
  Refusing f64 in the allocator would make the "f64 runs on the CPU engine over the same tensors"
  lane (design §9) impossible. The **engine**, not the allocator, refuses f64/C64 ops.
- `context.rs` — `pub struct MetalContext` (`Arc<ContextInner>`: `Mutex<Runtime>`, `Mutex<MetalCounters>`,
  domain id, `MetalAllocationDomain`, paired `CpuEngine`) with:
  - `open(config: MetalConfig) -> Result<Self, MetalError>`: probe, then `Runtime::open`
    (`Config::default()`), then `AllocationDomainId::fresh()`, then
    `CpuEngine::serial().with_allocation_domain(domain)`. A failed probe is a typed
    `MetalError::Unavailable`, never a panic.
  - `upload(&self, tensor: TensorRead<'_>) -> Result<Tensor>`: the source must be a host tensor
    (else typed `Unsupported` naming `download`/`to_contiguous`); allocate in-domain, `memcpy` the
    compact source into the shared buffer, count `upload_bytes`/`uploads`.
  - `download(&self, tensor: TensorRead<'_>) -> Result<Tensor>`: the source must be a
    shared-domain Metal tensor (else typed `Unsupported`), `wait()`, then copy into a fresh host
    tensor; count `download_bytes`/`downloads`.
  - `engine(&self) -> MetalEngine`, `cpu_engine(&self) -> &CpuEngine`, `allocation_domain()`,
    `device()`, `synchronize()`, `stats() -> MetalStats` (`allocations`, `live_bytes`,
    `upload_bytes`, `download_bytes`, `staged_reads`, `submissions`, `waits`, `artifact_entries`,
    `launches` from the runtime's own `Stats`).

**Concurrency shape: one lock, no second path.** `MetalContext` is `Clone` (an `Arc`), so the
`Runtime` lives behind `Mutex<Runtime>` and *every* runtime use — `alloc_zeroed`, `compile`, `bind`,
launch encoding, `commit`, `synchronize`, `stats` — takes that one lock. A domain `allocate`, an
upload, a download and an engine submission therefore serialise on it. `Completion::wait` needs
only `&Completion`, so waiting for pending work (`shared_host_slice`, `download`, and before a host
write) happens **without** holding the lock; only encoding and committing hold it. Lock order is
state it once and never nest it the other way.
- `transfer.rs` — the tensor-level form of the two copies above, so `TensorDeviceTransfer` and the
  context share one implementation.
- `engine.rs` — `pub struct MetalEngine` (a `MetalContext` clone + an op counter), `MetalEngineError`,
  and the trait impls:
  - `tenferro_tensor::TensorStructural`: `cast` (same dtype → `copy_exact` launch; cross dtype →
    typed `Unsupported` naming U4 slice 2), `transpose_read`/`reshape_read`/`broadcast_in_dim_read`
    (view constructors over the shared allocation, no launch, no copy), `extract_diagonal`/
    `embed_diagonal`/`tril`/`triu` → typed `Unsupported` naming U4 slice 2.
  - `tenferro_tensor::TensorElementwise`: `add_read`/`sub_read`/`mul_read`/`div_read`/`neg_read`/
    `conj_read`/`abs_read`/`sign_read` and `elementwise_read_into` for those, over f32 and
    Complex32, one `Plan` per call and one `commit` per call; `maximum`/`minimum`/`compare`/
    `select`/`clamp` → typed `Unsupported` (their MSL entries are not in `SYMBOLS`).
  - `tenferro_tensor::TensorDeviceTransfer` → `upload_host_tensor`/`download_to_host` (context).
  - `tenferro_tensor::TensorReduction`, `TensorIndexing`, `TensorDot`, `TensorAnalytic`,
    `TensorFusion` (empty impls where the trait has no required method; typed `Unsupported` for
    every required method) — each message names the unit that serves it (U4 slice 3 / U5 / later).
  - `SessionCachedDot`, `TensorBuffer`, `Engine` impls with the defaults.
  - `TensorBackendCapability`: `backend_id() = BackendId::Other("metal")` (a `BackendId::Metal`
    variant is a public-API decision, deliberately not taken here — see §3),
    `capabilities() = metal_capabilities()`.
- `capability.rs` in `metal/` — a `const` table written down (not derived), mirroring
  `crates/tenferro-gpu/src/capability.rs:68-216`, with one test that holds every `Native` entry to
  what the engine actually runs and every refusal to its typed error.
- `mod.rs` re-exports: `MetalContext`, `MetalEngine`, `MetalEngineError`, `MetalDeviceInfo`,
  `MetalDeviceError`, `metal_devices`, `metal_available`, `metal_capabilities`, `MetalBackend`.

Staging rule (one path, documented): an elementwise or `cast` operand that is not compact is
materialized first through `to_contiguous_read`, which in slice 1 copies **on the paired CPU
engine into a same-domain allocation** (a host-side layout copy of shared memory: no transfer, no
queue round trip, counted as `staged_reads`). The replacement is the catalogue's
`materialize_strided`/`tiled_transpose` Metal entry in U3 slice 3. This is not an op fallback: the
op still runs on the GPU, and the refusal for a *device* (non-shared) operand keeps the historical
error.

### 1.3 CI, docs, tests, bench

- `scripts/ci/run_profile.py` `"macos"` (`:59-64`): add, after the existing nextest run,
  1. `TENFERRO_REQUIRE_METAL=1 cargo nextest run -p tenferro-gpu --features metal <profile> --no-fail-fast`,
  2. a no-link check on the built test binary — `otool -L target/debug/deps/metal_engine-* | grep -Ei 'Metal|MetalPerformanceShaders|Accelerate'` must print nothing (issue #2402 "Checks to require").
  The Linux lanes keep compiling the crate with `metal` enabled but no Apple target:
  `crates/tenferro-gpu/src/metal` is gated on `feature = "metal"` for every target, so the
  module compiles everywhere; the runtime's probe refuses at run time, so `metal_devices()`
  is empty and `metal_available() == false`. That gives design requirement 6's observable
  result (no Metal device on non-Apple) without the literal compile-out its wording implies.
- `docs/guides/devices-and-gpu.md`: replace the "WebGPU / Metal / Apple shared CPU/Metal" row with
  a `Metal (experimental)` row naming the `metal` feature and exactly what slice 1 serves; add a
  "Metal coverage" table rendered from `metal_capabilities()` plus a default-features test that
  checks the two agree (the mechanism `capability.rs:218` already uses for CUDA).
- `docs/worklogs/2026-11-15-metal-u4-slice1-engine.md`: scope, decisions, the refusals, the
  measured M5 Max numbers, the review verdict, and the follow-up slices.
- `crates/tenferro-gpu/tests/metal_engine.rs` (`required-features = ["metal"]`): the hardware lane.
- `crates/tenferro-gpu/src/metal/*/tests.rs` (unit): the host-only lane (dtype admission,
  refusal messages, capability table, allocation-domain identity), runnable with no GPU.
- `crates/tenferro-gpu/benches/metal_engine.rs`: upload/download GB/s and elementwise GFLOP/s for
  f32 and c32 against the paired CPU engine on the same shared buffers, plus the per-launch cost.
- `api/tenferro-tensor.txt`, `api/tenferro-gpu.txt` (if present): regenerated by
  `scripts/api-snapshot.sh --write`; the only expected change is the new `tenferro_gpu::metal`
  surface.

## 2. Verification: exact commands, acceptance bars

Run in `~/projects/tensor4all/tenferro-rs-metal-u4` on the M5 Max. `python3` below means
`/opt/homebrew/bin/python3.12` locally (the system `python3` is 3.9 and has no `tomllib`); CI's
`python3` is 3.11+.

Correctness (M5 Max):

1. `TENFERRO_REQUIRE_METAL=1 cargo nextest run -p tenferro-gpu --features metal --no-fail-fast`
   — bar: 0 failures, and these named tests exist and pass:
   - `probe_reports_a_device` — probe names the device and `metal_available()`.
   - `round_trip_is_bit_exact` — f32 and c32 upload→download bit-exact for lengths
     {0, 1, 3, 255, 256, 257, 4096, 100_003}.
   - `elementwise_matches_the_cpu_reference` — 0 ulp for add/sub/neg/conj/abs/sign against the CPU
     engine over the same values, bit-exact for div (the U3 device lane already asserts
     bit-identical division).
   - `a_cpu_read_after_a_gpu_write_sees_the_values_and_counts_one_wait` — the wait rule of design §5.
   - `a_gpu_write_after_a_read_only_launch_waits` — a launch that only *reads* an allocation is
     registered, so a following `map_write` counts one wait (the reviewer's IMPORTANT: "waiting for
     pending GPU reads", design §5).
   - `f32_on_metal_and_f64_on_the_cpu_share_one_domain_with_no_transfer` — design §9 lane: one
     context, an f32 tensor produced by a Metal launch and an f64 tensor produced by the paired
     `CpuEngine` (e.g. `reduce_sum` over an in-domain f64 tensor), with `upload_bytes == 0` and
     `download_bytes == 0` and the Metal op reading its f32 operand afterwards.
   - `a_live_host_write_guard_makes_a_submission_report_mapped_for_host` — design §9 lane: hold
     `map_write` open on an allocation, submit a launch that names it, require the typed
     `MappedForHost` and no committed submission.
   - `a_foreign_domain_tensor_is_refused` — a shared tensor from a second `MetalContext` passed to
     the first context's engine is refused with the domain-identity error before any launch.
   - `two_threads_submit_and_allocate_without_deadlock` — one thread runs Metal ops, another
     allocates in the domain and runs CPU ops, both over one context; no deadlock, values correct
     (the reviewer's BLOCKER on locking).
2. `cargo test -p tenferro-gpu --features metal --lib` — the host-only unit lane (dtype admission,
   every refusal message, capability table, domain identity), which must also pass on a host with
   no Metal device.
3. `cargo test -p tenferro-tensor -p tenferro-cpu --lib` — bar: unchanged pass counts (351/360 on
   `origin/main`); the shared-domain admission is not touched.
4. `cargo doc -p tenferro-gpu --features metal --no-deps` — bar: no warnings, every new public item
   documented, `# Errors` sections on every fallible public function (the repo's rustdoc gate).

Additivity and off-Apple compilation (this host and the Linux lane):

5. `cargo clippy -p tenferro-gpu --features metal --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc`.
6. `cargo check --workspace --all-targets` (no GPU features) and
   `cargo check -p tenferro-gpu --features cuda,metal --all-targets` (both features at once; the
   0.3.0 catalogue bump must not disturb the CUDA engine).
7. `cargo fmt --all -- --check`.
8. `/opt/homebrew/bin/python3.12 scripts/check-git-pin-content.py` — bar: 0 errors, 0 warnings.
9. `/opt/homebrew/bin/python3.12 scripts/check-publish-layout.py`,
   `/opt/homebrew/bin/python3.12 scripts/api-snapshot.sh --check` (after `--write`).
10. `python3 scripts/ci/run_profile.py macos` on the M5 Max — bar: the new Metal commands pass and
    the `otool -L` grep is empty. Hosted `macos-15` (Apple Silicon) must be observed on the PR's
    own CI run; if the hosted runner has no Metal device, record that in the worklog and switch
    the required gate to the self-hosted M5 Max (design §9), not to "skipped".

Measurement (recorded in the PR body, with the capability dump and the MSL source hash from
`t4a_cuda_kernels::metal::source::compose`):

11. `cargo bench -p tenferro-gpu --features metal --bench metal_engine` — report, for
    n = 1e3 / 1e6 / 1e7 in f32 and c32: upload GB/s, download GB/s, add/mul GFLOP/s on Metal, the
    same op on the paired CPU engine over the same shared buffers (no copy between them), and the
    per-launch wall time. **This is a recorded observation, not a gate**: the PR states the numbers
    and the interpretation, including the per-launch overhead, and says plainly if Metal does not
    beat the CPU at some n. Tuning is U6's work (fusion); this slice does not iterate on numbers.

## 3. Edge cases, compatibility, deliberately unchanged

- **Empty tensors** (any extent 0): `Plan::dispatch.is_empty()` ⇒ no launch; the result is a
  zero-byte in-domain allocation. `upload`/`download` of an empty tensor succeeds and counts 0
  bytes.
- **Length > u32::MAX**: the catalogue refuses with `Error::LaunchLimit`; the engine maps it to
  `tenferro_tensor::Error::Unsupported`/`Validation` with the catalogue's message, never a wrapped
  launch.
- **f64 / Complex64**: typed `Unsupported` for the op, message
  `"... belongs to the paired CpuEngine, which reads these tensors without a transfer"`; no dtype
  narrowing, no CPU route.
- **Device-local / foreign domain**: a non-shared operand keeps today's refusal (`Unsupported`
  naming the backend family); a shared tensor from another `MetalContext` is refused by the
  domain-identity check before any launch (design §4.2).
- **Subnormal flush**: Metal's default `MTLCompileOptions` has fast math on and this device flushes
  subnormals to zero (measured by the U3 device lane). Slice 1 tests use normal-range values,
  asserts the flush once so the property is recorded rather than assumed, and states it in the
  guide. Removing it needs a compile-options seam in `t4a-metal-runtime` (cross-repo, recorded).
- **Aliasing** in `elementwise_read_into`: the op is computed into a fresh in-domain tensor and
  copied through `copy_read_into`, which applies the existing overlap validation; there is no
  in-place launch.
- **Threading/ownership**: `MetalEngine` is `!Sync` (submission borrows `&mut self`), `MetalContext`
  is `Clone` (an `Arc`) and `Send + Sync` (every path through one `Mutex<Runtime>`), matching
  `CudaEngine`'s ownership shape. `MetalStorage` is `Send + Sync`. A GPU write to an allocation a
  caller already holds a borrow of cannot happen: every GPU write targets an allocation created
  inside the same call (§1.2), and `map_read`/`map_write` guards make a submission report
  `MappedForHost`. The residual — a `shared_host_slice` borrow cannot raise that count because the
  U1 signature returns no guard — is written down in `mod.rs` and revisited by slice 2.
- **`map_write`/`shared_host_slice` unsafe**: the argument is stated once in `storage.rs`; the
  pointer aliases the GPU's address space and every touch is preceded by a wait.
- **Deliberately unchanged**: all CUDA paths and `CudaEngine`; the U1 admission rule and its
  follow-ups U1b (direct `TensorOp` surface domain check) and U1c (`is_host_memory` payload
  predicate); `REPOSITORY_RULES.md` (U1 already recorded the shared-domain rule);
  `HostAccessError::GpuAccessInProgress` on the managed-Cholesky tests; the `gpu` seam's
  vocabulary; `api/tenferro-tensor.txt` (no new public item in `tenferro-tensor`: `BackendId::Metal`
  is not added here).
- **No traced-program registration**: `MetalEngine` does not register an `ExecutionEngine` id, so
  a traced program cannot silently mix engines (design §5: one engine per traced program). Eager
  code is the unit's surface.

## 4. Risks and rejected alternatives

Risks, each with its mitigation:

1. **Release chain (resolved).** The branch originally pinned unpublished siblings:
   `t4a-metal-runtime` 0.1.1 lacked the `Send` impl and `t4a-cuda-kernels` 0.2.0 was not on
   crates.io. Both are now published: the runtime as `0.1.2` (from `e695c24` plus the version
   bump) and the catalogue as `0.3.0`, with `t4a-metal-sys` `0.1.0` transitive. The pins are
   registry requirements, the exception entries are gone, and `tenferro-gpu`'s own release is
   no longer blocked by this dependency.
2. **Catalogue 0.1.2 → 0.3.0.** The catalogue moved its CUDA ABI to revision 14 and changed
   reduction entries. Mitigation: `cargo check --features cuda,metal` locally plus the hosted Linux
   CUDA-compile lane; the CUDA *runtime* tests cannot run here (no NVIDIA device) and the GPU lane
   is spent once on the finished PR.
3. **Per-launch overhead.** One command buffer per elementwise call is the slice-1 shape; U6's
   fusion emitter is what removes it. The benchmark records the overhead so the claim is measured.
4. **Hosted `macos-15` may have no Metal device.** Mitigation: observe the PR's lane; if absent,
   the requirement is a self-hosted Apple Silicon runner (design §9) and the worklog says so
   instead of skipping the correctness tests.
5. **CPU layout staging inside a Metal op.** Documented and counted; replaced by the Metal
   materialization kernel in U4 slice 2.
6. **A second cross-repository change (the `Send` declaration) rides along.** Mitigation: it is
   three lines plus a contract test, it is pinned by revision, and the alternative (a local
   `unsafe impl` over a foreign type) makes the thread-safety claim where nobody maintains it.

Rejected alternatives:

- `newBufferWithBytesNoCopy` wrapping host buffers (issue body): rejected by the design §3; the
  shared allocation domain replaces it, and U1's admission already consumes it.
- Routing f64 to the CPU inside the Metal engine (issue body): rejected by the design §2
  requirement 1; typed `Unsupported` pointing at the paired engine instead.
- Shipping MSL text from `tenferro-gpu`: design §7 puts the source set in the kernel catalogue,
  shipped as data; duplicating it here would create a second path.
- Implementing reductions now with hand-written MSL: the task and design make MPS the owner of
  reductions; doing it with MSL first would build the path we are told not to build. Deferred to
  U4 slice 3 with MPS first.
- Adding `BackendId::Metal` to `tenferro-tensor` in this PR: a public enum variant is a
  compatibility decision (exhaustive matches downstream), so slice 1 uses `BackendId::Other("metal")`
  and the variant is raised separately.
- Registering the engine as a `tenferro-runtime` `ExecutionEngine` now: design §5 keeps one engine
  per traced program; eager execution is this unit's scope.
- One giant U4 PR covering reductions, structural kernels, GEMM and fusion: it would not be
  reviewable and would defer every partial verification; the repository's own precedent
  (`cuda/structural.rs:3-7`, U3 slice 1) is a typed-Unsupported first slice.

## 5. Review record

Reviewer: `ollama-cloud/glm-5.3-flash` (pi subagent `glm-brainstormer`, read-only), reviewing this
plan before any code is written. Nothing was edited by the reviewer.

**Verdict: CHANGES** (round 1). Every finding was checked against the tree; all seven were valid
and are addressed here:

- BLOCKER — locking/exclusivity unspecified (`&mut Runtime` behind `allocate(&self)`):
  §1.0 (`Runtime: Send` sibling change) and §1.2 ("one lock, no second path") plus the
  `two_threads_submit_and_allocate_without_deadlock` bar in §2.1.
- BLOCKER — a live host borrow is not excluded from a concurrent GPU write: §1.2 "Who may write an
  allocation while a host borrow is live" (live-host count + `MappedForHost`, all GPU writes into
  fresh allocations, `copy_read_into` on the host) and the `a_live_host_write_guard_...` bar.
- IMPORTANT — the wait rule must cover GPU reads: `PendingTouches` registers reads as well as
  writes (§1.2), with `a_gpu_write_after_a_read_only_launch_waits` in §2.1.
- IMPORTANT — two design §9 lane items missing from the acceptance bars: the f64-on-the-CPU /
  f32-on-Metal same-domain test and the `MappedForHost` test are now named bars in §2.1 (this also
  corrected a real error in the draft: `MetalAllocationDomain::allocate` must serve **all** `DType`s,
  not only f32/c32, or design §4.3's CPU outputs could not live in the domain).
- MINOR — the benchmark bar contradicted itself: §2.11 is now a recorded observation with a
  written interpretation and no tuning loop.
- MINOR — the pins block `tenferro-gpu`'s release: recorded in §1.1 and §4.1.
- MINOR — the foreign-domain refusal had no acceptance command: `a_foreign_domain_tensor_is_refused`
  is a named bar in §2.1.

## 6. Implementation status

U4 slice 1 is implemented in this worktree. What ran, on the Apple M5 Max:

- `TENFERRO_REQUIRE_METAL=1 cargo test -p tenferro-gpu --features metal --test metal_engine`
  — 24 passed, 0 failed.
- `cargo test -p tenferro-gpu --features metal --lib` — 6 passed (3 of them new Metal table
  tests).
- `cargo clippy -p tenferro-gpu --features metal --all-targets -- -D warnings
  -D clippy::missing_errors_doc -D clippy::missing_panics_doc` — clean; the same flags over
  `--workspace --all-targets` — clean.
- `cargo clippy -p tenferro-gpu --features cuda,metal --all-targets -- -D warnings
  -D clippy::missing_errors_doc -D clippy::missing_panics_doc` — clean.
- `cargo doc -p tenferro-gpu --features metal --no-deps` — no warnings; `cargo test --doc`
  — 7 passed.
- `cargo fmt --all -- --check` — clean; `cargo check --workspace --all-targets` — clean;
  `cargo test -p tenferro-tensor --lib` 350 passed and `-p tenferro-cpu --lib` 361 passed
  (both crates are untouched by this diff; the counts differ from the draft's 351/360 because
  the draft quoted a different feature set).
- `scripts/api-snapshot.sh --check` — ok, with a new `tenferro-gpu+metal` profile written to
  `api/tenferro-gpu+metal.txt`; `scripts/check-publish-layout.py` — ok;
  `scripts/check-macos-no-framework-link.sh` — ok on the built `metal_engine` binary.
- `scripts/check-git-pin-content.py` — **passes**: `0 warning(s), no content drift`. The
  registry requirements (`t4a-metal-runtime =0.1.2`, `t4a-cuda-kernels =0.3.0`,
  `t4a-metal-sys =0.1.0` transitive) replaced the temporary git pins, so §1.1 and §4.1 record
  no remaining release follow-up.
- Benchmark (`cargo bench -p tenferro-gpu --features metal --bench metal_engine`): recorded in
  `docs/worklogs/2026-11-15-metal-u4-slice1-engine.md` and the PR body.

## 7. Post-implementation review record

Reviewer: `gpt-6.1-sol` (pi subagent `reviewer-gpt`, read-only: no edits, no builds), reviewing
the finished diff against the design constraints and the bars in §2.

**Verdict: CHANGES** (round 1), five IMPORTANT findings. Every finding was re-checked against the
tree; all five were real, and all five are fixed in the worktree:

1. IMPORTANT — the host-admission check ran *before* `submit` took the runtime lock, so a guard
   could appear between the check and the commit. Fixed: the engine collects
   `MetalStorage::live_counter()` and `submit` performs the check under the same lock that the
   guard needs to raise the count (`engine.rs::submit`), before anything is encoded.
2. IMPORTANT — `map_read`, `map_write` and `write_from_slice` waited for pending work while
   holding the runtime mutex, stalling the whole context for up to the completion timeout.
   Fixed: the live count is raised under the lock, the lock is released, then the wait runs
   (`storage.rs`), so the ordering argument holds without the lock being held across a host
   completion wait. `commit` itself remains under the lock and can wait on the runtime's
   in-flight bound; `mod.rs` states that back-pressure case instead of claiming every wait is
   outside the lock.
   The same change fixed a real defect the review did not name: two host mappings of one
   allocation were allowed to overlap, including a write next to a read. The count is now an
   exclusive slot (compare-exchange) and a second mapping returns
   `HostAccessError::OverlappingHostMapping`; `a_second_host_mapping_is_refused_as_overlapping`
   covers it.
3. IMPORTANT — `scripts/check-macos-no-framework-link.sh` searched `target/debug` while the
   hosted lane builds the `ci` profile. Fixed: it searches the whole target directory (honouring
   `CARGO_TARGET_DIR`), and it was verified against a real `--profile ci` build
   (`target/ci/build/tenferro-gpu/.../metal_engine-...`: links no Apple GPU framework).
4. IMPORTANT — the pin exceptions cannot rescue the gate and the file still said "no exceptions
   are current". The review round recorded the two pins as a temporary deviation, but the final
   state removes them: `t4a-metal-runtime 0.1.2` and `t4a-cuda-kernels 0.3.0` are published and
   the workspace depends on the registry directly, so the exception file is back to "no
   exceptions are current" and the gate passes.
5. IMPORTANT — two bars could pass without the protection they claim. Fixed: `PendingTouches`
   counts only a wait that had a submission to wait for, so `MetalStats::waits` now evidences a
   recorded completion; `a_gpu_write_after_a_read_only_launch_waits` proves the rule with values
   (a host overwrite after a read-only launch still sees the launch's inputs) as well as the
   counter; the guard test asserts that a refused submission commits nothing and that the same
   call succeeds once the guard is dropped; the two-thread test asserts the computed values, not
   only the length. A deterministic test of the check/commit interleaving is not constructible
   without a hook inside `submit`; finding 1 is now closed by construction — one lock covers the
   count and the commit — which the code shows and no timing test could add to.

Re-verified after the fixes, on the Apple M5 Max: 19 tests pass under
`cargo nextest run --cargo-profile ci -p tenferro-gpu --features metal --test metal_engine` with
`TENFERRO_REQUIRE_METAL=1` (the exact hosted-lane command), 6 lib tests pass, clippy with
`-D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc` is clean, and
`cargo fmt --all -- --check` is clean.

The reviewer reported no BLOCKER. Its open-question caveats (dependency-internal probe behaviour,
exhaustive refusal-message coverage, API snapshot, guide consistency) were outside what a
read-only reviewer without command execution could check; they are covered by §6's command log
(snapshot, doc, doctests, workspace check) and by inspection of the refusal messages in
`engine.rs`.
