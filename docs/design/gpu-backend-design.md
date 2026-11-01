# GPU Backend Design

This document describes the implementation; it does not override it. The
manifests and `crates/tenferro-gpu/src/cuda/` are authoritative if prose drifts.
Execution uses a caller-held `tenferro_gpu::cuda::CudaEngine`, opened with
`CudaEngine::open(device_id)` and borrowed mutably for operations. The `cuda`
feature is opt-in. WebGPU, Metal execution, and the Apple shared CPU/Metal path
are not provided today; there is no `webgpu` feature or provider module.
`rocm` is an empty reserved feature, not an execution backend.

The implementation is in `crates/tenferro-gpu/src/cuda/`. It builds on
`t4a-cuda-runtime` and the `t4a-cuda-kernels` catalogue, not CubeCL or CubeK.
`CudaBackend` is the CUDA storage-seam marker, not the operation entry point.
There is no `CudaExecSession`, `with_backend_session`, or `with_cubecl` entry.
The raw vendor-call boundary is `CudaEngine::with_raw` in `cuda/raw.rs`;
`cuda/engine.rs` owns the engine state, and `cuda/launch.rs` binds and launches
the external catalogue's kernels. CubeCL is retired; the legacy names above
do not describe an available execution path.

Historical CubeCL/CubeK fork evaluations and earlier design records do not
specify the current dependency or execution contract.

See also:

- [backend-contract.md](../spec/backend-contract.md) for engine and placement rules,
- [Devices and GPU](../guides/devices-and-gpu.md) for current operation/dtype
  coverage and CUDA library compatibility,
- [CUDA engines and pinned transfers](../guides/engines-and-transfers.md) for
  caller-held engines and explicit transfer completion,
- `crates/tenferro-gpu/Cargo.toml` and the root `Cargo.toml` for features and
  dependency pins.

## Current Module Structure

```text
crates/tenferro-gpu/src/gpu/
    mod.rs                 backend-neutral storage identity, allocations and errors

crates/tenferro-gpu/src/cuda/
    mod.rs                 CUDA seam marker and exports
    engine.rs              caller-held CudaEngine and engine-owned state
    runtime.rs             t4a-cuda-runtime adapter and submission locking
    storage.rs             CUDA allocations wrapped as tensor storage
    transfer.rs            explicit upload/download and pending transfers
    pinned.rs              reusable pinned host buffers
    pending.rs             pending transfer handles
    publication.rs         published values and completion frontiers
    dispatch.rs            typed catalogue planning and argument marshalling
    launch.rs              NVRTC artifacts, bound functions and launches
    structural.rs          engine structural/read/write operations
    fusion.rs              fused elementwise catalogue integration
    cutensor.rs            cuTENSOR contraction integration
    vendor.rs              cuBLAS integration
    raw.rs                 engine-scoped vendor stream, spans and scratch
    runtime_adapter.rs     prepared-runtime engine registration
    event_domain.rs        prepared-runtime event integration
```

`crates/tenferro-linalg/src/gpu/` owns cuSOLVER/cuBLAS linalg integration.
`crates/tenferro-fft/src/cuda/` owns cuFFT loading, plans and execution. These
operation crates use the CUDA engine rather than owning another execution
session or runtime.

## Kernel Ownership

Reusable static CUDA kernel sources, entry ABIs and `plan_*` validation belong
to the external `t4a-cuda-kernels` catalogue. They do not live under
`tenferro-gpu/src/kernels/` or `src/cubecl/`; those directories are absent.

Tenferro owns tensor semantics, placement and allocation-domain checks,
conversion of tensor layouts to catalogue operands, argument marshalling,
engine dispatch and vendor-library integration. `cuda/launch.rs` composes the
catalogue's source modules, prepares NVRTC artifacts through the runtime,
binds functions and launches them. `cuda/fusion.rs` integrates the catalogue's
fusion support. Do not restore a parallel CubeCL kernel path.

## Dependency Source

The root workspace declares the `t4a-cuda-runtime` and `t4a-cuda-kernels` pins
with `git`, `rev` and `version`. `crates/tenferro-gpu/Cargo.toml` makes them
optional workspace dependencies, enabled together with `cudarc` by `cuda`.
The runtime comes from `tensor4all/t4a-gpu-backend`; the kernel catalogue comes
from `tensor4all/t4a-cuda-kernels`. The manifests, not a copied dependency
stanza here, are the source of truth for their revisions.

There are no current CubeCL/CubeK dependencies to update or restore. Historical
fork-version evaluations remain historical, not setup instructions.

## Runtime And Cache Ownership

A `CudaEngine` owns a `CudaRuntime`, prepared catalogue dispatchers, cuTENSOR
state, pending prepared runs and an extension-cache store. Its `&mut` borrow
is the operation-exclusivity contract; the engine is `Send`, not `Sync`.
`CudaRuntime` adapts `t4a_cuda_runtime::Runtime` and owns the stream,
allocation domain and transfer state. Explicit transfers use that same
runtime's submission lock and stream.

`CudaRuntime::synchronize()` is the explicit CUDA host barrier. There is no
CubeCL launch queue or physical-stream pool to flush or select. Vendor calls
bind to the stream lent by the engine's raw scope.

cuTENSOR and cuBLAS integration lives in `tenferro-gpu`; cuSOLVER integration
lives in `tenferro-linalg`. cuFFT loading and opaque plan entries remain owned
by `tenferro-fft`. Its `cufft-plans` entries live in the executing engine's
extension-cache store, not a caller-owned `FftExecutionCache`. The plan entry
retains the required work-area byte size, while execution obtains fresh scratch
through the raw scope and synchronizes before returning. See
`crates/tenferro-fft/src/cuda/plan.rs` for lifecycle and error handling.

The CUDA library path overrides are:

| Variable | Library owner |
| --- | --- |
| `TENFERRO_CUTENSOR_PATH` | cuTENSOR in `tenferro-gpu` |
| `TENFERRO_CUBLAS_PATH` | cuBLAS loader in `tenferro-linalg` |
| `TENFERRO_CUSOLVER_PATH` | cuSOLVER in `tenferro-linalg` |
| `TENFERRO_CUFFT_PATH` | cuFFT in `tenferro-fft` |

CUDA operation paths implemented through NVIDIA libraries report load or
provider errors when the required library is unavailable or lacks support;
they must not silently fall back to native kernels. See the guide's
[compatibility matrix](../guides/devices-and-gpu.md#cuda-and-nvidia-library-compatibility)
for the supported runtime and library combinations.

## Operation-Crate Interop Boundary

Operation crates receive the caller's engine. When handed an erased `Engine`,
the CUDA leaf's `cuda::raw::with_cuda_engine` visitor recovers the CUDA engine
through `Engine::native_engine()`; it does not construct or enter another
engine. Vendor execution then uses `CudaEngine::with_raw(op, closure)`.

The closure receives `cuda::raw::Session`, which lends the runtime's stream,
validated tensor spans and scratch. This is a raw CUDA scope, not the removed
backend-session admission mechanism. Work must use that scope's stream and
obey the raw method and vendor-library safety contracts. Whole-tensor
transfers run outside the scope: it holds the runtime submission lock, so
re-entering transfer admission from it is not supported.

`cuda/raw.rs` defines the stream, tensor-span, scratch, retention and
synchronization operations. `CudaRuntime::with_current_context` is the separate
context guard for vendor lifecycle work that only sets up state and enqueues
nothing. `tenferro-linalg` uses `with_raw` for cuSOLVER/cuBLAS calls;
`tenferro-fft` uses the context guard for plan lifecycle work and `with_raw`
for execution. There are no CubeCL sub-sessions or CubeCL bindings in this
contract.

For a checked downstream example, see
`docs/tutorial-code/src/bin/cuda_vendor_interop.rs`. For ordinary tensor calls,
use `CudaEngine`, `cuda::upload_tensor`, `cuda::download_tensor`, and the
[CUDA quickstart](../guides/devices-and-gpu.md#cuda-quickstart).

## Kernel Metadata And Launch Contract

Owned dense tensors use column-major logical layout. Borrowed CUDA views carry
shape, signed strides and offsets; a supported strided entry consumes that
layout, while an entry requiring compact input stages a same-device copy.
Neither case implies a host transfer.

For supported dtypes, `to_contiguous_read` materializes CUDA views on-device.
`copy_read_into` accepts valid source and non-overlapping destination regions of
the same runtime, including ranks above the catalogue's by-value layout limit.
Non-flat wide copies use `StridedToStrided`'s general symbol with device-resident
geometry; neither operand needs tensor-sized staging. Compact copies retain the
layout-free exact-copy path. Source and destination allocations must not alias.
Other write entries retain their documented by-value layout rank limit.

Catalogue planners receive validated dtype/layout/operation metadata and
produce the launch geometry. `cuda/dispatch.rs` calls the catalogue's `plan_*`
functions, then marshals runtime arguments in the entry's ABI order.
`cuda/launch.rs` uses catalogue `CLayout` parameters and runtime `Arg` values,
not CubeCL `TensorBinding`, `ArrayArg` or `#[comptime]` metadata. Shape extents,
strides, offsets and lengths must follow the selected catalogue entry's ABI;
do not reintroduce layout-specific CubeCL specializations.

Launch over the output or update domain using the catalogue plan. The dispatch
layer uses the plan's grid and block sizes rather than inventing a serial
fallback. Device addresses, byte spans and layout bounds must be validated
before unsafe launch arguments are submitted. The implementation sources are
`cuda/dispatch.rs`, `cuda/structural.rs` and `cuda/launch.rs`.

## Backend Allocation Ownership And Device Transfers

`StorageBuffer::Backend` owns provider storage. CUDA wraps a
`t4a-cuda-runtime` allocation in the GPU storage seam; views borrow that owner.
The engine runtime identifies the allocation domain. A matching device ordinal
alone does not make a foreign engine's allocation interchangeable.

Tenferro follows the PyTorch convention: no implicit CPU/GPU transfer at tensor
API boundaries. Upload host tensors with `cuda::upload_tensor` before CUDA
operations, and download with `cuda::download_tensor` before host inspection or
CPU-only execution. Same-device canonicalization is allowed, not a hidden
transfer. See `cuda/transfer.rs` for the blocking and pending routes and their
validation. The executable quickstart is
`crates/tenferro-gpu/examples/cuda_quickstart.rs`.

## Implemented Coverage

The [Devices and GPU coverage table](../guides/devices-and-gpu.md#coverage) is
the current operation/dtype summary. Backend implementation sources and tests
are authoritative; do not infer support from a historical CubeCL/CubeK design.
WebGPU, Metal and Apple shared CPU/Metal execution are not available, and ROCm
has no execution implementation.

### CUDA linalg driver notes

CUDA SVD follows the accuracy-preserving, JAX-compatible default driver
selection when the caller leaves `SvdOptions::driver` at `SvdDriver::Auto`: use
cuSOLVER Jacobi `gesvdj` when both matrix dimensions are at most 1024,
otherwise QR-based `gesvd`. `Auto` never selects the polar `Xgesvdp` (#2182).
`SvdDriver::Gesvdj` and `SvdDriver::Gesvd` force the routine regardless of
size, because the faster driver depends on the spectrum and the shape, not the
dimensions alone. The driver is part of the traced op identity and survives
pruning to the values-only op; CPU providers ignore it. `gesvdj` returns V, so
the backend materializes the public `vt` output by copying V to V^H on the
device. The singular-values-only path still passes scratch U/V buffers to
`gesvdj` because cuSOLVER rejects null U/V pointers on that path.

Ordinary CUDA QR (`qr`/`qr_with_options`) does not inspect the cuSOLVER
`geqrf`/`orgqr` device `info` flags, so forward QR issues no host barrier.
Synchronous provider return codes still propagate, and the doc-hidden compact
Householder hooks keep their own `geqrf` status read. The flags are not useful for QR: the illegal-argument
report is unreachable because shapes, dtypes, and dimensions are validated
before the vendor call, and an exactly zero `R` diagonal (rank deficiency) is a
valid factorization that cuSOLVER does not flag. A new CUDA decomposition should
add a status download only where the flag carries a failure the caller cannot
observe any other way, and never as an implicit host barrier on the forward
path.

CUDA `solve` and `lu_factor_solve` do not inspect the cuSOLVER `getrf` device
`info` flags and do not scan the packed LU diagonal, so the forward solve issues
no host barrier and performs no per-call non-singularity check. Shapes, dtypes,
dimensions, and leading dimensions are validated before the vendor call, and an
exactly singular factor is left to the caller's verification, as in
`torch.linalg.solve`. The public `lu_factor`/`lu` operations keep their `getrf`
status read and `Singular` error, and the forward solve also skips the
permutation-parity kernel that only the public LU outputs consume.

`SvdDriver::Xgesvdp` is the opt-in polar-decomposition routine; `Auto` never
selects it, and the explicit variant forces it at any size and dtype. It is the
fastest of the three drivers for `min(m, n) >= 256` on an A100 (see the
`cuda_svd_driver` benchmark), but its small singular values are only accurate to
about `1e-6` of the largest. Do not combine it with an `eps`-scale truncation
(a pseudoinverse or tensor-network truncation at `1e-12`..`1e-14`): the
perturbation can change the retained rank. Its 64-bit cuSOLVER call uses
matching real singular values and separate device/host byte workspaces. Both
matrix orientations are supported directly. Like `gesvdj`, it returns V and
needs an on-device adjoint for the public Vᴴ result. Values-only execution
selects `NoVector`, not a full decomposition. Host workspace and `h_err_sigma`
storage remain live through a completion barrier, including vendor-error paths.
`h_err_sigma` is cuSOLVER's absolute perturbation bound on the computed
singular values; tenferro rejects the factorization with a typed backend
failure when the bound is not small relative to the largest singular value,
instead of returning silently wrong factors. The check pairs each batch member
with its own spectrum and names the failing batch. That check is what makes an
explicit rank-deficient `F32`/`C32` request fail rather than return the perturbed
values. The existing `svd_full` API has no driver option and uses `Auto`; full
Xgesvdp factors are covered internally without extending that public API.

CUDA eigh routes by batch count, not by order. A real batch
(`batch_total > 1`) always takes a batched cuSOLVER routine, because that
replaces one launch per matrix with a single launch; a single matrix keeps the
per-matrix entry point. Which batched routine depends only on the driver:
`EighDriver::Auto` and `Syevd` take divide-and-conquer
`cusolverDnXsyevBatched`, and `Syevj` takes Jacobi `syevjBatched`.

There is deliberately no size threshold. An earlier revision capped batched
Jacobi at order 32 and described that as a cuSOLVER restriction; issue #1852
showed the manual documents that limit for `gesvdjBatched`, the SVD routine,
not for `syevjBatched`, which accepts any order. Measured on an A100, each
batched routine beats its own per-matrix loop at every order from 8 to 512, so
no order exists at which falling back is faster.

`Auto` therefore no longer reproduces the pre-driver behaviour bit for bit on
batched input: it is defined as the fastest divide-and-conquer routine
available, and the batched entry point is a different cuSOLVER call. It is the
same algorithm at the same accuracy, and it made the default about 100x to
280x faster on batches (105 ms to 0.37 ms for 1024 8x8 `f64`; 70 ms to 0.69 ms
for 256 32x32). `Syevj` retains a narrow advantage only up to about order 32.

`cusolverDnXsyevBatched` requires cuSOLVER 11.7.1 or newer, which is why the
supported CUDA floor is 12.6.2. That is a library floor: the loader resolves
cuSOLVER at runtime, and CUDA 12.x supports the same GPU architectures
throughout, so no hardware or driver requirement changes. Making the symbol
optional was rejected — the cuSOLVER loader treats every symbol as mandatory,
so an optional entry point would need a second loading mode, and leaving the
floor at 12.4 with a mandatory symbol would turn a missing optimisation into a
total cuSOLVER load failure.

As with the SVD driver, the eigh driver is part of the traced op identity and
survives pruning to the values-only op, and CPU providers ignore it.

## Tests

Use the repository's local GPU gate for CUDA validation, as described in
`AGENTS.md`. Focused ignored GPU tests can be run on a configured CUDA
12.6.2-or-newer machine with:

```sh
CUDA_PATH=/usr/local/cuda-12.6 \
LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/lib/x86_64-linux-gnu/libcutensor/12:$LD_LIBRARY_PATH \
  cargo test -p tenferro-gpu --features cuda -- --ignored
```

These are correctness tests, not benchmarks. CUDA 12.8-or-newer driver and
NVRTC libraries enable the full CUDA feature set; consult the guide's
compatibility matrix rather than the retired CubeCL runtime policy.
