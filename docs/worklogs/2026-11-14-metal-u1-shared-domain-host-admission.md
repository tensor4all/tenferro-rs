# Metal backend U1: CPU admission of shared allocations (tenferro-rs#2402)

## Scope

The approved design (`tensor4all-meta` `8befe98`, `docs/design/metal-backend.md`)
splits the Apple GPU backend into six units across three repositories. U2
(`t4a-metal-sys` / `t4a-metal-runtime`, the `dlopen`/`objc_getClass` tables and
the probe) and U3 (Metal kernel sources and `Target`) live in
`t4a-gpu-backend` and `t4a-gpu-kernels`; U4–U6 (the Metal engine, MPS GEMM and
the MSL fusion emitter) depend on them. **U1 is the only unit that is first in
sequence and implementable in this repository alone**, and it is exactly "shared
domain in tenferro, no Metal yet": `GpuBackendKind::Metal`, the CPU-boundary
admission of §4.3 with the wait hook of §5, and the `REPOSITORY_RULES.md` rule
for shared-domain host inspection.

No `metal` cargo feature is added: design §8 says the feature lands in the pull
request that gives it a working path, and `tenferro-gpu` has no Metal path until
U4.

## Decisions

- **One provider method is the whole admission rule.**
  `BackendStorage::shared_host_slice(&self) -> Result<&[T], HostAccessError>`
  defaults to `Unsupported`, so every existing backend (CUDA's `CudaStorage`,
  the opaque test handle) keeps today's behaviour without an edit. A shared
  allocation domain implements it as "wait for pending device writes, then lend
  the elements" — the call *is* the wait, so no second fence API exists.
- **Borrow, not a guard.** `map_read` keeps the closure-scoped
  `HostReadGuard`, which cannot supply the `StridedView<'a, T>` a CPU kernel
  needs. A shared allocation is never mapped/unmapped, so the borrow is the only
  lifetime that exists. This deviates from
  `docs/design/storage-ownership-contracts.md` G1's "prepared host access →
  guard lives through call" row, deliberately and only for shared allocations;
  exclusive CPU writes still go through `map_write`.
- **Reads only.** The slice is admitted by `typed_host_data`, `typed_view`,
  `typed_view_from_view` (`tenferro-cpu-basic`) and `operand_from_view`
  (`tenferro-cpu`). CPU *outputs* stay pooled host tensors: design §4.3 describes
  "CPU outputs are allocated in the same domain" as existing
  `with_allocation_domain` behaviour, but in this code only the managed FFT,
  managed Cholesky and `to_contiguous_read` copy paths allocate through the
  domain. Routing every kernel's output through the domain is follow-up work.
- **Engine-domain identity stays where it already is: the ingress contract.**
  Design §4.3 says a buffer in the engine's *own* shared domain is admitted and
  another domain keeps today's error. The second half already holds: a CPU
  engine registers its domain and `cpu_input_signature`
  (`runtime_adapter.rs:146-158`) accepts a managed input only when its
  `allocation_domain` equals that registered domain, so a foreign-domain or
  device-local input is refused before any kernel runs, and
  `runtime_adapter/tests.rs:20-58` covers exactly that (own accepted; foreign,
  missing and device inputs rejected). These helpers are below that contract,
  so they admit what a provider offers and pass a provider's own typed refusal
  (`ForeignDomain` included) through as `Error::HostAccess`.
  What is *not* covered: the direct `TensorOp` method surface
  (`session.add_read` and friends), which has no registration to compare
  against and therefore reads a foreign domain's shared allocation if handed
  one. Follow-up U1b adds `validate_shared_domain_read(op, inputs, own_domain)`
  to the delegated read entries in `exec_session.rs`, next to the existing
  `blas1::validate_cpu_read`, so the direct surface refuses it too. Neither the
  independent review of the whole change nor the second pass accepted the
  deferral as free, so it is recorded here as a gap with its enforcement point
  named, not as a satisfied requirement.
- **The payload predicate is untouched.** `is_host_memory()` still reports a
  shared allocation as a backend payload. Ops that narrow through
  `TypedTensorView::try_host` — reductions (`reduction.rs:75`), BLAS1
  (`blas1.rs`), einsum `dot` (`dot_runtime.rs`), clone/convert
  (`lib.rs`, `clone_host_tensor_read`) — therefore keep refusing shared tensors
  with their current typed error. `structural.rs` records the invariant "the two
  surfaces cannot disagree about one tensor", so admitting shared memory in
  `try_host` requires the predicate to agree, which is follow-up U1c.
  `a_shared_tensor_reduction_still_refuses_until_the_payload_predicate_admits`
  pins today's refusal so turning it on is a deliberate change.
- **The view path resolves through the root.** A view over a group-backed tensor
  holds `TensorStorageRef::Root`, so `TypedTensorView::shared_backend_storage`
  resolves the retained `BackendStorageAllocation` (one `pub(crate)` accessor,
  same interior-mutability argument as the existing root accessors) to keep the
  view's storage lifetime. Without it, only owned tensors would be admitted and
  strided shared operands would fail.

## Rejected alternatives

- Wrapping host buffers with `newBufferWithBytesNoCopy` (issue body): rejected by
  the design; a shared allocation domain replaces it, and nothing in this change
  can wrap host memory.
- Routing f64 to the CPU automatically (issue body): rejected by the design; a
  Metal op on f64/Complex64 will return typed `Unsupported` in U4.
- Threading `Option<&dyn SharedTensorAllocationDomain>` through the ~100
  `*_with_pool` kernel entries to enforce engine-domain identity now: 35
  `operand_from_view`/`typed_view_from_view` call sites and every kernel entry
  would carry a domain they never read, and the review of the admission rule
  itself would drown in that diff. Recorded as U1b: the registered-engine path
  already refuses a foreign domain through `cpu_input_signature`, so what is
  left is the direct `TensorOp` method surface, closed by one predicate at the
  delegated read entries. `shared_host_slice(&self)` offers no reader identity,
  so no provider can close it from below.
- Making `is_host_memory()` true for shared allocations: it silently changes
  transfer bookkeeping and the CUDA refusal paths; recorded as U1c.

## Verification

Run in `~/projects/tensor4all/tenferro-rs-metal-u1` on macOS 26.7.2 / Apple M5
Max (arm64), Rust 1.100.0:

- `cargo test -p tenferro-cpu --lib managed_copy_boundaries` — 16 passed,
  including the six new admission tests (elementwise over two shared operands
  with one wait per operand and zero domain allocations, a transposed shared
  view, a sliced shared view keeping its offset, an every-other-element sliced
  view keeping its step, a device-only operand keeping the historical download
  error, and a provider refusal surfacing as `Error::HostAccess`), plus the
  pinned reduction refusal.
- `cargo test -p tenferro-tensor --lib` 351, `cargo test -p tenferro-cpu --lib`
  360, `cargo test -p tenferro-cpu-basic --lib` 38,
  `cargo test -p tenferro-internal-cpu-kernels --lib` 32 — all passed.
- `cargo test --doc -p tenferro-tensor` 629,
  `cargo test --doc -p tenferro-cpu -p tenferro-cpu-basic` 97 — all passed.
- `cargo clippy -p tenferro-tensor -p tenferro-cpu-basic -p tenferro-cpu
  -p tenferro-internal-cpu-kernels --all-targets -- -D warnings` — clean (only
  the two pre-existing unused-workspace-dependency notes).
- `cargo check --workspace --all-targets` — no errors.
- `cargo fmt --all -- --check` — clean.
- `scripts/api-snapshot.sh --write` then `--check` — `GpuBackendKind::Metal`
  and the new `BackendStorage` method recorded in `api/`.
- `scripts/check-public-error-docs.py`, `check-api-consistency.py`,
  `bash scripts/check-pr-fast.sh`, `repository-rules-review.py` — see the PR
  body for the recorded results.

## Measured device facts for U2–U5 (not produced by repository code)

Because U2/U3 live in other repositories, this PR contains no Metal execution.
To keep the design's device claims checked on *this* machine, a throw-away
harness outside the repository measured on macOS 26.7.2 / Apple M5 Max
(40-core GPU, `hasUnifiedMemory = 1`, `maxThreadsPerThreadgroup = 1024`). Neither
binary is committed; the first links no framework at all (`otool -L` prints only
`/usr/lib/libSystem.B.dylib`) and the second links Metal, MPS and Accelerate only
because it measures them.

Runtime-only loading (design requirement 5), `dlopen` + `dlsym` + `objc_getClass`
in plain C:

| Probe | Result |
| --- | --- |
| `otool -L dlopen_probe` | `/usr/lib/libSystem.B.dylib` only — no Metal, MPS or Accelerate link |
| `dlopen` Metal / MetalPerformanceShaders / MetalPerformanceShadersGraph | all three succeed |
| `dlsym MTLCreateSystemDefaultDevice` | resolves, returns `Apple M5 Max` |
| `hasUnifiedMemory` | `1` |
| `objc_getClass("MPSMatrixMultiplication")`, `("MPSMatrix")` | present after the MPS framework is opened |
| `objc_getClass("MPSGraph")` (after opening its own framework) | **present** — confirms the design's "re-check": the earlier "MPSGraph absent" result only meant that framework had not been opened |
| `objc_getClass("MTLCommandQueue")`, `("MTLComputePipelineState")` | absent — they are protocols, so U2's probe must use `objc_getProtocol` for them |
| `fopen("/System/Library/Frameworks/Metal.framework/Metal")` | absent while `dlopen` on the same path succeeds (dyld shared cache) |

MSL compilation at run time (`newLibraryWithSource:`):

| Source | Result |
| --- | --- |
| f32 fused elementwise (`fma(a, x, tanh(y))`) | compiled, 0.5 ms warm / 42 ms on the first compile in a process |
| c32 fused (complex multiply over `float2`) | compiled, 0.0 ms warm |
| `double` kernel | **rejected in 10–28 ms**: `error: 'double' is not supported in Metal` |

The two compiled kernels were also dispatched over 4 194 304 elements
(256-thread threadgroups, `MTLResourceStorageModeShared` buffers) and checked
against a double-precision CPU reference:

| Kernel | Time | Effective traffic | Max abs diff |
| --- | ---: | ---: | ---: |
| f32 fused elementwise | 383 µs | 131 GB/s | 1.3e-7 (f32 rounding) |
| c32 fused complex multiply | 526 µs | 191 GB/s | 0 |

MPS GEMM (`MPSMatrixMultiplication`, f32, shared buffers, median of 3, one
warm-up; "8 per submission" amortizes the per-command-buffer cost):

| n | one submission | 8 per submission |
| --- | ---: | ---: |
| 1024 | 2.2 TFLOP/s | 4.3 TFLOP/s |
| 2048 | 8.8 TFLOP/s | 14.3 TFLOP/s |
| 4096 | 14.6 TFLOP/s | **15.0 TFLOP/s** |

- c32 as the decided four real GEMMs at n = 2048: 4.87 ms for all four encodes
  in one command buffer, i.e. **14.1 TFLOP/s of real work** and exactly the 4x
  cost of one real GEMM — the design's "first the four-GEMM version" is
also numerically verified here (max abs diff 0 against Accelerate for n = 128).
- Both f32 and c32 GEMMs matched the Accelerate reference exactly at n = 128
  (max abs diff 0), so the MPS path is not silently changing values.
- Compared with the issue's MLX reference (32.8 / 34.8 TFLOP/s at n = 4096),
  `MPSMatrixMultiplication`'s default f32 path reaches about 43% of it. That is
  an argument for profiling MPSGraph or a `simdgroup_matrix` GEMM in U5, not a
  reason to skip MPS: it is still ~26x the 0.573 TFLOP/s CPU (SME, 1T) figure
  recorded in the issue.

The numbers validate the device, not this change: nothing in this repository
executes on Metal yet.

## Remaining scope

- U1b engine-domain admission predicate; U1c payload-predicate admission for
  `try_host`-narrowed ops (reductions, BLAS1, einsum, clone/convert); CPU
  outputs allocated in the engine's shared domain; `TypedTensor::host_data()`
  and `host_data_mut()` over shared allocations.
- U2 (`t4a-metal-sys`/`t4a-metal-runtime` in `t4a-gpu-backend`), U3 (MSL
  sources in `t4a-gpu-kernels`), U4–U6 (`MetalContext`/`MetalEngine`, MPS GEMM,
  MSL fusion emitter).

## Review record

- Plan review: `ollama-cloud/glm-5.3-flash` (`glm-brainstormer`, read-only).
  Round 1 `VERDICT: CHANGES` (missing `api-snapshot`/`check-pr-fast` gates,
  unimplementable provider-level domain check, fake-buffer lifetime, missing
  refused-path list, method docs). Round 2 `VERDICT: CHANGES` with only two
  command-table corrections (`--write`, `--dry-run`) outstanding; applied.
- Implementation review: `reviewer-gpt` (`gpt-6.1-sol`), read-only, on the
  finished diff. `VERDICT: CHANGES`, two IMPORTANT findings, both addressed:
  (1) `shared_view_read_slice` truncated the provider's borrow to
  `view.n_elements()`, so a sliced or sparse-strided shared view was rejected
  out of bounds; the borrow is now the whole allocation and `StridedView::new`
  bounds the view, as it already does for a host view — with offset,
  every-other-element and transposed regression tests, and the truncated version
  reproduced to confirm the test catches it; (2) it read the deferral of
  engine-domain identity as accepting a foreign domain. Checked against the
  source: identity is enforced on the supported path by `cpu_input_signature`
  and the existing test in `runtime_adapter/tests.rs`, which the earlier
  wording failed to mention; the one unguarded surface is the direct `TensorOp`
  method path, now named with U1b as the fix. The test that pinned foreign
  admission on that internal path was removed rather than kept as an accepted
  deviation. Also applied: `docs/design/storage-ownership-contracts.md` G1
  records the shared-borrow exception as its own transition row (after the
  remaining rows, as the second pass asked) and qualifies the "host access goes
  through G1 guards" wording; the contradicted "already refused by the
  provider" sentence is gone. Reduction admission (U1c) stands as a follow-up
  with its pinned refusal test.
