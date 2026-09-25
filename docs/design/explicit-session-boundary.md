# Explicit Backend Session Boundary

Status: design (pre-implementation). This document records the target contract,
the verified source inventory at the stated baseline, and the migration and
enforcement plan for issue
[#1926](https://github.com/tensor4all/tenferro-rs/issues/1926), the session
workstream of the [#1929](https://github.com/tensor4all/tenferro-rs/issues/1929)
umbrella.

Inventory baseline: `a45833d4f` (`origin/main` at the time of writing, the
revision named by #1926).

Related authorities and records:

- [`exec-session.md`](./exec-session.md) — what a session *is* and how each
  backend maps it.
- [`session-oriented-concrete-apis.md`](./session-oriented-concrete-apis.md) —
  the session-explicit concrete surface (#1673). Its "the one-shot API stays
  available" clause is narrowed by this document; see
  [Supersession](#supersession-of-earlier-session-documents).
- [`cpu-session-open-cost.md`](./cpu-session-open-cost.md) — measured per-call
  cost of opening a CPU session.
- [`cpu-backend-execution.md`](./cpu-backend-execution.md) — `CpuOperationEntry`,
  execution-owner/reentrancy contract.
- [`api-and-convention-freeze.md`](./api-and-convention-freeze.md) — why
  removing implicit entry points is in scope before the release freeze closes.

## Problem

`BackendSession` is documented as the execution-time primitive surface and the
`_in`/`_read` session methods exist, but the source still contains several
*distinct* ways for ordinary operations to reach backend execution without the
caller ever naming an execution boundary:

1. **Operation-level public APIs that open a session internally.** Every
   `EagerTensor` method, and the one-shot `Tensor*` trait methods on the
   concrete CPU backend, create a session (or an equivalent permit + pool
   context) per call.
2. **A second CPU entry mechanism that is not a session at all.**
   `CpuBackend`'s `install_with_pool_context*` family installs a permit,
   `CpuOperationEntry`, and a buffer-pool loan, but hands the closure a
   `CpuExecutionContext` + `BufferPool` instead of a `BackendSession`.
3. **Extension owner routes** registered as `execute = ..._owner` that call
   `with_backend_session` on a `&mut B: TensorBackend`, even though a
   borrowed-session sibling already exists.
4. **Generic helpers** that reach a session-opening method on a `&mut B`
   type parameter rather than through a passed `&mut dyn BackendSession`.

The consequences are the ones the umbrella describes: an unbounded number of
fixed per-call entries that cannot be amortized by a caller, and no mechanical
way to tell a legitimate top-level entry from an accidental nested one. The
design is also not uniformly stated — #1673 kept the one-shot surface as a
deliberate convenience, while #1926 requires it to be removed.

## Definitions

- **Execution boundary** — a public entry that is *allowed* to create backend
  execution state (session, permit, pool context, device ordering). Boundaries
  are named, few, documented, and enumerable.
- **Hidden entry** — any other function that creates backend execution state.
- **Session surface** — operations that receive a borrowed
  `&mut dyn BackendSession` (or a concrete session reference) and never create
  one.

## Target contract

R1. A caller opens an execution boundary and passes the borrowed session
through user algorithms and helpers. **A helper that receives a session never
creates another one.**

R2. Operation-level public APIs do not open sessions. The one-shot spelling of
an operation is replaced by `boundary(|session| op_in(session))`, not kept as a
second implicit entry.

R3. The set of boundaries is exactly the allowlist in the audit gate. A crate
is not exempt as a unit; each allowed entry is named individually.

R4. Preserved during the whole migration:
- CPU `Send` bounds on `with_backend_session` (a managed CPU session may run
  the closure on a Rayon worker thread; see
  [`session-oriented-concrete-apis.md`](./session-oriented-concrete-apis.md)).
- Provider exclusion, resource-permit lifetime, and the reentrancy contract
  from [`cpu-backend-execution.md`](./cpu-backend-execution.md).
- GPU device ordering and enqueue-vs-synchronize semantics.
- Typed errors, validation, dtype promotion, broadcasting, and AD semantics —
  value/error parity for every migrated route.
- Cache ownership and prepared-plan lifetime are independent of the session.
  Removing an implicit wrapper must not discard prepared plans, force
  materialization, or change which cache slot a call uses.

R5. Compiled-program execution (`Runtime::run_compiled`) and its segmented
region boundaries remain valid top-level boundaries. They create *compatible
execution regions*, which is a scheduler decision at an explicit entry, not a
nested per-op entry. The same holds for the eager boundaries
(`with_eager_session`, `with_execution_session`,
`with_extension_execution_context`).

## Inventory at `a45833d4f`

Column meanings: **entry** is the mechanism that creates execution state;
**status** is one of `boundary` (allowed), `remove` (operation-level implicit
entry to be deleted), `thread` (helper to receive a session instead of creating
one), or `decide` (needs an explicit recorded decision).

### A. Allowed boundaries

| Public entry | Definition | Entry mechanism | Backends |
|---|---|---|---|
| `BackendSessionHost::with_backend_session` / `with_backend_session_cached` | default bodies `tenferro-tensor/src/backend.rs:3837`, `:3848` → `default_backend_session` `:4218` | session factory | all |
| `CpuBackend` host impl | `tenferro-cpu/src/backend.rs:3791` → `run_backend_session_cached` (`:3742`) | permit + `CpuOperationEntry` + `CpuExecSession` | CPU |
| `CudaBackend` host impl | `tenferro-gpu/src/cubecl/exec_session.rs:728` | `CudaExecSession` + debug entry guard | CUDA/CubeCL |
| `WebGpuBackend` host impl | `tenferro-gpu/src/webgpu/exec_session.rs:265` | `WebGpuExecSession` | WebGPU |
| `EagerRuntime::with_eager_session` | `tenferro-ad/src/eager.rs:988` | forwards to `with_backend_session` | all |
| `EagerRuntime::with_execution_session` | `tenferro-ad/src/eager.rs:1863` | forwards to `with_backend_session` | all |
| `EagerRuntime::with_extension_execution_context` | `tenferro-ad/src/eager.rs:1934` | session + extension cache, one lifetime | all |
| `Runtime::run_compiled` / `run_compiled_values` | `tenferro-runtime/src/runtime/snapshot.rs:1004`, `:1145` | scheduler-owned regions (`exec.rs`, `segment.rs`) | all |

`with_backend_session_cached` is the runtime-cache-aware entry used by the
scheduler and eager paths; `with_backend_session` is the canonical user entry.
Both are boundaries. The distinction between them is *cache access*, not
session existence.

### B. CPU backend operation-level entries

Two mechanisms coexist. `run_backend_session_cached` produces a
`CpuExecSession`; the `install_with_*` family produces a permit plus a
`CpuExecutionContext`/`BufferPool` loan with no session. Both are full
execution entries — `install_with_pool_context_unmarked`
(`tenferro-cpu/src/backend.rs:2720`, alongside `install_with_pool_unmarked`
`:2700`, `install_with_pool` `:2770`, `install_with_pool_context` `:2780`, and
`install_with_indexed_pool_context` `:2790`) acquires admission, constructs
`CpuOperationEntry`, and calls `entry.enter(...)` exactly like the session path.

| Family | Trait impls and sites | Entry | Status | Replacement route |
|---|---|---|---|---|
| BLAS-1 | `impl BackendSession for CpuBackend` `:2919` — `vdot_read` `:2921`, `norm_squared_read` `:2925`, `axpby_read_into_accum` `:2935` | `run_backend_session_cached` | boundary | these **are** session methods; the `CpuBackend`-as-session impl stays |
| Dense contraction | `impl TensorDot for CpuBackend` `:3451` — `dot_general` `:3458`, `dot_general_read` `:3467`, `dot_general_read_into` `:3479`, `dot_general_read_into_accum` `:3492`, `dot_general_with_conj` `:3505` | `run_backend_session_cached` | decide | `BackendSession::dot_general*_read*` on a caller session |
| Dense contraction (cached) | `impl BackendCachedDot for CpuBackend` `:3511` — `dot_general_cached` `:3520`, `dot_general_with_conj_cached` `:3535`, `dot_general_read_into_accum_cached` `:3550`, `grouped_gemm_cached` `:3571` | `run_backend_session_cached(Some(cache))` | decide | `with_backend_session_cached` + session `_cached` methods |
| Elementwise | `impl TensorElementwise for CpuBackend` `:2953` (~30 methods) | `install_with_pool_context{,_unmarked}` | decide | session-provided pool/context |
| Reductions | `impl TensorReduction for CpuBackend` `:3382` (5 methods) | `install_with_pool_context` | decide | session-provided pool/context |
| Indexing | `impl TensorIndexing for CpuBackend` `:3577` (~10 methods) | `install_with_indexed_pool_context` / `install_with_pool_context` | decide | session-provided pool/context + indexed plan cache |
| Fusion | `impl TensorFusion for CpuBackend` `:3882` (3 methods) | `install_with_pool_context` | decide | session-provided pool/context |
| Linalg pool helper | `CpuBackend::with_linalg_pool` `:2830` | `install_with_pool_context` | boundary | documented external-implementation boundary for `tenferro-linalg` |

`impl TensorAnalytic` `:3199` and `impl TensorStructural` `:3281` do not call
`install_with_*`; they are host-side and are not entries.

**Compatibility decision (recorded).** The `impl Tensor* for CpuBackend` blocks
listed as `decide` are public backend SPI. Removing them from the public
surface requires either (a) making the session method the only spelling and
migrating every caller, or (b) keeping one named one-shot boundary per family.
This document records the target as **(a)**: the session surface
(`BackendSession`) becomes the only operation primitive, and
`with_backend_session[_cached]` the only entry. Option (b) is retained only for
`with_linalg_pool`, which exists specifically so an operation-family crate can
own its backend implementation while sharing the CPU pool. That is a named,
documented boundary, not a crate-wide exemption.

Consequence for the audit gate: `install_with_*` becomes private to the
`CpuBackend` boundary implementations themselves, and `run_backend_session_cached`
is called only from `BackendSessionHost`/`BackendSession`.

### C. Eager per-op entries (`tenferro-ad`)

Every public `EagerTensor` operation reaches a session through this chain. The
public entry points are `eager_ops.rs` methods such as `add` `:161`, `mul`
`:202`, `reduce_sum` `:278`, `dot_general` `:335`, `matmul` `:544`, `transpose`
`:595`, `reshape` `:634`.

| Site | Entry | Status | Replacement route |
|---|---|---|---|
| `eager_exec.rs:461` in `exec_standard_op_on_tensor_reads` (`:450`) | `backend.with_backend_session` | thread | existing `exec_standard_op_on_tensor_reads_in_session` (`:467`) |
| `eager_exec.rs:261` in `exec_dot_general_with_conj_on_tensor_reads` (`:254`) | `backend.with_backend_session` | thread | session `dot_general_with_conj_read` |
| `eager_exec.rs:319` in `exec_op_on_tensor_reads_with_runtime` (`:302`) | `backend.with_backend_session` | thread | session `to_contiguous_read` / `concrete_tensor_reads` |
| `eager_exec.rs:733` in `exec_standard_op_on_tensors` (`:726`) | `backend.with_backend_session` | thread | session variant already exists inline |
| `eager.rs:2009` `exec_outputs_read` → `eager.rs:4456` `exec_single_output_read` | transitively opens a session per eager op | remove | `with_eager_session` / `with_execution_session` + session op |
| `eager.rs:2548`, `:2556` in `EagerRuntime::store_grads` | `backend.with_backend_session` | decide | AD-boundary session (see below) |
| `eager.rs:2032` `exec_standard_graph_outputs` | `backend.with_backend_session` | thread | test-only (`#[cfg(test)]`) |
| `eager_backend.rs:430`, `:581` | `with_backend_session` dispatch wrappers | boundary | these are the backend-erasure host plumbing |

**Eager AD decision (explicit, not an exemption).** The eager AD surface
(`EagerTensor` in `eager_ops.rs`, `store_grads`) is migrated the same way as
non-AD eager code: the AD boundary is the existing
`with_eager_session`/`with_extension_execution_context` entry, and the
differentiation entry point documents that it is the boundary for a whole
forward/backward group. The migration must therefore establish that gradient
accumulation in `store_grads` runs inside one boundary rather than one session
per slot. If a single boundary for a forward/backward group cannot be expressed
without changing AD value lifetime, that is a scope change and goes back to the
issue — it is not silently exempted.

### D. Extension owner routes

Each `*_owner` route has a borrowed-session sibling already. The `_owner`
variant is registered as the extension's `execute`/`execute_reads` fallback,
which is why it still opens sessions.

| Crate | Owner route (entry) | Session sibling | Status |
|---|---|---|---|
| `tenferro-fft` | `execute_fft_extension_reads_owner` `lib.rs:1462` (registration `:1547`–`:1548`) | `execute_fft_extension_reads_session` `:1473` → `_on_session` `:1514` | remove/thread |
| `tenferro-linalg` | `execute_linalg_extension_reads_owner` `extension.rs:904` (registration `:1118`–`:1119`) | `execute_linalg_extension_reads` `:896` → `_on_session` `:916` | remove/thread |
| `tenferro-einsum` | `execute_einsum_extension` `extension.rs:1021`, `execute_einsum_extension_reads` `:1050` (registration `:1015`–`:1016`) | `execute_einsum_extension_session_reads` `:1089` | remove/thread |
| `tenferro-linalg` helper | `matmul_preserve_trailing_batch` `tensor_ext.rs:2812` (`dot_general_read` `:2824`), `linalg_matmul_read` `:2831` (`:2846`) call `backend.dot_general_read(...)` on `&mut B: LinalgBackend` | session `dot_general_read` | thread |

The registration macro already supports a session-capable route
(`execute_..._session`); the migration is to make the session route the primary
one and keep the owner route only as the scheduler's explicit region fallback,
with the region construction owned by the runtime rather than by the extension.

### E. GPU backend entries

`CudaBackend` and `WebGpuBackend` implement the `Tensor*` operation traits by
calling the *same* functions as their `BackendSession` impls. There is no
distinct one-shot session struct: the backend is effectively its own session.

| Backend | `Tensor*` impls | `BackendSession` impl | Status |
|---|---|---|---|
| CUDA/CubeCL | `TensorElementwise` `cubecl/mod.rs:4719`, `TensorAnalytic` `:5601`, `TensorStructural` `:5879`, `TensorReduction` `:6669`, `TensorDot` `:6902`, `TensorIndexing` `:6959`, `TensorFusion` `:7895` | `:8109` | decide |
| WebGPU | `TensorElementwise` `webgpu/mod.rs:758`, `TensorAnalytic` `:836`, `TensorStructural` `:878`, `TensorReduction` `:952`, `TensorDot` `:970`, `TensorIndexing` `:992`, `TensorFusion` `:1047` | `:1077` | decide |

The GPU-specific risk is different from CPU: a one-shot `Tensor*` call on the
GPU backend does not *open* a session, it runs unbatched. The migration question
is therefore whether the unbatched spelling is a boundary or a hidden entry. This
document records it as **a hidden entry**: it must be spelled as
`with_backend_session(|s| op_in(s))` so that the ordering/lifetime contract is
explicit at the call site. Nested-entry detection for the GPU overrides is
currently debug-only (the portable in-session guard); release-mode enforcement
is a follow-up recorded in
[`session-oriented-concrete-apis.md`](./session-oriented-concrete-apis.md).

### F. Explicitly not entries

- `Runtime::run_compiled` region construction and the scheduler's session
  regions (`exec.rs:633`–`:796`, `segment.rs:313`–`:750`).
- `tenferro-fft`'s plan/cache types and `*_on_session` bodies.
- `tenferro-ad`'s `to_contiguous_host_read` (`eager.rs:1881`), which is
  deliberately session-free and returns `None` when no such path exists.
- Device transfer, allocation-domain, and buffer-retirement helpers, which do
  not execute tensor-sized ops.

## Migration slices

Ordered so each slice is independently reviewable and revertible, and so the
gate can land after the last hidden entry is gone.

1. **Design + inventory** (this document). No behavior change.
2. **Eager threading.** Convert `eager_exec` helpers to session-taking
   (`*_in_session`) and route every public `EagerTensor` operation through the
   caller's boundary. Add nested-entry tests; verify value/error parity on the
   eager AD paths. Highest volume, and the precondition for removing the
   per-op entry.
3. **Extension owner routes.** Make the `_session` route primary; keep the owner
   route as an explicit runtime region fallback with the region formed by the
   runtime. Cover einsum, linalg (including `tensor_ext.rs` helpers), and FFT.
4. **CPU backend SPI.** Collapse `install_with_*` onto the session-provided
   pool/context and reduce the `impl Tensor* for CpuBackend` blocks to session
   methods (or the recorded named boundary). Migrate `tenferro-linalg`'s
   `LinalgBackend`-generic helpers.
5. **GPU backend.** Apply the same split to `CudaBackend`/`WebGpuBackend`; add
   release-mode nested-entry enforcement for the overrides.
6. **Audit gate.** Add the deterministic check and its allowlist (below), then
   delete the removed public spellings.
7. **Measurement.** The matched one-shot / `with_execution_scope` /
   one-shared-session comparison (below), reported separately from the API
   change.

Slices 2–5 may be split further per crate, but each must land with its own
nested-entry test and parity evidence. 6 depends on 2–5.

## Audit gate design

The gate must reject *session factories in operation or helper implementations*,
not merely the text `with_backend_session`. Requirements taken from #1926 and
made concrete here:

- **Allowlist by exact source location**, not by crate or by regex. Each allowed
  entry is one function body; adding one requires a reviewed edit to the
  allowlist file.
- **Indirect detection.** The check must follow the entry *mechanism*, not only
  the public name: `run_backend_session_cached`, `install_with_pool_context*`,
  `install_with_indexed_pool_context*`, `default_backend_session`,
  `with_backend_session[_cached]`, and the GPU exec-session constructors. A
  helper that only calls an alias must still fail.
- **Exclusions that are not exemptions.** Test/bench/example code, and the
  bodies of the allowed boundary implementations themselves, are excluded
  because they *are* the boundary or are non-library code — the exclusion is
  source-location-scoped, and moving the code into the library must fail.
- **Reachability sanity.** The gate must include at least one indirect/aliased
  case in its own tests so it cannot silently degrade to a text grep, plus a
  negative test that a renamed alias is still caught.
- Integration follows the existing repository-rules review path
  (`scripts/repository-rules-review.py` routing, `REPOSITORY_RULES.md`
  section). No new CI workflow.
- The gate text must be mirrored by a normative bullet in `REPOSITORY_RULES.md`
  so the rule has one owner.

The gate cannot land before slice 5, because at `a45833d4f` it would fail on
the inventory in sections B–E.

## Measurement protocol

Removing syntax does not by itself save time; #1926 requires measurement
separately. The comparison must hold kernel, cache state, thread count, and
output contract fixed, and report the session-entry count alongside timing:

- cases: (i) current one-shot per-op entry, (ii) `with_execution_scope`,
  (iii) one shared session across the chain;
- same kernels, same warm cache state, same explicit CPU thread count with the
  effective thread count recorded (single-worker backend for the overhead
  baseline, per the umbrella's 1T requirement);
- report session-entry count, executor-admission time, and body time separately;
- GPU: report enqueue vs synchronized completion distinctly.

Do not attribute an executor-admission saving to the API change. Benchmarks
themselves live in `tenferro-benchmark` (#107) and
`strided-rs-benchmark-suite` (#41); this document only fixes the protocol that
this repository's claims must follow.

## Supersession of earlier session documents

`session-oriented-concrete-apis.md` (#1673) states "The one-shot API stays
available and becomes a thin wrapper" and lists "the one-shot API is not broken
for aesthetic consistency" as a non-goal. That was correct for its own scope
(adding a session-explicit surface). This document supersedes those two clauses
for **operation-level session-opening APIs**: #1926 removes them, because a
thin wrapper that opens a session per call is still an implicit entry. What
#1673 established and remains unchanged: the shape of the session surface
(`_in`/`_read` methods, `&mut dyn BackendSession`, preserved `Send` bounds), the
nested-entry prohibition, and cache parity.

`exec-session.md` remains the owner for "what a session is". Its
`CpuBackend::with_backend_session` code sample predates the permit-based
`run_backend_session_cached` implementation and was corrected in the same
change that added this document.

## Non-goals

- No new public API, backend, dependency, feature flag, or cache.
- No universal execution pipeline and no second operation registry.
- No claim that the migration improves performance without the measurement
  above.
- No change to `Runtime::run_compiled` region formation, GPU device ordering,
  AD semantics, or numerical behavior.
- No removal of the session surface itself, and no session handle inside
  `Tensor`/`TypedTensor`/`EagerTensor` values.

## Residual risks

- **Scope.** Slices 2–5 touch the hottest execution paths in three crates. Each
  slice must be independently revertible; a combined multi-crate rewrite would
  put numerical and AD behavior at risk with no smaller fallback.
- **Eager AD boundary granularity.** Whether one boundary per forward/backward
  group is expressible without changing AD value lifetime is unproven until
  slice 2 is attempted.
- **Test churn.** 740 source occurrences of `with_backend_session` exist, most
  in tests/benches/examples. Removing one-shot spellings will rewrite many
  test call sites; the replacement must keep the tests meaningful rather than
  mechanically wrapping each call in its own boundary.
- **GPU release-mode enforcement** stays debug-only until the follow-up in
  #1673's record lands.

## B2 work list: which operation methods invert, and what must not change

Derived by parsing the trait declarations in
`crates/tenferro-tensor/src/backend.rs` and the `impl Trait for Owner` blocks
for `CpuBackend`, `CpuExecSession`, `CudaBackend`, and `WebGpuBackend`. The
parse expands the method-generating macros (`delegate_with_pool_context!`,
`delegate_with_pool!`) used by `CpuExecSession`, so the counts below are not
`fn`-line counts.

### Invertible pairs: 31

A method pair inverts only when both halves exist and the read half is the
provided one. Pairs: `TensorElementwise` 13 (`add`/`sub`/`mul`/`neg`/`conj`/
`div`/`abs`/`sign`/`maximum`/`minimum`/`compare`/`select`/`clamp`),
`TensorAnalytic` 10 (`exp`/`log`/`sin`/`cos`/`tanh`/`sqrt`/`rsqrt`/`pow`/
`expm1`/`log1p`), `TensorStructural` 3 (`transpose`/`reshape`/
`broadcast_in_dim`), `TensorReduction` 4 (`reduce_sum`/`reduce_prod`/
`reduce_max`/`reduce_min`), `TensorDot` 1 (`dot_general`).

### Not invertible: 13 required one-shots with no `_read` sibling

`TensorStructural`: `cast`, `extract_diagonal`, `embed_diagonal`, `tril`,
`triu`. `TensorIndexing`: `gather`, `scatter`, `slice`, `dynamic_slice`,
`dynamic_update_slice`, `pad`, `concatenate`, `reverse`.

These have a single spelling today. `TensorIndexing`'s methods are already
session-safe — a session implementation does not open an entry — so they are
not hidden entries, and adding `_read` siblings would be a **new capability**,
not a migration. They stay as they are.

### Per-backend `_read` coverage

| Backend object | of the 31 `_read` methods | note |
|---|---|---|
| `CpuBackend` | 31 present | none relies on the default |
| `CpuExecSession` | 31 present | many generated by the delegation macros |
| `CudaBackend` | 31 present | none relies on the default |
| `WebGpuBackend` | **0 present** | implements only the required one-shots, mostly `unsupported!`; `dot_general`/`dot_general_with_conj` are real |

So requiring `_read` is free for CPU and CUDA and needs 31 explicit
implementations for WebGPU.

### The default `_read` semantics are a tested contract, not an accident

`crates/tenferro-tensor/src/tests/backend_default_read_tests.rs` (1916 lines)
pins the current defaults. Its central test,
`default_read_methods_delegate_owned_tensors_and_reject_views`, asserts two
behaviours for a backend that implements only the one-shot methods:

1. `TensorRead::Tensor` inputs delegate to the one-shot method (the test
   asserts `calls` contains `"add"` and `"dot_general"`);
2. `TensorRead::View` inputs are **rejected** with an error whose message
   contains `"borrowed tensor views"` (the elementwise/analytic/structural and
   `TensorDot` defaults instead materialize through `read_tensor`).

Deleting a default therefore deletes a tested contract, and the view policy
(accept-and-materialize versus reject-with-`Unsupported`) becomes the
implementing backend's explicit responsibility. "Do not change the acceptance
surface" means: for every backend and every one of the 31 operations, views
must be accepted or rejected exactly as the old default plus the backend's
override did. The affected test backends
(`DefaultReadBackend`, `DefaultOnlyBackend`, `DefaultOnlyExec`,
`DefaultOnlyLinalgBackend`, and the per-crate `macro_rules!` test backends)
must migrate with it.

### Staging

**Step 1 — require `_read`, keep every caller working.** Delete the 31 provided
defaults and make the read halves required. CPU and CUDA need no new code. For
WebGPU, add 31 implementations that reproduce the old default exactly:
`unsupported!` where the old chain ended in the one-shot's `unsupported!`, and
the old default body (materialize, then call the operation) where the operation
is real, which is `dot_general_read`. Migrate the default-contract tests to
assert the explicit implementations instead.
This step is mechanical, leaves the tree compiling, changes no caller, and is
the precondition for deletion. It must cover all 31 at once because the
default-contract test file is organized around the defaults as a set.

**Step 2 — delete the one-shot methods per family and migrate callers.**
Compile errors are the work list. Per family, the deletion also forces the
generic helpers that called the one-shot on a `&mut B` to become
session-taking, so each family slice carries the part of B3 it triggers. The
pilot is `TensorDot` (one pair): `dot_general` has 33 non-test call sites in 19
files, 113 test/bench/example call sites, and 2 doctest call sites, and the
provided `dot_general_read`, `dot_general_cached`, `dot_general_read_into`,
`dot_general_read_into_accum`, and `dot_general_with_conj[_read]` bodies all
reference it.

**Ordering constraint.** Deletion cannot be staged per family before Step 1
lands: while a read half is still provided, deleting its one-shot sibling makes
the default recurse into itself.
