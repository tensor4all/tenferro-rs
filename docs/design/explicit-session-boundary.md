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
| CUDA/CubeCL | `TensorElementwise` `cubecl/mod.rs:4719`, `TensorAnalytic` `:5601`, `TensorStructural` `:5879`, `TensorReduction` `:6669`, `TensorDot` `:6902`, `TensorIndexing` `:6959`, `TensorFusion` `:7895` | `:8109` | remove the owner impls |
| WebGPU | `TensorElementwise` `webgpu/mod.rs:758`, `TensorAnalytic` `:836`, `TensorStructural` `:878`, `TensorReduction` `:952`, `TensorDot` `:970`, `TensorIndexing` `:992`, `TensorFusion` `:1047` | `:1077` | remove the owner impls |

**The `delegate!` shim is transitional and goes with them.** Both GPU exec
sessions implement the operation traits with a `delegate!` macro whose bodies are
`self.backend.<method>(..)` (`cubecl/exec_session.rs:444`,
`webgpu/exec_session.rs:76`), so the session is a thin forwarder and the real
implementation lives on the owner. That is precisely the owner-as-session shape
B3 removes, and it cannot survive the removal of the owner impls: once
`impl TensorElementwise for CudaBackend` is gone, the shim has nothing to forward
to.

Decision: B3(b) deletes the shim together with the owner impls, and the GPU
operation bodies are expressed **once** as module-level functions over the
backend — the shape `gemm::*`, `structural::*` and `dispatch::*` already use —
with the session methods calling those functions. The session stays the only
entry, the owner keeps only `BackendRuntimeCache`, `TensorDeviceTransfer` and
`BackendSessionHost`, and no implementation is duplicated between an owner trait
impl and a session trait impl.

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

The single documented exception to the entry inventory is
`with_cpu_exec_session` (`crates/tenferro-cpu`): it is a capability bridge for
crates that must run a CPU-specific step inside a session they already own, not
an alternative execution entry. It is retained deliberately on the
out-of-scope list of #1929 (together with `with_backend_session` returning
`Result<R>`, and release-mode nested-entry detection on the GPU), and any new use
of it must justify why the ordinary session route does not apply. Every other
entry below is excluded because it is *not* an operation entry at all.

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

### Audit gate: implemented

`scripts/audit-session-entry.py` implements the gate described above, earlier
than slice 5 because it is useful as a *freeze* rather than as a cleanup: it
records today's legitimate entry functions in
`scripts/session-entry-allowlist.json` and fails on any new one.

- Mechanisms tracked: `default_backend_session`, `with_session_entry_guard`,
  `install_with_pool_context[_fresh]`,
  `install_with_indexed_pool_context[_unmarked]`, `run_backend_session_cached`,
  and `CpuExecSession` / `CudaExecSession` / `WebGpuExecSession` construction.
- The allowlist is keyed by mechanism and holds `path::function` entries, so a
  reviewed edit is required to add one, and a removal is recorded with
  `--bless`.
- Aliases are resolved (`use ...::{Type as Alias}`, `use ... as alias`), so a
  renamed import is still reported; `--check` runs the negative tests (aliased
  import, unrelated struct literal, allowlisted boundary, method call) on every
  invocation, so the checker cannot silently degrade into a text grep.
- Library scope only: `tests/`, `benches/` and `examples/` are excluded as the
  boundary's users, and the exclusion is path-based rather than symbol-based.
- `scripts/check-pr-fast.sh` runs the audit for code changes, and
  `REPOSITORY_RULES.md` names it as the single owner of the rule (routed by
  `scripts/repository-rules-review.py` for backend/session paths).

The frozen inventory started at 76 `path::function` entries, dominated by the CPU
owner entries that B3 removes (`install_with_pool_context` in 27 functions,
`run_backend_session_cached` in 14). B3's commits shrank it to **16**, and any
entry that reappears outside the allowlist fails the local gate:

| Mechanism | Entries |
| --- | --- |
| `CpuExecSession construction` | 2 |
| `CudaExecSession construction` | 1 |
| `WebGpuExecSession construction` | 1 |
| `install_with_pool_context` | 3 |
| `run_backend_session_cached` | 3 |
| `with_session_entry_guard` | 6 |
| `default_backend_session`, `install_with_indexed_pool_context*`, `install_with_pool_context_fresh` | 0 |

The zero rows are kept deliberately: the mechanism stays tracked, so one
reappearing entry is reported rather than silently untracked.

#### Demonstration: a hidden entry fails, the boundary implementations do not

Run on the final tree. With an aliased hidden entry appended to a library file
(`use run_backend_session_cached as __audit_demo_alias;` plus a new private
function calling it), the gate reports and exits non-zero:

```text
self-test passed: alias, boundary, method and grouped-import cases
run_backend_session_cached: unallowlisted session entry at crates/tenferro-tensor/src/backend.rs::audit_demo_hidden_entry
exit=1
```

On the pristine tree — where the same mechanism is reached only by the
allowlisted boundary implementations — the same command reports no finding and
exits zero:

```text
self-test passed: alias, boundary, method and grouped-import cases
exit=0
```

The temporary entry is not in the tree; the demonstration is reproducible by
appending those two lines and re-running `python3 scripts/audit-session-entry.py
--check`.

### B3(iii) blocks on an acceptance-surface decision

A third attempt deleted the eight CPU owner implementations and migrated the
resulting 317 errors (about 140 wrapped mechanically, the cached family moved to
`with_backend_session_cached`, and the contract tests retargeted to
`exec_session.rs`). The CPU crate's own suite passed, and the session-entry
allowlist shrank from 76 entries to 17, which is the shape B3(iii) is supposed to
have.

It then failed in `tenferro-ad`, and the reason is a design question rather than
a migration gap:

* `eager::tests::untracked_nary_ops_consume_lazy_views_without_materializing_inputs`
  reduces a lazy view and now sees `Unsupported { op: "reduce_sum", message:
  "backend does not accept borrowed tensor views at this execution boundary" }`.
  The ad layer reached the CPU *owner*, whose read half materializes a view; the
  CPU *session* deliberately rejects one instead. Both behaviours are documented
  (the CPU session's read halves reject borrowed views, the owner's accepted
  them), and Step-1's contract test pins the session behaviour.
* `eager::tests::eager_backend_session_identity_projects_to_owner` compares
  session identities and now sees the same id on both sides, because the ad
  layer's arrangement assumes the owner is its own session.

So "one implementation set on the session" is not behaviour-preserving for
consumers that relied on the owner's wider acceptance: either the session's
accepted input surface is widened to the owner's (changing the session contract
that Step 1 pinned and that the session-route benchmark measures), or those
consumers materialize explicitly before the call (a real change in the ad layer).
The issue's direction resolves it: #1926 keeps the *session* interface, and the
session's read halves are the contract Step 1 pinned and that
`docs/testing/session-route-baseline.json` measures, so the session's accepted
input surface is authoritative and the consumers that relied on the owner's
materialization have to materialize explicitly (the ad layer already has
`to_contiguous_read` for exactly that). That is a behaviour-preserving migration
of the consumer, not a widening of the session contract, and it is the path
B3(iii) should take. The `eager_backend_session_identity_projects_to_owner` test
then needs to state the new arrangement (the owner is the session provider, not a
session).

The attempt was reverted because that consumer migration is a real change of its
own and had not been reviewed yet; the tree stays at B3(i)+(ii)+B4.

The rest of the CPU-side work is then mechanical: the eight `impl ... for
CpuBackend` blocks, the now-dead `CpuBackendSessionMarker` and
`install_with_indexed_pool_context*`, the source-text contract retargets to
`exec_session.rs`, and the allowlist re-bless.

### B3(iii): the exact consumer that needs the decision

A fourth attempt reached the same point with the CPU-side work complete (the eight
owner impls, the dead marker and the indexed pool helpers deleted; the
session-entry allowlist down to 17 entries; the three source-text contracts
retargeted to `exec_session.rs`; the CPU suite green) and stopped in
`tenferro-ad` again. The failing path is now pinned:

* `EagerTensor::reduce_sum` builds `StdTensorOp::ReduceSum` and runs it through
  the untracked eager path, which reached the CPU owner; the owner's
  `reduce_sum_read` materialized a borrowed view before delegating, while the CPU
  session's `reduce_sum_read` rejects one (`crates/tenferro-ad/src/eager_ops.rs`
  `unary_op` → the untracked n-ary execution → the backend entry). The traced
  path is unaffected because the runtime materializes slots before
  `eager_exec.rs` calls `exec.reduce_sum_read` (`:524`).
* `eager_backend_session_identity_projects_to_owner` compares session identities
  and assumes the owner is its own session.

So the consumer-side fix is not a single mechanical edit; it is a policy choice
about *where* the ad layer materializes: always, on the untracked path only, or
per operation (only for the read halves whose session contract rejects views,
which is the reductions and the same family the trait's provided defaults
reject). Each option has a different cost profile for the untracked path, which is
exactly what the Phase-C baseline measures, so it should be decided with the
issue rather than picked inside a migration slice.

### B3(iii) status: CPU and WebGPU done, CUDA outlined

The earlier "acceptance surface" blocker was a misdiagnosis and is resolved:
the failure came from routing the composite `EagerBackend`'s
`BackendSessionHost::with_backend_session` to `with_session_entry_guard(||
f(self))`. The enum is not a session; forwarding to the concrete backend's
session (as before) keeps the untracked eager path on the same accepted input
surface. With that, B3(iii) needed no consumer policy change at all.

Landed so far:

* **CPU** (`007da98a7`): the eight owner impls, the dead marker and the indexed
  pool helpers are gone, ~380 call sites migrated, three source-text contracts
  retargeted, the allowlist down to 16 entries, workspace tests green.
* **WebGPU** (`ad6b41720`): the operation bodies moved from
  `impl Tensor* for WebGpuBackend` into `impl Tensor* for WebGpuExecSession<'_>`
  (the `delegate!` invocations for the operation families became real impls),
  the owner's `BackendSession`/`BackendCachedDot` impls and the session marker
  deleted, and `SessionCachedDot` implemented directly on the session (WebGPU has
  no runtime cache, so the trait defaults are what the owner's blanket impl
  provided). Verified with and without the `webgpu` feature.

**CUDA** is the remaining half, and the procedure is now known:

1. move each `impl Tensor* for CudaBackend` body into
   `impl Tensor* for CudaExecSession<'_>`, rewriting receivers to `self.backend`
   *and* `self` passed as an argument (`structural::transpose(self, ..)`,
   `gemm::dot_general(self, ..)`, `promotion::*(self, ..)`), which is the part a
   naive rewrite misses;
2. add the imports the moved bodies need in `cubecl/exec_session.rs`
   (`dispatch`, `elementwise`, `gemm`, `permutation`, `fusion`, the promotion
   helpers, `DType`, ...);
3. fix the E0599 calls that were owner methods (`to_contiguous_read`,
   `dot_general_with_conj`, ...) to the session form, and add
   `impl SessionCachedDot for CudaExecSession<'_> {}` in place of
   `delegate_cached!`;
4. delete `impl BackendSession for CudaBackend`/`BackendCachedDot for CudaBackend`
   and the marker, then retarget the CUDA source-text contracts
   (`cuda_launch_contract`, `public_surface_contract`, `session_contract`,
   `backend_read_contract`) that name `cubecl/mod.rs` sections.

Attempted twice and reverted. What the attempts measured, in addition to the
1113 errors of a move-only pass (572 E0425, 302 E0599, 201 E0433, 36 E0277):

* the receiver rewrite must also handle `self` followed by a *newline* before the
  dot (about 130 sites per pass; a literal `self.` rewrite misses them), and `self`
  passed as a bare argument (`structural::transpose(self, ..)`);
* the moved bodies invoke the `dispatch::launch_*` macros, which expand at the
  call site, so `cubecl/exec_session.rs` needs the macro bodies' names in scope
  (`CubeclCudaRuntime`, `ArrayArg`, `ComputeClient`, the promotion helpers, ...),
  not just the module path;
* the CUDA-specific `*_typed` helpers (`gather_typed`, `dynamic_slice_typed`,
  `scatter_float_typed`, ...) are inherent `impl CudaBackend` methods in
  `cubecl/mod.rs` and stay; only the trait entries move;
* `delegate_cached!` becomes `impl SessionCachedDot for CudaExecSession<'_> {}`
  like WebGPU's, since the CUDA owner's cached behaviour was the trait default
  plus the provider's own caching.

**The module-function shape landed, and it is what the attempt history argued.
The CUDA operation bodies now live in `crates/tenferro-gpu/src/cubecl/ops.rs` as
57 free functions over `&mut CudaBackend`, the seven owner impls and the owner
`BackendSession`/`BackendCachedDot` impls are deleted, and `exec_session.rs`
keeps one delegation macro per direction:

1. each owner method became `pub(super) fn <op>(backend: &mut CudaBackend,
   <args>) -> crate::Result<...> { <body> }` with a plain `self` -> `backend`
   rename, which is also what keeps `self`-taking macro invocations valid;
2. `delegate_ops!` generates the session impls and forwards
   `ops::<op>(self.backend, <args>)`, so the session's method table still comes
   from one list per family;
3. `delegate!` continues to serve the families that stay on the owner
   (`TensorBuffer`, `TensorDeviceTransfer`);
4. `impl SessionCachedDot for CudaExecSession<'_>` keeps the two device-path
   `_read_cached` overrides and takes the trait defaults for the rest, matching
   WebGPU.

The receiver rename is mechanical and reviewable per function, and step 2 keeps
one implementation per operation, which is what B3 requires. The measured
fallout supports the choice: the in-place attempts produced 1113 errors, while
this shape left six — four `to_contiguous_read` calls in `blas1.rs` and an
inherent helper, the owner `BackendCachedDot` impl, and one helper that needs a
session-typed receiver.

Two traps worth recording, both caught by the compiler rather than by review:

* `TensorElementwise::elementwise_read_into` cannot move to a free function: its
  allocating fallback is generic over `TensorElementwise`, which only the session
  implements once the owner impl is gone. It stays a hand-written session method,
  and `delegate_ops!` grew an optional `override { ... }` group for it;
* that first version silently dropped the native read-into kernels, which showed
  up only as three `never used` warnings (`elementwise_read_into_native` and two
  launch helpers). The override now tries the native path first and falls back to
  the allocating helper, so no work moved onto the allocating path.

The CUDA tests and example migrated through the same wrapper as CPU and WebGPU.
Two shapes needed hand work: `upload(&gpu, ..)` inside a session closure borrows
`gpu` immutably while the session needs it mutably (E0502), so the upload is
hoisted into a `let` before the enclosing statement, and two object-safety tests
that wrote `let exec: &mut dyn BackendSession = &mut gpu;` now obtain the erased
session from `with_backend_session`, which still proves the same interface.

The source-text contracts were retargeted to the module that now holds the
bodies: `cubecl_launch_contract` (eight anchors), `public_surface_contract` (two)
and `backend_read_contract`, whose session half became a compile-time assertion
that `CudaExecSession<'static>` implements all six operation traits — the same
shape `webgpu_backend_contract` uses, and stronger than the source scan it
replaced. `backend_read_contract` is now `#![cfg(feature = "cuda")]`, because
that assertion names the CUDA session type.

Verification on this host: `cargo check --workspace --all-targets` and
`cargo check -p tenferro-gpu --features cuda --all-targets` are error- and
warning-free; the CUDA suite reports 108 passed / 189 ignored in the lib and 73
passed in the integration target; the WebGPU suite reports 89 + 31 + 3 + 2
passed; and `scripts/audit-session-entry.py --check` stays green without an
allowlist change. What this host cannot do is run the hardware-gated kernels, so
their behaviour remains with the hosted GPU matrix; the bodies themselves are
verbatim moves, and no receiver rewrite changes an argument.

### Local gate state (mid-Phase-B)

`bash scripts/check-pr-fast.sh --no-fetch --coverage-reviewed --test 'cargo test
-p tenferro-gpu --features webgpu --lib --tests'` passes on the current branch
after the fixes it surfaced, which are worth remembering:

* the standalone `ext/tenferro-cpu-tblis` manifest is outside the root workspace,
  so its provider test kept the deleted one-shot spelling;
* `docs/guides/devices-and-gpu.md` is generated from a snippet source and was
  stale for the same reason (`check-doc-snippets.py` syncs it);
* clippy (`-D warnings`) caught seven orphaned `# Errors` doc blocks left behind
  by deleted trait items and the needless `&` borrows the migration introduced.

Run it with `RUSTC_WRAPPER=""` locally: the kache wrapper's path remapping breaks
trybuild `.stderr` comparisons, which is also why the pre-existing
`tenferro-ad::eager_backend_capability_contract` fixture mismatches here while the
`tenferro-gpu` session contracts pass. That fixture mismatch is span-only: the
same `E0432` is reported, with a wider underline than the recorded `.stderr`, so it
is a rustc-rendering difference on this toolchain rather than a behaviour change.

Two other fixtures of that contract *were* invalidated by this refactor and are
blessed with it. Both reported `E0308`/`E0576` for the removed owner-projection
APIs, and the only change in their `.stderr` is the removal of the compiler's
suggestion that the expected type could become a session, which is false now that
the owner is not a session. The third, span-only fixture is deliberately left
untouched, so after this change the contract reports exactly the same single
mismatch it reports on the baseline worktree — the pre-existing claim above is a
comparison, not an assumption.

### Phase-B completion audit

Each Phase-B requirement mapped to the artifact that satisfies it. "Evidence"
means a file, a command output, or a test that runs in CI, not an intention.

| Requirement | Evidence |
| --- | --- |
| B1 fail: owner `add`/`add_read` on three backends | `tenferro-cpu/src/lib.rs` crate docs pin the deleted owner spellings for `add`, `mul`, `exp`, `reduce_sum`, `transpose`, `dot_general`; `cubecl/exec_session.rs` and `webgpu/mod.rs` pin that the CUDA/WebGPU owners do not implement an operation trait (a call cannot resolve for the same reason) |
| B1 fail: `dyn BackendSession` old `add` | `tenferro-tensor/src/backend.rs::BackendSession` `compile_fail` example |
| B1 fail: `BackendCachedDot` | `tenferro-cpu/src/lib.rs` owner-bound `compile_fail` example |
| B1 fail: `default_backend_session` | `tenferro-tensor/src/backend.rs::BackendSessionHost` `compile_fail` example |
| B1 pass: `Tensor::add(.., session)`, cached operations, typed-view operations, scheduler/extension `dyn BackendSession`, scope nesting | five trybuild fixtures in `tenferro-runtime/tests/ui/session_surface/pass/`, driven by `session_surface_contract.rs`, which runs under the CI nextest profile (verified with `cargo nextest run -p tenferro-runtime --test session_surface_contract`) |
| B2 delegation inverted, one-shot spelling deleted | 31 per-operation deletion commits; `_read`/`_into` are required items |
| B2 acceptance ranges unchanged | `backend_default_read_tests.rs`: `default_read_methods_delegate_owned_tensors_and_reject_views`, `structural_runtime_materialization_rejects_views_by_default` |
| B3 supertraits trimmed | `pub trait TensorBackend: BackendRuntimeCache + TensorDeviceTransfer + BackendSessionHost` |
| B3 one operation implementation set per backend | CPU `CpuExecSession`; CUDA `ops.rs` bodies with `delegate_ops!`; WebGPU real impls on `WebGpuExecSession` |
| B3 deletions | owner impls, the non-public `install_with_pool_context*`, every `BackendCachedDot` impl, `default_backend_session`, the backend-as-session markers and factory |
| B3 CPU linalg policy preserved | `preferred_linalg_mode` consumed at `tenferro-cpu/src/backend.rs:2752`, defined in `provider.rs` |
| B3 GPU `delegate!` shim decided and documented | CUDA operations moved to `ops.rs` and deleted from the shim; `delegate!` remains only for `TensorBuffer`/`TensorDeviceTransfer` |
| B3 generic-bound and call-site migration (~215 sites, doctests, examples) | `cargo check --workspace --all-targets` is clean, which is the machine-checkable form of the migration |
| B4 extension owner path is session-primary | `define_extension_runtime!`'s owner entry forms the session and hands extensions `&mut dyn BackendSession`; owner routes are limited to the runtime-formed fallback (`execute_owner_extension_fallback`); the linalg helper is session-typed |
| B5 deterministic audit gate | `scripts/audit-session-entry.py` + `scripts/session-entry-allowlist.json`, run by `scripts/check-pr-fast.sh`; tracked mechanisms and the 16-entry inventory are in the table above |
| B5 alias negative test, deny on a hidden entry, pass on the boundary | the `--check` self-test plus the recorded demonstration below (exit 1 with the hidden entry, exit 0 without) |
| B5 single rules owner and doc consistency | `REPOSITORY_RULES.md` "Backend Session Entry" section, routed by `scripts/repository-rules-review.py`; `docs/design/explicit-session-boundary.md` and `docs/design/index.md` updated together |
| B5 `with_cpu_exec_session` as the single documented exception | `explicit-session-boundary.md` "capability bridge" entry plus the allowlist row |
| Final state: one-shot spellings do not compile | the `compile_fail` fixtures above |
| Final state: `with_backend_session`/`with_execution_scope` are the only boundaries | the 16-entry allowlist, whose remaining rows are session construction and the runtime cache entry |
| Final state: operations are not reachable from `TensorBackend` | the trimmed supertrait list together with the owner `compile_fail` fixtures |
| Evidence: build/check and tests pass | the table under "Phase-B closing state" |
| Evidence: doctests compile and run, deleted-API doctests moved | `cargo test --doc` for the affected crates, including the fail fixtures that now run as doctests |
| Evidence: pushed to the work branch | `refactor/1929-session-route-unification` |
| Evidence: one batched local gate at the end | `scripts/check-pr-fast.sh` and `scripts/repository-rules-review.py --dry-run` |

Outside this goal's scope, by the task's own split: the harness change forced by
the deletions (before-only arms cannot outlive the spellings) means the recorded
baseline is the before-reference and the certification run that re-measures the
baseline side, including the CUDA target, is the umbrella's Phase C.

### Phase-B closing state

With the CUDA body move in, Phase B is closed on this host:

| Check | Result |
| --- | --- |
| `scripts/check-pr-fast.sh --no-fetch --coverage-reviewed --test 'cargo test -p tenferro-gpu --features cuda --test integration'` | pass (`fast PR checks passed`) |
| `cargo check --workspace --all-targets` | 0 error, 0 warning |
| `cargo check -p tenferro-gpu --features cuda --all-targets` | 0 error, 0 warning |
| `cargo test --workspace --no-fail-fast` | all targets pass except one fixture of the pre-existing `tenferro-ad` trybuild contract, verified pre-existing by running the same test on the baseline worktree |
| `cargo test --manifest-path ext/tenferro-cpu-tblis/Cargo.toml` | 5 + 3 passed |
| `scripts/audit-session-entry.py --check` | pass, 16 entries, no allowlist change |
| `scripts/repository-rules-review.py --dry-run` | pass |

What Phase B does not include is evidence that the refactor is
performance-neutral. That is Phase C, which is still open and needs the quiet
window and a CUDA host described below.

### Phase-C pre-flight (tooling verified, measurement pending)

`scripts/compare-session-route-baseline.py` was run against the recorded baseline
and the Phase-A candidate logs to validate the comparison path before the real
measurement. Result: 98 `PAIRED_OK`, 1 `NOISY`, 0 `REGRESSION`, 0 `DELETED`, and
two expected failures that prove the fail-closed behaviour:

* `tenferro-gpu|route_matrix_gpu` has no candidate log — the CUDA target cannot run
  on this host;
* `session_chain/broadcast/execution_scope` is absent from those older candidate
  logs, and the comparator reports a missing non-deleted baseline case instead of
  skipping it.

So the harness, the capture/comparison pair and the thresholds work; what remains
for Phase C is the measurement itself, which needs the documented quiet window
(load below the recorded threshold with no live `cargo`/`rustc`), plus a CUDA host
for the GPU target.

#### Harness identity bug found during the first candidate run

The first candidate run compared route-matrix cases against routes the candidate
no longer has. `route_matrix` still registered `oneshot` arms for `dot_general`
and `reduce_sum`, whose one-shot spellings Step 2 deleted, and later runs found the
same for `slice` and `cast`, whose arms the Step-2 migration had already rewritten
to `with_backend_session` — every one of them timed the session route under a
before-route name, so the comparator compared them against owner-route baseline
numbers. `route_matrix_gpu` had the same shape for `dot_general`: its `oneshot`
helper had become a byte-for-byte duplicate of its `session` helper.

All before-only arms and their helpers are now removed from the campaign
harnesses, so no live registration uses an `oneshot`/`one_shot` label. That is
what the baseline's deleted-route predicate expects: the recorded numbers stay the
before reference and the comparator reports those cases as `DELETED` rather than
as regressions. The comparable pairs are `session/*` and `scope/*` against their
own recorded values, and the price of the unification is read off the recorded
`oneshot/*` numbers against the candidate's `session/*`.

The pruning was then verified without re-benchmarking, by deleting the
`oneshot` blocks from an existing candidate log of `route_matrix` and re-running
the comparator on a scratch logs directory: `deleted_route=16` and no regression
came from a pruned arm. The three `+5.6%`/`+6.4%` entries it still reported are on
`session`/`scope` cases of that loaded diagnostic capture, which is exactly the
noise this host cannot resolve; they are not a result, only evidence that the
classification path works. One caveat found while doing this: the comparator
reads the gate's merged stdout/stderr form (`Benchmarking <case>: Analyzing`), so a
stdout-only capture (`cargo bench ... > log`) cannot feed it and must be re-run
through `scripts/run-session-route-performance-gate.sh`.

#### Measurement attempt on a shared host (diagnostic only)

A candidate campaign was then attempted on this host: ten benchmark runs over
three passes for `route_matrix`, `session_chain`, `eager_dispatch_baseline`,
`eager_backward_shape_churn`, `elementwise_fusion` and `linalg_vjp_gate`, with the
campaign's pinned criterion settings (`--warm-up-time 2 --measurement-time 5
--sample-size 100`, which are at or below criterion's own defaults) and
`taskset -c 0` under the 1T environment. The host was **not** quiet: a foreign
`cargo` process was live throughout and the load average stayed near 6.8.

Result as a diagnostic: the `session_chain` cases showed no reproducible
deviation above +5% across the three passes, and of the 34 `route_matrix` cases
two exceeded +5%, both with a pass-to-pass spread (12.2% and 17.7%) larger than
their deviation (+9.5% and +5.4%). The single-pass `session_chain` deviations of
+5.5% to +14.5% seen in the first attempt did not reproduce.

Those numbers are **not** a Phase-C certificate and must not be reported as one:
on a shared host with concurrent compiler processes, the absence of a
reproducible deviation is not evidence that no regression exists. The same
qualification applies to the tempting explanation that the first attempt's
deviations were CPU frequency or thermal drift (the baseline is a recorded value
while the candidate is measured under long sustained single-core load) — that
hypothesis is consistent with the data but cannot be confirmed on a loaded host.

Phase C certification therefore remains open and requires all of:

* the documented quiet window, with the effective thread count recorded;
* three alternating baseline/candidate pairs, where the baseline side is
  re-measured from the pinned baseline commit with this harness rather than
  compared against the recorded numbers across a loaded window;
* the `tenferro-gpu|route_matrix_gpu` target on a CUDA host, which this host
  cannot run at all.

The criterion settings stay at the pinned defaults for certification; cheaper
settings are acceptable for a diagnostic pass only, because they change the
confidence intervals the comparator uses to separate `NOISY` from `REGRESSION`.

#### Certification runbook

The pieces below exist and were verified to the point this host allows; only the
quiet window and a CUDA host are missing. Every run goes through the campaign
script, which pins the criterion settings, the 1T thread environment and the CPU
affinity:

1. **Baseline side.** In the baseline worktree (`bench/1929-route-baseline`, which
   holds `25da8d431`), apply the current harness source — the seven campaign
   bench files — and confirm it builds:
   `cargo bench --no-run -p tenferro-cpu --bench route_matrix`,
   `-p tenferro-runtime --bench session_chain`,
   `-p tenferro-gpu --features cuda --bench route_matrix_gpu`. This was verified
   here: all three targets compile against the pinned code.
2. **Pairs.** Alternate three times, each pair writing to its own output
   directory:
   baseline `bash scripts/run-session-route-performance-gate.sh --mode run
   --label baseline --output-dir target/pair-N/baseline` in the baseline
   worktree, then candidate with `--label candidate` and
   `--output-dir target/pair-N/candidate` in the refactor worktree.
3. **Compare.** For each pass,
   `python3 scripts/compare-session-route-baseline.py --logs-dir
   target/pair-N/candidate --label candidate` against the recorded
   `docs/testing/session-route-baseline.json`. The `oneshot/*` and
   `one_shot` rows are before-only and must come back as `DELETED`; a baseline row
   that is neither deleted-route nor present in the candidate log is a
   fail-closed error, not a skip.
4. **GPU.** Steps 1–3 run on a CUDA host, because the campaign's
   `route_matrix_gpu` target needs a device; the protocol also requires reporting
   GPU enqueue and synchronized-completion cost separately.
5. **Record.** Write the comparator report and the alternating-pair logs into
   `docs/testing/` and a worklog entry, then remove the harness files from the
   baseline worktree (`git -C .worktrees/issue-1929-bench checkout -- .`) so the
   frozen harness is restored.

The recorded baseline keeps the before-only rows: re-capturing produces
`session`/`scope` rows only, and the before-reference stays the JSON that was
captured while the deleted spellings still existed.

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
   contains `"borrowed tensor views"`.

The rejection is uniform, not per family: the defaults for all 31 read halves
return the owned tensor through `read_tensor(op, input)` and raise
`read_boundary_error(op)` for a view. `read_tensor` is
`input.as_tensor().ok_or_else(|| read_boundary_error(op))`, so the default is
"delegate owned, reject views" everywhere. Materialization appears only where a
method overrides the default with `to_contiguous_read`: `dot_general_read`, the
`_read_into` outputs, and the cached dot paths.

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

**Step 1 — require `_read`, keep every caller working.** Delete the provided
defaults and make the read halves required, family by family. CPU and CUDA need
no new code. Every other implementor, including the test backends, reproduces
the old default in one line per operation, because the old default body *is*
`self.op(read_tensor(name, input)?, ..)`. That makes the two libraries below
the read-half migration mechanical:

- expose `read_tensor` as a `#[doc(hidden)] pub` bridge in `tenferro-tensor`,
  alongside the bridges this crate already uses for cross-crate internals
  (`with_cpu_exec_session`, `session_type_id`, `with_backend_session_cached`),
  so an external implementor writes
  `self.reduce_sum(read_tensor("reduce_sum", input)?, axes)` instead of a
  six-line match;
- for WebGPU, reproduce **both** branches of the old chain: `unsupported!` with
  the same op literal where the owned branch ended in the one-shot's
  `unsupported!`, and `read_boundary_error` for the view branch. Where the
  operation is real (`dot_general`), the old default body moves into
  `dot_general_read` unchanged.

Measured churn. Making only the four `TensorReduction` read halves required in a
throwaway probe broke nine `impl TensorReduction` sites in four files
(`tenferro-ad/src/eager_backend.rs` twice,
`tenferro-cpu/src/tests/cpu_tests/backend_misc.rs` twice,
`tenferro-runtime/tests/integration/session_ops.rs`, and
`tenferro-tensor/src/tests/backend_default_read_tests.rs`) while the
`--workspace --all-targets` check ran with default features. That is
approximately linear, so the full 31 read halves imply on the order of seventy
sites, each needing one line per operation. WebGPU's 31 implementations are
additionally required but are hidden from a default-feature check.

Per-family staging is viable and preferred for revertibility: the contract test
file asserts each family separately, so a family slice updates only its own
assertions. The ordering constraint is unchanged — a read half must become
required before its one-shot sibling is deleted, or the still-provided default
would recurse into itself.

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

## Local gate limitation for `compile_fail` fixtures

`trybuild` `compile_fail` contracts compare rendered diagnostics against a
committed `.stderr`, and two local conditions break that comparison
independently of any source change:

- **kache path remapping.** The local build wrapper passes
  `--remap-path-prefix`, so diagnostics render absolute `/kache/...` paths while
  the committed `.stderr` files carry crate-relative paths. Every `compile_fail`
  fixture in the affected crate then reports a mismatch. Running with
  `RUSTC_WRAPPER=""` restores the crate-relative form.
- **rustc version rendering.** At least `tenferro-ad`'s
  `tests/ui/eager_backend_owner_private.rs` still mismatches with the wrapper
  disabled: the committed `.stderr` expects a two-part span
  (`^^^^^^^^^^^^^------------` with the label on its own line) and rustc 1.97.1
  renders a single span. This reproduces with the working tree reverted, so it is
  pre-existing, not a consequence of any change here.

Consequences for the B1/B5 fixture sets:

- `pass` fixtures are unaffected — `trybuild` does not compare their output, so
  the session-surface contract in `tenferro-runtime` runs normally.
- The `fail` fixtures that must prove the one-shot spellings are gone are
  `compile_fail`, so their `.stderr` files are version-sensitive. They must be
  blessed in the same rustc that CI uses, and a local mismatch is not evidence
  that the expected error changed. Prefer fixtures whose diagnostic is
  structurally stable across versions (a missing-method or missing-trait-item
  error) over ones relying on span geometry.
- A local `compile_fail` failure therefore has to be checked against the
  pristine tree before it is attributed to a change. This was done once here:
  `git stash` + rerun reproduced the same mismatch.

### Step-1 mechanical aids and their limits

For the wider families the reproduction sites are numerous — 86 read halves
across nine impls for the ten `TensorAnalytic` operations — so a throwaway
generator was driven from `cargo check` E0046 output plus the `help: implement
the missing item` lines. Four limits were hit; knowing them matters before the
`TensorElementwise` family is attempted:

- Method-generating macros hide methods from a `^    fn ` scan. The
  implementation inventory must expand `delegate_with_pool_context!`,
  `delegate_with_pool!`, `panic_backend_methods!` and
  `unreachable_backend_methods!`, or the estimate is wrong.
- Some E0046 sites are reported inside a `macro_rules!` definition: the GPU
  `delegate!` shim and the per-crate `panic_*!` factories. Generating a method
  body inside a macro definition corrupts the macro. Those sites must be edited
  by hand — add entries to the delegation macro invocation, or bodies inside the
  factory macro.
- The trait declaration's receiver must be dropped when deriving parameters, or
  the generated signature contains `&mut self: `.
- Locating an impl's end by the first line equal to the impl's indentation plus
  `}` is not reliable in files with nested modules; in the tenferro-cpu test
  module it placed bodies outside the impl. Anchoring on the tail of the
  family's `panic_*!` or delegation block is reliable, and is how the earlier
  families were patched.

Every generated site uses the delegating form
(`self.op(read_owned_tensor("<op>", input)?)`), so Step 2 must inline the
one-shot bodies into these read halves. That second visit is expected and
mechanical: the one-shot bodies at these sites are panics, markers, or
unsupported errors. WebGPU is the deliberate exception and was written directly
in its final shape (`unsupported!` after evaluating the read input) across all
four families it implements, so it never needs a second visit.

### Step 1 complete

All 31 read halves of the invertible pairs are now required, verified by
scanning the trait declarations: `TensorElementwise` 13, `TensorAnalytic` 10,
`TensorStructural` 3, `TensorReduction` 4, `TensorDot` 1. The delegation
direction is inverted: a session implementation is now the primary
implementation, and no read half is derived from its one-shot sibling.

Four read-shaped methods remain provided and are deliberately outside the pair
set, because each lacks a required one-shot sibling:

| Method | Provided default | Hidden session entry? |
|---|---|---|
| `TensorElementwise::rem_read` | delegates to `rem` | no — `rem` is also provided and terminates in `Unsupported` |
| `TensorStructural::to_contiguous_read` | materializes compact host tensors, rejects views and device placement | no |
| `TensorReduction::reduce_sum_squares_read` | `Unsupported`, with a contract test asserting an explicit override is required | no |
| `TensorDot::dot_general_with_conj_read` | materializes, then calls `dot_general_with_conj` | no, but both reference the one-shot `dot_general` |

Step 2 must therefore also redirect the trait-internal defaults that reference a
deleted one-shot: `dot_general_with_conj` (which calls `dot_general`),
`dot_general_with_conj_read`, the `dot_general_read_into*` family, `SessionCachedDot::dot_general_cached` and its cached siblings, and the
`_read_into` elementwise defaults. Those are trait-internal edits, not new
implementor sites.

Evidence for Step 1: `cargo fmt` clean; `cargo check --workspace
--all-targets` clean and warning-free; 5111 workspace tests pass with the single
known pre-existing environmental trybuild mismatch in `tenferro-ad`.

### Step 2 sizing (measured)

Call sites of the one-shot spellings, counted by `\.(op)\(` per family. The
counts are an upper bound: a match may be a `Tensor`-side call on the session
extension surface or a `_read`-adjacent helper, and the real work list is
whatever `cargo check` reports after the methods are deleted.

| Family | non-test lib | tests/benches/examples | doctests | total |
|---|---:|---:|---:|---:|
| `TensorDot` (`dot_general`) | 35 | 133 | 2 | 170 |
| `TensorStructural` (`transpose`, `reshape`, `broadcast_in_dim`) | 78 | 157 | 12 | 247 |
| `TensorReduction` (`reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_min`) | 87 | 300 | 23 | 410 |
| `TensorAnalytic` (10 operations) | 131 | 484 | 48 | 663 |
| `TensorElementwise` (13 operations) | 337 | 1088 | 120 | 1545 |

Step 2 is therefore per-family in the order above: smallest first, so the
migration pattern is proven before the families that dominate the diff.
`TensorDot` is the pilot. Each family slice deletes the one-shot trait items,
inlines the one-shot bodies into the read halves that currently delegate to
them, redirects the trait-internal defaults listed above, and migrates callers.
The trait-internal redirect belongs to the same slice as the deletion, because
a default that still calls a deleted method cannot compile.

### Step 2 pilot: attempted, measured, reverted

The `TensorDot` deletion was attempted and reverted. The tree at
`5e9c7519a` (Step 1 complete) is the last verified state. Recording what the
attempt established, because it changes how Step 2 should be run.

**Identification is the first hard part, not the edit.** Of the 35 non-test
`\.dot_general\(` matches, only five are the deleted trait method: two
read halves that had to absorb the removed body, and three session read calls
(`tenferro-runtime/src/tensor.rs`, `tenferro-einsum/src/concrete.rs`,
`tenferro-ad/src/eager_exec.rs`). Every other match is a different API that must
not be touched — `EagerTensor::dot_general`, `TracedTensor::dot_general`,
`capabilities.dot_general()`, `CpuGeneralContractionProvider::dot_general`
(`&self`, provider request), and `gemm::dot_general` free functions. A raw
`\.dot_general\(` grep is therefore mostly false positives, and the same will
hold for `add`, `reshape`, and the rest.

**The test-side sites are three shapes.** Session-receiver calls already inside a
`with_backend_session` closure (rewrite to `_read` plus `TensorRead::from_tensor`
wrapping); owner-receiver calls (`backend.dot_general(..)`, including a
multi-line `backend\n .dot_general(` form) that need a boundary; and read halves
that still delegate to the removed method and must absorb its body.

**Six concrete codemod failure modes, all hit:**

1. Diagnostics that point *inside* a `macro_rules!` definition — the GPU
   `delegate!` shims and the per-crate `panic_backend_methods!` /
   `unreachable_backend_methods!` factories. Editing there corrupts the macro.
2. Factory *invocation* lines (`dot_general(...) -> Ret;` entries) that must be
   deleted individually.
3. Receivers that are not simple identifiers: `&mut B` generic session
   parameters, multi-line `receiver\n .method(` forms, `&mut dyn BackendSession`.
4. Argument lists with nested calls, macros and struct literals
   (`black_box(..)`, `DotGeneralConfig { .. }`), which need real paren matching —
   the indent/brace heuristics failed here repeatedly.
5. `E0407` stays hidden while a crate's earlier caller errors exist, so the work
   list arrives in stages rather than once.
6. String, raw-string and char literals break naive sanitizers.

**Recommendation for Step 2.** Migrate per *operation* rather than per family,
so each slice is one trait item and its call sites; migrate the library sites
first, then tests crate by crate; and keep the deletion in the same slice as its
migrations so the tree compiles after every slice. If a codemod is used again, it
should take an explicit file allowlist, refuse any site whose diagnostic points
into a `macro_rules!` definition, and require `cargo check --workspace
--all-targets` to pass after each file rather than once per family.

### Step 2 complete: 31 of 31 pairs deleted

Every paired one-shot in the Step-2 inventory is gone, one operation per commit,
in the smallest-first order the sizing table implies: the 10 analytic
operations, the 13 elementwise ones, the three structural ones, the four
reductions, and finally `TensorDot::dot_general`. `dot_general_with_conj`
remains, because it is one of the 13 required one-shots with no `_read` sibling,
and its body now calls `dot_general_read`.

The deletion exposed two failures that a compile-and-test pass alone would not
have caught, and both are now guarded:

* **A real implementation can hide behind the one-shot.** `WebGpuExecSession`'s
  `transpose_read` had been written as `unsupported!` while the working device
  transpose lived in the one-shot, so deleting `transpose` silently downgraded a
  supported operation. The read halves of every deleted operation were therefore
  re-checked against the removed body, and WebGPU transpose now keeps its
  `structural::transpose` route.
* **A migrated read half can call itself.** When a read half delegated to the
  one-shot and the one-shot was removed first, rewriting the delegate produced
  `fn op_read(..) { .. self.op_read(..) }` in test fixtures. A repo-wide scan for
  self-recursive `*_read` bodies is now clean; the fixtures that had one carry
  the removed behaviour directly (delegate to the wrapped backend, return the
  fixture result, or keep the same rejection after materializing a view).

The migration tooling that made 31 slices tractable:

* the work list came from `cargo check` diagnostics, never from a name grep, so
  the look-alike APIs (`EagerTensor::dot_general`, `TracedTensor::transpose`,
  provider capability queries, `gemm::*` free functions) were never touched;
* call sites were rewritten from the diagnostic's own span, with a
  string/comment-aware delimiter matcher over the original text, and only
  arguments whose `_read` parameter is a `TensorRead` were wrapped;
* the structural half of each slice (trait item, CUDA body absorption into the
  read half, CPU session delegation, runtime extension, eager dispatcher) was
  driven by a per-operation table, and every count was asserted before writing;
* `cargo fmt --all`, `cargo check --workspace --all-targets`, the workspace
  doctests and the focused suites ran per operation, and the full workspace suite
  once at the end.

Verification for the completed Step 2: `cargo check --workspace --all-targets`
is clean and warning-free; `cargo test --workspace` passes 5207 tests with the
single pre-existing environmental `tenferro-ad` trybuild span mismatch
(`eager_backend_capability_contract`); the workspace doctests pass; and no
`*_read` method recurses into itself.

The remaining Phase-B work is unchanged: B3 (drop `BackendSession`,
`TensorBackendOps` and `BackendCachedDot` from the `TensorBackend` supertraits
and keep only the session implementations), B4 (extension owner routes), B5
(audit gate, `REPOSITORY_RULES.md` owner, batched repository gate), then the
Phase-C paired benchmark against the recorded baseline.

### B3 reconnaissance: what removing the owner supertraits surfaces

Dropping `BackendSession`, `TensorBackendOps` and `BackendCachedDot` from
`TensorBackend` was measured once, then reverted, so the next session starts from
a known work list instead of rediscovering it.

Inside `tenferro-tensor` the change is small and self-contained:

* `impl<T> SessionCachedDot for T where T: TensorBackend` has to name `TensorDot`
  itself once the blanket access through `TensorBackendOps` is gone;
* `default_backend_session` and the default bodies of
  `BackendSessionHost::with_backend_session[_cached]` are what make an owner
  usable as a session. Removing them makes `with_backend_session` a required
  method, which in turn forces the test fixtures that write
  `impl BackendSessionHost for Fixture {}` to open a real session
  (`with_session_entry_guard(|| f(self))`) instead of borrowing the owner's own
  operation implementations.

Everything else is in `tenferro-runtime`, which is the crate that still treats
the owner as an operation host:

* `exec/dispatch.rs` calls `dot_general_read` / `dot_general_with_conj_read` on a
  `&mut B: TensorBackend`, so the FFI dispatch functions must take a session (or
  open one) rather than the owner;
* `HostExecution` (`TensorBackendOps + TensorDeviceTransfer`) and the host
  dispatch table lose their blanket source, so `execute_host_dispatch` and its
  `B: TensorBackend` re-exports in `exec.rs` have to become session-taking;
* `BackendCachedDot::dot_general_read_cached` / `..._with_conj_read_cached` need
  the cached trait bound back explicitly, or a session route;
* `reclaim_exec_slot_with_backend` uses `TensorBuffer::reclaim_buffer` on the
  owner, which was previously reachable through `TensorBackendOps`.

The public boundary this reaches is `TensorBackend`-bounded generic API: 103
sites in 21 files, of which the runtime dispatch layer, `tenferro-ad`'s eager
execution, and the `ext/*` extension modules dominate. B3 therefore splits into
(i) the supertrait trim plus the `tenferro-tensor` fixture changes above, (ii)
the runtime dispatch/execution layer moved to session-taking, and (iii) the
owner-side operation implementations and `BackendCachedDot` impls deleted once
nothing calls them. (i) alone does not compile the workspace, so it must ship in
the same commit as (ii); (iii) is what makes the removal observable.

### B3 depends on B4: the session-region path still falls back to the owner

A second B3 attempt converted the runtime dispatch layer and then stopped,
because it ran into the extension route. The finding is an ordering constraint
that the slice list does not show.

What converts cleanly: `exec/dispatch.rs`'s host table and the
`execute_*_host` functions become `&mut dyn BackendSession` (the session path
already calls them), the four owner-based evaluators in `segment.rs`/`exec.rs`
open one session and use `execute_segment_in_session` /
`execute_value_segment_in_session` / `execute_ffi_instruction_exec`, and the
owner wrappers (`execute_host_instruction`, `execute_ffi_instruction[_cached]`,
`reclaim_exec_slot_with_backend`, `reclaim_last_use_inputs_backend`) disappear.

What does not convert yet is the *extension* operation route. Two facts pin it:

1. `execute_prepared_extension_instruction` builds `ErasedExecutionContext`, which
   requires a `Sized + 'static` type, so it needs the owner (`B: TensorBackend +
   'static`), not `&mut dyn BackendSession`. The session equivalent
   (`execute_prepared_extension_instruction_in_session`) exists and is used by
   `execute_ffi_instruction_exec`, but the owner route is still reachable.
2. `segment_is_session_compatible` excludes FFI ops that are not session
   compatible, and the session-region evaluator falls back to
   `execute_ffi_instruction_cached(backend, ..)` for them. With the owner-based
   FFI dispatch deleted, that fallback has no callee.

So B3's owner-side operation impls cannot be deleted while the extension fallback
still runs operations on the owner, and the FFI dispatch table cannot become
session-only while that fallback exists. B4 (extension routes primarily
session-based, owner route limited to the runtime-formed region fallback) is
therefore a prerequisite for B3(iii), not a follow-up.

**Why the owner is still needed: capability identity, not execution.** A second
look at the extension routes narrows the reason. `execute_linalg_extension_reads_owner`
(and the fft equivalent) already opens a session internally
(`backend.with_backend_session(..)`) and runs the same session executor, so the
*execution* is session-based today. What needs the owner is
`ErasedExecutionContext<'_, B>` with `B: TensorBackend + 'static`, which the
extension runtime uses to identify the concrete backend for capability
dispatch. The session-based equivalent already exists
(`BackendSession::session_type_id`, `with_cpu_exec_session`,
`with_cuda_exec_session`), and `execute_prepared_extension_instruction_in_session`
uses it, but the owner path is still reachable.

It is reachable rather than dead because session support is per operation:
`linalg_session_supported` returns `true` for the whole family on CPU, and
`false` on CUDA for `FullPivLu`, `FullPivLuSolve` and general `eig`
(`extension.rs:1060`, issue #1665). For those operations the scheduler keeps the
owner path, and that path's `ErasedExecutionContext` cannot be built from
`&mut dyn BackendSession`.

So B4's deliverable is precise: move extension capability dispatch from the
owner's type identity to the session's, and give the remaining per-op
exceptions a session route (or an explicit documented refusal), after which no
extension operation needs the owner and the runtime's owner extension path can
be deleted. That is the prerequisite for B3.

The workable order is:

* B4: make the extension runtime registers session primary, move capability
  dispatch to `session_type_id`, and keep the owner route only where the runtime
  forms a region explicitly.
* B3(i)+(ii): trim the supertraits, delete `default_backend_session`, thread a
  session through the dispatch layer — after B4 the only remaining owner bound is
  gone and the FFI table can be session-only.
* B3(iii): delete the owner-side operation impls (`TensorElementwise` and the
  other families for `CpuBackend`, `CudaBackend`, `WebGpuBackend`), the
  `BackendCachedDot` impls and the GPU `delegate!` shims, then shrink
  `scripts/session-entry-allowlist.json`.

### B4 landed: the owner entry now forms the session

`8468debd5` made `execute_reads` optional in `define_extension_runtime!`. A
family that registers the session route (`execute_in_session` +
`session_supported`) now gets an owner entry that calls
`with_backend_session` itself and hands the extension nothing but
`&mut dyn BackendSession`; supplying both routes, or neither, is a compile
error. The einsum, linalg and fft owner routes (`execute_*_reads_owner`,
`execute_einsum_extension_reads`) are deleted, the linalg session-context
variant that only the owner route used is deleted, and the three tests that
drove the owner routes now drive the session route. No operation runs on the
owner through the extension path any more.

That removes the reason B3 could not start. What remains for B3(ii) is a
mechanical, now-safe conversion of the runtime dispatch layer, with these exact
call sites:

| Site | Today | After |
|---|---|---|
| `segment.rs:287`, `:390`, `:495`, `:579` | `execute_ffi_instruction_cached(backend, ..)` | session run, or the extension fallback |
| `segment.rs:296`, `:301`, `:400`, `:411`, `:504`, `:589` | `reclaim_last_use_inputs_backend(slots, inst, backend)` | `reclaim_last_use_inputs_exec` inside the run |
| `segment.rs:300`, `:409` | `execute_host_instruction(backend, ..)` | `execute_host_instruction_exec` inside the run |
| `exec.rs:649`–`:663`, `:694`–`:710` | the same pair in the unsegmented evaluators | the same run structure |
| `exec.rs:873` | `execute_ffi_instruction(backend, ..)` fallback | deleted with the owner wrapper |
| `runtime/execution.rs:1270`–`:1287` | owner host/ffi/reclaim | the run structure |

Two things to keep in mind while doing it:

* the fallback instruction is always an extension op: `is_session_compatible_instruction`
  returns `true` for non-FFI ops and for `DotGeneral`/`DotGeneralWithConj`
  (`is_exec_session_ffi_op`), so a `false` result can only come from an
  extension whose `supports_session()` is false. The fallback therefore calls
  `execute_extension_instruction` and can report an internal error for anything
  else, and it needs no `BackendCachedDot` bound.
* the unsegmented evaluators execute instruction-by-instruction, so the
  conversion must group each maximal run of session-compatible instructions into
  *one* session (`with_backend_session_cached`) and run the fallback outside it.
  Opening a session per instruction would be a needless change in session-entry
  count for the very path the Phase-C baseline measures.

### B3(iii) work list: deleting the owner implementations

`bed79ade0` removed the supertraits, so nothing *requires* the owner-side
operation implementations any more; deleting them is the last step, and the
measured shape of that step is recorded here because it is a call-site migration,
not a deletion.

Deleting the eight CPU owner impls (`BackendSession`, `TensorElementwise`,
`TensorAnalytic`, `TensorStructural`, `TensorReduction`, `TensorDot`,
`BackendCachedDot`, `TensorIndexing`) leaves 261 errors in the CPU crate alone,
across about a dozen test/bench files. They split into four shapes, and only the
first two are mechanical:

| Shape | Count (CPU crate) | Rewrite |
|---|---:|---|
| paired one-shot with a `_read` sibling (`slice`, `pad`, `gather`, `reverse`, `concatenate`, `cast`, `copy_read_into`, `to_contiguous_read`, `reduce_*_read`) | ~120 | `recv.op(args)` → `recv.with_backend_session(\|__s\| __s.op_read(args))` |
| session method with the same name (`dot_general_read_into_accum`, `elementwise_read_into`, `grouped_gemm_cached`) | ~30 | `recv.op(args)` → `recv.with_backend_session(\|__s\| __s.op(args))` |
| cached dot family (`dot_general[_with_conj]_cached`, `_read_cached`, `grouped_gemm_cached`) | ~20 | the owner form takes `(&mut cache, cache_slot, ..)` and the session form takes `(cache_slot, ..)`, so the cache becomes the receiver of `with_backend_session_cached` |
| receivers that are not a plain identifier (`CpuBackend::new().op(..)`, trait-qualified `BackendCachedDot::op(&mut backend, ..)`) and `&mut dyn` sites | ~15 | hand edits |

Two codemods were used and are worth reusing, outside the repository:

* a diagnostic-driven wrapper that reacts to `E0599 no method named `op`` and
  rewrites only the reported call span, one edit per file per round so later
  line numbers stay valid; it cleared ~80 of the first two shapes in a few
  rounds and left a short manual list;
* a cached-shape rewriter for the third shape. It must match "first argument is
  the cache" only when the call is *not* already cache-bound, or it re-wraps its
  own output — the version that ran here oscillated and was reverted.

Remaining work, in order: finish the CPU crate (about 20 hand sites), repeat for
the CUDA and WebGPU owner impls (which also removes the `delegate!` shim and
moves their bodies to the module functions the sessions already call), migrate
the downstream call sites in `tenferro-ad`, `tenferro-linalg`, `tenferro-einsum`
and the runtime tests/benches, delete `install_with_pool_context*` once the CPU
owner impls are gone, and re-bless
`scripts/session-entry-allowlist.json` (the CPU entries it lists are exactly the
ones that disappear).

### B1 fail fixtures: the deleted spellings are pinned by `compile_fail` doctests

The B1 contract file covers the surviving surface with trybuild pass fixtures.
The *fail* side is pinned with rustdoc `compile_fail` examples instead of
trybuild `.stderr` files, because `compile_fail` only requires compilation to
fail and therefore does not depend on the compiler's span rendering or on any
local build wrapper rewriting paths:

* `tenferro_tensor::BackendSession` documents that `exec.add(a, b)` inside
  `with_backend_session` no longer compiles;
* `tenferro_cpu`'s crate docs pin the deleted owner spellings for one operation
  per family (`add`, `mul`, `exp`, `reduce_sum`, `transpose`, `dot_general`).

The `pass` side is a trybuild contract that originally skipped itself when the
`NEXTEST` environment variable was set. That skip was wrong for this repository:
CI's workspace profile is `cargo nextest run --workspace` plus `cargo test --doc
--workspace`, so a nextest-only skip made the surviving-surface fixtures a CI
*target* that never ran. The skip is removed after verifying the contract under
nextest here (1 passed, about 60 s cold, about 3 s warm), so the fixtures and the
`compile_fail` examples both execute in CI.

The fixtures whose targets B3 removes have landed with that slice:
`tenferro_tensor::BackendSessionHost` pins the deleted
`default_backend_session` factory, and `tenferro_cpu`'s crate docs pin the
owner-level `BackendCachedDot` bound in addition to the one-shot spellings. The
positive counterpart is the `pass` fixture for the cache-aware session route
(`with_backend_session_cached` plus a session `_cached` contraction), which
still compiles, so the two sides together pin that the cached route moved from
the owner to the session rather than disappearing.
