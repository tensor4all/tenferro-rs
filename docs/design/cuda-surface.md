# The documented `tenferro_gpu::cuda` surface (issue #2298)

Status: agreed on issue #2298 (comment 6768140379) and applied in #2579. The table below is generated;
the verdicts are reviewed judgement recorded in `docs/internals/cuda-surface-verdicts.json`. The five
`private` verdicts that no code reads are applied (section 5); the kernel-launch family stays
`workspace-internal` by that decision (section 4).

The migration rule that produced the `#[doc(hidden)]` marks ("keep the whole new surface hidden until
the cutover") no longer holds: `tenferro_gpu::cuda` is the only CUDA path, and
`cuda/mod.rs` re-exports the engine, the device types, the transfers, the publication types and the raw
scope. What is missing is the decision about which of those items is a contract a user may keep using.

## 1. The table

`scripts/cuda_surface_inventory.py` generates `docs/internals/cuda-surface-inventory.md`:

```bash
python3 scripts/cuda_surface_inventory.py           # regenerate
python3 scripts/cuda_surface_inventory.py --check   # fail on drift (CI-shaped)
python3 scripts/cuda_surface_inventory.py --missing # items with no verdict
```

The Rust sources are the membership authority: every `pub` / `pub(crate)` / `pub(super)` item declared
under `crates/tenferro-gpu/src/cuda/**` — modules, structs, enums, unions, traits, type aliases,
functions, constants and statics, plus the methods of their `impl` blocks, excluding `#[cfg(test)]`
items, `#[cfg(test)]` impls and test modules — with its kind, its visibility, whether it carries
`#[doc(hidden)]`, whether a `pub` path reaches it from `tenferro_gpu::cuda`, and who references it:
other workspace crates, the crate's own `examples/`/`tests/`/`benches/`, and the user-facing guides.
Every row cites up to three `file:line` examples. Struct fields and enum variants are not rows; a
`#[doc(hidden)]` on one still counts in the mark total.

Three rules matter when reading the reference columns:

- Reachability is not the same as "the module it is written in". `raw` is `pub`, but `device`,
  `identity` and `vendor` are `pub(crate)`, so their items are reachable only through the `cuda::`
  re-exports; the table prints the path a user can actually write.
- Counts are name-based and therefore a lead, not proof. A module-level item counts only in files that
  name `tenferro_gpu`. A method counts its qualified `Type::name` uses and its `.name(` receiver calls
  in files that name the owner type, so a same-named method on another type can still raise a count,
  and a call through a variable whose type the file never spells out is missed. Where the count misses
  a real call site, the row's evidence carries an `evidence` `file:line` that the script checks exists
  in the checkout.
- Design notes, plans and worklogs are not counted: they record what someone once meant, not who uses
  the item. Only the guides count as documentation evidence.

### Recount at `origin/main` `90348dc0`

| | `tenferro-gpu/src/cuda/` | `tenferro-gpu/src/` |
|---|---:|---:|
| body (`95cd1b849`, issue #2298) | 123 | 135 |
| after #2312 (`094f25f10`) | 117 | 129 |
| the issue's last report (`55e6a8c06`) | 127 | 139 |
| this table (`90348dc0`) | **126** | **138** |

Against the issue's own baseline the count is **+3** (123 → 126 over four weeks of work), and against
the last report on the issue it is **−1**: `CudaEngine::runtime_mut` lost its mark when #2556
(`e4dfa2f4`, "retire the seven unused U02 internal pub items") deleted it. `cuda/engine.rs` is now 11
marks (reported as 12 at `55e6a8c06`). The table covers **292** exported items, of which **130** carry
`#[doc(hidden)]`: the 126 at `90348dc0` plus the four `KernelModules` launch methods the decision keeps
`workspace-internal` (section 4).

## 2. The verdicts

Three verdicts, applied by these rules:

- **public** — a user of the crate needs it: a guide or tutorial uses it, or it is unavoidable to name
  in the signature of such an API (a returned type, a parameter, an error payload). `#[doc(hidden)]` is
  not part of this test: a hidden item can still be supported surface.
- **workspace-internal** — another tenferro crate calls it today and no guide documents it. Removing its
  `pub` would mean editing a sibling crate, which is the signal for this verdict rather than a reason to
  edit the sibling. The kernel-launch family is the one deliberate exception (section 4): no caller
  today, but the issue decision keeps it hidden `pub` for the RRQR hook.
- **private** — nothing outside `crates/tenferro-gpu/src/cuda/` reads it. Zero references means private
  unless the table shows a reason otherwise.

| verdict | items | of the 130 hidden | what it means |
|---|---:|---:|---|
| public | 101 | 66 | 64 lose `#[doc(hidden)]` and get docs (section 10); 2 stay hidden on purpose (section 4) |
| workspace-internal | 42 | 42 | keep `pub` and keep `#[doc(hidden)]`, documented as internal (section 4) |
| private | 149 | 22 | lose `pub`: 5 now, 33 more after the owning test suite is ported (sections 5 and 6) |

Per module (an item is listed under the module its `impl`/declaration sits in; `CudaBackend` and
`gpu_available` sit directly in `cuda/mod.rs`):

| module | public | workspace-internal | private |
|---|---:|---:|---:|
| `mod.rs` (`CudaBackend`) | 0 | 0 | 1 |
| `mod.rs` (`gpu_available`) | 1 | 0 | 0 |
| `capability` | 1 | 0 | 0 |
| `cutensor` | 0 | 0 | 9 |
| `device` (incl. the `pub(crate)` module) | 15 | 0 | 4 |
| `dispatch` | 0 | 0 | 32 |
| `engine` | 7 | 4 | 7 |
| `event_domain` | 0 | 0 | 5 |
| `fusion` | 0 | 0 | 5 |
| `identity` (incl. the `pub(crate)` module) | 5 | 0 | 2 |
| `launch` | 0 | 7 | 2 |
| `pending` | 8 | 0 | 3 |
| `pinned` | 6 | 0 | 0 |
| `publication` | 11 | 0 | 1 |
| `raw` (incl. the `pub` module) | 29 | 0 | 12 |
| `runtime` | 12 | 5 | 30 |
| `runtime_adapter` | 2 | 0 | 11 |
| `storage` | 0 | 0 | 6 |
| `structural` (incl. `activation`) | 0 | 26 | 10 |
| `transfer` | 4 | 0 | 2 |
| `vendor` (incl. the `pub(crate)` module) | 0 | 0 | 7 |

## 3. The surface that should be documented

These lose `#[doc(hidden)]` and get prose, examples and the `# Errors` / `# Panics` sections the crate
audits (`scripts/check-public-error-docs.py`). The guides already use most of them, so this is mostly
about the rustdoc mark, not about new documentation.

- **Engine**: `CudaEngine`, `CudaEngine::open`, `CudaEngine::runtime`, `CudaEngine::publish`,
  `CudaEngine::with_raw`, `CudaEngineError`.
- **Device**: `cuda_devices`, `CudaDeviceId` (+ `from_ordinal`, `ordinal`), `CudaDeviceInfo`
  (+ its five accessors), `CudaDeviceError` (+ its four accessors), `CudaComputeCapability`,
  `CudaDeviceUuid` (+ `from_bytes`, `as_bytes`, `to_bytes`), `GpuExtensionCapability`,
  `cuda_capabilities`.
- **Availability**: `gpu_available` — the documented device probe the quickstart and every GPU test
  uses (83 references from other crates, 13 in the guides). It also *is* the migration item #2298 asks
  about: keeping it public is the point, so it loses the mark rather than its `pub`.
- **Transfers**: `upload_tensor`, `download_tensor`, `upload_pending`, `download_pending`,
  `PendingUpload`/`PendingDownload` (+ `is_ready`, `wait`), `RejectedTransfer`, `TransferSubmitError`,
  `PinnedHostBuffer` (+ `new`, `as_slice`, `as_mut_slice`, `len`, `is_empty`).
- **Publication**: `SourcePublication` (+ `bytes`, `frontier_ready`, `await_frontier`,
  `download_into`), `PublicationDownload` (+ `is_ready`, `wait`), `PublicationError`,
  `PublicationAdmissionError`, `RejectedPublication`. No other crate exercises publication yet, but
  `CudaEngine::publish` returns these types, so a user who publishes must name them; the group is a
  single contract and is documented as one.
- **Raw extension scope** (issue #1597): `raw::Session` and its eleven methods, `StreamRef`
  (+ `raw_handle`), `TensorRef`/`TensorMut` (+ `byte_len`, `raw_ptr`, `with_ptr`), `DeviceBytes`
  (+ `byte_len`, `with_ptr`), `CudaResourceGuard`, `with_cuda_engine`. `tenferro-fft` and
  `tenferro-linalg` call these today, so they are also the workspace's own extension contract.
- **Runtime**: `CudaRuntime`, `CudaRuntime::new`, `synchronize`, `pinned_staging_stats`,
  `set_pinned_staging_limit_bytes`, `clear_pinned_staging`, `artifact_dir_stats`,
  `artifact_dir_max_bytes`, `artifact_dir_evictions`, `clear_artifact_dir`,
  `DEFAULT_PINNED_STAGING_LIMIT_BYTES`, `PinnedStagingStats` — the knobs the devices and transfers
  guides already document.
- **Registration**: `cuda_runtime_engine_registration`, `cuda_runtime_hardware_class`.

## 4. What stays `#[doc(hidden)]`, and why

**The 42 `workspace-internal` items.** Thirty-five of them are a contract *of the workspace*, not of
the crate: another crate in the workspace calls them today. They are `CudaEngine::{runtime_engine_id,
runtime_identity, launch_completion_syncs, vendor_resource_inits, alloc_output, ensure_gpu_resident,
scale_tensor_write, to_contiguous_view}`, the 21 `linalg_*` auxiliaries and the `DeviceOperand` bound
they take (called from `tenferro-linalg`), `CudaRuntime::{allocation_domain_id, device_id,
with_current_context, stream_waits}` and `CudaRuntimeIdentity` (called from `tenferro-fft` and
`tenferro-linalg`). They stay `pub` and stay hidden: `#[doc(hidden)]` is exactly the "narrow,
owner-scoped hook" of R-api-design-09, and hiding them keeps `cargo doc` honest about what a user may
rely on while the workspace still compiles. Each row's evidence names the call site the automatic count
cannot see — `runtime().device_id()` and `runtime().stream_waits()` are receiver calls in files that
never spell out `CudaRuntime`.

The other seven are the kernel-launch family — `KernelModules`, its `for_entries`, `function`, `launch`
and `modules` methods, `device_buffer` and `layout_arg`. Nothing outside this crate names them today;
the issue decision keeps them `workspace-internal` because `docs/design/cuda-extension-hook-2208.md`
says the RRQR kernels will need a module and launch surface, and narrowing them now would reopen the
surface in a few weeks. They stay `pub` with `#[doc(hidden)]`, so there is no user-facing promise and
no churn.

`CudaRuntime::identity` was on this list by mistake: `tenferro-fft` keys its cuFFT plan cache on
`CudaEngine::runtime_identity`, not on the runtime accessor. It is private and was narrowed with the
rest.

**Two deliberate public-but-hidden hooks.** `CudaEngine::vendor_resource_stats` and
`CudaEngine::clear_vendor_resources` have no reader anywhere, and the crate's own redundancy audit
(`docs/audit-redundancy-2026-11-04.md`, item 9) already decided they are downstream-facing hooks that
must not be deleted. They are therefore `public` and stay hidden — the clearest example of why
"doc-hidden" and "private" must not be conflated. The same audit names a third,
`CudaEngine::runtime_mut`, which #2556 deleted; that line of the audit is stale.

**The `private` items that are already crate-private.** They carry no decision left to make; the table
records them so the next reader does not re-derive that they are unreachable.

## 5. What lost `pub` in this change

Five items, each with no reader of any kind outside `src/cuda/`, and each still read by non-test code
inside the crate, so `pub(crate)` alone keeps the crate warning-free:

| item | now | why it is private |
|---|---|---|
| `cuda::CudaBackend` | `pub(crate)` | the seam marker: nothing names it. Its only use is the crate-private `gpu::Buffer<CudaBackend>` inside `cuda/storage.rs`, and `docs/design/cuda-extension-hook-2208.md` records that "the generic seam stays internal" (R-api-design-90). The design notes that still name it predate the cutover and do not make it surface. |
| `CudaRuntime::alloc_pinned` | `pub(crate)` | read only by `cuda/pinned.rs` |
| `CudaRuntime::identity` | `pub(crate)` | read only by `cuda/engine.rs` and `cuda/runtime.rs`; `tenferro-fft` uses `CudaEngine::runtime_identity` instead |
| `CudaRuntime::completion_deferred` | `pub(crate)` | read only by `cuda/structural.rs` |
| `CudaEngine::to_contiguous` | `pub(crate)` | read only by `cuda/structural.rs`; it is a one-line forward to `TensorStructural::to_contiguous_read` |

The sixth reduction, `KernelModules::function`, was withdrawn: the issue decision keeps the whole
kernel-launch family `workspace-internal`, so it is `pub` with `#[doc(hidden)]` again (section 4).

`CudaBackend` stays narrowed because it is not the public extension point: there is no second
implementer, and #2552 reshapes the execution contract, so publishing the seam now would freeze a shape
that moves next.

`cuda::structural::DeviceOperand` was going to be narrowed with the rest, and the compiler stopped it: it is the
`impl DeviceOperand` bound on the `linalg_*` auxiliaries that `tenferro-linalg` calls, so narrowing it
trips `private_bounds` on all 21 of them. Its verdict is `workspace-internal` for that reason — a useful
check on the rule that a bound of a workspace-internal item inherits its visibility.

Narrowing `to_contiguous` surfaced one more thing worth recording: `clippy::wrong_self_convention`
exempts exported API, so a `to_*` method that takes `&mut self` starts failing `-D warnings` the moment
it becomes crate-private. It carries an `#[expect(clippy::wrong_self_convention)]` with the reason
beside it, because the name mirrors the `TensorStructural` method it forwards to. Consolidating that
second path into `to_contiguous_read` is its own change, not part of this one.

## 6. Private, but not removed in this change

33 items still carry `pub` while their verdict is private. For 32 of them the only reader is this
crate's own test code — the integration suite under `crates/tenferro-gpu/tests/` (`seam_*.rs`) and, for
three, a `#[cfg(test)]` module or the test-only `PreparedFusion::counters`. Those tests are separate
crates and can only reach the public surface, so removing `pub` means moving or rewriting them. The
33rd, `GpuExtensionCapability`, is `pub` only because narrowing an item nobody constructs would make it
dead code: it needs deleting, not hiding. Both are real changes and neither belongs in an API-shape
commit.

| family | items | reader today |
|---|---:|---|
| cuTENSOR plan surface: `Contraction`, `Cutensor` (+ five methods), `Operand` | 8 | `tests/seam_contraction.rs` |
| fusion preparation: `PreparedFusion` (+ `dtype`, `key`, `launch`, `prepare`) | 5 | `tests/seam_fusion.rs`, `cuda/fusion/tests.rs` |
| runtime measurement and tuning: `artifact_cache_hits`, `compile_attempts`, `completion_checkpoints`, `module_loads`, `pending_transfer_retirements`, `pinned_staging_allocations`, `pool_release_threshold_bytes`, `stats`, `with_config`, `copy_to`, `trim_pool`, `device_info`, `device_ordinal`, `download_bytes`, `upload_bytes` | 15 | `tests/seam_*.rs` and the test-only `PreparedFusion::counters` |
| cuBLAS wrapper: `Cublas` (+ `gemm`, `new`), `GemmRequest` | 4 | `tests/seam_vendor.rs` |

`artifact_cache_hits`, `module_loads` and `PreparedFusion::key` have no non-test reader at all: narrowing
them now would make them dead code in a normal build, so they wait for the same change that gates the
test-only `PreparedFusion::counters`, which already follows the #2312 precedent of a `#[cfg(test)]`
accessor.

## 7. Not to be removed while a sibling crate uses it

Explicitly, so a later pass does not remove them: the 42 `workspace-internal` items in section 4,
including the kernel-launch family, and the `DeviceOperand` bound the linalg auxiliaries take. The `linalg_*` auxiliaries in particular are called from
`tenferro-linalg/src/gpu/linalg.rs` and its `householder_qr.rs` / `rank_revealing_qr.rs` siblings, and
`to_contiguous_view` from `tenferro-linalg`, and `alloc_output`, `ensure_gpu_resident`,
`scale_tensor_write` from `tenferro-fft` and `tenferro-linalg`.
Removing any of them is a change to those crates, not to this one.

## 8. Corrections to the issue's premises

- The issue says `KernelModules`, `Dispatcher` and the linalg auxiliaries "leaked into `pub` because a
  sibling crate needed them". That is true for the `linalg_*` auxiliaries only. `KernelModules`,
  `device_buffer`, `layout_arg`, `Cutensor`, `Contraction`, `Operand`, `Cublas` and `GemmRequest` have
  **zero** references from other crates in the workspace; their only readers are this crate's own
  `seam_*` tests. Those items are private surface the tests exercise, not a workspace contract. The
  kernel-launch family is nevertheless kept `workspace-internal` by the decision below, for the future
  RRQR hook, not because a sibling calls it today.
- `Dispatcher` was already `pub(crate)` after #2312, and `PrepareCounters` no longer exists.
- `cuda::CudaBackend` and `gpu_available()` are the two migration items; their fate is decided above
  (`private` and `public` respectively), not deferred.
- The issue says "others are internals that leaked into `pub` because a sibling crate needed them
  (`KernelModules`, `Dispatcher`, the `linalg_*` auxiliaries, `Resolved`-adjacent helpers)". Two of
  those are also *dead* rather than leaked: `GpuExtensionCapability` has no reader and no method
  returning it, and its rustdoc names `CudaEngine::supports`, which does not exist. Deleting it is the
  honest verdict, not documenting it.

## 9. Decisions recorded on the issue

1. **`CudaBackend` stays `pub(crate)`.** It is not the public extension point: there is no second
   implementer, and #2552 reshapes the execution contract, so publishing the seam now would freeze a
   shape that moves next.
2. **The kernel-launch family stays `workspace-internal`** (`pub` + `#[doc(hidden)]`, not
   `pub(crate)`): the RRQR kernels will need a module and launch surface, and narrowing it now means
   reopening it in a few weeks.

Everything else is accepted as reported: 101 public (64 lose the mark and get docs), 42
workspace-internal, 149 private of which 5 here and 33 after the owning test suites are ported. The
recount at `90348dc0` (126 marks, +3 against the issue body, −1 against the last comment, explained by
#2556) is the number to use from then on; this branch adds four marks for the kept launch methods.

## 10. Follow-up work, in order

1. This change: the five `pub` removals in section 5.
2. Drop `#[doc(hidden)]` from the **64** public items (not 66: the 66 hidden `public` rows of
   section 2 minus `CudaEngine::vendor_resource_stats` and `CudaEngine::clear_vendor_resources`, which
   are `public` and stay hidden by section 4) and check them against
   `scripts/check-public-error-docs.py`; make the guides agree (`devices-and-gpu.md`,
   `engines-and-transfers.md`, `tutorial-code/src/bin/cuda_*.rs`).
3. Port or move the `seam_*` suite so the 32 test-read items in section 6 can lose `pub`, and gate the
   three counters together.
4. Delete `GpuExtensionCapability` and its `cuda::` re-export, and the dangling `CudaEngine::supports`
   link in its rustdoc.
5. Consolidate `CudaEngine::to_contiguous` into `to_contiguous_read`, the second path to the same
   operation.
6. Give publication a guide: it is documented surface with no user-facing walkthrough yet.
