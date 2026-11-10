Status for #2298 — decided (comment 6768140379) and applied in #2579. The private verdicts that no
code reads are applied; the kernel-launch family stays workspace-internal.

## Shape of the final surface

292 exported items under `crates/tenferro-gpu/src/cuda/**` (modules, types, functions, constants, and
their `impl` methods; `#[cfg(test)]` items and test modules excluded), 130 of them carrying
`#[doc(hidden)]`:

| verdict | items | of the 130 hidden | next step |
|---|---:|---:|---|
| public — a user of the crate needs it | 101 | 66 | 64 lose the mark and get docs; 2 stay hidden on purpose |
| workspace-internal — another tenferro crate calls it, no guide documents it | 42 | 42 | stay `pub`, stay hidden |
| private — nothing outside `cuda/` reads it | 149 | 22 | lose `pub`: 5 now, 33 more after the owning test suite is ported |

## Recount at `origin/main` `90348dc0`

126 `#[doc(hidden)]` lines under `cuda/` (138 under `tenferro-gpu/src`) — **+3** against the issue
body's 123 (`95cd1b849`) and **−1** against the last comment's 127 (`55e6a8c06`). The one that went is
`CudaEngine::runtime_mut`, which #2556 (`e4dfa2f4`) deleted; `cuda/engine.rs` is 11 marks now. Nothing
was hidden in between.

## The generator

```bash
python3 scripts/cuda_surface_inventory.py           # docs/internals/cuda-surface-inventory.md
python3 scripts/cuda_surface_inventory.py --check   # drift check
python3 scripts/cuda_surface_inventory.py --missing # items without a verdict
python3 scripts/cuda_surface_inventory.py --list-ids
```

Every row records the path a user can actually write, kind, visibility, the mark, reachability from
`tenferro_gpu::cuda`, and references from other workspace crates / the crate's own targets / the
user-facing guides, with up to three `file:line` examples. Reachability respects module visibility:
`raw` is `pub`, while `device`, `identity` and `vendor` are `pub(crate)`, so their items are reachable
only through the `cuda::` re-exports. Counts are name-based, so they are leads rather than proof; where
the automatic count misses a real receiver call the row carries an `evidence` `file:line` that the
script checks exists in the checkout. The verdicts are reviewed input in
`docs/internals/cuda-surface-verdicts.json`, so the table regenerates after `main` moves without
re-deciding anything.

## Applied in this change

Five private-verdict items lose `pub`. Each has no reference of any kind outside
`crates/tenferro-gpu/src/cuda/` and is still read by non-test crate code, so `pub(crate)` keeps the
build warning-free with no gated test: `CudaBackend`, `CudaRuntime::{alloc_pinned, identity,
completion_deferred}`, `CudaEngine::to_contiguous`. (`CudaRuntime::identity`
was on the workspace-internal list at first: `tenferro-fft` keys its plan cache on
`CudaEngine::runtime_identity`, not on the runtime accessor.) `to_contiguous` needs
`#[expect(clippy::wrong_self_convention)]`, because the lint exempts exported API and narrowing a `to_*`
method taking `&mut self` is what surfaces it. The sixth candidate, `KernelModules::function`, is
withdrawn: the decision keeps the kernel-launch family `workspace-internal` (`pub` + `#[doc(hidden)]`).

`structural::DeviceOperand` is *not* narrowed: it is the bound on the `linalg_*` auxiliaries that
`tenferro-linalg` calls, and `pub(crate)` trips `private_bounds` on all 21 of them. Its verdict is
workspace-internal for that reason, and no workspace-internal item is touched.

33 private items keep `pub`. 32 because their only reader is `crates/tenferro-gpu/tests/seam_*.rs`
(separate crates, which cannot see a crate-private item), so removing `pub` means porting that suite;
three counters among them (`artifact_cache_hits`, `module_loads`, `PreparedFusion::key`) have no
non-test reader at all and wait for the change that gates the test-only `PreparedFusion::counters`,
following the #2312 precedent. The 33rd, `GpuExtensionCapability`, is `pub` only because narrowing an
item nobody constructs would make it dead code: it needs deleting, not hiding.

## Evidence from this machine (`broccoli`, Linux x86_64, 4 cores, no CUDA device)

- `cargo check --workspace --all-targets --features cuda,tenferro-einsum/autodiff` — passes, one
  pre-existing warning (`unused variable: trace`,
  `crates/tenferro-einsum/tests/integration/cuda_eager_tensor.rs:61`, a file this change does not touch;
  the same warning appears at `90348dc0`).
- `cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D
  clippy::missing_panics_doc` (the CI command) — clean.
- `cargo clippy -p tenferro-gpu --features cuda --all-targets -- -D warnings -D
  clippy::missing_errors_doc -D clippy::missing_panics_doc` — clean.
- `cargo doc --no-deps -p tenferro-gpu` — warning-free before and after (0 warnings both times). It does
  not document the CUDA module, which is feature-gated; the `docs.rs` metadata builds with
  `features = ["native", "cuda"]`.
- `cargo check --workspace --all-features` — **cannot run on Linux, before or after this change**:
  `blas-accelerate` pulls the macOS-only `accelerate-src` (`library kind 'framework' is only supported
  on Apple targets`), and this workspace's provider features are mutually exclusive, so
  `--all-features` is not a valid configuration anywhere. That is pre-existing and unrelated to this
  diff.
- GPU tests were not run: this host has no CUDA device (`nvidia-smi` absent, no `/usr/local/cuda-*`).
  `cargo check` compiled the `seam_*` test targets with the `cuda` feature, so no visibility or
  type-checking failure was found in them; it does not link or run those executables.

## Decisions recorded on the issue

1. **`CudaBackend` stays `pub(crate)`.** It is not the public extension point: there is no second
   implementer, and #2552 reshapes the execution contract, so publishing the seam now would freeze a
   shape that moves next.
2. **The kernel-launch family stays `workspace-internal`** (`pub` + `#[doc(hidden)]`, not `pub(crate)`):
   the RRQR kernels will need a module and launch surface, and narrowing it now means reopening it in a
   few weeks.

Full reasoning, the per-module breakdown, the per-item evidence and the ordered follow-up are in
`docs/design/cuda-surface.md` and `docs/internals/cuda-surface-inventory.md`.
