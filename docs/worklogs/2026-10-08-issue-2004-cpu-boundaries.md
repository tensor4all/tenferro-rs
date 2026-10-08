# Issue #2004 — CPU boundary integration

Status: implemented locally on `refactor/2004-cpu-boundaries`; one integration
PR, required CI, and the consumer migrations in `tenferro-benchmark` and
`tensor4all-rs` are still pending.

## Decisions

- The CPU provider is compile-time. `native` (the default) keeps
  cpueinsum/tprims plus tlinalg; `blas` adds `cpueinsum-blas`/`tlinalg-blas` as
  compiled adapters with the native path kept for declined steps. Enabling both,
  or neither, is a `compile_error!`.
- CPU floating/complex `dot_general`, grouped GEMM, and N-ary concrete einsum
  delegate to cpueinsum; CPU linalg delegates to tlinalg/tlinalg-blas. tenferro
  keeps tensor semantics, placement, output allocation, axis/layout/dtype
  validation, the `MaybeUninit` complete-coverage proof, and the typed
  unsupported mapping.
- Lane selection, vendor batching, and scratch belong to the lower library.
  tenferro passes one parallelism token: `Sequential`, or the selected pool with
  a thread budget. The host `CpuBatchPolicy`, its thresholds, and the host
  tlinalg lane plan are gone.
- tenferro uses only pools it builds. The external executor/domain surface
  (`ExternalCpuDomain`, `CpuDomainExecutor`, caller-managed admission,
  `for_domain`, `from_external_managed_domains*`), the runtime provider bundle
  and kernel slots, and the runtime provider kind are removed.
- Session entry is admission-only: permit, caller-affinity guard, workspace
  lease, callback on the calling thread. Affinity narrows to `O ∩ D` and never
  widens; `AllAllowed` performs no syscall. Reentry rejection relies on
  caller-local state and non-blocking worker admission instead of pool-wide
  worker-scope registration.
- `CpuBackend::builder()` is the placement API (`threads`, `cpus`, `numa_node`,
  `worker_stack`, `buffer_limit`). `CpuContext` is a doc-hidden
  internal resource holder (reachable only by this crate's benchmark), and the
  cross-crate seam is `CpuExecutionContext`.

## Deleted

`ext/tenferro-cpu-tblis`, `ext/tenferro-cpu-tprims`, `third_party/t4a-tblis-src`
and their gates; `provider.rs`'s request/trait machinery; the old
`dot_runtime`/`gemm` provider route; `domain_executor.rs`,
`provider_capability.rs`, `provider_extensions.rs`, `batch_policy.rs`,
`affinity_policy.rs`; and the tests that exercised them.

## Verification

- `cargo check -j 16 --workspace --all-targets`: 0 errors.
- `cargo clippy -j 16 --workspace --all-targets -- -D warnings
  -D clippy::missing_errors_doc -D clippy::missing_panics_doc`: clean.
- `cargo check -j 16 -p tenferro-cpu --no-default-features --features blas
  --all-targets`: 0 errors; both-enabled and neither-enabled hit the
  `compile_error!`.
- `cargo test -j 16 --workspace --no-fail-fast`: only three **trybuild UI**
  targets fail, because the kache rustc wrapper rewrites diagnostic paths
  (`/kache/...`, `/proc/self/cwd/...`). With `RUSTC_WRAPPER= cargo test -p
  tenferro-ad --test integration eager_backend_capability_boundary` the four UI
  fixtures pass, so this is a machine-local environment artifact.
- `python3 scripts/check-doc-snippets.py`,
  `scripts/test-doc-consistency.py`, `scripts/check-crate-boundaries.py`,
  `scripts/audit-session-entry.py`, `scripts/test-check-boundary-scope.py`,
  `scripts/check-public-boundary-inventory.py`, `scripts/check-agent-skills.py`,
  `scripts/check-prelude-contract.py`, `scripts/check-operation-categories.py`,
  `scripts/check-guide-dependency-snippets.py`, `scripts/check-api-consistency.py`,
  `scripts/check-publish-layout.py`: pass.

## Independent post-review (gpt-6.1-sol) and corrections

The finished diff was reviewed read-only by a separate model. Findings accepted
and fixed in the same session:

- **BLOCKER — `blas,provider-inject` did not compile.** `inject_tests.rs` still
  used the removed provider/executor API. It is now a focused
  symbol-registration test on the public `CpuBackend`, and the dual-ABI
  registration tests share one process-wide lock with it because the injected
  pointer registry is global.
- **Integer contractions wrote instead of rejecting.** `dot_general_read_into`
  reached the removed non-floating elementwise path. Non-floating dtypes now
  reject before any output mutation, and the dead all-batch elementwise path
  plus its dispatch is deleted.
- **Floating paths skipped host validation.** `dot_fresh`/`dot_into` now call
  the shared host validators (`validate_dot_operands` / `validate_dot_general`),
  grouped GEMM validates host placement, and the CPU N-ary einsum route
  validates input/output placement through a doc-hidden helper.
- **Builder-selected NUMA/CPU-set placement was treated as the wildcard.** The
  domain now carries an explicit caller-affinity target, so an explicit
  `.cpus(...)`/`.numa_node(...)` narrows the calling thread to `O ∩ D`.
- **Retention-limit mutation raced lazy engine creation.** The buffer-limit and
  reset setters take the configuration guard the lazy constructor already uses.
- **Ambient Rayon in FFT and one faer matmul.** Both install the engine's
  selected pool for the duration of the numerical call.
- **Lower route declines were classified as backend failures.** A lower
  `Unsupported`/declined step now maps to `Unsupported` in the binary, grouped,
  and N-ary adapters, with the lower diagnostic retained in the message.
- **Cache shrink did not count evictions.** `BinaryCache`/`GroupCache`
  `set_capacity` records evicted entries.
- **Test-only executor-install counter was vacuous** and is removed; the
  doc-hidden `CpuContext` is documented as internal rather than crate-private;
  `builder()` is infallible and `worker_stack` validates its minimum.

Findings not fixed here:

- `tlinalg-blas` (the pinned revision) has no parallelism-token entry point, so
  the BLAS linalg adapter cannot pass the selected token the way the faer
  adapter does. The requirement is **amended** rather than the provider
  extended: `tlinalg-blas` documents that it deliberately takes no token (its
  batch loop is serial, LAPACK/BLAS own their threading, and a Rayon fan-out
  around a vendor call would fight the vendor's pool), and the accepted design
  has tenferro place the vendor call on the coordinator thread instead. Adding a
  token would put an inert parameter on 19 entry points. The contract is now
  stated in `docs/design/cpu-backend-execution.md` and in the adapter's module
  documentation.
- `CpuBackend::with_threads_isolated_arbiter_for_test` remains a doc-hidden
  public test helper used by `tenferro-ad`'s own test.
