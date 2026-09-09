# CPU static replay and basic/fused boundaries

## Scope and provenance

This work follows the maintainer-approved compile-time investigation: use crate
boundaries for lightweight CPU use, retain ordinary CPU automatic fusion, and
share typed replay without dynamic callbacks. Crate-path compatibility is not a
requirement. No package publication is authorized.

The tenferro implementation is rebased onto current main
`167a8d28dde14cf2d601ab1f7c397ae4de25ca7b`.
Its strided dependency is now the current split candidate
`1db88be3bba8882f1c1a1de00315fc341648e75e` (strided-rs PR #254), which includes
the consumer's prerequisite indexed-kernel work. The initial experiments used
older commits and are not final performance evidence.

Read: current repository rules and PERFORMANCE_TIPS, current backend/session and
kernel callers, buffer guard source contracts, the exact strided pin candidate,
and the earlier static-sharing transforms. Existing unpublished user changes in
the original worktree were not copied or modified.

## Implemented

- Static replay sharing remains in rebased commit `7c7b2a8`, preserving
  owned/read arithmetic,
  analytic/pow, unary, select/clamp and ordered min/max behavior, allocation and
  validation order, integer domain checks, complex abs dtype, and specialized
  multiply/comparison kernels.
- Added `tenferro-cpu-basic`, which owns the reusable host buffer pool,
  full-overwrite `PooledUninitOutput`, host strided adapters, raw descriptor
  boundary helpers, and shared complex-order validation.
- `tenferro-internal-cpu-kernels` now owns ordinary dtype-dispatch kernels and
  pool-aware one-shot elementwise read-into replay. It consumes the pool and
  output guard from basic; their implementations are not duplicated.
- Added `tenferro-cpu-fused`, which owns the dtype-erased runtime-DAG fusion
  adapter and depends on `strided-fused` plus basic, not on the ordinary kernel
  implementation. `CpuBackend` and `CpuExecSession` call it with the existing
  installed context and borrowed pool, preserving automatic fusion and resource
  identity.
- Moved `TensorElementwise::elementwise_read_into`'s CPU host one-shot
  implementation out of `tenferro-tensor` into the internal CPU kernel crate;
  tensor no longer depends on `strided-kernel`. The trait hook is required, and
  CUDA uses the explicit allocating backend helper while WebGPU returns its
  typed unsupported error. CPU and GPU session adapters delegate explicitly.
- Moved affected tests with their owners and updated test-only backend
  implementations, source-contract paths, trybuild stderr, architecture docs,
  and the architecture SVG. No user-facing facade or compatibility shim was
  added.

## Verification

Wrappers disabled; Cargo jobs = 4. Correctness runs used one test-harness thread,
with Rayon pinned to one where serial behavior was being checked and explicit
four-worker contexts for parallel suites. These are not timing measurements.

Passed checks include:

- `cargo test --profile ci --workspace --lib --tests --offline` after the rebase:
  **3268 passed, 130 ignored, 0 failed**; the run excluded doctests by design.
- Focused suites: tensor **275**, CPU **538**, CPU integration **50**, ordinary
  internal kernels **25**, fused adapter **6**, einsum session tests **3**,
  runtime integration **146**, linalg integration **160**, FFT capability **6**.
- New basic pool tests **32** and new fused/read-into tests pass; all moved
  read-into and ownership/source-contract checks pass.
- `cargo check -p tenferro-gpu --features cuda` and `--features webgpu` pass;
  CPU `cpu-faer`, `cpu-blas`, and both-provider feature checks also pass.
- Workspace `--no-run` compilation passes for all test targets, and the full
  workspace doctest run after the rebase passes (**1832 doctests**, 0 failed).
- `cargo fmt --all`, `git diff --check`, public-error-docs, docs-site and
  publish-layout checks pass. The initial trybuild mismatch was only a compiler diagnostic
  span after removing the tensor implementation; the checked stderr was
  regenerated and the test passed.
- Full workspace llvm-cov was run: the changed/new basic, fused and read-into
  files meet their thresholds. The repository baseline still reports 41
  unrelated GPU/FFT/linalg/tutorial files below existing thresholds, so the
  aggregate checker remains nonzero; no unrelated threshold was lowered.
- One predeclared cold build pair (`cargo build -p tenferro-cpu`, jobs=4,
  wrappers disabled, fresh targets, pinned provider variables) measured baseline
  **49.02 s** and candidate **49.36 s**. The host was an AMD EPYC 7713P with
  64 CPUs and load averages 7.42/20.10/16.58; under the declared noise policy
  this pair is **INCONCLUSIVE**, not a performance-neutrality claim. Raw logs,
  times, environment and protocol are retained under `/tmp/tenferro-build-*`.
- `--all-features --no-run` is not a valid Linux check because the existing
  Accelerate provider requests an Apple framework; the applicable provider and
  GPU feature combinations above pass.
- Cargo metadata confirms `tenferro-tensor` has no normal strided dependency;
  `tenferro-cpu-basic` depends only on tensor/basic strided primitives, fused
  depends on basic + `strided-fused`, and ordinary kernels remain the only
  owner of the normal `strided-kernel` dependency.

An earlier pre-rebase full workspace test command timed out while running the
large doctest sequence; its partial log is retained at
`/tmp/tenferro-split-workspace-tests.log` and is not counted as evidence. The
post-rebase lib/test and doctest matrices completed successfully.
No CUDA hardware execution or final runtime benchmark has yet been run on this
tenferro candidate. Release verification passed for the three kernel owners
(**87 tests**) and CPU integration (**50 tests**). `cargo package --no-verify`
was attempted for the new CPU packages and correctly stopped because
`strided-basic` is not yet on crates.io; this is the expected publication-order
blocker while strided PR #254 is unmerged/unpublished, not a package-success
claim. The build pair was run but is inconclusive because of host load, and
full aggregate coverage still contains unrelated baseline failures as described
above.

## Remaining before PR/merge readiness

- Reconcile the temporary `1db88be` strided git pin after strided PR #254 is
  merged and use the final merged commit for a tenferro PR. Until then, this
  worktree is a local integration candidate, not a publishable package state.
- Run the repository's final local PR gate, applicable feature/no-default and
  release checks, final diff/self-review, and resolve any new coverage finding.
  Hosted CI owns the broader matrix after PR creation.
- Measure the final fusion-preserving light/extensions/combined configurations
  under a predeclared protocol. Historical fusion-disabled timings and the
  earlier scalar-division regression do not certify this composition.
- Create the tenferro PR only after the exact dependency/order decision is
  resolved. Do not merge or publish without explicit authorization.
