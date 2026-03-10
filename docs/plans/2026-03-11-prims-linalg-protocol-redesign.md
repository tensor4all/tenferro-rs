# Prims and Linalg Protocol Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the old monolithic prim protocol with a family-based semiring/scalar/linalg substrate, keep `einsum` performance intact, and reorganize `docs/design/` so the new architecture is easy to understand.

**Architecture:** Introduce new protocol families in `tenferro-prims`, add a new `tenferro-linalg-prims` crate for backend-facing factorization/solve kernels, migrate `tenferro-einsum` and `tenferro-linalg` onto those layers without compatibility shims, and update design docs plus benchmark gates to reflect the final architecture.

**Tech Stack:** Rust workspace crates, trait-based protocol redesign, existing CPU/CUDA backends, oracle replay tests, sibling `tenferro-einsum-benchmark`, rustdoc design docs.

---

### Task 1: Add top-level protocol migration tests

**Files:**
- Modify: `tenferro-prims/src/tests/mod.rs`
- Modify: `tenferro-einsum/src/tests/mod.rs`
- Modify: `tenferro-linalg/tests/oracle_db/main.rs`

**Step 1: Write failing semiring-core smoke tests**

Add tests that express the target contract:

- `tenferro-einsum` compiles and runs using semiring-core traits only
- protocol tests no longer depend on prim `Permute`
- oracle/linalg tests can still be expressed through prim contracts

**Step 2: Run the targeted tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims protocol_smoke -- --nocapture
cargo test -p tenferro-einsum semiring_core_only -- --nocapture
```

Expected: failures caused by the old protocol surface still being in use.

**Step 3: Commit the failing tests**

```bash
git add tenferro-prims/src/tests/mod.rs tenferro-einsum/src/tests/mod.rs tenferro-linalg/tests/oracle_db/main.rs
git commit -m "test: expose protocol migration targets"
```

### Task 2: Introduce the new semiring-family protocol surface in `tenferro-prims`

**Files:**
- Modify: `tenferro-prims/src/lib.rs`
- Create: `tenferro-prims/src/semiring_core.rs`
- Create: `tenferro-prims/src/semiring_fast_path.rs`
- Create: `tenferro-prims/src/scalar_prims.rs`
- Create: `tenferro-prims/src/analytic_prims.rs`

**Step 1: Define new public family traits and descriptors**

Add the new families:

- `TensorSemiringCore<Alg: Semiring>`
- `TensorSemiringFastPath<Alg: Semiring>`
- `TensorScalarPrims<Alg>`
- `TensorAnalyticPrims<Alg>`

and the new descriptor enums / op enums for each family.

**Step 2: Remove `Permute` from the public prim surface**

Delete the public `Permute` descriptor from the new protocol and make the
surface reflect structural-view ownership by `tenferro-tensor`.

**Step 3: Run the targeted protocol tests**

Run:

```bash
cargo test -p tenferro-prims protocol_smoke -- --nocapture
```

Expected: tests still fail on missing backend implementations, but compile
against the new public surface.

**Step 4: Commit**

```bash
git add tenferro-prims/src/lib.rs tenferro-prims/src/semiring_core.rs tenferro-prims/src/semiring_fast_path.rs tenferro-prims/src/scalar_prims.rs tenferro-prims/src/analytic_prims.rs
git commit -m "feat: add family-based prim protocol surface"
```

### Task 3: Port CPU semiring core and fast path implementations without changing hot loops

**Files:**
- Modify: `tenferro-prims/src/cpu.rs`
- Modify: `tenferro-prims/src/tests/mod.rs`

**Step 1: Add failing CPU execution tests for the new descriptors**

Cover:

- `ReduceAdd`
- `Trace`
- `AntiTrace`
- `AntiDiag`
- `MakeContiguous`
- `ElementwiseBinary { Add, Mul }`

**Step 2: Lower new public descriptors to specialized CPU plans**

Keep specialized internal plans such as dedicated `ElementwiseMul`-style paths.
Do not leave per-element dynamic dispatch in execution loops.

**Step 3: Re-run the CPU protocol tests**

Run:

```bash
cargo test -p tenferro-prims cpu_protocol -- --nocapture
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tenferro-prims/src/cpu.rs tenferro-prims/src/tests/mod.rs
git commit -m "feat: port cpu backend to family-based semiring protocol"
```

### Task 4: Port CUDA/stub protocol support for semiring core and fast paths

**Files:**
- Modify: `tenferro-prims/src/cuda.rs`
- Modify: `tenferro-prims/src/gpu_stubs.rs`
- Modify: `tenferro-prims/src/cuda/tests/mod.rs`

**Step 1: Add failing capability and plan tests for the new GPU protocol**

Cover:

- semiring fast-path capability queries
- descriptor planning for `Contract` and semiring binary ops

**Step 2: Rewire CUDA/stub planning to the new family enums**

Preserve existing optimized mappings such as cuTENSOR contraction and binary
elementwise execution where already supported.

**Step 3: Re-run GPU/stub protocol tests**

Run:

```bash
cargo test -p tenferro-prims cuda_protocol -- --nocapture
```

Expected: PASS for CUDA-enabled or stub expectations.

**Step 4: Commit**

```bash
git add tenferro-prims/src/cuda.rs tenferro-prims/src/gpu_stubs.rs tenferro-prims/src/cuda/tests/mod.rs
git commit -m "feat: port gpu protocol surface to family-based prims"
```

### Task 5: Migrate `tenferro-einsum` to `TensorSemiringCore` / `TensorSemiringFastPath`

**Files:**
- Modify: `tenferro-einsum/src/lib.rs`
- Modify: `tenferro-einsum/src/dispatch.rs`
- Modify: `tenferro-einsum/src/prepare.rs`
- Modify: `tenferro-einsum/src/tests/mod.rs`

**Step 1: Write failing tests that enforce the final lowering**

Keep or extend tests that assert:

- CPU uses `permute view -> MakeContiguous -> BatchedGemm`
- dynamic semiring fast path for `ElementwiseBinary(Mul)` still works
- GPU prefers `Contract` when capability is present

**Step 2: Port `tenferro-einsum` to the new trait bounds**

Ensure the crate depends only on `TensorSemiringCore`, optionally checking
`TensorSemiringFastPath` capabilities for optimized routes.

**Step 3: Re-run `tenferro-einsum` tests**

Run:

```bash
cargo test -p tenferro-einsum --lib
```

Expected: PASS with the same current lowering behavior.

**Step 4: Commit**

```bash
git add tenferro-einsum/src/lib.rs tenferro-einsum/src/dispatch.rs tenferro-einsum/src/prepare.rs tenferro-einsum/src/tests/mod.rs
git commit -m "refactor: move einsum to semiring core and fast path traits"
```

### Task 6: Restore tropical + AD minimum coverage on the reduced semiring core

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/api/ad.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/tests/mod.rs`
- Modify: `docs/AD/scalar_ops.md`

**Step 1: Add failing tropical/minimum-AD regression tests**

Add tests that prove:

- tropical contraction still works via semiring core
- diagonal/scatter AD paths still use `Trace`, `AntiTrace`, `AntiDiag`

**Step 2: Port the AD plumbing to the new semiring trait families**

Keep AD dependent only on the semiring core for these minimum paths.

**Step 3: Re-run the targeted tests**

Run:

```bash
cargo test -p tenferro-dyadtensor tropical -- --nocapture
```

Expected: PASS.

**Step 4: Commit**

```bash
git add extension/tenferro-dyadtensor/src/api/ad.rs extension/tenferro-dyadtensor/src/api/tests/mod.rs docs/AD/scalar_ops.md
git commit -m "fix: restore tropical and minimum ad coverage on semiring core"
```

### Task 7: Add the new scalar and analytic protocol families

**Files:**
- Modify: `tenferro-prims/src/lib.rs`
- Modify: `tenferro-prims/src/cpu.rs`
- Modify: `tenferro-prims/src/cuda.rs`
- Modify: `tenferro-prims/src/tests/mod.rs`

**Step 1: Add failing tests for pointwise and reduction vocabulary**

Cover representative ops from:

- scalar unary
- scalar binary
- scalar reduction
- analytic unary/binary

**Step 2: Implement capability-query-driven planning**

Specialize public op enums into dedicated backend plans. Keep the current
optimized elementwise paths intact and extend them without introducing hot-loop
dynamic dispatch.

**Step 3: Re-run scalar/analytic tests**

Run:

```bash
cargo test -p tenferro-prims scalar_protocol -- --nocapture
cargo test -p tenferro-prims analytic_protocol -- --nocapture
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tenferro-prims/src/lib.rs tenferro-prims/src/cpu.rs tenferro-prims/src/cuda.rs tenferro-prims/src/tests/mod.rs
git commit -m "feat: add scalar and analytic prim families"
```

### Task 8: Create the new `tenferro-linalg-prims` crate

**Files:**
- Create: `tenferro-linalg-prims/Cargo.toml`
- Create: `tenferro-linalg-prims/src/lib.rs`
- Create: `tenferro-linalg-prims/src/tests/mod.rs`
- Modify: `Cargo.toml`
- Modify: `docs/design/index.md`

**Step 1: Add the new crate to the workspace**

Create the crate and wire it into `[workspace.members]` and any necessary
workspace dependencies.

**Step 2: Define failing protocol tests for kernel-basis operations**

Cover the kernel inventory:

- Cholesky factor
- LU factor / solve
- triangular solve
- QR factor
- Householder product
- SVD factor
- eigen
- least squares

**Step 3: Define the new public kernel contracts**

Add the new linalg kernel traits, result structs, and capability query surface.

**Step 4: Run the crate tests**

Run:

```bash
cargo test -p tenferro-linalg-prims
```

Expected: compile-only or contract tests pass; backend wiring may still be pending.

**Step 5: Commit**

```bash
git add Cargo.toml tenferro-linalg-prims/Cargo.toml tenferro-linalg-prims/src/lib.rs tenferro-linalg-prims/src/tests/mod.rs docs/design/index.md
git commit -m "feat: add tenferro-linalg-prims crate"
```

### Task 9: Port backend-facing linalg execution under `tenferro-linalg-prims`

**Files:**
- Modify: `tenferro-linalg/src/backend/mod.rs`
- Modify: `tenferro-linalg/src/backend/faer_backend.rs`
- Modify: `tenferro-linalg/src/backend/tensor_api.rs`
- Modify: `tenferro-linalg-prims/src/lib.rs`
- Modify: `tenferro-linalg/tests/linalg_tests.rs`

**Step 1: Add failing linalg-kernel tests against the new contracts**

Cover representative factorization/solve kernels via the new protocol.

**Step 2: Move backend-local linalg execution behind `tenferro-linalg-prims`**

The old backend execution logic should become the initial implementation of the
new kernel contracts rather than staying as an ad hoc `tenferro-linalg`
internal boundary.

**Step 3: Re-run targeted linalg kernel tests**

Run:

```bash
cargo test -p tenferro-linalg linalg_tests -- --nocapture
```

Expected: PASS for migrated kernel contracts.

**Step 4: Commit**

```bash
git add tenferro-linalg/src/backend/mod.rs tenferro-linalg/src/backend/faer_backend.rs tenferro-linalg/src/backend/tensor_api.rs tenferro-linalg-prims/src/lib.rs tenferro-linalg/tests/linalg_tests.rs
git commit -m "refactor: move linalg execution behind linalg prim contracts"
```

### Task 10: Rewrite `tenferro-linalg` as a pure public/composite lowering layer

**Files:**
- Modify: `tenferro-linalg/src/lib.rs`
- Modify: `tenferro-linalg/tests/oracle_db/replay.rs`
- Modify: `tenferro-linalg/tests/oracle_db/support.rs`

**Step 1: Add failing tests for composite lowering-only behavior**

Cover cases such as:

- `matrix_power`
- `cond`
- `tensorinv`
- `tensorsolve`

where the implementation must lower through prims rather than backend-local
code.

**Step 2: Remove backend-name checks and direct execution paths**

Rewrite `tenferro-linalg` so all execution is routed through:

- `tenferro-prims`
- `tenferro-linalg-prims`

with no direct `CpuBackend` / `CudaBackend` branching in the public/composite
layer.

**Step 3: Re-run linalg and oracle replay tests**

Run:

```bash
cargo test -p tenferro-linalg --test oracle_db
cargo test -p tenferro-linalg linalg_tests -- --nocapture
```

Expected: PASS.

**Step 4: Commit**

```bash
git add tenferro-linalg/src/lib.rs tenferro-linalg/tests/oracle_db/replay.rs tenferro-linalg/tests/oracle_db/support.rs
git commit -m "refactor: make tenferro-linalg a pure lowering layer"
```

### Task 11: Migrate dyadtensor and oracle replay to the new protocol stack

**Files:**
- Modify: `extension/tenferro-dyadtensor/src/api/mod.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/tests/mod.rs`
- Modify: `docs/generated/tensor-ad-oracles-support.md`

**Step 1: Add failing builder and replay regression tests**

Cover:

- primal builder availability
- AD surface availability
- oracle replay support accounting

under the new prim/linalg-prims substrate.

**Step 2: Port builder/replay wiring**

Keep the public dyadtensor and oracle behavior, but route all execution through
the new protocol layers.

**Step 3: Re-run targeted tests**

Run:

```bash
cargo test -p tenferro-dyadtensor
cargo test -p tenferro-linalg --test oracle_db
```

Expected: PASS.

**Step 4: Commit**

```bash
git add extension/tenferro-dyadtensor/src/api/mod.rs extension/tenferro-dyadtensor/src/api/tests/mod.rs docs/generated/tensor-ad-oracles-support.md
git commit -m "refactor: migrate dyadtensor and oracle replay to new protocol stack"
```

### Task 12: Remove the old protocol surface completely

**Files:**
- Modify: `tenferro-prims/src/lib.rs`
- Modify: `tenferro-einsum/src/lib.rs`
- Modify: `tenferro-linalg/src/lib.rs`

**Step 1: Delete the old monolithic trait/descriptors**

Remove the remaining old `TensorPrims` / `PrimDescriptor` / `Extension`
surface and any last internal users.

**Step 2: Run workspace compile and targeted tests**

Run:

```bash
cargo test -p tenferro-prims
cargo test -p tenferro-einsum --lib
cargo test -p tenferro-linalg linalg_tests -- --nocapture
```

Expected: PASS with no references to the old surface.

**Step 3: Commit**

```bash
git add tenferro-prims/src/lib.rs tenferro-einsum/src/lib.rs tenferro-linalg/src/lib.rs
git commit -m "refactor: remove old prim protocol surface"
```

### Task 13: Reorganize `docs/design` so the final architecture is readable

**Files:**
- Modify: `docs/design/architecture.md`
- Modify: `docs/design/index.md`
- Modify: `docs/design/tensor-prims.md`
- Create: `docs/design/linalg-prims.md`
- Modify: `docs/design/linalg.md`
- Modify: `docs/design/testing.md`
- Modify: `docs/design/linalg-backend-api.md`
- Modify: `docs/design/linalg-gemm-prims.md`

**Step 1: Add a failing docs-site expectation**

Update docs-site checks or a small golden-text assertion so the final design map
must expose:

- semiring core
- semiring fast path
- scalar/analytic families
- `tenferro-linalg-prims`
- `tenferro-linalg` as lowering/composite

**Step 2: Rewrite the canonical docs**

Make the design docs readable as one coherent story:

- `index.md` becomes the canonical map
- `architecture.md` shows final crate layering
- `tensor-prims.md` becomes the canonical protocol doc
- `linalg-prims.md` explains the new crate
- `linalg.md` explains the public/composite lowering layer

Absorb or clearly supersede the old proposal docs instead of leaving ambiguous
overlap.

**Step 3: Re-run docs checks**

Run:

```bash
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 4: Commit**

```bash
git add docs/design/architecture.md docs/design/index.md docs/design/tensor-prims.md docs/design/linalg-prims.md docs/design/linalg.md docs/design/testing.md docs/design/linalg-backend-api.md docs/design/linalg-gemm-prims.md
git commit -m "docs: reorganize design docs for the new protocol stack"
```

### Task 14: Run final verification including external `einsum` benchmark gating

**Files:**
- Modify: `docs/design/testing.md`
- Modify: `README.md`

**Step 1: Run full workspace verification**

Run:

```bash
cargo fmt --all --check
cargo test --workspace --release
cargo llvm-cov --workspace --release --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: all commands pass.

**Step 2: Run the external `einsum` benchmark gate**

Run from the sibling benchmark repo:

```bash
cd ../tenferro-einsum-benchmark
BENCH_INSTANCE=matrix_chain cargo run --release
BENCH_INSTANCE=perturbed_quantum_circuit cargo run --release
```

Then run the broader benchmark sweep appropriate for the machine.

Expected: no material regression versus the pre-refactor baseline.

**Step 3: Run in-tree hot-path benchmarks**

Run:

```bash
cargo bench -p tenferro-prims
cargo bench -p tenferro-linalg --bench linalg_benchmarks
```

Expected: no new hot-path regressions for scalar/elementwise and linalg kernel
paths.

**Step 4: Commit**

```bash
git add README.md docs/design/testing.md
git commit -m "chore: document verification and benchmark gates"
```
