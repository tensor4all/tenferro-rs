# Issue 246 Linalg Backend Cutover Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the old slice-based `tenferro-linalg` backend boundary with a tensor-level, explicit-context backend API, and migrate the crate's public and AD APIs to that new boundary in one cutover.

**Architecture:** Keep the backend/context split consistent with `tenferro-prims`, but make the linalg backend tensor-first and op-specific. Remove the slice backend entirely, use explicit contexts everywhere, and expose only device-oriented backend/context names (`Cpu*`, `Cuda*`, `Hip*`).

**Tech Stack:** Rust, Cargo features, `tenferro-tensor`, `tenferro-device`, existing `tenferro-linalg` math implementations, `faer` under `linalg-faer`

---

### Task 1: Establish backend module layout and feature policy

**Files:**
- Create: `tenferro-linalg/src/backend/tensor_api.rs`
- Create: `tenferro-linalg/src/backend/tensor_context.rs`
- Create: `tenferro-linalg/src/backend/tensor_helpers.rs`
- Create: `tenferro-linalg/src/backend/cpu.rs`
- Create: `tenferro-linalg/src/backend/cpu_faer.rs`
- Create: `tenferro-linalg/src/backend/cpu_lapack.rs`
- Create: `tenferro-linalg/src/backend/cuda.rs`
- Create: `tenferro-linalg/src/backend/hip.rs`
- Modify: `tenferro-linalg/src/backend/mod.rs`
- Modify: `tenferro-linalg/Cargo.toml`
- Delete: `tenferro-linalg/src/backend/tensor_backend.rs`
- Delete: `tenferro-linalg/src/backend/faer_backend.rs`

**Step 1: Write the failing compile-time tests**

- Add backend feature policy tests in `tenferro-linalg/tests/` or feature-check unit tests that expect:
  - `linalg-faer` only: compiles
  - `linalg-lapack` only: compiles
  - both: compile error
  - neither: compile error

**Step 2: Run the targeted checks to verify failure**

Run:

```bash
cargo check -p tenferro-linalg --no-default-features
cargo check -p tenferro-linalg --no-default-features --features "linalg-faer linalg-lapack"
```

Expected: both fail before the feature policy is added, or fail for the wrong reason.

**Step 3: Write the minimal module and feature scaffolding**

- Add `linalg-faer` / `linalg-lapack` features to `tenferro-linalg/Cargo.toml`
- Make one of them the default (`linalg-faer`)
- Add `compile_error!` guards in `backend/mod.rs`
- Split exports into the new file layout

**Step 4: Run the feature checks again**

Run:

```bash
cargo check -p tenferro-linalg
cargo check -p tenferro-linalg --no-default-features --features linalg-faer
cargo check -p tenferro-linalg --no-default-features --features linalg-lapack
```

Expected: each valid feature set compiles to the next missing-implementation error only.

**Step 5: Commit**

```bash
git add tenferro-linalg/Cargo.toml tenferro-linalg/src/backend
git commit -m "refactor: split tensor linalg backend modules"
```

### Task 2: Define the tensor-level backend API and context bridge

**Files:**
- Modify: `tenferro-linalg/src/backend/tensor_api.rs`
- Modify: `tenferro-linalg/src/backend/tensor_context.rs`
- Modify: `tenferro-linalg/src/backend/mod.rs`

**Step 1: Write the failing tests**

- Add unit tests for:
  - cloning result structs preserves tensor shapes
  - `TensorLinalgContextFor<T>` resolves a backend type for `tenferro_prims::CpuContext`
  - old `Faer*` names are removed entirely rather than kept as aliases
  - the public API does not introduce `CudaTensorLinalgContext` or `HipTensorLinalgContext`

**Step 2: Run the targeted tests to verify failure**

Run:

```bash
cargo test -p tenferro-linalg backend:: --lib
```

Expected: missing types, missing trait implementations, or stale names.

**Step 3: Write the minimal implementation**

- Move the current tensor result structs and `TensorLinalgBackend<T>` into `tensor_api.rs`
- Add `TensorLinalgContextFor<T>` in `tensor_context.rs`
- Export only `Cpu*`, `Cuda*`, and `Hip*` names

**Step 4: Run the targeted tests again**

Run:

```bash
cargo test -p tenferro-linalg backend:: --lib
```

Expected: the API-level backend tests pass or fail only on unimplemented execution methods.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/backend
git commit -m "feat: define tensor linalg backend api"
```

### Task 3: Bind shared CpuContext and implement faer-backed tensor execution

**Files:**
- Modify: `tenferro-linalg/src/backend/cpu.rs`
- Modify: `tenferro-linalg/src/backend/cpu_faer.rs`
- Modify: `tenferro-linalg/src/backend/tensor_helpers.rs`
- Modify: `tenferro-linalg/src/lib.rs`

**Step 1: Write the failing tests**

- Add backend unit tests that call:
  - `TensorLinalgBackend::solve`
  - `solve_triangular`
  - `qr`
  - `thin_svd`
  - `lu_factor`
  - `cholesky`
  - `eigen_sym`
  - `eig`
  through `tenferro_prims::CpuContext`
- Cover real and complex representative cases

**Step 2: Run the targeted tests to verify failure**

Run:

```bash
cargo test -p tenferro-linalg backend::cpu --lib
```

Expected: failures from missing implementations or placeholder stubs.

**Step 3: Write the minimal implementation**

- Implement `CpuTensorLinalgBackend`
- Bind `CpuTensorLinalgBackend::Context = tenferro_prims::CpuContext`
- For `linalg-faer`, perform tensor-level validation, contiguous handling, and output tensor allocation in `cpu_faer.rs`
- Use slices only as an internal implementation detail inside `cpu_faer.rs`
- Ensure `eig` returns `Tensor<T::Complex>` results correctly for both real and complex input

**Step 4: Run the targeted tests again**

Run:

```bash
cargo test -p tenferro-linalg backend::cpu --lib
```

Expected: CPU tensor backend tests pass under `linalg-faer`.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/backend tenferro-linalg/src/lib.rs
git commit -m "feat: implement cpu tensor linalg backend"
```

### Task 4: Add CUDA and HIP backend/context boundaries

**Files:**
- Modify: `tenferro-linalg/src/backend/cuda.rs`
- Modify: `tenferro-linalg/src/backend/hip.rs`
- Modify: `tenferro-linalg/src/backend/mod.rs`

**Step 1: Write the failing tests**

- Add unit tests that instantiate the CUDA/HIP context types and confirm their trait surface exists
- Add tests that unimplemented methods return a stable `tenferro_device::Error` rather than panic

**Step 2: Run the targeted tests to verify failure**

Run:

```bash
cargo test -p tenferro-linalg backend::cuda --lib
cargo test -p tenferro-linalg backend::hip --lib
```

Expected: missing type or missing impl failures.

**Step 3: Write the minimal implementation**

- Define the stub backends
- Bind `CudaTensorLinalgBackend::Context = tenferro_prims::CudaContext`
- Bind `HipTensorLinalgBackend::Context = tenferro_prims::RocmContext`
- Implement `TensorLinalgBackend<T>` for each with device-error stubs
- Export the types from `backend/mod.rs`

**Step 4: Run the targeted tests again**

Run:

```bash
cargo test -p tenferro-linalg backend::cuda --lib
cargo test -p tenferro-linalg backend::hip --lib
```

Expected: stub-surface tests pass.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/backend
git commit -m "feat: add gpu tensor linalg backend boundaries"
```

### Task 5: Migrate public linalg APIs to explicit contexts

**Files:**
- Modify: `tenferro-linalg/src/lib.rs`
- Modify: `tenferro-linalg/tests/linalg_tests.rs`
- Modify: `tenferro-capi/src/lib.rs`

**Step 1: Write the failing tests**

- Convert representative public API tests to the new shape:
  - `let mut ctx = tenferro_prims::CpuContext::new(4);`
  - `solve(&mut ctx, &a, &b)`
  - `qr(&mut ctx, &a)`
  - `svd(&mut ctx, &a, None)`
- Remove stale tests using `FaerBackend` and replace them in the same change

**Step 2: Run the targeted tests to verify failure**

Run:

```bash
cargo test -p tenferro-linalg linalg_tests -- --nocapture
```

Expected: widespread compile errors due to stale `FaerBackend` signatures before the full rename lands.

**Step 3: Write the minimal implementation**

- Change crate-root public APIs from `&mut backend` to `&mut ctx`
- Dispatch through `TensorLinalgContextFor<T>`
- Keep user-facing result shaping and option handling in crate root
- Update `tenferro-capi` call sites to construct and pass `tenferro_prims::CpuContext`

**Step 4: Run the targeted tests again**

Run:

```bash
cargo test -p tenferro-linalg linalg_tests -- --nocapture
cargo test -p tenferro-capi
```

Expected: public API tests pass with context-based signatures.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/lib.rs tenferro-linalg/tests/linalg_tests.rs tenferro-capi/src/lib.rs
git commit -m "refactor: switch linalg public api to explicit contexts"
```

### Task 6: Migrate AD APIs to explicit contexts

**Files:**
- Modify: `tenferro-linalg/src/lib.rs`
- Modify: `tenferro-linalg/tests/linalg_tests.rs`

**Step 1: Write the failing tests**

- Update reverse-mode and forward-mode tests for:
  - `svd_rrule`, `svd_frule`
  - `qr_rrule`, `qr_frule`
  - `lu_rrule`, `lu_frule`
  - `eigen_rrule`, `eigen_frule`
  - `cholesky_rrule`, `cholesky_frule`
  - `solve_rrule`, `solve_frule`
  - `eig_rrule`, `eig_frule`

**Step 2: Run the targeted tests to verify failure**

Run:

```bash
cargo test -p tenferro-linalg rrule -- --nocapture
cargo test -p tenferro-linalg frule -- --nocapture
```

Expected: compile errors from stale backend-typed parameters.

**Step 3: Write the minimal implementation**

- Change all AD entrypoints to accept `&mut ctx`
- Rewire internal helper calls to the tensor-level backend path
- Remove any remaining dependence on the deleted slice backend trait

**Step 4: Run the targeted tests again**

Run:

```bash
cargo test -p tenferro-linalg rrule -- --nocapture
cargo test -p tenferro-linalg frule -- --nocapture
```

Expected: AD tests pass with the new context-based backend.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/lib.rs tenferro-linalg/tests/linalg_tests.rs
git commit -m "refactor: move linalg ad apis to explicit contexts"
```

### Task 7: Fully replace old backend names and stale references

**Files:**
- Modify: `tenferro-linalg/src/backend/mod.rs`
- Modify: `tenferro-linalg/src/lib.rs`
- Modify: `tenferro-linalg/tests/linalg_tests.rs`
- Modify: `README.md`
- Modify: crate docs that mention `FaerBackend`

**Step 1: Write the failing checks**

- Search for stale names:
  - `FaerBackend`
  - `FaerTensorLinalgBackend`
  - `FaerTensorLinalgContext`
  - `LinalgBackend<`

**Step 2: Run the searches**

Run:

```bash
rg -n "FaerBackend|FaerTensorLinalgBackend|FaerTensorLinalgContext|LinalgBackend<" .
```

Expected: stale references remain before cleanup.

**Step 3: Write the minimal cleanup**

- Remove or rename stale docs, tests, exports, and downstream references in one pass
- Do not leave aliases, deprecated type aliases, or wrapper shims
- Ensure examples use `tenferro_prims::CpuContext`
- Keep docs aligned with the new explicit-context model

**Step 4: Run the searches again**

Run:

```bash
rg -n "FaerBackend|FaerTensorLinalgBackend|FaerTensorLinalgContext|LinalgBackend<" tenferro-linalg tenferro-capi README.md
```

Expected: only intentionally retained internal references remain, or none.

**Step 5: Commit**

```bash
git add tenferro-linalg tenferro-capi README.md
git commit -m "docs: remove stale linalg backend names"
```

### Task 8: Full verification and coverage

**Files:**
- Modify as needed based on failures

**Step 1: Run formatting**

Run:

```bash
cargo fmt --all
```

**Step 2: Run workspace tests**

Run:

```bash
cargo test --workspace
```

Expected: all tests pass with the new context-first API.

**Step 3: Run coverage**

Run:

```bash
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Expected: all files meet thresholds.

**Step 4: Run feature-matrix checks**

Run:

```bash
cargo check -p tenferro-linalg --no-default-features --features linalg-faer
cargo check -p tenferro-linalg --no-default-features --features linalg-lapack
cargo check -p tenferro-linalg --no-default-features
cargo check -p tenferro-linalg --no-default-features --features "linalg-faer linalg-lapack"
```

Expected:

- first two succeed (or `linalg-lapack` reaches only intentional stub errors if not fully implemented yet)
- last two fail with the intended compile errors

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: cut over linalg to tensor backend contexts"
```
