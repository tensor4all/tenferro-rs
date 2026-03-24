# ATen Dense Eager Core Compatibility Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an ATen-aligned dense eager core for deterministic constructors, metadata/int/bool/real/complex eager execution, and RNG, then rewrite the remaining tensor-linalg host cleanup paths onto that substrate.

**Architecture:** Implement the dense eager core bottom-up. `tenferro-device` owns raw kernels/runtime/generator state, `tenferro-tensor` owns object/view/materialize/constructor APIs, `tenferro-prims` owns family execution and mixed-dtype bridge semantics, and `tenferro-linalg` consumes the completed substrate without adding new low-level helpers.

**Tech Stack:** Rust workspace crates `tenferro-device`, `tenferro-tensor`, `tenferro-prims`, `tenferro-linalg-prims`, `tenferro-linalg`; CUDA runtime-loaded kernels; Philox for CUDA RNG; CPU generator engine; `strided-view`; existing family `plan/execute` protocols.

---

## Task 1: Add deterministic constructor regression tests

**Files:**
- Create: `tenferro-tensor/src/tests/constructors_phase2.rs`
- Modify: `tenferro-tensor/src/tests/mod.rs`

**Step 1: Write the failing tests**

Add tests for:

- `empty` shape/layout/device semantics
- `empty_strided` layout validation
- `full`
- `empty_like`
- `zeros_like`
- `ones_like`
- `full_like`
- `arange`
- `linspace`

Include CPU tests and `#[cfg(feature = "cuda")]` parity tests.

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor --lib constructors_phase2
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-tensor --features cuda --lib constructors_phase2
```

Expected: FAIL because the constructors do not exist or do not satisfy the desired contract.

**Step 3: Write minimal implementation**

None in this task.

**Step 4: Re-run to verify the failure is still the expected missing-feature failure**

Expected: FAIL for missing constructors / wrong semantics.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/tests/constructors_phase2.rs \
        tenferro-tensor/src/tests/mod.rs
git commit -m "test: add deterministic constructor regressions"
```

## Task 2: Implement deterministic constructors in `tenferro-tensor`

**Files:**
- Modify: `tenferro-tensor/src/tensor/constructors.rs`
- Modify: `tenferro-tensor/src/tensor/mod.rs`

**Step 1: Reuse the failing tests from Task 1**

No new tests.

**Step 2: Run the focused constructor tests**

Run:

```bash
cargo test -p tenferro-tensor --lib constructors_phase2
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Add:

- `empty`
- `empty_strided`
- `full`
- `empty_like`
- `zeros_like`
- `ones_like`
- `full_like`
- `arange`
- `linspace`

Implementation rules:

- CPU-first
- preserve existing `LogicalMemorySpace` / `MemoryOrder` conventions
- keep constructor logic in `tenferro-tensor`
- use `tenferro-device` only for allocation/transfer/runtime support, not family protocol

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor --lib constructors_phase2
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-tensor --features cuda --lib constructors_phase2
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/tensor/constructors.rs \
        tenferro-tensor/src/tensor/mod.rs
git commit -m "feat: add deterministic dense eager constructors"
```

## Task 3: Add metadata phase-2 regression tests

**Files:**
- Create: `tenferro-prims/src/tests/metadata_phase2.rs`
- Modify: `tenferro-prims/src/tests/mod.rs`

**Step 1: Write the failing tests**

Cover:

- `i32 + i32`
- `i32 - i32`
- `i32 * i32`
- `bool bitand`
- metadata `where`
- metadata `sum/all/any`
- broadcast sanity

Add CPU and CUDA tests.

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims --lib metadata_phase2
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib metadata_phase2
```

Expected: FAIL.

**Step 3: Write minimal implementation**

None in this task.

**Step 4: Re-run to confirm the failure is still the intended missing-op failure**

Expected: FAIL.

**Step 5: Commit**

```bash
git add tenferro-prims/src/tests/metadata_phase2.rs \
        tenferro-prims/src/tests/mod.rs
git commit -m "test: add metadata phase 2 regressions"
```

## Task 4: Implement metadata phase-2 CPU/CUDA support

**Files:**
- Modify: `tenferro-prims/src/cpu/metadata.rs`
- Modify: `tenferro-prims/src/cuda/metadata.rs`
- Modify: `tenferro-device/src/cuda/runtime/pointwise/pointwise_metadata.rs`
- Modify: `tenferro-device/src/cuda/runtime/kernels/metadata_scalar.rs`

**Step 1: Reuse the failing tests from Task 3**

No new tests.

**Step 2: Run focused tests**

Run:

```bash
cargo test -p tenferro-prims --lib metadata_phase2
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib metadata_phase2
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Add metadata binary support for:

- `Add`
- `Sub`
- `Mul`
- `BitAnd`

and metadata ternary/reduction coverage required by the tests.

Keep module split strict and avoid adding linalg-specific metadata helpers.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-prims --lib metadata_phase2
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib metadata_phase2
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-prims/src/cpu/metadata.rs \
        tenferro-prims/src/cuda/metadata.rs \
        tenferro-device/src/cuda/runtime/pointwise/pointwise_metadata.rs \
        tenferro-device/src/cuda/runtime/kernels/metadata_scalar.rs
git commit -m "feat: add metadata phase 2 eager ops"
```

## Task 5: Add metadata-to-scalar bridge regression tests

**Files:**
- Create: `tenferro-prims/src/tests/metadata_bridge_phase1.rs`
- Modify: `tenferro-prims/src/tests/mod.rs`

**Step 1: Write the failing tests**

Cover:

- bool metadata -> `f32`
- bool metadata -> `f64`
- `i32` metadata -> `f32`
- `i32` metadata -> `f64`
- metadata mask applied to scalar `where`
- same-shape CPU/CUDA parity

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims --lib metadata_bridge_phase1
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib metadata_bridge_phase1
```

Expected: FAIL.

**Step 3: Write minimal implementation**

None in this task.

**Step 4: Re-run to confirm the intended failure**

Expected: FAIL.

**Step 5: Commit**

```bash
git add tenferro-prims/src/tests/metadata_bridge_phase1.rs \
        tenferro-prims/src/tests/mod.rs
git commit -m "test: add metadata bridge regressions"
```

## Task 6: Implement metadata-to-scalar cast/select bridge

**Files:**
- Create: `tenferro-prims/src/families/cast.rs`
- Create: `tenferro-prims/src/cpu/cast.rs`
- Create: `tenferro-prims/src/cuda/cast.rs`
- Modify: `tenferro-prims/src/lib.rs`
- Modify: `tenferro-prims/src/cpu/mod.rs`
- Modify: `tenferro-prims/src/cuda/mod.rs`
- Modify: `tenferro-linalg/src/prims_bridge.rs`

**Step 1: Reuse the failing tests from Task 5**

No new tests.

**Step 2: Run focused tests**

Run:

```bash
cargo test -p tenferro-prims --lib metadata_bridge_phase1
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib metadata_bridge_phase1
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Define a new cast/bridge family instead of overloading `TensorScalarPrims`.

Minimum supported edges:

- `Bool -> f32`
- `Bool -> f64`
- `I32 -> f32`
- `I32 -> f64`

Then add metadata/scalar `where` composition support sufficient for linalg cleanup closure.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-prims --lib metadata_bridge_phase1
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib metadata_bridge_phase1
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-prims/src/families/cast.rs \
        tenferro-prims/src/cpu/cast.rs \
        tenferro-prims/src/cuda/cast.rs \
        tenferro-prims/src/lib.rs \
        tenferro-prims/src/cpu/mod.rs \
        tenferro-prims/src/cuda/mod.rs \
        tenferro-linalg/src/prims_bridge.rs
git commit -m "feat: add metadata to scalar bridge family"
```

## Task 7: Add representation-helper regression tests

**Files:**
- Create: `tenferro-tensor/src/tests/representation_helpers.rs`
- Modify: `tenferro-tensor/src/tests/mod.rs`

**Step 1: Write the failing tests**

Cover:

- `view_as_real`
- `view_as_complex`
- shape/stride contract
- CPU/CUDA parity

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor --lib representation_helpers
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-tensor --features cuda --lib representation_helpers
```

Expected: FAIL.

**Step 3: Write minimal implementation**

None in this task.

**Step 4: Re-run to verify intended failure**

Expected: FAIL.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/tests/representation_helpers.rs \
        tenferro-tensor/src/tests/mod.rs
git commit -m "test: add representation helper regressions"
```

## Task 8: Implement `view_as_real` and `view_as_complex`

**Files:**
- Modify: `tenferro-tensor/src/tensor/views.rs`

**Step 1: Reuse the failing tests from Task 7**

No new tests.

**Step 2: Run focused tests**

Run:

```bash
cargo test -p tenferro-tensor --lib representation_helpers
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-tensor --features cuda --lib representation_helpers
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Implement `ATen`-style directional equivalents for:

- complex -> real view
- real-last-dimension-of-2 -> complex view

Scope is only the layouts needed by the current dense eager core and linalg cleanup closure.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor --lib representation_helpers
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-tensor --features cuda --lib representation_helpers
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/tensor/views.rs
git commit -m "feat: add tensor real complex view helpers"
```

## Task 9: Add RNG regression tests

**Files:**
- Create: `tenferro-prims/src/tests/rng_phase1.rs`
- Modify: `tenferro-prims/src/tests/mod.rs`
- Create: `tenferro-tensor/src/tests/rng_constructors.rs`
- Modify: `tenferro-tensor/src/tests/mod.rs`

**Step 1: Write the failing tests**

Cover:

- seeded replay on CPU
- seeded replay on CUDA
- `rand`
- `randn`
- `randint`
- `*_like`
- shape/dtype/device semantics
- basic statistical sanity for `randn`

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims --lib rng_phase1
cargo test -p tenferro-tensor --lib rng_constructors
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib rng_phase1
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-tensor --features cuda --lib rng_constructors
```

Expected: FAIL.

**Step 3: Write minimal implementation**

None in this task.

**Step 4: Re-run to confirm intended failure**

Expected: FAIL.

**Step 5: Commit**

```bash
git add tenferro-prims/src/tests/rng_phase1.rs \
        tenferro-prims/src/tests/mod.rs \
        tenferro-tensor/src/tests/rng_constructors.rs \
        tenferro-tensor/src/tests/mod.rs
git commit -m "test: add rng phase 1 regressions"
```

## Task 10: Implement generator and RNG family core

**Files:**
- Create: `tenferro-device/src/generator.rs`
- Create: `tenferro-prims/src/families/rng.rs`
- Create: `tenferro-prims/src/cpu/rng.rs`
- Create: `tenferro-prims/src/cuda/rng.rs`
- Modify: `tenferro-device/src/lib.rs`
- Modify: `tenferro-device/src/cuda/mod.rs`
- Modify: `tenferro-prims/src/lib.rs`
- Modify: `tenferro-prims/src/cpu/mod.rs`
- Modify: `tenferro-prims/src/cuda/mod.rs`

**Step 1: Reuse the failing tests from Task 9**

No new tests.

**Step 2: Run focused tests**

Run:

```bash
cargo test -p tenferro-prims --lib rng_phase1
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib rng_phase1
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Add:

- CPU generator engine
- CUDA generator state with `Philox`
- RNG family descriptors and execution
- enough surface for:
  - uniform
  - normal
  - integer range

Keep CPU/CUDA API aligned even if implementation internals differ.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-prims --lib rng_phase1
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib rng_phase1
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-device/src/generator.rs \
        tenferro-prims/src/families/rng.rs \
        tenferro-prims/src/cpu/rng.rs \
        tenferro-prims/src/cuda/rng.rs \
        tenferro-device/src/lib.rs \
        tenferro-device/src/cuda/mod.rs \
        tenferro-prims/src/lib.rs \
        tenferro-prims/src/cpu/mod.rs \
        tenferro-prims/src/cuda/mod.rs
git commit -m "feat: add dense eager rng core"
```

## Task 11: Add tensor RNG constructors

**Files:**
- Modify: `tenferro-tensor/src/tensor/constructors.rs`

**Step 1: Reuse the failing tensor RNG constructor tests from Task 9**

No new tests.

**Step 2: Run focused tests**

Run:

```bash
cargo test -p tenferro-tensor --lib rng_constructors
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-tensor --features cuda --lib rng_constructors
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Add:

- `rand`
- `randn`
- `randint`
- `rand_like`
- `randn_like`
- `randint_like`

as `Tensor`-level constructor wrappers over the RNG substrate.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor --lib rng_constructors
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-tensor --features cuda --lib rng_constructors
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/tensor/constructors.rs
git commit -m "feat: add tensor rng constructors"
```

## Task 12: Rewrite `det`, `slogdet`, and `lu_solve` onto the dense eager core

**Files:**
- Modify: `tenferro-linalg/src/primal/linear_systems.rs`
- Modify: `tenferro-linalg/src/tests/runtime_capability.rs`
- Modify: `tenferro-linalg/src/tests/batch_a_contracts.rs`
- Modify: `tenferro-linalg/src/tests/mod.rs`

**Step 1: Write or extend failing cleanup tests**

Cover:

- `det` without host sign reconstruction
- real `slogdet` without host sign reconstruction
- complex `slogdet` without host metadata reconstruction
- `lu_solve` using pivot tensor metadata
- source-level regression guards against `tensor_from_data`, `Vec<i32>` bridge, and host parity reconstruction in public/composite paths

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg --lib det
cargo test -p tenferro-linalg --lib slogdet
cargo test -p tenferro-linalg --lib lu_solve
```

Expected: FAIL because host reconstruction remains.

**Step 3: Write minimal implementation**

Rewrite onto:

- metadata tensor arithmetic
- metadata/scalar cast bridge
- metadata/scalar `where`
- tensor-native pivot/info flow

Do not add new LU-specific helper APIs.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg --lib det
cargo test -p tenferro-linalg --lib slogdet
cargo test -p tenferro-linalg --lib lu_solve
cargo test -p tenferro-linalg --features cuda --lib --no-run
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/primal/linear_systems.rs \
        tenferro-linalg/src/tests/runtime_capability.rs \
        tenferro-linalg/src/tests/batch_a_contracts.rs \
        tenferro-linalg/src/tests/mod.rs
git commit -m "refactor: clean up lu metadata paths onto dense eager core"
```

## Task 13: Align the public LU surface toward PyTorch

**Files:**
- Modify: `tenferro-linalg/src/result_types/decomposition.rs`
- Modify: `tenferro-linalg/src/primal/decompositions.rs`
- Modify: `tenferro-linalg-prims/src/lib.rs`
- Modify: `tenferro-linalg/src/tests/mod.rs`

**Step 1: Write the failing tests**

Cover:

- `lu_factor` returns pivot tensor surface
- `lu_factor_ex` returns pivot tensor and info tensor surface
- `lu_solve` consumes pivot tensors instead of slice metadata

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg --lib lu_factor
cargo test -p tenferro-linalg --lib lu_factor_ex
cargo test -p tenferro-linalg --lib lu_solve
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Align public surface toward `PyTorch`:

- pivot tensors
- info tensors
- no reintroduction of host `Vec<usize>` in the main path

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-linalg --lib lu_factor
cargo test -p tenferro-linalg --lib lu_factor_ex
cargo test -p tenferro-linalg --lib lu_solve
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-linalg/src/result_types/decomposition.rs \
        tenferro-linalg/src/primal/decompositions.rs \
        tenferro-linalg-prims/src/lib.rs \
        tenferro-linalg/src/tests/mod.rs
git commit -m "feat: align lu public surface with tensor metadata"
```

## Task 14: Run the full verification gate

**Files:**
- None

**Step 1: Run focused crate gates**

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-device --features cuda --lib
cargo test -p tenferro-tensor --features cuda --lib
cargo test -p tenferro-prims --features cuda --lib
cargo test -p tenferro-linalg-prims --features cuda --lib
cargo test -p tenferro-linalg --features cuda --lib
```

Expected: PASS.

**Step 2: Run full workspace gate**

Run:

```bash
cargo test --workspace --release
cargo llvm-cov --workspace --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
cargo doc --workspace --no-deps
python3 scripts/check-docs-site.py
```

Expected: PASS.

**Step 3: Commit**

No commit in this task unless verification fixes were required.

## Execution Order Notes

- Do not start from Task 12.
- Tasks 1-6 define the deterministic dense eager core needed by linalg cleanup.
- Tasks 9-11 define RNG and complete the constructor layer of the `ATen`-aligned phase.
- `view_as_real` / `view_as_complex` are scheduled before the RNG/linalg tail because they are part of the target representation substrate and likely to be reused by follow-on cleanup.
