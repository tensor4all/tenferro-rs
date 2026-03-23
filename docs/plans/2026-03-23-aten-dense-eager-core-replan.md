# ATen Dense Eager Core Compatibility Replan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-sequence the ATen dense eager core phase so constructor fallibility is fixed first, then complete the remaining dense eager substrate and linalg cleanup without reintroducing host bridges or panic-based public APIs.

**Architecture:** Treat constructor fallibility as part of the dense eager core, not as a side issue. First migrate `tenferro-tensor` constructors from panic-based GPU allocation and edge-case aborts to `Result<Self>`-based APIs, then continue with metadata phase 2, cast/select bridges, representation helpers, RNG, and finally tensor-native linalg cleanup.

**Tech Stack:** Rust workspace crates `tenferro-device`, `tenferro-tensor`, `tenferro-prims`, `tenferro-linalg-prims`, `tenferro-linalg`; CUDA runtime-loaded kernels; Philox for CUDA RNG; CPU generator engine; `strided-view`; existing family `plan/execute` protocols.

---

## Current Baseline

- Worktree: `/home/shinaoka/tensor4all/tenferro-rs/.worktrees/complex-real-unary-substrate`
- Branch head when this replan was written: `1224b76`
- Relevant completed commits:
  - `04813a1` `docs: add aten dense eager core design and plan`
  - `d195f1e` `test: add deterministic constructor regressions`
  - `f26d919` `feat: add deterministic dense eager constructors`
  - `1224b76` `fix: return results from dense constructor inputs`

### What Is Already In Place

- ATen dense eager design doc exists
- Initial deterministic constructor tests exist
- Initial deterministic constructor implementations exist
- `arange` / `linspace` already moved to `Result<Self>`
- `*_like` layout policy is now test-enforced

### Why The Previous Plan Is No Longer Canonical

The previous implementation plan assumed Task 2 would be complete after `empty/full/arange/linspace` landed. Code-quality review showed that this was false:

- `finish_allocation(...)` still panics on GPU allocation failure
- public constructor APIs that route through it still cannot report errors
- `eye` still contains direct panic paths

That means the deterministic constructor tranche is not actually complete until the public constructor surface itself becomes fallible.

Therefore this replan moves **constructor API migration** to the front of the phase and treats it as a prerequisite for the rest of the dense eager core.

## Design Constraints

- No updates to historical plan files; this file supersedes them operationally.
- No new panic-based public constructors in library code.
- No `tenferro-tensor -> tenferro-prims` dependency.
- No linalg-specific parity/sign helper if generic metadata substrate can express the same behavior.
- No new CPU fallback in `tenferro-linalg`.
- GPU payload fallback to host remains forbidden outside tests and minimal `info` control flow.

## Revised Program Structure

### Program A: Fallible Constructor Core

Complete the constructor layer so public dense eager constructors behave like proper Rust APIs:

- return `Result<Self>` where allocation/layout/parameter errors are possible
- stop aborting the process on GPU allocation failure
- keep constructor logic in `tenferro-tensor`
- preserve CPU/CUDA shared semantics

### Program B: Deterministic Dense Eager Substrate

After constructor APIs are sound:

- metadata phase 2
- metadata/scalar bridge
- `where`
- conservative promotion/cast closure
- `view_as_real` / `view_as_complex`

### Program C: RNG Core

Once deterministic dense eager core is stable:

- generator abstraction
- CPU engine
- CUDA Philox
- `rand`, `randn`, `randint`, `*_like`

### Program D: Linalg Cleanup

Only after Programs A-C:

- `det`
- `slogdet`
- `lu_solve`
- LU public surface alignment
- remaining host metadata bridge removal

## Execution Order

1. Finish constructor API migration
2. Re-close deterministic constructor tranche
3. Implement metadata phase 2
4. Implement metadata-to-scalar cast/select bridge
5. Add representation helpers
6. Implement RNG core
7. Rewrite linalg consumers

Do not continue from metadata phase 2 until constructor migration is complete.

## Task 1: Add constructor-fallibility regression tests and guards

**Files:**
- Create: `tenferro-tensor/src/tests/constructor_fallibility.rs`
- Modify: `tenferro-tensor/src/tests/mod.rs`

**Step 1: Write the failing tests**

Add tests for:

- invalid `empty_strided` returns `Err`, not panic
- invalid `arange` inputs return `Err`
- invalid `linspace` inputs return `Err`
- source-level regression guard: public constructor code in `constructors.rs` no longer uses:
  - `panic!`
  - `unwrap_or_else(|err| panic!...)`
  in the public constructor path

The source-level guard should explicitly target:

- `empty`
- `zeros`
- `ones`
- `full`
- `empty_like`
- `zeros_like`
- `ones_like`
- `full_like`
- `eye`

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-tensor --lib constructor_fallibility
```

Expected: FAIL because public constructor code still contains panic paths.

**Step 3: Write minimal implementation**

None in this task.

**Step 4: Re-run to verify intended failure**

Expected: FAIL on the constructor panic guard or unfallible API mismatch.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/tests/constructor_fallibility.rs \
        tenferro-tensor/src/tests/mod.rs
git commit -m "test: add constructor fallibility regressions"
```

## Task 2: Convert deterministic public constructors to `Result<Self>`

**Files:**
- Modify: `tenferro-tensor/src/tensor/constructors.rs`
- Modify: `tenferro-tensor/src/tensor/mod.rs` (only if re-exports or docs need adjustment)

**Step 1: Reuse the failing tests from Task 1**

No new tests yet.

**Step 2: Run focused tests**

Run:

```bash
cargo test -p tenferro-tensor --lib constructors_phase2
cargo test -p tenferro-tensor --lib constructor_fallibility
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Change public constructor APIs so the following return `Result<Self>`:

- `zeros`
- `empty`
- `ones`
- `empty_strided`
- `full`
- `empty_like`
- `zeros_like`
- `ones_like`
- `full_like`
- `eye`
- `arange`
- `linspace`

Implementation notes:

- Replace `finish_allocation(...) -> Self` with a fallible helper.
- Preserve CPU fast paths.
- GPU placement should use `to_memory_space_async(...)` and propagate `Err`.
- `eye` overflow/layout issues should return `Error`, not panic.
- Keep deterministic semantics for `empty` for now unless a later design explicitly changes it.

**Step 4: Run focused tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor --lib constructors_phase2
cargo test -p tenferro-tensor --lib constructor_fallibility
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-tensor --features cuda --lib constructors_phase2
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/tensor/constructors.rs \
        tenferro-tensor/src/tensor/mod.rs
git commit -m "refactor: make dense constructors fallible"
```

## Task 3: Migrate constructor call sites inside `tenferro-tensor`

**Files:**
- Modify: `tenferro-tensor/src/tests/constructors_phase2.rs`
- Modify: any `tenferro-tensor/src/tests/*.rs` files that call the changed constructors
- Modify: relevant doctests in `tenferro-tensor/src/tensor/constructors.rs`

**Step 1: Add or adjust failing tests**

Ensure tests use `unwrap()` only in test code and that docs reflect `Result`-returning constructors.

**Step 2: Run tests to verify failures**

Run:

```bash
cargo test -p tenferro-tensor --lib
```

Expected: FAIL at compile time or runtime where call sites still assume infallible constructors.

**Step 3: Write minimal implementation**

Update local call sites and docs only within `tenferro-tensor`.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test -p tenferro-tensor --lib
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-tensor --features cuda --lib
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tenferro-tensor/src/tests \
        tenferro-tensor/src/tensor/constructors.rs
git commit -m "test: update tensor constructor call sites"
```

## Task 4: Migrate constructor call sites in downstream crates

**Files:**
- Modify: call sites across:
  - `tenferro-prims`
  - `tenferro-linalg-prims`
  - `tenferro-linalg`
  - `tenferro`
  - other workspace crates as needed

**Step 1: Find remaining compile errors**

Run:

```bash
cargo test --workspace --no-run
```

Expected: FAIL on call sites that still assume infallible constructors.

**Step 2: Write minimal implementation**

Update call sites with the narrowest appropriate behavior:

- propagate `Result` in library code where the surrounding API is already fallible
- use `unwrap()` only in tests / examples where appropriate
- avoid introducing panic-based bridges in production code

**Step 3: Run compile/test verification**

Run:

```bash
cargo test --workspace --no-run
cargo test -p tenferro-prims --lib
cargo test -p tenferro-linalg-prims --lib
cargo test -p tenferro-linalg --lib
```

Expected: PASS for the touched areas.

**Step 4: Commit**

```bash
git add tenferro-prims \
        tenferro-linalg-prims \
        tenferro-linalg \
        tenferro
git commit -m "refactor: propagate fallible constructor APIs"
```

## Task 5: Re-close the deterministic constructor tranche

**Files:**
- None required beyond touched constructor/test files

**Step 1: Run focused constructor gate**

Run:

```bash
cargo fmt --all --check
cargo test -p tenferro-tensor --lib constructors_phase2
cargo test -p tenferro-tensor --lib constructor_fallibility
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-tensor --features cuda --lib constructors_phase2
```

Expected: PASS.

**Step 2: Commit**

No commit unless fixes were needed.

## Task 6: Add metadata phase-2 regression tests

**Files:**
- Create: `tenferro-prims/src/tests/metadata_phase2.rs`
- Modify: `tenferro-prims/src/tests/mod.rs`

**Step 1: Write the failing tests**

Cover:

- `i32 + i32`
- `i32 - i32`
- `i32 * i32`
- bool `bitand`
- metadata `where`
- metadata `sum`
- metadata `all`
- metadata `any`
- shape/broadcast sanity

Add CPU tests and CUDA parity tests.

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims --lib metadata_phase2
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib metadata_phase2
```

Expected: FAIL.

**Step 3: Write minimal implementation**

None in this task.

**Step 4: Re-run to verify intended failure**

Expected: FAIL on unsupported metadata ops.

**Step 5: Commit**

```bash
git add tenferro-prims/src/tests/metadata_phase2.rs \
        tenferro-prims/src/tests/mod.rs
git commit -m "test: add metadata phase 2 regressions"
```

## Task 7: Implement metadata phase-2 CPU/CUDA support

**Files:**
- Modify: `tenferro-prims/src/cpu/metadata.rs`
- Modify: `tenferro-prims/src/cuda/metadata.rs`
- Modify: `tenferro-device/src/cuda/runtime/pointwise/pointwise_metadata.rs`
- Modify: `tenferro-device/src/cuda/runtime/kernels/metadata_scalar.rs`

**Step 1: Reuse the failing tests from Task 6**

No new tests.

**Step 2: Run focused tests**

Run:

```bash
cargo test -p tenferro-prims --lib metadata_phase2
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib metadata_phase2
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Add support for:

- `Add`
- `Sub`
- `Mul`
- `BitAnd`

plus any wiring needed by metadata `where` / reduction tests.

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

## Task 8: Add metadata-to-scalar bridge regression tests

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
- CPU/CUDA parity

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-prims --lib metadata_bridge_phase1
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib metadata_bridge_phase1
```

Expected: FAIL.

**Step 3: Write minimal implementation**

None in this task.

**Step 4: Re-run to verify intended failure**

Expected: FAIL.

**Step 5: Commit**

```bash
git add tenferro-prims/src/tests/metadata_bridge_phase1.rs \
        tenferro-prims/src/tests/mod.rs
git commit -m "test: add metadata bridge regressions"
```

## Task 9: Implement metadata-to-scalar cast/select bridge

**Files:**
- Create: `tenferro-prims/src/families/cast.rs`
- Create: `tenferro-prims/src/cpu/cast.rs`
- Create: `tenferro-prims/src/cuda/cast.rs`
- Modify: `tenferro-prims/src/lib.rs`
- Modify: `tenferro-prims/src/cpu/mod.rs`
- Modify: `tenferro-prims/src/cuda/mod.rs`
- Modify: `tenferro-linalg/src/prims_bridge.rs`

**Step 1: Reuse the failing tests from Task 8**

No new tests.

**Step 2: Run focused tests**

Run:

```bash
cargo test -p tenferro-prims --lib metadata_bridge_phase1
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-prims --features cuda --lib metadata_bridge_phase1
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Add a cast/bridge family for:

- `Bool -> f32`
- `Bool -> f64`
- `I32 -> f32`
- `I32 -> f64`

Then add metadata/scalar `where` composition support sufficient for linalg cleanup.

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

## Task 10: Add representation-helper regression tests

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

## Task 11: Implement `view_as_real` and `view_as_complex`

**Files:**
- Modify: `tenferro-tensor/src/tensor/views.rs`

**Step 1: Reuse the failing tests from Task 10**

No new tests.

**Step 2: Run focused tests**

Run:

```bash
cargo test -p tenferro-tensor --lib representation_helpers
TENFERRO_TEST_CUDA=1 cargo test -p tenferro-tensor --features cuda --lib representation_helpers
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Implement directional equivalents for:

- complex -> real view
- real-last-dimension-of-2 -> complex view

limited to the layouts needed by the dense eager core and linalg cleanup closure.

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

## Task 12: Add RNG regression tests

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

**Step 4: Re-run to verify intended failure**

Expected: FAIL.

**Step 5: Commit**

```bash
git add tenferro-prims/src/tests/rng_phase1.rs \
        tenferro-prims/src/tests/mod.rs \
        tenferro-tensor/src/tests/rng_constructors.rs \
        tenferro-tensor/src/tests/mod.rs
git commit -m "test: add rng phase 1 regressions"
```

## Task 13: Implement generator and RNG family core

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

**Step 1: Reuse the failing tests from Task 12**

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
- CUDA generator state with Philox
- RNG family descriptors and execution
- enough surface for:
  - uniform
  - normal
  - integer range

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

## Task 14: Add tensor RNG constructors

**Files:**
- Modify: `tenferro-tensor/src/tensor/constructors.rs`

**Step 1: Reuse the failing tensor RNG constructor tests from Task 12**

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

## Task 15: Rewrite `det`, `slogdet`, and `lu_solve` onto the dense eager core

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
- source-level guards against:
  - `tensor_from_data(...)`
  - forward-permutation reconstruction helpers
  - public/composite host parity reconstruction

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p tenferro-linalg --lib det
cargo test -p tenferro-linalg --lib slogdet
cargo test -p tenferro-linalg --lib lu_solve
```

Expected: FAIL.

**Step 3: Write minimal implementation**

Rewrite onto:

- metadata tensor arithmetic
- metadata/scalar cast bridge
- metadata/scalar `where`
- tensor-native pivot/info flow

Do not add LU-specific helper APIs.

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

## Task 16: Align the public LU surface toward PyTorch

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

Align public surface toward PyTorch:

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

## Task 17: Run the full verification gate

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

No commit unless verification fixes were required.

## Handoff Notes

- The previous file `2026-03-23-aten-dense-eager-core-implementation.md` should be treated as a historical draft before constructor fallibility was fully recognized as phase-critical.
- Start from Task 1 in this replan, not from metadata phase 2.
- When executing this plan, keep the review finding in mind: constructor API fallibility is not optional cleanup, it is part of the dense eager core contract.
